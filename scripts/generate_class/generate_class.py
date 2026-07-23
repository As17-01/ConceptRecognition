import json
import random
import sys

import anthropic
import hydra

from pathlib import Path
from omegaconf import DictConfig

# Fallback assumptions used only when data/corpus_stats.json (written by summarize_corpus.py)
# hasn't been produced yet - keeps this script runnable before that script's first run.
DEFAULT_WORDS_PER_MINUTE = 140.0
DEFAULT_PAUSE_FRACTION = 0.3
DEFAULT_MICRO_PAUSE_FRACTION = 0.02


# Upper bound on how much of a class can be pause/micro-pause time once scaled up, so aggressive
# pause_scale values can't starve the class of actual narration.
MAX_PAUSE_FRACTION_SUM = 0.7


def compute_targets(
    stats_path: Path,
    target_minutes: float,
    pause_scale: float = 1.0,
    micro_pause_scale: float = 1.0,
) -> tuple[int, int, int, int, int]:
    """Derives word-count and pause targets for a class of the requested length, calibrated
    against the real corpus's observed speaking rate and pause behavior when available.
    pause_scale / micro_pause_scale multiply the corpus's natural pause and micro-pause share to
    make the class roomier (more/longer breaks, correspondingly fewer narrated words) while keeping
    the same total duration. Returns (target_words, target_pause_count, target_pause_seconds,
    target_micro_pause_count, target_micro_pause_seconds)."""
    target_total_seconds = target_minutes * 60

    if stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        wpm = stats["words_per_minute"]
        avg_total_seconds = (
            stats["avg_words"] / wpm * 60 + stats["avg_pause_seconds"] + stats["avg_micro_pause_seconds"]
        )
        # Only fall back to the default split if the corpus average is degenerate (zero) -
        # a real corpus average, however skewed, is always preferred over the hardcoded guess.
        pause_fraction = stats["avg_pause_seconds"] / avg_total_seconds if avg_total_seconds else DEFAULT_PAUSE_FRACTION
        micro_pause_fraction = (
            stats["avg_micro_pause_seconds"] / avg_total_seconds if avg_total_seconds else DEFAULT_MICRO_PAUSE_FRACTION
        )
        avg_seconds_per_pause = stats["avg_pause_seconds"] / stats["avg_pause_count"] if stats["avg_pause_count"] else None
        avg_seconds_per_micro_pause = (
            stats["avg_micro_pause_seconds"] / stats["avg_micro_pause_count"] if stats["avg_micro_pause_count"] else None
        )
    else:
        wpm = DEFAULT_WORDS_PER_MINUTE
        pause_fraction = DEFAULT_PAUSE_FRACTION
        micro_pause_fraction = DEFAULT_MICRO_PAUSE_FRACTION
        avg_seconds_per_pause = None
        avg_seconds_per_micro_pause = None

    # Roomier class: give pauses a bigger share of the fixed total (so narration shrinks to match),
    # clamped so the two together never crowd out most of the actual speech.
    pause_fraction *= pause_scale
    micro_pause_fraction *= micro_pause_scale
    fraction_sum = pause_fraction + micro_pause_fraction
    if fraction_sum > MAX_PAUSE_FRACTION_SUM:
        scale_down = MAX_PAUSE_FRACTION_SUM / fraction_sum
        pause_fraction *= scale_down
        micro_pause_fraction *= scale_down

    target_pause_seconds = target_total_seconds * pause_fraction
    target_micro_pause_seconds = target_total_seconds * micro_pause_fraction
    target_speaking_seconds = target_total_seconds - target_pause_seconds - target_micro_pause_seconds
    target_words = round(target_speaking_seconds / 60 * wpm)
    target_pause_count = (
        round(target_pause_seconds / avg_seconds_per_pause)
        if avg_seconds_per_pause is not None
        else round(target_minutes / 6)  # no corpus pause data - guess roughly one pause every 6 minutes
    )
    target_micro_pause_count = (
        round(target_micro_pause_seconds / avg_seconds_per_micro_pause)
        if avg_seconds_per_micro_pause is not None
        else round(target_minutes * 1.5)  # no corpus micro-pause data - guess roughly 1.5 per minute
    )
    return (
        target_words,
        target_pause_count,
        round(target_pause_seconds),
        target_micro_pause_count,
        round(target_micro_pause_seconds),
    )


# Instructs Claude to write only the teacher's own instructional monologue - not the
# surrounding student interaction (greetings, Q&A, names) that's mixed into the real
# transcripts. Filtering that out is a prompt instruction here, not a preprocessing step:
# Claude's own reading comprehension handles it directly from the few-shot examples,
# without needing the examples pre-split or pre-labeled.
SYSTEM_INSTRUCTIONS = """You are ghostwriting a new contemporary dance / movement improvisation class script in \
the voice and style of a specific teacher, based on transcripts of their real classes.

Write ONLY the teacher's own continuous instructional monologue - as if narrating a class in real time, guiding a \
general audience. Do not include:
- Greetings or check-ins addressed to specific students
- Questions to, or answers from, students
- Any references to specific student names
- Personal corrections, adjustments, or feedback aimed at one individual student (e.g. "нет, у тебя колено \
уходит внутрь", "чуть выше руку", "Маша, расслабь плечи", hands-on fixes or reacting to what one person is \
doing). Keep every instruction addressed to everyone at once, never to a single person's specific mistake.
- Technical/logistical chatter (audio issues, "can you hear me", scheduling, etc.)

Insert the literal marker [ПАУЗА:N] (N = an integer number of seconds, e.g. [ПАУЗА:15]) at points that represent \
a real movement or music break with no narration - the same convention the example transcripts use. Some examples \
carry these markers throughout and some don't (it depends on how that particular recording was processed), so \
follow the convention regardless of whether a given example happens to show it.

Also insert the literal marker [МИКРОПАУЗА:N] (N = a small integer number of seconds, e.g. [МИКРОПАУЗА:3]) at \
brief settle or breath pauses - a short natural break in your speech rhythm between thoughts, not necessarily \
tied to any physical movement or music break. This is distinct from [ПАУЗА:N]: [ПАУЗА:N] marks a real break with \
no narration, while [МИКРОПАУЗА:N] marks the kind of short breath a real speaker naturally takes between \
sentences or ideas while still narrating the class overall.

The output should read as one continuous, flowing class script a student could follow directly - covering a \
warm-up, a technical or thematic focus, and a natural close - in the teacher's own vocabulary and phrasing style, \
including their characteristic code-switching between Russian and English movement terminology where the \
examples show it.

You may be asked to write the class in consecutive sections across several turns. When continuing a class you have \
already started, pick up seamlessly from where you left off: do not greet, do not recap or repeat earlier content, \
do not restart the warm-up, and only bring the class to a close when explicitly asked for the final section. Each \
turn should read as the direct continuation of the previous one, as if it were all one uninterrupted class."""


def build_system_prompt(examples: list[str], digest_path: Path) -> list[dict]:
    """Two cache breakpoints. Instructions + digest are identical across every class in a run, so
    that block hits the cache from the 2nd class onward. The example sample is re-randomized per
    class (see generate_one) so it never hits across classes - but since each class is now written
    as several section calls that all reuse the same examples, caching that block still pays off:
    it's written once on the first section and reused by the remaining sections of the same class
    (well within the ephemeral cache TTL, as the sections run back to back)."""
    static_parts = [SYSTEM_INSTRUCTIONS]
    if digest_path.is_file():
        static_parts.append(
            "\nHere are corpus-wide patterns distilled from the full set of this teacher's classes:\n"
            + digest_path.read_text(encoding="utf-8")
        )
    blocks = [{"type": "text", "text": "\n".join(static_parts), "cache_control": {"type": "ephemeral"}}]

    example_parts = ["Here are transcripts of several real classes by this teacher, for style and content reference:"]
    for i, example in enumerate(examples, 1):
        example_parts.append(f"\n--- Example class {i} ---\n{example}")
    blocks.append({"type": "text", "text": "\n".join(example_parts), "cache_control": {"type": "ephemeral"}})
    return blocks


def split_by_weight(total: int, weights: list[float]) -> list[int]:
    """Splits an integer total across sections proportionally to weights (normalized here, so they
    needn't sum to 1). Rounds each share, then puts any rounding remainder on the largest section
    so the parts still sum exactly to total."""
    weight_sum = sum(weights)
    shares = [round(total * w / weight_sum) for w in weights]
    drift = total - sum(shares)
    if drift and shares:
        shares[shares.index(max(shares))] += drift
    return shares


def build_section_user_message(
    section_index: int,
    num_sections: int,
    section_name: str,
    topic: str | None,
    sec_words: int,
    sec_pause_count: int,
    sec_pause_seconds: int,
    sec_micro_pause_count: int,
    sec_micro_pause_seconds: int,
) -> str:
    is_first = section_index == 0
    is_last = section_index == num_sections - 1

    if is_first:
        ask = "Now begin a new, original class script in this teacher's style"
        if topic:
            ask += f", focused on {topic}"
        ask += (
            f". This class will be written in {num_sections} consecutive sections; this is section "
            f"1 of {num_sections} - the {section_name}. Do not copy sentences from the examples - write new "
            "instructional content that matches the teacher's voice, structure, and vocabulary. Do NOT wrap up "
            "or conclude the class - this is only the beginning; end mid-flow so it can continue."
        )
    elif is_last:
        ask = (
            f"Continue the SAME class directly from where you stopped, and bring it to its natural close. "
            f"This is the final section ({section_index + 1} of {num_sections}) - the {section_name}. Do not greet, "
            "recap, or repeat earlier content. Keep the same voice and flow, then wind the class down and end it "
            "as this teacher naturally would."
        )
    else:
        ask = (
            f"Continue the SAME class directly from where you stopped. This is section {section_index + 1} of "
            f"{num_sections} - the {section_name}. Do not greet, recap, or repeat earlier content, and do not "
            "conclude the class yet. Keep the same voice and flow."
        )

    ask += (
        f" For this section, aim for approximately {sec_words} words of spoken narration, with roughly "
        f"{sec_pause_count} [ПАУЗА:N] pause markers totaling around {sec_pause_seconds} seconds, and about "
        f"{sec_micro_pause_count} brief [МИКРОПАУЗА:N] settle/breath pauses totaling around "
        f"{sec_micro_pause_seconds} seconds."
    )
    return ask


def generate_class(
    client: anthropic.Anthropic,
    examples: list[str],
    model: str,
    max_tokens: int,
    topic: str | None,
    digest_path: Path,
    sections: list,
    target_words: int,
    target_pause_count: int,
    target_pause_seconds: int,
    target_micro_pause_count: int,
    target_micro_pause_seconds: int,
) -> str:
    """Builds the class as a multi-turn continuation: each section is its own generation, but the
    running message history (prior sections as assistant turns) keeps the model writing one
    coherent class rather than restarting each time. Per-section targets are the totals split by
    each section's weight."""
    names = [str(s["name"]) for s in sections]
    weights = [float(s["weight"]) for s in sections]
    words_split = split_by_weight(target_words, weights)
    pause_count_split = split_by_weight(target_pause_count, weights)
    pause_seconds_split = split_by_weight(target_pause_seconds, weights)
    micro_count_split = split_by_weight(target_micro_pause_count, weights)
    micro_seconds_split = split_by_weight(target_micro_pause_seconds, weights)

    system = build_system_prompt(examples, digest_path)
    messages: list[dict] = []
    parts: list[str] = []
    num_sections = len(sections)
    for k in range(num_sections):
        messages.append({
            "role": "user",
            "content": build_section_user_message(
                k, num_sections, names[k], topic,
                words_split[k], pause_count_split[k], pause_seconds_split[k],
                micro_count_split[k], micro_seconds_split[k],
            ),
        })
        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        ) as stream:
            message = stream.get_final_message()
        section_text = next(block.text for block in message.content if block.type == "text").strip()
        parts.append(section_text)
        # Feed the section back as the assistant turn so the next section continues from it.
        messages.append({"role": "assistant", "content": section_text})
        label = names[k].split(":")[0].strip()
        print(f"    section {k + 1}/{num_sections} ({label}): ~{len(section_text.split())} words")

    return "\n\n".join(parts)


def generate_one(
    client: anthropic.Anthropic,
    files: list[Path],
    seed: int,
    num_examples: int,
    model: str,
    max_tokens: int,
    topic: str | None,
    digest_path: Path,
    sections: list,
    target_words: int,
    target_pause_count: int,
    target_pause_seconds: int,
    target_micro_pause_count: int,
    target_micro_pause_seconds: int,
    dst_path: Path,
) -> None:
    # A different seed per class means a different sample of examples - without this, every
    # class in a batch would be grounded in the same three transcripts and read as variations
    # on one style-transfer rather than genuinely separate classes.
    chosen = random.Random(seed).sample(files, num_examples)
    print(f"{dst_path.name}: examples = {', '.join(f.name for f in chosen)}")
    examples = [f.read_text(encoding="utf-8") for f in chosen]

    text = generate_class(
        client, examples, model, max_tokens, topic, digest_path, sections,
        target_words, target_pause_count, target_pause_seconds,
        target_micro_pause_count, target_micro_pause_seconds,
    )
    dst_path.write_text(text, encoding="utf-8")
    print(f"{dst_path.name}: done ({len(text.split())} words) -> {dst_path}")


@hydra.main(config_path="../conf", config_name="generate_class", version_base=None)
def main(cfg: DictConfig) -> None:
    src_dir = Path(cfg.src)
    if not src_dir.is_dir():
        print(f"Source folder not found: {src_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(src_dir.glob("*.txt"))
    if len(files) < cfg.num_examples:
        print(f"Only {len(files)} transcripts available, need {cfg.num_examples}", file=sys.stderr)
        sys.exit(1)

    dst_dir = Path(cfg.dst)
    dst_dir.mkdir(parents=True, exist_ok=True)

    digest_path = Path(cfg.digest_path)
    # Targets don't vary across classes in the same run, so compute them once rather than
    # re-deriving (and re-reading corpus_stats.json) on every iteration.
    target_words, target_pause_count, target_pause_seconds, target_micro_pause_count, target_micro_pause_seconds = (
        compute_targets(
            Path(cfg.stats_path), cfg.target_minutes,
            cfg.get("pause_scale", 1.0), cfg.get("micro_pause_scale", 1.0),
        )
    )

    client = anthropic.Anthropic()
    succeeded, skipped, failed = 0, 0, 0
    for i in range(cfg.num_classes):
        dst_path = dst_dir / f"class_{i + 1:03d}.txt"

        # Same resumability pattern as the rest of the pipeline: a crash or interruption
        # partway through a batch of generations shouldn't force redoing (and re-billing)
        # the ones that already finished.
        if dst_path.exists():
            print(f"{dst_path.name}: output already exists, skipping")
            skipped += 1
            continue

        try:
            generate_one(
                client, files, cfg.seed + i, cfg.num_examples, cfg.model, cfg.max_tokens, cfg.topic,
                digest_path, cfg.sections, target_words, target_pause_count, target_pause_seconds,
                target_micro_pause_count, target_micro_pause_seconds,
                dst_path,
            )
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            failed += 1
            continue
        succeeded += 1

    print(f"\n{succeeded} generated, {skipped} skipped (already done), {failed} failed, out of {cfg.num_classes} total")


if __name__ == "__main__":
    main()
