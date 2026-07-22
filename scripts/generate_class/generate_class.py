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


def compute_targets(stats_path: Path, target_minutes: float) -> tuple[int, int, int, int, int]:
    """Derives word-count and pause targets for a class of the requested length, calibrated
    against the real corpus's observed speaking rate and pause behavior when available.
    Returns (target_words, target_pause_count, target_pause_seconds, target_micro_pause_count,
    target_micro_pause_seconds)."""
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

Write ONLY the teacher's own continuous instructional monologue - as if narrating a class in real time. Do not \
include:
- Greetings or check-ins addressed to specific students
- Questions to, or answers from, students
- Any references to specific student names
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
examples show it."""


def build_system_prompt(examples: list[str], digest_path: Path) -> list[dict]:
    """Instructions + digest are identical across every class in a run, so they get their own
    cache_control breakpoint - that block actually hits the cache from the 2nd class onward.
    The example sample is re-randomized per class (see generate_one), so it's kept in a separate,
    uncached block: tagging ever-changing content with cache_control would never hit and would
    only pay the pricier cache-write cost on every single call instead of plain input pricing."""
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
    blocks.append({"type": "text", "text": "\n".join(example_parts)})
    return blocks


def build_user_message(
    topic: str | None,
    target_words: int,
    target_pause_count: int,
    target_pause_seconds: int,
    target_micro_pause_count: int,
    target_micro_pause_seconds: int,
    target_minutes: float,
) -> str:
    ask = "Now write a new, original class script in this teacher's style"
    if topic:
        ask += f", focused on {topic}"
    ask += ". Do not copy sentences from the examples - write new instructional content that matches the teacher's voice, structure, and vocabulary."
    ask += (
        f" Aim for approximately {target_words} words of spoken narration, with roughly {target_pause_count} "
        f"[ПАУЗА:N] pause markers totaling around {target_pause_seconds // 60} minutes of pause time, so the "
        f"full class (narration plus pauses) comes out to about {target_minutes:g} minutes overall."
    )
    ask += (
        f" Also include roughly {target_micro_pause_count} brief [МИКРОПАУЗА:N] settle/breath pauses, "
        f"totaling around {target_micro_pause_seconds} seconds."
    )
    return ask


def generate_class(
    client: anthropic.Anthropic,
    examples: list[str],
    model: str,
    max_tokens: int,
    topic: str | None,
    digest_path: Path,
    target_words: int,
    target_pause_count: int,
    target_pause_seconds: int,
    target_micro_pause_count: int,
    target_micro_pause_seconds: int,
    target_minutes: float,
) -> str:
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=build_system_prompt(examples, digest_path),
        messages=[{
            "role": "user",
            "content": build_user_message(
                topic, target_words, target_pause_count, target_pause_seconds,
                target_micro_pause_count, target_micro_pause_seconds, target_minutes,
            ),
        }],
    ) as stream:
        message = stream.get_final_message()
    return next(block.text for block in message.content if block.type == "text")


def generate_one(
    client: anthropic.Anthropic,
    files: list[Path],
    seed: int,
    num_examples: int,
    model: str,
    max_tokens: int,
    topic: str | None,
    digest_path: Path,
    target_words: int,
    target_pause_count: int,
    target_pause_seconds: int,
    target_micro_pause_count: int,
    target_micro_pause_seconds: int,
    target_minutes: float,
    dst_path: Path,
) -> None:
    # A different seed per class means a different sample of examples - without this, every
    # class in a batch would be grounded in the same three transcripts and read as variations
    # on one style-transfer rather than genuinely separate classes.
    chosen = random.Random(seed).sample(files, num_examples)
    print(f"{dst_path.name}: examples = {', '.join(f.name for f in chosen)}")
    examples = [f.read_text(encoding="utf-8") for f in chosen]

    text = generate_class(
        client, examples, model, max_tokens, topic, digest_path,
        target_words, target_pause_count, target_pause_seconds,
        target_micro_pause_count, target_micro_pause_seconds, target_minutes,
    )
    dst_path.write_text(text, encoding="utf-8")
    print(f"{dst_path.name}: done -> {dst_path}")


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
        compute_targets(Path(cfg.stats_path), cfg.target_minutes)
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
                digest_path, target_words, target_pause_count, target_pause_seconds,
                target_micro_pause_count, target_micro_pause_seconds, cfg.target_minutes,
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
