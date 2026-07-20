import random
import sys

import anthropic
import hydra

from pathlib import Path
from omegaconf import DictConfig

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

The output should read as one continuous, flowing class script a student could follow directly - covering a \
warm-up, a technical or thematic focus, and a natural close - in the teacher's own vocabulary and phrasing style, \
including their characteristic code-switching between Russian and English movement terminology where the \
examples show it."""


def build_system_prompt(examples: list[str]) -> list[dict]:
    """Examples go in the system prompt (not the user message) so they're covered by a single
    cache_control breakpoint - regenerating multiple classes from the same example set only pays
    the full input cost once."""
    parts = [SYSTEM_INSTRUCTIONS, "\nHere are transcripts of several real classes by this teacher, for style and content reference:"]
    for i, example in enumerate(examples, 1):
        parts.append(f"\n--- Example class {i} ---\n{example}")
    return [{"type": "text", "text": "\n".join(parts), "cache_control": {"type": "ephemeral"}}]


def build_user_message(topic: str | None) -> str:
    ask = "Now write a new, original class script in this teacher's style"
    if topic:
        ask += f", focused on {topic}"
    ask += ". Do not copy sentences from the examples - write new instructional content that matches the teacher's voice, structure, and vocabulary."
    return ask


def generate_class(client: anthropic.Anthropic, examples: list[str], model: str, max_tokens: int, topic: str | None) -> str:
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=build_system_prompt(examples),
        messages=[{"role": "user", "content": build_user_message(topic)}],
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
    dst_path: Path,
) -> None:
    # A different seed per class means a different sample of examples - without this, every
    # class in a batch would be grounded in the same three transcripts and read as variations
    # on one style-transfer rather than genuinely separate classes.
    chosen = random.Random(seed).sample(files, num_examples)
    print(f"{dst_path.name}: examples = {', '.join(f.name for f in chosen)}")
    examples = [f.read_text(encoding="utf-8") for f in chosen]

    text = generate_class(client, examples, model, max_tokens, topic)
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
            generate_one(client, files, cfg.seed + i, cfg.num_examples, cfg.model, cfg.max_tokens, cfg.topic, dst_path)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            failed += 1
            continue
        succeeded += 1

    print(f"\n{succeeded} generated, {skipped} skipped (already done), {failed} failed, out of {cfg.num_classes} total")


if __name__ == "__main__":
    main()
