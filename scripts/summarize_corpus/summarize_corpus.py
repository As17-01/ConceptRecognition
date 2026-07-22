import json
import re
import sys

import anthropic
import hydra

from pathlib import Path
from omegaconf import DictConfig

# Structural pause marker; format defined in transcribe.py - keep in sync.
PAUSE_RE = re.compile(r"\[ПАУЗА:(\d+)\]")

MAP_SYSTEM = """You are building a compact reference digest of a single Russian contemporary dance / \
movement improvisation class transcript, for later use as style/content reference when generating new \
classes.

Write a freeform digest of about 150-250 words covering:
- This class's structure and arc (e.g. warm-up, thematic/technical focus, close)
- Recurring themes, images, or metaphors used during instruction
- Characteristic vocabulary, phrasing, and code-switched Russian/English movement terms

Do not count words, pauses, or estimate duration - exact numeric stats for this class are computed \
separately in code. Output only the digest text, no headers or preamble."""

REDUCE_SYSTEM = """You are synthesizing a single corpus-wide digest from per-class digests of many \
Russian contemporary dance / movement improvisation classes by the same teacher, for use as extra \
context when generating new classes in their style.

Write a consolidated digest (roughly 400-1000 words) covering:
- The structure/arc most classes follow
- Recurring vocabulary, imagery, and themes across the corpus
- The typical code-switching style between Russian and English

The prompt also gives you exact numeric stats about the corpus, computed in code - state them \
verbatim in a short concluding paragraph as pacing guidance for someone writing a new class script \
in this style. Do not recompute, round differently, or estimate these numbers yourself."""


def compute_stats(text: str) -> tuple[int, int, int]:
    """Exact word/pause counts in plain Python - cheap enough to redo on every run, including for
    files whose digest already exists and is skipped below (the corpus-wide aggregate still needs
    every file's numbers)."""
    word_count = pause_count = pause_seconds = 0
    for token in text.split():
        match = PAUSE_RE.fullmatch(token)
        if match:
            pause_count += 1
            pause_seconds += int(match.group(1))
        else:
            word_count += 1
    return word_count, pause_count, pause_seconds


def summarize_file(client: anthropic.Anthropic, text: str, model: str, max_tokens: int) -> str:
    # Non-streaming: max_tokens is small (~1024) and each call is independent, well under the
    # ~16000-token threshold where the SDK requires streaming to avoid HTTP timeouts.
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=MAP_SYSTEM,
        messages=[{"role": "user", "content": f"Transcript:\n\n{text}"}],
    )
    return next(block.text for block in message.content if block.type == "text")


def build_reduce_prompt(digests: list[tuple[str, str]], stats: dict) -> str:
    parts = [f"Here are per-class digests for {len(digests)} real classes by the same dance teacher, produced by summarizing each class individually:"]
    for name, digest in digests:
        parts.append(f"\n--- {name} ---\n{digest}")
    parts.append(f"""
GIVEN FACTS about the full corpus (computed exactly in code from the real transcripts - report \
these, do not recompute or estimate them):
- Classes analyzed: {stats["num_files"]}
- Average narrated words per class (excluding pause markers): {stats["avg_words"]:.0f}
- Average number of structural pauses per class: {stats["avg_pause_count"]:.1f}
- Average total pause duration per class: {stats["avg_pause_seconds"]:.0f} seconds
- Assumed speaking rate: {stats["words_per_minute"]:.0f} words/minute

Synthesize ONE consolidated corpus-level digest covering the structure/arc most classes follow, \
recurring vocabulary/imagery/themes across the corpus, and the typical Russian/English \
code-switching style. End with a short concluding paragraph stating the GIVEN FACTS above, as \
pacing guidance for someone writing a new class script in this style.""")
    return "\n".join(parts)


def summarize_corpus(client: anthropic.Anthropic, digests: list[tuple[str, str]], stats: dict, model: str, max_tokens: int) -> str:
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=REDUCE_SYSTEM,
        messages=[{"role": "user", "content": build_reduce_prompt(digests, stats)}],
    ) as stream:
        message = stream.get_final_message()
    return next(block.text for block in message.content if block.type == "text")


@hydra.main(config_path="../conf", config_name="summarize_corpus", version_base=None)
def main(cfg: DictConfig) -> None:
    src_dir = Path(cfg.src)
    if not src_dir.is_dir():
        print(f"Source folder not found: {src_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(src_dir.glob("*.txt"))
    if not files:
        print(f"No transcript files found in {src_dir}")
        return

    summaries_dir = Path(cfg.summaries_dst)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic()

    # Running totals for the corpus-wide aggregate, accumulated every file regardless of whether
    # that file's digest is (re)generated this run or already existed.
    total_words = total_pause_count = total_pause_seconds = n_files = 0
    succeeded, skipped, failed = 0, 0, 0
    for src_path in files:
        dst_path = summaries_dir / src_path.name
        try:
            text = src_path.read_text(encoding="utf-8")
            words, pause_count, pause_seconds = compute_stats(text)
            total_words += words
            total_pause_count += pause_count
            total_pause_seconds += pause_seconds
            n_files += 1

            if dst_path.exists():
                print(f"{src_path.name}: output already exists, skipping")
                skipped += 1
                continue

            digest = summarize_file(client, text, cfg.map_model, cfg.map_max_tokens)
            dst_path.write_text(digest, encoding="utf-8")
            print(f"{src_path.name}: done -> {dst_path}")
        except Exception as e:
            # One bad file shouldn't lose progress on the rest of the batch; log it and move on
            # instead of crashing the job.
            print(f"  FAILED: {e}", file=sys.stderr)
            failed += 1
            continue
        succeeded += 1

    print(f"\n{succeeded} summarized, {skipped} skipped (already done), {failed} failed, out of {len(files)} total")

    if n_files == 0:
        print("No files processed successfully; skipping corpus digest and stats", file=sys.stderr)
        return

    stats = {
        "num_files": n_files,
        "avg_words": total_words / n_files,
        "avg_pause_count": total_pause_count / n_files,
        "avg_pause_seconds": total_pause_seconds / n_files,
        "words_per_minute": float(cfg.words_per_minute),
    }

    digest_path = Path(cfg.digest_dst)
    if digest_path.exists():
        print(f"{digest_path.name}: output already exists, skipping")
    else:
        # Re-glob rather than reuse the map loop's in-memory digests: a prior run may have
        # already produced most of them, so this run's map loop mostly just skipped ahead.
        digest_paths = sorted(summaries_dir.glob("*.txt"))
        digests = [(p.name, p.read_text(encoding="utf-8")) for p in digest_paths]
        if digests:
            print(f"Synthesizing corpus digest from {len(digests)} per-file digests...")
            corpus_digest = summarize_corpus(client, digests, stats, cfg.reduce_model, cfg.reduce_max_tokens)
            digest_path.write_text(corpus_digest, encoding="utf-8")
            print(f"Corpus digest -> {digest_path}")
        else:
            print("No per-file digests available; skipping corpus digest synthesis", file=sys.stderr)

    # Cheap/local and generate_class.py depends on it - written every run regardless of whether
    # the (expensive) digest synthesis above ran or was skipped.
    stats_path = Path(cfg.stats_dst)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Corpus stats -> {stats_path}")


if __name__ == "__main__":
    main()
