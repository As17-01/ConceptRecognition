import re
import sys

import hydra

from pathlib import Path
from omegaconf import DictConfig

# Whisper's own punctuation/casing is kept as-is (see preprocess_transcripts.yaml for why the
# RUPunct restoration pass was dropped); this only strips characters Whisper occasionally
# hallucinates from scripts unrelated to the recording (e.g. stray CJK glyphs). Keep Cyrillic,
# Latin (English words/code-switching), digits (counting, dates), whitespace, and the punctuation
# marks actually observed across this corpus's transcripts.
ALLOWED_PUNCT = ",.!?%—–…«»*':;-"  # hyphen last so it's literal, not a range, inside a [...] class
NON_TARGET_CHARS = re.compile(rf"[^0-9A-Za-zА-Яа-яЁё\s{ALLOWED_PUNCT}]")

# Standalone disfluencies only; never matched as a substring, so English words and
# digits are never touched.
FILLER_WORDS = {"ну", "э", "эм", "эмм", "ммм", "мм", "ыыы", "эээ", "ээ", "um", "umm", "uh", "uhh"}

# Real code-switching happens at the word level ("thread of the arms"), never mid-word -
# a token with both scripts glued together (e.g. "бедраader") is a reliable Whisper
# hallucination signature, not a genuine word.
CYRILLIC_CHAR = re.compile(r"[А-Яа-яЁё]")
LATIN_CHAR = re.compile(r"[A-Za-z]")


def clean_text(text: str) -> str:
    text = NON_TARGET_CHARS.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_word(word: str) -> str:
    return word.strip(ALLOWED_PUNCT).lower()


def collapse_repeated_ngrams(words: list[str], max_ngram: int, min_count: int) -> list[str]:
    """Compares words by normalize_word (punctuation-stripped, lowercased) rather than raw text,
    since Whisper's own punctuation on a hallucinated repeat isn't always identical run to run
    (e.g. "музыка." vs "музыка,"); the original text of the kept (first) occurrence is preserved."""
    keys = [normalize_word(w) for w in words]
    result = []
    i = 0
    n_words = len(words)
    while i < n_words:
        collapsed = False
        for n in range(min(max_ngram, n_words - i), 0, -1):
            key_ngram = keys[i : i + n]
            repeats = 1
            j = i + n
            while j + n <= n_words and keys[j : j + n] == key_ngram:
                repeats += 1
                j += n
            if repeats >= min_count:
                result.extend(words[i : i + n])
                i = j
                collapsed = True
                break
        if not collapsed:
            result.append(words[i])
            i += 1
    return result


def remove_fillers(words: list[str]) -> list[str]:
    return [w for w in words if normalize_word(w) not in FILLER_WORDS]


def remove_mixed_script_tokens(words: list[str]) -> list[str]:
    return [w for w in words if not (CYRILLIC_CHAR.search(w) and LATIN_CHAR.search(w))]


def preprocess_transcript(
    src_path: Path,
    dst_dir: Path,
    max_ngram: int,
    repeat_min_count: int,
    drop_fillers: bool,
    drop_mixed_script: bool,
) -> None:
    text = src_path.read_text(encoding="utf-8")
    words = clean_text(text).split()
    if not words:
        print(f"{src_path.name}: no words found, skipping")
        return

    if drop_mixed_script:
        words = remove_mixed_script_tokens(words)
    words = collapse_repeated_ngrams(words, max_ngram, repeat_min_count)
    if drop_fillers:
        words = remove_fillers(words)

    dst_path = dst_dir / src_path.name
    dst_path.write_text(" ".join(words), encoding="utf-8")
    print(f"{src_path.name}: {len(words)} words -> {dst_path}")


@hydra.main(config_path="../conf", config_name="preprocess_transcripts", version_base=None)
def main(cfg: DictConfig) -> None:
    src_dir = Path(cfg.src)
    dst_dir = Path(cfg.dst)

    if not src_dir.is_dir():
        print(f"Source folder not found: {src_dir}", file=sys.stderr)
        sys.exit(1)

    txt_files = sorted(src_dir.glob("*.txt"))
    if not txt_files:
        print(f"No transcript files found in {src_dir}")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    succeeded, skipped, failed = 0, 0, 0
    for txt_path in txt_files:
        dst_path = dst_dir / txt_path.name

        # Makes a 100+ file job resumable: a crash or timeout partway through shouldn't force
        # re-running files that already finished.
        if dst_path.exists():
            print(f"{txt_path.name}: output already exists, skipping ({dst_path})")
            skipped += 1
            continue

        try:
            preprocess_transcript(
                txt_path,
                dst_dir,
                cfg.max_ngram,
                cfg.repeat_min_count,
                cfg.remove_fillers,
                cfg.remove_mixed_script,
            )
        except Exception as e:
            # One bad file shouldn't lose progress on the rest of the batch; log it and move on
            # instead of crashing the job.
            print(f"  FAILED: {e}", file=sys.stderr)
            failed += 1
            continue
        succeeded += 1

    print(f"\n{succeeded} preprocessed, {skipped} skipped (already done), {failed} failed, out of {len(txt_files)} total")


if __name__ == "__main__":
    main()
