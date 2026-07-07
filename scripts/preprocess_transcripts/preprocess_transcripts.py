import re
import sys

import hydra
import torch

from pathlib import Path
from omegaconf import DictConfig
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Whisper occasionally hallucinates glyphs from scripts unrelated to the recording
# (e.g. stray CJK characters); keep Cyrillic, Latin (English words/code-switching),
# digits (counting, dates), and whitespace, drop everything else including the
# original sparse/inconsistent punctuation so the punctuation model gets a clean slate.
NON_TARGET_CHARS = re.compile(r"[^0-9A-Za-zА-Яа-яЁё\s]")

# Standalone disfluencies only; never matched as a substring, so English words and
# digits are never touched.
FILLER_WORDS = {"ну", "э", "эм", "эмм", "ммм", "мм", "ыыы", "эээ", "ээ", "um", "umm", "uh", "uhh"}

# Real code-switching happens at the word level ("thread of the arms"), never mid-word -
# a token with both scripts glued together (e.g. "бедраader") is a reliable Whisper
# hallucination signature, not a genuine word.
CYRILLIC_CHAR = re.compile(r"[А-Яа-яЁё]")
LATIN_CHAR = re.compile(r"[A-Za-z]")

# Mirrors the inference snippet in the RUPunct_big model card.
LABEL_TO_FORMATTER = {
    "LOWER_O": lambda t: t,
    "LOWER_PERIOD": lambda t: t + ".",
    "LOWER_COMMA": lambda t: t + ",",
    "LOWER_QUESTION": lambda t: t + "?",
    "LOWER_TIRE": lambda t: t + "—",
    "LOWER_DVOETOCHIE": lambda t: t + ":",
    "LOWER_VOSKL": lambda t: t + "!",
    "LOWER_PERIODCOMMA": lambda t: t + ";",
    "LOWER_DEFIS": lambda t: t + "-",
    "LOWER_MNOGOTOCHIE": lambda t: t + "...",
    "LOWER_QUESTIONVOSKL": lambda t: t + "?!",
    "UPPER_O": lambda t: t.capitalize(),
    "UPPER_PERIOD": lambda t: t.capitalize() + ".",
    "UPPER_COMMA": lambda t: t.capitalize() + ",",
    "UPPER_QUESTION": lambda t: t.capitalize() + "?",
    "UPPER_TIRE": lambda t: t.capitalize() + " —",
    "UPPER_DVOETOCHIE": lambda t: t.capitalize() + ":",
    "UPPER_VOSKL": lambda t: t.capitalize() + "!",
    "UPPER_PERIODCOMMA": lambda t: t.capitalize() + ";",
    "UPPER_DEFIS": lambda t: t.capitalize() + "-",
    "UPPER_MNOGOTOCHIE": lambda t: t.capitalize() + "...",
    "UPPER_QUESTIONVOSKL": lambda t: t.capitalize() + "?!",
    "UPPER_TOTAL_O": lambda t: t.upper(),
    "UPPER_TOTAL_PERIOD": lambda t: t.upper() + ".",
    "UPPER_TOTAL_COMMA": lambda t: t.upper() + ",",
    "UPPER_TOTAL_QUESTION": lambda t: t.upper() + "?",
    "UPPER_TOTAL_TIRE": lambda t: t.upper() + " —",
    "UPPER_TOTAL_DVOETOCHIE": lambda t: t.upper() + ":",
    "UPPER_TOTAL_VOSKL": lambda t: t.upper() + "!",
    "UPPER_TOTAL_PERIODCOMMA": lambda t: t.upper() + ";",
    "UPPER_TOTAL_DEFIS": lambda t: t.upper() + "-",
    "UPPER_TOTAL_MNOGOTOCHIE": lambda t: t.upper() + "...",
    "UPPER_TOTAL_QUESTIONVOSKL": lambda t: t.upper() + "?!",
}


def clean_text(text: str) -> str:
    text = NON_TARGET_CHARS.sub(" ", text)
    # RUPunct's LOWER_* labels are a no-op on the original word (they assume it's already
    # lowercase); since Whisper's own casing is no longer reliably absent (the new transcribe.py
    # produces decent native casing), leaving it in place would let words Whisper happened to
    # capitalize mid-sentence stay wrongly capitalized instead of being corrected by the model.
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def collapse_repeated_ngrams(words: list[str], max_ngram: int, min_count: int) -> list[str]:
    result = []
    i = 0
    n_words = len(words)
    while i < n_words:
        collapsed = False
        for n in range(min(max_ngram, n_words - i), 0, -1):
            ngram = words[i : i + n]
            repeats = 1
            j = i + n
            while j + n <= n_words and words[j : j + n] == ngram:
                repeats += 1
                j += n
            if repeats >= min_count:
                result.extend(ngram)
                i = j
                collapsed = True
                break
        if not collapsed:
            result.append(words[i])
            i += 1
    return result


def remove_fillers(words: list[str]) -> list[str]:
    return [w for w in words if w.lower() not in FILLER_WORDS]


def remove_mixed_script_tokens(words: list[str]) -> list[str]:
    return [w for w in words if not (CYRILLIC_CHAR.search(w) and LATIN_CHAR.search(w))]


def restore_punctuation(words: list[str], classifier, window_words: int, overlap_words: int) -> str:
    """Classifies in overlapping windows and keeps only each window's high-context "core"
    (the model has no context outside the window it's given, so the words right at a window's
    edge are the ones most likely to get punctuated as if sentence-initial when they're not).
    The overlap region is covered twice, by two different windows, and only the copy with the
    most surrounding context on both sides is kept - this is what makes sentence-boundary
    punctuation reliable across window splits, which matters since semantic_chunk.py treats
    sentences as its atomic unit.

    The model groups adjacent same-label words into a single prediction unit, and which words
    end up in the same group is itself context-dependent - the same word can be grouped
    differently by two different (overlapping) windows. Deciding what to keep by each group's
    own midpoint is therefore unsafe: a group straddling the seam between two windows can fall
    outside *both* windows' keep range and silently vanish. Reconstructing each window's full
    text first and only then slicing it by plain word position sidesteps this entirely, since
    that slicing no longer depends on how any window happened to group its words.
    """
    n_words = len(words)
    if n_words == 0:
        return ""

    half_overlap = overlap_words // 2
    stride = window_words - overlap_words

    parts = []
    start = 0
    while True:
        end = min(start + window_words, n_words)
        preds = classifier(" ".join(words[start:end]))
        formatted_words = " ".join(LABEL_TO_FORMATTER.get(p["entity_group"], lambda t: t)(p["word"].strip()) for p in preds).split()

        local_lo = 0 if start == 0 else half_overlap
        local_hi = (end - start) if end == n_words else (end - start) - half_overlap
        parts.extend(formatted_words[local_lo:local_hi])

        if end == n_words:
            break
        start += stride

    text = " ".join(parts)
    # LOWER_DEFIS glues "-" to the preceding word expecting it to also glue to the next one,
    # but the join above always inserts a space; LOWER_TIRE has the opposite problem, missing
    # the leading space that UPPER_TIRE/UPPER_TOTAL_TIRE include. Neither "-" nor "—" can occur
    # from any other source (clean_text strips both from the raw input before classification),
    # so these substitutions can't collide with anything else in the text.
    text = re.sub(r"-\s+", "-", text)
    text = re.sub(r"\s*—\s*", " — ", text)
    # RUPunct consistently mispredicts LOWER_DEFIS on these domain-specific cases:
    # "чуть-чуть" is split as two plain words, and "плие" (a ballet term used constantly
    # in these recordings) gets incorrectly hyphenated to whatever clause follows it.
    text = re.sub(r"\bчуть-чуть\b|\bчуть чуть\b", "чуть-чуть", text)
    text = re.sub(r"\bплие-", "плие, ", text)
    return text


def preprocess_transcript(
    src_path: Path,
    dst_dir: Path,
    classifier,
    window_words: int,
    overlap_words: int,
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

    restored = restore_punctuation(words, classifier, window_words, overlap_words)

    dst_path = dst_dir / src_path.name
    dst_path.write_text(restored, encoding="utf-8")
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

    print(f"Loading punctuation model '{cfg.model}' from '{cfg.model_dir}'...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, cache_dir=cfg.model_dir, strip_accents=False, add_prefix_space=True)
    # The saved tokenizer config leaves this unset (effectively infinite) even though the model
    # itself hard-caps at 512 position embeddings, so without this, a window that happens to
    # tokenize past 512 subwords would crash instead of truncating.
    tokenizer.model_max_length = 512
    model = AutoModelForTokenClassification.from_pretrained(cfg.model, cache_dir=cfg.model_dir)
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first", device=device)

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
                classifier,
                cfg.window_words,
                cfg.overlap_words,
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
