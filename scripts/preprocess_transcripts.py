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


def restore_punctuation(words: list[str], classifier, window_words: int) -> str:
    parts = []
    for i in range(0, len(words), window_words):
        window_text = " ".join(words[i : i + window_words])
        if not window_text:
            continue
        preds = classifier(window_text)
        formatted = [LABEL_TO_FORMATTER.get(p["entity_group"], lambda t: t)(p["word"].strip()) for p in preds]
        parts.append(" ".join(formatted))
    return " ".join(parts)


def preprocess_transcript(
    src_path: Path,
    dst_dir: Path,
    classifier,
    window_words: int,
    max_ngram: int,
    repeat_min_count: int,
    drop_fillers: bool,
) -> None:
    text = src_path.read_text(encoding="utf-8")
    words = clean_text(text).split()
    if not words:
        print(f"{src_path.name}: no words found, skipping")
        return

    words = collapse_repeated_ngrams(words, max_ngram, repeat_min_count)
    if drop_fillers:
        words = remove_fillers(words)

    restored = restore_punctuation(words, classifier, window_words)

    dst_path = dst_dir / src_path.name
    dst_path.write_text(restored, encoding="utf-8")
    print(f"{src_path.name}: {len(words)} words -> {dst_path}")


@hydra.main(config_path="conf", config_name="preprocess_transcripts", version_base=None)
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
    model = AutoModelForTokenClassification.from_pretrained(cfg.model, cache_dir=cfg.model_dir)
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first", device=device)

    for txt_path in txt_files:
        preprocess_transcript(txt_path, dst_dir, classifier, cfg.window_words, cfg.max_ngram, cfg.repeat_min_count, cfg.remove_fillers)


if __name__ == "__main__":
    main()
