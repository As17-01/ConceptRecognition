import re
import sys

import hydra

from pathlib import Path
from omegaconf import DictConfig

# Whisper's own punctuation/casing is kept as-is (see preprocess_transcripts.yaml for why the
# RUPunct restoration pass was dropped); this only strips characters Whisper occasionally
# hallucinates from scripts unrelated to the recording (e.g. stray CJK glyphs). Keep Cyrillic,
# Latin (English words/code-switching), digits (counting, dates), whitespace, and the punctuation
# marks actually observed across this corpus's transcripts. Hyphen is kept last (harmless now
# that NON_TARGET_CHARS below runs this through re.escape(), which makes literal-vs-range ordering
# inside a [...] class a non-issue).
ALLOWED_PUNCT = ",.!?%—–…«»*':;-"
NON_TARGET_CHARS = re.compile(rf"[^0-9A-Za-zА-Яа-яЁё\s{re.escape(ALLOWED_PUNCT)}]")

# Standalone disfluencies only; never matched as a substring, so English words and
# digits are never touched.
FILLER_WORDS = {"ну", "э", "эм", "эмм", "ммм", "мм", "ыыы", "эээ", "ээ", "um", "umm", "uh", "uhh"}

# Real code-switching happens at the word level ("thread of the arms"), never mid-word -
# a token with both scripts glued together (e.g. "бедраader") is a reliable Whisper
# hallucination signature, not a genuine word.
CYRILLIC_CHAR = re.compile(r"[А-Яа-яЁё]")
LATIN_CHAR = re.compile(r"[A-Za-z]")

# Whisper hallucinates YouTube-style subtitle-credit lines mid-transcript (found in 37 of 129
# files) - always sitting cleanly at a sentence boundary, so removing them doesn't disturb
# surrounding text. Names vary (editors: "А.Семкин", "М.Лосева"; always "Корректор А.Егорова" so
# far), and the verb before "DimaTorzok" varies too (делал/сделал/создавал/создал observed) - so
# this matches both patterns generically (by position, not by enumerating every verb form) rather
# than hardcoding names or exact wording.
SUBTITLE_CREDITS = re.compile(
    r"[Рр]едактор субтитров\s+\S+\s+[Кк]орректор\s+\S+\.?|[Сс]убтитры\s+\S+\s+DimaTorzok\.?"
)

# Whisper also hallucinates transcribe.py's own initial_prompt text back as fake "spoken" content
# (found near-verbatim in dozens of files, both glued to a SUBTITLE_CREDITS artifact and entirely
# standalone) - a known failure mode where a low-confidence moment causes the model to echo its
# own prompt instead of admitting uncertainty. The echo is often fragmentary/reordered (e.g.
# "Практика контемпорари и двигательной импульс, амплитуда, ..." skips "импровизации. Термины:
# вытяжение, ось, текстура," entirely), so this matches on DENSITY of known prompt vocabulary
# packed together with only light connectors between terms (comma/period/colon/"и"), rather than
# trying to match the prompt text exactly. Keep this list in sync with transcribe.yaml's
# initial_prompt - not shared code, no common module in this codebase (see other scripts).
#
# A plain "3+ terms in a row" rule is too aggressive on its own - tested against plausible real
# instruction ("Проверяем: ось, бедра, копчик.", a genuine anatomical checklist) and it wrongly
# stripped realistic sentences, since most of these terms (flow, vibe, range, бедра, ось,
# амплитуда, многозадачность, копчик, текстура) are ordinary enough to legitimately appear listed
# together. So a candidate span is only actually removed if it ALSO contains at least one term
# from DISTINCTIVE_ANCHOR - kept deliberately narrow, to only "кагами" (unusual enough that it's
# essentially never going to appear as coincidental real content) and the two distinctive opening-
# sentence fragments. "thread of the arms/legs" and "full body experience" are deliberately NOT
# anchors despite being English phrases - they're established, genuinely recurring real content in
# this corpus (that's why they were in initial_prompt from the very start), so gating on them risks
# stripping real speech the same way the ordinary single words did. This narrow anchor accepts more
# false negatives (a hallucinated echo that happens to skip "кагами" and both opening fragments
# won't be caught) in exchange for far fewer false positives - reasonable given VAD tuning is now
# the primary defense (see transcribe.yaml's vad_threshold) and this is just a cleanup safety net
# for whatever slips through, not the main defense. Unanchored comma-list echoes of the prompt
# vocabulary block are handled separately by strip_prompt_vocab_list_echo below.
PROMPT_TERM = (
    r"вытяжени[ея]|ось|текстура|импульс|амплитуда|многозадачность|бедра|копчик|кагами|"
    r"практика\s+контемпорари|двигательной(?:\s+импровизации)?|термины|"
    r"thread of the arms|thread of the legs|full body experience|range|flow|vibe|groove"
)
PROMPT_ECHO_CANDIDATE = re.compile(
    rf"(?:\b(?:{PROMPT_TERM})\b[,.:]?\s*(?:и\s+)?){{3,}}",
    re.IGNORECASE,
)
DISTINCTIVE_ANCHOR = re.compile(
    r"кагами|практика\s+контемпорари|двигательной\s+импровизации",
    re.IGNORECASE,
)

# Second pass for unanchored echoes: Whisper often replays the comma-separated vocabulary tail of
# transcribe.yaml's initial_prompt verbatim ("Термины, ось, текстура, ... flow, vibe, groove.")
# without ever hitting DISTINCTIVE_ANCHOR. Real instruction may list a few anatomical cues but
# won't open with "Термины," / "Вытяжение," and then enumerate 4+ core prompt terms in comma-list
# form; gate on that density instead of a single rare anchor word.
PROMPT_LIST_HEAD = r"(?:[Тт]ермины|[Вв]ытяжени[ея])"
PROMPT_LIST_ITEM = (
    r"вытяжени[ея]|ось|текстура|импульс|амплитуда|многозадачность|бедра|копчик|кагами|"
    r"практика\s+контемпорари|двигательной(?:\s+импровизации)?|"
    r"thread of the arms|thread of the legs|full body experience|range|flow|vibe|groove"
)
PROMPT_VOCAB_LIST = re.compile(
    rf"\b{PROMPT_LIST_HEAD}[,:]?\s*(?:\b(?:{PROMPT_LIST_ITEM})\b\s*,\s*)+(?:\b(?:{PROMPT_LIST_ITEM})\b)\s*\.?",
    re.IGNORECASE,
)
PROMPT_LIST_CORE = re.compile(
    r"\b(?:ось|текстура|импульс|амплитуда|многозадачность|бедра)\b",
    re.IGNORECASE,
)
PROMPT_LIST_ENGLISH_TAIL = re.compile(r"\b(?:flow|vibe|groove)\b", re.IGNORECASE)


def is_clear_prompt_vocab_list(span: str) -> bool:
    core_hits = PROMPT_LIST_CORE.findall(span)
    return len(core_hits) >= 4 or (len(core_hits) >= 3 and PROMPT_LIST_ENGLISH_TAIL.search(span))


def strip_prompt_vocab_list_echo(text: str) -> str:
    return PROMPT_VOCAB_LIST.sub(lambda m: " " if is_clear_prompt_vocab_list(m.group()) else m.group(), text)


def strip_prompt_echo(text: str) -> str:
    text = PROMPT_ECHO_CANDIDATE.sub(
        lambda m: " " if DISTINCTIVE_ANCHOR.search(m.group()) else m.group(), text
    )
    return strip_prompt_vocab_list_echo(text)

def clean_text(text: str) -> str:
    text = SUBTITLE_CREDITS.sub(" ", text)
    text = strip_prompt_echo(text)
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
