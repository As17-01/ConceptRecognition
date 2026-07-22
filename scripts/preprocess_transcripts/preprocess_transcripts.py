import re
import sys

import hydra
import torch
import whisper.audio

from pathlib import Path
from omegaconf import DictConfig

# Whisper's own punctuation/casing is kept as-is (see preprocess_transcripts.yaml for why the
# RUPunct restoration pass was dropped); this only strips characters Whisper occasionally
# hallucinates from scripts unrelated to the recording (e.g. stray CJK glyphs). Keep Cyrillic,
# Latin (English words/code-switching), digits (counting, dates), whitespace, and the punctuation
# marks actually observed across this corpus's transcripts.
# Square brackets are reserved for structural markers (e.g. the "[ПАУЗА:N]" pause marker emitted
# by transcribe.py) - never emitted by Whisper itself. Hyphen kept last (harmless now that
# NON_TARGET_CHARS below runs this through re.escape(), which makes literal-vs-range ordering
# inside a [...] class a non-issue).
ALLOWED_PUNCT = ",.!?%—–…«»*':;[]-"
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

# format defined in transcribe.py - keep in sync
PAUSE_MARKER_RE = re.compile(r"\[ПАУЗА:(\d+)\]")
SAMPLE_RATE = 16000  # Silero VAD's expected rate, same as transcribe.py uses


def load_vad(vad_model_dir: str):
    torch.hub.set_dir(vad_model_dir)
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=False, onnx=False)
    get_speech_timestamps, *_ = utils
    return model, get_speech_timestamps


def detect_pause_fractions(
    audio_path: Path,
    vad_model,
    get_speech_timestamps,
    vad_threshold: float,
    vad_min_silence_duration_ms: int,
    vad_speech_pad_ms: int,
    min_pause_seconds: float,
) -> list[tuple[float, int]]:
    """Returns [(fraction_of_total_VAD_speech_elapsed_before_the_pause, pause_seconds), ...] in
    ascending fraction order. This is an approximation for backfilling pause markers into
    transcripts that predate transcribe.py's native pause-marker support: it assumes roughly
    uniform speaking rate across the file to convert "this pause happened after X% of the total
    detected speech time" into "insert the marker after X% of the transcript's words" - good
    enough for downstream few-shot style/pacing reference, not claimed to be exact."""
    wav = torch.from_numpy(whisper.audio.load_audio(str(audio_path), sr=SAMPLE_RATE))
    speech_timestamps = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=vad_threshold,
        min_silence_duration_ms=vad_min_silence_duration_ms,
        speech_pad_ms=vad_speech_pad_ms,
    )
    if len(speech_timestamps) < 2:
        return []

    total_speech = sum(ts["end"] - ts["start"] for ts in speech_timestamps)
    min_pause_samples = int(min_pause_seconds * SAMPLE_RATE)
    pauses = []
    cumulative = 0
    for i, ts in enumerate(speech_timestamps):
        if i > 0:
            gap = ts["start"] - speech_timestamps[i - 1]["end"]
            if gap >= min_pause_samples:
                pauses.append((cumulative / total_speech, round(gap / SAMPLE_RATE)))
        cumulative += ts["end"] - ts["start"]
    return pauses


def insert_pause_markers(words: list[str], pauses: list[tuple[float, int]]) -> list[str]:
    """Inserts a "[ПАУЗА:N]" token into the word list at the position proportional to how far
    through the total VAD speech time each real pause occurred (see detect_pause_fractions).
    offset tracks how many markers have already been inserted so later insertion indices
    correctly account for the words list having grown."""
    result = list(words)
    total_words = len(words)
    for offset, (fraction, pause_seconds) in enumerate(pauses):
        index = min(round(fraction * total_words) + offset, len(result))
        result.insert(index, f"[ПАУЗА:{pause_seconds}]")
    return result


def clean_text(text: str) -> str:
    text = SUBTITLE_CREDITS.sub(" ", text)
    text = NON_TARGET_CHARS.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_word(word: str) -> str:
    return word.strip(ALLOWED_PUNCT).lower()


def collapse_repeated_ngrams(words: list[str], max_ngram: int, min_count: int) -> list[str]:
    """Compares words by normalize_word (punctuation-stripped, lowercased) rather than raw text,
    since Whisper's own punctuation on a hallucinated repeat isn't always identical run to run
    (e.g. "музыка." vs "музыка,"); the original text of the kept (first) occurrence is preserved.

    Pause markers get a unique, never-equal sentinel key instead: several distinct real pauses
    that happen to round to the same duration (e.g. three separate 10s breaks close together)
    are a plausible legitimate sequence, not a Whisper hallucination loop, and must never be
    collapsed down to one."""
    keys = [object() if PAUSE_MARKER_RE.fullmatch(w) else normalize_word(w) for w in words]
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
    audio_src: Path,
    vad_model,
    get_speech_timestamps,
    vad_threshold: float,
    vad_min_silence_duration_ms: int,
    vad_speech_pad_ms: int,
    min_pause_seconds: float,
) -> None:
    text = src_path.read_text(encoding="utf-8")
    words = clean_text(text).split()
    if not words:
        print(f"{src_path.name}: no words found, skipping")
        return

    # Fresh transcribe.py runs already embed real markers natively - only legacy transcripts
    # (predating that support) get VAD-only backfill, so a file with a marker is left untouched.
    if not PAUSE_MARKER_RE.search(text):
        audio_path = audio_src / f"{src_path.stem}.mp3"
        if audio_path.exists():
            try:
                pauses = detect_pause_fractions(
                    audio_path,
                    vad_model,
                    get_speech_timestamps,
                    vad_threshold,
                    vad_min_silence_duration_ms,
                    vad_speech_pad_ms,
                    min_pause_seconds,
                )
                if pauses:
                    words = insert_pause_markers(words, pauses)
            except Exception as e:
                # A corrupt audio file or VAD failure for one recording shouldn't fail that
                # file's entire preprocessing - just carry on without markers.
                print(f"{src_path.name}: VAD pause backfill failed: {e}", file=sys.stderr)
        else:
            print(f"{src_path.name}: no matching audio at {audio_path}, skipping pause backfill")

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
    audio_src = Path(cfg.audio_src)

    # Loaded once up front (same pattern as transcribe.py) rather than per-file - Silero VAD is
    # small/cheap, but there's no reason to reload it 129 times.
    print(f"Loading Silero VAD model from '{cfg.vad_model_dir}'...")
    vad_model, get_speech_timestamps = load_vad(cfg.vad_model_dir)

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
                audio_src,
                vad_model,
                get_speech_timestamps,
                cfg.vad_threshold,
                cfg.vad_min_silence_duration_ms,
                cfg.vad_speech_pad_ms,
                cfg.min_pause_seconds,
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
