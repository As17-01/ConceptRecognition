import sys

import hydra
import torch
import whisper
import whisper.audio

from pathlib import Path
from omegaconf import DictConfig

SAMPLE_RATE = 16000


def load_vad(vad_model_dir: str):
    torch.hub.set_dir(vad_model_dir)
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=False, onnx=False)
    get_speech_timestamps, _, _, _, collect_chunks = utils
    return model, get_speech_timestamps, collect_chunks


def pack_speech_chunks(speech_timestamps: list[dict], max_samples: int, min_split_samples: int) -> list[dict]:
    """Greedily groups VAD speech segments (in original order) into chunks no longer than
    max_samples of *speech* audio each, splitting any single segment that alone exceeds
    max_samples. This bounds how long Whisper goes without a fresh initial_prompt seed. Also
    splits a new group whenever the gap since the previous segment is at least
    min_split_samples - these are real, VAD-confirmed silences (VAD already merges anything
    below vad_min_silence_duration_ms into one segment). min_split_samples is the lower of the
    two pause thresholds, so every gap worth marking as EITHER pause type becomes its own group
    boundary - the caller classifies each boundary's exact gap length into the right marker text.
    Each returned group carries "pause_before": the sample-length of that real pause immediately
    preceding it (0 for the first group, and for splits caused only by max_samples overflow)."""
    groups = []
    current, current_len = [], 0
    pending_pause = 0
    prev_end = None

    def flush():
        nonlocal current, current_len, pending_pause
        if current:
            groups.append({"segments": current, "pause_before": pending_pause})
            current, current_len, pending_pause = [], 0, 0

    for ts in speech_timestamps:
        gap = ts["start"] - prev_end if prev_end is not None else 0
        is_real_pause = prev_end is not None and gap >= min_split_samples
        seg_len = ts["end"] - ts["start"]

        if seg_len > max_samples:
            flush()
            if is_real_pause:
                pending_pause = gap
            start = ts["start"]
            while start < ts["end"]:
                end = min(start + max_samples, ts["end"])
                groups.append({"segments": [{"start": start, "end": end}], "pause_before": pending_pause})
                pending_pause = 0
                start = end
            prev_end = ts["end"]
            continue

        if is_real_pause or current_len + seg_len > max_samples:
            flush()
            if is_real_pause:
                pending_pause = gap

        current.append(ts)
        current_len += seg_len
        prev_end = ts["end"]

    flush()
    return groups


def filter_hallucinated_segments(
    segments: list[dict], max_no_speech_prob: float, min_avg_logprob: float, max_compression_ratio: float
) -> tuple[str, int, int]:
    kept, dropped = [], 0
    for seg in segments:
        if seg["no_speech_prob"] <= max_no_speech_prob and seg["avg_logprob"] >= min_avg_logprob and seg["compression_ratio"] <= max_compression_ratio:
            kept.append(seg["text"])
        else:
            dropped += 1
    return "".join(kept).strip(), len(kept), dropped


def transcribe_mp3(
    mp3_path: Path,
    model,
    vad_model,
    get_speech_timestamps,
    collect_chunks,
    language: str,
    initial_prompt: str,
    condition_on_previous_text: bool,
    max_no_speech_prob: float,
    min_avg_logprob: float,
    max_compression_ratio: float,
    vad_threshold: float,
    vad_min_silence_duration_ms: int,
    vad_speech_pad_ms: int,
    max_chunk_seconds: float,
    min_pause_seconds: float,
    min_micro_pause_seconds: float,
) -> tuple[str, int, int]:
    wav = torch.from_numpy(whisper.audio.load_audio(str(mp3_path), sr=SAMPLE_RATE))
    speech_timestamps = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=vad_threshold,
        min_silence_duration_ms=vad_min_silence_duration_ms,
        speech_pad_ms=vad_speech_pad_ms,
    )
    if not speech_timestamps:
        # Nothing detected as speech; fall back to the full audio rather than transcribing nothing.
        speech_timestamps = [{"start": 0, "end": len(wav)}]

    max_samples = int(max_chunk_seconds * SAMPLE_RATE)
    min_pause_samples = int(min_pause_seconds * SAMPLE_RATE)
    min_micro_pause_samples = int(min_micro_pause_seconds * SAMPLE_RATE)
    chunk_groups = pack_speech_chunks(speech_timestamps, max_samples, min_micro_pause_samples)

    chunk_texts, total_kept, total_dropped = [], 0, 0
    for group in chunk_groups:
        if group["pause_before"] > 0:
            # Pause marker (structural or micro); format defined in transcribe.py - keep in sync.
            # A standalone entry so " ".join(...) below places it as its own atomic token.
            pause_seconds = round(group["pause_before"] / SAMPLE_RATE)
            if group["pause_before"] >= min_pause_samples:
                chunk_texts.append(f"[ПАУЗА:{pause_seconds}]")
            else:
                chunk_texts.append(f"[МИКРОПАУЗА:{pause_seconds}]")

        audio = collect_chunks(group["segments"], wav).numpy()
        result = model.transcribe(
            audio,
            language=language,
            initial_prompt=initial_prompt,
            condition_on_previous_text=condition_on_previous_text,
            # whisper's CLI defaults to beam search (vs greedy decoding) but the Python API
            # does not inherit that default, so it's set explicitly here.
            beam_size=5,
            best_of=5,
        )
        text, kept, dropped = filter_hallucinated_segments(result["segments"], max_no_speech_prob, min_avg_logprob, max_compression_ratio)
        chunk_texts.append(text)
        total_kept += kept
        total_dropped += dropped

    return " ".join(t for t in chunk_texts if t), total_kept, total_dropped


def transcribe_mp3s(
    src_dir: Path,
    dst_dir: Path,
    model_name: str,
    model_dir: Path,
    language: str,
    initial_prompt: str,
    condition_on_previous_text: bool,
    max_no_speech_prob: float,
    min_avg_logprob: float,
    max_compression_ratio: float,
    vad_model_dir: str,
    vad_threshold: float,
    vad_min_silence_duration_ms: int,
    vad_speech_pad_ms: int,
    max_chunk_seconds: float,
    min_pause_seconds: float,
    min_micro_pause_seconds: float,
) -> None:
    mp3_files = list(src_dir.glob("*.mp3"))
    if not mp3_files:
        print(f"No MP3 files found in {src_dir}")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Silero VAD model from '{vad_model_dir}'...")
    vad_model, get_speech_timestamps, collect_chunks = load_vad(vad_model_dir)

    print(f"Loading whisper model '{model_name}' from '{model_dir}'...")
    model = whisper.load_model(model_name, download_root=str(model_dir))

    succeeded, skipped, failed = 0, 0, 0
    for mp3_path in sorted(mp3_files):
        txt_path = dst_dir / mp3_path.with_suffix(".txt").name

        # Makes a 100+ file job resumable: a crash, preemption, or timeout partway through a
        # 72-hour run shouldn't force re-transcribing files that already finished.
        if txt_path.exists():
            print(f"{mp3_path.name}: output already exists, skipping ({txt_path})")
            skipped += 1
            continue

        print(f"{mp3_path.name} -> {txt_path}")
        try:
            text, kept, dropped = transcribe_mp3(
                mp3_path,
                model,
                vad_model,
                get_speech_timestamps,
                collect_chunks,
                language,
                initial_prompt,
                condition_on_previous_text,
                max_no_speech_prob,
                min_avg_logprob,
                max_compression_ratio,
                vad_threshold,
                vad_min_silence_duration_ms,
                vad_speech_pad_ms,
                max_chunk_seconds,
                min_pause_seconds,
                min_micro_pause_seconds,
            )
        except Exception as e:
            # One bad file (corrupt audio, an unexpected edge case) shouldn't lose progress on
            # the rest of a multi-hour batch; log it and move on instead of crashing the job.
            print(f"  FAILED: {e}", file=sys.stderr)
            failed += 1
            continue

        txt_path.write_text(text, encoding="utf-8")
        print(f"  done ({kept} segments kept, {dropped} dropped as likely hallucinations)")
        succeeded += 1

    print(f"\n{succeeded} transcribed, {skipped} skipped (already done), {failed} failed, out of {len(mp3_files)} total")


@hydra.main(config_path="../conf", config_name="transcribe", version_base=None)
def main(cfg: DictConfig) -> None:
    src_dir = Path(cfg.src)
    dst_dir = Path(cfg.dst)
    model_dir = Path(cfg.model_dir)

    if not src_dir.is_dir():
        print(f"Source folder not found: {src_dir}", file=sys.stderr)
        sys.exit(1)

    transcribe_mp3s(
        src_dir,
        dst_dir,
        cfg.model,
        model_dir,
        cfg.language,
        cfg.initial_prompt,
        cfg.condition_on_previous_text,
        cfg.max_no_speech_prob,
        cfg.min_avg_logprob,
        cfg.max_compression_ratio,
        cfg.vad_model_dir,
        cfg.vad_threshold,
        cfg.vad_min_silence_duration_ms,
        cfg.vad_speech_pad_ms,
        cfg.max_chunk_seconds,
        cfg.min_pause_seconds,
        cfg.min_micro_pause_seconds,
    )


if __name__ == "__main__":
    main()
