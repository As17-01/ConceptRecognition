import sys

import hydra
import torch

from pathlib import Path
from omegaconf import DictConfig
from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.audio import decode_audio

SAMPLE_RATE = 16000


def load_vad(vad_model_dir: str):
    torch.hub.set_dir(vad_model_dir)
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=False, onnx=False)
    get_speech_timestamps, *_ = utils
    return model, get_speech_timestamps


def load_model(model_name: str, model_dir: Path, device: str, compute_type: str) -> BatchedInferencePipeline:
    base_model = WhisperModel(model_name, device=device, compute_type=compute_type, download_root=str(model_dir))
    return BatchedInferencePipeline(model=base_model)


def split_into_clips(speech_timestamps: list[dict], max_clip_samples: int) -> list[dict]:
    """Splits any VAD segment longer than max_clip_samples into consecutive sub-clips. Passing
    clip_timestamps explicitly to transcribe() (done below, so our own VAD - not faster-whisper's
    internal one - decides what counts as speech) bypasses its own <=30s auto-chunking entirely:
    each clip_timestamps entry becomes exactly one model window, and a window longer than the
    model's fixed 30-second input is silently truncated (only the first 30s transcribed, the rest
    dropped) rather than raising an error. This has nothing to do with real pauses - it's purely
    to stay under that hard limit - so it must not be confused with classify_pause_events below,
    which operates on the original, unsplit speech_timestamps."""
    clips = []
    for ts in speech_timestamps:
        start = ts["start"]
        while start < ts["end"]:
            end = min(start + max_clip_samples, ts["end"])
            clips.append({"start": start, "end": end})
            start = end
    return clips


def classify_pause_events(
    speech_timestamps: list[dict], min_pause_seconds: float, min_micro_pause_seconds: float
) -> list[tuple[int, str]]:
    """Returns [(absolute_sample_position, marker_text), ...] in ascending order, one entry for
    every real gap between consecutive VAD speech segments that's at least min_micro_pause_seconds
    (the lower of the two thresholds) - "[ПАУЗА:N]" for gaps >= min_pause_seconds (a real
    movement/music break), "[МИКРОПАУЗА:N]" otherwise (a brief settle/breath pause). position is
    the end of the segment right before the gap.

    Because clip_timestamps (unlike VAD auto-chunking) preserves the original file's timeline -
    transcribe() adds each clip's own absolute offset back onto its returned segment timestamps -
    these positions line up directly with the model's own returned segment timestamps. No
    proportional/approximate placement is needed, unlike the old VAD-only backfill approach."""
    min_micro_pause_samples = int(min_micro_pause_seconds * SAMPLE_RATE)
    min_pause_samples = int(min_pause_seconds * SAMPLE_RATE)
    events = []
    for i in range(1, len(speech_timestamps)):
        gap = speech_timestamps[i]["start"] - speech_timestamps[i - 1]["end"]
        if gap >= min_micro_pause_samples:
            pause_seconds = round(gap / SAMPLE_RATE)
            marker = f"[ПАУЗА:{pause_seconds}]" if gap >= min_pause_samples else f"[МИКРОПАУЗА:{pause_seconds}]"
            events.append((speech_timestamps[i - 1]["end"], marker))
    return events


def keep_segment(seg, max_no_speech_prob: float, min_avg_logprob: float, max_compression_ratio: float) -> bool:
    return (
        seg.no_speech_prob <= max_no_speech_prob
        and seg.avg_logprob >= min_avg_logprob
        and seg.compression_ratio <= max_compression_ratio
    )


def transcribe_mp3(
    mp3_path: Path,
    model: BatchedInferencePipeline,
    vad_model,
    get_speech_timestamps,
    language: str,
    initial_prompt: str,
    condition_on_previous_text: bool,
    max_no_speech_prob: float,
    min_avg_logprob: float,
    max_compression_ratio: float,
    vad_threshold: float,
    vad_min_silence_duration_ms: int,
    vad_speech_pad_ms: int,
    max_clip_seconds: float,
    min_pause_seconds: float,
    min_micro_pause_seconds: float,
    batch_size: int,
) -> tuple[str, int, int]:
    wav = decode_audio(str(mp3_path), sampling_rate=SAMPLE_RATE)
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

    pause_events = classify_pause_events(speech_timestamps, min_pause_seconds, min_micro_pause_seconds)
    clips = split_into_clips(speech_timestamps, int(max_clip_seconds * SAMPLE_RATE))
    clip_timestamps = [{"start": c["start"] / SAMPLE_RATE, "end": c["end"] / SAMPLE_RATE} for c in clips]

    # One call for the whole file: batch_size controls how many of the (potentially many, since
    # every VAD-detected speech burst becomes its own clip) independent clips above get decoded
    # together per forward pass - real parallelism, not just a faster single-chunk engine.
    # initial_prompt is still applied fresh to every clip regardless of batching (each gets its
    # own copy of the same seeded prompt - confirmed in faster_whisper's batched generation path),
    # so this doesn't trade away the "keep the domain-vocabulary hint alive throughout the file"
    # benefit the old per-chunk-call design relied on - if anything it's re-seeded more often now.
    segments, _ = model.transcribe(
        wav,
        language=language,
        initial_prompt=initial_prompt,
        condition_on_previous_text=condition_on_previous_text,
        beam_size=5,
        best_of=5,
        vad_filter=False,
        clip_timestamps=clip_timestamps,
        batch_size=batch_size,
    )

    chunk_texts, total_kept, total_dropped, event_idx = [], 0, 0, 0
    for seg in segments:
        seg_start_samples = seg.start * SAMPLE_RATE
        while event_idx < len(pause_events) and pause_events[event_idx][0] <= seg_start_samples:
            chunk_texts.append(pause_events[event_idx][1])
            event_idx += 1

        if keep_segment(seg, max_no_speech_prob, min_avg_logprob, max_compression_ratio):
            chunk_texts.append(seg.text.strip())
            total_kept += 1
        else:
            total_dropped += 1

    # A trailing pause after the last kept/dropped segment still needs to be emitted.
    while event_idx < len(pause_events):
        chunk_texts.append(pause_events[event_idx][1])
        event_idx += 1

    return " ".join(t for t in chunk_texts if t), total_kept, total_dropped


def transcribe_mp3s(
    src_dir: Path,
    dst_dir: Path,
    model_name: str,
    model_dir: Path,
    device: str,
    compute_type: str,
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
    max_clip_seconds: float,
    min_pause_seconds: float,
    min_micro_pause_seconds: float,
    batch_size: int,
) -> None:
    mp3_files = list(src_dir.glob("*.mp3"))
    if not mp3_files:
        print(f"No MP3 files found in {src_dir}")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Silero VAD model from '{vad_model_dir}'...")
    vad_model, get_speech_timestamps = load_vad(vad_model_dir)

    print(f"Loading faster-whisper model '{model_name}' (device={device}, compute_type={compute_type}) from '{model_dir}'...")
    model = load_model(model_name, model_dir, device, compute_type)

    succeeded, skipped, failed = 0, 0, 0
    for mp3_path in sorted(mp3_files):
        txt_path = dst_dir / mp3_path.with_suffix(".txt").name

        # Makes a 100+ file job resumable: a crash, preemption, or timeout partway through a
        # long run shouldn't force re-transcribing files that already finished.
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
                language,
                initial_prompt,
                condition_on_previous_text,
                max_no_speech_prob,
                min_avg_logprob,
                max_compression_ratio,
                vad_threshold,
                vad_min_silence_duration_ms,
                vad_speech_pad_ms,
                max_clip_seconds,
                min_pause_seconds,
                min_micro_pause_seconds,
                batch_size,
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
        cfg.device,
        cfg.compute_type,
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
        cfg.max_clip_seconds,
        cfg.min_pause_seconds,
        cfg.min_micro_pause_seconds,
        cfg.batch_size,
    )


if __name__ == "__main__":
    main()
