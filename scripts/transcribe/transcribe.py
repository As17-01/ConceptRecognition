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


def pack_speech_chunks(speech_timestamps: list[dict], max_samples: int) -> list[list[dict]]:
    """Greedily groups VAD speech segments (in original order) into chunks no longer than
    max_samples of *speech* audio each, splitting any single segment that alone exceeds
    max_samples. This bounds how long Whisper goes without a fresh initial_prompt seed."""
    chunks, current, current_len = [], [], 0
    for ts in speech_timestamps:
        seg_len = ts["end"] - ts["start"]
        if seg_len > max_samples:
            if current:
                chunks.append(current)
                current, current_len = [], 0
            start = ts["start"]
            while start < ts["end"]:
                end = min(start + max_samples, ts["end"])
                chunks.append([{"start": start, "end": end}])
                start = end
            continue
        if current_len + seg_len > max_samples:
            chunks.append(current)
            current, current_len = [], 0
        current.append(ts)
        current_len += seg_len
    if current:
        chunks.append(current)
    return chunks


def filter_hallucinated_segments(segments: list[dict], max_no_speech_prob: float, min_avg_logprob: float, max_compression_ratio: float) -> str:
    kept = [
        seg["text"]
        for seg in segments
        if seg["no_speech_prob"] <= max_no_speech_prob
        and seg["avg_logprob"] >= min_avg_logprob
        and seg["compression_ratio"] <= max_compression_ratio
    ]
    return "".join(kept).strip()


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
) -> str:
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
    chunk_groups = pack_speech_chunks(speech_timestamps, max_samples)

    chunk_texts = []
    for group in chunk_groups:
        audio = collect_chunks(group, wav).numpy()
        result = model.transcribe(
            audio,
            language=language,
            initial_prompt=initial_prompt,
            condition_on_previous_text=condition_on_previous_text,
        )
        chunk_texts.append(filter_hallucinated_segments(result["segments"], max_no_speech_prob, min_avg_logprob, max_compression_ratio))

    return " ".join(t for t in chunk_texts if t)


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

    for mp3_path in sorted(mp3_files):
        txt_path = dst_dir / mp3_path.with_suffix(".txt").name
        print(f"{mp3_path.name} -> {txt_path}")

        text = transcribe_mp3(
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
        )
        txt_path.write_text(text, encoding="utf-8")
        print("  done")


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
    )


if __name__ == "__main__":
    main()
