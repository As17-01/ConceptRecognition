import re
import subprocess
import sys
import tempfile

import hydra

from pathlib import Path
from omegaconf import DictConfig
from elevenlabs.client import ElevenLabs

# Pause marker (structural or micro); format defined in transcribe.py - keep in sync.
PAUSE_RE = re.compile(r"\[(?:ПАУЗА|МИКРОПАУЗА):(\d+)\]")


def split_segments(text: str) -> list[tuple[str, int]]:
    """Splits text on pause markers into an ordered list of (spoken_text, pause_seconds_after)
    pairs. re.split with the capturing group interleaves [text0, num1, text1, num2, text2, ...];
    a segment not followed by a marker (including the last one) gets pause_seconds_after=0. A
    spoken_text can legitimately come out empty (two markers back to back)."""
    parts = PAUSE_RE.split(text)
    segments = []
    for i in range(0, len(parts), 2):
        spoken = parts[i].strip()
        pause_seconds = int(parts[i + 1]) if i + 1 < len(parts) else 0
        segments.append((spoken, pause_seconds))
    return segments


def synthesize_segment(client: ElevenLabs, text: str, voice_id: str, model_id: str, output_format: str) -> bytes:
    # convert() returns Iterator[bytes] (a streaming response) - not a single bytes object -
    # so the audio has to be assembled from chunks rather than written directly.
    chunks = client.text_to_speech.convert(text=text, voice_id=voice_id, model_id=model_id, output_format=output_format)
    return b"".join(chunk for chunk in chunks if isinstance(chunk, bytes))


def mp3_bytes_to_wav(audio: bytes, work_dir: Path, name: str) -> Path:
    # ffmpeg needs a real input path, so the segment bytes are written out first.
    src_path = work_dir / f"{name}.mp3"
    src_path.write_bytes(audio)
    wav_path = work_dir / f"{name}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src_path), "-ar", "44100", "-ac", "1", str(wav_path)],
        capture_output=True,
        check=True,
    )
    return wav_path


def silence_wav(seconds: int, work_dir: Path, name: str) -> Path:
    wav_path = work_dir / f"{name}.wav"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
            "-t", str(seconds),
            str(wav_path),
        ],
        capture_output=True,
        check=True,
    )
    return wav_path


def concat_to_mp3(wav_paths: list[Path], dst_path: Path, work_dir: Path, bitrate: str) -> None:
    # Concatenating independently-encoded mp3 chunks directly can leave audible clicks at the
    # splice points (encoder delay / frame boundaries don't line up) - going through a canonical
    # WAV intermediate and the concat demuxer avoids that, and re-encodes to mp3 in this same pass.
    list_path = work_dir / "concat_list.txt"
    list_path.write_text("".join(f"file '{p.resolve()}'\n" for p in wav_paths), encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-b:a", bitrate, str(dst_path)],
        capture_output=True,
        check=True,
    )


@hydra.main(config_path="../conf", config_name="synthesize_speech", version_base=None)
def main(cfg: DictConfig) -> None:
    src_dir = Path(cfg.src)
    if not src_dir.is_dir():
        print(f"Source folder not found: {src_dir}", file=sys.stderr)
        sys.exit(1)

    voice_id_path = Path(cfg.voice_id_file)
    if not voice_id_path.is_file():
        print(f"Voice ID file not found: {voice_id_path} - run clone_voice.py first", file=sys.stderr)
        sys.exit(1)
    voice_id = voice_id_path.read_text(encoding="utf-8").strip()

    txt_files = sorted(src_dir.glob("*.txt"))
    if not txt_files:
        print(f"No generated class scripts found in {src_dir}")
        return

    dst_dir = Path(cfg.dst)
    dst_dir.mkdir(parents=True, exist_ok=True)

    scratch_dir = Path(cfg.scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)  # TemporaryDirectory(dir=...) requires the parent to already exist

    # output_format is "<codec>_<samplerate>_<bitrate>" (e.g. "mp3_44100_128"); the final concat
    # pass re-encodes to mp3 itself, so only the bitrate part of it is needed here.
    bitrate = cfg.output_format.split("_")[-1] + "k"

    client = ElevenLabs()
    succeeded, skipped, failed = 0, 0, 0
    for txt_path in txt_files:
        dst_path = dst_dir / txt_path.with_suffix(".mp3").name

        # Same resumability pattern as the rest of the pipeline: a crash or interruption
        # partway through a batch shouldn't force re-synthesizing (and re-billing) classes
        # that already finished.
        if dst_path.exists():
            print(f"{txt_path.name}: output already exists, skipping")
            skipped += 1
            continue

        try:
            text = txt_path.read_text(encoding="utf-8")
            segments = split_segments(text)

            with tempfile.TemporaryDirectory(dir=scratch_dir) as tmp:
                work_dir = Path(tmp)
                pieces = []
                for i, (spoken, pause_seconds) in enumerate(segments):
                    if spoken:
                        audio = synthesize_segment(client, spoken, voice_id, cfg.model_id, cfg.output_format)
                        pieces.append(mp3_bytes_to_wav(audio, work_dir, f"seg_{i}"))
                    if pause_seconds > 0:
                        pieces.append(silence_wav(pause_seconds, work_dir, f"pause_{i}"))

                concat_to_mp3(pieces, dst_path, work_dir, bitrate)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            failed += 1
            continue
        succeeded += 1
        print(f"{txt_path.name}: {dst_path.stat().st_size} bytes -> {dst_path}")

    print(f"\n{succeeded} synthesized, {skipped} skipped (already done), {failed} failed, out of {len(txt_files)} total")


if __name__ == "__main__":
    main()
