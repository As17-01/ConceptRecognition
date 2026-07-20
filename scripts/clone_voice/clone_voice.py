import subprocess
import sys

import hydra

from pathlib import Path
from omegaconf import DictConfig
from elevenlabs.client import ElevenLabs


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def extract_clips(src_path: Path, dst_dir: Path, num_clips: int, clip_seconds: int) -> list[Path]:
    """Trims num_clips short reference clips, evenly spaced through the middle 80% of the source
    recording - skipping the first/last 10%, which tend to be greetings, audio-check chatter, or
    closing remarks rather than clean, sustained solo speech - for use as voice-cloning input."""
    duration = probe_duration(src_path)
    usable_start, usable_end = duration * 0.1, duration * 0.9
    usable_span = usable_end - usable_start

    dst_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for i in range(num_clips):
        offset = usable_start + usable_span * i / num_clips
        clip_path = dst_dir / f"sample_{i}.mp3"
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(offset),
                "-i", str(src_path),
                "-t", str(clip_seconds),
                "-acodec", "libmp3lame", "-q:a", "2",
                str(clip_path),
            ],
            capture_output=True,
            check=True,
        )
        clips.append(clip_path)
    return clips


@hydra.main(config_path="../conf", config_name="clone_voice", version_base=None)
def main(cfg: DictConfig) -> None:
    src_path = Path(cfg.src_dir) / cfg.sample_file
    if not src_path.is_file():
        print(f"Sample source file not found: {src_path}", file=sys.stderr)
        sys.exit(1)

    clips = extract_clips(src_path, Path(cfg.scratch_dir), cfg.num_clips, cfg.clip_seconds)
    print(f"Extracted {len(clips)} reference clips from {src_path.name}")

    client = ElevenLabs()
    voice = client.voices.ivc.create(
        name=cfg.voice_name,
        description=cfg.voice_description,
        files=[str(clip) for clip in clips],
    )

    dst_path = Path(cfg.dst_voice_id_file)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(voice.voice_id, encoding="utf-8")
    print(f"Cloned voice '{cfg.voice_name}' -> voice_id {voice.voice_id} (saved to {dst_path})")


if __name__ == "__main__":
    main()
