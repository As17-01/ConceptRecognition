import subprocess
import sys
import hydra

from pathlib import Path
from omegaconf import DictConfig


def convert_mp4_to_mp3(src_dir: Path, dst_dir: Path, quality: int) -> None:
    mp4_files = list(src_dir.glob("*.mp4"))
    if not mp4_files:
        print(f"No MP4 files found in {src_dir}")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    for mp4_path in sorted(mp4_files):
        mp3_path = dst_dir / mp4_path.with_suffix(".mp3").name
        print(f"{mp4_path.name} -> {mp3_path}")
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(mp4_path),
                "-vn",
                "-acodec", "libmp3lame",
                "-q:a", str(quality),
                str(mp3_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)
        else:
            print("  done")


@hydra.main(config_path="conf", config_name="mp4_to_mp3", version_base=None)
def main(cfg: DictConfig) -> None:
    src_dir = Path(cfg.src)
    dst_dir = Path(cfg.dst)

    if not src_dir.is_dir():
        print(f"Source folder not found: {src_dir}", file=sys.stderr)
        sys.exit(1)

    convert_mp4_to_mp3(src_dir, dst_dir, cfg.quality)


if __name__ == "__main__":
    main()
