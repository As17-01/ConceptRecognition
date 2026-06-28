import sys
import hydra
import whisper

from pathlib import Path
from omegaconf import DictConfig


def transcribe_mp3s(src_dir: Path, dst_dir: Path, model_name: str, model_dir: Path, language: str, initial_prompt: str) -> None:
    mp3_files = list(src_dir.glob("*.mp3"))
    if not mp3_files:
        print(f"No MP3 files found in {src_dir}")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading whisper model '{model_name}' from '{model_dir}'...")
    model = whisper.load_model(model_name, download_root=str(model_dir))

    for mp3_path in sorted(mp3_files):
        txt_path = dst_dir / mp3_path.with_suffix(".txt").name
        print(f"{mp3_path.name} -> {txt_path}")
        result = model.transcribe(str(mp3_path), language=language, initial_prompt=initial_prompt)
        txt_path.write_text(result["text"], encoding="utf-8")
        print("  done")


@hydra.main(config_path="conf", config_name="transcribe", version_base=None)
def main(cfg: DictConfig) -> None:
    src_dir = Path(cfg.src)
    dst_dir = Path(cfg.dst)
    model_dir = Path(cfg.model_dir)

    if not src_dir.is_dir():
        print(f"Source folder not found: {src_dir}", file=sys.stderr)
        sys.exit(1)

    transcribe_mp3s(src_dir, dst_dir, cfg.model, model_dir, cfg.language, cfg.initial_prompt)


if __name__ == "__main__":
    main()
