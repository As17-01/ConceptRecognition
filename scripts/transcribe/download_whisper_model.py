import hydra

from pathlib import Path
from omegaconf import DictConfig
from faster_whisper import WhisperModel


@hydra.main(config_path="../conf", config_name="transcribe", version_base=None)
def main(cfg: DictConfig) -> None:
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading faster-whisper model '{cfg.model}' into '{model_dir}'...")
    WhisperModel(cfg.model, device="cpu", download_root=str(model_dir))
    print("Done. The model is now cached locally and transcribe.py will not need internet access.")


if __name__ == "__main__":
    main()
