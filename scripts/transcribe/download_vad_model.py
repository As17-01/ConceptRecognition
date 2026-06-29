import hydra
import torch

from pathlib import Path
from omegaconf import DictConfig


@hydra.main(config_path="../conf", config_name="transcribe", version_base=None)
def main(cfg: DictConfig) -> None:
    vad_model_dir = Path(cfg.vad_model_dir)
    vad_model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Silero VAD model into '{vad_model_dir}'...")
    torch.hub.set_dir(str(vad_model_dir))
    torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=False, onnx=False)
    print("Done. The model is now cached locally and transcribe.py will not need internet access.")


if __name__ == "__main__":
    main()
