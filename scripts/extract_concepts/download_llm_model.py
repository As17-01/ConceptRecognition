import hydra

from pathlib import Path
from omegaconf import DictConfig
from huggingface_hub import snapshot_download


@hydra.main(config_path="../conf", config_name="extract_concepts", version_base=None)
def main(cfg: DictConfig) -> None:
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Plain file download, independent of quantization/GPU - the same cached weights are
    # quantized at load time by extract_concepts.py, so this can run on any machine.
    print(f"Downloading '{cfg.model}' into '{model_dir}'...")
    snapshot_download(cfg.model, cache_dir=str(model_dir))
    print("Done. The model is now cached locally and extract_concepts.py will not need internet access.")


if __name__ == "__main__":
    main()
