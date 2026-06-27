import hydra

from pathlib import Path
from omegaconf import DictConfig
from transformers import AutoModelForTokenClassification, AutoTokenizer


@hydra.main(config_path="conf", config_name="preprocess_transcripts", version_base=None)
def main(cfg: DictConfig) -> None:
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading punctuation model '{cfg.model}' into '{model_dir}'...")
    AutoTokenizer.from_pretrained(cfg.model, cache_dir=str(model_dir), strip_accents=False, add_prefix_space=True)
    AutoModelForTokenClassification.from_pretrained(cfg.model, cache_dir=str(model_dir))
    print("Done. The model is now cached locally and preprocess_transcripts.py will not need internet access.")


if __name__ == "__main__":
    main()
