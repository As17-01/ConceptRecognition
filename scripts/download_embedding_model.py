import hydra

from pathlib import Path
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer


@hydra.main(config_path="conf", config_name="semantic_chunk", version_base=None)
def main(cfg: DictConfig) -> None:
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading embedding model '{cfg.model}' into '{model_dir}'...")
    SentenceTransformer(cfg.model, cache_folder=str(model_dir))
    print("Done. The model is now cached locally and semantic_chunk.py will not need internet access.")


if __name__ == "__main__":
    main()
