import json
import sys
import hydra

from pathlib import Path
from omegaconf import DictConfig

# Whisper transcription bug: this phrase gets hallucinated repeatedly and is not actual speech
TRANSCRIPTION_ARTIFACTS = ["С вами был Игорь Негода."]


def clean_text(text: str) -> str:
    for artifact in TRANSCRIPTION_ARTIFACTS:
        text = text.replace(artifact, "")
    return text


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    stride = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), stride):
        chunk_words = words[start : start + chunk_size]
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_transcriptions(src_dir: Path, dst_dir: Path, chunk_size: int, overlap: int) -> None:
    txt_files = list(src_dir.glob("*.txt"))
    if not txt_files:
        print(f"No transcription files found in {src_dir}")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    for txt_path in sorted(txt_files):
        jsonl_path = dst_dir / txt_path.with_suffix(".jsonl").name
        print(f"{txt_path.name} -> {jsonl_path}")
        text = clean_text(txt_path.read_text(encoding="utf-8"))
        chunks = chunk_text(text, chunk_size, overlap)
        with jsonl_path.open("w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                record = {"source": txt_path.stem, "chunk_id": i, "text": chunk}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  {len(chunks)} chunks")


@hydra.main(config_path="conf", config_name="chunk_transcriptions", version_base=None)
def main(cfg: DictConfig) -> None:
    src_dir = Path(cfg.src)
    dst_dir = Path(cfg.dst)

    if not src_dir.is_dir():
        print(f"Source folder not found: {src_dir}", file=sys.stderr)
        sys.exit(1)

    if cfg.overlap >= cfg.chunk_size:
        print("overlap must be smaller than chunk_size", file=sys.stderr)
        sys.exit(1)

    chunk_transcriptions(src_dir, dst_dir, cfg.chunk_size, cfg.overlap)


if __name__ == "__main__":
    main()
