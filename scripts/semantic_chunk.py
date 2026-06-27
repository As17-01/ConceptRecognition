import re
import sys

import hydra
import numpy as np

from pathlib import Path
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s for s in sentences if s]


def semantic_chunks(
    sentences: list[str],
    model: SentenceTransformer,
    query_prefix: str,
    window: int,
    percentile: float,
    min_sentences: int,
) -> list[list[str]]:
    prefixed = [query_prefix + s for s in sentences]
    embeddings = model.encode(prefixed, normalize_embeddings=True)

    # Compare each sentence to the next `window` sentences combined,
    # so a single off-topic aside doesn't trigger a false split.
    sims = []
    for i in range(len(sentences) - window):
        left = embeddings[max(0, i - window + 1) : i + 1].mean(axis=0)
        right = embeddings[i + 1 : i + 1 + window].mean(axis=0)
        sims.append(float(np.dot(left, right)))

    # A "boundary" is a local dip in similarity, not just a low absolute value,
    # since similarity drifts over a long transcript.
    distances = 1 - np.array(sims)
    threshold = np.percentile(distances, percentile)
    boundaries = {i + 1 for i, d in enumerate(distances) if d >= threshold}

    chunks, current = [], [sentences[0]]
    for i, sentence in enumerate(sentences[1:], start=1):
        if i in boundaries:
            chunks.append(current)
            current = []
        current.append(sentence)
    chunks.append(current)

    # Short chunks are usually interjections/asides caught by the threshold,
    # not real topic shifts, so fold them into the chunk before them.
    merged = [chunks[0]]
    for chunk in chunks[1:]:
        if len(chunk) < min_sentences:
            merged[-1].extend(chunk)
        else:
            merged.append(chunk)
    return merged


def chunk_transcript(
    src_path: Path,
    dst_dir: Path,
    model: SentenceTransformer,
    query_prefix: str,
    window: int,
    percentile: float,
    min_sentences: int,
) -> None:
    text = src_path.read_text(encoding="utf-8")
    sentences = split_sentences(text)
    if not sentences:
        print(f"{src_path.name}: no sentences found, skipping")
        return

    chunks = semantic_chunks(sentences, model, query_prefix, window, percentile, min_sentences)

    out_dir = dst_dir / src_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_file in out_dir.glob("chunk_*.txt"):
        old_file.unlink()

    for i, chunk in enumerate(chunks, start=1):
        chunk_path = out_dir / f"chunk_{i:03d}.txt"
        chunk_path.write_text(" ".join(chunk), encoding="utf-8")

    print(f"{src_path.name}: {len(sentences)} sentences -> {len(chunks)} chunks -> {out_dir}")


@hydra.main(config_path="conf", config_name="semantic_chunk", version_base=None)
def main(cfg: DictConfig) -> None:
    src_dir = Path(cfg.src)
    dst_dir = Path(cfg.dst)

    if not src_dir.is_dir():
        print(f"Source folder not found: {src_dir}", file=sys.stderr)
        sys.exit(1)

    txt_files = sorted(src_dir.glob("*.txt"))
    if not txt_files:
        print(f"No transcript files found in {src_dir}")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model '{cfg.model}' from '{cfg.model_dir}'...")
    model = SentenceTransformer(cfg.model, cache_folder=cfg.model_dir)

    for txt_path in txt_files:
        chunk_transcript(txt_path, dst_dir, model, cfg.query_prefix, cfg.window, cfg.percentile, cfg.min_sentences)


if __name__ == "__main__":
    main()
