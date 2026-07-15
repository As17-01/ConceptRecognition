import re
import sys

import hydra
import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s for s in sentences if s]


def encode_with_context(model: SentenceTransformer, sentences: list[str], query_prefix: str) -> np.ndarray:
    """Encodes each sentence by mean-pooling only its own tokens' hidden states out of a forward
    pass over a window that also contains its neighbors, so self-attention lets ambiguous tokens
    (pronouns, references) pull in relevant context from nearby sentences before pooling. Unlike
    concatenating neighbor text into the input and pooling over everything, a neighbor's own
    tokens are never included in this sentence's average - only their influence on this
    sentence's tokens is, via attention - so there's no dilution from averaging in a whole extra
    sentence, and a sentence's vector stays anchored to its own length regardless of window size.

    Sentences are packed into non-overlapping windows capped at the model's max sequence length,
    so a sentence near a window edge only gets context from whichever side shares its window."""
    tokenizer = model.tokenizer
    transformer = model[0].auto_model
    device = model.device

    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    prefix_ids = tokenizer(query_prefix, add_special_tokens=False)["input_ids"] if query_prefix else []
    sentence_ids = [tokenizer(s, add_special_tokens=False)["input_ids"] for s in sentences]

    # Room left in the window for CLS + prefix + SEP, which are re-added to every window.
    budget = model.get_max_seq_length() - 2 - len(prefix_ids)

    embeddings: list[torch.Tensor] = [None] * len(sentences)
    start = 0
    while start < len(sentences):
        end, length = start, 0
        while end < len(sentences) and length + len(sentence_ids[end]) <= budget:
            length += len(sentence_ids[end])
            end += 1
        if end == start:
            # A single sentence alone exceeds the budget; take it truncated rather than stall.
            end = start + 1

        window_ids = prefix_ids + [tok for i in range(start, end) for tok in sentence_ids[i]]
        input_ids = torch.tensor([[cls_id, *window_ids[:budget], sep_id]], device=device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            hidden = transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[0]

        offset = 1 + len(prefix_ids)
        for i in range(start, end):
            n = len(sentence_ids[i])
            embeddings[i] = hidden[offset : offset + n].mean(dim=0)
            offset += n
        start = end

    return F.normalize(torch.stack(embeddings), dim=1).cpu().numpy()


def cluster_sentences(embeddings: np.ndarray, target_count: int, max_size_multiplier: float) -> list[tuple[int, int]]:
    """Constrained (adjacent-only) agglomerative clustering: starts with every sentence as its
    own cluster and repeatedly merges whichever *adjacent* pair is most similar, mean-pooling
    their embeddings into the merged cluster, until target_count clusters remain.

    Pure greedy merging (always take the globally most-similar adjacent pair) lets one locally
    homogeneous stretch keep absorbing merges far past the target average size while leaving
    other stretches comparatively fragmented, since "most similar" is judged globally, not
    relative to how big each side already is. max_size_multiplier bounds this: a candidate merge
    is skipped in favor of a less-similar one if it would make a cluster larger than
    max_size_multiplier times the target average size."""
    n = len(embeddings)
    starts = list(range(n))
    ends = list(range(1, n + 1))
    sums = [embeddings[i].copy() for i in range(n)]

    def sim(i: int, j: int) -> float:
        mi = sums[i] / np.linalg.norm(sums[i])
        mj = sums[j] / np.linalg.norm(sums[j])
        return float(np.dot(mi, mj))

    adjacent_sims = [sim(i, i + 1) for i in range(n - 1)]
    max_size = max_size_multiplier * (n / target_count)

    while len(starts) > target_count and len(starts) > 1:
        candidates = [i for i in range(len(adjacent_sims)) if (ends[i + 1] - starts[i]) <= max_size]
        if not candidates:
            # Every remaining adjacent merge would exceed the cap; honoring it would stall
            # progress entirely, so fall back to the otherwise-best merge for this one step.
            candidates = range(len(adjacent_sims))

        best = max(candidates, key=lambda i: adjacent_sims[i])
        starts[best : best + 2] = [starts[best]]
        ends[best : best + 2] = [ends[best + 1]]
        sums[best : best + 2] = [sums[best] + sums[best + 1]]
        del adjacent_sims[best]
        if best > 0:
            adjacent_sims[best - 1] = sim(best - 1, best)
        if best < len(starts) - 1:
            adjacent_sims[best] = sim(best, best + 1)

    return list(zip(starts, ends))


def chunk_embeddings_for(embeddings: np.ndarray, ranges: list[tuple[int, int]]) -> np.ndarray:
    pooled = np.stack([embeddings[start:end].mean(axis=0) for start, end in ranges])
    pooled /= np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled


def chunk_transcript(
    src_path: Path, dst_dir: Path, model: SentenceTransformer, query_prefix: str, avg_sentences: int, max_size_multiplier: float
) -> None:
    text = src_path.read_text(encoding="utf-8")
    sentences = split_sentences(text)
    if not sentences:
        print(f"{src_path.name}: no sentences found, skipping")
        return

    embeddings = encode_with_context(model, sentences, query_prefix)

    target_count = max(1, len(sentences) // avg_sentences)
    ranges = cluster_sentences(embeddings, target_count, max_size_multiplier)

    out_dir = dst_dir / src_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_file in out_dir.glob("chunk_*.txt"):
        old_file.unlink()

    for i, (start, end) in enumerate(ranges, start=1):
        chunk_path = out_dir / f"chunk_{i:03d}.txt"
        chunk_path.write_text(" ".join(sentences[start:end]), encoding="utf-8")

    np.save(out_dir / "embeddings.npy", chunk_embeddings_for(embeddings, ranges))
    print(f"{src_path.name}: {len(ranges)} chunks -> {out_dir}")


@hydra.main(config_path="../conf", config_name="semantic_chunk", version_base=None)
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

    succeeded, skipped, failed = 0, 0, 0
    for txt_path in txt_files:
        # Makes a 100+ file job resumable: a crash or timeout partway through shouldn't force
        # re-running files that already finished.
        if (dst_dir / txt_path.stem / "embeddings.npy").exists():
            print(f"{txt_path.name}: output already exists, skipping")
            skipped += 1
            continue

        try:
            chunk_transcript(txt_path, dst_dir, model, cfg.query_prefix, cfg.avg_sentences, cfg.max_size_multiplier)
        except Exception as e:
            # One bad file shouldn't lose progress on the rest of the batch; log it and move on
            # instead of crashing the job.
            print(f"  FAILED: {e}", file=sys.stderr)
            failed += 1
            continue
        succeeded += 1

    print(f"\n{succeeded} chunked, {skipped} skipped (already done), {failed} failed, out of {len(txt_files)} total")


if __name__ == "__main__":
    main()
