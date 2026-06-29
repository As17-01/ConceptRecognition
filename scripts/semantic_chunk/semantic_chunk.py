import json
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


def hierarchical_clusters(
    embeddings: np.ndarray, target_counts: dict[str, int], max_size_multiplier: float
) -> dict[str, list[tuple[int, int]]]:
    """Constrained (adjacent-only) agglomerative clustering: starts with every sentence as its
    own cluster and repeatedly merges whichever *adjacent* pair is most similar, mean-pooling
    their embeddings into the merged cluster. A level is just a snapshot of this single merge
    sequence at the cluster count target_counts[level] - since every later snapshot is reached
    by continuing to merge from the earlier one (never restarting), a coarser level's chunks are
    always exact unions of a finer level's chunks. Nesting is a byproduct of how merging works,
    not something enforced after the fact.

    Pure greedy merging (always take the globally most-similar adjacent pair) lets one locally
    homogeneous stretch keep absorbing merges far past a level's target average size while
    leaving other stretches comparatively fragmented, since "most similar" is judged globally,
    not relative to how big each side already is. max_size_multiplier bounds this: a candidate
    merge is skipped in favor of a less-similar one if it would make a cluster larger than
    max_size_multiplier times the *finest* not-yet-reached level's average size - that cap
    relaxes automatically as finer levels get captured and coarser, larger sizes become
    expected."""
    n = len(embeddings)
    starts = list(range(n))
    ends = list(range(1, n + 1))
    sums = [embeddings[i].copy() for i in range(n)]

    def sim(i: int, j: int) -> float:
        mi = sums[i] / np.linalg.norm(sums[i])
        mj = sums[j] / np.linalg.norm(sums[j])
        return float(np.dot(mi, mj))

    adjacent_sims = [sim(i, i + 1) for i in range(n - 1)]
    max_size_for_level = {name: max_size_multiplier * (n / target) for name, target in target_counts.items()}

    snapshots: dict[str, list[tuple[int, int]]] = {}
    remaining_targets = dict(target_counts)

    while True:
        for name, target in list(remaining_targets.items()):
            if len(starts) <= target:
                snapshots[name] = list(zip(starts, ends))
                del remaining_targets[name]
        if not remaining_targets or len(starts) == 1:
            break

        active_cap = min(max_size_for_level[name] for name in remaining_targets)
        candidates = [i for i in range(len(adjacent_sims)) if (ends[i + 1] - starts[i]) <= active_cap]
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

    # Any target at or below the final cluster count (e.g. target=1, or a very short transcript)
    # never gets caught inside the loop above since it exits as soon as len(starts) == 1.
    for name in remaining_targets:
        snapshots[name] = list(zip(starts, ends))

    return snapshots


def level_embeddings_for(embeddings: np.ndarray, ranges: list[tuple[int, int]]) -> np.ndarray:
    pooled = np.stack([embeddings[start:end].mean(axis=0) for start, end in ranges])
    pooled /= np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled


def children_per_parent(parent_ranges: list[tuple[int, int]], child_ranges: list[tuple[int, int]]) -> list[list[int]]:
    """For each parent chunk, the 0-based indices of the child-level chunks it contains."""
    children_by_parent = [[] for _ in parent_ranges]
    child_idx = 0
    for parent_idx, (_, p_end) in enumerate(parent_ranges):
        while child_idx < len(child_ranges) and child_ranges[child_idx][0] < p_end:
            children_by_parent[parent_idx].append(child_idx)
            child_idx += 1
    return children_by_parent


def write_level(level_dir: Path, sentences: list[str], embeddings: np.ndarray, ranges: list[tuple[int, int]]) -> None:
    level_dir.mkdir(parents=True, exist_ok=True)
    for old_file in level_dir.glob("chunk_*.txt"):
        old_file.unlink()

    for i, (start, end) in enumerate(ranges, start=1):
        chunk_path = level_dir / f"chunk_{i:03d}.txt"
        chunk_path.write_text(" ".join(sentences[start:end]), encoding="utf-8")

    np.save(level_dir / "embeddings.npy", level_embeddings_for(embeddings, ranges))


def chunk_transcript(
    src_path: Path, dst_dir: Path, model: SentenceTransformer, query_prefix: str, levels: dict[str, int], max_size_multiplier: float
) -> None:
    text = src_path.read_text(encoding="utf-8")
    sentences = split_sentences(text)
    if not sentences:
        print(f"{src_path.name}: no sentences found, skipping")
        return

    prefixed = [query_prefix + s for s in sentences]
    embeddings = model.encode(prefixed, normalize_embeddings=True)

    target_counts = {name: max(1, len(sentences) // avg_size) for name, avg_size in levels.items()}
    snapshots = hierarchical_clusters(embeddings, target_counts, max_size_multiplier)

    # Finest (most chunks/smallest avg size) to coarsest, so each level can record which chunks
    # of the level directly below it contains.
    level_names = sorted(levels.keys(), key=lambda name: levels[name])

    out_dir = dst_dir / src_path.stem
    hierarchy = {"levels": level_names}
    finer_name, finer_ranges = None, None
    for level_name in level_names:
        ranges = snapshots[level_name]
        write_level(out_dir / level_name, sentences, embeddings, ranges)

        level_info = {"ranges": [[start, end] for start, end in ranges]}
        if finer_ranges is not None:
            level_info["finer_level"] = finer_name
            level_info["children"] = children_per_parent(ranges, finer_ranges)
        hierarchy[level_name] = level_info

        print(f"{src_path.name}: level '{level_name}' -> {len(ranges)} chunks -> {out_dir / level_name}")
        finer_name, finer_ranges = level_name, ranges

    (out_dir / "hierarchy.json").write_text(json.dumps(hierarchy, ensure_ascii=False, indent=2), encoding="utf-8")


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

    levels = dict(cfg.levels)
    succeeded, skipped, failed = 0, 0, 0
    for txt_path in txt_files:
        # Makes a 100+ file job resumable: a crash or timeout partway through shouldn't force
        # re-running files that already finished.
        if (dst_dir / txt_path.stem / "hierarchy.json").exists():
            print(f"{txt_path.name}: output already exists, skipping")
            skipped += 1
            continue

        try:
            chunk_transcript(txt_path, dst_dir, model, cfg.query_prefix, levels, cfg.max_size_multiplier)
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
