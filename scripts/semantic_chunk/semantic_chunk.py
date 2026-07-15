import re
import statistics
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
    # "…" ends a clause the same way "." does (a trailing-off thought, usually followed by
    # capitalization signaling Whisper itself treated it as a break) but isn't in [.!?], so
    # without it here a sentence boundary silently gets missed and two sentences merge into one.
    sentences = re.split(r"(?<=[.!?…])\s+", text)
    return [s for s in sentences if s]


def split_overlong_sentences(sentences: list[str], max_words: int) -> list[str]:
    """Whisper doesn't always add terminal punctuation promptly in run-on speech - measured on
    real preprocessed transcripts, sentence length has a median of 8 words and a p95 of 40, but a
    long tail up to 149. A sentence that long is already past the point where a
    sentence-transformer's pooled embedding usefully represents one idea (see semantic_chunk
    granularity discussion), so anything over max_words is split at comma boundaries, greedily
    packing clauses back together up to the limit - like pack_speech_chunks in transcribe.py -
    rather than atomizing every comma, since ordinary Russian subordinate clauses use commas far
    more often than English does, not just at true topic breaks."""
    result = []
    for sentence in sentences:
        if len(sentence.split()) <= max_words:
            result.append(sentence)
            continue

        clauses = re.split(r"(?<=,)\s+", sentence)
        current, current_len = [], 0
        for clause in clauses:
            clause_len = len(clause.split())
            if current and current_len + clause_len > max_words:
                result.append(" ".join(current))
                current, current_len = [], 0
            current.append(clause)
            current_len += clause_len
        if current:
            result.append(" ".join(current))
    return result


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


def merge_trace(embeddings: np.ndarray) -> list[tuple[int, float]]:
    """Runs the same adjacent-only greedy agglomerative merge as cluster_sentences_by_zscore, but
    all the way down to a single cluster, recording one entry per merge in the order performed:
    (resulting cluster size in sentences, similarity of the pair that was merged). This is what
    cluster_sentences_by_zscore uses to learn a document's own mean/std merge similarity before
    deciding where to actually stop."""
    n = len(embeddings)
    starts = list(range(n))
    ends = list(range(1, n + 1))
    sums = [embeddings[i].copy() for i in range(n)]

    def sim(i: int, j: int) -> float:
        mi = sums[i] / np.linalg.norm(sums[i])
        mj = sums[j] / np.linalg.norm(sums[j])
        return float(np.dot(mi, mj))

    adjacent_sims = [sim(i, i + 1) for i in range(n - 1)]
    trace = []
    while len(starts) > 1:
        best = max(range(len(adjacent_sims)), key=lambda i: adjacent_sims[i])
        merge_sim = adjacent_sims[best]
        starts[best : best + 2] = [starts[best]]
        ends[best : best + 2] = [ends[best + 1]]
        sums[best : best + 2] = [sums[best] + sums[best + 1]]
        del adjacent_sims[best]
        if best > 0:
            adjacent_sims[best - 1] = sim(best - 1, best)
        if best < len(starts) - 1:
            adjacent_sims[best] = sim(best, best + 1)
        trace.append((ends[best] - starts[best], merge_sim))
    return trace


def cluster_sentences_by_zscore(embeddings: np.ndarray, z_score: float) -> list[tuple[int, int]]:
    """Adjacent-only agglomerative clustering: starts with every sentence as its own cluster and
    repeatedly merges whichever *adjacent* pair is most similar, mean-pooling their embeddings
    into the merged cluster, stopping once the best available adjacent similarity drops more than
    z_score standard deviations below this document's own mean merge similarity - instead of
    merging down to a fixed target chunk count. Absolute cosine
    similarity turned out to be useless as a stopping signal here - measured on real transcripts,
    even the least-similar merge in a whole document stayed above 0.91, so a fixed threshold like
    0.8 never fires. Chunk count instead falls out of how topically choppy or smooth *this*
    document actually is, relative to its own distribution, rather than an externally chosen
    average sentence count. Runs merge_trace first purely to get that document's own mean/std,
    then re-merges and stops at the derived cutoff - the two-pass cost is negligible next to the
    embedding step."""
    trace = merge_trace(embeddings)
    sims = [s for _, s in trace]
    tau = statistics.mean(sims) - z_score * statistics.stdev(sims)

    n = len(embeddings)
    starts = list(range(n))
    ends = list(range(1, n + 1))
    sums = [embeddings[i].copy() for i in range(n)]

    def sim(i: int, j: int) -> float:
        mi = sums[i] / np.linalg.norm(sums[i])
        mj = sums[j] / np.linalg.norm(sums[j])
        return float(np.dot(mi, mj))

    adjacent_sims = [sim(i, i + 1) for i in range(n - 1)]
    while len(starts) > 1 and max(adjacent_sims) >= tau:
        best = max(range(len(adjacent_sims)), key=lambda i: adjacent_sims[i])
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


def merge_short_chunks(embeddings: np.ndarray, ranges: list[tuple[int, int]], min_sentences: int) -> list[tuple[int, int]]:
    """Eliminates chunks under min_sentences sentences long by merging each one into whichever
    adjacent chunk its pooled embedding is more similar to. A pure similarity-driven boundary
    (see cluster_sentences_by_zscore) tends to strand short interjections ("Beautiful!", "Ага.")
    as their own chunk: they don't embed like the detailed instructional text on either side, so
    the algorithm reads them as a topic shift in both directions - but a one-word aside isn't a
    topic of its own and shouldn't be a standalone chunk."""
    ranges = list(ranges)
    while len(ranges) > 1:
        sizes = [end - start for start, end in ranges]
        shortest = min(range(len(ranges)), key=lambda i: sizes[i])
        if sizes[shortest] >= min_sentences:
            break

        pooled = chunk_embeddings_for(embeddings, ranges)
        if shortest == 0:
            merge_with = 1
        elif shortest == len(ranges) - 1:
            merge_with = shortest - 1
        else:
            left_sim = float(np.dot(pooled[shortest], pooled[shortest - 1]))
            right_sim = float(np.dot(pooled[shortest], pooled[shortest + 1]))
            merge_with = shortest - 1 if left_sim >= right_sim else shortest + 1

        lo, hi = sorted([shortest, merge_with])
        ranges[lo : hi + 1] = [(ranges[lo][0], ranges[hi][1])]
    return ranges


def split_long_chunks(
    sentences: list[str], embeddings: np.ndarray, ranges: list[tuple[int, int]], max_words: int
) -> list[tuple[int, int]]:
    """Force-splits any chunk whose sentences total more than max_words words after clustering,
    cutting at its weakest internal adjacent-sentence link (lowest cosine similarity between two
    consecutive sentences inside it) rather than an arbitrary midpoint - reusing the same
    locally-computed similarity signal cluster_sentences_by_zscore is built on, instead of an
    unrelated cut rule. Word count, not sentence count, is what actually drove the outliers this
    guards against - a chunk can have very few but very long sentences and still run past 300
    words, well past where a pooled embedding usefully represents one idea (the same dilution
    concern max_sentence_words addresses at the raw-sentence level). Splitting can leave an
    undersized fragment right at a cut's edge (the weakest link can sit next to either end of the
    chunk) - run merge_short_chunks again afterwards to clean those up."""
    result = []
    for start, end in ranges:
        while end - start > 1 and sum(len(s.split()) for s in sentences[start:end]) > max_words:
            weakest = min(range(start, end - 1), key=lambda i: float(np.dot(embeddings[i], embeddings[i + 1])))
            result.append((start, weakest + 1))
            start = weakest + 1
        result.append((start, end))
    return result


def chunk_transcript(
    src_path: Path,
    dst_dir: Path,
    model: SentenceTransformer,
    query_prefix: str,
    max_sentence_words: int,
    boundary_z_score: float,
    min_chunk_sentences: int,
    max_chunk_words: int,
) -> None:
    text = src_path.read_text(encoding="utf-8")
    sentences = split_sentences(text)
    sentences = split_overlong_sentences(sentences, max_sentence_words)
    if not sentences:
        print(f"{src_path.name}: no sentences found, skipping")
        return

    embeddings = encode_with_context(model, sentences, query_prefix)

    ranges = cluster_sentences_by_zscore(embeddings, boundary_z_score)
    ranges = merge_short_chunks(embeddings, ranges, min_chunk_sentences)
    ranges = split_long_chunks(sentences, embeddings, ranges, max_chunk_words)
    ranges = merge_short_chunks(embeddings, ranges, min_chunk_sentences)

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
            chunk_transcript(
                txt_path,
                dst_dir,
                model,
                cfg.query_prefix,
                cfg.max_sentence_words,
                cfg.boundary_z_score,
                cfg.min_chunk_sentences,
                cfg.max_chunk_words,
            )
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
