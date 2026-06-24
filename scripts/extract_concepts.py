import json
import re
import sys
import hydra
import numpy as np
import torch

from pathlib import Path
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

CONCEPT_PROMPT = (
    "Ты анализируешь фрагмент расшифровки лекции на русском языке.\n"
    "Назови главную тему (концепт), которая обсуждается в этом фрагменте.\n"
    "Игнорируй организационные реплики (приветствия, технические неполадки, расписание) "
    "и называй содержательную тему, если она есть.\n"
    "Ответь ТОЛЬКО короткой именной группой из 2-3 слов, без кавычек, пояснений и специальных символов.\n\n"
    "Фрагмент:\n{text}\n\nТема:"
)


def load_chunks(jsonl_path: Path) -> list[dict]:
    records = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def embed_texts(model: SentenceTransformer, texts: list[str], prefix: str, batch_size: int) -> np.ndarray:
    return np.asarray(model.encode([f"{prefix}{text}" for text in texts], batch_size=batch_size, show_progress_bar=False))


def clean_label(raw_label: str) -> str:
    label = raw_label.strip().strip("\"'«»")
    label = re.split(r"[\n.]", label)[0]
    return label.strip().rstrip(",;:").lower().replace("_", " ")


def extract_chunk_labels(
    tokenizer: AutoTokenizer,
    label_model: AutoModelForCausalLM,
    texts: list[str],
    max_chunk_chars: int,
    max_new_tokens: int,
) -> list[str]:
    labels = []
    with torch.inference_mode():
        for text in texts:
            prompt = CONCEPT_PROMPT.format(text=text[:max_chunk_chars])
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            output_ids = label_model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated_ids = output_ids[0, input_ids.shape[1] :]
            labels.append(clean_label(tokenizer.decode(generated_ids, skip_special_tokens=True)))
    return labels


def build_concepts(chunk_vectors: np.ndarray, chunk_labels: list[str]) -> list[dict]:
    concepts = []
    for chunk_id, (label, vector) in enumerate(zip(chunk_labels, chunk_vectors)):
        concepts.append(
            {
                "chunk_id": chunk_id,
                "label": label,
                "vector": vector.tolist(),
            }
        )
    return concepts


def extract_concepts(
    src_dir: Path,
    dst_dir: Path,
    embed_model_name: str,
    label_model_name: str,
    batch_size: int,
    max_label_new_tokens: int,
    max_chunk_chars: int,
) -> None:
    jsonl_files = list(src_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No chunk files found in {src_dir}")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model '{embed_model_name}'...")
    embed_model = SentenceTransformer(embed_model_name, local_files_only=True)

    print(f"Loading label model '{label_model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(label_model_name, local_files_only=True)
    label_model = AutoModelForCausalLM.from_pretrained(
        label_model_name,
        local_files_only=True,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    label_model.eval()

    for jsonl_path in sorted(jsonl_files):
        records = load_chunks(jsonl_path)
        texts = [r["text"] for r in records]
        chunk_vectors = embed_texts(embed_model, texts, "passage: ", batch_size)

        print(f"{jsonl_path.name}: extracting per-chunk concepts with '{label_model_name}'...")
        chunk_labels = extract_chunk_labels(tokenizer, label_model, texts, max_chunk_chars, max_label_new_tokens)

        concepts = build_concepts(chunk_vectors, chunk_labels)

        out_path = dst_dir / jsonl_path.name
        with out_path.open("w", encoding="utf-8") as f:
            for concept in concepts:
                concept["source"] = jsonl_path.stem
                f.write(json.dumps(concept, ensure_ascii=False) + "\n")

        print(f"{jsonl_path.name}: {len(records)} chunks -> {len(concepts)} concepts -> {out_path}")


@hydra.main(config_path="conf", config_name="extract_concepts", version_base=None)
def main(cfg: DictConfig) -> None:
    src_dir = Path(cfg.src)
    dst_dir = Path(cfg.dst)

    if not src_dir.is_dir():
        print(f"Source folder not found: {src_dir}", file=sys.stderr)
        sys.exit(1)

    extract_concepts(
        src_dir,
        dst_dir,
        cfg.embed_model,
        cfg.label_model,
        cfg.batch_size,
        cfg.max_label_new_tokens,
        cfg.max_chunk_chars,
    )


if __name__ == "__main__":
    main()
