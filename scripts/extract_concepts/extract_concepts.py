import json
import re
import sys

import hydra
import torch

from pathlib import Path
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SPECIFIC_SYSTEM_PROMPT = (
    "Ты помогаешь структурировать транскрипт занятия по импровизации и контемпорари-дэнсу. "
    "Из отрывка транскрипта извлеки тему отрывка и список ключевых идей или инструкций. "
    "Игнорируй приветствия, обращения к участникам по имени и прочий шум, не относящийся к "
    "содержанию занятия. Отвечай ТОЛЬКО валидным JSON без markdown-разметки и без пояснений, "
    'в формате: {"topic": "...", "key_points": ["...", "..."]}'
)

SYNTHESIZE_SYSTEM_PROMPT = (
    "Ты помогаешь структурировать транскрипт занятия по импровизации и контемпорари-дэнсу. "
    "Ниже даны темы и ключевые идеи нескольких последовательных частей занятия в формате JSON. "
    "Объедини их в одну более общую тему и список ключевых идей более высокого уровня, убирая "
    "повторы и оставляя только действительно важное. Отвечай ТОЛЬКО валидным JSON без "
    'markdown-разметки и без пояснений, в формате: {"topic": "...", "key_points": ["...", "..."]}'
)

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> dict:
    fenced = JSON_FENCE_RE.search(text)
    candidate = fenced.group(1) if fenced else text
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in model output")
    return json.loads(candidate[start : end + 1])


def generate_json(model, tokenizer, system_prompt: str, user_content: str, max_new_tokens: int) -> dict:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    try:
        return extract_json(response)
    except (ValueError, json.JSONDecodeError):
        # Surfaced as data instead of crashing the batch - a single malformed response
        # shouldn't lose progress on a job processing thousands of chunks.
        return {"topic": None, "key_points": [], "parse_error": True, "raw_output": response}


def extract_leaf_level(src_level_dir: Path, dst_level_dir: Path, model, tokenizer, max_new_tokens: int) -> tuple[int, int]:
    dst_level_dir.mkdir(parents=True, exist_ok=True)
    done, total = 0, 0
    for chunk_path in sorted(src_level_dir.glob("chunk_*.txt")):
        total += 1
        out_path = dst_level_dir / f"{chunk_path.stem}.json"
        if out_path.exists():
            continue
        text = chunk_path.read_text(encoding="utf-8")
        result = generate_json(model, tokenizer, SPECIFIC_SYSTEM_PROMPT, text, max_new_tokens)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        done += 1
    return done, total


def extract_synthesized_level(
    dst_level_dir: Path,
    finer_dst_level_dir: Path,
    ranges: list,
    children: list[list[int]],
    model,
    tokenizer,
    max_new_tokens: int,
) -> tuple[int, int]:
    dst_level_dir.mkdir(parents=True, exist_ok=True)
    done, total = 0, len(ranges)
    for i in range(total):
        out_path = dst_level_dir / f"chunk_{i + 1:03d}.json"
        if out_path.exists():
            continue
        child_concepts = [
            json.loads((finer_dst_level_dir / f"chunk_{child_idx + 1:03d}.json").read_text(encoding="utf-8"))
            for child_idx in children[i]
        ]
        user_content = json.dumps(child_concepts, ensure_ascii=False)
        result = generate_json(model, tokenizer, SYNTHESIZE_SYSTEM_PROMPT, user_content, max_new_tokens)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        done += 1
    return done, total


def extract_transcript(src_dir: Path, dst_dir: Path, model, tokenizer, max_new_tokens: int) -> None:
    hierarchy = json.loads((src_dir / "hierarchy.json").read_text(encoding="utf-8"))
    level_names = hierarchy["levels"]  # finest to coarsest, matching semantic_chunk.py's ordering

    out_dir = dst_dir / src_dir.name
    finest = level_names[0]
    done, total = extract_leaf_level(src_dir / finest, out_dir / finest, model, tokenizer, max_new_tokens)
    print(f"  level '{finest}': {done} generated, {total - done} already done, {total} total")

    for level_name in level_names[1:]:
        finer_name = hierarchy[level_name]["finer_level"]
        ranges = hierarchy[level_name]["ranges"]
        children = hierarchy[level_name]["children"]
        done, total = extract_synthesized_level(out_dir / level_name, out_dir / finer_name, ranges, children, model, tokenizer, max_new_tokens)
        print(f"  level '{level_name}': {done} generated, {total - done} already done, {total} total")


@hydra.main(config_path="../conf", config_name="extract_concepts", version_base=None)
def main(cfg: DictConfig) -> None:
    src_dir = Path(cfg.src)
    dst_dir = Path(cfg.dst)

    if not src_dir.is_dir():
        print(f"Source folder not found: {src_dir}", file=sys.stderr)
        sys.exit(1)

    transcript_dirs = sorted(p for p in src_dir.iterdir() if p.is_dir() and (p / "hierarchy.json").exists())
    if not transcript_dirs:
        print(f"No chunked transcripts (with hierarchy.json) found in {src_dir}")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading LLM '{cfg.model}' from '{cfg.model_dir}' (quantization={cfg.quantization})...")
    quantization_config = None
    if cfg.quantization == "4bit":
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    elif cfg.quantization == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model, cache_dir=cfg.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, cache_dir=cfg.model_dir, quantization_config=quantization_config, device_map="auto", dtype=torch.bfloat16
    )
    model.eval()

    succeeded, failed = 0, 0
    for transcript_dir in transcript_dirs:
        print(f"{transcript_dir.name}:")
        try:
            extract_transcript(transcript_dir, dst_dir, model, tokenizer, cfg.max_new_tokens)
        except Exception as e:
            # Per-chunk skip-checks inside extract_transcript already make a resumed run cheap,
            # so a failure here just means this file's remaining chunks wait for the next run -
            # no separate "skipped" bookkeeping needed at this level.
            print(f"  FAILED: {e}", file=sys.stderr)
            failed += 1
            continue
        succeeded += 1

    print(f"\n{succeeded} transcripts processed, {failed} failed, out of {len(transcript_dirs)} total")


if __name__ == "__main__":
    main()
