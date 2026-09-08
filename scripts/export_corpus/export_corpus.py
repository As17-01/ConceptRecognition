import re
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+")


def add_title(title: str, section: str, text: str) -> str:
    return f"# {title}\n\n## {section}\n\n{text.strip()}\n"


def format_transcript(text: str, sentences_per_paragraph: int) -> str:
    sentences = [sentence.strip() for sentence in SENTENCE_BOUNDARY.split(text.strip()) if sentence.strip()]
    paragraphs = [
        " ".join(sentences[start : start + sentences_per_paragraph])
        for start in range(0, len(sentences), sentences_per_paragraph)
    ]
    return "\n\n".join(paragraphs)


def collect_by_stem(directory: Path) -> dict[str, Path]:
    return {path.stem: path for path in sorted(directory.glob("*.txt"))}


def validate_inputs(
    transcripts: dict[str, Path], summaries: dict[str, Path], digest_path: Path
) -> None:
    missing_summaries = sorted(transcripts.keys() - summaries.keys())
    missing_transcripts = sorted(summaries.keys() - transcripts.keys())
    problems = []
    if not transcripts:
        problems.append("no preprocessed transcripts found")
    if missing_summaries:
        problems.append(f"missing summaries: {', '.join(missing_summaries)}")
    if missing_transcripts:
        problems.append(f"missing transcripts: {', '.join(missing_transcripts)}")
    if not digest_path.is_file():
        problems.append(f"corpus digest not found: {digest_path}")
    if problems:
        raise ValueError("; ".join(problems))


def export_corpus(
    transcripts_dir: Path,
    summaries_dir: Path,
    digest_path: Path,
    dst_dir: Path,
    sentences_per_paragraph: int,
) -> int:
    if sentences_per_paragraph < 1:
        raise ValueError("sentences_per_paragraph must be at least 1")

    transcripts = collect_by_stem(transcripts_dir)
    summaries = collect_by_stem(summaries_dir)
    validate_inputs(transcripts, summaries, digest_path)

    dst_dir.mkdir(parents=True, exist_ok=True)
    digest = digest_path.read_text(encoding="utf-8")
    (dst_dir / "corpus_digest.md").write_text(
        f"# Corpus digest\n\n{digest.strip()}\n", encoding="utf-8"
    )

    for class_name, transcript_path in transcripts.items():
        class_dir = dst_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        transcript = format_transcript(
            transcript_path.read_text(encoding="utf-8"), sentences_per_paragraph
        )
        summary = summaries[class_name].read_text(encoding="utf-8")
        (class_dir / "transcript.md").write_text(
            add_title(class_name, "Transcript", transcript), encoding="utf-8"
        )
        (class_dir / "summary.md").write_text(
            add_title(class_name, "Summary", summary), encoding="utf-8"
        )

    return len(transcripts)


@hydra.main(config_path="../conf", config_name="export_corpus", version_base=None)
def main(cfg: DictConfig) -> None:
    try:
        exported = export_corpus(
            Path(cfg.transcripts_src),
            Path(cfg.summaries_src),
            Path(cfg.digest_src),
            Path(cfg.dst),
            int(cfg.sentences_per_paragraph),
        )
    except (OSError, ValueError) as error:
        print(f"Export failed: {error}", file=sys.stderr)
        sys.exit(1)

    print(f"Exported {exported} classes and the corpus digest to {cfg.dst}")


if __name__ == "__main__":
    main()
