# ConceptRecognition

## Installation

Use Python 3.12 or 3.13; Hydra is not compatible with Python 3.14's `argparse` changes.

The default environment contains only the lightweight dependencies used by API-backed and local
preprocessing stages, including `summarize_corpus`:

```bash
poetry install
```

Copy `.env.example` to `.env` and add the API key used by the configured OpenAI-compatible
endpoint. The example also sets `OPENAI_BASE_URL` to `https://direct.router-cheap.com/v1`:

```bash
cp .env.example .env
```

`summarize_corpus` loads both variables from this file. The `.env` file is excluded from Git.

Export the preprocessed transcripts, per-class summaries, and corpus digest as a browsable
Markdown directory:

```bash
poetry run python scripts/export_corpus/export_corpus.py
```

The result is written to `data/corpus_markdown`: each class gets a folder containing
`transcript.md` and `summary.md`, and `corpus_digest.md` is written at the root.

Install the optional local neural-network stack for training, embedding, clustering, or
transcription with Whisper/Silero:

```bash
poetry install --with local-ml
```

The `local-ml` group is optional because it includes large packages such as PyTorch,
Transformers, and the audio inference runtimes. Development tooling can be added independently
with `poetry install --with dev` or together with the ML stack using
`poetry install --with dev,local-ml`.

## Group classes by topic

Run the LLM topic distributor after generating the summaries, corpus digest, and Markdown export:

```bash
poetry run python scripts/distribute_corpus/distribute_corpus.py
```

Settings are in `scripts/conf/distribute_corpus.yaml`. The script loads `.env`, derives a shared
Russian topic list from `data/corpus_digest.txt` and all class summaries, then sends each class's `summary.md` together
with that digest and topic list to the LLM. It copies the entire class folder into
`data/corpus_by_topic_detailed/<topic>/<class>/` for every focused matching topic.
The taxonomy aims for 20–40 specific topics where supported, rather than copying digest headings.
Assignments require a primary focus or a substantial secondary block, with an exact summary
excerpt as evidence; routine background techniques do not qualify. Classes without a
matching topic go into `_unmatched`. Source folders are preserved.

`manifest.json` records the topic definitions, assignments, primary/secondary roles, evidence excerpts, and reasons. Reruns reuse completed
LLM results and retry missing assignments/copies. If the source files, digest, prompts, or model
settings change, use a new destination to keep output versions separate:

```bash
poetry run python scripts/distribute_corpus/distribute_corpus.py dst=data/corpus_by_topic_v2
```

Malformed or incomplete LLM responses are reported as failures; rerun to retry. The script exits
with a nonzero status if any class fails. It makes one taxonomy request plus one request per
uncached class when you run it.

The detailed distributor can use an existing digest because it also reads all class summaries.
To optionally regenerate a more detailed digest with the revised summarization prompt while
preserving the existing digest, run:

```bash
poetry run python scripts/summarize_corpus/summarize_corpus.py digest_dst=data/corpus_digest_detailed.txt
poetry run python scripts/distribute_corpus/distribute_corpus.py digest_src=data/corpus_digest_detailed.txt dst=data/corpus_by_topic_detailed_v2
```

Existing per-class summaries are reused by the summarizer. Neither command updates the previously
exported `data/corpus_markdown/corpus_digest.md` automatically.
