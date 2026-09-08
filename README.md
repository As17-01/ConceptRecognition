# ConceptRecognition

## Installation

The default environment contains only the lightweight dependencies used by API-backed and local
preprocessing stages, including `summarize_corpus`:

```bash
poetry install
```

Install the optional local neural-network stack for training, embedding, clustering, or
transcription with Whisper/Silero:

```bash
poetry install --with local-ml
```

The `local-ml` group is optional because it includes large packages such as PyTorch,
Transformers, and the audio inference runtimes. Development tooling can be added independently
with `poetry install --with dev` or together with the ML stack using
`poetry install --with dev,local-ml`.
