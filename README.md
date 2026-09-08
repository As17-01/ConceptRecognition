# ConceptRecognition

## Installation

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

Install the optional local neural-network stack for training, embedding, clustering, or
transcription with Whisper/Silero:

```bash
poetry install --with local-ml
```

The `local-ml` group is optional because it includes large packages such as PyTorch,
Transformers, and the audio inference runtimes. Development tooling can be added independently
with `poetry install --with dev` or together with the ML stack using
`poetry install --with dev,local-ml`.
