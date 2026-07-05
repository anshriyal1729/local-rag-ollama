# Local/On-Prem RAG with Ollama

A fully offline-capable RAG pipeline: TF-IDF retrieval (no external
embedding API) + generation via a **locally-running Ollama model** — so
neither your documents nor your queries ever leave the machine. Built to
mirror a common enterprise requirement (e.g. telecom/finance) where
customer data can't be sent to a third-party LLM API.

If Ollama isn't running, the pipeline automatically falls back to an
extractive answer instead of crashing — see `generated_by` in the result.

## How it works

- `src/ollama_client.py` — thin HTTP client for Ollama's local REST API
  (`/api/generate`, `/api/tags`), with a clean `OllamaUnavailableError` for
  graceful fallback handling.
- `src/rag.py` — `LocalRAGPipeline`: TF-IDF chunking/retrieval + tries
  Ollama for generation, falls back to extractive answers if Ollama isn't
  reachable.
- `data/sample_docs.py` — small bundled knowledge base about on-prem AI/RAG.
- `demo.py` — runs a few sample queries end-to-end.

## Quickstart

**Without Ollama** (fallback mode, zero setup):
```bash
pip install -r requirements.txt
python demo.py
```

**With Ollama** (real local LLM generation):
```bash
# 1. Install Ollama: https://ollama.com/download
ollama serve
ollama pull llama3

# 2. Run the demo — it will detect Ollama and use it automatically
python demo.py
```

## Running tests

```bash
pip install pytest
pytest tests/ -v
```
Tests run without Ollama installed — they specifically verify the fallback
path works correctly, since that's what most CI environments will exercise.

## Project layout

```
local-rag-ollama/
├── src/
│   ├── ollama_client.py  # HTTP client for local Ollama server
│   └── rag.py            # LocalRAGPipeline (retrieval + generation + fallback)
├── data/
│   └── sample_docs.py
├── tests/
│   └── test_rag.py
└── demo.py
```

## License

MIT — see [LICENSE](LICENSE).
