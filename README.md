# RAG from Scratch — PDF Q&A

A small **Retrieval-Augmented Generation** app: upload a PDF, ask questions, get
answers grounded in that document. Runs **fully local** — no cloud, no API keys.
Ollama for the LLM and embeddings, ChromaDB for the vector store, FastAPI for the
API, and a single-file HTML chat UI.

```
PDF ─► extract text ─► chunk ─► embed ─► ChromaDB (cosine)
                                              │
question ─► embed ─► nearest chunks ─► prompt ─► LLM ─► streamed answer
```

> **Demo:** _add a screenshot or GIF here_ — e.g. `docs/demo.gif`. This is the
> single highest-impact thing on the page; a 5-second clip of dropping a PDF and
> watching the answer stream in sells the project better than any paragraph.

---

## Stack

- **FastAPI** — `/upload` + `/ask` (streaming) API, also serves the frontend.
- **Ollama** — local LLM (`qwen3:4b`) + embeddings (`qwen3-embedding:0.6b`).
- **ChromaDB** — persistent vector store, cosine distance.
- **pypdf** — PDF text extraction.
- **Vanilla HTML/JS** — drag-drop upload + token-by-token streaming, zero build step.

---

## Run it

Prerequisites: **[Ollama](https://ollama.com/)** and **[uv](https://docs.astral.sh/uv/)** installed.

**1. Pull the models** (one-time, ~3 GB download)
```bash
ollama pull qwen3:4b              # writes the answers
ollama pull qwen3-embedding:0.6b  # turns text into vectors
```

**2. Install dependencies**
```bash
uv sync                           # creates .venv from pyproject.toml
```
Not using uv? `python -m venv .venv`, activate it, then `pip install -r requirements.txt`.

**3. Start Ollama** (must be running *before* the app)
```bash
ollama serve                      # or just launch the Ollama desktop app
```

**4. Start the app**
```bash
uv run python main.py             # (pip venv activated: python main.py)
```
It boots the server **and** auto-opens <http://127.0.0.1:8000> in your browser.

**5. Use it** — drop a PDF onto the upload strip, wait for **ready ✓**, type a
question, press Enter. The answer streams in token by token.

---

## What's inside

| Path | Purpose |
|------|---------|
| `main.py` | The app — the FastAPI endpoints, serves `index.html`, launches uvicorn. |
| `index.html` | Browser chat UI (upload + streaming answers + sources). |
| `tests/` | pytest suite; runs offline, no Ollama needed (see [Tests](#tests)). |
| `.env.example` | Every `RAG_*` setting with its default (see [Configuration](#configuration)). |
| `rag_app.spec` | PyInstaller recipe to bundle a standalone exe (see [Packaging](#packaging)). |
| `Learning_RAG/` | Notebooks + earlier variants showing how the pieces were built up. |
| `RAG_02.ipynb` | Loaders, chunking, and the ChromaDB cosine-vs-L2 lesson. |
| `pyproject.toml` / `requirements.txt` | Dependencies (uv / pip). |

> `data/` (uploaded PDFs + the vector store) is git-ignored — it's local runtime
> state, created on first run.

---

## How it works

**Ingest (`/upload`)**
1. `process_pdf` — read PDF bytes with `pypdf`, yielding `(page_number, text)` per
   page that has text.
2. `split_into_chunks` — `RecursiveCharacterTextSplitter`, ~1000-char chunks, 150
   overlap, each chunk tagged with its source filename and page.
3. `generate_embedding` — `OllamaEmbeddings.embed_documents` → one vector per chunk.
4. `collection.add` — store vectors + text + metadata in ChromaDB.

**Query (`/ask`)**
1. `get_context` — embed the question, pull the `top_k` nearest chunks with metadata.
2. `build_context` — prefix each chunk with `[file.pdf p.4]` so the model can cite it.
3. `answer` — fill `RAG_Prompt` with context + question, `app.state.llm.stream(...)`
   yields the answer token by token → `StreamingResponse`.

The chunks the answer came from are returned in an `X-Sources` response header
(percent-encoded JSON, e.g. `["report.pdf p.2","report.pdf p.7"]`) and shown
under the answer in the UI.

### Endpoints

| Method | Path | What it does |
|--------|------|--------------|
| `POST` | `/upload` | Index a PDF. Multipart, field name `file`. Returns page and chunk counts. |
| `POST` | `/ask` | Answer a question. JSON body `{"query": "..."}`. Streams `text/plain`; cites its chunks in the `X-Sources` header. |
| `GET` | `/health` | Liveness, the configured models, and how many chunks are indexed. |
| `GET` | `/documents` | Which files are indexed, with page and chunk counts. |
| `DELETE` | `/documents` | Empty the vector store without restarting. |
| `GET` | `/` | The browser UI. |

Error responses use FastAPI's `{"detail": "..."}` shape: `400` for a non-PDF, an
unreadable PDF, an empty question or a question asked before anything was
uploaded; `413` over the size cap; `422` for a malformed body; `404` when
retrieval matches nothing.

### API directly

```bash
# upload (field name must be "file")
curl -F "file=@your.pdf" http://127.0.0.1:8000/upload

# ask (question goes in a JSON body)
curl -X POST http://127.0.0.1:8000/ask      -H "Content-Type: application/json"      -d '{"query": "who created spider-man"}'
```

Interactive docs at <http://127.0.0.1:8000/docs>.

---

## Configuration

Everything is read from the environment at startup, with the defaults below. Copy
`.env.example` to `.env` and edit it (loaded automatically if `python-dotenv` is
installed), or export the variables before starting the app.

| Variable | Default | Meaning |
|----------|---------|---------|
| `RAG_LLM_MODEL` | `qwen3:4b` | Ollama model that writes answers |
| `RAG_EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | Ollama model that makes vectors |
| `RAG_TEMPERATURE` | `0.0` | sampling temperature for the answer |
| `RAG_TOP_K` | `3` | chunks retrieved per question |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `1000` / `150` | chunking granularity |
| `RAG_VECTOR_STORE_PATH` | `data/vector_store` | where ChromaDB persists |
| `RAG_MAX_UPLOAD_MB` | `10` | rejected above this, with a `413` |
| `RAG_HOST` / `RAG_PORT` | `127.0.0.1` / `8000` | where uvicorn binds |
| `RAG_ALLOWED_ORIGINS` | `*` | comma-separated CORS origins |

Bad values fail at startup with a message naming the variable, rather than
surfacing later as an empty retrieval or a confusing ChromaDB error.

---

## Tests

```bash
uv run pytest                     # or: pytest
```

30 tests, no Ollama required and nothing written to disk: the embedder and the
LLM are replaced with deterministic fakes, while ChromaDB runs in memory so
retrieval, metadata and the cosine ranking are exercised for real. They cover
PDF text extraction and page numbering, chunk metadata, the `<think>`-stripping
in `answer()`, and every endpoint including the error paths.

---

## Packaging

Bundle everything into a single Windows exe:

```bash
uv run pyinstaller rag_app.spec    # → dist/rag_app.exe
```

The exe boots the same server and serves the UI (Ollama still required on the host).
Note it's a large onefile build and can trip antivirus heuristics — for real
distribution prefer a `--onedir` build and code-signing. For running the project,
`python main.py` is the intended path.

---

## Notes & gotchas

- **Cosine, not L2.** Collection created with `metadata={"hnsw:space": "cosine"}`.
  ChromaDB defaults to squared-L2, which breaks `similarity = 1 - distance`. The metric
  is locked at creation — to change it you must delete + recreate the collection.
- **Fresh store each restart.** Startup deletes and recreates the collection, so
  uploaded PDFs do **not** survive a restart — re-upload after restarting. (Intentional
  for a demo; swap `delete_collection` for `get_or_create_collection` to persist.)
- **`reasoning=False`** on the LLM stops qwen3's `<think>` block, so answers are clean.
- **CORS defaults to `*`** — fine for local dev; set `RAG_ALLOWED_ORIGINS` for
  anything public. `X-Sources` is in `expose_headers`, or the browser would hide
  it whenever the UI points at a different origin.
- **Re-uploading the same file adds it twice.** Chunks are only ever appended, so
  the same PDF uploaded twice is indexed twice. `DELETE /documents` clears the
  store.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| "Can't reach the API…" | server not running | `python main.py` |
| `/ask` errors or hangs | Ollama down / model not pulled | `ollama serve`; `ollama list` |
| `400 No PDF has been uploaded yet` | nothing indexed | upload a PDF first |
| `400 Could not read any text` | scanned/image-only PDF | `pypdf` cannot OCR; use a text-based PDF |
| `404 Nothing ... matched` | question unrelated to the PDF | rephrase, or raise `RAG_TOP_K` |
| `model not found` | model missing in Ollama | `ollama pull qwen3:4b` |
| Editor shows red imports | wrong interpreter | select `.venv` as the IDE interpreter |
