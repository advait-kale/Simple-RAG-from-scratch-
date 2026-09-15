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
| `main.py` | The app — FastAPI `/upload` + `/ask`, serves `index.html`, launches uvicorn. |
| `index.html` | Browser chat UI (drag-drop upload + streaming answers). |
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

Constants at the top of `main.py`:

| Name | Default | Meaning |
|------|---------|---------|
| `llm_model` | `qwen3:4b` | Ollama model that writes answers |
| `embedding_model` | `qwen3-embedding:0.6b` | Ollama model that makes vectors |
| `top_k` | `3` | chunks retrieved per question |
| `chunk_size` / `chunk_overlap` | `1000` / `150` | chunking granularity |
| `vector_store_path` | `data/vector_store` | where ChromaDB persists |

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
- **CORS `allow_origins=["*"]`** — fine for local dev; tighten for anything public.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| "Can't reach the API…" | server not running | `python main.py` |
| `/ask` errors or hangs | Ollama down / model not pulled | `ollama serve`; `ollama list` |
| Empty / "no answer" | nothing uploaded, or PDF had no text | upload a text-based PDF first |
| `model not found` | model missing in Ollama | `ollama pull qwen3:4b` |
| Editor shows red imports | wrong interpreter | select `.venv` as the IDE interpreter |
