# RAG from Scratch — PDF Q&A

A small **Retrieval-Augmented Generation** app: upload a PDF, ask questions, get
answers grounded in that document. Runs fully **local** — Ollama for the LLM and
embeddings, ChromaDB for the vector store, FastAPI for the API, and a single-file
HTML chat frontend.

```
PDF ─► extract text ─► chunk ─► embed ─► ChromaDB (cosine)
                                              │
question ─► embed ─► nearest chunks ─► prompt ─► LLM ─► streamed answer
```

---

## What's inside

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app: `/upload` and `/ask` (streaming). The real app. |
| `index.html` | Browser chat frontend (drag-drop upload + streaming answers). |
| `main_test.py` | Earlier self-contained variant (kept for reference). |
| `data/pdfs/` | Sample PDFs (`spiderman_sample.pdf`). |
| `data/vector_store/` | ChromaDB persistent store (auto-created). |
| `pyproject.toml` | Dependencies (managed with **uv**). |
| `requirements.txt` | Same dependencies for plain `pip install -r`. |

---

## Prerequisites

1. **[uv](https://docs.astral.sh/uv/)** — Python package/venv manager.
2. **[Ollama](https://ollama.com/)** running locally (`localhost:11434`).
3. Pull the two models used:
   ```bash
   ollama pull qwen3:4b              # the LLM that writes answers
   ollama pull qwen3-embedding:0.6b  # turns text into vectors
   ```
   > Models live wherever `OLLAMA_MODELS` points (this project used
   > `C:\Advait\Ollama`). Set it, restart Ollama, then pull.

---

## Setup

```bash
# from the project folder
uv sync          # creates .venv and installs everything from pyproject.toml
```

Not using uv? Use pip with the provided `requirements.txt`:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Run

```bash
# 1. make sure Ollama is running (ollama serve, or the desktop app)
# 2. start the API
uv run uvicorn main:app --reload
# (with a pip venv activated, just: uvicorn main:app --reload)
```

- API: <http://127.0.0.1:8000>
- Interactive docs: <http://127.0.0.1:8000/docs>

Then open **`index.html`** in your browser (double-click it). Drop a PDF, ask away.

---

## Usage

### Frontend
1. Drag a PDF onto the upload strip (or click to choose). Wait for **ready ✓**.
2. Type a question, press **Enter**. The answer streams in token by token.

### API directly (curl)
```bash
# upload a PDF (field name must be "file")
curl -F "file=@data/pdfs/spiderman_sample.pdf" http://127.0.0.1:8000/upload

# ask a question (query is a query-string param)
curl -X POST "http://127.0.0.1:8000/ask?query=who%20created%20spider-man"
```

---

## How it works

**Ingest (`/upload`)**
1. `process_pdf` — read the PDF bytes with `pypdf`, one text string per page.
2. `split_into_chunks` — `RecursiveCharacterTextSplitter` cuts even ~1000-char
   chunks (150 overlap) so retrieval is fine-grained.
3. `generate_embedding` — `OllamaEmbeddings.embed_documents` turns each chunk into
   a vector.
4. `collection.add` — store vectors + text in ChromaDB.

**Query (`/ask`)**
1. `get_context` — embed the question (`embed_query`), pull the `top_k` nearest
   chunks, join them.
2. `answer` — fill the `RAG_Prompt` with that context + question, then
   `app.llm.stream(...)` yields the answer token by token → `StreamingResponse`.

---

## Configuration

Edit the constants at the top of `main.py`:

| Name | Default | Meaning |
|------|---------|---------|
| `llm_model` | `qwen3:4b` | Ollama model that writes answers |
| `embedding_model` | `qwen3-embedding:0.6b` | Ollama model that makes vectors |
| `top_k` | `3` | how many chunks to retrieve per question |
| `chunk_size` / `chunk_overlap` | `1000` / `150` | chunking granularity |
| `vector_store_path` | `data/vector_store` | where ChromaDB persists |

---

## Notes & gotchas

- **Cosine, not L2.** The collection is created with `metadata={"hnsw:space": "cosine"}`.
  ChromaDB defaults to **squared-L2**, which breaks `similarity = 1 - distance`. The
  metric is locked at creation — to change it you must delete + recreate the collection.
- **Fresh store each restart.** Startup deletes and recreates the collection, so
  uploaded PDFs do **not** survive a server restart — re-upload after restarting.
- **`reasoning=False`** on the LLM stops qwen3 from emitting a `<think>` block, so
  answers are clean with no manual stripping.
- **CORS is enabled** (`allow_origins=["*"]`) so the `file://` frontend can call the
  API. Fine for local dev; tighten for anything public.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Frontend: "Can't reach the API…" | server not running / CORS | start uvicorn; CORS is already on |
| `/ask` errors or hangs | Ollama not running, or model not pulled | `ollama serve`; `ollama list` |
| Empty / "no answer" | nothing uploaded, or PDF had no text | upload a text-based PDF first |
| `model not found` | model missing in Ollama | `ollama pull qwen3:4b` etc. |
| Editor shows red imports | wrong interpreter | select `.venv` in your IDE; running uses `uv run` regardless |
