"""
main_test.py — single-file RAG FastAPI app.

Pipeline (proven in RAG_02.ipynb):
  upload PDF -> extract text -> chunk -> embed -> store in ChromaDB (COSINE)
  ask query -> embed -> nearest chunks -> stuff into prompt -> stream LLM answer

Run:
  uv run uvicorn main_test:app --reload
  open http://127.0.0.1:8000/docs   (interactive API)
"""

import io
import uuid

import chromadb
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------------- config ----------------
EMBED_MODEL = "all-MiniLM-L6-v2"      # 384-dim sentence embeddings
LLM_MODEL = "qwen3:4b"                 # local Ollama model (reasoning model)
CHROMA_PATH = "data/vector_store"
COLLECTION = "pdf_documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 3

RAG_PROMPT = """Use the following context to answer the question. If the answer is not in the context, say "I can't find the answer in the given context".

<context>
{context}
</context>

<question>
{question}
</question>
""".strip()


# ---------------- app + heavy objects (built once at startup) ----------------
app = FastAPI(title="RAG API (test)")


@app.on_event("startup")
def _startup():
    # embedding model: load once, reuse for every request
    app.embedder = SentenceTransformer(EMBED_MODEL)

    # LLM: reasoning=False stops qwen3 from emitting a <think> block
    app.llm = init_chat_model(
        model=LLM_MODEL,
        model_provider="ollama",
        temperature=0.0,
        reasoning=False,
    )

    # text splitter: even ~1000-char chunks with overlap
    app.splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ".", ""],
    )

    # ChromaDB: fresh COSINE collection each server start.
    # metric is locked at creation, so delete-then-create guarantees cosine
    # (default is L2, which breaks `similarity = 1 - distance`).
    app.chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        app.chroma.delete_collection(COLLECTION)
    except Exception:
        pass
    app.collection = app.chroma.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------- helpers ----------------
def extract_pdf_text(pdf_bytes: bytes) -> list[str]:
    """One non-empty text string per PDF page."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(text)
    return pages


def build_context(query: str, k: int = TOP_K) -> str:
    """Embed the query, pull the k nearest chunks, join into one context string."""
    if app.collection.count() == 0:
        return ""
    qv = app.embedder.encode([query])[0]
    res = app.collection.query(query_embeddings=[qv.tolist()], n_results=k)
    docs = res["documents"][0]   # nested: one inner list per query
    return "\n\n".join(docs)


def stream_answer(query: str):
    """Generator: stream the LLM answer token by token."""
    context = build_context(query)
    if not context:
        yield "No documents uploaded yet, or no relevant context found."
        return
    prompt = RAG_PROMPT.format(context=context, question=query)
    for chunk in app.llm.stream(prompt):
        yield chunk.content


# ---------------- routes ----------------
@app.post("/upload", description="Upload a PDF: extract, chunk, embed, store.")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are allowed")

    content = await file.read()
    pages = extract_pdf_text(content)
    if not pages:
        raise HTTPException(400, "Could not read any text from the PDF")

    # chunk every page into even pieces (create_documents splits internally)
    chunks = app.splitter.create_documents(pages)
    texts = [c.page_content for c in chunks]

    # embed + store
    vecs = app.embedder.encode(texts)
    app.collection.add(
        ids=[f"{uuid.uuid4().hex[:8]}_{i}" for i in range(len(texts))],
        embeddings=[v.tolist() for v in vecs],
        documents=texts,
        metadatas=[{"source": file.filename} for _ in texts],
    )

    return {
        "filename": file.filename,
        "pages": len(pages),
        "chunks_added": len(texts),
        "total_chunks": app.collection.count(),
    }


@app.post("/ask", description="Ask a question about the uploaded PDF(s).")
async def ask(query: str = Query(..., description="Your question")):
    return StreamingResponse(stream_answer(query), media_type="text/plain")


@app.get("/health")
def health():
    return {"status": "ok", "chunks": app.collection.count()}
