from langchain.chat_models import init_chat_model

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field

import sys, os
from contextlib import asynccontextmanager

try:                       # optional: pip install python-dotenv to use a .env file
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

import chromadb
import uuid
import json
from urllib.parse import quote

from pypdf import PdfReader
import io


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _env_number(name: str, default, cast):
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be a {cast.__name__}, got {raw!r}") from None


# Every setting can be overridden from the environment (see .env.example); the
# defaults below are the ones the README documents and are what you get if you
# just run `python main.py`.
llm_model = _env_str("RAG_LLM_MODEL", "qwen3:4b")
embedding_model = _env_str("RAG_EMBEDDING_MODEL", "qwen3-embedding:0.6b")
temperature = _env_number("RAG_TEMPERATURE", 0.0, float)
pdf_path = _env_str("RAG_PDF_PATH", "data/pdfs")
vector_store_path = _env_str("RAG_VECTOR_STORE_PATH", "data/vector_store")
top_k = _env_number("RAG_TOP_K", 3, int)
chunk_size = _env_number("RAG_CHUNK_SIZE", 1000, int)
chunk_overlap = _env_number("RAG_CHUNK_OVERLAP", 150, int)
max_upload_mb = _env_number("RAG_MAX_UPLOAD_MB", 10, int)
MAX_UPLOAD_BYTES = max_upload_mb * 1024 * 1024
allowed_origins = [
    origin.strip()
    for origin in _env_str("RAG_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
host = _env_str("RAG_HOST", "127.0.0.1")
port = _env_number("RAG_PORT", 8000, int)

# Catch bad combinations here rather than halfway through an upload.
if top_k < 1:
    raise RuntimeError(f"RAG_TOP_K must be at least 1, got {top_k}")
if chunk_size < 1:
    raise RuntimeError(f"RAG_CHUNK_SIZE must be at least 1, got {chunk_size}")
if chunk_overlap < 0 or chunk_overlap >= chunk_size:
    raise RuntimeError(
        f"RAG_CHUNK_OVERLAP must be between 0 and RAG_CHUNK_SIZE-1 "
        f"({chunk_size - 1}), got {chunk_overlap}"
    )
if max_upload_mb < 1:
    raise RuntimeError(f"RAG_MAX_UPLOAD_MB must be at least 1, got {max_upload_mb}")


RAG_Prompt = """Use the context to answer the query

<context>
{context}
</context>

<query>
{query}
</query>

Asnwer: 
"""

COLLECTION_NAME = "pdf_documents"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the models and the vector store once, before the first request."""
    app.state.embedder = OllamaEmbeddings(
        model=embedding_model
    )

    app.state.llm = init_chat_model(
        model=llm_model,
        model_provider="ollama",
        temperature=temperature,
        reasoning=False
    )

    app.state.splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ".", ""]
    )

    app.state.chroma = chromadb.PersistentClient(path=vector_store_path)
    try:
        app.state.chroma.delete_collection(name=COLLECTION_NAME)
    except Exception:
        # Nothing to delete on a first run. That is the normal path, so it
        # should not print anything that looks like a failure.
        pass

    app.state.collection = app.state.chroma.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    yield


app = FastAPI(title="RAG API", lifespan=lifespan)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question to answer from the uploaded PDFs")


def process_pdf(content: bytes):
    """Extract text per page as (page_number, text), skipping empty pages.

    The page number is the one printed in a PDF reader (1-based). Pages with
    no extractable text are dropped, which is why the number has to be
    carried along rather than recovered from the list index later.
    """
    reader = PdfReader(io.BytesIO(content))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((page_number, text))
    return pages


def split_into_chunks(pages, splitter, source: str):
    """Chunk each page, tagging every chunk with the page it came from."""
    texts = [text for _, text in pages]
    metadatas = [{"source": source, "page": number} for number, _ in pages]
    # create_documents copies each metadata dict onto every chunk cut from
    # that page, so a chunk always knows its own page.
    return splitter.create_documents(texts, metadatas=metadatas)

def generate_embedding(chunks):
    return app.state.embedder.embed_documents(chunks)   # list[str] -> list[list[float]]

def get_context(query: str, k: int | None = None):
    query_embedding = app.state.embedder.embed_query(query)   # str -> list[float]
    results = app.state.collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k if k is None else k,
    )
    # Chroma returns one list per query embedding; an empty store gives
    # back [[]] or even [], so do not index into it blindly.
    matches = results.get("documents") or []
    documents = matches[0] if matches else []
    meta_matches = results.get("metadatas") or []
    metadatas = meta_matches[0] if meta_matches else []
    return documents, metadatas


def describe_source(metadata) -> str:
    """Human-readable label for one retrieved chunk, e.g. "report.pdf p.4"."""
    metadata = metadata or {}
    source = metadata.get("source") or "unknown"
    page = metadata.get("page")
    return f"{source} p.{page}" if page else source


def build_context(documents, metadatas) -> str:
    """Prefix each chunk with where it came from so the model can cite it."""
    blocks = []
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        blocks.append(f"[{describe_source(metadata)}]\n{document}")
    return "\n\n".join(blocks)


def unique_sources(metadatas):
    """Ordered, de-duplicated source labels for the retrieved chunks."""
    labels = []
    for metadata in metadatas:
        label = describe_source(metadata)
        if label not in labels:
            labels.append(label)
    return labels


def answer(context: str, query: str):
    prompt = RAG_Prompt.format(context=context, query=query)

    buffer = ""
    passed_think = False
    started = False        # so we can strip leading blank lines off the answer
    for ans in app.state.llm.stream(prompt):
        token = ans.content or ""
        if passed_think:
            if not started:
                token = token.lstrip()
                if not token:
                    continue
                started = True
            yield token
            continue
        buffer += token
        if "</think>" in buffer:
            passed_think = True
            after = buffer.split("</think>", 1)[1].lstrip()
            buffer = ""
            if after:
                started = True
                yield after

    if not passed_think and buffer:
        yield buffer.lstrip()


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are allowed")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "PDF too large (max 10 MB)")

    pages = process_pdf(content)
    if not pages:
        raise HTTPException(400, "Could not read any text from the PDF")

    chunks = split_into_chunks(pages, app.state.splitter, file.filename)
    chunks_page_content = [chunk.page_content for chunk in chunks]
    embeddings = generate_embedding(chunks_page_content)

    # One id prefix per upload keeps the chunks of a single file groupable.
    batch = uuid.uuid4().hex[:8]
    app.state.collection.add(
        ids=[f"{batch}_{i}" for i in range(len(chunks_page_content))],
        embeddings=embeddings,
        documents=chunks_page_content,
        metadatas=[chunk.metadata for chunk in chunks]
    )

    return {
        "filename": file.filename,
        "pages_indexed": len(pages),
        "chunks_indexed": len(chunks_page_content),
    }

@app.post("/ask")
async def ask(request: AskRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(400, "Question cannot be empty")

    # Retrieval has to happen here, not inside the generator. Once
    # StreamingResponse starts writing, the status line is already on the
    # wire and an HTTPException raised later would arrive as a broken 200
    # instead of a 4xx the frontend can show.
    if app.state.collection.count() == 0:
        raise HTTPException(400, "No PDF has been uploaded yet")

    documents, metadatas = get_context(query)
    context = build_context(documents, metadatas)
    if not context.strip():
        raise HTTPException(404, "Nothing in the uploaded PDFs matched that question")

    # The body is a stream, so the citations ride along in a header the
    # client can read as soon as the response starts.
    return StreamingResponse(
        answer(context, query),
        media_type="text/plain",
        headers={"X-Sources": quote(json.dumps(unique_sources(metadatas)))},
    )


@app.get("/health")
def health():
    """Cheap readiness probe: is the app up, and what is it holding?"""
    return {
        "status": "ok",
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "chunks_indexed": app.state.collection.count(),
    }


@app.get("/documents")
def list_documents():
    """Which PDFs are currently indexed, and how much of each."""
    stored = app.state.collection.get(include=["metadatas"])

    chunks_per_file = {}
    pages_per_file = {}
    for metadata in stored.get("metadatas") or []:
        metadata = metadata or {}
        source = metadata.get("source") or "unknown"
        chunks_per_file[source] = chunks_per_file.get(source, 0) + 1
        page = metadata.get("page")
        if page is not None:
            pages_per_file.setdefault(source, set()).add(page)

    return {
        "documents": [
            {
                "filename": filename,
                "chunks": count,
                "pages": len(pages_per_file.get(filename, ())),
            }
            for filename, count in sorted(chunks_per_file.items())
        ]
    }


@app.delete("/documents")
def clear_documents():
    """Empty the vector store without restarting the server."""
    removed = app.state.collection.count()
    app.state.chroma.delete_collection(name=COLLECTION_NAME)
    app.state.collection = app.state.chroma.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return {"deleted_chunks": removed}


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,   # "*" by default — fine for local dev
    allow_methods=["*"],             # GET, POST, OPTIONS, ...
    allow_headers=["*"],
    expose_headers=["X-Sources"],    # custom headers are hidden cross-origin otherwise
)


def resource_path(rel: str) -> str:
    # works in dev AND inside a PyInstaller exe (bundled files land in sys._MEIPASS)
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)

@app.get("/")
def index():
    return FileResponse(resource_path("index.html"))


if __name__ == "__main__":
    import uvicorn
    import webbrowser, threading
    threading.Timer(1.5, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    uvicorn.run(app, host=host, port=port)


