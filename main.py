import ollama
from langchain.chat_models import init_chat_model

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

import chromadb
from sklearn.metrics.pairwise import cosine_similarity
import os, glob, uuid
from pathlib import Path

from pypdf import PdfReader
import io


llm_model = "qwen3:4b"
embedding_model = "qwen3-embedding:0.6b"
temprature = 0.0
pdf_path = "data/pdfs"
vector_store_path = "data/vector_store"
top_k = 3
chunk_size = 1000
chunk_overlap = 150
RAG_Prompt = """Use the context to answer the query

<context>
{context}
</context>

<query>
{query}
</query>

Asnwer: 
"""

app = FastAPI(title="RAG API ")

@app.on_event("startup")
def _startup():

    app.embedder = OllamaEmbeddings(
        model=embedding_model
    )

    app.llm = init_chat_model(
        model=llm_model,
        model_provider="ollama",
        temperature=temprature,
        reasoning=False
    )

    app.splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ".", ""]
    )

    app.chroma = chromadb.PersistentClient(path=vector_store_path)
    try:
        app.chroma.delete_collection(name="pdf_documents")

    except Exception as e:
        print("Error {e}")
        

    app.collection = app.chroma.create_collection(
        name="pdf_documents",
        metadata={"hnsw:space": "cosine"}
    )

def process_pdf(content: bytes):
    reader = PdfReader(io.BytesIO(content))
    pages = []
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(text)
    return pages

def split_into_chunks(documents, splitter):
    return splitter.create_documents(documents)   # list[str] -> chunked Documents

def generate_embedding(chunks):
    return app.embedder.embed_documents(chunks)   # list[str] -> list[list[float]]

def get_context(query: str, top_k: int = top_k):
    query_embedding = app.embedder.embed_query(query)   # str -> list[float]
    results = app.collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    return "\n\n".join(results["documents"][0])

def answer(query: str):
    context = get_context(query)
    prompt = RAG_Prompt.format(context=context, query=query)
    for ans in app.llm.stream(prompt):
        yield ans.content
def response(query: str):
    full_response = ""
    in_thinking = False
    for content in answer(query):

        full_response += content
        
        if "<think>" in content:
            in_thinking = True
            content = content.replace("<think>", "")
        if in_thinking:
            if "</think>" in content:
            content = content.replace("</think>", "")
        

    yield full_response


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are allowed")

    content = await file.read()
    pages = process_pdf(content)
    if not pages:
        raise HTTPException(400, "Could not read any text from the PDF")

    chunks = split_into_chunks(pages, app.splitter)
    chunks_page_content = [chunk.page_content for chunk in chunks]
    embeddings = generate_embedding(chunks_page_content)

    app.collection.add(
        ids=[f"{uuid.uuid4().hex[:8]}_{i}" for i in range(len(chunks_page_content))],
        embeddings=embeddings,
        documents=chunks_page_content,
        metadatas=[{"source": file.filename} for _ in chunks_page_content]
    )

    return "Uploaded"

@app.post("/ask")
async def ask(query: str):
    return StreamingResponse(response(query), media_type="text/plain")


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # any origin — fine for local dev
    allow_methods=["*"],     # GET, POST, OPTIONS, ...
    allow_headers=["*"],
)


