import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

import pypdfium2  # FIX: was `from pypdfium2 import pypdfium2` (wrong import)
from langchain.chat_models import init_chat_model

# ---- config ----
model = "qwen3-vl:4b"
TEMPERATURE = 0.0

RAG_PROMPT = """Use the following context to answer the question. If you cannot find the answer in the context say "I can't find the answer to this in the given context"

<contexts>
{context}
</contexts>

<question>
{query}
</question>
""".strip()

# ---- app + state ----
app = FastAPI(title="RAG API")
app.chunks = []           # uploaded text chunks (one per PDF page)
app.vectorize = None      # the fitted TfidfVectorizer
app.chunk_vectors = None  # TF-IDF matrix of chunks
app.llm = init_chat_model(
    model=model,
    model_provider="ollama",
    temperature=TEMPERATURE,  # FIX: was `temprature`
    reasoning=False,          # kill qwen3 <think> block
)


# ---- pdf -> text ----
def extract_pdf_pages(pdf_content):
    pdf = pypdfium2.PdfDocument(pdf_content)
    pages = []
    for page_i in range(len(pdf)):
        page = pdf.get_page(page_i)
        textpage = page.get_textpage()
        page_text = textpage.get_text_bounded().strip()
        if page_text:
            pages.append(page_text)
    return pages


# ---- retrieval ----
def retrieve(query: str, k: int = 6):
    # FIX: read app.vectorize / app.chunk_vectors (server state), not globals
    if not app.chunks or app.vectorize is None:
        return [], np.array([])
    query_vector = app.vectorize.transform([query])
    similarity = cosine_similarity(query_vector, app.chunk_vectors).flatten()
    top_k_indices = np.argsort(similarity)[::-1][:k]
    return [app.chunks[i] for i in top_k_indices], similarity[top_k_indices]


# ---- generation ----
def answer(query: str, k: int = 2):
    if not app.chunks:
        raise HTTPException(400, "Upload doc first")
    context_chunks, scores = retrieve(query, k)
    context_string = "\n\n".join(
        f"<context>\n{c}\n</context>" for c in context_chunks
    )
    prompt = RAG_PROMPT.format(context=context_string, query=query)
    for chunk in app.llm.stream(prompt):  # FIX: app.llm, not global llm
        yield chunk.content


# ---- routes ----
@app.post("/upload", description="Upload the PDF file")
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files is allowed")
    content = await file.read()
    pages = extract_pdf_pages(content)

    if not pages:
        raise HTTPException(400, "Error in reading the pdf file")

    app.chunks.extend(pages)
    app.vectorize = TfidfVectorizer()
    app.chunk_vectors = app.vectorize.fit_transform(app.chunks)

    return {"filename": file.filename, "page_added": len(pages)}


@app.post("/ask", description="Ask a question about the uploaded PDF")
async def ask(query: str = Query(..., description="Your question")):
    # stream the LLM answer back token by token
    return StreamingResponse(answer(query), media_type="text/plain")

