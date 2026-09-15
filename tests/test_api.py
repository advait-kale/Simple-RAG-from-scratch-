"""End-to-end tests against the HTTP surface, with fake models behind it."""

import json
from urllib.parse import unquote

import main

from helpers import make_pdf


def upload(client, pages, filename="report.pdf"):
    return client.post(
        "/upload",
        files={"file": (filename, make_pdf(pages), "application/pdf")},
    )


# --------------------------------------------------------------------------
# /upload
# --------------------------------------------------------------------------

def test_upload_reports_what_it_indexed(client):
    response = upload(client, ["Spider-Man was created by Stan Lee and Steve Ditko."])

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "report.pdf"
    assert body["pages_indexed"] == 1
    assert body["chunks_indexed"] >= 1


def test_upload_rejects_anything_that_is_not_a_pdf(client):
    response = client.post(
        "/upload", files={"file": ("notes.txt", b"plain text", "text/plain")}
    )

    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_upload_rejects_a_pdf_over_the_size_cap(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 128)

    response = upload(client, ["This document is comfortably over 128 bytes once " * 5])

    assert response.status_code == 413


def test_upload_rejects_a_pdf_with_no_extractable_text(client):
    # Two pages, neither carrying a text stream - the scanned-document case.
    response = upload(client, ["", ""])

    assert response.status_code == 400
    assert "text" in response.json()["detail"].lower()


def test_upload_is_case_insensitive_about_the_extension(client):
    assert upload(client, ["Some text"], filename="REPORT.PDF").status_code == 200


# --------------------------------------------------------------------------
# /ask
# --------------------------------------------------------------------------

def test_ask_before_uploading_anything_is_a_400(client):
    response = client.post("/ask", json={"query": "who created spider-man"})

    assert response.status_code == 400
    assert "No PDF" in response.json()["detail"]


def test_ask_with_an_empty_query_is_rejected(client):
    upload(client, ["Spider-Man was created by Stan Lee."])

    response = client.post("/ask", json={"query": ""})

    assert response.status_code == 422   # pydantic min_length, before the handler


def test_ask_with_a_whitespace_only_query_is_a_400(client):
    upload(client, ["Spider-Man was created by Stan Lee."])

    response = client.post("/ask", json={"query": "   "})

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_ask_requires_the_query_field(client):
    assert client.post("/ask", json={}).status_code == 422


def test_ask_streams_the_answer(client):
    upload(client, ["Spider-Man was created by Stan Lee and Steve Ditko."])

    response = client.post("/ask", json={"query": "who created spider-man"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "The answer is 42."   # the fake LLM's script


def test_ask_sends_the_retrieved_chunk_to_the_model(client, llm):
    upload(client, ["Spider-Man was created by Stan Lee and Steve Ditko."])

    client.post("/ask", json={"query": "who created spider-man"})

    prompt = llm.prompts[0]
    assert "Stan Lee" in prompt, "retrieved context should reach the prompt"
    assert "who created spider-man" in prompt


def test_ask_returns_the_sources_it_used(client):
    upload(client, ["Spider-Man was created by Stan Lee."], filename="comics.pdf")

    response = client.post("/ask", json={"query": "who created spider-man"})

    sources = json.loads(unquote(response.headers["X-Sources"]))
    assert sources == ["comics.pdf p.1"]


def test_ask_cites_the_page_the_answer_came_from(client):
    upload(
        client,
        ["Nothing relevant on this page.", "", "Spider-Man was created by Stan Lee."],
        filename="comics.pdf",
    )

    response = client.post("/ask", json={"query": "spider man stan lee created"})

    sources = json.loads(unquote(response.headers["X-Sources"]))
    assert "comics.pdf p.3" in sources, f"expected page 3, got {sources}"


# --------------------------------------------------------------------------
# /health and /documents
# --------------------------------------------------------------------------

def test_health_reports_the_configured_models(client):
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["llm_model"] == main.llm_model
    assert body["embedding_model"] == main.embedding_model
    assert body["chunks_indexed"] == 0


def test_health_chunk_count_tracks_uploads(client):
    upload(client, ["Some text to index."])

    assert client.get("/health").json()["chunks_indexed"] >= 1


def test_documents_is_empty_to_start_with(client):
    assert client.get("/documents").json() == {"documents": []}


def test_documents_lists_each_file_with_page_and_chunk_counts(client):
    upload(client, ["Page one text.", "", "Page three text."], filename="a.pdf")
    upload(client, ["Only page."], filename="b.pdf")

    documents = client.get("/documents").json()["documents"]

    assert [entry["filename"] for entry in documents] == ["a.pdf", "b.pdf"]
    assert documents[0]["pages"] == 2      # the blank page is not indexed
    assert documents[1]["pages"] == 1
    assert all(entry["chunks"] >= 1 for entry in documents)


def test_delete_documents_empties_the_store(client):
    upload(client, ["Some text to index."])
    assert client.get("/health").json()["chunks_indexed"] >= 1

    response = client.delete("/documents")

    assert response.status_code == 200
    assert response.json()["deleted_chunks"] >= 1
    assert client.get("/health").json()["chunks_indexed"] == 0
    assert client.get("/documents").json() == {"documents": []}


def test_asking_after_a_clear_is_a_400_again(client):
    upload(client, ["Some text to index."])
    client.delete("/documents")

    response = client.post("/ask", json={"query": "anything"})

    assert response.status_code == 400
