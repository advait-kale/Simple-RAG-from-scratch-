"""Fixtures wiring the app to fakes so the suite runs without Ollama.

ChromaDB itself is real but in-memory, so retrieval, metadata round-tripping
and the cosine ranking are genuinely exercised rather than mocked away.
"""

import chromadb
import pytest
from fastapi.testclient import TestClient

import main
from helpers import FakeEmbedder, FakeLLM


@pytest.fixture
def splitter():
    return main.RecursiveCharacterTextSplitter(
        chunk_size=main.chunk_size,
        chunk_overlap=main.chunk_overlap,
        separators=["\n\n", "\n", " ", ".", ""],
    )


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def client(monkeypatch, splitter, llm):
    """A TestClient backed by fakes and a fresh in-memory Chroma collection.

    TestClient only runs the lifespan handler when it is used as a context
    manager. Constructing it plainly therefore skips lifespan entirely, which
    is what keeps Ollama and the on-disk vector store out of the tests.
    """
    # EphemeralClient hands back a cached client for identical settings, so the
    # collection outlives the test that made it. Drop it first, the same way
    # the app's own lifespan handler does, to start each test empty.
    chroma = chromadb.EphemeralClient()
    try:
        chroma.delete_collection(name=main.COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma.create_collection(
        name=main.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    state = main.app.state
    monkeypatch.setattr(state, "embedder", FakeEmbedder(), raising=False)
    monkeypatch.setattr(state, "llm", llm, raising=False)
    monkeypatch.setattr(state, "splitter", splitter, raising=False)
    monkeypatch.setattr(state, "chroma", chroma, raising=False)
    monkeypatch.setattr(state, "collection", collection, raising=False)

    return TestClient(main.app)
