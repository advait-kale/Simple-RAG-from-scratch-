"""Text extraction, chunking, and the bits of answer() that shape output."""

import main

from helpers import FakeLLM, make_pdf


def test_process_pdf_returns_one_entry_per_page_with_text():
    pages = main.process_pdf(make_pdf(["First page text", "Second page text"]))

    assert [number for number, _ in pages] == [1, 2]
    assert "First page text" in pages[0][1]
    assert "Second page text" in pages[1][1]


def test_process_pdf_skips_empty_pages_but_keeps_real_page_numbers():
    # Page 2 has no text, so the third page must still report itself as page 3.
    pages = main.process_pdf(make_pdf(["Alpha", "", "Gamma"]))

    assert [number for number, _ in pages] == [1, 3]
    assert "Gamma" in pages[1][1]


def test_process_pdf_returns_nothing_for_an_image_only_pdf():
    assert main.process_pdf(make_pdf(["", ""])) == []


def test_split_into_chunks_tags_every_chunk_with_its_page(splitter):
    pages = [(1, "alpha " * 400), (7, "omega " * 400)]

    chunks = main.split_into_chunks(pages, splitter, "report.pdf")

    assert len(chunks) > 2, "long pages should produce more than one chunk each"
    assert {chunk.metadata["source"] for chunk in chunks} == {"report.pdf"}
    assert {chunk.metadata["page"] for chunk in chunks} == {1, 7}

    for chunk in chunks:
        expected_page = 1 if "alpha" in chunk.page_content else 7
        assert chunk.metadata["page"] == expected_page


def test_describe_source_formats_file_and_page():
    assert main.describe_source({"source": "a.pdf", "page": 4}) == "a.pdf p.4"
    assert main.describe_source({"source": "a.pdf"}) == "a.pdf"
    assert main.describe_source(None) == "unknown"


def test_unique_sources_dedupes_but_keeps_order():
    metadatas = [
        {"source": "a.pdf", "page": 2},
        {"source": "a.pdf", "page": 2},
        {"source": "a.pdf", "page": 1},
    ]

    assert main.unique_sources(metadatas) == ["a.pdf p.2", "a.pdf p.1"]


def test_build_context_labels_each_chunk():
    context = main.build_context(
        ["chunk one", "chunk two"],
        [{"source": "a.pdf", "page": 1}, {"source": "b.pdf", "page": 9}],
    )

    assert "[a.pdf p.1]\nchunk one" in context
    assert "[b.pdf p.9]\nchunk two" in context


def test_build_context_survives_missing_metadata():
    context = main.build_context(["only chunk"], [])

    assert context == "[unknown]\nonly chunk"


def test_answer_strips_the_models_think_block(monkeypatch):
    tokens = ["<think>", "wondering", "</think>", "\n\n", "Real answer."]
    monkeypatch.setattr(main.app.state, "llm", FakeLLM(tokens), raising=False)

    assert "".join(main.answer("context", "q")) == "Real answer."


def test_answer_passes_through_output_with_no_think_block(monkeypatch):
    monkeypatch.setattr(
        main.app.state, "llm", FakeLLM(["  ", "Plain ", "answer."]), raising=False
    )

    assert "".join(main.answer("context", "q")) == "Plain answer."


def test_answer_puts_context_and_query_into_the_prompt(monkeypatch):
    llm = FakeLLM(["ok"])
    monkeypatch.setattr(main.app.state, "llm", llm, raising=False)

    list(main.answer("THE CONTEXT", "THE QUERY"))

    assert "THE CONTEXT" in llm.prompts[0]
    assert "THE QUERY" in llm.prompts[0]
