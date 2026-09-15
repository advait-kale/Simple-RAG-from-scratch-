"""Test doubles and a tiny PDF writer.

The tests run without Ollama, so the two pieces that need a model - the
embedder and the LLM - are replaced with deterministic fakes.
"""

import math
import re
import zlib

EMBEDDING_DIM = 64


def fake_embed(text: str):
    """Hash words into a small unit vector so similar text lands near each other.

    Not a real embedding, but it is deterministic and it preserves the one
    property the retrieval tests care about: a query sharing words with a chunk
    scores higher than one that does not.
    """
    vector = [0.0] * EMBEDDING_DIM
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        # crc32 rather than hash(): hashing of str is salted per process.
        vector[zlib.crc32(word.encode()) % EMBEDDING_DIM] += 1.0

    length = math.sqrt(sum(value * value for value in vector))
    if length == 0:
        # An all-zero vector has no direction, so cosine distance is undefined.
        return [1.0] + [0.0] * (EMBEDDING_DIM - 1)
    return [value / length for value in vector]


class FakeEmbedder:
    def embed_documents(self, texts):
        return [fake_embed(text) for text in texts]

    def embed_query(self, text):
        return fake_embed(text)


class FakeToken:
    """Stands in for a LangChain message chunk, which is read via .content."""

    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Replays a fixed list of tokens and records the prompts it was given."""

    def __init__(self, tokens=None):
        self.tokens = ["The ", "answer ", "is ", "42."] if tokens is None else tokens
        self.prompts = []

    def stream(self, prompt):
        self.prompts.append(prompt)
        for token in self.tokens:
            yield FakeToken(token)


def make_pdf(pages):
    """Build a minimal single-font PDF from a list of page strings.

    Written by hand so the tests need nothing beyond pypdf, which the project
    already depends on. An empty string produces a page with no text, which is
    how the skip-empty-pages behaviour gets covered.
    """
    # Object ids: 1 = catalog, 2 = page tree, 3 = the shared font, then a page
    # object and a content stream for each page.
    page_object_ids = [4 + index * 2 for index in range(len(pages))]
    kids = " ".join(f"{object_id} 0 R" for object_id in page_object_ids)

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    for index, text in enumerate(pages):
        content_id = page_object_ids[index] + 1

        if text:
            escaped = (
                text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            )
            stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode()
        else:
            stream = b""

        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> "
                f"/Contents {content_id} 0 R >>"
            ).encode()
        )
        objects.append(
            b"<< /Length "
            + str(len(stream)).encode()
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_offset = len(out)
    size = len(objects) + 1
    out += f"xref\n0 {size}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {size} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n"
    ).encode()
    out += b"%%EOF\n"
    return bytes(out)
