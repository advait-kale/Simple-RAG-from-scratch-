# RAG from Scratch — Notebook Notes

`01_RAG_YT.ipynb` builds a minimal RAG retrieval step with TF-IDF + cosine similarity (no LLM yet).

## Skills

- **`rag-teacher`** (`.claude/skills/rag-teacher/SKILL.md`) — Socratic RAG tutor following the curiousily/AI-Bootcamp curriculum. Teaches one concept at a time with worked examples and check questions. Invoke with `/rag-teacher` or "teach me RAG".

## Pipeline

1. **Knowledge base** — `KNOWLEDGE_BASE` string holds 5 sections (1.1–1.5) on handling customer complaints.
2. **Chunking** — `KNOWLEDGE_BASE.strip().split("\n\n")` splits on blank lines → 5 chunks (one per section).
   - Common bug: `split("/n")` or `split("/n/n")` uses literal slash-n, not a newline. Must be backslash `"\n\n"`.
3. **Vectorize** — `TfidfVectorizer().fit_transform(chunks)` → sparse matrix shape `(5, vocab)`. Each row = one chunk as a TF-IDF vector.
4. **Retrieve** — embed query with the same vectorizer, score against all chunks, return top k.

## `retrieve(query, k)`

```python
def retrieve(query: str, k: int):
    query_vector = vectorize.transform([query])                       # query → TF-IDF (reuse fitted vocab)
    similarity = cosine_similarity(query_vector, chunks_vectors).flatten()  # (5,) score per chunk
    top_k_indices = np.argsort(similarity)[::-1][:k]                  # indices of best k chunks
    return [chunks[i] for i in top_k_indices], similarity[top_k_indices]
```

### `top_k_indices`

Array of chunk positions (indices into `chunks`) for the k highest-scoring chunks, ordered best→worst.

- `np.argsort(similarity)` — indices sorted by score **ascending**.
- `[::-1]` — flip to **descending** (best first).
- `[:k]` — keep top k.

It stores *which* chunks win, not the text. `chunks[i]` then pulls the actual text.

### Gotchas

- Use `vectorize.transform`, not `fit_transform`, on the query — must reuse the vocab learned from chunks.
- Wrong: `np.argsort(similarity[-k:][::-1])` / `similarity[:k][::-1]` — slices by position, not score, and indices no longer map to `chunks`.
