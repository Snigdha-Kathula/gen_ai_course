# Session 10 — Full RAG Pipeline
> Snigdhaa's GenAI Course · Save as `session10/concepts.md`

---

## The one line summary
RAG = Retrieve relevant chunks → Feed to LLM → Grounded answer. Retrieval quality determines everything.

---

## The two phases of RAG

### Phase 1 — Indexing (runs once)
```
Load document
     ↓
Chunk with RecursiveCharacterTextSplitter
     ↓
Embed each chunk → get_embedding(chunk)
     ↓
Store in vector DB with metadata
```

### Phase 2 — Retrieval + Generation (runs every query)
```
User query
     ↓
Embed query → get_embedding(query)
     ↓
Semantic search → top K chunks
     ↓
Filter by similarity threshold (> 0.3)
     ↓
Inject chunks into LLM prompt as context
     ↓
LLM generates grounded answer
```

---

## The complete RAG loop in code

### Retrieval
```python
def retrieve(query: str, n_results: int = 3) -> list[dict]:
    collection = chroma_client.get_collection("dev_knowledge")
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances"]
    )
    chunks = []
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        similarity = round(1 - dist, 3)
        if similarity > 0.3:  # threshold — ignore low quality matches
            chunks.append({"text": doc, "similarity": similarity})
    return chunks
```

### Generation
```python
def generate(query: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)

    system = """You are a precise technical assistant.
Answer questions strictly based on the provided context.
If the answer is not in the context, say "I don't have information about that."
Never make up information."""

    user_message = f"""Context:
{context}

Question: {query}"""

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user_message)
    ]
    response = llm.invoke(messages)
    return response.content.strip()
```

### Full RAG query
```python
def rag_query(query: str) -> dict:
    chunks = retrieve(query)
    if not chunks:
        return {"answer": "I don't have information about that.", "chunks_used": 0}
    answer = generate(query, [c["text"] for c in chunks])
    return {"answer": answer, "chunks_used": len(chunks)}
```

---

## Similarity threshold — why it matters

```python
if similarity > 0.3:  # only use high quality matches
    chunks.append(doc)
```

Without threshold → low quality chunks reach the LLM → hallucination risk.
With threshold → only relevant chunks pass → grounded answers.

### From your results today
```
"Git merge vs rebase?" → below threshold → "I don't have information" ✅
```
Git IS in the knowledge base but merge vs rebase specifically isn't covered.
The threshold correctly rejected the low-quality match.

---

## History management in RAG chatbot

Context injection creates a problem — if you store the full context message in history,
history grows massive very fast (context + query every turn).

**The fix — replace context message with clean query in history:**
```python
# Add context-enriched message for LLM
history.append(HumanMessage(content=context_message))  # context + query
messages = [SystemMessage(content=SYSTEM)] + history[-6:]
response = llm.invoke(messages)

# Replace with clean version in history — context not stored
history[-1] = HumanMessage(content=user_query)  # ← clean query only
history.append(AIMessage(content=answer))
```

This keeps history small while still giving the LLM full context per turn.

---

## The most critical insight from today

**Retrieval is the most impactful step — not generation.**

```
Bad retrieval → wrong chunks → LLM generates wrong answer (no recovery possible)
Good retrieval → right chunks → LLM generates correct answer
```

The LLM can only work with what retrieval gives it.
If retrieval fails, generation cannot fix it.

**Rule: garbage in → garbage out. Good retrieval → good answers.**

This is why chunking strategy (Session 9) and embedding quality (Session 8)
matter more than prompt engineering in RAG systems.

---

## Sources display — production trust feature

```python
if show_sources:
    print(f"[Sources used: {len(chunks)}]")
    for c in chunks:
        print(f"  [{c['similarity']}] {c['text'][:80]}...")
```

Showing sources builds user trust — they can verify where the answer came from.
Every production RAG system should have this capability.

---

## RAG results from today

```
"What is FastAPI and what makes it special?" → answered correctly ✅
"How does Redis store data?"                 → answered correctly ✅
"What is a Dockerfile?"                      → answered correctly ✅
"What is the capital of France?"             → rejected correctly ✅
"Git merge vs rebase?"                       → rejected correctly ✅
"What is Kubernetes?" + sources              → answered with sources ✅
```

---

## What you built today

| File                 | What it does                    | Key concept       |
|----------------------|---------------------------------|-------------------|
| `01_indexing.py`     | Load → chunk → embed → store    | Indexing phase    |
| `02_retrieval.py`    | Query → embed → search → chunks | Retrieval phase   |
| `03_generation.py`   | Chunks + query → LLM → answer   | Generation phase  |
| `04_mini_project.py` | Full interactive RAG chatbot    | Complete pipeline |

---

## What's coming — Session 11

Reranking + hybrid search.
Today's retrieval was pure semantic search.
Session 11 adds keyword search alongside semantic search (hybrid),
and reranking to reorder results by relevance after initial retrieval.
This improves retrieval quality significantly for production systems.

---

*Session 10 complete · Next: Session 11 — Reranking + Hybrid Search*