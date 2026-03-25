# Session 11 — Reranking + Hybrid Search
> Snigdhaa's GenAI Course · Save as `session11/concepts.md`

---

## The one line summary
Pure semantic search has blind spots. Hybrid search + reranking fills them.

---

## Why pure semantic search fails

```
Failure 1 — keyword mismatch:
Query: "FastAPI async await"
Semantic finds: loosely related Python chunks
BM25 finds: exact "async and await keywords" chunk immediately

Failure 2 — semantic drift:
Query: "What does BM25 stand for"
Semantic finds: loosely related chunks (score 0.098)
No good answer exists — but score isn't low enough to reject
```

---

## BM25 — keyword search algorithm

BM25 (Best Match 25) scores documents by:
- Term frequency — how often the query term appears in the document
- Inverse document frequency — how rare the term is across all documents
- Document length normalisation — penalises very long documents

```python
from rank_bm25 import BM25Okapi
import re

def tokenise(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())

bm25 = BM25Okapi([tokenise(chunk) for chunk in chunks])

def keyword_search(query: str, n_results: int = 5) -> list[dict]:
    scores = bm25.get_scores(tokenise(query))
    top_indices = sorted(range(len(scores)),
                        key=lambda i: scores[i], reverse=True)[:n_results]
    return [{"text": chunks[i], "score": scores[i]} for i in top_indices
            if scores[i] > 0]
```

**BM25 strength:** Exact keyword match → very high score
**BM25 weakness:** No semantic understanding — "API" matches any chunk containing "API"

---

## Hybrid search — combining both signals

```python
def hybrid_search(query: str, semantic_weight: float = 0.7,
                  keyword_weight: float = 0.3) -> list[dict]:
    semantic_scores = semantic_search(query)   # embedding similarity
    keyword_scores = keyword_search(query)     # BM25 scores (normalised 0-1)

    all_indices = set(semantic_scores.keys()) | set(keyword_scores.keys())
    combined = {
        i: semantic_scores.get(i, 0) * 0.7 + keyword_scores.get(i, 0) * 0.3
        for i in all_indices
    }
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)
```

### Score improvement from today
```
Query: "FastAPI async await keywords"
  Semantic alone: 0.631
  Hybrid:         0.742  ← +17% boost from keyword signal

Query: "Redis memory data structures"
  Semantic alone: 0.597
  Hybrid:         0.718  ← +20% boost
```

### Weights — 70/30 is a good starting point
```
semantic_weight = 0.7  # semantic usually more reliable
keyword_weight  = 0.3  # keyword adds precision for exact terms
```
Tune based on your use case. Technical docs with specific terminology → increase keyword weight.
Conversational queries → increase semantic weight.

---

## Reranking — second pass precision

```
Step 1: Retrieve top 10 chunks fast (semantic or hybrid)
Step 2: Score each of the 10 with LLM — "how relevant is this to the query?"
Step 3: Return top 3 by rerank score
```

### Why two steps?
Searching all chunks with LLM scoring = too slow (N × LLM calls).
Retrieve a candidate set fast → rerank small set precisely.

### Reranking result from today
```
Query: "how does FastAPI handle asynchronous operations"

Before reranking (original order):
  0: Python is a high-level programming language...  ← wrong at top
  1: Python is widely used in web development...
  2: FastAPI is a modern Python web framework...

After reranking:
  [10/10] async and await keywords chunk            ← moved to top ✅
  [2/10]  FastAPI intro chunk
  [1/10]  Redis use cases chunk
```

LLM correctly identified the most relevant chunk and gave a precise reason.

### Reranking code pattern
```python
def rerank(query: str, candidate_chunks: list[str], top_k: int = 3) -> list[dict]:
    scored = []
    for chunk in candidate_chunks:
        prompt = f"""Score relevance of this text to the query (0-10).
Query: {query}
Text: {chunk}
Respond ONLY with JSON: {{"score": <0-10>, "reason": "<one sentence>"}}"""
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip().replace("```json","").replace("```","")
        result = json.loads(raw)
        scored.append({"text": chunk, "score": result["score"]})
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]
```

---

## Known weakness — hybrid score inflation

```
Query: "What does BM25 stand for"  (not in knowledge base)
  Semantic score: 0.098  (loose meaning match)
  Hybrid score:   0.369  (inflated — higher than deserved)
```

**Why:** Common words like "does" and "for" get tiny BM25 scores across many chunks.
When normalised and combined with semantic score, they inflate the final hybrid score.

**Production fix:** Apply minimum threshold on each signal independently:
```python
# Only include BM25 score if it's meaningful
if bm25_score / max_score > 0.1:   # minimum keyword threshold
    keyword_scores[i] = normalised_score
```

---

## Retrieval pipeline comparison

| Method          | Strength            | Weakness                           |
|-----------------|---------------------|------------------------------------|
| Pure semantic   | Understands meaning | Misses exact terms                 |
| Pure BM25       | Exact keyword match | No semantic understanding          |
| Hybrid          | Best of both        | Score inflation from noise         |
| Hybrid + rerank | Most accurate       | Slower — N LLM calls for reranking |

**For production:** Hybrid retrieval + reranking on top 5-10 candidates.

---

## What you built today

| File                  | What it does                   | Key concept                |
|-----------------------|--------------------------------|----------------------------|
| `01_bm25_search.py`   | Keyword search with BM25       | Term frequency scoring     |
| `02_hybrid_search.py` | Semantic + keyword combined    | Weighted score fusion      |
| `03_reranking.py`     | LLM-based relevance scoring    | Second pass precision      |
| `04_mini_project.py`  | Full RAG with hybrid retrieval | Production-grade retrieval |

---

## What's coming — Session 12

RAG evaluation — hallucination detection + faithfulness scoring.
You'll build a pipeline that automatically measures how good your RAG system is.
Key metrics: faithfulness, relevancy, hallucination rate.

---

*Session 11 complete · Next: Session 12 — RAG Evaluation*