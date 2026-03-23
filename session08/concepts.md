# Session 08 — Embeddings + Vector Databases
> Snigdhaa's GenAI Course · Save as `session08/concepts.md`

---

## The one line summary
Embeddings convert text into numbers that capture meaning. Vector databases find the closest meanings fast.

---

## What are embeddings

An embedding converts text into a list of floating point numbers — a vector.
Each number captures some aspect of the text's meaning.

```python
"I love Python"   → [-0.009, -0.002, 0.009, -0.089, ...] # 3072 numbers
"Python is great" → [-0.008, -0.003, 0.011, -0.091, ...] # very similar
"I love mangoes"  → [ 0.067,  0.123, -0.034, 0.021, ...] # different
```

**Key insight:** Similar meaning = similar numbers = close in vector space.

---

## Generating embeddings with Gemini

```python
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_embedding(text: str) -> list[float]:
    result = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    return result.embeddings[0].values

embedding = get_embedding("I love Python")
print(len(embedding))    # 3072 dimensions
print(embedding[:5])     # first 5 values
```

**Model:** `text-embedding-004` — Google's embedding model
**Dimensions:** 3072 — each text becomes 3072 numbers

---

## Cosine similarity — how closeness is measured

```python
def cosine_similarity(vec_a: list, vec_b: list) -> float:
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = sum(a ** 2 for a in vec_a) ** 0.5
    magnitude_b = sum(b ** 2 for b in vec_b) ** 0.5
    return dot_product / (magnitude_a * magnitude_b)
```

| Score    | Meaning            |
|----------|--------------------|
| 1.000    | Identical meaning  |
| 0.8+     | Very similar       |
| 0.5–0.8  | Somewhat related   |
| 0.0–0.5  | Different topic    |

### Your live results today
```
Base: "I love programming in Python"

0.868 → "Python is my favourite language"    (very similar ✅)
0.643 → "Machine learning is fascinating"    (interest/enthusiasm context)
0.623 → "I enjoy eating mangoes"             (love/enjoy similar)
0.552 → "I had mango juice today"            (lower — no emotional word)
0.523 → "Deep learning is subset of ML"      (lowest — most different)
```

**Observation:** "Machine learning" scored higher than "mangoes" because
"fascinating" shares emotional/interest context with "love programming."
Embeddings understand conceptual relationships, not just keywords.

---

## Vector databases

A vector database stores embeddings and finds nearest neighbours fast.
Not PostgreSQL. Not Redis. A dedicated engine built for one job — similarity search.

### ChromaDB — what you used today

```python
import chromadb

# In-memory (lost on restart)
client = chromadb.Client()

# Persistent (saved to disk)
client = chromadb.PersistentClient(path="session08/chroma_db")

# Create or get collection
collection = client.create_collection(name="my_docs")
```

### Storing documents

```python
collection.add(
    documents=["The speed of light is 299,792 km/s"],
    embeddings=[get_embedding("The speed of light is 299,792 km/s")],
    ids=["physics_001"],
    metadatas=[{"category": "physics"}]
)
```

### Searching

```python
results = collection.query(
    query_embeddings=[get_embedding("how fast does light travel")],
    n_results=3,
    include=["documents", "metadatas", "distances"]
)
```

**Key:** You never search for keywords. You search for meaning.
"how fast does light travel" finds "speed of light is 299,792 km/s"
even though none of the words match.

---

## In-memory vs persistent

|                  | In-memory           | Persistent                            |
|------------------|---------------------|---------------------------------------|
| Setup            | `chromadb.Client()` | `chromadb.PersistentClient(path=...)` |
| Survives restart | ❌                   | ✅                                     |
| Use case         | Testing, dev        | Production                            |

### Get or create pattern — always use this for persistent

```python
def get_or_create_collection(name: str):
    try:
        return chroma_client.get_collection(name)   # exists → get it
    except:
        return chroma_client.create_collection(name) # doesn't exist → create
```

Without this — second run crashes because collection already exists.

---

## The retrieval quality lesson

From your "planets and space exploration" query — scores were low (0.154 max).
Your knowledge base had no document about planets specifically.

```
Query: "planets and space exploration"
Best result: 0.154 — Mars has the largest volcano...
```

**Retrieval is only as good as what's stored.**
This is the most important RAG design principle:
- Good documents → good retrieval → good answers
- Missing documents → low similarity scores → wrong answers

In Session 10 you'll learn how to design document collections for maximum retrieval quality.

---

## Metadata — filter + search combined

```python
# Store with metadata
collection.add(
    documents=[text],
    embeddings=[embedding],
    ids=[doc_id],
    metadatas=[{"category": "physics", "source": "textbook"}]
)

# Search results include metadata
for doc, meta, dist in zip(results["documents"][0],
                           results["metadatas"][0],
                           results["distances"][0]):
    similarity = round(1 - dist, 3)
    print(f"[{similarity}] {doc} — {meta['category']}")
```

Metadata lets you filter by category, source, date etc. alongside semantic search.
You'll use this heavily in Phase 3 RAG pipeline.

---

## What you built today

| File                 | What it does                            | Key concept                 |
|----------------------|-----------------------------------------|-----------------------------|
| `01_embeddings.py`   | Generate embeddings + cosine similarity | Text → numbers → distance   |
| `02_vector_db.py`    | Store + search movies semantically      | ChromaDB in-memory          |
| `03_mini_project.py` | Persistent science knowledge base       | ChromaDB on disk + metadata |

---

## What's coming — Session 9

Chunking strategies + document loading.
Real documents (PDFs, text files) are too large to embed whole.
You'll learn how to split them into chunks optimally before storing.

---

*Session 08 complete · Next: Session 09 — Chunking Strategies*