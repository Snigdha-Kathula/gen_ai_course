# Session 09 — Chunking Strategies + Document Loading
> Snigdhaa's GenAI Course · Save as `session09/concepts.md`

---

## The one line summary
How you split documents determines retrieval quality. One clear idea per chunk — not one topic that mentions many ideas.

---

## Why chunking matters

```
Too large  → too much noise, relevant info diluted, dominant topic drowns out details
Too small  → loses context, incomplete answers
Just right → one clear idea per chunk, retrievable
```

Real documents are pages long. You can't embed an entire document as one vector.
Chunking splits it into meaningful pieces before embedding.

---

## 3 chunking strategies compared

| Strategy            | How it works                            | Chunks | Problem                     |
|---------------------|---------------------------------------- |--------|-----------------------------|
| Fixed size          | Split every N characters                | 14     | Cuts mid-sentence, mid-word |
| Sentence-aware      | Split at sentence boundaries            | 15     | Overlap adds duplicates     |
| LangChain recursive | Split on paragraphs → sentences → words | 10     | Best — cleanest boundaries  |

### Fixed size — what goes wrong
```
Chunk 2: "g, and natural language processing.
AI systems can perform..."
```
Starts mid-word. No RAG system can retrieve this correctly.

### LangChain recursive — why it's best
```
Chunk 3: "Machine learning is a subset of artificial intelligence..."
```
Clean, complete thought. One idea. Retrievable.

---

## LangChain RecursiveCharacterTextSplitter

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,       # max characters per chunk
    chunk_overlap=50,     # overlap between chunks for context continuity
    separators=["\n\n", "\n", ". ", " ", ""]
    # tries paragraph first, then sentence, then word, then character
)

chunks = splitter.split_text(text)
```

**Why separators order matters:**
Tries `\n\n` (paragraph) first. If chunk still too large, tries `\n` (line).
Then `. ` (sentence). Then ` ` (word). Then `""` (character — last resort).
This hierarchy ensures the most natural split possible.

---

## The golden rule of chunking

```
One clear idea per chunk.
Not one topic that mentions many ideas.
```

### Why this matters — from your live results today

Query: "what are neural networks" → score only 0.335

The retrieved chunk was about deep learning — it *mentioned* neural networks
but its dominant topic was deep learning. The embedding captured the
dominant meaning of the chunk, not every keyword inside it.

**Embedding represents dominant meaning, not keyword presence.**

If the user asks about neural networks, the ideal chunk should be
ABOUT neural networks — not a deep learning chunk that mentions them in passing.

---

## Document ingestion pipeline — the pattern

```
Load document (txt, pdf, etc.)
         ↓
Chunk with RecursiveCharacterTextSplitter
         ↓
For each chunk → get_embedding(chunk)
         ↓
collection.add(document, embedding, id, metadata)
         ↓
Ready for retrieval
```

### Full ingestion function
```python
def ingest_document(filepath: str, collection_name: str,
                    chunk_size: int = 300, chunk_overlap: int = 50) -> int:
    # Load
    with open(filepath, "r") as f:
        text = f.read()

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)

    # Embed + store
    collection = get_or_create_collection(collection_name)
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"chunk_{i}"],
            metadatas=[{
                "source": filepath,
                "chunk_index": i,
                "chunk_size": len(chunk)
            }]
        )
    return len(chunks)
```

---

## Chunk overlap — why it matters

```python
chunk_overlap=50  # last 50 chars of chunk N appear at start of chunk N+1
```

Without overlap — a sentence split across two chunks loses context at the boundary.
With overlap — the boundary region appears in both chunks, preserving context.

```
Chunk 3: "...enables systems to learn from data. Instead of being explicitly programmed..."
Chunk 4: "...improve over time. Common machine learning algorithms..."
                   ↑ overlap region connects the two
```

---

## Retrieval quality from today

```
"how do machines learn from data"        → ML chunk     [0.507] ✅
"what are neural networks"               → DL chunk     [0.335] ← lower (dominant topic mismatch)
"how does NLP work"                      → NLP chunk    [0.392] ✅
"image recognition and computer vision"  → CV chunk     [0.445] ✅
```

Lower score on neural networks = chunk design issue, not embedding issue.
Fix: create a dedicated chunk ABOUT neural networks.

---

## Metadata in chunks

Always store metadata with every chunk:
```python
metadatas=[{
    "source": filepath,      # which document
    "chunk_index": i,        # position in document
    "chunk_size": len(chunk) # size of this chunk
}]
```

In production RAG you'll also add:
- `"page_number"` for PDFs
- `"section"` for structured documents
- `"date"` for time-sensitive content

---

## What you built today

| File                       | What it does                     | Key concept                   |
|----------------------------|----------------------------------|-------------------------------|
| `01_fixed_chunking.py`     | Split by character count         | Baseline — shows the problem  |
| `02_sentence_chunking.py`  | Split at sentence boundaries     | Better but adds duplicates    |
| `03_langchain_splitter.py` | Recursive splitting + comparison | Best approach                 |
| `04_mini_project.py`       | Full ingestion pipeline          | Load → chunk → embed → store  |

---

## What's coming — Session 10

Full RAG pipeline — retrieve + generate.
You'll connect the ingestion pipeline from today to an LLM.
User asks a question → retrieve relevant chunks → feed to LLM → answer.
This is the complete RAG loop.

---

*Session 09 complete · Next: Session 10 — Full RAG Pipeline*