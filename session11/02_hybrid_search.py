import os
import re
from dotenv import load_dotenv
from google import genai
import chromadb
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Load and chunk
with open("session10/knowledge_base.txt", "r") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_text(text)

# BM25 setup
def tokenise(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())

bm25 = BM25Okapi([tokenise(c) for c in chunks])

# Chroma setup
chroma_client = chromadb.PersistentClient(path="session10/chroma_db")
collection = chroma_client.get_collection("dev_knowledge")

def get_embedding(text: str) -> list[float]:
    result = client.models.embed_content(
        model=os.getenv("EMBEDDING_MODEL"),
        contents=text
    )
    return result.embeddings[0].values

def semantic_search(query: str, n_results: int = 5) -> dict[int, float]:
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances"]
    )
    scores = {}
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        # Find chunk index by matching text
        for i, chunk in enumerate(chunks):
            if chunk[:50] == doc[:50]:
                scores[i] = round(1 - dist, 3)
                break
    return scores

def keyword_search(query: str, n_results: int = 5) -> dict[int, float]:
    tokenised = tokenise(query)
    bm25_scores = bm25.get_scores(tokenised)
    # Normalise BM25 scores to 0-1
    max_score = max(bm25_scores) if max(bm25_scores) > 0 else 1
    top_indices = sorted(range(len(bm25_scores)),
                        key=lambda i: bm25_scores[i], reverse=True)[:n_results]
    return {i: round(bm25_scores[i] / max_score, 3)
            for i in top_indices if bm25_scores[i] > 0}

def hybrid_search(query: str, semantic_weight: float = 0.7,
                  keyword_weight: float = 0.3, n_results: int = 3) -> list[dict]:
    semantic_scores = semantic_search(query)
    keyword_scores = keyword_search(query)

    # Combine scores
    all_indices = set(semantic_scores.keys()) | set(keyword_scores.keys())
    combined = {}
    for i in all_indices:
        sem = semantic_scores.get(i, 0)
        kw = keyword_scores.get(i, 0)
        combined[i] = round(sem * semantic_weight + kw * keyword_weight, 3)

    # Sort by combined score
    top = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:n_results]
    return [{"text": chunks[i], "score": score, "index": i} for i, score in top]

# Compare semantic vs hybrid
queries = [
    "FastAPI async await keywords",
    "Redis memory data structures",
    "What does BM25 stand for",  # not in KB — should score low
]

print("=== Semantic vs Hybrid Search ===\n")
for query in queries:
    print(f"Query: '{query}'")

    semantic = semantic_search(query)
    top_semantic = sorted(semantic.items(), key=lambda x: x[1], reverse=True)[:1]

    hybrid = hybrid_search(query)

    print(f"  Semantic top: [{top_semantic[0][1]}] {chunks[top_semantic[0][0]][:80]}...")
    print(f"  Hybrid top:   [{hybrid[0]['score']}] {hybrid[0]['text'][:80]}...")
    print()