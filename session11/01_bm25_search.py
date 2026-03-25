import os
import re
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("session10/knowledge_base.txt", "r") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_text(text=text)

# Tokenise for BM25
def tokenise(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())

tokenised_chunks = [tokenise(chunk) for chunk in chunks]
bm25 = BM25Okapi(tokenised_chunks)

def keyword_search(query: str, n_results: int = 5) -> list[dict]:
    tokenised_query = tokenise(query)
    scores = bm25.get_scores(tokenised_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
    return [
        {"text": chunks[i], "score": round(scores[i], 3), "index": i}
        for i in top_indices
        if scores[i] > 0
    ]

# Test
queries = [
    "FastAPI async await",
    "Redis memory cache",
    "Kubernetes pods deployments",
    "What does API stand for",
]
print("=== BM25 Keyword Search ===\n")
for query in queries:
    print(f"Query: '{query}'")
    results = keyword_search(query)
    for r in results[:3]:
        print(f"  [{r['score']}] {r['text'][:80]}...")
    print()