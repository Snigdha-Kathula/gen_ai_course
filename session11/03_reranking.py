import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0
)

with open("session10/knowledge_base.txt", "r") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_text(text)

def rerank(query: str, candidate_chunks: list[str], top_k: int = 3) -> list[dict]:
    """Use LLM to score relevance of each chunk to the query."""
    scored = []
    for i, chunk in enumerate(candidate_chunks):
        prompt = f"""Score how relevant this text is to answering the query.
Query: {query}
Text: {chunk}

Respond with ONLY a JSON object: {{"score": <0-10>, "reason": "<one sentence>"}}
No other text."""
        response = llm.invoke([HumanMessage(content=prompt)])
        try:
            raw = response.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)
            scored.append({
                "text": chunk,
                "rerank_score": result["score"],
                "reason": result["reason"],
                "original_index": i
            })
        except:
            scored.append({
                "text": chunk,
                "rerank_score": 0,
                "reason": "parse error",
                "original_index": i
            })

    return sorted(scored, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

# Test reranking
query = "how does FastAPI handle asynchronous operations"
candidates = chunks[:8]  # first 8 chunks as candidates

print(f"Query: '{query}'\n")
print("=== Before reranking (original order) ===")
for i, c in enumerate(candidates[:3]):
    print(f"  {i}: {c[:80]}...")

print("\n=== After reranking ===")
reranked = rerank(query, candidates)
for r in reranked:
    print(f"  [score: {r['rerank_score']}/10] {r['text'][:80]}...")
    print(f"  Reason: {r['reason']}")
    print()