import os
import re
from dotenv import load_dotenv
from google import genai
import chromadb
from rank_bm25 import BM25Okapi
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)

# Load and chunk
with open("session10/knowledge_base.txt", "r") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_text(text)

# BM25
def tokenise(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())

bm25 = BM25Okapi([tokenise(c) for c in chunks])

# Chroma
chroma_client = chromadb.PersistentClient(path="session10/chroma_db")
collection = chroma_client.get_collection("dev_knowledge")

def get_embedding(text: str) -> list[float]:
    result = gemini_client.models.embed_content(
        model= os.getenv("EMBEDDING_MODEL"),
        contents=text
    )
    return result.embeddings[0].values

def hybrid_retrieve(query: str, n_results: int = 3) -> list[str]:
    # Semantic
    query_emb = get_embedding(query)
    sem_results = collection.query(
        query_embeddings=[query_emb],
        n_results=5,
        include=["documents", "distances"]
    )
    semantic_scores = {}
    for doc, dist in zip(sem_results["documents"][0], sem_results["distances"][0]):
        for i, chunk in enumerate(chunks):
            if chunk[:50] == doc[:50]:
                semantic_scores[i] = round(1 - dist, 3)
                break

    # Keyword
    bm25_scores = bm25.get_scores(tokenise(query))
    max_score = max(bm25_scores) if max(bm25_scores) > 0 else 1
    keyword_scores = {i: round(bm25_scores[i] / max_score, 3)
                      for i in range(len(chunks)) if bm25_scores[i] > 0}

    # Combine — 70% semantic, 30% keyword
    all_indices = set(semantic_scores.keys()) | set(keyword_scores.keys())
    combined = {
        i: round(semantic_scores.get(i, 0) * 0.7 + keyword_scores.get(i, 0) * 0.3, 3)
        for i in all_indices
    }

    top = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:n_results]
    return [chunks[i] for i, score in top if score > 0.2]

def generate(query: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    messages = [
        SystemMessage(content="""You are a precise technical assistant.
Answer strictly from the context provided.
If not in context say "I don't have information about that."
Max 4 sentences."""),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
    ]
    return llm.invoke(messages).content.strip()

def main():
    print("=== RAG Assistant with Hybrid Search ===")
    print("Powered by semantic + keyword search\n")

    while True:
        query = input("You: ").strip()
        if query.lower() == "quit":
            break
        if not query:
            continue

        chunks_retrieved = hybrid_retrieve(query)
        if not chunks_retrieved:
            print("Bot: I don't have information about that.\n")
            continue

        answer = generate(query, chunks_retrieved)
        print(f"Bot: {answer}\n")

if __name__ == "__main__":
    main()