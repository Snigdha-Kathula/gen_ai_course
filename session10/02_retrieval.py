import os
from google import genai
from dotenv import load_dotenv
import chromadb

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

collection = chromadb.PersistentClient(path="session10/chroma_db")

def get_embedding(text:str):
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )
    return result.embeddings[0].values

def retrieve(query:str, collection_name:str, n_results:int = 3):
    get_collection = collection.get_collection(collection_name)
    get_embedding_query = get_embedding(query)
    results = get_collection.query(
        query_embeddings=[get_embedding_query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    output = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        output.append(
            {
                "text": doc,
                "similarity": round(1-dist, 3),
                "chunk_index": meta["chunk_id"]
            }
        )

    return output


if __name__ == "__main__":
    queries = [
        "how does FastAPI handle async requests",
        "what is Redis used for",
        "how does Kubernetes manage containers",
    ]
    for q in queries:
        print(f"query: {q}")
        response = retrieve(q, "dev_knowledge")
        for r in response:
            print(f"  [{r['similarity']}] {r['text'][:100]}...")
        print()

