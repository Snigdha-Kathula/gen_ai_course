import os
import re
from dotenv import load_dotenv
from google import genai
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_embedding(text: str) -> list[float]:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values

# Vector DB setup
chroma_client = chromadb.PersistentClient(path="session09/chroma_db")

def get_or_create_collection(name: str):
    try:
        return chroma_client.get_collection(name)
    except:
        return chroma_client.create_collection(name)

def ingest_document(filepath: str, collection_name: str, chunk_size: int = 300, chunk_overlap: int = 50) -> int:
    # Load
    with open(filepath, "r") as f:
        text = f.read()
    print(f"Loaded: {filepath} ({len(text)} chars)")

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    print(f"Chunked into: {len(chunks)} pieces")

    # Embed + store
    collection = get_or_create_collection(collection_name)

    # Clear existing docs if any
    existing = collection.count()
    if existing > 0:
        collection.delete(ids=collection.get()["ids"])
        print(f"Cleared {existing} existing documents")

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

    print(f"Stored {len(chunks)} chunks in '{collection_name}'\n")
    return len(chunks)

def search(collection_name: str, query: str, n_results: int = 3) -> list[dict]:
    collection = get_or_create_collection(collection_name)
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        output.append({
            "text": doc,
            "chunk_index": meta["chunk_index"],
            "similarity": round(1 - dist, 3)
        })
    return output

def main():
    print("=== Document Ingestion Pipeline ===\n")

    # Ingest document
    ingest_document(
        filepath="session09/sample_doc.txt",
        collection_name="ai_knowledge",
        chunk_size=300,
        chunk_overlap=50
    )

    # Test retrieval
    print("=== Testing Retrieval ===\n")
    queries = [
        "how do machines learn from data",
        "what are neural networks",
        "how does NLP work",
        "image recognition and computer vision",
    ]

    for query in queries:
        print(f"Query: '{query}'")
        results = search("ai_knowledge", query)
        for r in results[:2]:
            print(f"  [{r['similarity']}] chunk_{r['chunk_index']}: {r['text'][:100]}...")
        print()

if __name__ == "__main__":
    main()