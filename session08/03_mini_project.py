import os
from google import genai
import chromadb
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_embedding(text:str):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values
     

chroma_client = chromadb.PersistentClient(path="session08/chroma_db")

def get_or_create_collection(name:str):
      try:
           return chroma_client.get_collection(name=name)
      except:
           return chroma_client.create_collection(name=name)

collection = get_or_create_collection("science_facts")  

def add_document(doc_id:int, text:str, metadata:dict = {}):
    embedding = get_embedding(text=text)
    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[doc_id],
        metadatas=[metadata]
    )
    print(f"Embedded and Added: {doc_id}")

def search(query: str, n_results: int = 3) -> list[dict]:
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
            "metadata": meta,
            "similarity": round(1 - dist, 3)
        })
    return output

     
def main():
    print("=== Science Knowledge Base ===")
    print(f"Documents in DB: {collection.count()}\n")
    if collection.count() == 0:
        print("Loading initial documents...\n")
        docs = [
            ("physics_001", "The speed of light in a vacuum is approximately 299,792 km per second.", {"category": "physics"}),
            ("physics_002", "Black holes are regions of spacetime where gravity is so strong that nothing can escape.", {"category": "physics"}),
            ("physics_003", "Einstein's theory of relativity states that energy equals mass times the speed of light squared.", {"category": "physics"}),
            ("biology_001", "DNA is a double helix molecule that carries genetic information in all living organisms.", {"category": "biology"}),
            ("biology_002", "Photosynthesis is the process by which plants convert sunlight into glucose using carbon dioxide and water.", {"category": "biology"}),
            ("biology_003", "The human brain contains approximately 86 billion neurons connected by trillions of synapses.", {"category": "biology"}),
            ("chemistry_001", "Water is a molecule made of two hydrogen atoms bonded to one oxygen atom.", {"category": "chemistry"}),
            ("chemistry_002", "The periodic table organises all known chemical elements by their atomic number.", {"category": "chemistry"}),
            ("space_001", "The Milky Way galaxy contains between 100 and 400 billion stars.", {"category": "space"}),
            ("space_002", "Mars has the largest volcano in the solar system called Olympus Mons.", {"category": "space"}),
            ("space_003", "A light year is the distance light travels in one year — about 9.46 trillion kilometres.", {"category": "space"}),
            ("earth_001", "The Earth's core is made primarily of iron and nickel and reaches temperatures of 5,000 degrees Celsius.", {"category": "earth"}),
        ] 
        for doc_id, text, meta in docs:
            add_document(doc_id, text, meta)
        print(f"\nLoaded {collection.count()} documents\n")
    # Interactive search
    print("Type your query to search. Type 'quit' to exit.\n")
    while True:
        query = input("Search: ").strip()
        if query.lower() == "quit":
            break
        if not query:
            continue

        results = search(query, n_results=3)
        print(f"\nTop {len(results)} results for '{query}':")
        for i, r in enumerate(results, 1):
            print(f"\n  {i}. [{r['similarity']}] {r['text']}")
            print(f"     Category: {r['metadata'].get('category', 'N/A')}")
        print()

if __name__ == "__main__":
    main()