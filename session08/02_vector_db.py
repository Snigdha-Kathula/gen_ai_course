import os
from google import genai
from dotenv import load_dotenv
import chromadb

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_embeddings(sentence:str):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=sentence
    )
    return response.embeddings[0].values

chroma_client = chromadb.Client()
collection = chroma_client.create_collection("fin-docs")

# Documents to store
documents = [
    "The Dark Knight is a 2008 superhero film directed by Christopher Nolan featuring Batman.",
    "Inception is a 2010 sci-fi thriller about a thief who enters people's dreams.",
    "Interstellar follows astronauts travelling through a wormhole near Saturn to find a new home for humanity.",
    "The Shawshank Redemption is a 1994 drama about hope and friendship inside a prison.",
    "Forrest Gump is a 1994 film about a man with low IQ who witnesses major historical events.",
    "The Matrix is a 1999 sci-fi film where humanity is trapped in a simulated reality.",
    "Parasite is a 2019 South Korean film about class inequality directed by Bong Joon-ho.",
    "Schindler's List is a 1993 historical drama about Oskar Schindler saving Jewish lives in World War II.",
    "Pulp Fiction is a 1994 crime film by Quentin Tarantino with interconnected storylines.",
    "The Godfather is a 1972 crime drama about the powerful Corleone mafia family.",
]

print("=== Storing movies in vector DB ===")
for i, doc in enumerate(documents):
    embedding = get_embeddings(doc)
    collection.add(
        documents=[doc],
        embeddings=[embedding],
        ids = [f"Movie: {i}"]
    )
    print(f"Stored: {doc[:60]}...")

print(f"\nTotal documents stored: {collection.count()}\n")



# semantic search
def search(query:str, n_results:int =3):
    query_embedding = get_embeddings(query)
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return result["documents"][0]
    




print("=== Semantic search ===\n")
queries = [
    "films about dreams and the subconscious",
    "movies about war and survival",
    "crime and gangster films",
    "space exploration movies",
]

for q in queries:
    print(f"QUERY: {q}")
    results = search(q)
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result[:80]}...")
    print()

