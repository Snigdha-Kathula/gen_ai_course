import re 
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from dotenv import load_dotenv
from google import genai
import os

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
collection = chromadb.PersistentClient(path="session10/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
def get_embedding(text:str):
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )
    return result.embeddings[0].values


def get_or_create_collection(collection_name:str):
    try:
        return collection.get_collection(collection_name)
    except:
        return collection.create_collection(collection_name)


def index_document(filepath:str, collection_name:str):
    with open(filepath, "r") as f:
        text = f.read()
    splitting = RecursiveCharacterTextSplitter(
        chunk_size =300,
        chunk_overlap = 50,
        separators=["\n\n", "\n", ". ", " ", ""] # ask claude, what does it mean
    )
    chunks = splitting.split_text(text=text)
    collection = get_or_create_collection(collection_name)
    if collection.count() > 0 :
        collection.delete(ids=collection.get()["ids"])
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"chunk_{i}"],
            metadatas=[{"source": filepath, "chunk_id": i}]
        )
    print(f"Indexed {len(chunks)} chunks from {filepath}")
    return len(chunks)


if __name__ == "__main__":
    count = index_document("session10/knowledge_base.txt", "dev_knowledge")
    print(f"Total chunks indexed: {count}")
