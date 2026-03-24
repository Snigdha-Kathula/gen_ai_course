import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import chromadb
from google import genai
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()
llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash", temperature=0.0)
collection = chromadb.PersistentClient(path="session10/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_embedding(text:str):
    response = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
    return response.embeddings[0].values

def retrieve(query:str, collection_name: str):
    embedding = get_embedding(query)
    get_collection = collection.get_collection(collection_name)
    result = get_collection.query(
        query_embeddings=[embedding],
        n_results=3,
        include=["documents", "distances"]
    )
    chunks = []
    for doc, dist in zip(result["documents"][0],result["distances"][0]):
        similarity = round(1-dist, 3)
        if similarity > 0.3:
            chunks.append({"text": doc, "similarity":similarity})
    return chunks


history =[ ]
SYSTEM = """You are a helpful technical assistant.
Answer questions strictly based on the provided context.
If the answer is not in the context say "I don't have information about that."
Be concise and precise. Max 4 sentences."""


def chat(query:str, show_sources: bool = False):
    retrieve_result = retrieve(query, "dev_knowledge")
    if not retrieve_result:
        return  "I don't have information about that in my knowledge base"
    
    context = "\n\n".join([c["text"] for c in retrieve_result])

    user_msg = f"""Context from knowledge base:
{context}

Question: {query}"""
    user_msg = f"""Context:
{context}

Question: {query}
"""
    history.append(HumanMessage(content=user_msg))
    messages = [SystemMessage(content=SYSTEM)]+ history[-6:]
    result = llm.invoke(messages)
    answer = result.content.strip()
    history[-1] = HumanMessage(content=user_msg)
    history.append(AIMessage(content=answer))

    if show_sources:
        print(f"\n[Sources used: {len(retrieve_result)}]")
        for c in retrieve_result:
            print(f"  [{c['similarity']}] {c['text'][:80]}...")


    return answer


def main():
    print("=== RAG Knowledge Assistant ===")
    print("Ask questions about Python, FastAPI, PostgreSQL, Redis, Docker, Kubernetes, Git, REST APIs")
    print("Commands: 'sources' to toggle source display | 'quit' to exit\n")

    show_sources = False

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        elif user_input.lower() == "sources":
            show_sources = not show_sources
            print(f"[Sources display: {'ON' if show_sources else 'OFF'}]\n")
            continue
        answer = chat(user_input, show_sources=show_sources)
        print(f"\nBot: {answer}\n")




if __name__ == "__main__":

    main()
