import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import chromadb
from google import genai
from langchain.messages import HumanMessage, AIMessage, SystemMessage

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
            chunks.append(doc)
    return chunks

def generate(query:str, context:list[str]):
    context = "\n\n".join(context)
    system = """You are a precise technical assistant.
Answer questions strictly based on the provided context.
If the answer is not in the context, say "I don't have information about that in my knowledge base."
Be concise — max 4 sentences.
Never make up information."""
    user_msg = f"""Context:
{context}

Question: {query}
"""
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user_msg)
    ]
    result = llm.invoke(messages)
    return result.content.strip()


def rag_query(query:str):
    retrieve_result = retrieve(query, "dev_knowledge")
    if not retrieve_result:
        return {
            "query": query,
            "answer": "I don't have information about that in my knowledge base",
            "chunks_used": 0
        }
    result = generate(query, context =retrieve_result)
    return {
            "query": query,
            "answer": result,
            "chunks_used": len(retrieve_result),
            "context": retrieve_result
        }





if __name__ == "__main__":
    queries = [
        "What is FastAPI and what makes it special?",
        "How does Redis store data?",
        "What is a Dockerfile?",
        "What is the capital of France?",  # out of context — should be rejected
    ]
    for query in queries:
        print(f"Query: {query}")
        result = rag_query(query)
        print(f"result: {result['answer']}")
        print(f"[chunks: {result['chunks_used']}]\n")
