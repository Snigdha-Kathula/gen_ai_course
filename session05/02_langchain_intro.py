import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash",
    temperature =0.7
)
# response = llm.invoke("What is langchain?. explain it in short")
# print(response.content)
messages = [
    SystemMessage("You are a helpful assistant. Be concise - max 2 sentences")
]

def chat(msg):
    messages.append(HumanMessage(content=msg))
    print(f"You: {msg}")
    response = llm.invoke(messages)
    messages.append(response.content)
    return response.content
print("Bot:", chat("Hi! My name is Snigdhaa and I am a backend developer."))
print("Bot:", chat("What is my name?"))
print("Bot:", chat("What do I do for work?"))
print("Bot:", chat("Give me one Python tip for backend developers."))
