import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

conversation_history = []
def chat_no_memory(user_msg:str, max_history: int=10)-> str:
    conversation_history.append(
        {
            "role": "user",
            "parts": [{"text": user_msg}]
        }
    )

    trimmed_history=conversation_history[-max_history:]
    
    reponse = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=trimmed_history,
        config=types.GenerateContentConfig(
            system_instruction="you ar an helpful assistant, you will remember the user tells you",
            temperature=0.7
        )
    )
    assistant_reply = reponse.text.strip()
    conversation_history.append(
        {
            "role": "user",
            "parts": [{"text": assistant_reply}]
        }
    )
    return assistant_reply


print("=== Chatbot WITH buffer memory ===\n")
print("Bot:", chat_no_memory("Hi! My name is Snigdhaa. I am a backend developer."))
print("\n")
print("Bot:", chat_no_memory("What is my name?"))
print("\n")
print("Bot:", chat_no_memory("What do I do for work?"))
print("\n")
print("Bot:", chat_no_memory("Give me a Python tip relevant to my job."))
print("\n")







# print(chat_no_memory("hi my name is Snigdha! I am a backend developer\n"))
# print(chat_no_memory("What is my name\n"))
# # print(chat_no_memory("what did i told you in first message\n"))
# print(chat_no_memory("I works on java, springboot and learning gen ai"))
# print(chat_no_memory("i will go to the office at 11:00 clock"))
# print(chat_no_memory("i works in a product based company"))
# print(chat_no_memory("my company has 4 values"))
# print(chat_no_memory("my collegue name is nandita"))
# print(chat_no_memory("my manager name is steve jobs"))
# print(chat_no_memory("My phone lost display"))
