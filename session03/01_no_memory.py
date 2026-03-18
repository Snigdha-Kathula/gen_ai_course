import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
def chat_no_memory(prompt: str):
    reponse = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7
        )
    )
    return reponse.text

print(chat_no_memory("hi my name is Snigdha! I am a backend developer"))
print(chat_no_memory("What is my name"))
print(chat_no_memory("what did i told you in first message"))
