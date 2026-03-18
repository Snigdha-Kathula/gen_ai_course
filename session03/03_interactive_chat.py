import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

history = []
SYSTEM="""You are Pandu, a smart and friendly AI assistant.
You remember everything the user tells you in the conversation.
You are concise — never more than 4 sentences unless asked.
You are talking to a backend developer named Snigdhaa."""

def chat(user_msg:str)-> None:
    history.append({
        "role": "user",
        "parts":[{"text":user_msg}]
    })

    full_string = ""
    for chunk in client.models.generate_content_stream(
        model="gemini-2.0-flash",
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            temperature=0.7
        )
    ):
        print(chunk.text, end="", flush=True)
        full_string += chunk.text
    
    print("\n")
    history.append({
        "role": "user",
        "parts":[{"text":full_string}]
    })

def main():
    print("type quit to exit")
    while True:
        ip = input("You: ").strip()
        if(ip.lower() == "quit"):
            print("Aria: Goodbye Snigdhaa! 👋")
            break
        chat(ip)
if __name__ == "__main__":
    main()

