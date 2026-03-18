import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

history = []
SYSTEM="""You are a helpful AI assistant for a software company.
You help developers with coding questions, architecture decisions, and best practices.
You are concise and practical. Max 5 sentences per response."""

total_tokens=0
def chat(user_msg:str)-> None:
    global total_tokens
    history.append({
        "role": "user",
        "parts":[{"text":user_msg}]
    })

    # Trim to last 20 messages
    trimmed = history[-20:]

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=trimmed,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            temperature=0.7
        )
    )

    reply = response.text.strip()
    tokens_used = response.usage_metadata.total_token_count
    total_tokens += tokens_used
    
    print("\n")
    history.append({
        "role": "user",
        "parts":[{"text":reply}]
    })
    print(f"\nBot: {reply}")
    print(f"[Turn tokens: {tokens_used} | Total tokens: {total_tokens}]\n")

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

