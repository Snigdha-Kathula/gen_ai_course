import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def count_tokens(messages: list) -> int:
    result = client.models.count_tokens(
        model="gemini-2.0-flash",
        contents=messages
    )
    return result.total_tokens

def chat_with_tracking(history: list, user_message: str) -> tuple[str, int]:
    history.append({
        "role": "user",
        "parts": [{"text": user_message}]
    })

    tokens_before = count_tokens(history)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=history,
        config=types.GenerateContentConfig(temperature=0.7)
    )

    reply = response.text.strip()
    history.append({
        "role": "model",
        "parts": [{"text": reply}]
    })

    return reply, tokens_before

# Simulate a long conversation
history = []
print("=== Simulating growing context ===\n")

questions = [
    "My name is Snigdhaa and I work at InCred as a backend developer.",
    "I am building a loan management system using Python and PostgreSQL.",
    "The system handles 10,000 transactions per day.",
    "We use Redis for caching and RabbitMQ for async processing.",
    "What is my name?",
    "What database am I using?",
    "What is my message queue system?",
    "Summarise everything I have told you so far.",
]

for q in questions:
    reply, tokens = chat_with_tracking(history, q)
    print(f"You: {q}")
    print(f"Bot: {reply}")
    print(f"[Tokens in this request: {tokens}]")
    print()