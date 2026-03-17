import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def call(system: str, prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0
        )
    )
    return response.text.strip()

# Strict system prompt — removes the helpfulness conflict
system = """You are a review classification engine.
You ONLY output one word: Positive, Negative, or Mixed.
No explanations. No questions. No extra text.
If the review has both good and bad aspects, output Mixed."""

review = "The battery dies after 3 hours but the screen is gorgeous."

# Zero-shot
zero_shot = f"Classify this review:\n{review}"

# Few-shot
few_shot = f"""Classify each review.

Review: "Absolutely love it, works perfectly!" → Positive
Review: "Broke after one day, waste of money." → Negative
Review: "Great camera but terrible battery life." → Mixed

Review: "{review}" →"""

print("=== Zero-shot ===")
print(call(system, zero_shot))

print("\n=== Few-shot ===")
print(call(system, few_shot))