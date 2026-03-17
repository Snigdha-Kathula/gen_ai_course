import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
def call_gemini(prompt:str)-> str:
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0)
    )
    return response.text.strip()

review = "The battery dies after 3 hours but the screen is gorgeous."

zero_shot = f"Classify the review as Positive, Negative, Mixed:\n {review}"

few_shot = f"""Classify each review as positive, negative, mixed:
Review: "Absolutely love it, works perfectly!" → Positive
Review: "Broke after one day, waste of money." → Negative
Review: "Great camera but terrible battery life." → Mixed
Review:{review}
"""
print("--- zero shot ----")
print(call_gemini(review))
print("\n")
print("---- few shot ----")
print(call_gemini(review))


