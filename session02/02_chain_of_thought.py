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

problem = """A store sells an apple for Rs.12 each.
Priya buys 5 apples and paid with a Rs.100 note.
She gets Rs.40 change. Did the cashier make an error?
"""

# without CoT
direct = f"{problem}\nAnswer Yes or No"

# with CoT
CoT = f"""{problem}
Think through the step by step and give the final answer"""

print("==== without cot ====")
print(call_gemini(prompt=problem))
print("\n ===== with CoT =====")
print(call_gemini(prompt=problem))


