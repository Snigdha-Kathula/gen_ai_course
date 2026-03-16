import os
from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def gemini_api_call(prompt: str, temperature: float)-> str: 
    response = client.models.generate_content(
        model = "gemini-2.0-flash",
        contents = prompt,
        config = types.GenerateContentConfig(
            temperature = temperature,
            max_output_tokens = 200
        )
    )
    return response.text


prompt = "Write a one-line tagline for a coffee shop."

print("Temperature 0.0 (Deterministic..)")
for _ in range(3):
    print("-> ", gemini_api_call(prompt, temperature=0.0))

print("\nTemperature 1.0 (Creative..)")
for _ in range(3):
    print("-> ", gemini_api_call(prompt, temperature=1.0))

print("\nTemperature 2.0 (Choatic..)")
for _ in range(3):
    print("-> ", gemini_api_call(prompt, temperature=2.0))


