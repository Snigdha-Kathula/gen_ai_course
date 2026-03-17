import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model = "gemini-2.0-flash",
    contents="what is streaming in LLms Apis?explain it in short"
)
print(response.text)
print("\n")
for chunk in client.models.generate_content_stream(
    model = "gemini-2.0-flash",
    contents="what is streaming in LLms Apis?explain it in short"
):
    print(chunk.text, end="", flush=True)