import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

# configure SDK with your key
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# first call
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="what is the capital of Telangana?"
)

print(response.text)
