import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

texts = [
    "The cat sat on the mat",
    "unhappiness automation Snapchat",
    "def hello(): print('Hello')",
    "9472 + 1337 = ?",
    "नमस्ते दुनिया",  # Hello world in Hindi
]

for text in texts :
    result = client.models.count_tokens(
        model="gemini-2.0-flash",
        contents=text
    )
    word = len(text.split())
    print(f"Text: {text}, word:{word}, Tokens:{result.total_tokens}")
    print(f"Ratio: {result.total_tokens/max(word, 1):.2f}")
    print("\n")

# Code > Numbers > English > Common languages
# (most tokens)              (fewest tokens)