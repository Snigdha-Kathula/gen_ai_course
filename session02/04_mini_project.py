import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import json

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
def review_code(code: str)-> any:
    prompt = f"""You are an expert Python code reviewer.
        Analyse the following code and respond ONLY in valid JSON.
        No explanation outside the JSON. No markdown backticks.

        JSON format:
        {{
            "rating": <integer 1-10>,
            "issues": [<list of issues found>],
            "improvements": [<list of specific improvements>],
            "verdict": "<one sentence summary>"
        }}

        Code to review:
        ````python
        {code}
        ```"""
    response = client.models.generate_content(
        model = "gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0)
    )

    raw = response.text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()  # clean backticks

    print(raw)
    return json.loads(raw)


# Test 1 — bad code
bad_code = """
def get_user(id):
    import requests
    r = requests.get("http://api.example.com/users/" + id)
    data = r.json()
    return data
"""

# Test 2 — better code
good_code = """
import requests
from typing import Optional

BASE_URL = "http://api.example.com"

def get_user(user_id: int) -> Optional[dict]:
    try:
        response = requests.get(f"{BASE_URL}/users/{user_id}", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching user {user_id}: {e}")
        return None
"""

print("=" * 50)
print("REVIEWING BAD CODE")
print("=" * 50)
review_code(bad_code)

print("\n" + "=" * 50)
print("REVIEWING GOOD CODE")
print("=" * 50)
review_code(good_code)