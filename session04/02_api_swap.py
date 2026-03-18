import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

PROMPT = "Explain what a REST API is in exactly 2 sentences."

# ── Provider 1: Gemini ──────────────────────────────────────
def call_gemini(prompt: str) -> str:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0)
    )
    return response.text.strip()

# ── Provider 2: OpenAI ──────────────────────────────────────
def call_openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

# ── Provider 3: Anthropic ───────────────────────────────────
def call_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()

# ── Run all three ───────────────────────────────────────────
print(f"Prompt: {PROMPT}\n")

print("=== Gemini 2.0 Flash ===")
print(call_gemini(PROMPT))

# Uncomment when you have API keys:
# print("\n=== OpenAI GPT-4o Mini ===")
# print(call_openai(PROMPT))

# print("\n=== Anthropic Claude Sonnet ===")
# print(call_anthropic(PROMPT))

'''

**The key observation** — look at the 3 functions side by side:

|              |       Gemini         |   OpenAI                              |       Anthropic            |
|--------------|----------------------|---------------------------------------|----------------------------|
| Client       | `genai.Client()`     | `OpenAI()`                            | `anthropic.Anthropic()`    |
| Method       | `generate_content()` | `chat.completions.create()`           | `messages.create()`        |
| History role |    `"model"`.        |              `"assistant"`            | `"assistant"`              |
| Response     | `response.text`      | `response.choices[0].message.content` | `response.content[0].text` |

Different SDKs, same concept. Once you understand one, you understand all of them.

Run with Gemini for now — the OpenAI and Anthropic calls are commented out since you don't have those keys. The important thing is reading the code and understanding the pattern.

---

### Step 3 — Phase 1 Capstone: Smart Q&A bot (45 min)

`session04/03_capstone_qna.py`

This ties together everything from Phase 1 — API calls, prompt engineering, system prompts, memory, streaming. You give it any text file and it answers questions about it.

First create a sample text file `session04/sample.txt`:
```
InCred Financial Services is a technology-led lending company.
It offers personal loans, business loans, and education loans.
The company uses advanced data analytics and machine learning to assess creditworthiness.
InCred was founded in 2016 by Bhupinder Singh.
The technology stack includes Python for backend services, PostgreSQL for databases,
Redis for caching, and RabbitMQ for message queuing.
The engineering team follows microservices architecture.
Deployments are managed using Docker and Kubernetes.
The platform processes over 10,000 loan applications per day.
Customer support is available 24/7 through the web portal and mobile app. '''