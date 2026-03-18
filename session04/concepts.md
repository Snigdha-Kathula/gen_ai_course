# Session 04 — Context Windows + Phase 1 Capstone
> Snigdhaa's GenAI Course · Save as `session04/concepts.md`

---

## The one line summary
Context windows limit how much an LLM can read at once. Every provider has a different SDK but the same underlying concept.

---

## Context windows

Every LLM has a maximum token limit per API call — input + output combined.

```
Gemini 2.0 Flash  → 1,048,576 tokens input  (~750,000 words)
GPT-4o            →   128,000 tokens
Claude Sonnet     →   200,000 tokens
```

### What happens when you exceed it
```
Option 1 → API throws an error and rejects the request
Option 2 → Model silently truncates the oldest messages
Option 3 → Response quality degrades — model forgets early context
```

### Token growth observation from today
```
Turn 1 →  23 tokens   (first question)
Turn 2 →  38 tokens   (history growing)
Turn 3 →  62 tokens   (history growing)
Turn 4 →  79 tokens   (history growing)
```
Every turn sends all previous messages — cost grows continuously.
Buffer memory (`history[-N:]`) is the fix for long sessions.

---

## 3 API providers — same concept, different SDK

| | Gemini | OpenAI | Anthropic |
|---|---|---|---|
| Client | `genai.Client()` | `OpenAI()` | `anthropic.Anthropic()` |
| Method | `generate_content()` | `chat.completions.create()` | `messages.create()` |
| History role (model) | `"model"` | `"assistant"` | `"assistant"` |
| Response text | `response.text` | `response.choices[0].message.content` | `response.content[0].text` |

### Gemini
```python
from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt,
    config=types.GenerateContentConfig(temperature=0.0)
)
print(response.text)
```

### OpenAI
```python
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0
)
print(response.choices[0].message.content)
```

### Anthropic
```python
import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=200,
    messages=[{"role": "user", "content": prompt}]
)
print(response.content[0].text)
```

---

## Phase 1 Capstone — Smart Q&A bot

### Architecture
```
Text file (any document)
       ↓
Load into system prompt
       ↓
User asks questions
       ↓
Model answers strictly from document
       ↓
Refuses to answer if not in document → hallucination prevention
```

### Key pattern — document in system prompt
```python
def build_system_prompt(document: str) -> str:
    return f"""You are a precise Q&A assistant. Answer strictly from the document below.

Rules:
- Only use information from the document
- If answer not in document → "This information is not in the document."
- Be concise — max 3 sentences
- Never make up information

Document:
{document}"""
```

### Hallucination prevention
When the user asked "What is the CEO's salary?" — the bot responded:
"This information is not in the document."

This is the most important safety feature in any RAG or Q&A system.
The system prompt rule + temperature 0.0 together enforce factual, grounded responses.

### Token counting
```python
token_count = client.models.count_tokens(
    model="gemini-2.0-flash",
    contents=history
).total_tokens
```

Always track tokens in production — costs money per token.

---

## Phase 1 — fully closed ✅

| Topic | Status |
|---|---|
| LLMs — how they work, tokens, temperature | ✅ |
| Prompt engineering — zero-shot, few-shot, CoT, system prompts | ✅ |
| Gemini API + Python SDK | ✅ |
| Memory patterns — buffer memory, conversation history | ✅ |
| Context windows — live code | ✅ |
| OpenAI / Anthropic API — pattern understood | ✅ |
| Phase 1 Capstone — Smart Q&A bot | ✅ |

---

*Session 04 complete · Phase 1 closed · Next: Phase 2 — Chatbots and Memory*