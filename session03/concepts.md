# Session 03 — Chatbots and Memory
> Snigdhaa's GenAI Course · Save as `session03/concepts.md`

---

## The one line summary
LLMs have zero memory. What looks like memory is you sending the full conversation history with every API call.

---

## How memory actually works

Every turn, you send ALL previous messages + the new one:

```
Turn 1 → ["Hi my name is Snigdhaa"]
Turn 2 → ["Hi my name is Snigdhaa", "Nice to meet you!", "What is my name?"]
Turn 3 → ["Hi my name is Snigdhaa", "Nice to meet you!", "What is my name?", "Snigdhaa!", "What do I do?"]
```

The model reads it all fresh every time. It has no persistent state.

---

## The problem — token cost grows every turn

From your live output today:
```
Turn 1 →  81 tokens   (just the intro)
Turn 2 → 279 tokens   (history growing)
Turn 3 → 469 tokens   (history growing further)
Total  → 829 tokens   after just 3 turns
```

After 100 turns → cost is massive + context window exceeded + performance degrades.

---

## 3 memory patterns — how to solve it

| Pattern | How it works | When to use |
|---|---|---|
| Buffer memory | Keep only last N messages | Simple chatbots, short sessions |
| Summary memory | Summarise old messages, keep recent | Long conversations, cost control |
| Vector memory | Store all, retrieve only relevant parts | RAG systems, Phase 3 |

---

## Pattern 1 — Buffer memory (built today)

Keep only the last N messages. Simple, effective, cheap.

```python
conversation_history = []

def chat(user_message: str, max_history: int = 10) -> str:
    conversation_history.append({
        "role": "user",
        "parts": [{"text": user_message}]
    })

    # Buffer — send only last N messages, not full history
    trimmed = conversation_history[-max_history:]

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=trimmed,
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful assistant.",
            temperature=0.7
        )
    )

    reply = response.text.strip()

    # CRITICAL — always append model reply back to history
    conversation_history.append({
        "role": "model",
        "parts": [{"text": reply}]
    })

    return reply
```

**Critical rule:** Always append the model's reply to history after every turn.
If you forget → model never sees its own previous replies → conversation breaks.

---

## Pattern 2 — Summary memory (Phase 2)

When history gets too long, summarise old messages into one paragraph.
Send: summary + last 5 messages. History stays small, context preserved.

```
Instead of 50 messages →
"Summary: Snigdhaa is a backend developer who asked about K9s 
and phone backup. Prefers concise answers." + last 5 messages
```

---

## Pattern 3 — Vector memory (Phase 3)

Store everything. Retrieve only the relevant parts using embeddings.
Used in RAG systems — covered fully in Phase 3.

---

## Streaming in chatbots

Always use streaming in interactive chatbots — better UX.

```python
full_reply = ""

for chunk in client.models.