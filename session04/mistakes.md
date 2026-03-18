# Session 04 — Mistakes
> Snigdhaa's GenAI Course · Save as `session04/mistakes.md`

---

## Mistake 1 — Wrong Gemini message format

**What was written:**
```python
{"role": "user", "text": "Who founded the company?"}
```

**Error it caused:**
`pydantic_core.ValidationError` — 14 validation errors. API rejected the request entirely.

**The fix:**
```python
{"role": "user", "parts": [{"text": "Who founded the company?"}]}
```

**Why it works:**
Gemini's SDK requires messages in `Content` format — role + parts array containing text objects. The `text` key must be nested inside `parts`, not directly on the message object.

**The rule to remember:**
```python
# Gemini message format — always
{
    "role": "user",          # or "model"
    "parts": [
        {"text": "your message here"}
    ]
}
```

**Why this trips people up:**
OpenAI and Anthropic use a flat format — `{"role": "user", "content": "message"}`. When switching between providers, this is the #1 format mistake. Always check the SDK docs when switching providers.

---

## Pattern to remember — 3 provider formats side by side

```python
# Gemini
{"role": "user", "parts": [{"text": "message"}]}

# OpenAI
{"role": "user", "content": "message"}

# Anthropic
{"role": "user", "content": "message"}
```

Gemini is the odd one out — `parts` array instead of `content` string.

---

*Session 04 mistakes logged · Phase 1 closed*