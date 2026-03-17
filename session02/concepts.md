# Session 02 — Prompt Engineering Reference
> Snigdhaa's GenAI Course · Save as `session02/concepts.md`

---

## The one line summary
Same model. Same question. Prompt quality = output quality. This is the highest-leverage skill in GenAI.

---

## Technique 1 — Zero-shot

No examples given. Just an instruction.

```python
prompt = "Classify this review as Positive, Negative, or Mixed:\nThe battery dies fast but screen is gorgeous."
```

**When to use:** Simple, unambiguous tasks where the model clearly understands what you want.
**Weakness:** Model uses its own judgement on format and behaviour. Can go off-track on ambiguous input.
**Rating: 4/10** on complex tasks.

---

## Technique 2 — Few-shot

Give examples before the actual question. Model learns the pattern from examples.

```python
prompt = """Classify each review as Positive, Negative, or Mixed.

Review: "Absolutely love it, works perfectly!" → Positive
Review: "Broke after one day, waste of money." → Negative
Review: "Great camera but terrible battery life." → Mixed

Review: "The battery dies fast but screen is gorgeous." →"""
```

**When to use:** When you want consistent format. Classification, extraction, transformation tasks.
**Why it works:** Model pattern-matches your examples and follows the same structure.
**Rating: 6/10** — better format, but still no strict behaviour control.

---

## Technique 3 — Chain of Thought (CoT)

Force the model to reason step by step before giving a final answer.

```python
prompt = f"""{problem}
Think through this step by step, then give your final answer."""
```

**When to use:** Math problems, logic problems, multi-step reasoning, debugging, architecture decisions.
**Why it works:** Forces the model to not skip steps — like showing work in an exam.
**Important lesson from today:** CoT does NOT help on simple problems. Gemini 2.0 gave the same answer with and without CoT on the apple problem because it was too easy. CoT shines on genuinely complex reasoning.
**Rating: 7/10** on complex tasks.

---

## Technique 4 — System Prompts

Set the persona, behaviour rules, and constraints BEFORE the conversation starts.

```python
config = types.GenerateContentConfig(
    system_instruction="""You are a senior backend engineer with 10 years experience.
You give practical, opinionated advice.
You always mention what NOT to do.
Keep responses under 150 words.""",
    temperature=0.7
)
```

**When to use:** Every production application. Always.
**What it controls:**
- Persona (who the model acts as)
- Format rules (bullet points, JSON, word limits)
- Behaviour rules (never assume, always ask, no markdown)
- Constraints (scope, tone, language)

**Key observation from today:**
Same question "How should I structure my Python project?" gave completely different answers:
- Senior engineer → opinionated, structured, included Don'ts
- Teaching assistant → house analogy, beginner-friendly, encouraging

**Rating: 8/10** — strongest single technique.

---

## Technique 5 — Structured Output

Force the model to respond in a specific format (JSON, CSV, XML).

```python
prompt = f"""Analyse the following code and respond ONLY in valid JSON.
No explanation outside the JSON. No markdown backticks.

JSON format:
{{
    "rating": <integer 1-10>,
    "issues": [<list of issues found>],
    "improvements": [<list of specific improvements>],
    "verdict": "<one sentence summary>"
}}

Code to review:
{code}"""
```

Then parse it:
```python
raw = response.text.strip()
raw = raw.replace("```json", "").replace("```", "").strip()  # model adds backticks anyway
result = json.loads(raw)
```

**When to use:** Any time downstream code needs to process the LLM output programmatically.
**Real world use cases:** Code review tools, data extraction, classification pipelines, RAG evaluation.
**Key lesson:** Always clean markdown backticks — even when you tell the model not to add them, it sometimes does anyway. Defensive coding.

---

## The combination is where real power is

| Technique | Alone | Combined |
|---|---|---|
| Zero-shot | 4/10 | — |
| Few-shot | 6/10 | + zero-shot context |
| Chain-of-thought | 7/10 | + system prompt |
| System prompt | 8/10 | + few-shot + CoT |
| All combined | — | 9/10 |

Production prompt = system prompt + few-shot examples + CoT instruction + structured output format.

---

## Critical bugs you debugged today — remember these forever

### Bug 1 — f-string outside quotes (same as Session 1)
```python
# Wrong ❌ — {code} is never substituted
prompt = """...{code}..."""

# Correct ✅
prompt = f"""...{code}..."""
```

### Bug 2 — Curly braces inside f-strings
```python
# Wrong ❌ — Python tries to evaluate {"rating": ...} as Python code
prompt = f"""
{{
    "rating": ...   # Wrong — Python sees this as a set expression
}}
{code}
"""

# Correct ✅ — double braces = literal curly brace
prompt = f"""
{{
    "rating": ...   # {{ and }} render as { and }
}}
{code}             # single braces = variable substitution
"""
```

### Bug 3 — Calling a function on itself
```python
# Wrong ❌ — passes the return value of review_code back into review_code
review_code(review_code(bad_code))

# Correct ✅
review_code(bad_code)
```

### Bug 4 — .strip without parentheses (same as Session 1)
```python
raw = response.text.strip   # ❌ references the function, doesn't call it
raw = response.text.strip() # ✅ actually calls it
```

---

## The most important lesson of Session 2

**Instruction following vs helpfulness conflict.**

When your prompt is ambiguous, Gemini's helpfulness instinct overrides your instruction.
The model tried to identify what device the review was about instead of classifying it —
because the review had no product context.

Fix: Always add a system prompt with strict behaviour rules for production use cases.

```python
system = """You are a classification engine.
You ONLY output one word: Positive, Negative, or Mixed.
No explanations. No questions. No extra text."""
```

This is the most common bug in real AI applications — the model being "too helpful."

---

## Streaming — added to your toolkit today

```python
# Without streaming — waits for full response ❌
response = client.models.generate_content(...)
print(response.text)

# With streaming — prints each token as it arrives ✅
for chunk in client.models.generate_content_stream(...):
    print(chunk.text, end="", flush=True)
print()
```

`end=""` — no newline between chunks
`flush=True` — forces immediate print, no buffering

---

## What you built today

| File | What it does | Key concept |
|---|---|---|
| `01_zero_vs_fewshot.py` | Classifies product reviews | Zero-shot vs few-shot |
| `02_chain_of_thought.py` | Solves a math word problem | CoT reasoning |
| `03_system_prompt.py` | Same question, two personas | System prompts + streaming |
| `04_mini_project.py` | Code review tool with JSON output | Structured output + f-string escaping |

---

## What's coming in Session 3

**Multi-turn conversations and memory.**
You'll build a chatbot that actually remembers what you said 10 messages ago.
This is where conversation history, context window management, and memory patterns come in.
Everything you learned about system prompts today becomes the foundation for Session 3.

---

*Session 02 complete · Next: Session 03 — Chatbots and Memory*