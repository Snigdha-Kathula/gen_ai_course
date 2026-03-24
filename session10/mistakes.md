# Session 10 — Mistakes
> Snigdhaa's GenAI Course · Save as `session10/mistakes.md`

---

## Mistake 1 — Retrieval returning None

**What happened:**
```
query: how does FastAPI handle async requests
result: None
```

**Root cause:**
Print statement used wrong variable — printed `None` instead of iterating results.

**The fix:**
Always verify what variable you're printing. Run a quick debug print
of the raw results object before formatting to confirm its structure.

```python
# Debug pattern — always use this when results look wrong
print(type(results))
print(results)
```

**Rule:** When output looks wrong but no error is thrown — print the raw object first.

---

## Conceptual insight — Dockerfile answer was short

**What happened:**
```
Query: "What is a Dockerfile?"
Answer: "A Dockerfile is used to define the build steps for creating a container image."
```
Only one sentence — much shorter than other answers.

**Why:**
The knowledge base has one sentence about Dockerfile buried inside a Docker chunk.
The chunk's dominant topic is Docker containers, not Dockerfile specifically.
Only one chunk passed the 0.3 similarity threshold.

**The lesson:**
If answers for specific topics are too short — check your knowledge base.
Add a dedicated chunk focused on that topic.

```
Bad ❌  → "Docker uses a Dockerfile to define build steps." (1 sentence in a Docker chunk)
Good ✅ → Dedicated chunk: "A Dockerfile is a text file containing instructions
          for building a Docker image. Each instruction creates a new layer..."
```

---

## Key design decision — similarity threshold

```python
if similarity > 0.3:  # threshold
    chunks.append(doc)
```

**Too low (0.1):** Irrelevant chunks reach the LLM → hallucination risk
**Too high (0.6):** Too many valid chunks rejected → "I don't have information" for valid questions
**Sweet spot (0.3):** Rejects clearly irrelevant results, accepts relevant ones

Tune this threshold based on your knowledge base and document quality.
Always test with adversarial queries (questions outside the KB) to verify rejection works.

---

## History management insight

Storing full context messages in history causes history to grow very fast.
Context (300 chars × 3 chunks) + query = ~1000 chars per turn.
After 20 turns → 20,000 chars in history → expensive.

**The fix:**
```python
# Store context for LLM call
history.append(HumanMessage(content=context_message))
response = llm.invoke(messages)

# Replace with clean query — don't store context in history
history[-1] = HumanMessage(content=user_query)  # ← key line
history.append(AIMessage(content=answer))
```

Always separate what you send to the LLM from what you store in history.

---

*Session 10 mistakes logged*