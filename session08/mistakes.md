# Session 08 — Mistakes
> Snigdhaa's GenAI Course · Save as `session08/mistakes.md`

---

## Mistake 1 — Printed wrong variable for embedding values

**What happened:**
```
First 5 values: ['I love programming in Python', 'Python is my favourite language', ...]
```
Printed the `sentences` list instead of `sample[:5]`.

**The fix:**
```python
# Wrong ❌
print(f"First 5 values: {sentences[:5]}")

# Correct ✅
print(f"First 5 values: {sample[:5]}")
```

**Why it matters:**
This is a silent bug — no crash, just wrong output. Always verify
what variable you're printing, especially when variable names are similar.

---

## Key insight logged — retrieval quality depends on document quality

Query: "planets and space exploration" → best score was only 0.154.

**Why:** The knowledge base had no document about planets generally.
Only specific facts about Mars and Milky Way.

**The lesson:**
RAG retrieval quality = document collection quality.
Before building a RAG system, always ask:
- Does my knowledge base cover the topics users will ask about?
- Are my documents specific enough to match likely queries?
- Are there gaps that will cause low similarity scores?

This is called **knowledge base design** — covered in Session 10.

---

## Pattern learned — get or create collection

```python
# Wrong ❌ — crashes on second run, collection already exists
collection = chroma_client.create_collection(name="science_facts")

# Correct ✅ — safe for any run
def get_or_create_collection(name: str):
    try:
        return chroma_client.get_collection(name)
    except:
        return chroma_client.create_collection(name)
```

Always use get-or-create for persistent collections.

---

*Session 08 mistakes logged*