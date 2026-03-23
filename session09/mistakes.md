# Session 09 — Mistakes
> Snigdhaa's GenAI Course · Save as `session09/mistakes.md`

---

## No code bugs today ✅

All 4 files ran correctly first try.

---

## Conceptual mistake caught — chunk content vs keyword presence

**The assumption:**
"If a chunk contains the word 'neural networks', a query about neural networks will score high."

**What actually happened:**
```
Query: "what are neural networks"
Best chunk: "Deep learning is a subset of ML that uses neural networks with many layers..."
Score: 0.335  ← lower than expected
```

**Why:**
The embedding captures the **dominant meaning** of the chunk — deep learning.
Neural networks is mentioned but not the focus.
The score reflects how well the *overall chunk* matches the *overall query*.
Not whether individual keywords match.

**The fix for production RAG:**
Design chunks so each one has one clear, focused idea.
Don't let important concepts get buried inside chunks about other topics.

```
Bad chunk ❌ — neural networks mentioned inside a deep learning chunk
Good chunk ✅ — dedicated chunk: "Neural networks are computational models
               inspired by the brain, consisting of layers of interconnected nodes..."
```

---

## The rule to remember

```
Embedding = dominant meaning of the entire chunk
NOT = presence of keywords inside the chunk
```

This is the most important chunking insight for production RAG.
If retrieval scores are low, the first thing to check is chunk design —
not embedding model, not vector DB, not the query itself.

---

## Chunk overlap insight

```python
chunk_overlap=50  # ✅ always use overlap
chunk_overlap=0   # ❌ sentences at boundaries lose context
```

Without overlap, a concept split across two chunks becomes
unretrievable — neither chunk has the full context.

---

*Session 09 mistakes logged*