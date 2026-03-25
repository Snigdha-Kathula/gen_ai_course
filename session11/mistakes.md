# Session 11 — Mistakes
> Snigdhaa's GenAI Course · Save as `session11/mistakes.md`

---

## No code bugs today ✅

All 4 files ran correctly first try. Pattern from Sessions 8-10 holding strong.

---

## Conceptual insight — hybrid score inflation

**What happened:**
```
Query: "What does BM25 stand for"  (not in knowledge base)
  Semantic: 0.098
  Hybrid:   0.369  ← higher than expected
```

**Why this is a problem:**
0.369 is above the 0.3 retrieval threshold — meaning a bad result
would have been passed to the LLM and possibly generated a hallucinated answer.

**Root cause:**
Common words ("does", "for", "what") get tiny BM25 scores across many chunks.
When normalised (divided by max score) and multiplied by 0.3 weight,
they inflate the combined score beyond what the semantic signal alone deserved.

**Production fix:**
```python
# Apply minimum threshold to each signal BEFORE combining
MIN_BM25 = 0.1   # ignore noisy keyword matches

keyword_scores = {
    i: normalised_score
    for i, normalised_score in raw_keyword_scores.items()
    if normalised_score > MIN_BM25   # filter noise first
}
```

**Rule:** Always apply independent thresholds to each signal before fusion.
Never let noise from one signal inflate the combined score.

---

## Reranking trade-off to remember

```
Reranking accuracy  ↑  as candidate pool grows
Reranking speed     ↓  as candidate pool grows
```

For production — retrieve top 10, rerank to top 3.
Never rerank the entire corpus — too slow.
Never rerank only top 3 — candidate pool too small to find the best result.

**Sweet spot:** Retrieve 5-10 × n_results, rerank to n_results.

---

## BM25 normalisation — always required

```python
# Wrong ❌ — raw BM25 scores are unbounded (0 to infinity)
keyword_scores[i] = bm25_scores[i]

# Correct ✅ — normalise to 0-1 before combining with semantic scores
max_score = max(bm25_scores) if max(bm25_scores) > 0 else 1
keyword_scores[i] = bm25_scores[i] / max_score
```

Raw BM25 and cosine similarity are on completely different scales.
Always normalise before combining — otherwise BM25 dominates regardless of weight.

---

*Session 11 mistakes logged*