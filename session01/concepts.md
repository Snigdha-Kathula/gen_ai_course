# Session 01 — Core Concepts Reference
> Snigdhaa's GenAI Course · Save this file in your `genai-course/` folder as `session01/concepts.md`

---

## 1. Tokenization

Before an LLM reads a single word of your prompt, it converts everything into **tokens**. The model never sees text — it only ever sees numbers. Tokenization is that conversion step.

Think of it like this — your Python code never sees the word `"hello"` either. It sees bytes. Tokenization is the LLM's version of that.

### Examples

**Simple sentence:**
```
"The cat sat on the mat"
→ [The][cat][sat][on][the][mat]
→ [464][6245][9139][319][264][2611]
```
Common words like "the", "on", "cat" each get one token.
Notice "The" (capital) = token 464 and "the" (lowercase) = token 264 — they are different tokens. Case matters.

**Long/rare words get split:**
```
"unhappiness"  →  [un][happi][ness]       = 3 tokens
"automation"   →  [autom][ation]          = 2 tokens
"Snapchat"     →  [Snap][chat]            = 2 tokens
```
Rare or long words get broken into subword pieces. The model still understands them —
it learns that `un + happi + ness` combines to mean unhappiness.
This is why LLMs sometimes struggle with spelling — they never see individual letters.

**Code tokenizes differently:**
```python
def hello():       →  [def][ hello][()][:]
    print("Hello") →  [    ][print]["Hello"]
```
Spaces and indentation are part of the token! The 4-space indent is its own token.
This is why LLMs are sensitive to formatting in code prompts — every whitespace is meaningful.

**Numbers and math (why LLMs struggle):**
```
1, 2, 3     → one token each   (small numbers)
9472        → [9][4][72]        (large numbers split arbitrarily)
9472 + 1337 = ?  → 13 tokens for 5 "words"
```
The model has no concept of numerical value — it's just pattern matching on token sequences.
This is why base LLMs are bad at arithmetic.

### The 4 rules every developer must know

1. **~1 token ≈ 0.75 words** — 100 tokens ≈ 75 English words. Use this to estimate API costs — you pay per token, not per word.
2. **Context window = token limit** — Gemini 1.5 Flash has a 1M token context window. That's ~750,000 words. But it still costs money per token sent.
3. **Non-English = more tokens** — Hindi, Tamil, Chinese text uses 2–4× more tokens than English for the same meaning. Costs multiply for multilingual apps.
4. **Prompt + response = total tokens** — You're billed for input tokens (your prompt) AND output tokens (the response).

### Token cost order (memorise this)
```
Code > Numbers > English > Common languages
(most tokens per word)     (fewest tokens per word)
```

### Your live results from `04_tokens.py`
```
"The cat sat on the mat"         → Words: 6  | Tokens: 7  | Ratio: 1.17
"unhappiness automation Snapchat"→ Words: 3  | Tokens: 5  | Ratio: 1.67
"def hello(): print('Hello')"    → Words: 3  | Tokens: 8  | Ratio: 2.67
"9472 + 1337 = ?"                → Words: 5  | Tokens: 13 | Ratio: 2.60
"नमस्ते दुनिया"                    → Words: 2  | Tokens: 3  | Ratio: 1.50
```

---

## 2. Embeddings — Why LLMs Convert Text to Numbers

### Tokenization vs Embeddings — two separate steps
```
Your text
   ↓
Tokenization  →  breaks text into token IDs (integers)
   ↓
Embeddings    →  converts each token ID into a vector (list of numbers)
   ↓
Transformer layers process the vectors
```

### What is an Embedding?
An embedding converts a token into a **list of ~768 to 4096 floating point numbers** called a vector.
Each number captures some aspect of meaning.

```python
"king"  →  [0.23, -0.87, 0.45, 0.12, -0.33, ...]  # 768 numbers
"queen" →  [0.21, -0.85, 0.44, 0.91, -0.31, ...]  # 768 numbers
"apple" →  [0.89,  0.12, -0.67, 0.03, 0.77, ...]  # 768 numbers
```
Notice `king` and `queen` have very similar numbers. `apple` looks completely different.
**Similar meaning = similar numbers.**

### Why convert to numbers at all?
Because computers cannot do math on words. Neural networks are giant math operations —
matrix multiplications, additions, activations. You can't multiply `"king"` by a matrix.
You absolutely can multiply `[0.23, -0.87, 0.45...]` by a matrix.

**The entire point: turn meaning into math.**

### The magic — meaning becomes geometry
```
king - man + woman ≈ queen
paris - france + italy ≈ rome
```
When you do that subtraction and addition on the actual vectors, the result is closest to the
correct answer. Meaning has literally become spatial distance. Words that are conceptually
related end up close together in 768-dimensional space.

### Why this matters for your course
In Phase 3 (RAG systems), embeddings are the entire engine:
```
User asks: "What is our refund policy?"
         ↓
Convert question to embedding vector
         ↓
Search document database for vectors CLOSEST to it
         ↓
Return most similar chunks → feed to LLM
```
You're not searching for matching words — you're searching for matching *meaning*.

### Code preview (Phase 3)
```python
from google import genai

client = genai.Client(api_key="...")

result = client.models.embed_content(
    model="text-embedding-004",
    contents="What is the refund policy?"
)

print(result.embeddings[0].values[:5])
# → [0.023, -0.412, 0.891, -0.234, 0.567]
```

---

## 3. Temperature — How It Affects LLM Output

### What temperature controls
Temperature controls how the model picks the next token from its probability distribution.

```
At every step, the model has probabilities like:
"force"      → 42%
"pull"        → 31%
"attraction"  → 18%
"banana"      →  0.1%
```

- **Temperature 0.0** — always picks the highest probability token. Deterministic. Same input = same output every time.
- **Temperature 1.0** — samples from the distribution as-is. Some creativity, still coherent.
- **Temperature 2.0** — flattens the distribution. Low probability tokens get a chance. Output gets chaotic.

### Your live results from `02_tokens_and_temp.py`
```
Temperature 0.0 → "Your daily grind, elevated."  (identical all 3 times)
Temperature 1.0 → "Your daily grind, elevated."  (Gemini 2.0 is conservative)
Temperature 2.0 → "Your daily grind, perfectly grounded."
                  "Your daily grind, redefined."
                  "Your daily grind, redefined."
```

### Key observation from your run
Gemini 2.0 Flash resisted randomness even at 1.0 — that's because it's a
production-hardened model. Different models respond to temperature differently.
You'll see this vary across OpenAI, Anthropic, and Google models.

### When to use what temperature
| Use case | Temperature |
|---|---|
| Factual Q&A, data extraction | 0.0 – 0.3 |
| Chatbots, assistants | 0.5 – 0.7 |
| Creative writing, brainstorming | 0.8 – 1.0 |
| Experimental / chaotic | 1.5+ |

---

## 4. Streaming in LLM APIs

### The problem without streaming
```python
response = client.models.generate_content(...)
print(response.text)  # nothing appears until FULL response is ready
```
For a 500 word response — you stare at a blank screen for 8 seconds, then wall of text.
That's terrible UX. Nobody builds production apps this way.

### What streaming does
Instead of waiting for the full response, you receive each token as it's generated —
exactly like ChatGPT's typing effect.
```
Without streaming:  [8 seconds of nothing]... "Gravity is a force that..."
With streaming:     "Grav" → "ity" → " is" → " a" → " force" → " that" → ...
```

### The code — only 2 things change
```python
# Without streaming ❌
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Explain quantum computing in detail."
)
print(response.text)

# With streaming ✅
for chunk in client.models.generate_content_stream(
    model="gemini-2.0-flash",
    contents="Explain quantum computing in detail."
):
    print(chunk.text, end="", flush=True)
print()
```

Two differences:
- `generate_content` → `generate_content_stream`
- Loop over chunks instead of reading `.text` once

### Why `flush=True` matters
```python
print(chunk.text, end="", flush=True)
#                          ↑
# Forces Python to immediately write to terminal
# Without this, Python buffers output and you still get wall of text
```

### How it works under the hood
```
Your code          Gemini API
   |                   |
   |── HTTP request ──→|
   |                   | generating token 1...
   |←── chunk 1 ───────| "Grav"
   | print "Grav"      | generating token 2...
   |←── chunk 2 ───────| "ity"
   | print "ity"       | ...
   |←── stream end ────|
```
This is a persistent HTTP connection — stays open until the model signals done.
Your `for` loop keeps reading until that signal arrives.

### Where you'll use streaming in this course
| Project | Why streaming matters |
|---|---|
| Chatbot (Phase 2) | Users see response forming — feels alive |
| RAG system (Phase 3) | Long answers appear progressively |
| Voice bot (Phase 5) | Start speaking before full response — cuts latency 2-3 seconds |

---

*Session 01 complete · Next: Session 02 — Prompt Engineering*