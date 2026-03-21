# Session 07 — Mistakes
> Snigdhaa's GenAI Course · Save as `session07/mistakes.md`

---

## No bugs today ✅

Session 7 ran cleanly. All features worked first try.

This is a direct result of fixing and understanding every bug from Sessions 3–6 properly.
When you understand *why* a bug happens, you stop making it.

---

## Patterns that prevented bugs today

### Pattern 1 — Filter before slice
```python
# Learned the hard way in Session 5
chat_only = [m for m in history if not isinstance(m, SystemMessage)]
if len(chat_only) > EXCEED:
    old_messages = chat_only[:-SIZE]   # clean boundaries
```

### Pattern 2 — Append after stream, not inside
```python
# Learned in Session 3
full_reply = ""
for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)
    full_reply += chunk.content
history.append(AIMessage(content=full_reply))  # after loop, not inside
```

### Pattern 3 — Reset all state on switch
```python
# Prevents persona context leaking
def reset_session():
    global history, summary, total_tokens, total_turns
    history = []
    summary = ""
    total_tokens = 0
    total_turns = 0
```

### Pattern 4 — keyword arguments always
```python
# Explicit is better than implicit
HumanMessage(content=user_input)      # ✅
AIMessage(content=full_reply)         # ✅
SystemMessage(content=system_content) # ✅
```

### Pattern 5 — Handle commands before LLM
```python
# Commands checked first — LLM never sees "quit" or "switch"
cmd = user_input.lower()
if cmd == "quit": ...
elif cmd == "switch": ...
else: chat(user_input)  # only reaches here if not a command
```

---

## The bigger lesson

Sessions 3–6 had 8+ bugs between them. Session 7 had zero.
The bugs weren't wasted time — they built the muscle memory that made Session 7 clean.

Every mistake you fixed manually is a pattern you now carry automatically.

---

*Session 07 mistakes logged · Phase 2 complete*