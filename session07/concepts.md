# Session 07 — Phase 2 Capstone: Production Customer Support Bot
> Snigdhaa's GenAI Course · Save as `session07/concepts.md`

---

## The one line summary
A production bot is not one feature — it's all features combined: personas, memory, streaming, token tracking, and commands working together.

---

## What this capstone combines

| Feature | Session learned | How it's used |
|---|---|---|
| Streaming responses | Session 3 | `llm.stream(messages)` — tokens appear live |
| Buffer + summary memory | Session 5 | History compressed after EXCEED turns |
| LangChain messages | Session 5 | `HumanMessage`, `AIMessage`, `SystemMessage` |
| Production system prompts | Session 6 | 6-component prompts per persona |
| Multi-persona access control | Session 6 | Customer / Developer / Admin |
| Token tracking | Session 3 | Estimated cost per session |

---

## Full architecture

```
User selects persona at startup
         ↓
User types message
         ↓
Append to history as HumanMessage
         ↓
Filter chat_only (no SystemMessage)
         ↓
len(chat_only) > EXCEED?
    Yes → summarise old, keep recent, inject summary into SystemMessage
    No  → continue
         ↓
Build messages = [SystemMessage + summary] + history
         ↓
llm.stream(messages) → print tokens live
         ↓
Append full reply to history as AIMessage
         ↓
Update token estimate
```

---

## The complete session state

```python
history = []          # conversation messages
summary = ""          # compressed memory of old messages
total_tokens = 0      # running token estimate
total_turns = 0       # turn counter
current_persona = {}  # active persona dict
```

**Reset all state on persona switch** — never carry old persona context forward.

```python
def reset_session():
    global history, summary, total_tokens, total_turns
    history = []
    summary = ""
    total_tokens = 0
    total_turns = 0
```

---

## Summary memory with system prompt injection

```python
# Summary injected into SystemMessage — not as a separate message
system_content = current_persona["system"]
if summary:
    system_content += f"\n\nContext from earlier:\n{summary}"

messages = [SystemMessage(content=system_content)] + history
```

Key rule — summary lives inside SystemMessage, not as a standalone message in history.

---

## Streaming with LangChain

```python
# LangChain streaming — use .stream() not .invoke()
for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)
    full_reply += chunk.content

print("\n")

# Always append full reply AFTER the stream loop completes
history.append(AIMessage(content=full_reply))
```

---

## Command handling pattern

```python
cmd = user_input.lower()

if cmd == "quit":    → exit with stats
elif cmd == "switch" → select persona + reset session
elif cmd == "status" → show session stats
elif cmd == "history"→ show conversation log
elif cmd == "help"   → show commands
else:                → chat(user_input)
```

Always handle commands before passing to the LLM.
Always strip and lowercase user input before command matching.

---

## Production bot checklist — what makes this deployable

- [x] Persona-based access control
- [x] Summary memory — history never grows unbounded
- [x] Streaming — real-time UX
- [x] Token tracking — cost visibility
- [x] Graceful fallbacks in every persona
- [x] Clean session reset on persona switch
- [x] KeyboardInterrupt handled — Ctrl+C exits cleanly
- [x] Commands — help, status, history, switch, quit

---

## Phase 2 — fully closed ✅

| Session | Topic | Status |
|---|---|---|
| Session 5 | Summary memory + LangChain | ✅ |
| Session 6 | Production system prompts + personas | ✅ |
| Session 7 | Phase 2 Capstone | ✅ |

---

## What's coming — Phase 3

RAG systems — Retrieval Augmented Generation.
You'll give the bot access to real documents and it will answer questions based on them.
Uses everything from Phase 1 and 2, plus embeddings and vector databases.

Phase 3 index will be provided at the start of Session 8.

---

*Session 07 complete · Phase 2 closed · Next: Phase 3 — RAG Systems*