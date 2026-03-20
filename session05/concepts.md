# Session 05 — Summary Memory + LangChain Intro
> Snigdhaa's GenAI Course · Save as `session05/concepts.md`

---

## The one line summary
Buffer memory drops old messages. Summary memory compresses them — key facts survive forever.

---

## Why buffer memory is not enough

Buffer memory keeps last N messages. But what if your name was mentioned in turn 1 and your buffer is 10? After turn 11, your name is gone.

```
Turn 1  → "My name is Snigdhaa"     ← outside buffer after turn 11
Turn 2  → "I work at InCred"        ← outside buffer after turn 12
...
Turn 11 → "What is my name?"        ← bot says "I don't know"
```

Buffer memory trades old context for cost control. Summary memory solves this.

---

## Summary memory — how it works

```
All messages
     ↓
Filter out SystemMessage → chat_only
     ↓
If len(chat_only) > EXCEED:
     ↓
old_messages = chat_only[:-SIZE]   → summarise these
recent_messages = chat_only[-SIZE:] → keep these as-is
     ↓
summary = summarise(old_messages, previous_summary)
     ↓
Rebuild messages:
[SystemMessage("You are Aria... + Context: {summary}")] + recent_messages
     ↓
llm.invoke(messages) — LLM sees summary + recent context
```

**Key rule:** Summary lives inside the SystemMessage — model always reads it first.

---

## The summarise function — final correct version

```python
def summarise(old_messages: list, previous_summary: str = "") -> str:
    conversation = ""
    for msg in old_messages:
        if isinstance(msg, HumanMessage):
            conversation += f"User: {msg.content}\n"    # += not =
        if isinstance(msg, AIMessage):
            conversation += f"Assistant: {msg.content}\n"  # += not =

    # Carry forward previous summary so facts are never lost
    prior = f"Previous summary:\n{previous_summary}\n\n" if previous_summary else ""

    # prompt OUTSIDE the loop
    prompt = f"""Summarise this conversation in 3-5 sentences.
Preserve ALL key facts — names, company, role, preferences, technical context.
If a previous summary exists, merge it with the new conversation into one summary.

{prior}New conversation:
{conversation}"""

    response = summary_llm.invoke(prompt)
    return response.content.strip()
```

**3 rules that make this work:**
1. `+=` not `=` — accumulate all messages, don't overwrite
2. `prompt` outside the loop — built after all messages are collected
3. Pass `previous_summary` — each new summary merges with the old one, facts never lost

---

## The chat function — final correct version

```python
def chat(message: str):
    global messages, summary, total_turns
    total_turns += 1
    messages.append(HumanMessage(content=message))

    # Filter SystemMessage FIRST — clean slice boundaries
    chat_only = [m for m in messages if not isinstance(m, SystemMessage)]

    if len(chat_only) > EXCEED:
        old_messages = chat_only[:-SIZE]       # old → summarise
        recent_messages = chat_only[-SIZE:]    # recent → keep

        # Pass previous summary so facts carry forward
        summary = summarise(old_messages, previous_summary=summary)

        updated_system = f"{SYSTEM}\n\nContext from earlier:\n{summary}"
        messages = [SystemMessage(content=updated_system)] + recent_messages

    # LLM called AFTER messages updated — sees the summary
    response = llm.invoke(messages)
    messages.append(AIMessage(content=response.content))
    return response.content
```

---

## LangChain — what it is and why it exists

LangChain is a framework that gives you pre-built components for LLM applications.

```python
# Raw SDK (what you built from scratch)
history.append({"role": "user", "parts": [{"text": user_message}]})
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=history,
    config=types.GenerateContentConfig(system_instruction=SYSTEM)
)
reply = response.text.strip()

# LangChain (same result, cleaner)
messages.append(HumanMessage(content=user_message))
response = llm.invoke(messages)
reply = response.content
```

Same behaviour. Half the code.

### LangChain message types

```python
from langchain.messages import HumanMessage, AIMessage, SystemMessage

SystemMessage(content="You are Aria...")      # system prompt
HumanMessage(content="What is my name?")     # user message
AIMessage(content="Your name is Snigdhaa")   # model response
```

Always use keyword arguments — `HumanMessage(content=...)` not `HumanMessage(...)`.

### LangChain vs raw SDK

| | Raw SDK | LangChain |
|---|---|---|
| Message format | `{"role": "user", "parts": [{"text": "..."}]}` | `HumanMessage(content="...")` |
| API call | `client.models.generate_content(...)` | `llm.invoke(messages)` |
| Response | `response.text` | `response.content` |
| Streaming | `generate_content_stream()` | `llm.stream(messages)` |

---

## Tuning parameters

```python
SIZE = 8      # keep last 8 messages in full
EXCEED = 10   # summarise when chat messages exceed 10
```

**Rule of thumb:**
- `SIZE` should be large enough to keep recent context meaningful
- `EXCEED` should be `SIZE + 2` minimum — otherwise you summarise immediately after rebuilding
- If summary triggers too frequently → increase EXCEED
- If bot loses facts → decrease SIZE or ensure previous_summary is passed

---

## What you built today

| File | What it does | Key concept |
|---|---|---|
| `01_summary_memory.py` | Summary memory from scratch | Compression pattern |
| `02_langchain_intro.py` | Same chatbot with LangChain | SDK abstraction |
| `03_mini_project.py` | Full chatbot with summary memory + LangChain | Everything combined |

---

## What's coming — Session 6

Production system prompts and personas.
You'll build a chatbot that behaves differently for different users — a customer, a developer, an admin — all from the same codebase.

---

*Session 05 complete · Next: Session 06 — Production System Prompts and Personas*