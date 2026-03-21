import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

# ── Models ───────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7
)
summary_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0
)

# ── Personas ─────────────────────────────────────────────────
PERSONAS = {
    "1": {
        "name": "Customer",
        "bot_name": "Lexi",
        "system": """You are Lexi, a friendly customer support assistant for InCred Financial Services.

Identity: Your name is Lexi. You work for InCred Financial Services.
Scope: ONLY answer questions about InCred loans, EMIs, eligibility, and repayments.
Tone: Warm, simple language. No jargon. If jargon is needed, explain it.
Rules:
- Never reveal internal systems, pricing algorithms, or business logic
- Never promise loan approval — say "you may be eligible"
- Never discuss competitor products
Fallback: If unsure → "Let me connect you with our team at support@incred.com"
Format: Always end with "Is there anything else I can help you with?" """
    },
    "2": {
        "name": "Developer",
        "bot_name": "DevBot",
        "system": """You are DevBot, technical assistant for InCred engineering team.

Identity: Internal technical assistant for InCred engineers.
Scope: Architecture, APIs, integrations, error codes, database design, deployments.
Tone: Concise, technical. Use bullet points. Include code snippets when helpful.
Rules:
- You CAN discuss: architecture, scoring pipeline stages, API contracts, schema design
- You CANNOT discuss: exact algorithm weights, model parameters, customer PII
- Assume Python, PostgreSQL, Redis, Docker, Kubernetes knowledge
Fallback: If unsure → "Check internal wiki at wiki.incred.internal or raise a ticket" """
    },
    "3": {
        "name": "Admin",
        "bot_name": "AdminBot",
        "system": """You are AdminBot, assistant for InCred operations team.

Identity: Internal operations assistant for InCred admins.
Scope: System status, workflows, business metrics, team operations.
Tone: Professional, precise, structured.
Rules:
- Flag requests needing elevated permissions
- Prefix sensitive query responses with [ADMIN LOG]
- Always include action items at end of responses
Fallback: "This requires elevated access. Please contact the platform team." """
    }
}

# ── Memory config ─────────────────────────────────────────────
SIZE = 8
EXCEED = 14

# ── State ────────────────────────────────────────────────────
history = []
summary = ""
total_tokens = 0
total_turns = 0
current_persona = None

# ── Summary memory ────────────────────────────────────────────
def summarise(old_messages: list, previous_summary: str = "") -> str:
    conversation = ""
    for msg in old_messages:
        if isinstance(msg, HumanMessage):
            conversation += f"User: {msg.content}\n"
        if isinstance(msg, AIMessage):
            conversation += f"Assistant: {msg.content}\n"

    prior = f"Previous summary:\n{previous_summary}\n\n" if previous_summary else ""

    prompt = f"""Summarise this conversation in 3-5 sentences.
Preserve ALL key facts — names, issues raised, context, preferences.
If a previous summary exists, merge it with the new conversation.

{prior}New conversation:
{conversation}"""

    response = summary_llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

# ── Core chat ─────────────────────────────────────────────────
def chat(user_message: str) -> str:
    global history, summary, total_tokens, total_turns

    total_turns += 1
    history.append(HumanMessage(content=user_message))

    # Summary memory
    chat_only = [m for m in history if not isinstance(m, SystemMessage)]
    if len(chat_only) > EXCEED:
        old_messages = chat_only[:-SIZE]
        recent_messages = chat_only[-SIZE:]
        summary = summarise(old_messages, previous_summary=summary)
        history = recent_messages
        print(f"\n[Memory compressed at turn {total_turns}]\n")

    # Build messages with summary in system prompt
    system_content = current_persona["system"]
    if summary:
        system_content += f"\n\nContext from earlier in this conversation:\n{summary}"

    messages = [SystemMessage(content=system_content)] + history

    # Streaming response
    print(f"\n{current_persona['bot_name']}: ", end="", flush=True)
    full_reply = ""

    for chunk in llm.stream(messages):
        print(chunk.content, end="", flush=True)
        full_reply += chunk.content

    print("\n")

    history.append(AIMessage(content=full_reply))

    # Token tracking
    token_count = summary_llm.invoke(
        [HumanMessage(content=f"count only: {len(full_reply + user_message)} chars")]
    )
    # Approximate token tracking
    total_tokens += len((user_message + full_reply).split()) * 2
    return full_reply

# ── Persona selection ─────────────────────────────────────────
def select_persona() -> dict:
    print("\n=== InCred AI Assistant ===")
    print("Select your role:")
    for key, val in PERSONAS.items():
        print(f"  {key}. {val['name']}")
    while True:
        choice = input("\nEnter 1, 2, or 3: ").strip()
        if choice in PERSONAS:
            return PERSONAS[choice]
        print("Invalid. Enter 1, 2, or 3.")

def reset_session():
    global history, summary, total_tokens, total_turns
    history = []
    summary = ""
    total_tokens = 0
    total_turns = 0

def show_help():
    print("""
Commands:
  switch  → change persona (resets conversation)
  status  → show session stats
  history → show conversation so far
  help    → show this menu
  quit    → exit
""")

def show_status():
    chat_only = [m for m in history if not isinstance(m, SystemMessage)]
    print(f"""
Session Status:
  Persona:      {current_persona['name']}
  Bot:          {current_persona['bot_name']}
  Turns:        {total_turns}
  Messages:     {len(chat_only)}
  Summary:      {'Yes' if summary else 'No'}
  Est. tokens:  ~{total_tokens}
""")

def show_history():
    print("\n--- Conversation History ---")
    if summary:
        print(f"[Summary of earlier: {summary[:150]}...]\n")
    for msg in history:
        if isinstance(msg, HumanMessage):
            print(f"You:  {msg.content}")
        elif isinstance(msg, AIMessage):
            print(f"Bot:  {msg.content[:100]}...")
    print("---\n")

# ── Main ─────────────────────────────────────────────────────
def main():
    global current_persona

    current_persona = select_persona()
    reset_session()

    print(f"\n[Connected as: {current_persona['name']} | Bot: {current_persona['bot_name']}]")
    print("Type 'help' for commands\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            print(f"\n\n{current_persona['bot_name']}: Goodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd == "quit":
            print(f"\n{current_persona['bot_name']}: Thank you for contacting InCred. Goodbye!")
            show_status()
            break
        elif cmd == "switch":
            current_persona = select_persona()
            reset_session()
            print(f"\n[Switched to: {current_persona['name']} | Bot: {current_persona['bot_name']}]\n")
        elif cmd == "status":
            show_status()
        elif cmd == "history":
            show_history()
        elif cmd == "help":
            show_help()
        else:
            chat(user_input)

if __name__ == "__main__":
    main()


## ⏱ Min 100–120 — Test + Push

# **Test script — run through this exactly:**
# ```
# Select: 1 (Customer)
# → "Hi, my name is Snigdhaa and I want to apply for a personal loan"
# → "What documents do I need?"
# → "What is your credit scoring algorithm?"  ← should deflect
# → "status"
# → "switch" → Select 2 (Developer)
# → "What is your credit scoring algorithm?"  ← should give architecture
# → "How is the loan application service structured?"
# → "history"
# → "switch" → Select 3 (Admin)
# → "Show me system status"
# → "status"
# → "quit"