import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7
)

PERSONAS = {
    "1": {
        "name": "Customer",
        "bot_name": "Lexi",
        "system": """You are Lexi, a friendly customer support assistant for InCred Financial Services.
- Use simple, warm language
- Only discuss loan products, EMIs, eligibility, and repayments
- Never reveal internal systems or algorithms
- If unsure, say "Let me connect you with our team at support@incred.com"
- End every response with "Is there anything else I can help you with?" """
    },
    "2": {
        "name": "Developer",
        "bot_name": "DevBot",
        "system": """You are DevBot, technical assistant for InCred engineering team.
    - Assume strong Python, PostgreSQL, REST API knowledge
    - Be concise — use bullet points
    - Include code snippets when helpful
    - You CAN discuss internal architecture, scoring pipeline stages, API contracts, 
    database schema design, and integration patterns
    - You CANNOT discuss exact algorithm weights, model parameters, or customer PII
    - When discussing the credit scoring system, explain it at architecture level:
    data ingestion → feature engineering → model inference → decision engine
    - Never discuss customer personal or financial data"""
    },
    "3": {
        "name": "Admin",
        "bot_name": "AdminBot",
        "system": """You are AdminBot for InCred operations team.
- Speak professionally and precisely
- Discuss system status, workflows, business metrics
- Flag requests needing elevated permissions
- Prefix sensitive query responses with [ADMIN LOG]"""
    }
}

def select_persona() -> dict:
    print("\n=== InCred AI Assistant ===")
    print("Select your role:")
    for key, val in PERSONAS.items():
        print(f"  {key}. {val['name']}")
    
    while True:
        choice = input("\nEnter 1, 2, or 3: ").strip()
        if choice in PERSONAS:
            return PERSONAS[choice]
        print("Invalid choice. Enter 1, 2, or 3.")

def main():
    persona = select_persona()
    history = []
    total_turns = 0

    print(f"\n[Connected as: {persona['name']}]")
    print(f"[Bot: {persona['bot_name']}]")
    print("Type 'switch' to change persona | 'quit' to exit\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "quit":
            print(f"\n{persona['bot_name']}: Goodbye! Have a great day.")
            break
        if user_input.lower() == "switch":
            persona = select_persona()
            history = []  # reset history on persona switch
            total_turns = 0
            print(f"\n[Switched to: {persona['name']}]\n")
            continue
        if not user_input:
            continue

        total_turns += 1
        messages = [SystemMessage(content=persona["system"])] + history[-10:] + [
            HumanMessage(content=user_input)
        ]

        response = llm.invoke(messages)
        reply = response.content.strip()

        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=reply))

        print(f"\n{persona['bot_name']}: {reply}\n")

if __name__ == "__main__":
    main()