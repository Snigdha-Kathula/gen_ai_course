import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7
)

# Three personas — same product, different users
PERSONAS = {
    "customer": """You are Lexi, customer support for InCred Financial Services.
- Speak in simple, friendly language
- Focus on loan eligibility, EMIs, repayments
- Never use technical jargon
- Always offer to connect with human support if unsure
- End every response with "Is there anything else I can help you with?" """,

    "developer": """You are DevBot, a technical assistant for InCred engineering team.
- Speak technically — assume Python, PostgreSQL, REST API knowledge
- You can discuss internal architecture, APIs, error codes, and integrations
- Be concise — developers prefer bullet points over paragraphs
- Include code snippets when relevant
- Never discuss customer PII or financial data""",

    "admin": """You are AdminBot, an assistant for InCred operations team.
- You have access to operational context
- Speak professionally and precisely
- You can discuss system status, team workflows, and business metrics
- Always flag if a request requires elevated permissions
- Log all sensitive queries with: [ADMIN LOG: {query_summary}]"""
}

def chat_as_persona(persona_name: str, user_message: str, history: list) -> str:
    system = PERSONAS.get(persona_name, PERSONAS["customer"])

    messages = [SystemMessage(content=system)] + history + [
        HumanMessage(content=user_message)
    ]

    response = llm.invoke(messages)
    return response.content.strip()

def run_persona_demo(persona: str, questions: list) -> None:
    print(f"\n{'='*60}")
    print(f"PERSONA: {persona.upper()}")
    print(f"{'='*60}")
    history = []
    for q in questions:
        print(f"\nUser: {q}")
        reply = chat_as_persona(persona, q, history)
        print(f"Bot:  {reply[:300]}")
        history.append(HumanMessage(content=q))
        history.append(AIMessage(content=reply))

# Same questions, three different personas
questions = [
    "How does the loan approval process work?",
    "What happens when something goes wrong?",
    "Can you give me more details on that?",
]

run_persona_demo("customer", questions)
run_persona_demo("developer", questions)
run_persona_demo("admin", questions)