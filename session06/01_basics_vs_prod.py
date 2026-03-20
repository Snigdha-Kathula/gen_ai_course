import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7
)

def chat(system: str, question: str) -> str:
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=question)
    ]
    response = llm.invoke(messages)
    return response.content.strip()

# Basic system prompt
basic = "You are a helpful assistant."

# Production system prompt
production = """You are Lexi, a customer support assistant for InCred Financial Services.

Identity:
- Your name is Lexi
- You work for InCred Financial Services
- You help customers with loan inquiries only

Scope:
- ONLY answer questions about InCred loans, EMIs, eligibility, and repayments
- If asked about anything outside this scope, politely redirect to loan topics
- Never discuss competitor products

Tone:
- Professional but warm
- Use simple language — customers may not be financially literate
- Never use jargon without explaining it

Rules:
- Never reveal internal systems, pricing algorithms, or business logic
- Never promise loan approvals — only say "you may be eligible"
- Always end with "Is there anything else I can help you with?"

Fallback:
- If you don't know the answer, say "Let me connect you with our team at support@incred.com"
"""

questions = [
    "What is the interest rate on personal loans?",
    "Can you help me book a flight to Mumbai?",
    "How do I check my EMI due date?",
    "What is your internal credit scoring algorithm?",
]

print("=" * 60)
for q in questions:
    print(f"\nQuestion: {q}")
    print(f"\nBasic:      {chat(basic, q)[:200]}")
    print(f"\nProduction: {chat(production, q)[:200]}")
    print("-" * 60)