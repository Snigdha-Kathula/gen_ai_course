import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
EXCEED = 10
SIZE = 6
history =[]
summary=[]

def summarise_history(old_msgs:list)-> str:
    only_text =""
    for each_msg in old_msgs:
        role="User" if each_msg["role"] == "user" else "Assistant"
        only_text += f"{role}: {each_msg['parts'][0]['text']}\n"

        prompt = f"""Summarise this conversation in 3-5 sentences.
        Preserve all key facts — names, preferences, context mentioned.
        Be concise.

        Conversation:
        {only_text}"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0)
    )
    return response.text.strip()


def build_system_with_summary(summary:str):
    if not summary:
        return "You are an helpful assistant. Remember all the facts that user tells you"
    return f"""You are an helpful assistant. Remember all the facts that user tells you
    summary:{summary}"""

def chat(msg:str)-> str:
    global summary, history
    history.append({
        "role":"user",
        "parts":[{"text":msg}]
    })
    if len(history) > EXCEED:
        old_messages = history[:-SIZE]
        recent_messages = history[-SIZE:]
        summary = summarise_history(old_messages)
        history = recent_messages

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction=build_system_with_summary(summary),
            temperature=0.7
        )
    )
    resply= response.text.strip()
    history.append({
        "role":"model",
        "parts":[{"text":resply}]
    })
    return resply


def main():
    print("=== Chatbot with Summary Memory ===\n")

    # Feed it lots of facts
    messages = [
        "My name is Snigdhaa.",
        "I work at InCred as a backend developer.",
        "I am building a loan management system.",
        "My tech stack is Python, PostgreSQL and Redis.",
        "I prefer concise answers.",
        "My team has 5 engineers.",
        "We deploy using Docker and Kubernetes.",
        "I have been coding for 4 years.",
        "I love working on distributed systems.",
        "My manager's name is Rahul.",
        # These next ones will trigger summary
        "What is my name?",
        "Where do I work?",
        "What is my tech stack?",
        "Who is my manager?",
    ]  
    for msg in messages:
        print(f"You: {msg}")
        reply = chat(msg)
        print(f"Bot:{reply}")
        print(f"History Length ={len(history)}")
      

if __name__ == "__main__":
    main()
