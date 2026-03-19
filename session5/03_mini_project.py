from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import AIMessage, SystemMessage, HumanMessage

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7
)
summary_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0
)
SYSTEM = "You are Aria, a smart assistant for backend developers. Be concise and practical."
SIZE = 8
EXCEED = 12

messages = [SystemMessage(content=SYSTEM)]
summary = ""
total_turns = 0

def summarise(old_messages):
    conversation=""
    for msg in old_messages:
        if isinstance(msg,HumanMessage):
            conversation = f"User:{msg.content}"
        if isinstance(msg,AIMessage):
            conversation = f"Assistant:{msg.content}"
        prompt = f"""Summarise this conversation in 3-5 sentences.
    Preserve all key facts — names, preferences, technical context.

    Conversation:
    {conversation}"""
    response = summary_llm.invoke(prompt)
    return response.content.strip()


def chat(message:str):
    global messages, summary, total_turns
    total_turns+=1
    messages.append(HumanMessage(content=message))
    if len(messages) > EXCEED:
        old_messages = messages[-SIZE:]
        recent_messages = messages[:-SIZE]
        summary = summarise(old_messages)
        updated_system = f"{SYSTEM}\n\nContext from earlier:\n{summary}"
        messages = [SystemMessage(content=updated_system)] + recent_messages
        print(f"\n[Summary triggered at turn {total_turns}]")
        print(f"[Summary: {summary[:100]}...]\n")
    response = llm.invoke(messages)
    messages.append(AIMessage(response.content))
    return response.content




def main():
    print("type 'status' to look history. type 'quit' to exit the chat")
    while True:
        input_type = input("You: ").strip()
        if input_type.lower() == "quit":
            break
        elif input_type.lower() == "status":
            chat_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
            print(f"\n[Turns: {total_turns} | Messages in memory: {len(chat_msgs)} | Summary exists: {bool(summary)}]\n")
            continue
       
        response = chat(input_type)
        print(f"BOT: {response}")

if __name__ == "__main__":
    main()