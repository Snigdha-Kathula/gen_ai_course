import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def build_prompt(topic: str, age: int)-> str:
    return f""" Explain the {topic} who is {age} years old.
    Use simple language approriate to their age.
    Explain it in 5 sentences"""

def explain_topic(topic:str, age:int) -> str :
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=build_prompt(topic=topic, age=age),
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=300
        )
    )
    return response.text

def main():
    Topic = "What is Gen AI?"
    for age in [5, 15, 30]:
        print(f"- - - - - {age} - - - - -")
        print(explain_topic(topic=Topic, age=age))
    
    while True:
        topic = input("Topic or quit: ").strip()
        if topic.lower() == "quit":
            break
        age = int(input("Age: ").strip())
        print(explain_topic(topic=topic, age=age))

if __name__ == "__main__":
    main()