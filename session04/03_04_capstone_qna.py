import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def load_document(path:str)-> str:
    with open(path,"r") as f:
        return f.read()

def build_system_prompt(document:str)-> str:
    return f"""You are a precise Q&A assistant. You answer questions strictly based on the document provided below.

    Rules:
    - Only use information from the document to answer
    - If the answer is not in the document, say "This information is not in the document."
    - Be concise — max 3 sentences
    - Never make up information

    Document:
    {document}"""

def main():
    history =[]
    total_tokens=0
    document = load_document("session04/sample.txt")

    while True:
        input_string = input("Type 'quit' to leave\n YOU:").strip()
        if input_string.lower() == "quit" :
            print(f"total Tokens used {total_tokens}")
            break
        
        history.append({
            "role": "user",
            "parts":[{"text": input_string}],
        })

        full_string = ""
        for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents=history[-10:],
            config=types.GenerateContentConfig(
                system_instruction=build_system_prompt(document=document),
                temperature=0.0
            )
        ):
            full_string += chunk.text
            print(chunk.text, end="", flush=True)
        print("\n")
        history.append({
            "role": "model",
            "parts":[{"text": full_string}],
        })
        token_count = client.models.count_tokens(
            model="gemini-2.0-flash",
            contents=history
        )
        total_tokens = token_count.total_tokens
        print(f"Tokens so far: {total_tokens}")


if __name__ == "__main__":
    main()


