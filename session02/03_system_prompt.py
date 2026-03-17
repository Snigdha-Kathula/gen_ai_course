import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def call_gemini_stream_system(system: str,user_msg:str)-> str:
    for chunk in client.models.generate_content_stream(
        model="gemini-2.0-flash",
        contents=user_msg,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.7
        )
    ):
        print(chunk.text, end="", flush=True)
    print("\n")

question = "How should i structure my python project?"
# Personality 1 Senior Engineer
print( "= = = Senior Engineer = = =")
call_gemini_stream_system(
    system="""You are a senior backend engineer with a experiance of 10 years
    You give practical opinionated advice
    you always mention what not to do. not just what to do
    keep responses under 150 words""",
    user_msg=question
) 

print( "= = = Teaching Assistant = = =")
call_gemini_stream_system(
    system="""You are a patient coding teacher explaining to a beginner
    Use simple words. use analogies
    keep responses under 150 words""",
    user_msg=question
) 
