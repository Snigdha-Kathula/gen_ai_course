import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# List all available models
# for model in client.models.list():
#     print(model.name, "->", model.supported_actions)

def get_embedding(sentence:str)-> list[float]:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=sentence
    )
    return result.embeddings[0].values

def cosine_similarity(vec_a:list, vec_b:list):
    dot_product = sum(a*b for a, b in zip(vec_a, vec_b))
    magnitude_a = sum(a**2 for a in vec_a)**0.5
    magnitude_b = sum(b**2 for b in vec_b)**0.5
    return dot_product / (magnitude_a * magnitude_b)


# Test sentences
sentences = [
    "I love programming in Python",
    "Python is my favourite language",
    "I enjoy eating mangoes",
    "Machine learning is fascinating",
    "Deep learning is a subset of machine learning",
    "I had mango juice today",
]

print("=== Embedding dimensions ===")
sample = get_embedding(sentences[0])
print(f"{sentences[0]}")
print(f"Dimensions: {len(sample)}")
print(f"First 5 values: {sample[:5]}")
print("=== Similarity scores ===")
base = sentences[0]
base_vec = get_embedding(sentence=base)
print(f"Base: '{base}'\n")

for sentence in sentences:
    vec = get_embedding(sentence=sentence)
    score = cosine_similarity(base_vec, vec)
    bar = "█" * int(score * 20)
    print(f"{score:.3f} {bar} '{sentence}'")
