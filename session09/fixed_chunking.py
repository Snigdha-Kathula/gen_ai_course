import os

def fixed_size_chunking(text:str, chunk_size:int, overlap:int):
    chunks = []
    start = 0
    while start < len(text):
        end = start +chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks

with open(os.path.join(os.path.dirname(__file__), "sample_doc.txt")) as f:
    text = f.read()

print("=== Fixed Size Chunking ===\n")
chunks = fixed_size_chunking(text, chunk_size =200, overlap=50)
print(f"Total chunks: {len(chunks)}\n")

for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i} ({len(chunk)} chars):")
    print(chunk)
    print("-" * 40)