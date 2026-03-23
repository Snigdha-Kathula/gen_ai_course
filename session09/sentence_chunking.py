import re, os
def sentence_chunking(text:str, max_chunk_size:int, overlap_sentences:int):
#  split into sentences
    sentences =  re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []
    current_chunk = []
    current_size = 0
    for sentence in sentences:
        sentence_size = len(sentence)   
        if current_size + sentence_size > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap_sentences:]
            current_size = sum(len(s) for s in current_chunk)
        current_chunk.append(sentence)
        current_size += sentence_size
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

with open(os.path.join(os.path.dirname(__file__), "sample_doc.txt")) as f:
    text = f.read()

print("=== Sentence-Aware Chunking ===\n")
chunks = sentence_chunking(text, max_chunk_size =300, overlap_sentences=1)
print(f"Total chunks: {len(chunks)}\n")

for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i} ({len(chunk)} chars):")
    print(chunk)
    print("-" * 40)