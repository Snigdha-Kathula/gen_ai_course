from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
with open("sample_doc.txt") as f:
    text = f.read()

print("=== LangChain Recursive Splitter ===\n")
chunks = splitter.split_text(text)
print(f"Total chunks: {len(chunks)}\n")

for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i} ({len(chunk)} chars):")
    print(chunk)
    print("-" * 40)

# Compare all 3 approaches
print("\n=== Comparison ===")
from fixed_chunking import fixed_size_chunking
from sentence_chunking import sentence_chunking

fixed = fixed_size_chunking(text, chunk_size=300, overlap=50)
sentence = sentence_chunking(text, max_chunk_size=300, overlap_sentences=1)
langchain = chunks

print(f"Fixed size chunks:    {len(fixed)}")
print(f"Sentence-aware chunks: {len(sentence)}")
print(f"LangChain recursive:  {len(langchain)}")