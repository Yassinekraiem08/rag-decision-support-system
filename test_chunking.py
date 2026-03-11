from app.services.chunking import chunk_text

text = """
Artificial intelligence is transforming many industries.
RAG systems improve LLM responses by retrieving relevant documents first.
Chunking is important because embeddings and retrieval work better on smaller text segments.
"""

chunks = chunk_text(text, chunk_size=10, overlap=3)

print("Number of chunks:", len(chunks))
for i, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {i}:")
    print(repr(chunk))