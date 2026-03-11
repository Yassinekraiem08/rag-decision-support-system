from app.utils.loaders import load_text_file
from app.services.chunking import chunk_text

text = load_text_file("sample_doc.txt")
chunks = chunk_text(text, chunk_size=20, overlap=5)

print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {i}:")
    print(chunk)