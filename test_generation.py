from app.utils.loaders import load_text_file
from app.services.chunking import chunk_text
from app.services.retrieval import retrieve_top_chunks
from app.services.generation import generate_answer

text = load_text_file("sample_doc.txt")
chunks = chunk_text(text, chunk_size=20, overlap=5)

query = "How do RAG systems improve answers?"

retrieved = retrieve_top_chunks(query, chunks, top_k=2)
answer = generate_answer(query, retrieved)

print("Retrieved chunks:")
for chunk, score in retrieved:
    print(f"\nScore: {score:.4f}")
    print(chunk)

print("\nGenerated answer:\n")
print(answer)