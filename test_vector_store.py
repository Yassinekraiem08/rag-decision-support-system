from app.utils.loaders import load_text_file
from app.services.chunking import chunk_text
from app.services.vector_store import InMemoryVectorStore

text = load_text_file("sample_doc.txt")
chunks = chunk_text(text, chunk_size=20, overlap=5)

store = InMemoryVectorStore()

print("Indexing document...")
store.add_chunks(chunks)

query = "How do RAG systems improve answers?"

results = store.search(query, top_k=2)

print("\nTop results:")

for chunk, score in results:
    print(f"\nScore: {score:.4f}")
    print(chunk)