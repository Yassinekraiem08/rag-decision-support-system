from app.utils.loaders import load_text_file
from app.services.chunking import chunk_text
from app.services.pgvector_store import (
    clear_chunks_table,
    store_chunks_in_db,
    search_chunks_in_db,
)

text = load_text_file("sample_doc.txt")
chunks = chunk_text(text, chunk_size=20, overlap=5)

print("Clearing old chunks...")
clear_chunks_table()

print("Storing chunks in PostgreSQL...")
store_chunks_in_db(chunks)

query = "How do RAG systems improve answers?"
results = search_chunks_in_db(query, top_k=2)

print("\nTop results from native pgvector search:")
for chunk, score in results:
    print(f"\nScore: {score:.4f}")
    print(chunk)