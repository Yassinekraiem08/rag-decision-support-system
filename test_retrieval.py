from app.services.retrieval import retrieve_top_chunks

chunks = [
    "Artificial intelligence is transforming healthcare and finance.",
    "RAG systems improve language model responses by retrieving relevant context.",
    "Chunking and embeddings are essential parts of semantic search systems.",
    "Pizza dough is made from flour, water, yeast, and salt."
]

query = "How do RAG systems improve answers?"

results = retrieve_top_chunks(query, chunks, top_k=2)

print("Top retrieved chunks:")
for chunk, score in results:
    print(f"\nScore: {score:.4f}")
    print(chunk)