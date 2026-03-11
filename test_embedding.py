from app.services.embeddings import generate_embedding

text = "Artificial intelligence helps analyze documents."

embedding = generate_embedding(text)

print("Embedding length:", len(embedding))
print("First 10 values:", embedding[:10])