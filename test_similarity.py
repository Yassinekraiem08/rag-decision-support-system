from app.services.embeddings import generate_embedding
from app.services.similarity import cosine_similarity

text1 = "Artificial intelligence improves document analysis."
text2 = "AI helps analyze documents more efficiently."
text3 = "Pizza recipes use cheese and tomato sauce."

embedding1 = generate_embedding(text1)
embedding2 = generate_embedding(text2)
embedding3 = generate_embedding(text3)

score_similar = cosine_similarity(embedding1, embedding2)
score_different = cosine_similarity(embedding1, embedding3)

print("Similarity between related sentences:", score_similar)
print("Similarity between unrelated sentences:", score_different)