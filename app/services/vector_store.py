from app.services.embeddings import generate_embedding
from app.services.similarity import cosine_similarity


class InMemoryVectorStore:

    def __init__(self):
        self.vectors = []
        self.texts = []

    def add_chunks(self, chunks: list[str]):
        for chunk in chunks:
            embedding = generate_embedding(chunk)
            self.vectors.append(embedding)
            self.texts.append(chunk)

    def search(self, query: str, top_k: int = 3):
        query_embedding = generate_embedding(query)

        results = []

        for text, vector in zip(self.texts, self.vectors):
            score = cosine_similarity(query_embedding, vector)
            results.append((text, score))

        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]