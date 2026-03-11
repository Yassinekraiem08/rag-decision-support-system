from app.services.embeddings import generate_embedding
from app.services.similarity import cosine_similarity


def retrieve_top_chunks(query: str, chunks: list[str], top_k: int = 3):
    query_embedding = generate_embedding(query)
    scored_chunks = []

    for chunk in chunks:
        chunk_embedding = generate_embedding(chunk)
        score = cosine_similarity(query_embedding, chunk_embedding)
        scored_chunks.append((chunk, score))

    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return scored_chunks[:top_k]