from openai import OpenAI
import os
from dotenv import load_dotenv
from app.services.cache import TTLCache

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedding_cache = TTLCache(ttl_seconds=3600)


def generate_embedding(text: str):
    cached = embedding_cache.get(text)
    if cached is not None:
        return cached

    response = client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL"),
        input=text
    )
    embedding = response.data[0].embedding
    embedding_cache.set(text, embedding)
    return embedding


def generate_embeddings(texts: list[str]):
    uncached = []
    uncached_indices = []
    results = [None] * len(texts)

    for i, text in enumerate(texts):
        cached = embedding_cache.get(text)
        if cached is not None:
            results[i] = cached
        else:
            uncached.append(text)
            uncached_indices.append(i)

    if uncached:
        response = client.embeddings.create(
            model=os.getenv("EMBEDDING_MODEL"),
            input=uncached
        )
        new_embeddings = [item.embedding for item in response.data]

        for idx, text, embedding in zip(uncached_indices, uncached, new_embeddings):
            results[idx] = embedding
            embedding_cache.set(text, embedding)

    return results