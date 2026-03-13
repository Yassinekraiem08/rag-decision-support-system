from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _score_chunk(query: str, chunk: str, filename: str, vector_score: float) -> tuple[str, str, float]:
    """Score a single chunk — runs concurrently in parallel reranking."""
    prompt = f"""You are a retrieval reranker.

Question: {query}

Candidate chunk: {chunk[:800]}

Rate relevance 1-10. Return only the number."""

    try:
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL"),
            messages=[
                {"role": "system", "content": "You score retrieval relevance."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=5
        )
        rerank_score = float(response.choices[0].message.content.strip())
    except Exception:
        rerank_score = 0.0

    return (chunk, filename, vector_score + rerank_score)


def rerank_chunks(query: str, retrieved_chunks: list[tuple[str, str, float]], top_k: int = 3):
    """
    Rerank chunks using parallel LLM scoring.
    All chunks are scored concurrently — reduces latency from O(n) to O(1) LLM calls.
    """
    with ThreadPoolExecutor(max_workers=len(retrieved_chunks)) as executor:
        futures = {
            executor.submit(_score_chunk, query, chunk, filename, score): i
            for i, (chunk, filename, score) in enumerate(retrieved_chunks)
        }
        scored = [future.result() for future in as_completed(futures)]

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]