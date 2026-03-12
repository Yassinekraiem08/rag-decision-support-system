from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def rerank_chunks(query: str, retrieved_chunks: list[tuple[str, str, float]], top_k: int = 3):
    scored = []

    for chunk, filename, score in retrieved_chunks:
        prompt = f"""
You are a retrieval reranker.

Question:
{query}

Candidate chunk:
{chunk}

Rate how relevant this chunk is for answering the question on a scale from 1 to 10.
Return only the number.
"""

        try:
            response = client.chat.completions.create(
                model=os.getenv("LLM_MODEL"),
                messages=[
                    {"role": "system", "content": "You score retrieval relevance."},
                    {"role": "user", "content": prompt}
                ]
            )

            content = response.choices[0].message.content.strip()
            rerank_score = float(content)

        except Exception:
            rerank_score = 0.0

        final_score = score + rerank_score
        scored.append((chunk, filename, final_score))

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]