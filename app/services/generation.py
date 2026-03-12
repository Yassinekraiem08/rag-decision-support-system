from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def build_context(retrieved_chunks: list[tuple[str, str, float]]) -> str:
    """Build numbered context for citation"""
    return "\n\n".join(
        [
            f"[{i+1}] {filename}\nContent: {chunk}"
            for i, (chunk, filename, score) in enumerate(retrieved_chunks)
        ]
    )


def generate_answer(query: str, retrieved_chunks: list[tuple[str, str, float]]):
    context = build_context(retrieved_chunks)

    prompt = f"""
You are a helpful assistant answering questions using only the provided sources.

Question:
{query}

Sources:
{context}

Instructions:
- Answer using only the sources above.
- If the sources do not contain enough information, say so.
- Cite sources using numbered brackets like [1], [2], [3] corresponding to the source numbers above.
- Do not make up sources.
"""

    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL"),
        messages=[
            {"role": "system", "content": "You answer questions using retrieved context only."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


def format_answer_with_references(answer: str, retrieved_chunks: list[tuple[str, str, float]]) -> dict:
    """
    Format answer with separate references section.

    Returns:
        {
            "answer": str,  # Original answer with citations
            "references": list[str]  # List of referenced sources
        }
    """
    references = []
    for i, (chunk, filename, score) in enumerate(retrieved_chunks, 1):
        # Create reference entry with filename and score
        references.append(f"[{i}] {filename} (relevance: {score:.3f})")

    return {
        "answer": answer,
        "references": references
    }


def stream_answer(query: str, retrieved_chunks: list[tuple[str, str, float]]):
    context = build_context(retrieved_chunks)

    prompt = f"""
You are a helpful assistant answering questions using only the provided sources.

Question:
{query}

Sources:
{context}

Instructions:
- Answer using only the sources above.
- If the sources do not contain enough information, say so.
- Cite sources using numbered brackets like [1], [2], [3] corresponding to the source numbers above.
- Do not make up sources.
"""

    stream = client.chat.completions.create(
        model=os.getenv("LLM_MODEL"),
        messages=[
            {"role": "system", "content": "You answer questions using retrieved context only."},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta