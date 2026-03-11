from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def build_context(retrieved_chunks: list[tuple[str, str, float]]) -> str:
    return "\n\n".join(
        [
            f"Source {i+1} - File: {filename}\nContent: {chunk}"
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
- Cite filenames in your answer, like [sample_doc.txt].
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
- Cite filenames in your answer, like [sample_doc.txt].
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