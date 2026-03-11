from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_answer(query: str, retrieved_chunks: list[tuple[str, float]]):
    context = "\n\n".join(
        [f"Source {i+1}: {chunk}" for i, (chunk, score) in enumerate(retrieved_chunks)]
    )

    prompt = f"""
You are a helpful assistant answering questions using only the provided sources.

Question:
{query}

Sources:
{context}

Instructions:
- Answer using only the sources above.
- If the sources do not contain enough information, say so.
- Cite the source numbers in your answer like [Source 1].
"""

    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL"),
        messages=[
            {"role": "system", "content": "You answer questions using retrieved context only."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content