import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def verify_answer(question: str, answer: str, retrieved_chunks: list[tuple[str, str, float]]) -> dict:
    """
    Verifies whether the generated answer is supported by the retrieved chunks.

    Args:
        question: The user's question.
        answer: The generated answer.
        retrieved_chunks: List of tuples in the format (chunk_content, filename, score).

    Returns:
        dict with:
            - verdict: SUPPORTED | PARTIALLY_SUPPORTED | UNSUPPORTED
            - reason: short explanation
    """
    context = "\n\n".join(
        [
            f"File: {filename}\nScore: {score:.4f}\nContent: {chunk}"
            for chunk, filename, score in retrieved_chunks
        ]
    )

    system_prompt = (
        "You are a groundedness verifier for a Retrieval-Augmented Generation system. "
        "Your job is to determine whether the answer is supported by the retrieved sources.\n\n"
        "You must return valid JSON only with this exact schema:\n"
        '{\n'
        '  "verdict": "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED",\n'
        '  "reason": "short explanation"\n'
        '}\n\n'
        "Decision rules:\n"
        "- SUPPORTED: All important claims in the answer are directly supported by the sources.\n"
        "- PARTIALLY_SUPPORTED: Some of the answer is supported, but at least one important claim is weak, inferred, or missing from the sources.\n"
        "- UNSUPPORTED: The answer is mostly unsupported, contradicted, or invents claims not found in the sources.\n"
        "Keep the reason concise and factual."
    )

    user_prompt = f"""
Question:
{question}

Answer:
{answer}

Retrieved Sources:
{context}
"""

    try:
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL"),
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)

        verdict = parsed.get("verdict", "UNSUPPORTED")
        reason = parsed.get("reason", "No reason provided.")

        allowed_verdicts = {"SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"}
        if verdict not in allowed_verdicts:
            verdict = "UNSUPPORTED"
            reason = "Verifier returned an invalid verdict."

        return {
            "verdict": verdict,
            "reason": reason,
        }

    except Exception as e:
        return {
            "verdict": "UNSUPPORTED",
            "reason": f"Verification failed: {str(e)}",
        }