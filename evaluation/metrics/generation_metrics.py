"""
Generation Quality Metrics

Evaluates the quality of generated answers:
- LLM-as-judge: Use GPT to evaluate answer correctness
- Semantic similarity: Cosine similarity between embeddings
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
from openai import OpenAI
from typing import Dict
from app.services.embeddings import generate_embedding


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def answer_correctness_llm_judge(generated: str, ground_truth: str) -> Dict:
    """
    Use LLM as a judge to evaluate answer correctness.

    Args:
        generated: Generated answer from RAG system
        ground_truth: Ground truth answer

    Returns:
        {
            "score": float (0-1),
            "reasoning": str,
            "verdict": "CORRECT" | "PARTIALLY_CORRECT" | "INCORRECT"
        }
    """
    if not ground_truth:
        return {
            "score": None,
            "reasoning": "No ground truth available",
            "verdict": "UNABLE_TO_JUDGE"
        }

    prompt = f"""You are evaluating the correctness of a RAG system's answer.

Ground Truth Answer:
{ground_truth}

Generated Answer:
{generated}

Task:
Rate the generated answer's correctness compared to the ground truth on a 0-1 scale:
- 1.0: Completely correct, captures all key points accurately
- 0.7-0.9: Mostly correct, minor omissions or slight inaccuracies
- 0.4-0.6: Partially correct, missing important information or some errors
- 0.0-0.3: Incorrect, mostly wrong or hallucinated

Return ONLY valid JSON with this exact structure:
{{
  "score": 0.85,
  "reasoning": "Brief explanation of the score",
  "verdict": "CORRECT" | "PARTIALLY_CORRECT" | "INCORRECT"
}}

Verdict mapping:
- CORRECT: score >= 0.7
- PARTIALLY_CORRECT: 0.4 <= score < 0.7
- INCORRECT: score < 0.4
"""

    try:
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an objective answer quality evaluator. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ]
        )

        content = response.choices[0].message.content.strip()
        result = json.loads(content)

        # Validate and normalize
        score = float(result.get("score", 0.0))
        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]

        reasoning = result.get("reasoning", "No reasoning provided")
        verdict = result.get("verdict", "INCORRECT")

        # Ensure verdict matches score
        if score >= 0.7:
            verdict = "CORRECT"
        elif score >= 0.4:
            verdict = "PARTIALLY_CORRECT"
        else:
            verdict = "INCORRECT"

        return {
            "score": score,
            "reasoning": reasoning,
            "verdict": verdict
        }

    except Exception as e:
        return {
            "score": 0.0,
            "reasoning": f"Evaluation failed: {str(e)}",
            "verdict": "INCORRECT"
        }


def semantic_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity between two texts using embeddings.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Cosine similarity score (0-1, typically 0.6-1.0 for similar texts)
    """
    if not text1 or not text2:
        return 0.0

    try:
        # Generate embeddings
        emb1 = generate_embedding(text1)
        emb2 = generate_embedding(text2)

        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        magnitude1 = sum(a * a for a in emb1) ** 0.5
        magnitude2 = sum(b * b for b in emb2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        similarity = dot_product / (magnitude1 * magnitude2)

        # Clamp to [0, 1]
        return max(0.0, min(1.0, similarity))

    except Exception as e:
        print(f"Error calculating semantic similarity: {e}")
        return 0.0


def evaluate_answer_quality(
    generated: str,
    ground_truth: str = None,
    use_llm_judge: bool = True
) -> Dict:
    """
    Comprehensive answer quality evaluation.

    Args:
        generated: Generated answer
        ground_truth: Ground truth answer (optional)
        use_llm_judge: Whether to use LLM-as-judge (requires API call)

    Returns:
        Dict with multiple quality metrics
    """
    metrics = {}

    # Semantic similarity (always compute if ground truth exists)
    if ground_truth:
        metrics["semantic_similarity"] = semantic_similarity(generated, ground_truth)

    # LLM-as-judge (optional, requires API call)
    if use_llm_judge and ground_truth:
        llm_eval = answer_correctness_llm_judge(generated, ground_truth)
        metrics["llm_judge_score"] = llm_eval["score"]
        metrics["llm_judge_reasoning"] = llm_eval["reasoning"]
        metrics["llm_judge_verdict"] = llm_eval["verdict"]

    # Basic length metrics
    metrics["answer_length"] = len(generated)
    metrics["answer_word_count"] = len(generated.split())

    return metrics


if __name__ == "__main__":
    # Example usage
    ground_truth = "Embodied AI refers to artificial intelligence systems that have a physical form and interact with the real world."

    good_answer = "Embodied AI is artificial intelligence integrated into physical systems that can interact with and learn from their environment through sensors and actuators."

    bad_answer = "Embodied AI is a type of cloud computing service that runs on virtual machines."

    print("Testing Generation Metrics")
    print("=" * 80)

    print("\nGood Answer:")
    metrics = evaluate_answer_quality(good_answer, ground_truth, use_llm_judge=True)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\nBad Answer:")
    metrics = evaluate_answer_quality(bad_answer, ground_truth, use_llm_judge=True)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
