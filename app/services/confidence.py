"""
Confidence scoring for RAG answers.

Calculates a multi-factor confidence score combining:
- Retrieval quality (average chunk scores)
- Verification quality (SUPPORTED/PARTIALLY/UNSUPPORTED)
- Consistency (semantic similarity between retrieved chunks)
"""

import numpy as np


def calculate_confidence(
    retrieved_chunks: list[tuple[str, str, float]],
    verification_verdict: str,
) -> dict:
    """
    Calculate confidence score for a RAG answer.

    Args:
        retrieved_chunks: List of (content, filename, score) tuples
        verification_verdict: One of SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED

    Returns:
        {
            "confidence": float (0-1),
            "reasoning": str,
            "breakdown": {
                "retrieval_quality": float,
                "verification_quality": float,
                "consistency": float
            }
        }
    """
    # 1. Calculate retrieval quality (average of chunk scores)
    if not retrieved_chunks:
        retrieval_quality = 0.0
    else:
        retrieval_quality = np.mean([score for _, _, score in retrieved_chunks])
        # Normalize to 0-1 (scores can be > 1 due to keyword boosting)
        retrieval_quality = min(retrieval_quality, 1.0)

    # 2. Calculate verification quality
    verification_map = {
        "SUPPORTED": 1.0,
        "PARTIALLY_SUPPORTED": 0.5,
        "UNSUPPORTED": 0.0
    }
    verification_quality = verification_map.get(verification_verdict, 0.0)

    # 3. Calculate consistency (how similar are the chunks to each other?)
    # High consistency means chunks discuss the same topic
    consistency = calculate_chunk_consistency(retrieved_chunks)

    # 4. Weighted combination
    confidence = (
        0.5 * retrieval_quality +
        0.3 * verification_quality +
        0.2 * consistency
    )

    # 5. Generate reasoning
    reasoning = generate_reasoning(
        retrieval_quality,
        verification_quality,
        consistency,
        verification_verdict
    )

    return {
        "confidence": float(confidence),
        "reasoning": reasoning,
        "breakdown": {
            "retrieval_quality": float(retrieval_quality),
            "verification_quality": float(verification_quality),
            "consistency": float(consistency)
        }
    }


def calculate_chunk_consistency(retrieved_chunks: list[tuple[str, str, float]]) -> float:
    """
    Calculate semantic consistency between retrieved chunks.

    High consistency = chunks are semantically similar (discuss same topic)
    Low consistency = chunks are disparate (may indicate retrieval issues)

    Uses simple word overlap for efficiency. Could be improved with embeddings.
    """
    if len(retrieved_chunks) < 2:
        return 1.0  # Single chunk is perfectly consistent with itself

    # Calculate pairwise word overlap
    chunks_words = [set(chunk.lower().split()) for chunk, _, _ in retrieved_chunks]

    similarities = []
    for i in range(len(chunks_words)):
        for j in range(i + 1, len(chunks_words)):
            # Jaccard similarity
            intersection = len(chunks_words[i] & chunks_words[j])
            union = len(chunks_words[i] | chunks_words[j])
            if union > 0:
                sim = intersection / union
                similarities.append(sim)

    if not similarities:
        return 0.5  # Default middle value

    return float(np.mean(similarities))


def generate_reasoning(
    retrieval_quality: float,
    verification_quality: float,
    consistency: float,
    verdict: str
) -> str:
    """Generate human-readable reasoning for confidence score."""

    parts = []

    # Retrieval quality
    if retrieval_quality >= 0.8:
        parts.append("high retrieval scores")
    elif retrieval_quality >= 0.6:
        parts.append("moderate retrieval scores")
    else:
        parts.append("low retrieval scores")

    # Verification
    if verdict == "SUPPORTED":
        parts.append("fully supported answer")
    elif verdict == "PARTIALLY_SUPPORTED":
        parts.append("partially supported answer")
    else:
        parts.append("unsupported answer")

    # Consistency
    if consistency >= 0.5:
        parts.append("consistent sources")
    else:
        parts.append("inconsistent sources")

    reasoning = f"{parts[0].capitalize()}, {parts[1]}, and {parts[2]}."

    return reasoning
