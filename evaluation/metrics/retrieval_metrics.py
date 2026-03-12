"""
Retrieval Quality Metrics

Implements standard information retrieval metrics:
- Precision@K: Accuracy of top-K results
- Recall@K: Coverage of relevant docs in top-K
- Mean Reciprocal Rank (MRR): Position of first relevant result
- nDCG@K: Normalized Discounted Cumulative Gain (graded relevance)
- Average Precision: Precision at each relevant position
"""

from typing import List, Set, Dict
import numpy as np


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Precision@K: What fraction of top-K results are relevant?

    Formula: |{retrieved[:k]} ∩ {relevant}| / K

    Args:
        retrieved: Ordered list of retrieved document IDs
        relevant: Set of relevant document IDs
        k: Top-k to evaluate

    Returns:
        Precision@K score (0-1)

    Example:
        >>> precision_at_k(['doc1', 'doc2', 'doc3'], {'doc1', 'doc3'}, 3)
        0.667  # 2 out of 3 are relevant
    """
    if k <= 0 or not retrieved:
        return 0.0

    if not relevant:
        return 1.0  # No relevant docs means we can't be wrong

    top_k = retrieved[:k]
    hits = len(set(top_k) & relevant)
    return hits / k


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Recall@K: What fraction of relevant docs are in top-K?

    Formula: |{retrieved[:k]} ∩ {relevant}| / |{relevant}|

    Args:
        retrieved: Ordered list of retrieved document IDs
        relevant: Set of relevant document IDs
        k: Top-k to evaluate

    Returns:
        Recall@K score (0-1)

    Example:
        >>> recall_at_k(['doc1', 'doc2', 'doc3'], {'doc1', 'doc3', 'doc4'}, 3)
        0.667  # Found 2 out of 3 relevant docs
    """
    if not relevant:
        return 1.0  # No relevant docs to find

    if k <= 0 or not retrieved:
        return 0.0

    top_k = retrieved[:k]
    hits = len(set(top_k) & relevant)
    return hits / len(relevant)


def mean_reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """
    MRR: Reciprocal of the position of the first relevant result.

    Returns: 1/rank if found, else 0.0

    Args:
        retrieved: Ordered list of retrieved document IDs
        relevant: Set of relevant document IDs

    Returns:
        MRR score (0-1)

    Example:
        >>> mean_reciprocal_rank(['doc1', 'doc2', 'doc3'], {'doc3'})
        0.333  # First relevant at position 3 → 1/3
    """
    if not relevant or not retrieved:
        return 0.0

    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i

    return 0.0


def ndcg_at_k(
    retrieved: List[str],
    relevant_with_scores: Dict[str, int],
    k: int
) -> float:
    """
    Normalized Discounted Cumulative Gain: Graded relevance ranking quality.

    nDCG accounts for both position and degree of relevance.

    Args:
        retrieved: Ordered list of retrieved document IDs
        relevant_with_scores: Dict mapping doc_id → relevance score (0-3)
            0 = not relevant
            1 = marginally relevant
            2 = relevant
            3 = highly relevant
        k: Top-k to evaluate

    Returns:
        nDCG@K score (0-1)

    Formula:
        DCG@k = Σ(relevance_i / log2(i + 1)) for i=1 to k
        IDCG@k = DCG of perfect ranking
        nDCG@k = DCG@k / IDCG@k

    Example:
        >>> ndcg_at_k(
        ...     ['doc1', 'doc2', 'doc3'],
        ...     {'doc1': 3, 'doc2': 1, 'doc3': 2},
        ...     3
        ... )
        0.95  # Almost perfect ranking
    """
    if k <= 0 or not retrieved:
        return 0.0

    # Calculate DCG
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k], start=1):
        relevance = relevant_with_scores.get(doc, 0)
        dcg += relevance / np.log2(i + 1)

    # Calculate IDCG (ideal DCG with perfect ranking)
    ideal_scores = sorted(relevant_with_scores.values(), reverse=True)[:k]
    idcg = sum(score / np.log2(i + 1) for i, score in enumerate(ideal_scores, start=1))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Average Precision: Precision at each relevant position, averaged.

    Formula: (1/|relevant|) * Σ P(k) for each relevant k
    Where P(k) = precision at position k (only counted if k is relevant)

    Args:
        retrieved: Ordered list of retrieved document IDs
        relevant: Set of relevant document IDs

    Returns:
        AP score (0-1)

    Example:
        >>> average_precision(
        ...     ['doc1', 'doc2', 'doc3', 'doc4'],
        ...     {'doc1', 'doc3', 'doc4'}
        ... )
        0.83  # High AP because relevant docs are ranked early
    """
    if not relevant or not retrieved:
        return 0.0

    ap = 0.0
    hits = 0

    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            hits += 1
            precision_at_i = hits / i
            ap += precision_at_i

    return ap / len(relevant)


def f1_score(precision: float, recall: float) -> float:
    """
    F1 Score: Harmonic mean of precision and recall.

    Formula: 2 * (precision * recall) / (precision + recall)

    Args:
        precision: Precision score
        recall: Recall score

    Returns:
        F1 score (0-1)
    """
    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def compute_all_retrieval_metrics(
    retrieved: List[str],
    relevant: Set[str],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict:
    """
    Compute all retrieval metrics at multiple K values.

    Args:
        retrieved: Ordered list of retrieved document IDs
        relevant: Set of relevant document IDs
        k_values: List of K values to compute metrics for

    Returns:
        Dict with all metrics
    """
    metrics = {}

    # Compute metrics for each K
    for k in k_values:
        precision = precision_at_k(retrieved, relevant, k)
        recall = recall_at_k(retrieved, relevant, k)
        f1 = f1_score(precision, recall)

        metrics[f"precision@{k}"] = precision
        metrics[f"recall@{k}"] = recall
        metrics[f"f1@{k}"] = f1

    # Single-value metrics
    metrics["mrr"] = mean_reciprocal_rank(retrieved, relevant)
    metrics["average_precision"] = average_precision(retrieved, relevant)

    return metrics


if __name__ == "__main__":
    # Example usage
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc1", "doc3", "doc5"}

    print("Example Retrieval Metrics:")
    print("=" * 60)
    print(f"Retrieved: {retrieved}")
    print(f"Relevant: {relevant}")
    print()

    metrics = compute_all_retrieval_metrics(retrieved, relevant)
    for metric, value in metrics.items():
        print(f"{metric:25s}: {value:.4f}")
