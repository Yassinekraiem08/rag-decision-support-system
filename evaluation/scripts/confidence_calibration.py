#!/usr/bin/env python3
"""
Confidence Calibration Study

Core question: When the system reports confidence=0.8, is it actually correct 80% of the time?
Most RAG systems are never tested for this. We test it with data.

Methodology:
- Run all 50 failure-analysis queries through the retrieval + reranking pipeline
- Compute confidence score for each (retrieval quality + chunk consistency)
- Define correctness: answerable queries → ≥1 expected source in top-3
                      unanswerable queries → system correctly refuses (below threshold)
- Bin queries by confidence (5 bins: 0.0-0.2, ..., 0.8-1.0)
- Per bin: compare avg confidence vs actual accuracy
- Compute ECE (Expected Calibration Error) — industry standard calibration metric

ECE = Σ (|confidence - accuracy| × n_bin/n_total)
A perfectly calibrated system has ECE = 0.
Typical uncalibrated neural systems have ECE > 0.10.

Usage:
    python evaluation/scripts/confidence_calibration.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
import statistics
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from app.services.pgvector_store import search_chunks_in_db, get_max_vector_score
from app.services.reranker import rerank_chunks
from app.services.confidence import calculate_confidence


CONFIDENCE_THRESHOLD = 0.43  # Raw vector score threshold for refusal


def is_refused(query: str) -> bool:
    """Returns True if the system would refuse to answer this query."""
    return get_max_vector_score(query) < CONFIDENCE_THRESHOLD


def get_confidence_for_query(query: str) -> tuple[float, list]:
    """
    Run retrieval + reranking + confidence scoring for a query.
    Returns (confidence_score, reranked_chunks).
    Skips LLM generation — uses PARTIALLY_SUPPORTED as verification default
    to isolate retrieval-side calibration signal.
    """
    retrieved = search_chunks_in_db(query, top_k=6, domain_filter="technical")
    if not retrieved:
        return 0.0, []

    reranked = rerank_chunks(query, retrieved, top_k=3)
    result = calculate_confidence(reranked, verification_verdict="PARTIALLY_SUPPORTED")
    return result["confidence"], reranked


def is_correct(reranked: list, expected_sources: set, answerable: bool, refused: bool) -> bool:
    """
    Ground truth correctness:
    - Unanswerable: correct iff system refused
    - Answerable: correct iff ≥1 expected source in top-3
    """
    if not answerable:
        return refused
    if refused:
        return False  # Should have answered but didn't
    retrieved_files = {f for _, f, _ in reranked}
    return bool(retrieved_files & expected_sources)


def compute_ece(bins: list[dict]) -> float:
    """Expected Calibration Error across all bins."""
    total = sum(b["count"] for b in bins)
    if total == 0:
        return 0.0
    ece = sum(
        abs(b["avg_confidence"] - b["accuracy"]) * (b["count"] / total)
        for b in bins
        if b["count"] > 0
    )
    return ece


def run_calibration():
    with open("evaluation/datasets/failure_analysis_eval.json") as f:
        data = json.load(f)

    queries = data["queries"]

    print(f"\n{'='*80}")
    print("CONFIDENCE CALIBRATION STUDY")
    print("Question: Does the system's confidence score predict its correctness?")
    print(f"{'='*80}")
    print(f"Queries: {len(queries)} | Bins: 5 | Metric: ECE (Expected Calibration Error)")
    print(f"{'='*80}\n")

    results = []

    for i, q in enumerate(queries, 1):
        question = q["question"]
        answerable = q.get("answerable", True)
        expected = set(q.get("expected_sources") or [])
        category = q.get("failure_category", "unknown")

        print(f"[{i:02d}/{len(queries)}] {question[:65]}")

        # Step 1: Refusal check (cheap — one embedding call)
        refused = is_refused(question)

        if refused:
            confidence = 0.0
            reranked = []
            correct = is_correct(reranked, expected, answerable, refused=True)
            print(f"  REFUSED | correct={correct} (answerable={answerable})")
        else:
            # Step 2: Full retrieval + reranking + confidence
            confidence, reranked = get_confidence_for_query(question)
            correct = is_correct(reranked, expected, answerable, refused=False)
            retrieved_files = [f for _, f, _ in reranked]
            status = "CORRECT" if correct else "WRONG"
            print(f"  conf={confidence:.3f} | {status} | top3={retrieved_files}")

        results.append({
            "id": q["id"],
            "question": question,
            "category": category,
            "answerable": answerable,
            "expected_sources": list(expected),
            "refused": refused,
            "confidence": confidence,
            "correct": correct,
        })

    # ── Binning ─────────────────────────────────────────────────────────────────
    bin_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    bin_labels = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
    bin_data = defaultdict(list)

    for r in results:
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= r["confidence"] < bin_edges[i + 1]:
                bin_data[bin_labels[i]].append(r)
                break

    bins = []
    for label in bin_labels:
        items = bin_data[label]
        if not items:
            bins.append({"label": label, "count": 0, "avg_confidence": 0.0, "accuracy": 0.0})
            continue
        avg_conf = statistics.mean(r["confidence"] for r in items)
        accuracy = sum(1 for r in items if r["correct"]) / len(items)
        bins.append({
            "label": label,
            "count": len(items),
            "avg_confidence": avg_conf,
            "accuracy": accuracy,
            "gap": avg_conf - accuracy,  # positive = overconfident, negative = underconfident
        })

    ece = compute_ece(bins)
    total = len(results)
    overall_accuracy = sum(1 for r in results if r["correct"]) / total
    overall_confidence = statistics.mean(r["confidence"] for r in results)

    # ── Print results ────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("CALIBRATION RESULTS")
    print(f"{'='*80}")
    print(f"\n{'Bin':<12} {'Count':>6} {'Avg Conf':>10} {'Accuracy':>10} {'Gap':>8} {'Status':>14}")
    print(f"  {'-'*62}")
    for b in bins:
        if b["count"] == 0:
            print(f"  {b['label']:<10} {'0':>6} {'—':>10} {'—':>10} {'—':>8}")
            continue
        gap = b["avg_confidence"] - b["accuracy"]
        status = "overconfident" if gap > 0.05 else ("underconfident" if gap < -0.05 else "calibrated")
        print(f"  {b['label']:<10} {b['count']:>6} {b['avg_confidence']:>10.3f} {b['accuracy']:>10.3f} {gap:>+8.3f}  {status}")

    print(f"\n  Overall accuracy:   {overall_accuracy:.3f} ({sum(1 for r in results if r['correct'])}/{total})")
    print(f"  Overall confidence: {overall_confidence:.3f}")
    print(f"  Confidence gap:     {overall_confidence - overall_accuracy:+.3f}")
    print(f"\n  ECE (Expected Calibration Error): {ece:.4f}")

    # Interpret ECE
    if ece < 0.05:
        ece_verdict = "WELL-CALIBRATED (ECE < 0.05) — confidence scores are reliable"
    elif ece < 0.10:
        ece_verdict = "MODERATELY CALIBRATED (0.05 ≤ ECE < 0.10) — usable but imperfect"
    else:
        ece_verdict = "POORLY CALIBRATED (ECE ≥ 0.10) — confidence scores mislead users"
    print(f"  Verdict: {ece_verdict}")

    # ── Category breakdown ───────────────────────────────────────────────────────
    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    print(f"\nCalibration by query category:")
    print(f"  {'Category':<30} {'Avg Conf':>10} {'Accuracy':>10} {'Gap':>8} {'n':>4}")
    print(f"  {'-'*66}")
    for cat, items in sorted(by_category.items()):
        avg_c = statistics.mean(r["confidence"] for r in items)
        acc = sum(1 for r in items if r["correct"]) / len(items)
        gap = avg_c - acc
        print(f"  {cat:<30} {avg_c:>10.3f} {acc:>10.3f} {gap:>+8.3f} {len(items):>4}")

    # ── Worst-calibrated queries ─────────────────────────────────────────────────
    print(f"\nMost overconfident queries (high confidence, wrong answer):")
    overconfident = sorted(
        [r for r in results if not r["correct"] and r["confidence"] >= 0.4],
        key=lambda x: x["confidence"],
        reverse=True
    )[:5]
    for r in overconfident:
        print(f"  conf={r['confidence']:.3f} | {r['question'][:65]}")

    print(f"\nMost underconfident queries (low confidence, correct answer):")
    underconfident = sorted(
        [r for r in results if r["correct"] and r["confidence"] < 0.6],
        key=lambda x: x["confidence"]
    )[:5]
    for r in underconfident:
        print(f"  conf={r['confidence']:.3f} | {r['question'][:65]}")

    print(f"\n{'='*80}")

    # ── Key insight ──────────────────────────────────────────────────────────────
    gap = overall_confidence - overall_accuracy
    if ece < 0.05:
        insight = (
            f"The confidence scorer is well-calibrated (ECE={ece:.4f}). "
            f"When the system reports high confidence, it is correct at a matching rate. "
            f"This means the confidence score can be trusted as a reliability signal."
        )
    elif gap > 0.05:
        insight = (
            f"The system is overconfident (ECE={ece:.4f}, gap={gap:+.3f}). "
            f"It reports higher confidence than its actual accuracy warrants. "
            f"This is the most common failure mode in RAG systems — the retrieval score "
            f"looks high, but the retrieved documents don't fully answer the question."
        )
    elif gap < -0.05:
        insight = (
            f"The system is underconfident (ECE={ece:.4f}, gap={gap:+.3f}). "
            f"It is more accurate than its confidence scores suggest. "
            f"The scoring weights may be overly conservative."
        )
    else:
        insight = (
            f"The system is reasonably calibrated (ECE={ece:.4f}, gap={gap:+.3f}). "
            f"Confidence scores track accuracy well overall."
        )

    print(f"KEY INSIGHT: {insight}\n")

    # ── Save ─────────────────────────────────────────────────────────────────────
    output_dir = Path("evaluation/results/calibration")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": total,
        "overall": {
            "accuracy": overall_accuracy,
            "avg_confidence": overall_confidence,
            "confidence_gap": overall_confidence - overall_accuracy,
            "ece": ece,
            "ece_verdict": ece_verdict,
        },
        "bins": bins,
        "by_category": {
            cat: {
                "count": len(items),
                "avg_confidence": statistics.mean(r["confidence"] for r in items),
                "accuracy": sum(1 for r in items if r["correct"]) / len(items),
                "gap": statistics.mean(r["confidence"] for r in items) - sum(1 for r in items if r["correct"]) / len(items),
            }
            for cat, items in by_category.items()
        },
        "key_insight": insight,
        "per_query": results,
    }

    output_file = output_dir / f"confidence_calibration_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to: {output_file}")
    return summary


if __name__ == "__main__":
    run_calibration()
