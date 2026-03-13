#!/usr/bin/env python3
"""
Multi-Document Retrieval Benchmark

Compares standard single-pass RAG vs iterative multi-document retrieval
on synthesis and multi-hop queries from the failure analysis dataset.

Usage:
    python evaluation/scripts/benchmark_multi_doc.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
import time
from pathlib import Path
from datetime import datetime

from app.services.pgvector_store import search_chunks_in_db
from app.services.reranker import rerank_chunks
from app.services.multi_doc_retrieval import smart_retrieve, classify_query


def precision_at_k(retrieved: list, expected: set, k: int) -> float:
    if not expected:
        return 1.0
    top_k = retrieved[:k]
    hits = len(set(top_k) & expected)
    return hits / k if k > 0 else 0.0


def run_standard_pipeline(query: str) -> dict:
    """Baseline: single-pass retrieval, top_k=3."""
    start = time.time()
    chunks = search_chunks_in_db(query, top_k=6)
    reranked = rerank_chunks(query, chunks, top_k=3)
    latency = (time.time() - start) * 1000
    filenames = [f for _, f, _ in reranked]
    return {
        "filenames": filenames,
        "unique_docs": len(set(filenames)),
        "latency_ms": latency,
        "strategy": "standard"
    }


def run_smart_pipeline(query: str) -> dict:
    """Iterative retrieval with query classification."""
    start = time.time()
    result = smart_retrieve(query)
    latency = (time.time() - start) * 1000
    filenames = [f for _, f, _ in result["chunks"]]
    return {
        "filenames": filenames,
        "unique_docs": len(set(filenames)),
        "latency_ms": latency,
        "strategy": result["strategy"],
        "metadata": result.get("metadata", {})
    }


def main():
    # Load only synthesis + multi-hop queries from failure analysis dataset
    with open("evaluation/datasets/failure_analysis_eval.json") as f:
        data = json.load(f)

    target_categories = {"cross_document_synthesis", "multi_hop_reasoning", "comparative_analysis"}
    queries = [
        q for q in data["queries"]
        if q.get("failure_category") in target_categories and q.get("answerable", True)
    ]

    print(f"\n{'='*80}")
    print("MULTI-DOCUMENT RETRIEVAL BENCHMARK")
    print(f"{'='*80}")
    print(f"Queries: {len(queries)} (synthesis + multi-hop + comparative)")
    print(f"Comparing: Standard RAG vs Iterative Multi-Doc RAG")
    print(f"{'='*80}\n")

    results = []

    for i, q in enumerate(queries, 1):
        question = q["question"]
        expected = set(q.get("expected_sources", []))
        category = q["failure_category"]
        detected_strategy = classify_query(question)

        print(f"[{i:02d}/{len(queries)}] {question[:65]}...")
        print(f"         Category: {category} | Detected: {detected_strategy}")

        # Run both pipelines
        std = run_standard_pipeline(question)
        smart = run_smart_pipeline(question)

        # Score both
        std_p3 = precision_at_k(std["filenames"], expected, 3)
        smart_p3 = precision_at_k(smart["filenames"], expected, min(3, len(smart["filenames"])))
        smart_hit = len(set(smart["filenames"]) & expected)
        std_hit = len(set(std["filenames"]) & expected)

        std_success = std_hit >= 1 if expected else True
        smart_success = smart_hit >= 1 if expected else True

        delta_docs = smart["unique_docs"] - std["unique_docs"]
        improved = smart_success and not std_success

        print(f"         Standard : P@3={std_p3:.2f} | {std['unique_docs']} docs | {std['latency_ms']:.0f}ms | {'✅' if std_success else '❌'}")
        print(f"         Iterative: P@3={smart_p3:.2f} | {smart['unique_docs']} docs | {smart['latency_ms']:.0f}ms | {'✅' if smart_success else '❌'} {'🔼 IMPROVED' if improved else ''}")
        print()

        results.append({
            "query_id": q["id"],
            "question": question,
            "category": category,
            "detected_strategy": detected_strategy,
            "expected_sources": list(expected),
            "standard": {
                "filenames": std["filenames"],
                "unique_docs": std["unique_docs"],
                "precision_at_3": std_p3,
                "success": std_success,
                "latency_ms": std["latency_ms"]
            },
            "iterative": {
                "filenames": smart["filenames"],
                "unique_docs": smart["unique_docs"],
                "precision_at_3": smart_p3,
                "success": smart_success,
                "latency_ms": smart["latency_ms"],
                "strategy_used": smart["strategy"]
            },
            "delta_unique_docs": delta_docs,
            "improved": improved
        })

    # Summary
    total = len(results)
    std_successes = sum(1 for r in results if r["standard"]["success"])
    smart_successes = sum(1 for r in results if r["iterative"]["success"])
    improvements = sum(1 for r in results if r["improved"])
    regressions = sum(1 for r in results if r["standard"]["success"] and not r["iterative"]["success"])

    std_avg_p3 = sum(r["standard"]["precision_at_3"] for r in results) / total
    smart_avg_p3 = sum(r["iterative"]["precision_at_3"] for r in results) / total
    std_avg_docs = sum(r["standard"]["unique_docs"] for r in results) / total
    smart_avg_docs = sum(r["iterative"]["unique_docs"] for r in results) / total
    std_avg_latency = sum(r["standard"]["latency_ms"] for r in results) / total
    smart_avg_latency = sum(r["iterative"]["latency_ms"] for r in results) / total

    print(f"{'='*80}")
    print("BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"{'Metric':<30} {'Standard RAG':>15} {'Iterative RAG':>15} {'Delta':>10}")
    print(f"{'-'*70}")
    print(f"{'Success Rate':<30} {std_successes/total:>14.1%} {smart_successes/total:>14.1%} {(smart_successes-std_successes)/total:>+9.1%}")
    print(f"{'Avg Precision@3':<30} {std_avg_p3:>15.3f} {smart_avg_p3:>15.3f} {smart_avg_p3-std_avg_p3:>+10.3f}")
    print(f"{'Avg Unique Docs Retrieved':<30} {std_avg_docs:>15.1f} {smart_avg_docs:>15.1f} {smart_avg_docs-std_avg_docs:>+10.1f}")
    print(f"{'Avg Latency (ms)':<30} {std_avg_latency:>15.0f} {smart_avg_latency:>15.0f} {smart_avg_latency-std_avg_latency:>+10.0f}")
    print(f"{'-'*70}")
    print(f"Queries improved by iterative: {improvements}/{total}")
    print(f"Queries regressed:             {regressions}/{total}")
    print(f"{'='*80}\n")

    # Save results
    output_dir = Path("evaluation/results/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"multi_doc_benchmark_{timestamp}.json"

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": total,
        "standard_rag": {
            "success_rate": std_successes / total,
            "avg_precision_at_3": std_avg_p3,
            "avg_unique_docs": std_avg_docs,
            "avg_latency_ms": std_avg_latency
        },
        "iterative_rag": {
            "success_rate": smart_successes / total,
            "avg_precision_at_3": smart_avg_p3,
            "avg_unique_docs": smart_avg_docs,
            "avg_latency_ms": smart_avg_latency
        },
        "improvements": improvements,
        "regressions": regressions,
        "per_query": results
    }

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Benchmark saved to: {output_file}")

    # Print actionable insight
    if smart_successes > std_successes:
        pct_gain = (smart_successes - std_successes) / total * 100
        print(f"\n📈 Iterative retrieval improved success rate by {pct_gain:.0f} percentage points")
        print(f"   on synthesis and multi-hop queries — the hardest RAG failure mode.")
    else:
        print(f"\n📊 Results saved. Iterative retrieval increased document coverage by")
        print(f"   {smart_avg_docs - std_avg_docs:+.1f} docs/query on average.")

    return summary


if __name__ == "__main__":
    main()
