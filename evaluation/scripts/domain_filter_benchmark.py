#!/usr/bin/env python3
"""
Domain Filter Benchmark

Measures the P@3 impact of filtering out literary (Gutenberg) documents
from technical query retrieval.

Two conditions for 15 technical queries:
  A. No filter  — all documents searched (current contamination-prone behavior)
  B. domain_filter="technical" — only technical PDFs searched

Key metrics:
  - Contamination rate (Gutenberg doc in top-3)
  - Precision@3 before/after
  - P@3 improvement from domain filtering

Usage:
    python evaluation/scripts/domain_filter_benchmark.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
import statistics
from pathlib import Path
from datetime import datetime

from app.services.pgvector_store import search_chunks_in_db

LITERARY_DOCS = {"pg16713.txt", "pg35398.txt", "pg38304.txt", "pg52091.txt"}

TECHNICAL_QUERIES = [
    {"q": "What is embodied AI?", "expected": ["capa.pdf", "qya.pdf"]},
    {"q": "How do tactile sensors improve robot dexterity?", "expected": ["capa.pdf"]},
    {"q": "What does the robotic control paper say about sensor fusion?", "expected": ["cntrlrbt.pdf"]},
    {"q": "How does deep learning contribute to path planning?", "expected": ["hrb1.pdf"]},
    {"q": "What are the implications of embodied AI for manufacturing?", "expected": ["capa.pdf"]},
    {"q": "How do RAG systems handle multi-hop questions?", "expected": ["rags.pdf"]},
    {"q": "What is the main contribution of the embodied intelligence labs paper?", "expected": ["embodiedintelligencelabs.pdf"]},
    {"q": "What do papers say about human-robot collaboration?", "expected": ["hrb1.pdf", "cntrlrbt.pdf"]},
    {"q": "How does embodied AI differ from screen-based AI in education?", "expected": ["qya.pdf"]},
    {"q": "What carbon pathfinding algorithms are discussed?", "expected": ["carbonpathfinding.pdf"]},
    {"q": "What attention mechanisms are described in the deep learning paper?", "expected": ["ailadeep.pdf"]},
    {"q": "What are the safety considerations for robotic systems?", "expected": ["hrb1.pdf", "cntrlrbt.pdf"]},
    {"q": "How do robotic systems act as collaborative partners in art?", "expected": ["bkbj.pdf"]},
    {"q": "What datasets are used in the language triangulation paper?", "expected": ["triangulang.pdf"]},
    {"q": "What does probing reveal about neural network representations?", "expected": ["probing more.pdf"]},
]


def precision_at_k(retrieved: list, expected: set, k: int) -> float:
    if not expected:
        return 1.0
    hits = len(set(retrieved[:k]) & expected)
    return hits / k


def run_benchmark():
    print(f"\n{'='*80}")
    print("DOMAIN FILTER BENCHMARK")
    print("Question: Does domain filtering improve P@3 for technical queries?")
    print(f"{'='*80}")
    print(f"Queries: {len(TECHNICAL_QUERIES)} technical queries")
    print(f"Conditions: No filter | domain_filter='technical'")
    print(f"{'='*80}\n")

    results = []

    for i, item in enumerate(TECHNICAL_QUERIES, 1):
        query = item["q"]
        expected = set(item["expected"])

        print(f"[{i:02d}/{len(TECHNICAL_QUERIES)}] {query[:65]}")

        # Condition A: No filter
        no_filter = search_chunks_in_db(query, top_k=3, use_cache=False)
        no_filter_files = [f for _, f, _ in no_filter]
        p3_no_filter = precision_at_k(no_filter_files, expected, 3)
        contaminated = any(f in LITERARY_DOCS for f in no_filter_files)

        # Condition B: domain_filter="technical"
        filtered = search_chunks_in_db(query, top_k=3, use_cache=False, domain_filter="technical")
        filtered_files = [f for _, f, _ in filtered]
        p3_filtered = precision_at_k(filtered_files, expected, 3)
        contaminated_after = any(f in LITERARY_DOCS for f in filtered_files)

        status = "CONTAMINATED" if contaminated else "clean"
        effect = "IMPROVED" if p3_filtered > p3_no_filter else ("same" if p3_filtered == p3_no_filter else "worse")
        print(f"  No filter [{status:11s}] P@3={p3_no_filter:.2f} | Filtered P@3={p3_filtered:.2f} | {effect}")

        results.append({
            "query": query,
            "expected": list(expected),
            "no_filter": {
                "top3": no_filter_files,
                "precision_at_3": p3_no_filter,
                "contaminated": contaminated,
            },
            "domain_filtered": {
                "top3": filtered_files,
                "precision_at_3": p3_filtered,
                "contaminated": contaminated_after,
            },
            "p3_delta": p3_filtered - p3_no_filter,
        })

    # Aggregate
    total = len(results)
    contaminated_before = sum(1 for r in results if r["no_filter"]["contaminated"])
    contaminated_after = sum(1 for r in results if r["domain_filtered"]["contaminated"])
    avg_p3_before = statistics.mean(r["no_filter"]["precision_at_3"] for r in results)
    avg_p3_after = statistics.mean(r["domain_filtered"]["precision_at_3"] for r in results)
    delta = avg_p3_after - avg_p3_before

    print(f"\n{'='*80}")
    print("DOMAIN FILTER BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"\n{'Condition':<30} {'Avg P@3':>10} {'Contamination Rate':>20}")
    print(f"{'-'*62}")
    print(f"  {'No domain filter':<28} {avg_p3_before:>10.3f} {contaminated_before}/{total} ({contaminated_before/total:.1%})")
    print(f"  {'domain_filter=technical':<28} {avg_p3_after:>10.3f} {contaminated_after}/{total} ({contaminated_after/total:.1%})")
    print(f"\n  P@3 delta from domain filtering: {delta:+.3f}")
    if avg_p3_before > 0:
        print(f"  Relative improvement: {delta/avg_p3_before:+.1%}")
    print(f"  Contamination eliminated: {contaminated_before - contaminated_after} queries")
    print(f"{'='*80}\n")

    # Key insight
    if contaminated_before == 0:
        insight = "No contamination detected — literary documents did not appear in top-3 for these technical queries. Domain filter has no P@3 impact on this query set."
    elif delta > 0:
        insight = f"Domain filtering improves P@3 by {delta:.3f} ({delta/avg_p3_before:.1%} relative) and eliminates {contaminated_before} contamination event(s). The Gutenberg docs were displacing relevant technical chunks."
    elif delta == 0:
        insight = f"Domain filtering eliminates {contaminated_before} contamination event(s) with zero P@3 impact — the literary documents happened to appear in positions that didn't affect top-k precision."
    else:
        insight = f"Domain filtering reduces contamination but slightly lowers P@3 ({delta:.3f}) — the literary documents may have had high semantic similarity acting as useful context."

    print(f"KEY INSIGHT: {insight}\n")

    # Save results
    output_dir = Path("evaluation/results/domain_filter")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": total,
        "conditions": {
            "no_filter": {"avg_p3": avg_p3_before, "contamination_rate": contaminated_before / total},
            "domain_filtered": {"avg_p3": avg_p3_after, "contamination_rate": contaminated_after / total},
        },
        "p3_delta": delta,
        "contamination_eliminated": contaminated_before - contaminated_after,
        "key_insight": insight,
        "per_query": results,
    }

    output_file = output_dir / f"domain_filter_benchmark_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to: {output_file}")
    return summary


if __name__ == "__main__":
    run_benchmark()
