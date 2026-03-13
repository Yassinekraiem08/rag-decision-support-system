#!/usr/bin/env python3
"""
Latency Profiler

Breaks down pipeline latency per stage to identify bottlenecks.
Measures: embedding → pgvector retrieval → reranking → generation

Usage:
    python evaluation/scripts/latency_profiler.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time
import json
import statistics
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

from app.services.embeddings import generate_embedding
from app.services.pgvector_store import search_chunks_in_db
from app.services.reranker import rerank_chunks
from app.services.generation import generate_answer

load_dotenv()

TEST_QUERIES = [
    "What is embodied AI?",
    "How do tactile sensors improve robot dexterity?",
    "What are the implications of embodied AI for manufacturing supply chains?",
    "Compare approaches to human-robot collaboration across the papers.",
    "What do all papers say about generalization in AI systems?",
]


def profile_pipeline(query: str) -> dict:
    """Profile each stage of the RAG pipeline independently."""
    timings = {}

    # Stage 1: Embedding
    t0 = time.time()
    _ = generate_embedding(query)
    timings["embedding_ms"] = (time.time() - t0) * 1000

    # Stage 2: pgvector retrieval
    t0 = time.time()
    retrieved = search_chunks_in_db(query, top_k=6)
    timings["retrieval_ms"] = (time.time() - t0) * 1000

    # Stage 3: Reranking
    t0 = time.time()
    reranked = rerank_chunks(query, retrieved, top_k=3)
    timings["reranking_ms"] = (time.time() - t0) * 1000

    # Stage 4: Generation
    t0 = time.time()
    _ = generate_answer(query, reranked)
    timings["generation_ms"] = (time.time() - t0) * 1000

    timings["total_ms"] = sum(timings.values())
    timings["query"] = query

    # Percentage breakdown
    for stage in ["embedding_ms", "retrieval_ms", "reranking_ms", "generation_ms"]:
        timings[stage.replace("_ms", "_pct")] = timings[stage] / timings["total_ms"] * 100

    return timings


def main():
    print(f"\n{'='*80}")
    print("PIPELINE LATENCY PROFILER")
    print(f"{'='*80}")
    print(f"Queries: {len(TEST_QUERIES)}")
    print(f"{'='*80}\n")

    all_results = []

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"[{i}/{len(TEST_QUERIES)}] {query[:65]}...")
        result = profile_pipeline(query)
        all_results.append(result)

        print(f"  Embedding:  {result['embedding_ms']:7.0f}ms ({result['embedding_pct']:4.1f}%)")
        print(f"  Retrieval:  {result['retrieval_ms']:7.0f}ms ({result['retrieval_pct']:4.1f}%)")
        print(f"  Reranking:  {result['reranking_ms']:7.0f}ms ({result['reranking_pct']:4.1f}%)  ← bottleneck?")
        print(f"  Generation: {result['generation_ms']:7.0f}ms ({result['generation_pct']:4.1f}%)")
        print(f"  TOTAL:      {result['total_ms']:7.0f}ms\n")

    # Aggregate stats
    stages = ["embedding_ms", "retrieval_ms", "reranking_ms", "generation_ms"]
    print(f"{'='*80}")
    print("AGGREGATE RESULTS (avg across all queries)")
    print(f"{'='*80}")
    print(f"{'Stage':<15} {'Avg (ms)':>10} {'P50 (ms)':>10} {'P95 (ms)':>10} {'% of Total':>12}")
    print(f"{'-'*60}")

    total_avg = sum(statistics.mean(r[s] for r in all_results) for s in stages)
    bottleneck = max(stages, key=lambda s: statistics.mean(r[s] for r in all_results))

    for stage in stages:
        vals = [r[stage] for r in all_results]
        avg = statistics.mean(vals)
        p50 = statistics.median(vals)
        p95 = sorted(vals)[int(len(vals) * 0.95)] if len(vals) > 1 else vals[0]
        pct = avg / total_avg * 100
        marker = " ← BOTTLENECK" if stage == bottleneck else ""
        print(f"  {stage.replace('_ms',''):<13} {avg:>10.0f} {p50:>10.0f} {p95:>10.0f} {pct:>11.1f}%{marker}")

    print(f"  {'TOTAL':<13} {total_avg:>10.0f}")
    print(f"{'='*80}\n")

    # Save results
    output_dir = Path("evaluation/results/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"latency_profile_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "queries": len(TEST_QUERIES),
            "per_query": all_results,
            "bottleneck_stage": bottleneck,
            "avg_total_ms": total_avg
        }, f, indent=2)

    print(f"✅ Profile saved to: {output_file}")
    print(f"🔍 Bottleneck: {bottleneck.replace('_ms', '').upper()} ({statistics.mean(r[bottleneck] for r in all_results):.0f}ms avg)")

    return all_results


if __name__ == "__main__":
    main()
