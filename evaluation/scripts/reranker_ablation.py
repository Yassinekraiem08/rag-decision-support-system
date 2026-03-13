#!/usr/bin/env python3
"""
Reranker Ablation Study

The core question: Does the LLM-based reranker actually improve retrieval quality,
or does it add latency without meaningfully changing which documents are returned?

This is the question practitioners skip. We answer it with data.

Three conditions tested across 25 answerable queries:
  A. No reranker     — hybrid vector scores only, top-3 by score
  B. With reranker   — parallel LLM scoring on top-6, reranked to top-3
  C. Oracle top-3    — best possible P@3 from top-6 pool (upper bound)

Key metrics:
  - Precision@3 (did we get the right documents?)
  - Rank improvement (did reranker move correct docs up?)
  - Rank degradation (did reranker move correct docs DOWN?)
  - Latency cost per condition
  - Agreement rate (how often does reranker change the order?)

Usage:
    python evaluation/scripts/reranker_ablation.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
import time
import statistics
from pathlib import Path
from datetime import datetime

from app.services.pgvector_store import search_chunks_in_db
from app.services.reranker import rerank_chunks


def precision_at_k(retrieved: list, expected: set, k: int) -> float:
    if not expected:
        return 1.0
    hits = len(set(retrieved[:k]) & expected)
    return hits / k


def oracle_precision_at_k(retrieved_pool: list, expected: set, k: int) -> float:
    """Best possible P@3 from the retrieved pool — the upper bound."""
    if not expected:
        return 1.0
    # Put expected docs first
    hits = [f for f in retrieved_pool if f in expected]
    non_hits = [f for f in retrieved_pool if f not in expected]
    oracle_order = hits + non_hits
    return precision_at_k(oracle_order, expected, k)


def run_ablation():
    # Load answerable queries from failure analysis dataset
    with open("evaluation/datasets/failure_analysis_eval.json") as f:
        data = json.load(f)

    queries = [q for q in data["queries"] if q.get("answerable", True) and q.get("expected_sources")]
    # Cap at 25 to control cost
    queries = queries[:25]

    print(f"\n{'='*80}")
    print("RERANKER ABLATION STUDY")
    print("Question: Does the LLM reranker improve retrieval quality?")
    print(f"{'='*80}")
    print(f"Queries: {len(queries)} (answerable, with expected sources)")
    print(f"Conditions: No reranker | With reranker | Oracle upper bound")
    print(f"{'='*80}\n")

    results = []

    for i, q in enumerate(queries, 1):
        question = q["question"]
        expected = set(q["expected_sources"])
        category = q.get("failure_category", "unknown")

        print(f"[{i:02d}/{len(queries)}] {question[:60]}...")

        # Retrieve pool (top-6) — same for all conditions
        t0 = time.time()
        retrieved = search_chunks_in_db(question, top_k=6, use_cache=False)
        retrieval_ms = (time.time() - t0) * 1000

        pool_filenames = [f for _, f, _ in retrieved]

        # ── Condition A: No reranker (top-3 by hybrid score) ──────────────────
        no_rerank_top3 = pool_filenames[:3]
        p3_no_rerank = precision_at_k(no_rerank_top3, expected, 3)

        # ── Condition B: With reranker ─────────────────────────────────────────
        t0 = time.time()
        reranked = rerank_chunks(question, retrieved, top_k=3)
        rerank_ms = (time.time() - t0) * 1000
        reranked_filenames = [f for _, f, _ in reranked]
        p3_reranked = precision_at_k(reranked_filenames, expected, 3)

        # ── Condition C: Oracle upper bound ────────────────────────────────────
        p3_oracle = oracle_precision_at_k(pool_filenames, expected, 3)

        # ── Analysis ───────────────────────────────────────────────────────────
        order_changed = no_rerank_top3 != reranked_filenames
        reranker_helped = p3_reranked > p3_no_rerank
        reranker_hurt = p3_reranked < p3_no_rerank
        reranker_neutral = p3_reranked == p3_no_rerank

        # Rank movement of correct docs
        correct_in_pool = [f for f in pool_filenames if f in expected]
        rank_movements = []
        for doc in correct_in_pool:
            if doc in pool_filenames and doc in reranked_filenames:
                old_rank = pool_filenames.index(doc) + 1
                new_rank = reranked_filenames.index(doc) + 1
                rank_movements.append(old_rank - new_rank)  # positive = moved up

        avg_rank_movement = statistics.mean(rank_movements) if rank_movements else 0

        effect = "HELPED" if reranker_helped else ("HURT" if reranker_hurt else "neutral")
        print(f"  No rerank: P@3={p3_no_rerank:.2f} | Reranked: P@3={p3_reranked:.2f} | Oracle: P@3={p3_oracle:.2f} | {effect} | order_changed={order_changed}")

        results.append({
            "query_id": q["id"],
            "question": question,
            "category": category,
            "expected_sources": list(expected),
            "pool_filenames": pool_filenames,
            "no_rerank": {
                "top3": no_rerank_top3,
                "precision_at_3": p3_no_rerank,
                "latency_ms": retrieval_ms
            },
            "with_reranker": {
                "top3": reranked_filenames,
                "precision_at_3": p3_reranked,
                "latency_ms": retrieval_ms + rerank_ms,
                "rerank_only_ms": rerank_ms
            },
            "oracle": {
                "precision_at_3": p3_oracle
            },
            "order_changed": order_changed,
            "reranker_helped": reranker_helped,
            "reranker_hurt": reranker_hurt,
            "reranker_neutral": reranker_neutral,
            "avg_rank_movement_of_correct_docs": avg_rank_movement
        })

    # ── Aggregate results ──────────────────────────────────────────────────────
    total = len(results)
    helped = sum(1 for r in results if r["reranker_helped"])
    hurt = sum(1 for r in results if r["reranker_hurt"])
    neutral = sum(1 for r in results if r["reranker_neutral"])
    order_changed = sum(1 for r in results if r["order_changed"])

    avg_p3_no_rerank = statistics.mean(r["no_rerank"]["precision_at_3"] for r in results)
    avg_p3_reranked = statistics.mean(r["with_reranker"]["precision_at_3"] for r in results)
    avg_p3_oracle = statistics.mean(r["oracle"]["precision_at_3"] for r in results)

    avg_rerank_latency = statistics.mean(r["with_reranker"]["rerank_only_ms"] for r in results)
    avg_total_no_rerank = statistics.mean(r["no_rerank"]["latency_ms"] for r in results)
    avg_total_reranked = statistics.mean(r["with_reranker"]["latency_ms"] for r in results)

    # Rank movement
    all_movements = [r["avg_rank_movement_of_correct_docs"] for r in results if r["avg_rank_movement_of_correct_docs"] != 0]
    avg_movement = statistics.mean(all_movements) if all_movements else 0

    # P@3 gain/loss per category
    by_category = {}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = {"no_rerank": [], "reranked": [], "oracle": []}
        by_category[cat]["no_rerank"].append(r["no_rerank"]["precision_at_3"])
        by_category[cat]["reranked"].append(r["with_reranker"]["precision_at_3"])
        by_category[cat]["oracle"].append(r["oracle"]["precision_at_3"])

    print(f"\n{'='*80}")
    print("ABLATION RESULTS")
    print(f"{'='*80}")
    print(f"\n{'Condition':<25} {'Avg P@3':>10} {'vs No-Rerank':>14} {'Latency':>10}")
    print(f"{'-'*60}")
    print(f"  {'No Reranker':<23} {avg_p3_no_rerank:>10.3f} {'—':>14} {avg_total_no_rerank:>9.0f}ms")
    print(f"  {'With Reranker':<23} {avg_p3_reranked:>10.3f} {avg_p3_reranked-avg_p3_no_rerank:>+13.3f} {avg_total_reranked:>9.0f}ms")
    print(f"  {'Oracle (upper bound)':<23} {avg_p3_oracle:>10.3f} {avg_p3_oracle-avg_p3_no_rerank:>+13.3f} {'—':>10}")

    print(f"\nReranker effect breakdown ({total} queries):")
    print(f"  Helped  (P@3 improved): {helped:>3} queries ({helped/total:.1%})")
    print(f"  Hurt    (P@3 degraded): {hurt:>3} queries ({hurt/total:.1%})")
    print(f"  Neutral (no P@3 change): {neutral:>3} queries ({neutral/total:.1%})")
    print(f"  Order changed at all:   {order_changed:>3} queries ({order_changed/total:.1%})")

    print(f"\nLatency cost of reranking:")
    print(f"  Reranking alone:       {avg_rerank_latency:.0f}ms avg")
    print(f"  Total without rerank:  {avg_total_no_rerank:.0f}ms avg")
    print(f"  Total with rerank:     {avg_total_reranked:.0f}ms avg")
    print(f"  Latency overhead:      +{avg_rerank_latency:.0f}ms ({avg_rerank_latency/avg_total_no_rerank:.0%} of baseline)")

    print(f"\nP@3 by query category:")
    print(f"  {'Category':<30} {'No Rerank':>10} {'Reranked':>10} {'Delta':>8} {'Oracle':>8}")
    print(f"  {'-'*68}")
    for cat, vals in sorted(by_category.items()):
        nr = statistics.mean(vals["no_rerank"])
        rr = statistics.mean(vals["reranked"])
        oc = statistics.mean(vals["oracle"])
        print(f"  {cat:<30} {nr:>10.3f} {rr:>10.3f} {rr-nr:>+8.3f} {oc:>8.3f}")

    print(f"\nOracle gap analysis:")
    oracle_gap_no_rerank = avg_p3_oracle - avg_p3_no_rerank
    oracle_gap_reranked = avg_p3_oracle - avg_p3_reranked
    print(f"  Gap to oracle (no rerank):   {oracle_gap_no_rerank:.3f}")
    print(f"  Gap to oracle (with rerank): {oracle_gap_reranked:.3f}")
    if oracle_gap_no_rerank > 0:
        gap_closed = (oracle_gap_no_rerank - oracle_gap_reranked) / oracle_gap_no_rerank
        print(f"  Reranker closes oracle gap:  {gap_closed:.1%}")

    print(f"{'='*80}\n")

    # Key insight
    p3_delta = avg_p3_reranked - avg_p3_no_rerank
    if abs(p3_delta) < 0.01:
        insight = f"The reranker is NEUTRAL on P@3 (delta={p3_delta:+.3f}) but costs {avg_rerank_latency:.0f}ms. The ranking order changes {order_changed/total:.0%} of the time, but the set of retrieved documents is already correct — reranking shuffles within an already-good set."
    elif p3_delta > 0:
        insight = f"The reranker IMPROVES P@3 by {p3_delta:.3f} ({p3_delta/avg_p3_no_rerank:.1%} relative) at a cost of {avg_rerank_latency:.0f}ms. Worth the latency for quality-critical applications."
    else:
        insight = f"The reranker HURTS P@3 by {abs(p3_delta):.3f} on this corpus. The hybrid vector scorer is already well-calibrated — LLM reranking introduces noise."

    print(f"KEY INSIGHT: {insight}\n")

    # Save
    output_dir = Path("evaluation/results/ablations")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"reranker_ablation_{timestamp}.json"

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": total,
        "conditions": {
            "no_reranker": {"avg_p3": avg_p3_no_rerank, "avg_latency_ms": avg_total_no_rerank},
            "with_reranker": {"avg_p3": avg_p3_reranked, "avg_latency_ms": avg_total_reranked, "rerank_only_ms": avg_rerank_latency},
            "oracle": {"avg_p3": avg_p3_oracle}
        },
        "reranker_effect": {
            "helped": helped, "hurt": hurt, "neutral": neutral,
            "helped_pct": helped/total, "hurt_pct": hurt/total, "neutral_pct": neutral/total,
            "order_changed_pct": order_changed/total,
            "p3_delta": p3_delta,
            "latency_overhead_ms": avg_rerank_latency
        },
        "oracle_gap": {
            "no_rerank": oracle_gap_no_rerank,
            "with_rerank": oracle_gap_reranked
        },
        "by_category": {
            cat: {
                "avg_p3_no_rerank": statistics.mean(v["no_rerank"]),
                "avg_p3_reranked": statistics.mean(v["reranked"]),
                "avg_p3_oracle": statistics.mean(v["oracle"]),
                "delta": statistics.mean(v["reranked"]) - statistics.mean(v["no_rerank"])
            }
            for cat, v in by_category.items()
        },
        "key_insight": insight,
        "per_query": results
    }

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Results saved to: {output_file}")
    return summary


if __name__ == "__main__":
    run_ablation()
