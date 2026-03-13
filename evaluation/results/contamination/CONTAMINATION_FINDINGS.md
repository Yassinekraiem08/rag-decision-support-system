# Corpus Contamination Analysis — Findings

**Generated:** 2026-03-13
**Corpus:** 14 technical AI/robotics papers + 4 Gutenberg literary texts (84.5% of chunks)

---

## The Hypothesis vs The Reality

**Hypothesis:** Gutenberg texts contaminate technical query results via keyword boost.

**Reality (more nuanced):** The contamination effect is query-type dependent, and the keyword boost actually *helps* technical queries while *harming* common-word queries.

---

## Finding 1: Keyword Boost Helps Technical Queries

For specific technical queries (embodied AI, tactile sensors, robotics), the keyword boost improves retrieval:

| Scoring | Avg P@3 | Contamination Rate |
|---------|---------|-------------------|
| Hybrid (vector + keyword) | **0.244** | 0.0% |
| Vector-only | 0.200 | 6.7% |

Counter-intuitive result: Hybrid scoring outperforms vector-only on technical queries (+0.044 P@3) because technical terms appear frequently in technical docs but rarely in Gutenberg texts.

---

## Finding 2: Keyword Boost Dangerously Inflates Scores for Common-Word Queries

The same boost that helps technical queries creates a critical failure for common-word queries.

| Query | Hybrid Score | Raw Vector Score | Inflation |
|-------|-------------|-----------------|-----------|
| "What is the current price of NVIDIA stock?" | 10.40 | 0.204 | +10.20 |
| "What is the recipe for sourdough bread?" | 8.18 | 0.226 | +7.95 |
| "What is two plus two?" | 3.68 | 0.408 | +3.27 |

---

## Finding 3: This Creates a Hidden Confidence Threshold Failure

A naive system using hybrid scores for confidence thresholding would hallucinate on all these queries. Raw cosine similarity separates valid from unanswerable queries cleanly at threshold 0.43. Hybrid scores cannot — the ranges overlap completely.

---

## Finding 4: 84.5% Corpus Imbalance Is the Root Cause

Gutenberg files = 1,455 of 1,722 total chunks (84.5%). More chunks = more keyword matches = higher inflation. Production systems with shared knowledge bases face exactly this.

---

## Fix Implemented

Raw cosine similarity threshold (0.43) bypasses hybrid scores for confidence gating. Hallucination rate: 80% → 0%.

*Standard RAG benchmarks (MS MARCO, NQ, TriviaQA) use homogeneous corpora and are blind to this failure mode.*
