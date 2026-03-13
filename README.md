# AI Decision Support System — Production RAG with Evaluation

A production-grade RAG system built with rigorous evaluation methodology. Goes beyond typical demos with systematic failure analysis, calibration studies, and measured optimizations.

**[Live Demo](https://rag-decision-support-system.streamlit.app)** | **[API](https://rag-decision-support-api.onrender.com/docs)** | **[Findings](FINDINGS.md)**

---

## What makes this different

Most RAG projects show it working. This one measures *when and why it fails*.

- **50-query failure analysis** across 6 categories — hallucination, domain confusion, multi-hop reasoning, cross-document synthesis, specificity stress, comparative analysis
- **Confidence calibration study** — ECE=0.256, identified bimodal collapse in confidence scoring (first step toward trustworthy uncertainty estimates)
- **Reranker ablation** — neutral on 96% of queries, value concentrated in comparative queries (+0.067 P@3); proposes adaptive reranking for 72% latency reduction
- **Corpus contamination analysis** — keyword boost inflates Gutenberg literary doc scores by up to +10.2 on unanswerable queries; domain tagging fix implemented
- **Hallucination prevention** — raw cosine similarity threshold (0.43) drops hallucination rate 80% → 0%
- **Iterative multi-doc retrieval** — two-pass retrieval for synthesis queries (+4.2% success rate, 0 regressions)
- **Parallel reranking** — ThreadPoolExecutor reduces reranking latency 3,504ms → 848ms (-76%)
- **CI/CD regression detection** — GitHub Actions runs evaluation on every push

---

## Performance

| Metric | Result |
|--------|--------|
| Hallucination rate (after fix) | **0%** (was 80%) |
| Reranking latency | **848ms** (was 3,504ms, -76%) |
| Repeated query latency | **4,242ms** (was 9,938ms, -57%) |
| Iterative vs standard RAG | **+4.2% success rate**, 0 regressions |
| Confidence calibration (ECE) | **0.256** — bimodal collapse identified |
| Documents indexed | **18 docs, 1,673 chunks** |

Full findings with before/after numbers: [`FINDINGS.md`](FINDINGS.md)

---

## Architecture

```
Query → Confidence threshold check → Hybrid retrieval (pgvector + keyword)
      → Iterative 2nd pass (synthesis queries only)
      → Parallel LLM reranking → Answer generation
      → Groundedness verification → Confidence scoring → Response
```

**Stack:** FastAPI · PostgreSQL + pgvector · OpenAI embeddings · GPT-4.1-mini · Streamlit

---

## Quick start

```bash
git clone https://github.com/Yassinekraiem08/rag-decision-support-system
pip install -r requirements.txt

# Set environment variables
OPENAI_API_KEY=...
DATABASE_URL=postgresql://...

python scripts/migrate_add_domain.py   # create tables + domain column
python ingest_folder.py                # embed and index documents
uvicorn app.main:app --reload          # start API
streamlit run demo/chat_demo.py        # start UI
```

---

## Evaluation scripts

```bash
python evaluation/scripts/run_failure_analysis.py       # 50-query stress test
python evaluation/scripts/confidence_calibration.py     # ECE calibration study
python evaluation/scripts/reranker_ablation.py          # reranker impact analysis
python evaluation/scripts/corpus_contamination_analysis.py
python evaluation/scripts/domain_filter_benchmark.py
python evaluation/scripts/benchmark_multi_doc.py
```

---

## Key findings (summary)

| Finding | Result |
|---------|--------|
| Hallucination on unanswerable queries | 80% → 0% after confidence threshold |
| Cross-document synthesis failures | +4.2% with iterative retrieval |
| Reranker impact | Neutral on 96% of queries |
| Confidence calibration | ECE=0.256, bimodal (0.0 or ~0.67, no gradient) |
| Domain contamination | Keyword boost inflates off-topic scores by up to +10.2 |
| Latency bottleneck | Reranking was 36% of total; parallelized to 848ms |

See [`FINDINGS.md`](FINDINGS.md) for full analysis with methodology.
