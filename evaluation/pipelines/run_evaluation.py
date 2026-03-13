#!/usr/bin/env python3
"""
RAG Evaluation Pipeline

Runs comprehensive evaluation on the RAG system using a curated test dataset.

Usage:
    python evaluation/pipelines/run_evaluation.py
    python evaluation/pipelines/run_evaluation.py --quick  # Run on subset for testing
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Import RAG system components
from app.services.pgvector_store import search_chunks_in_db
from app.services.reranker import rerank_chunks
from app.services.generation import generate_answer
from app.services.verifier import verify_answer


class EvaluationRunner:
    """Main evaluation orchestrator"""

    def __init__(self, dataset_path: str, output_dir: str):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset
        with open(self.dataset_path) as f:
            data = json.load(f)
            self.queries = data["queries"]
            self.metadata = data.get("metadata", {})

        self.results = []

    def run(self, quick: bool = False) -> Dict:
        """
        Run evaluation on the dataset.

        Args:
            quick: If True, only run on first 3 queries for testing

        Returns:
            Summary dict with aggregate metrics
        """
        queries_to_eval = self.queries[:5] if quick else self.queries

        print(f"\n{'='*80}")
        print(f"STARTING EVALUATION")
        print(f"{'='*80}")
        print(f"Dataset: {self.dataset_path.name}")
        print(f"Queries: {len(queries_to_eval)} {'(quick mode)' if quick else ''}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*80}\n")

        for i, query_item in enumerate(queries_to_eval, 1):
            print(f"[{i}/{len(queries_to_eval)}] Evaluating: {query_item['question'][:60]}...")
            result = self.evaluate_query(query_item)
            self.results.append(result)

            # Print quick status
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"  {status} - P@3: {result['metrics']['precision_at_3']:.3f}\n")

        # Compute summary
        summary = self.compute_summary()

        # Save results
        self.save_results(summary)

        # Print final summary
        self.print_summary(summary)

        return summary

    def evaluate_query(self, query_item: Dict) -> Dict:
        """Evaluate a single query"""
        query_id = query_item["id"]
        question = query_item["question"]
        expected_sources = set(query_item.get("expected_sources", []))
        ground_truth_answer = query_item.get("ground_truth_answer")

        # Start timing
        start_time = time.time()

        try:
            # Run RAG pipeline
            retrieved = search_chunks_in_db(question, top_k=6)
            reranked = rerank_chunks(question, retrieved, top_k=3)
            answer = generate_answer(question, reranked)
            verification = verify_answer(question, answer, reranked)

            latency = time.time() - start_time

            # Extract retrieved filenames
            retrieved_filenames = [filename for _, filename, _ in reranked]

            # Basic metrics
            precision_at_3 = self.precision_at_k(retrieved_filenames, expected_sources, 3)

            # Check success: at least 1 expected source retrieved in top-3
            num_correct = len(set(retrieved_filenames[:3]) & expected_sources)
            is_correct = num_correct >= 1 if expected_sources else True

            return {
                "query_id": query_id,
                "question": question,
                "difficulty": query_item.get("difficulty"),
                "category": query_item.get("category"),
                "metrics": {
                    "precision_at_3": precision_at_3,
                    "num_expected": len(expected_sources),
                    "num_retrieved_correct": len(set(retrieved_filenames) & expected_sources)
                },
                "latency_ms": latency * 1000,
                "answer": answer,
                "verification": verification,
                "retrieved_filenames": retrieved_filenames,
                "expected_sources": list(expected_sources),
                "success": is_correct,
                "error": None
            }

        except Exception as e:
            return {
                "query_id": query_id,
                "question": question,
                "difficulty": query_item.get("difficulty"),
                "category": query_item.get("category"),
                "metrics": {
                    "precision_at_3": 0.0,
                    "num_expected": len(expected_sources),
                    "num_retrieved_correct": 0
                },
                "latency_ms": 0,
                "answer": None,
                "verification": None,
                "retrieved_filenames": [],
                "expected_sources": list(expected_sources),
                "success": False,
                "error": str(e)
            }

    @staticmethod
    def precision_at_k(retrieved: List[str], expected: set, k: int) -> float:
        """Calculate Precision@K"""
        if not expected:
            return 1.0  # No expected sources means we can't fail

        top_k = retrieved[:k]
        hits = len(set(top_k) & expected)
        return hits / k if k > 0 else 0.0

    def compute_summary(self) -> Dict:
        """Compute aggregate metrics across all results"""
        if not self.results:
            return {}

        total_queries = len(self.results)
        successful = sum(1 for r in self.results if r["success"])
        failed = sum(1 for r in self.results if not r["success"])

        # Aggregate metrics
        avg_precision = sum(r["metrics"]["precision_at_3"] for r in self.results) / total_queries
        avg_latency = sum(r["latency_ms"] for r in self.results) / total_queries

        # By difficulty
        by_difficulty = {}
        for result in self.results:
            diff = result.get("difficulty", "unknown")
            if diff not in by_difficulty:
                by_difficulty[diff] = {"count": 0, "success": 0, "precision": []}
            by_difficulty[diff]["count"] += 1
            if result["success"]:
                by_difficulty[diff]["success"] += 1
            by_difficulty[diff]["precision"].append(result["metrics"]["precision_at_3"])

        # Calculate success rate by difficulty
        difficulty_stats = {}
        for diff, stats in by_difficulty.items():
            difficulty_stats[diff] = {
                "count": stats["count"],
                "success_rate": stats["success"] / stats["count"],
                "avg_precision_at_3": sum(stats["precision"]) / len(stats["precision"])
            }

        return {
            "timestamp": datetime.now().isoformat(),
            "dataset": str(self.dataset_path),
            "total_queries": total_queries,
            "successful": successful,
            "failed": failed,
            "overall_accuracy": successful / total_queries,
            "avg_precision_at_3": avg_precision,
            "avg_latency_ms": avg_latency,
            "by_difficulty": difficulty_stats,
            "corpus_size": self.metadata.get("corpus_size", "unknown")
        }

    def save_results(self, summary: Dict):
        """Save detailed results and summary"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        detailed_file = self.output_dir / f"results_{timestamp}.json"
        with open(detailed_file, "w") as f:
            json.dump({
                "summary": summary,
                "per_query_results": self.results
            }, f, indent=2)

        print(f"\n✅ Detailed results saved to: {detailed_file}")

        # Save latest summary (for easy access)
        latest_file = self.output_dir / "latest_summary.json"
        with open(latest_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"✅ Latest summary saved to: {latest_file}")

        # Append to history
        history_file = self.output_dir / "evaluation_history.jsonl"
        with open(history_file, "a") as f:
            f.write(json.dumps(summary) + "\n")

        print(f"✅ Appended to history: {history_file}")

    def print_summary(self, summary: Dict):
        """Print formatted summary"""
        print(f"\n{'='*80}")
        print("EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"Overall Accuracy:     {summary['overall_accuracy']:.2%}")
        print(f"Avg Precision@3:      {summary['avg_precision_at_3']:.3f}")
        print(f"Avg Latency:          {summary['avg_latency_ms']:.0f}ms")
        print(f"Successful:           {summary['successful']}/{summary['total_queries']}")
        print(f"Failed:               {summary['failed']}/{summary['total_queries']}")

        if summary.get("by_difficulty"):
            print(f"\nBy Difficulty:")
            for diff, stats in summary["by_difficulty"].items():
                print(f"  {diff:10s}: {stats['success_rate']:.1%} success, "
                      f"P@3={stats['avg_precision_at_3']:.3f} "
                      f"({stats['count']} queries)")

        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument(
        "--dataset",
        default="evaluation/datasets/base_eval.json",
        help="Path to evaluation dataset"
    )
    parser.add_argument(
        "--output",
        default="evaluation/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run on subset (first 3 queries) for quick testing"
    )

    args = parser.parse_args()

    # Run evaluation
    runner = EvaluationRunner(args.dataset, args.output)
    summary = runner.run(quick=args.quick)

    # Exit with error code if evaluation failed
    threshold = 0.6
    if summary["overall_accuracy"] < threshold:
        print(f"⚠️  WARNING: Overall accuracy ({summary['overall_accuracy']:.1%}) below {threshold:.0%} threshold")
        sys.exit(1)
    else:
        print(f"✅ SUCCESS: Overall accuracy ({summary['overall_accuracy']:.1%}) meets quality threshold")
        sys.exit(0)


if __name__ == "__main__":
    main()
