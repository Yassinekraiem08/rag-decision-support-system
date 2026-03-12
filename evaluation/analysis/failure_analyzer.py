"""
Failure Analysis & Error Taxonomy

Systematically classifies evaluation failures to identify root causes:
- RETRIEVAL_MISS: Relevant chunk not retrieved at all
- RETRIEVAL_RANK: Relevant chunk retrieved but ranked too low
- RETRIEVAL_NOISE: Irrelevant chunks retrieved
- GEN_INCOMPLETE: Answer missing key information
- GEN_HALLUCINATION: Answer contains unsupported facts
- VERIFICATION_ERROR: Verification verdict is incorrect
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from enum import Enum
from typing import Dict, List, Set
from evaluation.metrics.generation_metrics import semantic_similarity


class FailureMode(Enum):
    """Taxonomy of failure modes"""
    # Retrieval failures
    RETRIEVAL_MISS = "relevant_chunk_not_retrieved"
    RETRIEVAL_RANK = "relevant_chunk_ranked_too_low"
    RETRIEVAL_NOISE = "irrelevant_chunks_retrieved"

    # Generation failures
    GEN_INCOMPLETE = "answer_missing_key_information"
    GEN_HALLUCINATION = "answer_contains_hallucinated_facts"
    GEN_CITATION_ERROR = "incorrect_or_missing_citations"

    # Hybrid failures
    RETRIEVAL_AND_GEN = "both_retrieval_and_generation_failed"

    # Edge cases
    UNANSWERABLE_CORRECT = "correctly_identified_unanswerable"
    UNANSWERABLE_HALLUCINATE = "hallucinated_answer_for_unanswerable"

    # Verification failures
    VERIFICATION_FALSE_POSITIVE = "marked_supported_but_actually_wrong"
    VERIFICATION_FALSE_NEGATIVE = "marked_unsupported_but_actually_correct"


class FailureAnalyzer:
    """Analyzes failures and classifies them by failure mode"""

    def __init__(self):
        self.analyses = []

    def analyze_failure(
        self,
        query_item: Dict,
        result: Dict
    ) -> Dict:
        """
        Analyze a single failure case.

        Args:
            query_item: Original query from dataset
            result: Evaluation result for this query

        Returns:
            Dict with failure analysis
        """
        query_id = result["query_id"]
        question = result["question"]
        expected_sources = set(result["expected_sources"])
        retrieved_filenames = result["retrieved_filenames"]
        answer = result.get("answer")
        ground_truth = query_item.get("ground_truth_answer")
        verification = result.get("verification", {})
        is_answerable = query_item.get("answerable", True)

        # Initialize analysis
        analysis = {
            "query_id": query_id,
            "question": question,
            "failure_modes": [],
            "root_cause": None,
            "explanation": "",
            "severity": "moderate",
            "recommended_fix": ""
        }

        # Check if this is actually a failure
        if result["success"]:
            analysis["explanation"] = "Query succeeded - no failure to analyze"
            return analysis

        # 1. Check retrieval failures
        retrieval_issues = self._check_retrieval(expected_sources, retrieved_filenames)
        analysis["failure_modes"].extend(retrieval_issues)

        # 2. Check generation failures (if we have ground truth)
        if ground_truth and answer:
            gen_issues = self._check_generation(answer, ground_truth)
            analysis["failure_modes"].extend(gen_issues)

        # 3. Check for unanswerable edge cases
        if not is_answerable:
            if answer and len(answer) > 50:  # Generated a substantial answer
                analysis["failure_modes"].append(FailureMode.UNANSWERABLE_HALLUCINATE)
            else:
                analysis["failure_modes"].append(FailureMode.UNANSWERABLE_CORRECT)

        # 4. Check verification accuracy
        if ground_truth and answer:
            # Calculate actual correctness via semantic similarity
            similarity = semantic_similarity(answer, ground_truth)
            verdict = verification.get("verdict", "UNKNOWN")

            # False positive: marked SUPPORTED but answer is wrong
            if verdict == "SUPPORTED" and similarity < 0.5:
                analysis["failure_modes"].append(FailureMode.VERIFICATION_FALSE_POSITIVE)

            # False negative: marked UNSUPPORTED but answer is correct
            elif verdict == "UNSUPPORTED" and similarity >= 0.7:
                analysis["failure_modes"].append(FailureMode.VERIFICATION_FALSE_NEGATIVE)

        # 5. Determine root cause
        analysis["root_cause"] = self._determine_root_cause(analysis["failure_modes"])

        # 6. Generate explanation
        analysis["explanation"] = self._generate_explanation(
            analysis["failure_modes"],
            expected_sources,
            retrieved_filenames,
            answer,
            ground_truth
        )

        # 7. Determine severity
        analysis["severity"] = self._determine_severity(analysis["failure_modes"])

        # 8. Recommend fix
        analysis["recommended_fix"] = self._recommend_fix(
            analysis["root_cause"],
            analysis["failure_modes"]
        )

        self.analyses.append(analysis)
        return analysis

    def _check_retrieval(
        self,
        expected: Set[str],
        retrieved: List[str]
    ) -> List[FailureMode]:
        """Check for retrieval failures"""
        issues = []

        if not expected:
            return issues

        retrieved_set = set(retrieved)

        # Check if expected sources are missing entirely
        missing = expected - retrieved_set
        if missing:
            issues.append(FailureMode.RETRIEVAL_MISS)

        # Check if expected sources are retrieved but ranked low
        # (assuming top-3 matters most)
        top_3_retrieved = set(retrieved[:3])
        in_results_but_not_top = (expected & retrieved_set) - top_3_retrieved
        if in_results_but_not_top:
            issues.append(FailureMode.RETRIEVAL_RANK)

        # Check if irrelevant sources dominate
        irrelevant_in_top_3 = top_3_retrieved - expected
        if len(irrelevant_in_top_3) >= 2:  # 2+ irrelevant in top-3
            issues.append(FailureMode.RETRIEVAL_NOISE)

        return issues

    def _check_generation(
        self,
        answer: str,
        ground_truth: str
    ) -> List[FailureMode]:
        """Check for generation failures"""
        issues = []

        # Calculate semantic similarity
        similarity = semantic_similarity(answer, ground_truth)

        # If similarity is low, answer is likely incorrect
        if similarity < 0.5:
            issues.append(FailureMode.GEN_HALLUCINATION)
        elif similarity < 0.7:
            issues.append(FailureMode.GEN_INCOMPLETE)

        return issues

    def _determine_root_cause(self, failure_modes: List[FailureMode]) -> str:
        """Determine the primary root cause"""
        if not failure_modes:
            return "unknown"

        # Prioritize retrieval failures (they cascade to generation)
        retrieval_failures = [
            FailureMode.RETRIEVAL_MISS,
            FailureMode.RETRIEVAL_RANK,
            FailureMode.RETRIEVAL_NOISE
        ]

        for mode in failure_modes:
            if mode in retrieval_failures:
                return "retrieval_failure"

        # Then check generation
        generation_failures = [
            FailureMode.GEN_INCOMPLETE,
            FailureMode.GEN_HALLUCINATION,
            FailureMode.GEN_CITATION_ERROR
        ]

        for mode in failure_modes:
            if mode in generation_failures:
                return "generation_failure"

        # Check for edge cases
        if FailureMode.UNANSWERABLE_HALLUCINATE in failure_modes:
            return "unanswerable_hallucination"

        return "multiple_failures"

    def _generate_explanation(
        self,
        failure_modes: List[FailureMode],
        expected: Set[str],
        retrieved: List[str],
        answer: str,
        ground_truth: str
    ) -> str:
        """Generate human-readable explanation"""
        explanations = []

        if FailureMode.RETRIEVAL_MISS in failure_modes:
            missing = expected - set(retrieved)
            explanations.append(
                f"Expected sources {missing} were not retrieved at all. "
                "This indicates semantic mismatch or keyword gap between query and document content."
            )

        if FailureMode.RETRIEVAL_RANK in failure_modes:
            explanations.append(
                "Expected sources were retrieved but ranked too low (outside top-3). "
                "Reranking may not be prioritizing relevant chunks correctly."
            )

        if FailureMode.RETRIEVAL_NOISE in failure_modes:
            explanations.append(
                "Top-3 results dominated by irrelevant sources. "
                "This suggests poor discrimination between relevant and irrelevant content."
            )

        if FailureMode.GEN_INCOMPLETE in failure_modes:
            explanations.append(
                "Generated answer is partially correct but missing key information from ground truth."
            )

        if FailureMode.GEN_HALLUCINATION in failure_modes:
            explanations.append(
                "Generated answer contains information not supported by retrieved sources or ground truth."
            )

        if FailureMode.UNANSWERABLE_HALLUCINATE in failure_modes:
            explanations.append(
                "Question is unanswerable (not in corpus) but system generated an answer anyway. "
                "Need better detection of when to say 'I don't know'."
            )

        return " ".join(explanations) if explanations else "Failure mode unclear."

    def _determine_severity(self, failure_modes: List[FailureMode]) -> str:
        """Determine severity level"""
        critical_modes = [
            FailureMode.GEN_HALLUCINATION,
            FailureMode.UNANSWERABLE_HALLUCINATE,
            FailureMode.VERIFICATION_FALSE_POSITIVE
        ]

        if any(mode in critical_modes for mode in failure_modes):
            return "critical"

        if len(failure_modes) >= 3:
            return "high"

        if len(failure_modes) >= 2:
            return "moderate"

        return "low"

    def _recommend_fix(self, root_cause: str, failure_modes: List[FailureMode]) -> str:
        """Recommend specific fixes"""
        if root_cause == "retrieval_failure":
            if FailureMode.RETRIEVAL_MISS in failure_modes:
                return (
                    "1. Expand query with synonyms/paraphrases\n"
                    "2. Increase top-K candidates before reranking (6→10)\n"
                    "3. Tune keyword boost weight (currently 0.1)\n"
                    "4. Consider query expansion or multi-query retrieval"
                )
            elif FailureMode.RETRIEVAL_RANK in failure_modes:
                return (
                    "1. Improve reranking model (try different prompts)\n"
                    "2. Increase weight of semantic similarity vs keyword matching\n"
                    "3. Consider ensemble reranking (multiple strategies)"
                )
            elif FailureMode.RETRIEVAL_NOISE in failure_modes:
                return (
                    "1. Increase score threshold (min_score > 0.5)\n"
                    "2. Improve chunk quality during ingestion (filter low-value chunks)\n"
                    "3. Fine-tune embedding model on domain data"
                )

        elif root_cause == "generation_failure":
            return (
                "1. Improve prompt to emphasize completeness\n"
                "2. Increase context window to include more chunks\n"
                "3. Add chain-of-thought reasoning step before final answer"
            )

        elif root_cause == "unanswerable_hallucination":
            return (
                "1. Add pre-generation check: 'Are sources sufficient?'\n"
                "2. Tune verification to catch unsupported answers\n"
                "3. Prompt LLM to say 'not enough information' when uncertain"
            )

        return "Requires deeper investigation"

    def generate_report(self) -> str:
        """Generate markdown failure analysis report"""
        if not self.analyses:
            return "No failures to analyze."

        # Aggregate statistics
        failure_mode_counts = {}
        for analysis in self.analyses:
            for mode in analysis["failure_modes"]:
                mode_name = mode.value
                failure_mode_counts[mode_name] = failure_mode_counts.get(mode_name, 0) + 1

        # Root cause distribution
        root_cause_counts = {}
        for analysis in self.analyses:
            rc = analysis["root_cause"]
            root_cause_counts[rc] = root_cause_counts.get(rc, 0) + 1

        # Build report
        lines = []
        lines.append("# Failure Analysis Report\n")
        lines.append(f"**Total Failures Analyzed:** {len(self.analyses)}\n")

        lines.append("## Failure Mode Distribution\n")
        lines.append("| Failure Mode | Count | % of Failures |")
        lines.append("|--------------|-------|---------------|")

        sorted_modes = sorted(failure_mode_counts.items(), key=lambda x: x[1], reverse=True)
        for mode, count in sorted_modes:
            pct = (count / len(self.analyses)) * 100
            lines.append(f"| {mode} | {count} | {pct:.1f}% |")

        lines.append("\n## Root Cause Distribution\n")
        lines.append("| Root Cause | Count | % of Failures |")
        lines.append("|------------|-------|---------------|")

        sorted_causes = sorted(root_cause_counts.items(), key=lambda x: x[1], reverse=True)
        for cause, count in sorted_causes:
            pct = (count / len(self.analyses)) * 100
            lines.append(f"| {cause} | {count} | {pct:.1f}% |")

        lines.append("\n## Critical Failures (Severity: Critical)\n")
        critical = [a for a in self.analyses if a["severity"] == "critical"]

        if critical:
            for analysis in critical:
                lines.append(f"\n### {analysis['query_id']}: {analysis['question']}\n")
                lines.append(f"**Failure Modes:** {', '.join(m.value for m in analysis['failure_modes'])}\n")
                lines.append(f"**Explanation:** {analysis['explanation']}\n")
                lines.append(f"**Recommended Fix:**\n{analysis['recommended_fix']}\n")
        else:
            lines.append("No critical failures.\n")

        lines.append("\n## Key Insights\n")

        # Generate insights
        if "retrieval_failure" in root_cause_counts:
            ret_pct = (root_cause_counts["retrieval_failure"] / len(self.analyses)) * 100
            lines.append(f"- **Retrieval is the bottleneck:** {ret_pct:.0f}% of failures stem from retrieval issues\n")

        if FailureMode.RETRIEVAL_MISS.value in failure_mode_counts:
            lines.append("- **Semantic gap:** Many expected sources not retrieved - suggests embedding/keyword mismatch\n")

        if FailureMode.RETRIEVAL_RANK.value in failure_mode_counts:
            lines.append("- **Ranking quality:** Relevant sources retrieved but not prioritized correctly\n")

        return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    analyzer = FailureAnalyzer()

    # Simulate a failure
    query_item = {
        "id": "Q001",
        "question": "What is embodied AI?",
        "expected_sources": ["capa.pdf", "qya.pdf"],
        "ground_truth_answer": "Embodied AI refers to AI systems with physical form.",
        "answerable": True
    }

    result = {
        "query_id": "Q001",
        "question": "What is embodied AI?",
        "expected_sources": ["capa.pdf", "qya.pdf"],
        "retrieved_filenames": ["bkbj.pdf", "hrb1.pdf", "qya.pdf"],  # Missing capa.pdf
        "answer": "Embodied AI is about robots.",
        "verification": {"verdict": "SUPPORTED"},
        "success": False
    }

    analysis = analyzer.analyze_failure(query_item, result)

    print("Failure Analysis Example:")
    print("=" * 80)
    print(f"Query: {analysis['question']}")
    print(f"Failure Modes: {[m.value for m in analysis['failure_modes']]}")
    print(f"Root Cause: {analysis['root_cause']}")
    print(f"Severity: {analysis['severity']}")
    print(f"\nExplanation:\n{analysis['explanation']}")
    print(f"\nRecommended Fix:\n{analysis['recommended_fix']}")
