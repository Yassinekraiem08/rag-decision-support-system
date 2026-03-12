"""
Cost & Latency Tracking

Tracks token usage and latency for cost optimization:
- Token counting (embeddings, LLM input/output)
- Cost calculation (OpenAI pricing)
- Latency breakdown by component
"""

import time
from functools import wraps
from typing import Dict, List
from collections import defaultdict


class CostTracker:
    """Track API costs and latencies across evaluation"""

    # OpenAI Pricing (as of 2026)
    PRICING = {
        "text-embedding-3-small": 0.020 / 1_000_000,  # per token
        "text-embedding-3-large": 0.130 / 1_000_000,
        "gpt-4.1-mini-input": 0.150 / 1_000_000,
        "gpt-4.1-mini-output": 0.600 / 1_000_000,
        "gpt-4-input": 5.00 / 1_000_000,
        "gpt-4-output": 15.00 / 1_000_000,
    }

    def __init__(self):
        self.embedding_tokens = 0
        self.llm_input_tokens = 0
        self.llm_output_tokens = 0
        self.latencies = defaultdict(list)  # component -> list of latencies

    def track_embedding(self, num_tokens: int):
        """Track embedding tokens"""
        self.embedding_tokens += num_tokens

    def track_llm(self, input_tokens: int, output_tokens: int):
        """Track LLM tokens"""
        self.llm_input_tokens += input_tokens
        self.llm_output_tokens += output_tokens

    def track_latency(self, component: str, latency_seconds: float):
        """Track component latency"""
        self.latencies[component].append(latency_seconds)

    def calculate_cost(self, embedding_model: str = "text-embedding-3-small", llm_model: str = "gpt-4.1-mini") -> Dict:
        """
        Calculate total cost based on tracked usage.

        Args:
            embedding_model: Embedding model used
            llm_model: LLM model used

        Returns:
            Dict with cost breakdown
        """
        # Embedding cost
        emb_price_key = embedding_model
        emb_cost = self.embedding_tokens * self.PRICING.get(emb_price_key, self.PRICING["text-embedding-3-small"])

        # LLM cost
        llm_input_price = self.PRICING.get(f"{llm_model}-input", self.PRICING["gpt-4.1-mini-input"])
        llm_output_price = self.PRICING.get(f"{llm_model}-output", self.PRICING["gpt-4.1-mini-output"])

        llm_input_cost = self.llm_input_tokens * llm_input_price
        llm_output_cost = self.llm_output_tokens * llm_output_price
        llm_total_cost = llm_input_cost + llm_output_cost

        total_cost = emb_cost + llm_total_cost

        return {
            "total_cost_usd": total_cost,
            "embedding_cost_usd": emb_cost,
            "llm_cost_usd": llm_total_cost,
            "breakdown": {
                "embedding_tokens": self.embedding_tokens,
                "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens,
                "llm_total_tokens": self.llm_input_tokens + self.llm_output_tokens
            }
        }

    def get_latency_stats(self) -> Dict:
        """
        Calculate latency statistics per component.

        Returns:
            Dict with mean, median, p95, p99 latencies per component
        """
        import numpy as np

        stats = {}

        for component, times in self.latencies.items():
            if not times:
                continue

            times_ms = [t * 1000 for t in times]  # Convert to milliseconds

            stats[component] = {
                "count": len(times),
                "mean_ms": np.mean(times_ms),
                "median_ms": np.median(times_ms),
                "p95_ms": np.percentile(times_ms, 95) if len(times) > 1 else times_ms[0],
                "p99_ms": np.percentile(times_ms, 99) if len(times) > 1 else times_ms[0],
                "min_ms": min(times_ms),
                "max_ms": max(times_ms),
                "total_ms": sum(times_ms)
            }

        return stats

    def print_report(self, num_queries: int = None):
        """Print formatted cost and latency report"""
        cost_data = self.calculate_cost()
        latency_data = self.get_latency_stats()

        print("\n" + "=" * 80)
        print("COST & LATENCY REPORT")
        print("=" * 80)

        # Cost breakdown
        print("\n📊 Cost Breakdown:")
        print(f"  Total Cost:        ${cost_data['total_cost_usd']:.4f}")
        print(f"  Embedding Cost:    ${cost_data['embedding_cost_usd']:.4f}")
        print(f"  LLM Cost:          ${cost_data['llm_cost_usd']:.4f}")

        if num_queries:
            cost_per_query = cost_data['total_cost_usd'] / num_queries
            print(f"  Cost per Query:    ${cost_per_query:.4f}")

        # Token usage
        print(f"\n🔢 Token Usage:")
        print(f"  Embedding:         {cost_data['breakdown']['embedding_tokens']:,}")
        print(f"  LLM Input:         {cost_data['breakdown']['llm_input_tokens']:,}")
        print(f"  LLM Output:        {cost_data['breakdown']['llm_output_tokens']:,}")
        print(f"  LLM Total:         {cost_data['breakdown']['llm_total_tokens']:,}")

        # Latency stats
        if latency_data:
            print(f"\n⏱️  Latency Breakdown:")
            for component, stats in sorted(latency_data.items()):
                print(f"  {component:20s}: "
                      f"mean={stats['mean_ms']:.0f}ms, "
                      f"p95={stats['p95_ms']:.0f}ms, "
                      f"calls={stats['count']}")

        print("=" * 80 + "\n")

    def get_summary(self) -> Dict:
        """Get summary dict for saving to file"""
        return {
            "cost": self.calculate_cost(),
            "latency": self.get_latency_stats()
        }


def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text.

    Rough estimate: ~4 characters per token for English text.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    return max(1, len(text) // 4)


# Decorator for tracking latency
def track_latency(component_name: str):
    """
    Decorator to track function latency.

    Usage:
        @track_latency("query_embedding")
        def my_function():
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Look for cost_tracker in kwargs
            tracker = kwargs.get('cost_tracker')

            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start

            if tracker:
                tracker.track_latency(component_name, elapsed)

            return result
        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage
    tracker = CostTracker()

    # Simulate some API calls
    tracker.track_embedding(1000)  # 1 embedding call
    tracker.track_llm(500, 200)    # 1 LLM call (generation)
    tracker.track_llm(400, 150)    # 1 LLM call (verification)

    # Track latencies
    tracker.track_latency("query_embedding", 0.045)
    tracker.track_latency("retrieval", 0.120)
    tracker.track_latency("reranking", 0.450)
    tracker.track_latency("generation", 0.280)
    tracker.track_latency("verification", 0.320)

    # Print report
    tracker.print_report(num_queries=1)
