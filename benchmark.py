"""
Comprehensive Benchmark Suite for Holographic Attention

Tests:
1. Needle-in-haystack retrieval (single needle, multiple needles)
2. Copy task (exact retrieval)
3. Soft attention (weighted average)
4. Scaling behavior (1k to 100k+ tokens)
5. Query time consistency
6. Memory efficiency
7. Adversarial cases (similar keys, clustered keys)
8. Multi-query scenarios
9. Numerical stability
10. LLM-realistic scenarios
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from holographic_attention import HolographicAttention
from holographic_attention_prod import HolographicAttentionProd


@dataclass
class BenchmarkResult:
    name: str
    accuracy: float
    mean_error: float
    max_error: float
    query_time_us: float
    passed: bool
    details: Dict[str, Any]


class HolographicBenchmark:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.results: List[BenchmarkResult] = []

    def run_all(self, verbose: bool = True) -> List[BenchmarkResult]:
        """Run all benchmarks."""
        self.results = []

        benchmarks = [
            self.bench_needle_in_haystack,
            self.bench_multiple_needles,
            self.bench_copy_task,
            self.bench_soft_attention,
            self.bench_scaling,
            self.bench_query_time_consistency,
            self.bench_similar_keys,
            self.bench_clustered_keys,
            self.bench_sequential_positions,
            self.bench_random_positions,
            self.bench_high_dimension,
            self.bench_low_dimension,
            self.bench_sparse_values,
            self.bench_numerical_stability,
            self.bench_llm_realistic,
        ]

        for bench_fn in benchmarks:
            try:
                result = bench_fn()
                self.results.append(result)
                if verbose:
                    status = "✓" if result.passed else "✗"
                    print(f"{status} {result.name}: acc={result.accuracy:.3f}, "
                          f"err={result.mean_error:.4f}, time={result.query_time_us:.0f}μs")
            except Exception as e:
                print(f"✗ {bench_fn.__name__}: FAILED with {e}")

        return self.results

    def bench_needle_in_haystack(self) -> BenchmarkResult:
        """Single distinctive needle in random noise."""
        np.random.seed(self.seed)

        n, d = 10000, 64
        K = np.random.randn(n, d) * 0.3
        V = np.zeros((n, d))

        # Plant needle
        needle_pos = n // 2
        K[needle_pos] = np.ones(d) * 2.0
        V[needle_pos] = np.ones(d) * 42.0

        holo = HolographicAttention(d, n_features=1024, seed=self.seed)
        holo.ingest(K, V)

        q = np.ones(d) * 2.0
        t0 = time.perf_counter()
        output = holo.query(q)
        query_time = (time.perf_counter() - t0) * 1e6

        target = 42.0
        error = abs(output[0] - target)
        accuracy = max(0, 1 - error / target)

        return BenchmarkResult(
            name="needle_in_haystack",
            accuracy=accuracy,
            mean_error=error,
            max_error=error,
            query_time_us=query_time,
            passed=accuracy > 0.95,
            details={"retrieved": output[0], "target": target}
        )

    def bench_multiple_needles(self) -> BenchmarkResult:
        """Multiple needles at different positions."""
        np.random.seed(self.seed)

        n, d = 10000, 64
        K = np.random.randn(n, d) * 0.3
        V = np.random.randn(n, d) * 0.1

        # Plant 10 distinct needles
        needle_positions = [100, 500, 1000, 2500, 5000, 7500, 8000, 9000, 9500, 9900]
        needle_values = list(range(10, 110, 10))

        for pos, val in zip(needle_positions, needle_values):
            direction = np.random.randn(d)
            direction = direction / np.linalg.norm(direction)
            K[pos] = direction * 3.0
            V[pos] = np.ones(d) * val

        holo = HolographicAttention(d, n_features=2048, seed=self.seed)
        holo.ingest(K, V)

        errors = []
        times = []
        for pos, val in zip(needle_positions, needle_values):
            q = K[pos].copy()
            t0 = time.perf_counter()
            output = holo.query(q)
            times.append((time.perf_counter() - t0) * 1e6)
            errors.append(abs(output[0] - val))

        accuracy = np.mean([1 - e/v for e, v in zip(errors, needle_values)])

        return BenchmarkResult(
            name="multiple_needles",
            accuracy=accuracy,
            mean_error=np.mean(errors),
            max_error=np.max(errors),
            query_time_us=np.mean(times),
            passed=accuracy > 0.90,
            details={"errors": errors, "positions": needle_positions}
        )

    def bench_copy_task(self) -> BenchmarkResult:
        """Exact value retrieval (copy task)."""
        np.random.seed(self.seed)

        n, d = 1000, 64
        K = np.random.randn(n, d)
        K = K / np.linalg.norm(K, axis=1, keepdims=True) * 2  # Normalize + scale
        V = np.random.randn(n, d)

        holo = HolographicAttention(d, n_features=2048, seed=self.seed)
        holo.ingest(K, V)

        # Query each key and compare to its value
        n_test = min(100, n)
        test_indices = np.random.choice(n, n_test, replace=False)

        errors = []
        perfect = 0
        times = []

        for idx in test_indices:
            q = K[idx]
            t0 = time.perf_counter()
            output = holo.query(q)
            times.append((time.perf_counter() - t0) * 1e6)

            error = np.mean(np.abs(output - V[idx]))
            errors.append(error)
            if error < 0.1:
                perfect += 1

        return BenchmarkResult(
            name="copy_task",
            accuracy=perfect / n_test,
            mean_error=np.mean(errors),
            max_error=np.max(errors),
            query_time_us=np.mean(times),
            passed=perfect / n_test > 0.3,
            details={"perfect_retrievals": perfect, "total": n_test}
        )

    def bench_soft_attention(self) -> BenchmarkResult:
        """Weighted average retrieval (soft attention)."""
        np.random.seed(self.seed)

        n, d = 1000, 64
        K = np.random.randn(n, d)
        V = np.random.randn(n, d)

        holo = HolographicAttention(d, n_features=1024, seed=self.seed)
        holo.ingest(K, V)

        # Random queries
        n_queries = 50
        Q = np.random.randn(n_queries, d)

        errors = []
        times = []

        for q in Q:
            # Standard attention output
            scores = q @ K.T
            weights = np.exp(scores - scores.max())
            weights = weights / weights.sum()
            standard_out = weights @ V

            t0 = time.perf_counter()
            holo_out = holo.query(q)
            times.append((time.perf_counter() - t0) * 1e6)

            error = np.mean(np.abs(holo_out - standard_out))
            errors.append(error)

        return BenchmarkResult(
            name="soft_attention",
            accuracy=1 - np.mean(errors) / np.std(V),
            mean_error=np.mean(errors),
            max_error=np.max(errors),
            query_time_us=np.mean(times),
            passed=np.mean(errors) < 1.0,
            details={"vs_standard_attention": True}
        )

    def bench_scaling(self) -> BenchmarkResult:
        """Query time scaling with sequence length."""
        np.random.seed(self.seed)

        d = 64
        sizes = [1000, 5000, 10000, 50000, 100000]
        query_times = []

        for n in sizes:
            K = np.random.randn(n, d)
            V = np.random.randn(n, d)

            holo = HolographicAttention(d, n_features=1024, seed=self.seed)
            holo.ingest(K, V)

            q = np.random.randn(d)

            # Warm up
            holo.query(q)

            # Measure
            times = []
            for _ in range(10):
                t0 = time.perf_counter()
                holo.query(q)
                times.append((time.perf_counter() - t0) * 1e6)

            query_times.append(np.median(times))

        # Check O(1) behavior: query time should not grow significantly
        ratio = query_times[-1] / query_times[0]
        is_constant = ratio < 2.0  # Allow 2x variance

        return BenchmarkResult(
            name="scaling_o1",
            accuracy=1.0 if is_constant else 0.5,
            mean_error=0,
            max_error=0,
            query_time_us=np.mean(query_times),
            passed=is_constant,
            details={
                "sizes": sizes,
                "times_us": query_times,
                "ratio_100k_vs_1k": ratio
            }
        )

    def bench_query_time_consistency(self) -> BenchmarkResult:
        """Consistent query times across many queries."""
        np.random.seed(self.seed)

        n, d = 10000, 64
        K = np.random.randn(n, d)
        V = np.random.randn(n, d)

        holo = HolographicAttention(d, n_features=1024, seed=self.seed)
        holo.ingest(K, V)

        times = []
        for _ in range(100):
            q = np.random.randn(d)
            t0 = time.perf_counter()
            holo.query(q)
            times.append((time.perf_counter() - t0) * 1e6)

        std_time = np.std(times)
        mean_time = np.mean(times)
        cv = std_time / mean_time  # Coefficient of variation

        return BenchmarkResult(
            name="query_time_consistency",
            accuracy=1.0 if cv < 0.5 else 0.5,
            mean_error=0,
            max_error=0,
            query_time_us=mean_time,
            passed=cv < 0.5,
            details={"cv": cv, "std_us": std_time, "mean_us": mean_time}
        )

    def bench_similar_keys(self) -> BenchmarkResult:
        """Keys that are very similar to each other (hard case)."""
        np.random.seed(self.seed)

        n, d = 1000, 64
        base = np.random.randn(d)
        base = base / np.linalg.norm(base)

        # Create keys that are slight perturbations of base
        K = np.tile(base, (n, 1)) + np.random.randn(n, d) * 0.1
        V = np.arange(n).reshape(-1, 1) * np.ones((1, d))  # Value = index

        holo = HolographicAttention(d, n_features=2048, seed=self.seed)
        holo.ingest(K, V)

        # Query specific keys
        errors = []
        times = []
        test_indices = [0, 100, 500, 999]

        for idx in test_indices:
            q = K[idx]
            t0 = time.perf_counter()
            output = holo.query(q)
            times.append((time.perf_counter() - t0) * 1e6)

            error = abs(output[0] - idx)
            errors.append(error)

        return BenchmarkResult(
            name="similar_keys",
            accuracy=1 - np.mean(errors) / 500,
            mean_error=np.mean(errors),
            max_error=np.max(errors),
            query_time_us=np.mean(times),
            passed=np.mean(errors) < 100,
            details={"test_indices": test_indices, "errors": errors}
        )

    def bench_clustered_keys(self) -> BenchmarkResult:
        """Keys in distinct clusters."""
        np.random.seed(self.seed)

        n_clusters = 10
        n_per_cluster = 100
        n = n_clusters * n_per_cluster
        d = 64

        K = np.zeros((n, d))
        V = np.zeros((n, d))

        # Create clusters
        for c in range(n_clusters):
            center = np.random.randn(d)
            center = center / np.linalg.norm(center) * 3
            start = c * n_per_cluster
            end = start + n_per_cluster
            K[start:end] = center + np.random.randn(n_per_cluster, d) * 0.2
            V[start:end] = np.ones((n_per_cluster, d)) * c * 10

        holo = HolographicAttention(d, n_features=1024, seed=self.seed)
        holo.ingest(K, V)

        # Query cluster centers
        errors = []
        times = []

        for c in range(n_clusters):
            center_idx = c * n_per_cluster + n_per_cluster // 2
            q = K[center_idx]
            expected_val = c * 10

            t0 = time.perf_counter()
            output = holo.query(q)
            times.append((time.perf_counter() - t0) * 1e6)

            error = abs(output[0] - expected_val)
            errors.append(error)

        return BenchmarkResult(
            name="clustered_keys",
            accuracy=1 - np.mean(errors) / 50,
            mean_error=np.mean(errors),
            max_error=np.max(errors),
            query_time_us=np.mean(times),
            passed=np.mean(errors) < 5,
            details={"n_clusters": n_clusters}
        )

    def bench_sequential_positions(self) -> BenchmarkResult:
        """Retrieving from different positions in sequence."""
        np.random.seed(self.seed)

        n, d = 10000, 64
        K = np.random.randn(n, d)
        K = K / np.linalg.norm(K, axis=1, keepdims=True) * 2
        V = np.arange(n).reshape(-1, 1) * np.ones((1, d))

        holo = HolographicAttention(d, n_features=2048, seed=self.seed)
        holo.ingest(K, V)

        # Test positions: beginning, middle, end
        positions = [0, 10, 100, 1000, 5000, 9000, 9990, 9999]
        errors = []
        times = []

        for pos in positions:
            q = K[pos]
            t0 = time.perf_counter()
            output = holo.query(q)
            times.append((time.perf_counter() - t0) * 1e6)

            error = abs(output[0] - pos)
            errors.append(error)

        return BenchmarkResult(
            name="sequential_positions",
            accuracy=1 - np.mean(errors) / 5000,
            mean_error=np.mean(errors),
            max_error=np.max(errors),
            query_time_us=np.mean(times),
            passed=np.mean(errors) < 500,
            details={"positions": positions, "errors": errors}
        )

    def bench_random_positions(self) -> BenchmarkResult:
        """Random position retrieval."""
        np.random.seed(self.seed)

        n, d = 10000, 64
        K = np.random.randn(n, d)
        K = K / np.linalg.norm(K, axis=1, keepdims=True) * 2
        V = np.arange(n).reshape(-1, 1) * np.ones((1, d))

        holo = HolographicAttention(d, n_features=2048, seed=self.seed)
        holo.ingest(K, V)

        # Random positions
        n_test = 100
        positions = np.random.choice(n, n_test, replace=False)
        errors = []
        times = []

        for pos in positions:
            q = K[pos]
            t0 = time.perf_counter()
            output = holo.query(q)
            times.append((time.perf_counter() - t0) * 1e6)

            error = abs(output[0] - pos)
            errors.append(error)

        perfect = sum(1 for e in errors if e < 1)

        return BenchmarkResult(
            name="random_positions",
            accuracy=perfect / n_test,
            mean_error=np.mean(errors),
            max_error=np.max(errors),
            query_time_us=np.mean(times),
            passed=perfect / n_test > 0.3,
            details={"perfect": perfect, "total": n_test}
        )

    def bench_high_dimension(self) -> BenchmarkResult:
        """High dimensional keys (d=512)."""
        np.random.seed(self.seed)

        n, d = 5000, 512
        K = np.random.randn(n, d)
        V = np.zeros((n, d))

        needle_pos = n // 2
        K[needle_pos] = np.ones(d) * 0.5
        V[needle_pos] = np.ones(d) * 42.0

        holo = HolographicAttention(d, n_features=2048, seed=self.seed)
        holo.ingest(K, V)

        q = np.ones(d) * 0.5
        t0 = time.perf_counter()
        output = holo.query(q)
        query_time = (time.perf_counter() - t0) * 1e6

        error = abs(output[0] - 42.0)

        return BenchmarkResult(
            name="high_dimension_d512",
            accuracy=max(0, 1 - error / 42),
            mean_error=error,
            max_error=error,
            query_time_us=query_time,
            passed=error < 5,
            details={"d": d, "retrieved": output[0]}
        )

    def bench_low_dimension(self) -> BenchmarkResult:
        """Low dimensional keys (d=8)."""
        np.random.seed(self.seed)

        n, d = 10000, 8
        K = np.random.randn(n, d) * 0.5
        V = np.zeros((n, d))

        needle_pos = n // 2
        K[needle_pos] = np.ones(d) * 2.0
        V[needle_pos] = np.ones(d) * 42.0

        holo = HolographicAttention(d, n_features=512, seed=self.seed)
        holo.ingest(K, V)

        q = np.ones(d) * 2.0
        t0 = time.perf_counter()
        output = holo.query(q)
        query_time = (time.perf_counter() - t0) * 1e6

        error = abs(output[0] - 42.0)

        return BenchmarkResult(
            name="low_dimension_d8",
            accuracy=max(0, 1 - error / 42),
            mean_error=error,
            max_error=error,
            query_time_us=query_time,
            passed=error < 5,
            details={"d": d, "retrieved": output[0]}
        )

    def bench_sparse_values(self) -> BenchmarkResult:
        """Sparse value vectors."""
        np.random.seed(self.seed)

        n, d = 5000, 64
        K = np.random.randn(n, d)
        K = K / np.linalg.norm(K, axis=1, keepdims=True) * 2

        # Sparse values: only 10% nonzero
        V = np.zeros((n, d))
        for i in range(n):
            sparse_idx = np.random.choice(d, d // 10, replace=False)
            V[i, sparse_idx] = np.random.randn(d // 10)

        holo = HolographicAttention(d, n_features=1024, seed=self.seed)
        holo.ingest(K, V)

        # Query and compare
        n_test = 50
        test_indices = np.random.choice(n, n_test, replace=False)
        errors = []
        times = []

        for idx in test_indices:
            q = K[idx]
            t0 = time.perf_counter()
            output = holo.query(q)
            times.append((time.perf_counter() - t0) * 1e6)

            error = np.mean(np.abs(output - V[idx]))
            errors.append(error)

        return BenchmarkResult(
            name="sparse_values",
            accuracy=1 - np.mean(errors),
            mean_error=np.mean(errors),
            max_error=np.max(errors),
            query_time_us=np.mean(times),
            passed=np.mean(errors) < 0.5,
            details={"sparsity": 0.9}
        )

    def bench_numerical_stability(self) -> BenchmarkResult:
        """Test numerical stability with extreme values."""
        np.random.seed(self.seed)

        n, d = 1000, 64

        # Mix of scales
        K = np.vstack([
            np.random.randn(n//3, d) * 0.01,  # Very small
            np.random.randn(n//3, d) * 1.0,   # Normal
            np.random.randn(n//3 + n%3, d) * 10.0,  # Large
        ])
        V = np.random.randn(n, d)

        holo = HolographicAttention(d, n_features=1024, seed=self.seed)
        holo.ingest(K, V)

        # Query different scales
        queries = [
            np.random.randn(d) * 0.01,
            np.random.randn(d) * 1.0,
            np.random.randn(d) * 10.0,
        ]

        all_finite = True
        times = []

        for q in queries:
            t0 = time.perf_counter()
            output = holo.query(q)
            times.append((time.perf_counter() - t0) * 1e6)

            if not np.all(np.isfinite(output)):
                all_finite = False

        return BenchmarkResult(
            name="numerical_stability",
            accuracy=1.0 if all_finite else 0.0,
            mean_error=0,
            max_error=0,
            query_time_us=np.mean(times),
            passed=all_finite,
            details={"all_finite": all_finite}
        )

    def bench_llm_realistic(self) -> BenchmarkResult:
        """LLM-realistic scenario: d_head=64, seq_len=4096."""
        np.random.seed(self.seed)

        n, d = 4096, 64  # Typical LLM head size

        # Simulate learned key/value embeddings with some structure
        K = np.random.randn(n, d) * 0.5
        V = np.random.randn(n, d) * 0.5

        # Add some structure: recent tokens have higher norms
        for i in range(n):
            decay = 1.0 + 0.5 * (i / n)
            K[i] *= decay

        holo = HolographicAttention(d, n_features=2048, seed=self.seed)

        t0 = time.perf_counter()
        holo.ingest(K, V)
        ingest_time = (time.perf_counter() - t0) * 1000

        # Query recent positions (most common in LLM)
        recent_positions = range(n - 100, n)
        errors = []
        times = []

        for pos in recent_positions:
            q = K[pos]
            t0 = time.perf_counter()
            output = holo.query(q)
            times.append((time.perf_counter() - t0) * 1e6)

            # Compare to standard attention
            scores = q @ K.T
            weights = np.exp(scores - scores.max())
            weights = weights / weights.sum()
            standard_out = weights @ V

            error = np.mean(np.abs(output - standard_out))
            errors.append(error)

        return BenchmarkResult(
            name="llm_realistic",
            accuracy=1 - np.mean(errors),
            mean_error=np.mean(errors),
            max_error=np.max(errors),
            query_time_us=np.mean(times),
            passed=np.mean(errors) < 1.0,
            details={
                "seq_len": n,
                "d_head": d,
                "ingest_time_ms": ingest_time,
                "compression": holo.compression_ratio()
            }
        )

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.results:
            return {}

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        avg_accuracy = np.mean([r.accuracy for r in self.results])
        avg_query_time = np.mean([r.query_time_us for r in self.results])

        return {
            "passed": passed,
            "total": total,
            "pass_rate": passed / total,
            "avg_accuracy": avg_accuracy,
            "avg_query_time_us": avg_query_time,
            "failed_tests": [r.name for r in self.results if not r.passed]
        }


def main():
    print("=" * 60)
    print("Holographic Attention Benchmark Suite")
    print("=" * 60)
    print()

    bench = HolographicBenchmark(seed=42)
    bench.run_all(verbose=True)

    print()
    print("-" * 60)
    summary = bench.summary()
    print(f"SUMMARY: {summary['passed']}/{summary['total']} tests passed "
          f"({summary['pass_rate']*100:.1f}%)")
    print(f"Average accuracy: {summary['avg_accuracy']:.3f}")
    print(f"Average query time: {summary['avg_query_time_us']:.0f}μs")

    if summary['failed_tests']:
        print(f"\nFailed tests: {', '.join(summary['failed_tests'])}")

    return bench.results


if __name__ == "__main__":
    main()
