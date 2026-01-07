"""
Comparison benchmark for all holographic attention versions.
"""

import numpy as np
import time
from typing import Dict, Any, List
from dataclasses import dataclass

from holographic_attention import HolographicAttention
from holographic_attention_v2 import (
    HolographicAttentionV2,
    HolographicAttentionV3,
    HolographicAttentionV4
)
from holographic_attention_v5 import HolographicAttentionV5


@dataclass
class TestResult:
    name: str
    accuracy: float
    mean_error: float
    query_time_us: float


def run_test(holo, K, V, queries, targets, name: str) -> TestResult:
    """Run a single test and return results."""
    holo.ingest(K, V)

    errors = []
    times = []

    for q, target in zip(queries, targets):
        t0 = time.perf_counter()
        output = holo.query(q)
        times.append((time.perf_counter() - t0) * 1e6)

        if isinstance(target, np.ndarray):
            error = np.mean(np.abs(output - target))
        else:
            error = abs(output[0] - target)
        errors.append(error)

    mean_error = np.mean(errors)
    # Compute accuracy based on error relative to target magnitude
    target_mag = np.mean([np.abs(t).mean() if isinstance(t, np.ndarray) else abs(t) for t in targets])
    accuracy = max(0, 1 - mean_error / (target_mag + 1e-8))

    return TestResult(
        name=name,
        accuracy=accuracy,
        mean_error=mean_error,
        query_time_us=np.mean(times)
    )


def compare_versions():
    """Compare all versions on key benchmarks."""
    np.random.seed(42)
    d = 64

    # Define all model constructors
    models = {
        'V1 (original)': lambda: HolographicAttention(d, n_features=1024, seed=42),
        'V2 (multi-table)': lambda: HolographicAttentionV2(d, n_features=512, n_tables=8, seed=42),
        'V3 (hierarchical)': lambda: HolographicAttentionV3(d, n_features=256, n_buckets=64, n_tables=4, seed=42),
        'V4 (polynomial)': lambda: HolographicAttentionV4(d, n_features=512, n_scales=4, seed=42),
        'V5 (residual)': lambda: HolographicAttentionV5(d, n_features=1024, n_heads=4, seed=42),
    }

    tests = {}

    # Test 1: Needle in haystack
    print("\n" + "="*60)
    print("TEST 1: Needle in Haystack (n=10000)")
    print("="*60)
    n = 10000
    K = np.random.randn(n, d) * 0.3
    V = np.zeros((n, d))
    K[n//2] = np.ones(d) * 2.0
    V[n//2] = np.ones(d) * 42.0
    queries = [np.ones(d) * 2.0]
    targets = [42.0]

    for name, model_fn in models.items():
        holo = model_fn()
        result = run_test(holo, K, V, queries, targets, name)
        print(f"  {name:20s}: acc={result.accuracy:.3f}, err={result.mean_error:.4f}, time={result.query_time_us:.0f}μs")

    # Test 2: Multiple needles
    print("\n" + "="*60)
    print("TEST 2: Multiple Needles (10 needles in n=10000)")
    print("="*60)
    K = np.random.randn(n, d) * 0.3
    V = np.random.randn(n, d) * 0.1
    positions = [100, 500, 1000, 2500, 5000, 7500, 8000, 9000, 9500, 9900]
    values = list(range(10, 110, 10))
    queries = []
    targets = []
    for pos, val in zip(positions, values):
        direction = np.random.randn(d)
        direction = direction / np.linalg.norm(direction) * 3.0
        K[pos] = direction
        V[pos] = np.ones(d) * val
        queries.append(direction)
        targets.append(val)

    for name, model_fn in models.items():
        holo = model_fn()
        result = run_test(holo, K, V, queries, targets, name)
        print(f"  {name:20s}: acc={result.accuracy:.3f}, err={result.mean_error:.4f}, time={result.query_time_us:.0f}μs")

    # Test 3: Copy task
    print("\n" + "="*60)
    print("TEST 3: Copy Task (exact retrieval, n=1000)")
    print("="*60)
    n = 1000
    K = np.random.randn(n, d)
    K = K / np.linalg.norm(K, axis=1, keepdims=True) * 2
    V = np.random.randn(n, d)
    test_idx = np.random.choice(n, 50, replace=False)
    queries = [K[i] for i in test_idx]
    targets = [V[i] for i in test_idx]

    for name, model_fn in models.items():
        holo = model_fn()
        result = run_test(holo, K, V, queries, targets, name)
        print(f"  {name:20s}: acc={result.accuracy:.3f}, err={result.mean_error:.4f}, time={result.query_time_us:.0f}μs")

    # Test 4: Similar keys
    print("\n" + "="*60)
    print("TEST 4: Similar Keys (perturbations of same vector)")
    print("="*60)
    n = 1000
    base = np.random.randn(d)
    base = base / np.linalg.norm(base)
    K = np.tile(base, (n, 1)) + np.random.randn(n, d) * 0.1
    V = np.arange(n).reshape(-1, 1) * np.ones((1, d))
    test_idx = [0, 100, 500, 999]
    queries = [K[i] for i in test_idx]
    targets = [float(i) for i in test_idx]

    for name, model_fn in models.items():
        holo = model_fn()
        result = run_test(holo, K, V, queries, targets, name)
        print(f"  {name:20s}: acc={result.accuracy:.3f}, err={result.mean_error:.4f}, time={result.query_time_us:.0f}μs")

    # Test 5: High dimension
    print("\n" + "="*60)
    print("TEST 5: High Dimension (d=512, n=5000)")
    print("="*60)
    d_high = 512
    n = 5000
    K = np.random.randn(n, d_high)
    V = np.zeros((n, d_high))
    K[n//2] = np.ones(d_high) * 0.5
    V[n//2] = np.ones(d_high) * 42.0
    queries = [np.ones(d_high) * 0.5]
    targets = [42.0]

    models_hd = {
        'V1 (original)': lambda: HolographicAttention(d_high, n_features=2048, seed=42),
        'V2 (multi-table)': lambda: HolographicAttentionV2(d_high, n_features=512, n_tables=8, seed=42),
        'V3 (hierarchical)': lambda: HolographicAttentionV3(d_high, n_features=256, n_buckets=64, n_tables=4, seed=42),
        'V4 (polynomial)': lambda: HolographicAttentionV4(d_high, n_features=512, n_scales=4, seed=42),
        'V5 (residual)': lambda: HolographicAttentionV5(d_high, n_features=1024, n_heads=4, seed=42),
    }

    for name, model_fn in models_hd.items():
        holo = model_fn()
        result = run_test(holo, K, V, queries, targets, name)
        print(f"  {name:20s}: acc={result.accuracy:.3f}, err={result.mean_error:.4f}, time={result.query_time_us:.0f}μs")

    # Test 6: O(1) scaling verification
    print("\n" + "="*60)
    print("TEST 6: O(1) Scaling (query time vs sequence length)")
    print("="*60)
    d = 64
    sizes = [1000, 10000, 50000, 100000]

    for name, model_fn_base in [
        ('V1', lambda n: HolographicAttention(d, n_features=1024, seed=42)),
        ('V2', lambda n: HolographicAttentionV2(d, n_features=512, n_tables=4, seed=42)),
    ]:
        times = []
        for n in sizes:
            K = np.random.randn(n, d)
            V = np.random.randn(n, d)
            q = np.random.randn(d)

            holo = model_fn_base(n)
            holo.ingest(K, V)

            # Warm up
            holo.query(q)

            # Measure
            t_list = []
            for _ in range(10):
                t0 = time.perf_counter()
                holo.query(q)
                t_list.append((time.perf_counter() - t0) * 1e6)
            times.append(np.median(t_list))

        print(f"  {name}: ", end="")
        for n, t in zip(sizes, times):
            print(f"n={n//1000}k:{t:.0f}μs  ", end="")
        ratio = times[-1] / times[0]
        print(f"[ratio 100k/1k: {ratio:.2f}x]")

    # Test 7: Memory efficiency
    print("\n" + "="*60)
    print("TEST 7: Memory Efficiency (compression ratio)")
    print("="*60)
    d = 64
    n = 10000
    K = np.random.randn(n, d)
    V = np.random.randn(n, d)

    for name, model_fn in models.items():
        holo = model_fn()
        holo.ingest(K, V)
        mem = holo.memory_bytes()
        ratio = holo.compression_ratio()
        print(f"  {name:20s}: {mem/1024:.0f}KB, {ratio:.1f}x compression")


if __name__ == "__main__":
    compare_versions()
