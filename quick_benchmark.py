"""Quick benchmark comparison: Original vs Production."""

import numpy as np
import time
from holographic_attention import HolographicAttention
from holographic_attention_prod import HolographicAttentionProd


def benchmark(name, holo_fn, d=64, n=10000, seed=42):
    """Run key benchmarks."""
    np.random.seed(seed)

    results = {}

    # Test 1: Needle in haystack
    K = np.random.randn(n, d) * 0.3
    V = np.zeros((n, d))
    K[n//2] = np.ones(d) * 2.0
    V[n//2] = np.ones(d) * 42.0

    holo = holo_fn(d)
    holo.ingest(K, V)

    q = np.ones(d) * 2.0
    out = holo.query(q)
    results['needle'] = out[0]

    # Test 2: Multiple needles
    K = np.random.randn(n, d) * 0.3
    V = np.random.randn(n, d) * 0.1
    positions = [100, 1000, 5000, 9000]
    values = [10, 20, 30, 40]
    for pos, val in zip(positions, values):
        direction = np.random.randn(d)
        direction = direction / np.linalg.norm(direction) * 3.0
        K[pos] = direction
        V[pos] = np.ones(d) * val

    holo = holo_fn(d)
    holo.ingest(K, V)

    errors = []
    for pos, val in zip(positions, values):
        out = holo.query(K[pos])
        errors.append(abs(out[0] - val))
    results['multi_needle_err'] = np.mean(errors)

    # Test 3: Query time
    K = np.random.randn(n, d)
    V = np.random.randn(n, d)
    holo = holo_fn(d)
    holo.ingest(K, V)

    q = np.random.randn(d)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        holo.query(q)
        times.append((time.perf_counter() - t0) * 1e6)
    results['query_time'] = np.median(times)

    # Test 4: Memory
    results['memory_kb'] = holo.memory_bytes() / 1024

    return results


def main():
    print("=" * 60)
    print("QUICK BENCHMARK: Original vs Production")
    print("=" * 60)

    configs = [
        ("Original (1024 feat)", lambda d: HolographicAttention(d, n_features=1024, seed=42)),
        ("Original (4096 feat)", lambda d: HolographicAttention(d, n_features=4096, seed=42)),
        ("Production (2048 feat)", lambda d: HolographicAttentionProd(d, n_features=2048, seed=42)),
        ("Production (4096 feat)", lambda d: HolographicAttentionProd(d, n_features=4096, seed=42)),
    ]

    print(f"\n{'Config':<25} {'Needle':>10} {'MultiErr':>10} {'Time(μs)':>10} {'Mem(KB)':>10}")
    print("-" * 70)

    for name, holo_fn in configs:
        results = benchmark(name, holo_fn)
        print(f"{name:<25} {results['needle']:>10.2f} {results['multi_needle_err']:>10.2f} "
              f"{results['query_time']:>10.0f} {results['memory_kb']:>10.0f}")

    print("\nTarget: Needle=42.0, MultiErr=0.0")


if __name__ == "__main__":
    main()
