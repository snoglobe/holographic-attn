"""
LLM-Scale Stress Tests for Holographic Attention

Tests:
1. Long sequence handling (100k+ tokens)
2. Real-world attention patterns
3. Streaming/autoregressive generation
4. Multi-head integration
5. Memory pressure tests
"""

import numpy as np
import time
import gc
from typing import Dict, List, Tuple

from holographic_attention import HolographicAttention
from holographic_attention_v5 import HolographicAttentionV5, MultiHeadHolographicAttention


def test_long_sequence_scaling():
    """Test behavior at extreme sequence lengths."""
    print("\n" + "=" * 60)
    print("TEST: Long Sequence Scaling (1k to 500k tokens)")
    print("=" * 60)

    d = 64
    sizes = [1000, 10000, 50000, 100000, 250000, 500000]

    for version_name, create_model in [
        ("V1", lambda: HolographicAttention(d, n_features=1024, seed=42)),
        ("V5 (no residual)", lambda: HolographicAttentionV5(d, n_features=1024, n_heads=4, use_residual=False, seed=42)),
    ]:
        print(f"\n{version_name}:")
        for n in sizes:
            try:
                gc.collect()

                K = np.random.randn(n, d).astype(np.float32)
                V = np.random.randn(n, d).astype(np.float32)
                q = np.random.randn(d).astype(np.float32)

                holo = create_model()

                # Measure ingest time
                t0 = time.perf_counter()
                holo.ingest(K, V)
                ingest_time = (time.perf_counter() - t0) * 1000

                # Warm up
                holo.query(q)

                # Measure query time
                times = []
                for _ in range(10):
                    t0 = time.perf_counter()
                    holo.query(q)
                    times.append((time.perf_counter() - t0) * 1e6)

                query_time = np.median(times)
                mem = holo.memory_bytes() / (1024 * 1024)

                print(f"  n={n//1000:3d}k: ingest={ingest_time:7.1f}ms, query={query_time:6.0f}μs, mem={mem:.1f}MB")

                del K, V, holo
                gc.collect()

            except MemoryError:
                print(f"  n={n//1000:3d}k: OUT OF MEMORY")
                break


def test_autoregressive_generation():
    """Simulate autoregressive LLM generation."""
    print("\n" + "=" * 60)
    print("TEST: Autoregressive Generation Simulation")
    print("=" * 60)

    d = 64
    context_len = 4096
    gen_len = 100

    np.random.seed(42)

    # Initial context
    K_context = np.random.randn(context_len, d)
    V_context = np.random.randn(context_len, d)

    # V5 with causal mode
    holo = HolographicAttentionV5(d, n_features=1024, n_heads=4, use_residual=False, seed=42)
    holo.ingest(K_context, V_context, causal=True)

    # Simulate generation
    query_times = []
    update_times = []

    for t in range(gen_len):
        # New token's key/value
        k_new = np.random.randn(d)
        v_new = np.random.randn(d)

        # Query at current position
        q = np.random.randn(d)
        t0 = time.perf_counter()
        output = holo.query(q, position=context_len + t - 1)
        query_times.append((time.perf_counter() - t0) * 1e6)

        # Update with new token (simulated - actual update would need streaming)
        t0 = time.perf_counter()
        holo.update(k_new, v_new)
        update_times.append((time.perf_counter() - t0) * 1e6)

    print(f"Context length: {context_len}")
    print(f"Generation length: {gen_len}")
    print(f"Mean query time: {np.mean(query_times):.0f}μs")
    print(f"Mean update time: {np.mean(update_times):.0f}μs")
    print(f"Total per-token overhead: {np.mean(query_times) + np.mean(update_times):.0f}μs")


def test_realistic_attention_patterns():
    """Test with realistic LLM attention patterns."""
    print("\n" + "=" * 60)
    print("TEST: Realistic Attention Patterns")
    print("=" * 60)

    d = 64
    n = 4096

    np.random.seed(42)

    # Pattern 1: Recent bias (LLM typically attends more to recent tokens)
    print("\n1. Recent Bias Pattern:")
    K = np.random.randn(n, d)
    V = np.random.randn(n, d)

    # Make recent tokens more salient
    for i in range(n):
        recency = (i + 1) / n
        K[i] *= (0.5 + 0.5 * recency)

    holo = HolographicAttentionV5(d, n_features=1024, n_heads=4, use_residual=False, seed=42)
    holo.ingest(K, V)

    # Query should naturally prefer recent tokens
    q = np.random.randn(d)
    output = holo.query(q)
    print(f"  Output norm: {np.linalg.norm(output):.3f}")

    # Pattern 2: Sparse attention (only a few tokens are relevant)
    print("\n2. Sparse Attention Pattern:")
    K = np.random.randn(n, d) * 0.3
    V = np.zeros((n, d))

    # 10 important tokens spread throughout
    important_positions = np.linspace(0, n-1, 10, dtype=int)
    for i, pos in enumerate(important_positions):
        K[pos] = np.random.randn(d) * 2.0
        V[pos] = np.ones(d) * (i + 1) * 10

    holo = HolographicAttentionV5(d, n_features=1024, n_heads=4, use_residual=True, seed=42)
    holo.ingest(K, V)

    errors = []
    for i, pos in enumerate(important_positions):
        q = K[pos]
        output = holo.query(q)
        expected = (i + 1) * 10
        errors.append(abs(output[0] - expected))

    print(f"  Mean retrieval error: {np.mean(errors):.2f}")
    print(f"  Max retrieval error: {np.max(errors):.2f}")

    # Pattern 3: Clustered attention (semantic clustering)
    print("\n3. Clustered Attention Pattern:")
    n_clusters = 20
    cluster_size = n // n_clusters

    K = np.zeros((n, d))
    V = np.zeros((n, d))

    for c in range(n_clusters):
        center = np.random.randn(d)
        center = center / np.linalg.norm(center) * 2
        start = c * cluster_size
        end = start + cluster_size
        K[start:end] = center + np.random.randn(cluster_size, d) * 0.2
        V[start:end] = np.ones((cluster_size, d)) * c

    holo = HolographicAttentionV5(d, n_features=1024, n_heads=4, use_residual=False, seed=42)
    holo.ingest(K, V)

    # Query cluster centers
    errors = []
    for c in range(n_clusters):
        pos = c * cluster_size + cluster_size // 2
        q = K[pos]
        output = holo.query(q)
        errors.append(abs(output[0] - c))

    print(f"  Mean cluster retrieval error: {np.mean(errors):.2f}")


def test_multi_head_integration():
    """Test multi-head holographic attention as LLM layer."""
    print("\n" + "=" * 60)
    print("TEST: Multi-Head Holographic Attention Layer")
    print("=" * 60)

    d_model = 512
    n_heads = 8
    seq_len = 2048

    np.random.seed(42)
    X = np.random.randn(seq_len, d_model)

    mha = MultiHeadHolographicAttention(
        d_model=d_model,
        n_heads=n_heads,
        n_features=256,
        causal=True,
        seed=42
    )

    t0 = time.perf_counter()
    output = mha.forward(X)
    forward_time = (time.perf_counter() - t0) * 1000

    print(f"Model dimension: {d_model}")
    print(f"Attention heads: {n_heads}")
    print(f"Sequence length: {seq_len}")
    print(f"Forward pass time: {forward_time:.1f}ms")
    print(f"Output shape: {output.shape}")
    print(f"Memory usage: {mha.memory_bytes() / 1024:.0f}KB")


def test_memory_pressure():
    """Test behavior under memory pressure."""
    print("\n" + "=" * 60)
    print("TEST: Memory Pressure (multiple concurrent contexts)")
    print("=" * 60)

    d = 64
    n = 10000
    n_contexts = 10

    np.random.seed(42)

    # Create multiple attention contexts (like batch processing)
    contexts = []
    for i in range(n_contexts):
        K = np.random.randn(n, d)
        V = np.random.randn(n, d)

        holo = HolographicAttentionV5(d, n_features=512, n_heads=2, use_residual=False, seed=42+i)
        holo.ingest(K, V)
        contexts.append(holo)

    total_mem = sum(h.memory_bytes() for h in contexts) / (1024 * 1024)
    print(f"Contexts: {n_contexts}")
    print(f"Tokens per context: {n}")
    print(f"Total memory: {total_mem:.1f}MB")

    # Measure query across all contexts
    q = np.random.randn(d)
    t0 = time.perf_counter()
    for holo in contexts:
        holo.query(q)
    batch_time = (time.perf_counter() - t0) * 1e6

    print(f"Batch query time ({n_contexts} contexts): {batch_time:.0f}μs")
    print(f"Per-context query time: {batch_time/n_contexts:.0f}μs")


def test_accuracy_vs_features():
    """Test how accuracy scales with number of features."""
    print("\n" + "=" * 60)
    print("TEST: Accuracy vs Number of Features")
    print("=" * 60)

    d = 64
    n = 5000
    feature_counts = [128, 256, 512, 1024, 2048, 4096]

    np.random.seed(42)

    # Needle in haystack setup
    K = np.random.randn(n, d) * 0.3
    V = np.zeros((n, d))
    K[n//2] = np.ones(d) * 2.0
    V[n//2] = np.ones(d) * 42.0
    q = np.ones(d) * 2.0

    print(f"{'Features':>8} {'Accuracy':>10} {'Error':>10} {'Time':>10} {'Memory':>10}")
    print("-" * 50)

    for m in feature_counts:
        holo = HolographicAttentionV5(d, n_features=m, n_heads=4, use_residual=False, seed=42)
        holo.ingest(K, V)

        t0 = time.perf_counter()
        output = holo.query(q)
        query_time = (time.perf_counter() - t0) * 1e6

        error = abs(output[0] - 42.0)
        accuracy = max(0, 1 - error / 42)
        mem = holo.memory_bytes() / 1024

        print(f"{m:>8} {accuracy:>10.3f} {error:>10.4f} {query_time:>8.0f}μs {mem:>8.0f}KB")


def main():
    print("=" * 60)
    print("HOLOGRAPHIC ATTENTION LLM-SCALE STRESS TESTS")
    print("=" * 60)

    test_accuracy_vs_features()
    test_long_sequence_scaling()
    test_autoregressive_generation()
    test_realistic_attention_patterns()
    test_multi_head_integration()
    test_memory_pressure()

    print("\n" + "=" * 60)
    print("ALL STRESS TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
