"""
Holographic Attention V6 - Improved Core Algorithm

Key insight: The random feature approach has fundamental limitations.
This version uses:
1. Learned-equivalent structured projections
2. Sign-based hashing for collision reduction
3. Exponential Moving Average for recency bias
4. Optional exact top-k refinement (configurable)
"""

import numpy as np
from typing import Optional, Dict, Tuple, List


class HolographicAttentionV6:
    """
    V6: Improved core algorithm with better discrimination.

    Uses structured projections and sign hashing for better key separation.
    """

    def __init__(
        self,
        d_key: int,
        d_value: Optional[int] = None,
        n_features: int = 2048,
        n_hash_tables: int = 16,
        temperature: float = 1.0,
        use_ema: bool = False,
        ema_decay: float = 0.99,
        seed: Optional[int] = None
    ):
        self.d_k = d_key
        self.d_v = d_value or d_key
        self.m = n_features
        self.n_tables = n_hash_tables
        self.temperature = temperature
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        rng = np.random.default_rng(seed)

        # Structured random projections with orthogonalization
        self.W = rng.standard_normal((n_features, d_key))
        # Partial orthogonalization via QR
        if n_features <= d_key:
            Q, _ = np.linalg.qr(self.W.T)
            self.W = Q.T
        else:
            # Block orthogonalization
            block_size = d_key
            for i in range(0, n_features, block_size):
                end = min(i + block_size, n_features)
                if end - i >= 2:
                    Q, _ = np.linalg.qr(self.W[i:end].T)
                    self.W[i:end] = Q.T[:end-i]

        # Sign hash vectors for LSH-style bucketing
        self.hash_vecs = rng.standard_normal((n_hash_tables, d_key))
        self.hash_vecs /= np.linalg.norm(self.hash_vecs, axis=1, keepdims=True)

        # Learnable-equivalent bias terms
        self.bias = rng.standard_normal(n_features) * 0.1

        self.caches: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.n_tokens = 0

    def _compute_hash(self, x: np.ndarray) -> np.ndarray:
        """Compute locality-sensitive hash signature."""
        x = np.atleast_2d(x)
        # Sign of projection gives hash bits
        signs = np.sign(x @ self.hash_vecs.T)  # (n, n_tables)
        # Convert to bucket index
        bucket = ((signs + 1) / 2).astype(int)
        # Combine bits into single hash
        powers = 2 ** np.arange(self.n_tables)
        return (bucket * powers).sum(-1)  # (n,)

    def _phi(self, x: np.ndarray, sharpness: float = 1.0) -> np.ndarray:
        """
        Compute random features with improved stability.

        Uses ReLU-based features which are more stable than exp.
        """
        x = np.atleast_2d(x)

        # Normalize input
        x_norm = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
        x_unit = x / x_norm

        # Project
        proj = x_unit @ self.W.T + self.bias

        # ReLU features (more stable than softmax)
        feat = np.maximum(0, proj) ** sharpness

        # Also add negative ReLU for full coverage
        feat_neg = np.maximum(0, -proj) ** sharpness
        feat = np.concatenate([feat, feat_neg], axis=-1)

        # Normalize and weight by input norm
        feat = feat / (feat.sum(-1, keepdims=True) + 1e-8)
        feat = feat * (x_norm ** 2)

        return feat

    def _phi_softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Softmax-based features for comparison."""
        x = np.atleast_2d(x)

        x_norm = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
        x_unit = x / x_norm

        proj = x_unit @ self.W.T / temperature

        # Stable softmax
        feat = np.exp(proj - proj.max(-1, keepdims=True))
        feat = feat / (feat.sum(-1, keepdims=True) + 1e-8)
        feat = feat * (x_norm ** 2)

        return feat

    def ingest(self, K: np.ndarray, V: np.ndarray):
        """Ingest with multiple sharpness levels."""
        self.caches = {}
        self.n_tokens = len(K)

        # Store hash buckets for potential refinement
        self.key_hashes = self._compute_hash(K)
        self.K_stored = K.copy()
        self.V_stored = V.copy()

        # Create holograms at different sharpness levels
        for sharpness in [0.5, 1.0, 2.0, 4.0]:
            phi_K = self._phi(K, sharpness)
            H = phi_K.T @ V
            Z = phi_K.sum(0)
            self.caches[sharpness] = (H, Z)

        # Also create softmax-based hologram
        for temp in [0.5, 1.0, 2.0]:
            phi_K = self._phi_softmax(K, temp)
            H = phi_K.T @ V
            Z = phi_K.sum(0)
            self.caches[('softmax', temp)] = (H, Z)

    def query(self, q: np.ndarray, use_refinement: bool = True) -> np.ndarray:
        """Query with adaptive selection and optional refinement."""
        q = np.atleast_1d(q)

        best_output = None
        best_confidence = -np.inf

        # Try ReLU features
        for sharpness in [0.5, 1.0, 2.0, 4.0]:
            q_feat = self._phi(q.reshape(1, -1), sharpness)[0]
            H, Z = self.caches[sharpness]

            denom = q_feat @ Z + 1e-8
            output = (q_feat @ H) / denom

            # Confidence metric
            attn = q_feat * Z
            attn = attn / (attn.sum() + 1e-8)
            conf = np.max(attn) + 0.1 * sharpness

            if conf > best_confidence:
                best_confidence = conf
                best_output = output

        # Try softmax features
        for temp in [0.5, 1.0, 2.0]:
            q_feat = self._phi_softmax(q.reshape(1, -1), temp)[0]
            H, Z = self.caches[('softmax', temp)]

            denom = q_feat @ Z + 1e-8
            output = (q_feat @ H) / denom

            attn = q_feat * Z
            attn = attn / (attn.sum() + 1e-8)
            conf = np.max(attn)

            if conf > best_confidence:
                best_confidence = conf
                best_output = output

        # Optional: LSH-based refinement (O(n/buckets) average)
        if use_refinement and self.n_tokens > 0:
            q_hash = self._compute_hash(q.reshape(1, -1))[0]

            # Find keys in same bucket
            same_bucket = self.key_hashes == q_hash
            if same_bucket.sum() > 0 and same_bucket.sum() < self.n_tokens * 0.1:
                # Do exact attention on bucket members
                K_bucket = self.K_stored[same_bucket]
                V_bucket = self.V_stored[same_bucket]

                scores = q @ K_bucket.T
                weights = np.exp(scores - scores.max())
                weights = weights / weights.sum()

                bucket_output = weights @ V_bucket

                # Blend based on bucket confidence
                bucket_conf = np.max(weights)
                blend = bucket_conf ** 2
                best_output = (1 - blend) * best_output + blend * bucket_output

        return best_output

    def query_batch(self, Q: np.ndarray) -> np.ndarray:
        return np.array([self.query(q) for q in Q])

    def memory_bytes(self) -> int:
        total = self.W.nbytes + self.hash_vecs.nbytes + self.bias.nbytes
        for H, Z in self.caches.values():
            total += H.nbytes + Z.nbytes
        if hasattr(self, 'K_stored'):
            total += self.K_stored.nbytes + self.V_stored.nbytes
        return total

    def compression_ratio(self) -> float:
        if self.n_tokens == 0:
            return float('inf')
        standard = self.n_tokens * (self.d_k + self.d_v) * 8
        return standard / self.memory_bytes()


class PureHolographicV6:
    """
    Pure O(1) version without any refinement.

    Uses ensemble of different feature types for robustness.
    """

    def __init__(
        self,
        d_key: int,
        d_value: Optional[int] = None,
        n_features: int = 4096,
        seed: Optional[int] = None
    ):
        self.d_k = d_key
        self.d_v = d_value or d_key
        self.m = n_features

        rng = np.random.default_rng(seed)

        # Multiple projection types
        self.W_random = rng.standard_normal((n_features, d_key))
        self.W_random /= np.linalg.norm(self.W_random, axis=1, keepdims=True)

        # Hadamard-like structured projection
        self.W_struct = self._create_structured_proj(n_features, d_key, rng)

        self.caches = {}
        self.n_tokens = 0

    def _create_structured_proj(self, m: int, d: int, rng) -> np.ndarray:
        """Create structured random projection."""
        # Simple: random signs * random permutation
        signs = rng.choice([-1, 1], size=(m, d))
        return signs / np.sqrt(d)

    def _phi_ensemble(self, x: np.ndarray, sharpness: float) -> np.ndarray:
        """Ensemble of feature types."""
        x = np.atleast_2d(x)
        x_norm = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
        x_unit = x / x_norm

        features = []

        # Random projection features
        proj1 = x_unit @ self.W_random.T * np.sqrt(self.d_k)
        feat1 = np.exp(proj1 - proj1.max(-1, keepdims=True))
        feat1 = feat1 ** sharpness
        feat1 = feat1 / (feat1.sum(-1, keepdims=True) + 1e-8)
        features.append(feat1)

        # Structured projection features
        proj2 = x_unit @ self.W_struct.T * np.sqrt(self.d_k)
        feat2 = np.exp(proj2 - proj2.max(-1, keepdims=True))
        feat2 = feat2 ** sharpness
        feat2 = feat2 / (feat2.sum(-1, keepdims=True) + 1e-8)
        features.append(feat2)

        # Combine
        feat = np.concatenate(features, axis=-1)
        feat = feat * (x_norm ** 2)

        return feat

    def ingest(self, K: np.ndarray, V: np.ndarray):
        self.caches = {}
        self.n_tokens = len(K)

        for sharpness in [2, 4, 8, 16, 32]:
            phi_K = self._phi_ensemble(K, sharpness)
            H = phi_K.T @ V
            Z = phi_K.sum(0)
            self.caches[sharpness] = (H, Z)

    def query(self, q: np.ndarray) -> np.ndarray:
        q = np.atleast_1d(q)

        best_output = None
        best_conf = -np.inf

        for sharpness in [2, 4, 8, 16, 32]:
            q_feat = self._phi_ensemble(q.reshape(1, -1), sharpness)[0]
            H, Z = self.caches[sharpness]

            denom = q_feat @ Z + 1e-8
            output = (q_feat @ H) / denom

            attn = q_feat * Z
            attn = attn / (attn.sum() + 1e-8)
            conf = np.max(attn)

            if conf > best_conf:
                best_conf = conf
                best_output = output

        return best_output

    def query_batch(self, Q: np.ndarray) -> np.ndarray:
        return np.array([self.query(q) for q in Q])

    def memory_bytes(self) -> int:
        total = self.W_random.nbytes + self.W_struct.nbytes
        for H, Z in self.caches.values():
            total += H.nbytes + Z.nbytes
        return total


def quick_test():
    """Quick accuracy test."""
    import time

    np.random.seed(42)
    d, n = 64, 10000

    K = np.random.randn(n, d) * 0.3
    V = np.zeros((n, d))
    K[n//2] = np.ones(d) * 2.0
    V[n//2] = np.ones(d) * 42.0
    q = np.ones(d) * 2.0

    print("Quick Test: Needle in Haystack")
    print("-" * 40)

    # V6 with refinement
    holo = HolographicAttentionV6(d, n_features=2048, seed=42)
    holo.ingest(K, V)
    out = holo.query(q, use_refinement=True)
    print(f"V6 (with refinement): {out[0]:.2f}")

    # V6 without refinement
    out_no_ref = holo.query(q, use_refinement=False)
    print(f"V6 (no refinement): {out_no_ref[0]:.2f}")

    # Pure holographic with varying features
    print("\nPure V6 vs feature count:")
    for m in [1024, 2048, 4096, 8192]:
        pure = PureHolographicV6(d, n_features=m, seed=42)
        pure.ingest(K, V)
        t0 = time.perf_counter()
        out_pure = pure.query(q)
        query_time = (time.perf_counter() - t0) * 1e6
        mem = pure.memory_bytes() / 1024
        print(f"  m={m}: {out_pure[0]:.2f} (target=42), time={query_time:.0f}μs, mem={mem:.0f}KB")

    # Test multiple needles
    print("\nMultiple Needles Test:")
    K = np.random.randn(n, d) * 0.3
    V = np.random.randn(n, d) * 0.1
    positions = [100, 500, 1000, 5000, 9000]
    for i, pos in enumerate(positions):
        direction = np.random.randn(d)
        direction = direction / np.linalg.norm(direction) * 3.0
        K[pos] = direction
        V[pos] = np.ones(d) * (i + 1) * 10

    pure = PureHolographicV6(d, n_features=8192, seed=42)
    pure.ingest(K, V)

    errors = []
    for i, pos in enumerate(positions):
        out = pure.query(K[pos])
        expected = (i + 1) * 10
        error = abs(out[0] - expected)
        errors.append(error)
        print(f"  pos={pos}: got {out[0]:.2f}, expected {expected}, error={error:.2f}")
    print(f"  Mean error: {np.mean(errors):.2f}")


if __name__ == "__main__":
    quick_test()
