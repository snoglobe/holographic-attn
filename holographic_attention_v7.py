"""
Holographic Attention V7 - Position-Augmented Keys

Key insight: Similar keys are fundamentally indistinguishable in holographic
space. But if we augment keys with positional information, we can retrieve
based on both content AND position.

This version adds:
1. Position encoding mixed into keys
2. Separate content and position holograms
3. Content-position joint retrieval
"""

import numpy as np
from typing import Optional, Dict, Tuple


class PositionAugmentedHolographic:
    """
    V7: Position-augmented holographic attention.

    Augments keys with positional encodings to disambiguate similar keys
    at different positions.
    """

    def __init__(
        self,
        d_key: int,
        d_value: Optional[int] = None,
        n_features: int = 2048,
        max_seq_len: int = 100000,
        pos_weight: float = 0.3,
        seed: Optional[int] = None
    ):
        self.d_k = d_key
        self.d_v = d_value or d_key
        self.m = n_features
        self.max_seq_len = max_seq_len
        self.pos_weight = pos_weight

        rng = np.random.default_rng(seed)

        # Content projection
        self.W_content = rng.standard_normal((n_features, d_key))
        self.W_content /= np.linalg.norm(self.W_content, axis=1, keepdims=True)

        # Position encoding dimension
        self.d_pos = min(64, d_key)

        # Position projection
        self.W_pos = rng.standard_normal((n_features, self.d_pos))
        self.W_pos /= np.linalg.norm(self.W_pos, axis=1, keepdims=True)

        self.caches = {}
        self.n_tokens = 0

    def _positional_encoding(self, positions: np.ndarray) -> np.ndarray:
        """Sinusoidal positional encodings."""
        positions = np.atleast_1d(positions)
        pe = np.zeros((len(positions), self.d_pos))

        for i, pos in enumerate(positions):
            for j in range(self.d_pos // 2):
                freq = 1 / (10000 ** (2 * j / self.d_pos))
                pe[i, 2*j] = np.sin(pos * freq)
                pe[i, 2*j + 1] = np.cos(pos * freq)

        return pe

    def _phi_content(self, x: np.ndarray, sharpness: float) -> np.ndarray:
        """Content-based features."""
        x = np.atleast_2d(x)
        x_norm = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
        x_unit = x / x_norm

        proj = x_unit @ self.W_content.T * np.sqrt(self.d_k)
        feat = np.exp(proj - proj.max(-1, keepdims=True))
        feat = np.power(feat, sharpness)
        feat = feat / (feat.sum(-1, keepdims=True) + 1e-8)
        feat = feat * (x_norm ** 2)

        return feat

    def _phi_position(self, pos_enc: np.ndarray, sharpness: float) -> np.ndarray:
        """Position-based features."""
        pos_enc = np.atleast_2d(pos_enc)

        proj = pos_enc @ self.W_pos.T * np.sqrt(self.d_pos)
        feat = np.exp(proj - proj.max(-1, keepdims=True))
        feat = np.power(feat, sharpness)
        feat = feat / (feat.sum(-1, keepdims=True) + 1e-8)

        return feat

    def _phi_joint(self, x: np.ndarray, pos_enc: np.ndarray, sharpness: float) -> np.ndarray:
        """Joint content-position features."""
        phi_c = self._phi_content(x, sharpness)
        phi_p = self._phi_position(pos_enc, sharpness)

        # Combine via weighted product
        alpha = self.pos_weight
        combined = (1 - alpha) * phi_c + alpha * phi_p

        return combined

    def ingest(self, K: np.ndarray, V: np.ndarray, positions: Optional[np.ndarray] = None):
        """Ingest with position augmentation."""
        self.caches = {}
        self.n_tokens = len(K)

        if positions is None:
            positions = np.arange(len(K))

        pos_enc = self._positional_encoding(positions)

        for sharpness in [2, 4, 8, 16]:
            # Content-only hologram
            phi_c = self._phi_content(K, sharpness)
            H_c = phi_c.T @ V
            Z_c = phi_c.sum(0)

            # Joint hologram
            phi_j = self._phi_joint(K, pos_enc, sharpness)
            H_j = phi_j.T @ V
            Z_j = phi_j.sum(0)

            self.caches[sharpness] = {
                'content': (H_c, Z_c),
                'joint': (H_j, Z_j)
            }

        # Store for position-aware queries
        self.positions = positions
        self.K_stored = K

    def query(self, q: np.ndarray, query_position: Optional[int] = None) -> np.ndarray:
        """
        Query with optional position awareness.

        Args:
            q: Query vector
            query_position: If provided, uses position-augmented retrieval
        """
        q = np.atleast_1d(q)

        best_output = None
        best_conf = -np.inf

        for sharpness in [2, 4, 8, 16]:
            # Content-only query
            q_feat = self._phi_content(q.reshape(1, -1), sharpness)[0]
            H, Z = self.caches[sharpness]['content']

            denom = q_feat @ Z + 1e-8
            output = (q_feat @ H) / denom

            attn = q_feat * Z
            attn = attn / (attn.sum() + 1e-8)
            conf = np.max(attn)

            if conf > best_conf:
                best_conf = conf
                best_output = output

            # Joint query (if position provided)
            if query_position is not None:
                q_pos = self._positional_encoding(np.array([query_position]))[0]
                q_feat_j = self._phi_joint(q.reshape(1, -1), q_pos.reshape(1, -1), sharpness)[0]
                H_j, Z_j = self.caches[sharpness]['joint']

                denom_j = q_feat_j @ Z_j + 1e-8
                output_j = (q_feat_j @ H_j) / denom_j

                attn_j = q_feat_j * Z_j
                attn_j = attn_j / (attn_j.sum() + 1e-8)
                conf_j = np.max(attn_j)

                if conf_j > best_conf:
                    best_conf = conf_j
                    best_output = output_j

        return best_output

    def query_by_position(self, position: int) -> np.ndarray:
        """Query by position alone (for position-based retrieval)."""
        pos_enc = self._positional_encoding(np.array([position]))[0]

        # Find nearest position in stored data
        stored_pe = self._positional_encoding(self.positions)
        sims = pos_enc @ stored_pe.T
        nearest_idx = np.argmax(sims)

        # Use that key for content-based retrieval
        return self.query(self.K_stored[nearest_idx], query_position=position)

    def query_batch(self, Q: np.ndarray) -> np.ndarray:
        return np.array([self.query(q) for q in Q])

    def memory_bytes(self) -> int:
        total = self.W_content.nbytes + self.W_pos.nbytes
        for cache in self.caches.values():
            for H, Z in cache.values():
                total += H.nbytes + Z.nbytes
        if hasattr(self, 'K_stored'):
            total += self.K_stored.nbytes
        return total


class ContentAddressedHolographic:
    """
    Content-addressed holographic attention.

    Instead of using key-value pairs, uses content hashing for
    associative memory-style retrieval.
    """

    def __init__(
        self,
        d_key: int,
        d_value: Optional[int] = None,
        n_features: int = 4096,
        n_hash_bits: int = 12,
        seed: Optional[int] = None
    ):
        self.d_k = d_key
        self.d_v = d_value or d_key
        self.m = n_features
        self.n_buckets = 2 ** n_hash_bits

        rng = np.random.default_rng(seed)

        # Hash projection for content addressing
        self.H_hash = rng.standard_normal((n_hash_bits, d_key))
        self.H_hash /= np.linalg.norm(self.H_hash, axis=1, keepdims=True)

        # Feature projection per bucket
        self.W = rng.standard_normal((n_features, d_key))
        self.W /= np.linalg.norm(self.W, axis=1, keepdims=True)

        self.bucket_caches = {}
        self.n_tokens = 0

    def _compute_hash(self, x: np.ndarray) -> int:
        """Compute hash bucket for a key."""
        x = np.atleast_1d(x)
        x_unit = x / (np.linalg.norm(x) + 1e-8)
        bits = (x_unit @ self.H_hash.T > 0).astype(int)
        return int((bits * (2 ** np.arange(len(bits)))).sum())

    def _phi(self, x: np.ndarray, sharpness: float) -> np.ndarray:
        """Compute features."""
        x = np.atleast_2d(x)
        x_norm = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
        x_unit = x / x_norm

        proj = x_unit @ self.W.T * np.sqrt(self.d_k)
        feat = np.exp(proj - proj.max(-1, keepdims=True))
        feat = np.power(feat, sharpness)
        feat = feat / (feat.sum(-1, keepdims=True) + 1e-8)
        feat = feat * (x_norm ** 2)

        return feat

    def ingest(self, K: np.ndarray, V: np.ndarray):
        """Ingest into content-addressed buckets."""
        self.bucket_caches = {}
        self.n_tokens = len(K)

        # Group by hash bucket
        buckets = {}
        for i, k in enumerate(K):
            h = self._compute_hash(k)
            if h not in buckets:
                buckets[h] = {'K': [], 'V': [], 'idx': []}
            buckets[h]['K'].append(K[i])
            buckets[h]['V'].append(V[i])
            buckets[h]['idx'].append(i)

        # Create hologram per bucket
        for bucket_id, data in buckets.items():
            K_b = np.array(data['K'])
            V_b = np.array(data['V'])

            bucket_cache = {}
            for sharpness in [4, 8, 16]:
                phi_K = self._phi(K_b, sharpness)
                H = phi_K.T @ V_b
                Z = phi_K.sum(0)
                bucket_cache[sharpness] = (H, Z)

            self.bucket_caches[bucket_id] = {
                'cache': bucket_cache,
                'size': len(K_b)
            }

    def query(self, q: np.ndarray) -> np.ndarray:
        """Query via content addressing."""
        q = np.atleast_1d(q)
        bucket_id = self._compute_hash(q)

        if bucket_id not in self.bucket_caches:
            # Fall back to random bucket
            bucket_id = list(self.bucket_caches.keys())[0]

        bucket = self.bucket_caches[bucket_id]

        best_output = None
        best_conf = -np.inf

        for sharpness in [4, 8, 16]:
            q_feat = self._phi(q.reshape(1, -1), sharpness)[0]
            H, Z = bucket['cache'][sharpness]

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
        total = self.H_hash.nbytes + self.W.nbytes
        for bucket in self.bucket_caches.values():
            for H, Z in bucket['cache'].values():
                total += H.nbytes + Z.nbytes
        return total


def test_v7():
    """Test V7 implementations."""
    import time

    np.random.seed(42)
    d, n = 64, 10000

    print("V7 Tests: Position-Augmented and Content-Addressed")
    print("=" * 60)

    # Test 1: Similar keys at different positions
    print("\n1. Similar Keys at Different Positions")
    base = np.random.randn(d)
    base = base / np.linalg.norm(base)

    K = np.tile(base, (n, 1)) + np.random.randn(n, d) * 0.05  # Very similar
    V = np.arange(n).reshape(-1, 1) * np.ones((1, d))  # Value = position

    # Test position-augmented
    holo = PositionAugmentedHolographic(d, n_features=2048, seed=42)
    holo.ingest(K, V)

    test_positions = [0, 100, 500, 5000, 9999]
    print("\n   Position-Augmented Retrieval:")
    for pos in test_positions:
        out = holo.query(K[pos], query_position=pos)
        error = abs(out[0] - pos)
        print(f"      pos={pos:5d}: got {out[0]:.1f}, error={error:.1f}")

    # Test content-addressed
    print("\n   Content-Addressed Retrieval:")
    ca_holo = ContentAddressedHolographic(d, n_features=2048, n_hash_bits=10, seed=42)
    ca_holo.ingest(K, V)

    for pos in test_positions:
        out = ca_holo.query(K[pos])
        error = abs(out[0] - pos)
        print(f"      pos={pos:5d}: got {out[0]:.1f}, error={error:.1f}")

    # Test 2: Needle in haystack (standard test)
    print("\n2. Needle in Haystack")
    K = np.random.randn(n, d) * 0.3
    V = np.zeros((n, d))
    K[n//2] = np.ones(d) * 2.0
    V[n//2] = np.ones(d) * 42.0

    holo = PositionAugmentedHolographic(d, n_features=2048, seed=42)
    holo.ingest(K, V)
    q = np.ones(d) * 2.0
    out = holo.query(q)
    print(f"   Position-Augmented: {out[0]:.2f}/42.0")

    ca_holo = ContentAddressedHolographic(d, n_features=2048, n_hash_bits=10, seed=42)
    ca_holo.ingest(K, V)
    out = ca_holo.query(q)
    print(f"   Content-Addressed: {out[0]:.2f}/42.0")

    # Test 3: O(1) verification
    print("\n3. O(1) Scaling")
    sizes = [1000, 10000, 50000]
    for n in sizes:
        K = np.random.randn(n, d)
        V = np.random.randn(n, d)
        q = np.random.randn(d)

        holo = PositionAugmentedHolographic(d, n_features=1024, seed=42)
        holo.ingest(K, V)

        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            holo.query(q)
            times.append((time.perf_counter() - t0) * 1e6)

        print(f"   n={n//1000:2d}k: {np.median(times):.0f}μs")


if __name__ == "__main__":
    test_v7()
