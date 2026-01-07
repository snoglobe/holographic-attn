"""
Holographic Attention V2 - Improved O(1) Query Complexity

Key improvements over V1:
1. Multi-table hashing for collision reduction
2. Learnable-equivalent temperature adaptation
3. Dimension-adaptive feature scaling
4. Hierarchical resolution with refinement
5. Better numerical stability for extreme dimensions
"""

import numpy as np
from typing import Optional, Dict, Tuple, List


class HolographicAttentionV2:
    """
    Improved holographic attention with better collision handling.

    Uses multiple independent hash tables to reduce the probability
    of key collisions, similar to locality-sensitive hashing.
    """

    def __init__(
        self,
        d_key: int,
        d_value: Optional[int] = None,
        n_features: int = 512,
        n_tables: int = 8,
        base_sharpness: float = 4.0,
        seed: Optional[int] = None
    ):
        self.d_k = d_key
        self.d_v = d_value or d_key
        self.m = n_features
        self.n_tables = n_tables
        self.base_sharpness = base_sharpness

        rng = np.random.default_rng(seed)

        # Multiple independent projection tables
        self.projections: List[np.ndarray] = []
        self.proj_norms: List[np.ndarray] = []

        for _ in range(n_tables):
            # Orthogonalized random projection
            P = rng.standard_normal((n_features, d_key))
            # Normalize rows for stability
            norms = np.linalg.norm(P, axis=1, keepdims=True)
            P = P / (norms + 1e-8)
            self.projections.append(P)
            self.proj_norms.append(norms.flatten())

        # Dimension-adaptive scaling
        self.dim_scale = np.sqrt(d_key)

        # Sharpness levels for adaptive selection
        self.sharpness_levels = [1, 2, 4, 8, 16, 32]

        # Cache structure: table_idx -> sharpness -> (H, Z)
        self.caches: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self.n_tokens = 0

    def _phi(self, x: np.ndarray, table_idx: int, sharpness: float) -> np.ndarray:
        """
        Compute features for a specific hash table.

        Uses normalized dot product with adaptive temperature.
        """
        x = np.atleast_2d(x)
        P = self.projections[table_idx]

        # Compute normalized projection
        x_norm = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
        x_unit = x / x_norm

        # Scaled cosine similarity
        proj = x_unit @ P.T

        # Temperature scaling based on dimension
        temp = self.base_sharpness * sharpness / self.dim_scale

        # Softmax with temperature
        scaled = proj * temp * self.dim_scale
        feat = np.exp(scaled - scaled.max(-1, keepdims=True))
        feat = feat ** min(sharpness, 32)  # Additional sharpening
        feat = feat / (feat.sum(-1, keepdims=True) + 1e-8)

        # Weight by original norm (keys with larger norms are more "important")
        feat = feat * (x_norm ** 2)

        return feat

    def _phi_with_position(self, x: np.ndarray, table_idx: int, sharpness: float,
                           add_position: bool = False, positions: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute features with optional positional encoding."""
        base_feat = self._phi(x, table_idx, sharpness)

        if not add_position or positions is None:
            return base_feat

        # Add sinusoidal position encoding to features
        n = len(positions)
        pos_enc = np.zeros((n, self.m))
        for i, pos in enumerate(positions):
            for j in range(self.m // 2):
                freq = 1 / (10000 ** (2 * j / self.m))
                pos_enc[i, 2*j] = np.sin(pos * freq)
                pos_enc[i, 2*j + 1] = np.cos(pos * freq)

        return base_feat * (1 + 0.1 * pos_enc)

    def ingest(self, K: np.ndarray, V: np.ndarray):
        """Ingest key-value pairs into multiple hash tables."""
        self.caches = {}
        self.n_tokens = len(K)

        for table_idx in range(self.n_tables):
            for sharpness in self.sharpness_levels:
                phi_K = self._phi(K, table_idx, sharpness)

                H = phi_K.T @ V
                Z = phi_K.sum(0)

                self.caches[(table_idx, sharpness)] = (H, Z)

    def query(self, q: np.ndarray) -> np.ndarray:
        """
        Query with multi-table consensus and adaptive sharpness.

        Aggregates results from all tables, weighted by confidence.
        """
        q = np.atleast_1d(q)

        # Collect outputs from all tables and sharpness levels
        outputs = []
        confidences = []

        for table_idx in range(self.n_tables):
            best_output = None
            best_conf = -np.inf

            for sharpness in self.sharpness_levels:
                q_feat = self._phi(q.reshape(1, -1), table_idx, sharpness)[0]
                H, Z = self.caches[(table_idx, sharpness)]

                # Compute output
                denom = q_feat @ Z + 1e-8
                output = (q_feat @ H) / denom

                # Confidence: how peaked is the attention?
                attn = q_feat * Z
                attn = attn / (attn.sum() + 1e-8)
                conf = np.max(attn)

                if conf > best_conf:
                    best_conf = conf
                    best_output = output

            outputs.append(best_output)
            confidences.append(best_conf)

        # Weighted average by confidence (softmax weighting)
        outputs = np.array(outputs)
        confidences = np.array(confidences)

        # Sharpen confidence weights
        conf_weights = np.exp(confidences * 10)
        conf_weights = conf_weights / conf_weights.sum()

        return (conf_weights[:, None] * outputs).sum(0)

    def query_batch(self, Q: np.ndarray) -> np.ndarray:
        """Batch query."""
        return np.array([self.query(q) for q in Q])

    def memory_bytes(self) -> int:
        """Total memory usage."""
        total = sum(P.nbytes for P in self.projections)
        for H, Z in self.caches.values():
            total += H.nbytes + Z.nbytes
        return total

    def compression_ratio(self) -> float:
        """Compression vs standard KV cache."""
        if self.n_tokens == 0:
            return float('inf')
        standard = self.n_tokens * (self.d_k + self.d_v) * 8
        return standard / self.memory_bytes()


class HolographicAttentionV3:
    """
    V3: Hierarchical holographic attention with learned-equivalent bucketing.

    Key innovation: Uses multiple resolution levels and a voting mechanism
    to achieve both soft attention AND sharp retrieval.
    """

    def __init__(
        self,
        d_key: int,
        d_value: Optional[int] = None,
        n_features: int = 256,
        n_buckets: int = 64,
        n_tables: int = 4,
        seed: Optional[int] = None
    ):
        self.d_k = d_key
        self.d_v = d_value or d_key
        self.m = n_features
        self.n_buckets = n_buckets
        self.n_tables = n_tables

        rng = np.random.default_rng(seed)

        # Hash functions for bucketing (LSH-style)
        self.hash_projections = [
            rng.standard_normal((n_buckets, d_key))
            for _ in range(n_tables)
        ]

        # Random features for soft attention within buckets
        self.feature_projections = [
            rng.standard_normal((n_features, d_key))
            for _ in range(n_tables)
        ]

        # Normalize
        for i in range(n_tables):
            self.hash_projections[i] /= np.linalg.norm(
                self.hash_projections[i], axis=1, keepdims=True
            )
            self.feature_projections[i] /= np.linalg.norm(
                self.feature_projections[i], axis=1, keepdims=True
            )

        self.caches: Dict = {}
        self.n_tokens = 0

    def _compute_bucket(self, x: np.ndarray, table_idx: int) -> np.ndarray:
        """Compute soft bucket assignment (LSH-style)."""
        x = np.atleast_2d(x)
        proj = x @ self.hash_projections[table_idx].T

        # Soft bucket assignment with temperature
        bucket_weights = np.exp(proj * 4 - proj.max(-1, keepdims=True))
        bucket_weights = bucket_weights / (bucket_weights.sum(-1, keepdims=True) + 1e-8)

        return bucket_weights

    def _compute_features(self, x: np.ndarray, table_idx: int) -> np.ndarray:
        """Compute random features for soft attention."""
        x = np.atleast_2d(x)
        x_norm = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
        x_unit = x / x_norm

        proj = x_unit @ self.feature_projections[table_idx].T * 8
        feat = np.exp(proj - proj.max(-1, keepdims=True))
        feat = feat ** 4
        feat = feat / (feat.sum(-1, keepdims=True) + 1e-8)
        feat = feat * (x_norm ** 2)

        return feat

    def ingest(self, K: np.ndarray, V: np.ndarray):
        """Ingest with hierarchical bucketing."""
        self.caches = {}
        self.n_tokens = len(K)

        for table_idx in range(self.n_tables):
            # Compute bucket assignments for all keys
            bucket_weights = self._compute_bucket(K, table_idx)  # (n, n_buckets)
            features = self._compute_features(K, table_idx)       # (n, n_features)

            # Create hologram per bucket
            # H[b] = Σ_i bucket_weights[i,b] * features[i] ⊗ values[i]
            # Shape: (n_buckets, n_features, d_v)

            H = np.zeros((self.n_buckets, self.m, self.d_v))
            Z = np.zeros((self.n_buckets, self.m))

            for b in range(self.n_buckets):
                weighted_feat = features * bucket_weights[:, b:b+1]
                H[b] = weighted_feat.T @ V
                Z[b] = weighted_feat.sum(0)

            self.caches[table_idx] = (H, Z, bucket_weights.sum(0))

    def query(self, q: np.ndarray) -> np.ndarray:
        """Query with hierarchical retrieval."""
        q = np.atleast_1d(q)

        outputs = []
        confidences = []

        for table_idx in range(self.n_tables):
            H, Z, bucket_counts = self.caches[table_idx]

            # Get query's bucket weights
            q_bucket = self._compute_bucket(q.reshape(1, -1), table_idx)[0]
            q_feat = self._compute_features(q.reshape(1, -1), table_idx)[0]

            # Aggregate from relevant buckets
            output = np.zeros(self.d_v)
            total_weight = 0

            for b in range(self.n_buckets):
                if q_bucket[b] < 0.01:  # Skip near-zero buckets
                    continue

                denom = q_feat @ Z[b] + 1e-8
                bucket_output = (q_feat @ H[b]) / denom

                weight = q_bucket[b]
                output += weight * bucket_output
                total_weight += weight

            if total_weight > 0:
                output = output / total_weight

            # Confidence based on bucket peakedness
            conf = np.max(q_bucket)

            outputs.append(output)
            confidences.append(conf)

        # Weighted combination
        outputs = np.array(outputs)
        confidences = np.array(confidences)

        conf_weights = np.exp(confidences * 5)
        conf_weights = conf_weights / conf_weights.sum()

        return (conf_weights[:, None] * outputs).sum(0)

    def query_batch(self, Q: np.ndarray) -> np.ndarray:
        return np.array([self.query(q) for q in Q])

    def memory_bytes(self) -> int:
        total = sum(P.nbytes for P in self.hash_projections)
        total += sum(P.nbytes for P in self.feature_projections)
        for H, Z, _ in self.caches.values():
            total += H.nbytes + Z.nbytes
        return total

    def compression_ratio(self) -> float:
        if self.n_tokens == 0:
            return float('inf')
        standard = self.n_tokens * (self.d_k + self.d_v) * 8
        return standard / self.memory_bytes()


class HolographicAttentionV4:
    """
    V4: Polynomial feature expansion with multi-scale retrieval.

    Uses polynomial feature expansion for better key discrimination
    and multi-scale hashing for robust retrieval.
    """

    def __init__(
        self,
        d_key: int,
        d_value: Optional[int] = None,
        n_features: int = 512,
        n_scales: int = 4,
        max_poly_degree: int = 2,
        seed: Optional[int] = None
    ):
        self.d_k = d_key
        self.d_v = d_value or d_key
        self.m = n_features
        self.n_scales = n_scales
        self.max_degree = max_poly_degree

        rng = np.random.default_rng(seed)

        # Multi-scale random projections
        self.projections = []
        for scale in range(n_scales):
            # Different projection density at each scale
            P = rng.standard_normal((n_features, d_key))
            P = P / np.linalg.norm(P, axis=1, keepdims=True)
            self.projections.append(P)

        # Scale-specific temperatures
        self.temps = [2 ** (scale + 1) for scale in range(n_scales)]

        self.caches: Dict = {}
        self.n_tokens = 0

    def _poly_features(self, x: np.ndarray, degree: int = 2) -> np.ndarray:
        """Compute polynomial features up to given degree."""
        x = np.atleast_2d(x)
        features = [x]

        if degree >= 2:
            # Quadratic features (element-wise squares)
            features.append(x ** 2)

        return np.concatenate(features, axis=-1)

    def _phi(self, x: np.ndarray, scale: int) -> np.ndarray:
        """Compute features at a given scale."""
        x = np.atleast_2d(x)

        # Normalize
        x_norm = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
        x_unit = x / x_norm

        P = self.projections[scale]
        temp = self.temps[scale]

        proj = x_unit @ P.T * temp

        # Softmax features
        feat = np.exp(proj - proj.max(-1, keepdims=True))
        feat = feat ** min(temp, 32)
        feat = feat / (feat.sum(-1, keepdims=True) + 1e-8)

        # Norm weighting
        feat = feat * (x_norm ** 2)

        return feat

    def ingest(self, K: np.ndarray, V: np.ndarray):
        """Ingest at multiple scales."""
        self.caches = {}
        self.n_tokens = len(K)

        # Also store polynomial expanded values
        K_poly = self._poly_features(K, self.max_degree)

        for scale in range(self.n_scales):
            phi_K = self._phi(K, scale)

            H = phi_K.T @ V
            Z = phi_K.sum(0)

            # Also create hologram with polynomial keys
            phi_K_poly = self._phi(K_poly[:, :self.d_k], scale)  # Use original dims
            H_poly = phi_K_poly.T @ V
            Z_poly = phi_K_poly.sum(0)

            self.caches[scale] = {
                'linear': (H, Z),
                'poly': (H_poly, Z_poly)
            }

    def query(self, q: np.ndarray) -> np.ndarray:
        """Multi-scale query with automatic scale selection."""
        q = np.atleast_1d(q)

        best_output = None
        best_conf = -np.inf

        for scale in range(self.n_scales):
            for feat_type in ['linear', 'poly']:
                H, Z = self.caches[scale][feat_type]

                if feat_type == 'poly':
                    q_input = self._poly_features(q.reshape(1, -1), self.max_degree)[0, :self.d_k]
                else:
                    q_input = q

                q_feat = self._phi(q_input.reshape(1, -1), scale)[0]

                denom = q_feat @ Z + 1e-8
                output = (q_feat @ H) / denom

                # Confidence
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
        total = sum(P.nbytes for P in self.projections)
        for scale_cache in self.caches.values():
            for H, Z in scale_cache.values():
                total += H.nbytes + Z.nbytes
        return total

    def compression_ratio(self) -> float:
        if self.n_tokens == 0:
            return float('inf')
        standard = self.n_tokens * (self.d_k + self.d_v) * 8
        return standard / self.memory_bytes()
