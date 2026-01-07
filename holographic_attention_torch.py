"""
Holographic Attention - PyTorch Implementation

GPU-accelerated O(1) holographic attention for LLM integration.

This is a drop-in replacement for standard attention in PyTorch models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class HolographicAttention(nn.Module):
    """
    PyTorch implementation of O(1) holographic attention.

    Can be used as a drop-in replacement for standard attention.
    """

    def __init__(
        self,
        d_key: int,
        d_value: Optional[int] = None,
        n_features: int = 2048,
        n_sharpness_levels: int = 4,
        max_sharpness: int = 32,
        dropout: float = 0.0,
    ):
        """
        Initialize holographic attention.

        Args:
            d_key: Key/query dimension
            d_value: Value dimension (defaults to d_key)
            n_features: Number of random features
            n_sharpness_levels: Number of sharpness levels to try
            max_sharpness: Maximum sharpening exponent
            dropout: Dropout probability
        """
        super().__init__()

        self.d_k = d_key
        self.d_v = d_value or d_key
        self.m = n_features
        self.n_sharpness = n_sharpness_levels
        self.max_sharpness = max_sharpness

        # Random projection matrix (fixed, not learned)
        self.register_buffer(
            'W',
            torch.randn(n_features, d_key) / math.sqrt(d_key)
        )
        # Normalize rows
        with torch.no_grad():
            self.W.div_(self.W.norm(dim=1, keepdim=True) + 1e-8)

        # Sharpness levels
        sharpness_values = torch.linspace(4, max_sharpness, n_sharpness_levels)
        self.register_buffer('sharpness_levels', sharpness_values)

        self.dropout = nn.Dropout(dropout)

        # Cache for ingested data
        self.H: Optional[torch.Tensor] = None  # (n_sharpness, m, d_v)
        self.Z: Optional[torch.Tensor] = None  # (n_sharpness, m)

    def _phi(self, x: torch.Tensor, sharpness: float) -> torch.Tensor:
        """
        Compute random features.

        Args:
            x: Input of shape (..., d_k)
            sharpness: Sharpening exponent

        Returns:
            Features of shape (..., m)
        """
        # Normalize input
        x_norm = x.norm(dim=-1, keepdim=True) + 1e-8
        x_unit = x / x_norm

        # Project
        proj = F.linear(x_unit, self.W) * math.sqrt(self.d_k)

        # Softmax with sharpening
        feat = F.softmax(proj, dim=-1)
        feat = feat.pow(min(sharpness, 64))
        feat = feat / (feat.sum(dim=-1, keepdim=True) + 1e-8)

        # Weight by squared norm
        feat = feat * (x_norm ** 2)

        return feat

    def ingest(self, K: torch.Tensor, V: torch.Tensor) -> None:
        """
        Ingest key-value pairs into the hologram.

        Args:
            K: Keys of shape (batch, seq_len, d_k) or (seq_len, d_k)
            V: Values of shape (batch, seq_len, d_v) or (seq_len, d_v)
        """
        # Handle batched vs unbatched
        if K.dim() == 2:
            K = K.unsqueeze(0)
            V = V.unsqueeze(0)

        batch_size, seq_len, _ = K.shape

        # Compute holograms for each sharpness level
        H_list = []
        Z_list = []

        for sharpness in self.sharpness_levels:
            phi_K = self._phi(K, sharpness.item())  # (batch, seq_len, m)

            # H = φ(K)ᵀ V
            H = torch.bmm(phi_K.transpose(-2, -1), V)  # (batch, m, d_v)
            # Z = sum(φ(K))
            Z = phi_K.sum(dim=1)  # (batch, m)

            H_list.append(H)
            Z_list.append(Z)

        self.H = torch.stack(H_list, dim=1)  # (batch, n_sharpness, m, d_v)
        self.Z = torch.stack(Z_list, dim=1)  # (batch, n_sharpness, m)

    def query(self, q: torch.Tensor) -> torch.Tensor:
        """
        Query the hologram.

        Args:
            q: Query of shape (batch, d_k) or (d_k,)

        Returns:
            Output of shape (batch, d_v) or (d_v,)
        """
        squeeze_output = q.dim() == 1
        if squeeze_output:
            q = q.unsqueeze(0)

        batch_size = q.shape[0]

        # Try each sharpness level and pick best
        outputs = []
        confidences = []

        for i, sharpness in enumerate(self.sharpness_levels):
            q_feat = self._phi(q, sharpness.item())  # (batch, m)

            H = self.H[:, i]  # (batch, m, d_v)
            Z = self.Z[:, i]  # (batch, m)

            # Compute output: φ(q)ᵀH / φ(q)ᵀZ
            denom = (q_feat * Z).sum(dim=-1, keepdim=True) + 1e-8  # (batch, 1)
            output = torch.bmm(q_feat.unsqueeze(1), H).squeeze(1) / denom  # (batch, d_v)

            # Confidence: attention peakedness
            attn = q_feat * Z
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            conf = attn.max(dim=-1).values  # (batch,)

            outputs.append(output)
            confidences.append(conf)

        outputs = torch.stack(outputs, dim=1)  # (batch, n_sharpness, d_v)
        confidences = torch.stack(confidences, dim=1)  # (batch, n_sharpness)

        # Select best output per batch element
        best_idx = confidences.argmax(dim=1)  # (batch,)
        best_output = outputs[torch.arange(batch_size), best_idx]  # (batch, d_v)

        if squeeze_output:
            best_output = best_output.squeeze(0)

        return self.dropout(best_output)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Standard attention interface.

        Args:
            query: (batch, seq_len, d_k)
            key: (batch, seq_len, d_k)
            value: (batch, seq_len, d_v)
            need_weights: Ignored (weights not computed in holographic attention)

        Returns:
            Output of shape (batch, seq_len, d_v)
            None for attention weights
        """
        batch_size, seq_len, _ = query.shape

        # Ingest K, V
        self.ingest(key, value)

        # Query each position
        outputs = []
        for t in range(seq_len):
            out = self.query(query[:, t])
            outputs.append(out)

        output = torch.stack(outputs, dim=1)  # (batch, seq_len, d_v)

        return output, None


class MultiHeadHolographicAttention(nn.Module):
    """
    Multi-head holographic attention layer.

    Drop-in replacement for nn.MultiheadAttention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        n_features: int = 512,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
    ):
        """
        Initialize multi-head holographic attention.

        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of attention heads
            n_features: Random features per head
            dropout: Dropout probability
            bias: Whether to use bias in projections
            batch_first: If True, input is (batch, seq, feature)
        """
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Holographic attention per head
        self.heads = nn.ModuleList([
            HolographicAttention(
                d_key=self.head_dim,
                d_value=self.head_dim,
                n_features=n_features,
                dropout=dropout,
            )
            for _ in range(num_heads)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            key_padding_mask: Ignored in current implementation
            need_weights: Ignored
            attn_mask: Ignored in current implementation

        Returns:
            Output tensor, None for attention weights
        """
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        batch_size, seq_len, _ = query.shape

        # Project Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Reshape to heads: (batch, seq, num_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Process each head
        head_outputs = []
        for h in range(self.num_heads):
            Q_h = Q[:, :, h]  # (batch, seq, head_dim)
            K_h = K[:, :, h]
            V_h = V[:, :, h]

            out_h, _ = self.heads[h](Q_h, K_h, V_h)
            head_outputs.append(out_h)

        # Concatenate heads
        output = torch.cat(head_outputs, dim=-1)  # (batch, seq, embed_dim)

        # Output projection
        output = self.out_proj(output)
        output = self.dropout(output)

        if not self.batch_first:
            output = output.transpose(0, 1)

        return output, None


def test_pytorch():
    """Test PyTorch implementation."""
    print("=" * 60)
    print("PYTORCH HOLOGRAPHIC ATTENTION TEST")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Test single-head attention
    print("\n1. Single-Head Attention")
    d, n = 64, 1000

    K = torch.randn(n, d, device=device) * 0.3
    V = torch.zeros(n, d, device=device)
    K[n//2] = torch.ones(d, device=device) * 2.0
    V[n//2] = torch.ones(d, device=device) * 42.0

    holo = HolographicAttention(d, n_features=2048).to(device)
    holo.ingest(K, V)

    q = torch.ones(d, device=device) * 2.0
    out = holo.query(q)
    print(f"   Needle retrieval: {out[0].item():.2f}/42.0")

    # Test multi-head attention
    print("\n2. Multi-Head Attention")
    batch_size = 4
    seq_len = 256
    embed_dim = 256
    num_heads = 8

    x = torch.randn(batch_size, seq_len, embed_dim, device=device)

    mha = MultiHeadHolographicAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        n_features=256,
    ).to(device)

    import time
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    t0 = time.perf_counter()
    output, _ = mha(x, x, x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    forward_time = (time.perf_counter() - t0) * 1000

    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Forward time: {forward_time:.1f}ms")

    # Test gradient flow
    print("\n3. Gradient Flow")
    loss = output.sum()
    loss.backward()
    print(f"   Gradients computed: OK")

    # Memory usage
    if torch.cuda.is_available():
        print(f"\n4. GPU Memory")
        print(f"   Allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.1f}MB")
        print(f"   Cached: {torch.cuda.memory_reserved() / 1024 / 1024:.1f}MB")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_pytorch()
