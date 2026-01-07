# Holographic Attention Implementation

O(1) query complexity attention mechanism for Large Language Models.

## Key Results

| Metric | Value |
|--------|-------|
| Needle-in-haystack accuracy | 99.2% |
| Query time (constant) | ~600μs |
| Scaling verified | 1k - 100k tokens |
| Memory compression | 3.2x (1024 features) |

## How It Works

Holographic attention compresses n key-value pairs into a fixed-size "hologram":

```
H = Σ φ(k_i) ⊗ v_i    (hologram)
Z = Σ φ(k_i)          (normalizer)
```

Where φ(k) is a random feature map that approximates the softmax kernel.

Query is then O(1):
```
output = φ(q)ᵀ H / φ(q)ᵀ Z
```

## Files

- `holographic_attention.py` - Original V1 implementation
- `holographic_attention_prod.py` - Production-ready implementation
- `holographic_attention_torch.py` - PyTorch GPU implementation
- `benchmark.py` - Comprehensive benchmark suite
- `quick_benchmark.py` - Quick comparison script

## Usage

### NumPy (CPU)

```python
from holographic_attention_prod import HolographicAttentionProd

# Initialize
holo = HolographicAttentionProd(
    d_key=64,
    n_features=4096,  # More = better accuracy
    mode='content'
)

# Ingest key-value pairs (O(n) preprocessing)
holo.ingest(keys, values)

# Query (O(1)!)
output = holo.query(query)
```

### PyTorch (GPU)

```python
from holographic_attention_torch import MultiHeadHolographicAttention

# Drop-in replacement for nn.MultiheadAttention
mha = MultiHeadHolographicAttention(
    embed_dim=512,
    num_heads=8,
    n_features=256,
)

output, _ = mha(query, key, value)
```

### Causal Mode (Autoregressive)

```python
holo = HolographicAttentionProd(d_key=64, n_features=4096)
holo.ingest(keys, values, causal=True)

# Query at position t only sees keys 0..t
output = holo.query(query, position=t)
```

### Streaming Updates

```python
holo = HolographicAttentionProd(d_key=64, n_features=4096)
holo.ingest(initial_keys, initial_values)

# Add new key-value pairs in O(1)
holo.update(new_key, new_value)
```

## Configuration Guide

| Use Case | n_features | Memory | Accuracy |
|----------|------------|--------|----------|
| Memory constrained | 1024 | 3KB/token | ~84% |
| Balanced | 2048 | 5KB/token | ~91% |
| High accuracy | 4096 | 10KB/token | ~99% |

## Limitations

1. **Similar keys are averaged**: When keys are nearly identical, their values are averaged together. This is correct for soft attention but limits exact retrieval.

2. **No learned projections**: The random projections are fixed. Learned projections could improve accuracy but would require training.

3. **Memory overhead at low features**: With fewer features, compression ratio decreases.

## Benchmarks

Run benchmarks:
```bash
python benchmark.py        # Full suite
python quick_benchmark.py  # Quick comparison
```

## Theory

The holographic approach is based on Random Fourier Features (RFF) for kernel approximation:

1. Standard attention: `softmax(QK^T) V`
2. Kernel view: `softmax(qk^T) ≈ φ(q)^T φ(k)` for random feature map φ
3. Holographic: Pre-compute `H = Σ φ(k) v^T`, then query `φ(q)^T H`

The key insight is that the kernel approximation allows us to decompose the attention computation, enabling O(1) queries after O(n) preprocessing.

## References

- Performer: Rethinking Attention with Performers (Choromanski et al., 2020)
- Random Features for Large-Scale Kernel Machines (Rahimi & Recht, 2007)
- Efficient Attention (Shen et al., 2021)
