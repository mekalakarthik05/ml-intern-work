# Positional Encoding

**Giving Transformers a sense of order** — You will learn why Transformers cannot understand sequence order without help, how sinusoidal and learned positional encodings fix that, and the trade-offs between modern variants like RoPE and ALiBi.

---

## Learning Objectives

After completing this topic, the student should be able to:

**Explain & Intuit**
- Why Transformers need positional encoding (permutation invariance of self-attention)
- How sinusoidal encodings encode relative position via linear transformations
- The trade-off between absolute vs. relative positional representations

**Derive (Conceptually)**
- The sinusoidal positional encoding formula: why alternating sin/cos frequencies
- How dot products of shifted sinusoids depend only on relative offset
- The gradient flow through positional encodings during backpropagation

**Implement**
- Sinusoidal positional encoding from scratch (NumPy/PyTorch)
- Learned positional embeddings (nn.Embedding)
- Rotary Position Embedding (RoPE) forward pass
- Relative position biases (T5-style, ALiBi)

**Analyze & Debug**
- Diagnose length generalization failures caused by absolute positional encodings
- Compare positional encoding strategies on sequence length extrapolation
- Measure positional information retention via probing tasks

---

## Subtopic List

### Core Concepts & Intuition
1. **Permutation invariance of attention** — Why QK^T without position treats "I ate pizza" and "pizza ate I" identically.
2. **Absolute vs. relative position** — Why the model needs to know both where tokens are and how far apart they are.
3. **Sinusoidal intuition** — Each dimension encodes a frequency; the model learns to use these as "phase" signals.

### Mathematical Formulation (Final Equations)
4. **Sinusoidal encoding** — PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
5. **Learned absolute embedding** — PE = nn.Embedding(max_len, d_model), trained end-to-end.
6. **Rotary Position Embedding (RoPE)** — Rotates query/key pairs by a block-diagonal rotation matrix: f_q(x_m, m) = R_m W_q x_m, with R_m defined per 2D subspace at angle mθ_i.
7. **ALiBi (Attention with Linear Biases)** — Adds a static bias to attention scores: score(i, j) = q_i k_j^T + m · (j − i), where m is a head-specific slope.
8. **T5 relative bias** — Learned scalar bias b_{j−i} added to attention logits, bucketed for efficient parameterization.

### Algorithm Variants & Extensions
9. **Sinusoidal (Vaswani et al., 2017)** — Fixed, no learned params, generalizes to unseen lengths (theoretically).
10. **Learned absolute (BERT, GPT-2)** — Flexible per-position, but capped at max training length.
11. **Rotary (RoPE)** — Applied to Q and K; decays dot product with distance; used in LLaMA, Mistral, GPT-NeoX.
12. **ALiBi (Press et al., 2022)** — No positional embeddings added to inputs; only bias attention scores. Strong length extrapolation.
13. **T5 bias / Transformer-XL** — Learned relative bias with log-bucketing; extended context windows.
14. **xPos / FIRE** — Variants of RoPE for long-context (e.g., YaRN, NTK-aware scaling).

### Implementation Details
15. **From scratch (sinusoidal)** — Precompute PE matrix; add to token embeddings before the first encoder layer.
16. **PyTorch implementation patterns** — Register buffer vs. parameter; batch-wise broadcasting; half-precision considerations.
17. **RoPE implementation** — Complex multiply trick vs. explicit rotation; precomputing cos/sin tables.
18. **Integration with attention** — Where in the forward pass each PE variant injects positional information.
19. **Library usage** — HuggingFace Transformers' handling of PE variants (config flags, model-specific implementations).

### Common Failure Modes, Edge Cases & Misconceptions
20. **Length extrapolation failure** — Learned absolute embeddings degrade sharply beyond max training length; sinusoidal also degrades in practice (distribution shift).
21. **Misconception: sinusoidal encodes absolute position** — It primarily encodes relative offset via linear combinations; absolute info is weak by design.
22. **Edge case: batch padding** — Positional encoding should not be applied to pad tokens; masking must exclude them.
23. **Edge case: tied embeddings** — When tying input & output embeddings, positional encoding placement matters (pre- vs. post-embedding).
24. **Failure mode: untrained heads with RoPE** — The rotation is per-dimension; some attention heads may not learn to utilize it, requiring head-specific frequency tuning (learned θ_i).
25. **Numerical stability** — Large pos/denom ratio in sinusoids causes tiny gradients; precomputation with float32 avoids this.

---

## What Is Positional Encoding?

Self-attention is permutation-invariant: the attention score between "I" and "pizza" is the same whether the input is "I ate pizza" or "pizza ate I". That's a problem because language *is* order. Positional encoding solves this by injecting location signals into the model.

Think of a coat check. Each ticket has a number, but the attendant also remembers *where* each coat hangs (first shelf vs. last). The number is the token embedding; the shelf position is the positional encoding. Without the shelf location, grabbing "coat #7" could mean any coat #7 anywhere — or none at all.

Formally, positional encoding is a vector added to (or mixed with) the token embedding at the input of a Transformer. It tells each attention head, "these two tokens are 3 positions apart" or "this token is at position 5 in the sequence."

---

## Mathematical Formulation

### Sinusoidal Positional Encoding (Vaswani et al., 2017)

For position `pos` and dimension index `i`:

```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

**What it tells us:** Each dimension pair encodes a frequency. Low `i` → high frequency (fine-grained local position). High `i` → low frequency (coarse global position). The model can recover relative offset between any two positions because a dot product of shifted sinusoids depends only on `pos_1 - pos_2`, not on absolute values.

### Learned Absolute Embedding

```
PE = Embedding(max_len, d_model)
```

**What it tells us:** Each position gets a free vector learned via backprop. Flexible but capped — the model has never seen position 513 if trained on length 512.

### Rotary Position Embedding (RoPE)

```
f_q(x_m, m) = R_m W_q x_m    where R_m rotates (x, y) pairs by angle m·θ_i
```

**What it tells us:** Position is multiplied into Q and K via rotation. Dot products naturally decay with distance, which biases attention toward nearby tokens — a useful inductive bias.

### ALiBi

```
score(i, j) = q_i · k_j + m · (j - i)
```

**What it tells us:** No positional signal is added to embeddings at all. A fixed linear bias is added directly to attention scores. Simpler, parameter-free, and extrapolates well.

---

## How It Works — Step by Step

**Step 1 — Build the position matrix.** Create a matrix with shape `(max_len, d_model)` where each row corresponds to a position (0, 1, 2, ...). For sinusoidal, fill alternating sin/cos values at geometrically increasing frequencies. Think of it like assigning each seat in a theater a unique acoustic fingerprint.

**Step 2 — Add to token embeddings.** Take your `(batch, seq_len, d_model)` token embeddings and add the first `seq_len` rows of the position matrix element-wise. The model now sees: `x_pos = x_token + PE[:seq_len]`.

**Step 3 — Forward through attention.** The Q, K, V projections now see mixed token + position signals. When computing attention scores `Q K^T`, the dot product naturally encodes how far apart tokens are.

**Step 4 — (RoPE variant only)** Apply rotation to Q and K *after* projection but *before* the dot product. This keeps token embeddings position-free and only encodes position at the attention step.

**Step 5 — (ALiBi variant only)** Skip adding any position signal to embeddings. Instead, add a static linear bias to `Q K^T` proportional to `j - i`. Closer tokens get a boost; farther tokens get penalized.

---

## Key Assumptions

| Assumption | What happens if violated |
|---|---|
| Positions are integers from 0 to max_len-1 | Fractional positions (e.g., in interpolation) need special handling; naive rounding loses precision |
| Frequency scaling follows a geometric progression | Changing the base or scaling law changes resolution of position discrimination |
| Max sequence length during inference ≤ training length | Learned embeddings fail catastrophically; sinusoidal and RoPE degrade gracefully but still suffer |
| Position is independent of token content | Sometimes violated (e.g., sentence boundaries carry positional meaning); learned embeddings can compensate implicitly |

---

## When to Use / When Not to Use

| Use Positional Encoding When | Consider Alternatives When |
|---|---|
| You need to model sequence order in any Transformer | Your model processes sets or graphs (no inherent order) |
| Your task is NLP, time-series, or any sequential data | Your sequence length is extremely short (≤4 tokens); CNNs may suffice |
| You need length extrapolation (ALiBi / RoPE) | You have unlimited compute and can train with full-length data (learned absolute is fine) |
| You are building a modern LLM (use RoPE) | Your model already has recurrence (RNN / state-space model) |

---

## Implementation Overview

```python
import numpy as np

def sinusoidal_pe(max_len, d_model):
    pe = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            freq = pos / (10000 ** (i / d_model))
            pe[pos, i] = np.sin(freq)
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(freq)
    return pe  # shape: (max_len, d_model)

# Usage: token_embeds + pe[:seq_len]
```

| Aspect | From Scratch (NumPy) | Library (PyTorch / HuggingFace) |
|---|---|---|
| Storage | Precomputed numpy array | `nn.Embedding` or registered buffer |
| GPU support | Manual `.to(device)` | Automatic via tensor device management |
| RoPE | Manual complex-multiply | `transformers` config flag: `rope_scaling` |
| Batch handling | Manual broadcasting | Automatic (shape broadcast in forward) |
| Extensibility | Full control | Swap config params; no code changes |

---

## Prerequisite Knowledge

| Area | Specific Topics | Why It's Needed |
|---|---|---|
| Deep Learning Fundamentals | Feed-forward networks, backpropagation, gradient descent | Understand how any PE variant learns or doesn't learn |
| Sequence Models | RNNs, LSTMs, sequential vs. parallel processing | Contrast with Transformer's lack of inherent order |
| Attention Mechanism | Scaled dot-product attention, Q/K/V projections, multi-head attention | PE is integrated directly into attention computation |
| Linear Algebra | Matrix multiplication, dot product, rotation matrices (for RoPE) | Every PE variant manipulates or biases dot products |
| Trigonometry | Sine, cosine, angular frequency, phase shift | Sinusoidal PE is built on these |
| Embeddings | Token embedding lookup, embedding matrices | PE is added or concatenated with token embeddings |
| Basic Probability & Statistics | Distribution shift, interpolation vs. extrapolation | Understanding length generalization failures |

---

## Connections to Other ML Topics

| Topic | Connection |
|---|---|
| **RNNs / LSTMs** | RNNs encode position via timestep recurrence; Transformers replace recurrence with PEs, trading sequential for parallel computation |
| **Convolutional Networks** | CNNs encode relative position implicitly through kernel stride/padding; PEs make it explicit and learnable |
| **Neural ODEs / Coordinate Embeddings** | Sinusoidal PE is structurally identical to NeRF's Fourier features — mapping low-dim coordinates to high-dim sinusoids |
| **Graph Neural Networks** | Positional encodings in graphs (Laplacian PE) solve the same permutation invariance problem as in sequences |
| **Speech / Audio (WaveNet, FFT)** | Using frequency-based representations to encode temporal structure; PEs share the same multi-frequency design philosophy |
| **Time Series Decomposition** | Trend + seasonality + residual maps to position + frequency + noise; certain PE variants can be interpreted as learned seasonality bases |
| **Mixture of Experts (MoE)** | MoE routing often ignores token position; adding PE features to the router can improve expert load balancing |
| **Retrieval-Augmented Generation** | Long-context PE (YaRN, NTK-aware) directly enables RAG over documents exceeding training length |
| **Memory-Augmented Networks** | Positional signals help memory controllers associate read/write operations across distant timesteps |
| **Vision Transformers** | 2D/3D positional encodings for images/video are direct extensions of 1D sequence PEs |

---

## Top 5 Interview Questions

**Q1. Why can't self-attention model order?** — Show permutation invariance: `softmax(XW_Q · (XW_K)^T) XW_V` yields same output if rows of X are permuted. Sinusoidal PE breaks this via position-dependent bias. Alternating sin/cos allows recovering relative offset from dot products.

**Q2. Model trained on length 512 fails at 1024 — diagnose and fix.** — Issue: learned embedding OOD. Fixes: (a) interpolate embeddings (b) switch to RoPE + extend frequencies (YaRN) (c) use ALiBi (zero-shot cap). Trade-off: compute cost vs. quality retention.

**Q3. Explain RoPE.** — Rotate Q/K per 2D subspace using angle `m·θ_i`. Dot product `Q_m · K_n` depends on `m - n`. Distance decay is a feature: it biases attention locality, reducing attention to very distant tokens.

**Q4. Choose PE for 128k context.** — RoPE with NTK-aware scaling or YaRN. ALiBi also works but may underperform on long-range retrieval tasks. Sinusoidal and learned absolute are off the table.

**Q5. Implement sinusoidal PE in pseudocode; extend to T5 relative bias.** — Pseudocode: double loop over `pos` and `i`. T5 bias: replace pre-embedding addition with a learned scalar `b_{clip(j-i, -k, k)}` added per head in attention logits. Key difference: PE is input-side; bias is attention-side.

---

## Quick Reference Table

| Item | Detail |
|---|---|
| Algorithm Type | Input embedding / Attention bias |
| Time Complexity | O(seq_len · d_model) per forward pass (precomputed) |
| Space Complexity | O(max_len · d_model) for storage |
| Key Hyperparameters | `max_len`, `d_model`, base frequency (10000), RoPE θ_i, ALiBi slopes |
| Evaluation Metrics | Perplexity, length extrapolation accuracy, probing tasks |

---

## References & Further Reading

1. [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — Original sinusoidal PE proposal
2. [RoFormer: Rotary Position Embedding (Su et al., 2021)](https://arxiv.org/abs/2104.09864) — RoPE paper
3. [Train Short, Test Long: ALiBi (Press et al., 2022)](https://arxiv.org/abs/2108.12409) — Linear bias alternative
4. [The Annotated Transformer (Harvard NLP)](http://nlp.seas.harvard.edu/2018/04/03/attention.html) — Walkthrough with code
5. [Efficient Transformers: Positional Encodings — A Survey](https://paperswithcode.com/task/positional-encoding) — Paper list & benchmarks

---

*Diagram: `pe_diagram.png` — Transformer encoder block with positional encoding step highlighted.*
