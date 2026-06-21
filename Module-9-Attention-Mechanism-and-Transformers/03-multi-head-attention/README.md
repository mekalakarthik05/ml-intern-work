# Multi-Head Attention

> *Letting the model look at the same sentence through many lenses at once.*

**What you will learn:** By the end of this material, you will understand how transformers learn to relate every token in a sequence to every other token simultaneously — and why doing this with multiple independent "heads" is far more powerful than doing it once. You will be able to derive the parameter count of an MHA layer, implement it from scratch, and reason confidently about its failure modes in production settings.

---

## 1. What Is Multi-Head Attention?

At its core, attention is a mechanism for a model to decide *which parts of an input to focus on* when producing each element of an output. Think of it like reading a long contract: when you reach the clause "the party referred to in Section 3(b)," your eyes don't re-read the whole document — they jump directly to Section 3(b). Attention teaches a model to do exactly this, but in a differentiable, learnable way.

Scaled dot-product attention — the building block of multi-head attention — works by assigning each input token a **query** (what am I looking for?), a set of **keys** (what does each other token advertise?), and a set of **values** (what information should each other token contribute?). The model computes a similarity score between a token's query and every other token's key, turns those scores into weights via softmax, and takes a weighted sum of the values. The result is a new representation of that token — one that has "looked at" all others and gathered relevant context.

The *multi-head* part is the crucial upgrade. Instead of performing this lookup once in the full embedding space, MHA projects the input into `h` smaller subspaces and runs attention independently in each. Each head can then specialize: one head might learn to track subject-verb agreement, another might learn coreference (connecting pronouns to their referents), and another might capture long-range dependencies. The outputs of all heads are concatenated and projected back to the original dimension. **The real-world analogy:** it is like having a panel of h expert reviewers each reading the same document with a different mandate, then pooling their notes into a single consolidated report.

---

## 2. Mathematical Formulation

**Scaled dot-product attention** (one head):

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

*What it tells us:* The term $QK^\top$ computes pairwise similarity scores between every query-key pair — producing an $n \times n$ matrix for a sequence of length $n$. Dividing by $\sqrt{d_k}$ (the square root of the key dimension) prevents the dot products from growing so large that the softmax saturates into near-zero gradients. The softmax converts scores into a probability distribution over positions, and multiplying by $V$ produces a weighted average of value vectors.

**Multi-head attention:**

$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\,W^O$$

$$\text{head}_i = \text{Attention}(QW^Q_i,\ KW^K_i,\ VW^V_i)$$

*What it tells us:* Each head $i$ has its own learned projection matrices $W^Q_i \in \mathbb{R}^{d_\text{model} \times d_k}$, $W^K_i$, $W^V_i$, and $W^O \in \mathbb{R}^{hd_v \times d_\text{model}}$. Because each head projects into a smaller $d_k = d_\text{model}/h$ dimensional space, the total computation stays constant regardless of the number of heads. The output projection $W^O$ fuses all heads back into a single $d_\text{model}$-dimensional representation.

**Total parameter count** (one MHA layer, no bias):

$$\text{Params} = 4 \cdot d_\text{model}^2$$

*What it tells us:* Three projection matrices for Q, K, V (each $d_\text{model} \times d_\text{model}$) plus the output projection $W^O$ — four square matrices in total. This number is independent of the number of heads, which is a clean design property.

---

## 3. How It Works — Step by Step

**Step 1 — Project inputs into Q, K, V.**
Each token's embedding is linearly projected into three separate vectors: a query, a key, and a value — independently for each head. Think of it as each token "filling out three different forms": one advertising what it's searching for, one advertising what it offers, and one containing its actual payload.

**Step 2 — Compute attention scores.**
For each head, compute the dot product of every query with every key: $QK^\top$. This gives an $n \times n$ score matrix. Tokens that are highly relevant to each other get high scores. For example, in "The trophy didn't fit in the suitcase because **it** was too big," the query for "it" would score highly against "trophy."

**Step 3 — Scale and optionally mask.**
Divide all scores by $\sqrt{d_k}$. In a decoder (autoregressive generation), apply a causal mask — set scores for future positions to $-\infty$ so the softmax assigns them zero weight. This prevents the model from "cheating" by seeing tokens it hasn't generated yet.

**Step 4 — Softmax to get attention weights.**
Apply softmax row-wise, converting raw scores into a probability distribution over positions. Each row now sums to 1, representing how much attention token $i$ pays to every other token $j$.

**Step 5 — Weighted sum of values.**
Multiply the attention weights by $V$. The output for each token is a blend of all value vectors, weighted by relevance. A token that strongly attends to another "borrows" its context.

**Step 6 — Concatenate heads and project.**
Stack the outputs from all $h$ heads (each of dimension $d_v$) into a single vector of dimension $h \cdot d_v = d_\text{model}$, then multiply by $W^O$ to mix information across heads. This final projection is what allows heads to interact and produce a unified representation.

---

## 4. Key Assumptions

| Assumption | What happens if violated |
|---|---|
| Input tokens are embedded in a continuous vector space | Discrete inputs (e.g., raw integers) feed meaningless dot products — scores become noise, attention is random |
| Sequence length fits in memory | Attention requires $O(n^2)$ memory; sequences >4k–8k tokens will OOM on consumer hardware without sparse/flash attention |
| Positional information is injected externally | MHA is permutation-equivariant by design; without positional encodings, "cat sat on mat" and "mat on sat cat" are identical to the model |
| $d_k$ is sufficiently large | Very small $d_k$ limits expressivity; very large $d_k$ without scaling causes softmax saturation and vanishing gradients |
| Padding tokens are masked | Unmasked padding contributes spurious value signals and pollutes the attended representation |
| Heads receive gradient signal | If one head collapses (attention sink), it stops learning and wastes capacity — diversity requires a well-tuned learning rate and initialization |

---

## 5. When to Use / When Not to Use

| **Use MHA when…** | **Avoid or modify MHA when…** |
|---|---|
| The task requires modeling long-range dependencies (e.g., coreference, document QA) | Sequence length > 8k tokens with standard hardware — use Flash Attention or sparse variants |
| You need the model to learn multiple types of relationships simultaneously | Inference latency is critical and the KV cache becomes a memory bottleneck — consider GQA or MQA |
| Input modalities can be embedded (text, images, audio, graph nodes) | You only need local context (e.g., short sliding-window tasks) — convolutions may be faster and sufficient |
| You have GPU compute and can afford $O(n^2)$ memory scaling | Deployment is on edge devices with <1 GB RAM — standard MHA is prohibitively large |
| Task benefits from cross-modal reasoning (text + image cross-attention) | Training data is tiny (<10k samples) — attention has many parameters and overfits easily without pretraining |

---

## 6. Implementation Overview

| | **From Scratch (NumPy/PyTorch)** | **Library (PyTorch `nn.MultiheadAttention`)** |
|---|---|---|
| **Q/K/V projection** | Manual `nn.Linear` per head, or batched via reshape | Handled internally; pass `embed_dim` and `num_heads` |
| **Score computation** | `torch.bmm(Q, K.transpose(-2,-1)) / math.sqrt(d_k)` | Handled internally |
| **Masking** | Add `float('-inf')` to masked positions before softmax | Pass `attn_mask` (additive) or `key_padding_mask` (boolean) |
| **Softmax + weighted sum** | `F.softmax(..., dim=-1) @ V` | Handled internally |
| **Head concatenation** | `torch.cat(heads, dim=-1) @ W_O` | Handled internally |
| **When to use** | Learning, research, custom attention variants | Production, fine-tuning, rapid prototyping |
| **Gotcha** | Easy to get tensor shape bugs (batch × heads × seq × d) | Confusing `batch_first` default (False); mask convention differs from intuition |

```python
import torch
import torch.nn as nn

# Library usage: nn.MultiheadAttention
d_model, num_heads, seq_len, batch = 512, 8, 64, 32

mha = nn.MultiheadAttention(
    embed_dim=d_model,
    num_heads=num_heads,
    dropout=0.1,
    batch_first=True   # input shape: (batch, seq, d_model)
)

x = torch.randn(batch, seq_len, d_model)

# Self-attention (Q = K = V = x)
# key_padding_mask: True where tokens are padding
padding_mask = torch.zeros(batch, seq_len, dtype=torch.bool)

# Causal mask for autoregressive decoding
causal_mask = torch.triu(
    torch.ones(seq_len, seq_len), diagonal=1
).bool()   # upper triangle = True → masked out

attn_output, attn_weights = mha(
    query=x, key=x, value=x,
    attn_mask=causal_mask,
    key_padding_mask=padding_mask,
    need_weights=True   # set False in production for speed
)
# attn_output: (batch, seq_len, d_model)
```

---

## 7. Top 5 Interview Questions

**Q1 — Walk me through the full forward pass of MHA, including all matrix shapes.**

- Start from input $X \in \mathbb{R}^{n \times d_\text{model}}$
- Project to Q, K, V via learned matrices; reshape to separate heads: $(B, h, n, d_k)$
- Compute scores: $(B, h, n, n)$ → scale → mask → softmax
- Multiply by V: $(B, h, n, d_v)$ → concat → $(B, n, h \cdot d_v)$ → project via $W^O$
- Name all four weight matrices and their shapes; state parameter count = $4 d_\text{model}^2$

**Q2 — Why divide by $\sqrt{d_k}$? What fails without it?**

- Dot products grow with dimension: variance of $q \cdot k = d_k \cdot \sigma^2$
- Large values push softmax into near-zero gradient regions (saturation)
- Training stalls: gradients from attention vanish, weights stop updating
- Dividing by $\sqrt{d_k}$ keeps variance $\approx 1$ regardless of head dimension

**Q3 — You're serving a 70B LLM and memory is a bottleneck. Explain MHA → GQA trade-off.**

- Standard MHA: $h$ query heads, $h$ key heads, $h$ value heads → KV cache scales with $h$
- GQA: $h$ query heads but $g < h$ KV head groups → cache shrinks by $h/g$
- Quality: minimal degradation when $g \geq 4$; zero degradation at $g = h$ (MHA)
- Throughput: larger batch sizes fit in memory → higher GPU utilization
- Used in Llama 2 (GQA), Mistral (sliding window + GQA) — cite these

**Q4 — Self-attention is permutation-equivariant. Why does this matter, and how is it fixed?**

- If you shuffle input tokens, outputs shuffle identically — no notion of order
- Natural language is order-dependent: "dog bites man" ≠ "man bites dog"
- Fix 1: Absolute sinusoidal or learned positional embeddings added to input
- Fix 2: Relative position bias (inside the attention score computation) — more robust to length generalization
- Fix 3: RoPE — rotates Q and K vectors by angle proportional to position; used in GPT-NeoX, Llama
- ALiBi: adds a linear position penalty to scores; best for length extrapolation

**Q5 — Your model was trained on 8k-token sequences but inference needs 32k. You're OOM. Three approaches.**

- **Flash Attention 2**: reorders attention computation to use SRAM tiling; same math, 3–8× less HBM; enables 32k+ with no quality loss
- **Sliding window / sparse attention** (Longformer): each token attends to a local window + global tokens; O(n) memory; some long-range signal lost
- **RoPE + positional interpolation** (YaRN, LongRoPE): extend rope base frequency at inference; quality degrades gracefully; requires fine-tuning to recover
- Mention trade-offs: Flash is best when you have the GPU; sparse trades recall for memory; interpolation trades quality for zero hardware cost

---

## 8. Quick Reference Table

| Item | Detail |
|---|---|
| **Algorithm type** | Attention mechanism; core of Transformer encoder/decoder blocks |
| **Time complexity** | $O(n^2 \cdot d_\text{model})$ per layer — quadratic in sequence length |
| **Space complexity** | $O(n^2)$ for attention weight matrix (per head, per layer) |
| **Key hyperparameters** | `d_model`, `num_heads`, `dropout`, `d_k = d_model / num_heads` |
| **Parameter count** | $4 \cdot d_\text{model}^2$ per MHA layer (no bias) |
| **Evaluation metrics** | Task-dependent: perplexity (LM), BLEU/ROUGE (seq2seq), F1 (QA); attention entropy for head diversity analysis |
| **Common variants** | MQA, GQA, Flash Attention, Sparse/Sliding-Window, Cross-Attention |
| **Frameworks** | PyTorch `nn.MultiheadAttention`, HuggingFace `transformers`, JAX/Flax, TensorFlow Keras |

---

## 9. References & Further Reading

| Resource | Why read it |
|---|---|
| [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) | The original paper — read Sections 3.2 and 3.3 for the exact equations |
| [The Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/) | Best visual walkthrough of MHA; bridges math to intuition with animations |
| [Flash Attention paper (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) | Essential for production — explains the IO-aware recomputation trick cleanly |
| [GQA: Training Generalized Multi-Query Transformer Models (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245) | The paper behind Llama 2's memory efficiency; clear ablations on quality vs. group count |
| [Annotated Transformer — Harvard NLP](https://nlp.seas.harvard.edu/annotated-transformer/) | Line-by-line walkthrough of the original Transformer with runnable PyTorch code |
