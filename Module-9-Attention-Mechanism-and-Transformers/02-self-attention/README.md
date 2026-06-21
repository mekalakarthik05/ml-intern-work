# Self-Attention

**Tagline:** Letting every word in a sequence decide, for itself, what else in the sequence matters.

**What you will learn:** This README builds the intuition, math, and mechanics of self-attention — the mechanism that powers Transformers, BERT, and GPT. By the end, you'll understand how attention scores are computed, when self-attention is the right tool, and how to reason about it in an interview setting.

---

## 2. What is Self-Attention?

Self-attention is a mechanism that lets a model look at an entire sequence at once and decide, for each element, how much it should "pay attention to" every other element — including itself — when building its representation. Instead of processing a sentence word by word like an RNN, a self-attention layer lets every word directly query every other word in a single step, and weighs their contributions based on relevance.

Think of a classroom discussion. When you (a word) want to understand the topic better, you don't ask only the person next to you — you scan the whole room and decide whose comments are most relevant to your question. You give more "weight" to the insightful student and less to someone talking about something unrelated. Self-attention works the same way: each word issues a "query" (what am I looking for?), every word offers a "key" (what do I represent?), and the match between query and key determines how much of that word's "value" (its actual content) gets blended into the output.

This is powerful because relationships between words can be long-distance and content-dependent. In the sentence "The trophy didn't fit in the suitcase because *it* was too big," resolving "it" requires connecting to "trophy," which could be far away in the sentence. Self-attention computes this connection directly, in parallel, regardless of distance — something RNNs struggle with due to their sequential, step-by-step processing.

---

## 3. Mathematical Formulation

**Scaled Dot-Product Attention:**

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

*Significance:* This single equation says — compute similarity between queries and keys (QKᵀ), scale it so values don't explode, normalize into probabilities (softmax), then use those probabilities to take a weighted average of the values. The output for each token is literally "a blend of everyone else's information, weighted by relevance."

**Linear Projections:**

```
Q = X W_Q,   K = X W_K,   V = X W_V
```

*Significance:* Raw token embeddings (X) aren't naturally suited to being queries, keys, or values — these learned projection matrices transform the same input into three specialized "roles."

**Multi-Head Attention:**

```
MultiHead(X) = Concat(head_1, ..., head_h) W_O
```

*Significance:* Running several smaller attention computations in parallel lets the model attend to different *types* of relationships simultaneously (e.g., one head tracks syntax, another tracks coreference).

**Causal Masking (for decoders):**

```
scores_masked[i,j] = scores[i,j] if j ≤ i, else -∞
```

*Significance:* Prevents a token from "seeing the future" — essential for autoregressive generation.

---

## 4. How It Works — Step by Step

1. **Project inputs into Q, K, V.**
   Analogy: Each word fills out three index cards — "what I'm looking for" (Query), "what I offer" (Key), "what I actually contain" (Value).

2. **Compute similarity scores (QKᵀ).**
   Analogy: Every word's Query card is compared against every other word's Key card — like matching search terms to document tags.

3. **Scale by √d_k.**
   Analogy: If you used a giant magnifying glass, even tiny differences look huge. Scaling keeps the comparison fair and prevents one match from completely dominating.

4. **Apply softmax to get attention weights.**
   Analogy: Convert raw match scores into percentages that sum to 100% — "I'll devote 70% of my attention to word A, 20% to word B, 10% to the rest."

5. **Weighted sum of Values.**
   Analogy: Blend everyone's Value cards according to those percentages — the result is a new representation enriched with relevant context.

6. **Repeat across multiple heads, then concatenate.**
   Analogy: Multiple "committees" each focus on a different aspect of the discussion (grammar, meaning, tone), then their notes are combined into one final report.

---

## 5. Key Assumptions

| Assumption | If Violated |
|---|---|
| Relevance can be captured by a dot-product similarity between learned Q/K vectors | Model may fail to capture relationships that aren't linearly separable in this space; needs deeper/more heads to compensate |
| Sequence length is manageable (attention is O(n²)) | Extremely long sequences cause memory/compute blow-up; requires sparse or linear attention variants |
| Tokens have no inherent order unless told so | Without positional encoding, the model treats input as a "bag of tokens" and loses word order information |
| Padding tokens are properly masked | Real tokens can "attend to" meaningless padding, corrupting representations |
| Sufficient training data exists to learn meaningful Q/K/V projections | Underfitting; attention weights become close to uniform/noisy and uninformative |

---

## 6. When to Use / When Not to Use

| ✅ Use Self-Attention When | ❌ Avoid / Reconsider When |
|---|---|
| You need to model long-range dependencies in sequences (NLP, long documents) | Sequence length is extremely large (100K+) and compute/memory budget is tight (use sparse/linear attention instead) |
| Parallelization during training is a priority | You're working with very small datasets where a simpler model (CNN/RNN) generalizes better |
| Task benefits from global context per token (translation, summarization) | Latency-sensitive inference on edge devices with limited compute |
| You're building or fine-tuning Transformer-based architectures (BERT, GPT, ViT) | The relationships in your data are strictly local/sequential (e.g., short fixed-window signals), where a CNN may be more efficient |
| Interpretability via attention maps is a nice-to-have (not a hard guarantee) | You need rigorously interpretable, causal explanations — attention weights are not proof of causal importance |

---

## 7. Implementation Overview

| Aspect | From Scratch (NumPy) | Library (PyTorch / sklearn-style) |
|---|---|---|
| Q/K/V projections | Manually multiply input by weight matrices you initialize | Handled internally by `nn.MultiheadAttention` or `scaled_dot_product_attention` |
| Masking | You build the mask matrix and add it before softmax | Pass `attn_mask` / `key_padding_mask` arguments |
| Multi-head logic | You manually reshape and loop/batch over heads | Built-in head-splitting and concatenation |
| Numerical stability | You handle softmax overflow manually (subtract max) | Library functions are optimized and numerically stable by default |
| Speed | Educational, but slow — good for understanding internals | Highly optimized (cuDNN/FlashAttention kernels) for production use |
| Best use case | Learning, debugging, interviews | Production training and deployment |

Self-attention itself isn't a `sklearn` algorithm (sklearn has no Transformer module), but to fulfill the standard "library training" example, here's a quick sklearn snippet showing how a downstream classifier might be trained on pooled self-attention-derived embeddings (e.g., from a pretrained Transformer):

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# X_embeddings: sentence embeddings pooled from a self-attention model (e.g., [CLS] token)
# y: labels for a downstream classification task
X_train, X_test, y_train, y_test = train_test_split(
    X_embeddings, y, test_size=0.2, random_state=42
)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

print("Accuracy:", clf.score(X_test, y_test))
```

### Attention Block Diagram (Transformer Sub-layer Flow)

Transformers use **LayerNorm**, not BatchNorm — but here's how the self-attention sub-layer fits into the normalization flow:

```
        Input X
           │
           ▼
   ┌───────────────┐
   │ Self-Attention│
   └───────┬───────┘
           │  (residual add)
           ▼
   ┌───────────────┐
   │   LayerNorm    │
   └───────┬───────┘
           │
           ▼
   ┌───────────────┐
   │ Feed-Forward   │
   └───────┬───────┘
           │  (residual add)
           ▼
   ┌───────────────┐
   │   LayerNorm    │
   └───────┬───────┘
           │
           ▼
        Output
```

---

## 8. Top 5 Interview Questions

1. **Why scale attention scores by √d_k?**
   - Mention: dot products grow with dimension d_k
   - Mention: large values → softmax saturates → vanishing gradients
   - Conclude: scaling keeps variance stable regardless of d_k

2. **What's the time/space complexity, and how would you reduce it for 100K-length sequences?**
   - State O(n²·d) for both compute and memory
   - Mention sparse attention, linear attention (Performer/Linformer), FlashAttention as IO-aware solution
   - Note trade-off: approximation vs exactness

3. **Differentiate self-attention, cross-attention, and causal attention.**
   - Self: Q,K,V from same sequence
   - Cross: Q from one sequence, K/V from another (encoder-decoder)
   - Causal: masked self-attention restricting to past tokens only

4. **Why does multi-head attention help, and what happens with too many/few heads?**
   - Different heads specialize in different relationship types
   - Too few: limited representational diversity
   - Too many: each head's dimension shrinks, risking diluted, noisy attention patterns

5. **How do you handle masking for variable-length sequences in a batch, and what bugs commonly occur?**
   - Use padding masks to block attention to pad tokens
   - Common bug: forgetting to mask pad tokens → information leakage
   - Common bug: applying mask after softmax instead of before (incorrect probability mass)

---

## 9. Quick Reference Table

| Item | Detail |
|---|---|
| Algorithm Type | Sequence representation learning mechanism (core building block of Transformers) |
| Time Complexity | O(n² · d) — quadratic in sequence length n |
| Space Complexity | O(n²) for the attention score matrix (per head) |
| Key Hyperparameters | Number of heads (h), model dimension (d_model), head dimension (d_k), dropout rate |
| Evaluation Metrics | Task-dependent (perplexity for LM, BLEU for translation, accuracy/F1 for classification); attention maps used qualitatively, not as a metric |

---

## 10. References & Further Reading

- **Original Paper:** Vaswani et al., *"Attention Is All You Need"* (2017) — https://arxiv.org/abs/1706.03762
- **Best Tutorial:** Jay Alammar, *"The Illustrated Transformer"* — https://jalammar.github.io/illustrated-transformer/
- **Visual/Interactive Guide:** *"The Annotated Transformer"* (Harvard NLP) — https://nlp.seas.harvard.edu/annotated-transformer/
- **Kaggle Notebook:** Search "Transformer from scratch" notebooks on Kaggle for hands-on PyTorch implementations — https://www.kaggle.com/search?q=transformer+from+scratch
- **PyTorch Docs:** `torch.nn.MultiheadAttention` and `scaled_dot_product_attention` — https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
