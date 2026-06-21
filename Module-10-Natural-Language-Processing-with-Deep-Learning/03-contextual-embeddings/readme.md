# Contextual Embeddings
> *The same word, a different vector — because meaning lives in context*

After studying this topic, you will understand how transformer-based models produce dynamic word representations that shift with sentence context, solving the fundamental polysemy problem that static embeddings cannot. You will be able to extract contextual embeddings using HuggingFace, fine-tune BERT for downstream tasks, and reason confidently about architectural tradeoffs in interviews.

---

## What Are Contextual Embeddings?

In Word2Vec or GloVe, every word gets exactly one vector — forever. "Bark" gets the same representation whether you're talking about a dog or a tree. This is a hard ceiling on what those models can understand. Contextual embeddings break this ceiling: each word token gets a *different* vector depending on the sentence it appears in. The word "bank" in "I deposited money at the bank" and in "We sat on the river bank" will produce different vectors — and downstream models can act on that difference.

The key invention that made this possible is the **attention mechanism** inside a transformer. Instead of processing each word in isolation or in a fixed left-to-right sequence, a transformer lets every token "look at" every other token in the sentence simultaneously. The representation of each token is then an aggregation of information from the whole context, weighted by relevance. The result is a rich, context-aware vector for every position in the input.

**The real-world analogy:** Think about the word "cold" in three different sentences — "I caught a cold," "Turn on the cold water," and "She gave him a cold stare." A human reader instantly adjusts their interpretation of "cold" based on surrounding words. Contextual embeddings teach machines to do the same thing: the model reads the whole sentence first, then assigns a meaning-vector to each word that reflects its role in *that specific context*. It's the difference between looking up a word in a dictionary (one fixed definition) versus reading the sentence and inferring meaning from context like a fluent speaker would.

---

## Mathematical Formulation

### Scaled dot-product attention

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

**What this tells us:** For each token, $Q$ (query) asks "what am I looking for?", $K$ (key) says "what does each token offer?", and $V$ (value) holds what each token actually contributes. The dot product $QK^\top$ scores compatibility between every pair of tokens. Dividing by $\sqrt{d_k}$ prevents the scores from growing too large when the dimension $d_k$ is high — large values push the softmax into flat regions where gradients vanish. The softmax converts scores to weights, and the output is a weighted sum of values.

### Multi-head attention

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\,W^O$$
$$\text{where } \text{head}_i = \text{Attention}(QW_i^Q,\, KW_i^K,\, VW_i^V)$$

**What this tells us:** Running $h$ parallel attention operations on different learned projections lets the model simultaneously capture different types of relationships — one head might track syntactic dependencies, another co-reference, another semantic similarity — then merge them.

### Transformer block (residual + layer norm)

$$x' = \text{LayerNorm}(x + \text{MultiHead}(x))$$
$$\text{output} = \text{LayerNorm}(x' + \text{FFN}(x'))$$

**What this tells us:** Residual connections add the input back after each sub-layer, preventing information loss and enabling gradients to flow cleanly through many layers. Layer normalization stabilizes activations without depending on batch statistics — critical for sequence tasks where batch sizes are small.

### Masked language modeling (MLM) objective

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(w_i \mid w_{\setminus \mathcal{M}})$$

**What this tells us:** For each masked position $i$ in set $\mathcal{M}$, the model predicts the original token using all *non-masked* surrounding tokens — both left and right context. This is what makes BERT bidirectional. The training signal is entirely self-supervised: no human labels are needed, just raw text.

---

## How It Works — Step by Step

**Using BERT's pre-training and fine-tuning pipeline as the reference model:**

1. **Tokenize with subword vocabulary.** Input text is split using WordPiece into subword tokens. "unbelievable" might become ["un", "##believable"]. Special tokens `[CLS]` (start) and `[SEP]` (separator) are prepended/appended. *Why it matters: subword tokenization eliminates out-of-vocabulary words — any string can be represented.*

2. **Build input embeddings.** Each token ID is looked up in a learned token embedding matrix, added to a positional embedding (encoding its position in the sequence) and a segment embedding (indicating sentence A vs B). These three are summed to form the initial representation. *Think of it as: "what is this word" + "where does it sit" + "which sentence is it from."*

3. **Pass through stacked transformer layers.** The summed embeddings flow through $L$ transformer blocks (BERT-base: 12 layers). In each block, every token attends to every other token via multi-head attention, then passes through a feedforward sublayer. Lower layers build syntactic awareness; upper layers build semantic and task-specific representations.

4. **Apply masking (during pre-training).** 15% of tokens are selected. Of those: 80% are replaced with `[MASK]`, 10% with a random word, 10% left unchanged. The model must reconstruct the original tokens. *The 10%/10% split is deliberate: it prevents the model from only caring about `[MASK]` positions — representations must remain useful for all tokens.*

5. **Extract the final hidden states.** After the last transformer layer, each token has a context-aware vector of dimension $d_{model}$ (768 for BERT-base). These are the contextual embeddings. For sentence-level tasks, mean-pool all token vectors or use the `[CLS]` token (after fine-tuning).

6. **Fine-tune on a downstream task.** Add a lightweight task head (e.g., a linear classifier) on top of the frozen or partially frozen encoder. Train end-to-end on labeled data with a small learning rate (2e-5 to 5e-5). The pre-trained weights adapt to the task without losing general knowledge.

---

## Architecture: BERT Transformer Block

```
Input tokens + Positional + Segment embeddings
           │
    ┌──────▼──────────────────────────────┐
    │       Multi-Head Self-Attention      │
    │   (12 heads × 64-dim each = 768)     │
    └──────┬──────────────────────────────┘
           │   Residual connection
    ┌──────▼──────────────────────────────┐
    │         Layer Normalization          │
    └──────┬──────────────────────────────┘
           │
    ┌──────▼──────────────────────────────┐
    │   Feed-Forward Network (768→3072→768)│
    └──────┬──────────────────────────────┘
           │   Residual connection
    ┌──────▼──────────────────────────────┐
    │         Layer Normalization          │
    └──────┬──────────────────────────────┘
           │
    Contextual token vectors (batch × seq_len × 768)
           │
    ×12 layers (BERT-base) / ×24 layers (BERT-large)
```

---

## Key Assumptions

| Assumption | What breaks if violated |
|---|---|
| **Sequence fits within max length** (512 tokens for BERT) | Long documents are silently truncated — sentiment in the tail of a review or conclusion of a legal doc is lost; chunking + pooling strategies required |
| **Pretrain domain ≈ inference domain** | General BERT on clinical or legal text has poor coverage of domain vocabulary; use BioBERT / LegalBERT or run task-adaptive pre-training (TAPT) |
| **Fine-tuning data is sufficient** (~1K–100K examples) | With fewer than ~500 examples, catastrophic forgetting and overfitting dominate; use lower LR, freeze early layers, or use few-shot prompting instead |
| **CLS token represents the sentence** (after fine-tuning) | Raw pretrained `[CLS]` is a poor sentence embedding; mean pooling of last hidden layer outperforms `[CLS]` for similarity tasks without fine-tuning |
| **Subword tokens map cleanly to labels** (in NER) | A single word splits into multiple subword tokens — only first-token labeling causes subtle misalignment bugs; must be handled explicitly in the data pipeline |
| **Embedding space is isotropic** | BERT embeddings cluster in a narrow cone (anisotropy); cosine similarity is unreliable for semantic search without whitening or SBERT fine-tuning |

---

## When to Use / When Not to Use

| ✅ Use contextual embeddings when… | ❌ Avoid / reconsider when… |
|---|---|
| Word sense matters — "lead" (metal) vs "lead" (guide) | Latency is critical (< 5ms) and the task is simple — a TF-IDF + logistic regression may be sufficient |
| You have a pretrained model close to your domain | Your corpus is tiny (< 500 labeled examples) and no pretrained model matches the domain |
| Sentence-level semantic search or QA extraction | You need to process millions of tokens per second on CPU — compute cost is prohibitive |
| NER, sentiment, coreference, or classification on varied text | Interpretability is legally required — transformer internals are hard to audit vs linear models |
| You can afford GPU memory for a 110M+ parameter model | The task is pure keyword matching or exact-string retrieval — simpler models win |
| Transfer learning from a related domain is viable | Privacy constraints prohibit sending text to a hosted API (e.g., healthcare PHI data) |

---

## Implementation Overview

| | From Scratch (PyTorch) | Library (HuggingFace Transformers) |
|---|---|---|
| **Tokenization** | Hand-roll BPE or WordPiece vocabulary | `AutoTokenizer.from_pretrained()` — handles all special tokens |
| **Model architecture** | Build `nn.MultiheadAttention` + `nn.TransformerEncoderLayer` blocks | `AutoModel.from_pretrained()` — loads weights instantly |
| **Pre-training** | Write MLM masking pipeline, train on raw corpus (requires TPUs / multi-GPU weeks) | Not needed — use pretrained weights; optionally continue pre-training with `Trainer` |
| **Fine-tuning** | Custom training loop with gradient accumulation and LR warmup | `Trainer` + `TrainingArguments` — handles distributed training, mixed precision |
| **Embedding extraction** | Manually slice `output.last_hidden_state` tensors | `pipeline("feature-extraction")` or direct model call with `output_hidden_states=True` |
| **Best for** | Learning attention internals, custom architectures | Production fine-tuning, rapid prototyping, deployment |

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

dataset = load_dataset("imdb")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

tokenized = dataset.map(tokenize, batched=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,          # LR ramps up for first 500 steps
    learning_rate=2e-5,        # small LR to avoid catastrophic forgetting
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,                 # mixed-precision for ~2x speedup on GPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
```

---

## Top 5 Interview Questions

**Q1. BERT is bidirectional but GPT is unidirectional. For a dense retrieval system, which would you choose for encoding documents and queries, and why?**
- BERT encoder: sees full left + right context → richer token representations
- GPT decoder: causal mask blocks future tokens → suboptimal for encoding tasks
- Neither raw model is ideal — use SBERT (siamese BERT fine-tuned with contrastive loss)
- For asymmetric retrieval (short query, long doc): consider separate query and document encoders (DPR)
- Mention: BERT's `[CLS]` alone is unreliable without fine-tuning; mean pooling preferred

**Q2. Walk me through the scaled dot-product attention mechanism. Why divide by √dₖ, and what breaks without it?**
- Q, K, V matrices computed via learned linear projections of input
- Dot product $QK^\top$ scores each token pair's compatibility
- Without scaling: dot products grow proportionally to $d_k$ — large values push softmax to near-zero gradient regions (saturation)
- $\sqrt{d_k}$ normalizes variance to ~1 regardless of dimension
- Softmax of normalized scores → attention weights → weighted sum of V vectors

**Q3. You fine-tune BERT on a 3-class classifier. Training accuracy is 95% but test accuracy is 61%. What BERT-specific failure modes do you diagnose first?**
- Catastrophic forgetting: LR too high (> 5e-5) overwrote pretrained features
- Dataset too small: overfit to fine-tuning set; BERT needs ~1K+ examples to generalize
- Sequence truncation: if class signal lives in text beyond 512 tokens
- Label leakage in tokenized dataset construction
- Domain mismatch: if general BERT never saw your domain's vocabulary
- Remediation: lower LR, freeze early layers, augment data, or switch to domain-adapted model

**Q4. Describe exactly what BERT's MLM masking strategy is, including the 80/10/10 rule and why each percentage exists.**
- 15% of tokens are candidates for masking
- Of those 15%: 80% replaced with `[MASK]`, 10% replaced with random token, 10% unchanged
- 80% `[MASK]`: trains the model to reconstruct missing tokens using context
- 10% random: prevents model from only "trying" at `[MASK]` positions — forces all positions to maintain useful representations
- 10% unchanged: creates a discrepancy between pre-training (masked) and fine-tuning (no masks) — unchanged tokens help bridge this gap
- NSP (Next Sentence Prediction) is a second pre-training objective — later shown by RoBERTa to be largely unnecessary

**Q5. Your team must serve a BERT classifier at p99 < 10ms for 50K RPS. Walk through your optimization stack.**
- Step 1: model compression — switch to DistilBERT (40% smaller, 60% faster, ~97% accuracy retention)
- Step 2: quantization — INT8 post-training quantization via ONNX or TorchScript (2–4× speedup, minimal accuracy loss)
- Step 3: export to ONNX + TensorRT for GPU-optimized kernel fusion
- Step 4: reduce max sequence length to the 95th percentile of your actual input distribution
- Step 5: dynamic batching at the serving layer (Triton Inference Server)
- Step 6: caching layer for repeated/similar inputs (input hash → embedding cache)
- Know the math: 50K RPS at 10ms budget = 500 concurrent requests; GPU throughput vs CPU throughput decision

---

## Quick Reference Table

| Item | Detail |
|---|---|
| **Algorithm type** | Self-supervised representation learning via masked language modeling (transformer encoder) |
| **Pre-training complexity** | $O(n^2 \cdot d)$ per layer — $n$ = sequence length, $d$ = hidden dimension; quadratic in sequence length |
| **Inference complexity** | $O(L \cdot n^2 \cdot d)$ — $L$ layers; scales poorly for long sequences (Longformer, BigBird address this) |
| **Space complexity** | $O(n^2)$ for attention matrices per layer; 110M params (BERT-base), 340M (BERT-large) |
| **Key hyperparameters** | `max_seq_length` (128–512), `learning_rate` (1e-5 to 5e-5), `warmup_steps`, `num_layers` to freeze, `num_heads` |
| **Fine-tuning eval metrics** | F1 / accuracy (classification), span-level F1 (QA), entity-level F1 (NER), GLUE benchmark score |
| **Embedding eval metrics** | Spearman ρ on STS-B (similarity), BEIR benchmark (retrieval), probing accuracy per layer |
| **Typical embedding dim** | 768 (BERT-base / DistilBERT), 1024 (BERT-large), 384 (MiniLM) |

---

## References & Further Reading

| Resource | Why it matters |
|---|---|
| [Devlin et al. (2019) — BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) | Original BERT paper — read Section 3 (model architecture) and Section 4 (pre-training procedure) |
| [Vaswani et al. (2017) — Attention Is All You Need](https://arxiv.org/abs/1706.03762) | The transformer paper — required reading; Section 3.2 is the attention mechanism derivation |
| [Jay Alammar — The Illustrated BERT, ELMo, and co.](https://jalammar.github.io/illustrated-bert/) | Best visual walkthrough of contextual embeddings; pairs perfectly with the papers |
| [Reimers & Gurevych (2019) — Sentence-BERT](https://arxiv.org/abs/1908.10084) | The SBERT paper — essential for understanding why raw BERT CLS fails for similarity and how contrastive fine-tuning fixes it |
| [Rogers et al. (2020) — A Primer in BERTology](https://arxiv.org/abs/2002.12327) | Comprehensive survey of what BERT actually learns layer-by-layer; critical for interview Q4 and probing questions |