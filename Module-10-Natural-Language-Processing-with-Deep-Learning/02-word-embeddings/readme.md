# Word Embeddings
> *Teaching machines to understand meaning by placing words in geometric space*

After studying this topic, you will understand how raw text is transformed into numerical vectors that capture semantic meaning, and why this is the foundational step for nearly every modern NLP system. You will be able to implement Word2Vec from scratch, use pretrained embeddings in production, and reason about their failure modes in interviews.

---

## What Are Word Embeddings?

Every machine learning model needs numbers as input — but language is made of words. The naive approach is to give each word a unique integer ID, or a one-hot vector (a vector of all zeros with a single 1 at the word's position). These representations have a fatal flaw: they treat every word as equally unrelated to every other word. "Dog" and "puppy" look as different as "dog" and "refrigerator." The model has no way to generalize.

Word embeddings solve this by representing each word as a dense vector of real numbers — typically 100 to 300 dimensions — learned from data. Words used in similar contexts end up near each other in this vector space. "King" and "queen" cluster together. "Paris" and "Berlin" cluster together. The geometry of the space encodes meaning, and arithmetic in that space produces startlingly sensible results.

**The real-world analogy:** Think of a city map. Every location is described by two numbers (latitude and longitude), and distance on the map reflects how physically close two places are. Word embeddings create an analogous map for language — except instead of 2 dimensions, you might use 300, and "closeness" means semantic similarity rather than physical proximity. Just as you can reason about directions on a map ("go north from Paris to reach London"), you can reason about directions in embedding space ("go from 'man' toward 'woman' and the same direction from 'king' lands near 'queen'").

---

## Mathematical Formulation

### Skip-gram objective (softmax form)

$$P(w_o \mid w_c) = \frac{\exp(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c})}{\sum_{w=1}^{V} \exp(\mathbf{u}_w^\top \mathbf{v}_{w_c})}$$

**What this tells us:** For a center word $w_c$, the model assigns a probability to each possible context word $w_o$ using the dot product of their vectors (higher dot product = more likely to co-occur). Training maximizes this probability across all observed context pairs in the corpus. The denominator sums over the entire vocabulary $V$ — which is what makes this expensive.

### Negative sampling loss (the practical approximation)

$$\mathcal{L} = \log \sigma(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c}) + \sum_{k=1}^{K} \mathbb{E}_{w_k \sim P_n(w)}\left[\log \sigma(-\mathbf{u}_{w_k}^\top \mathbf{v}_{w_c})\right]$$

**What this tells us:** Instead of normalizing over all $V$ words, we reframe the problem as binary classification: is this a real word pair or a fake (noise) pair? We maximize the score for real pairs and minimize it for $K$ randomly sampled noise pairs. This reduces per-step cost from $O(V)$ to $O(K)$, where $K$ is typically 5–20.

### GloVe loss

$$\mathcal{L} = \sum_{i,j=1}^{V} f(X_{ij}) \left(\mathbf{v}_i^\top \mathbf{u}_j + b_i + b_j - \log X_{ij}\right)^2$$

**What this tells us:** Unlike Word2Vec which processes one context window at a time, GloVe directly fits word vectors to the full corpus co-occurrence matrix $X$. The weighting function $f(X_{ij})$ down-weights very common pairs (like "the the") and gives zero weight to pairs that never co-occur. The model learns vectors whose dot products best approximate the log of how often two words appear together globally.

### Cosine similarity

$$\text{sim}(A, B) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$

**What this tells us:** The angle between two vectors measures semantic relatedness, regardless of their magnitudes. A value of 1.0 means identical direction (near-synonyms), 0 means orthogonal (unrelated), −1 means opposite.

---

## How It Works — Step by Step

**Using Skip-gram with Negative Sampling as the example:**

1. **Build vocabulary.** Scan the corpus, assign an integer ID to every unique token. Common practice: discard words appearing fewer than 5 times. For a corpus like Wikipedia, vocabulary size is typically 1–3 million.

2. **Generate training pairs.** Slide a context window (size 2–5) over every sentence. For each center word, every word in the window becomes a positive (real) pair. *Example: In "the quick brown fox", with window size 1 and center word "brown": pairs are (brown→quick) and (brown→fox).*

3. **Sample noise pairs.** For each positive pair, randomly sample $K$ words from the vocabulary as fake context words. Words are sampled proportionally to their frequency raised to the 3/4 power — this slightly boosts rare words relative to pure frequency sampling.

4. **Look up vectors.** The model maintains two matrices: $V_{in}$ (center word embeddings) and $V_{out}$ (context word embeddings), each of shape $|vocabulary| \times d$. Look up the center word row in $V_{in}$ and each context/noise word row in $V_{out}$.

5. **Compute loss and update.** Compute the negative sampling loss. Real pairs should have high dot products; noise pairs should have low ones. Backpropagate — only the rows corresponding to the current center word and the $K+1$ context words are updated. This sparsity is what makes training fast.

6. **Extract embeddings.** After training, use $V_{in}$ as your word vectors (or average $V_{in}$ and $V_{out}$). Discard $V_{out}$.

---

## Key Assumptions

| Assumption | What breaks if violated |
|---|---|
| **Distributional hypothesis** — words in similar contexts have similar meaning | Fails for antonyms ("hot"/"cold" appear in similar contexts but have opposite meaning) and for technical/domain jargon with thin context |
| **Stationarity** — word meanings are consistent across the corpus | Fails with time-sensitive corpora (e.g., "tweet" changed meaning post-2006) or multi-domain text |
| **Sufficient corpus size** — rare words need enough context to converge | Words appearing fewer than ~5 times get poor embeddings; OOV at inference is completely unhandled |
| **Single meaning per word** — one vector per word token | Polysemous words ("bank," "crane," "bat") collapse all senses into one blurred vector |
| **Linear structure encodes semantics** — analogies are linear offsets | Breaks for non-linear relationships; the famous king−man+woman≈queen result doesn't generalize to all analogies |

---

## When to Use / When Not to Use

| ✅ Use word embeddings when… | ❌ Avoid / reconsider when… |
|---|---|
| You need a fast, lightweight semantic representation with no inference latency | Your vocabulary is highly specialized and no domain corpus exists for training |
| Tasks are word-level: NER, POS tagging, word similarity | Context matters critically — "I went to the river **bank**" vs "I went to the **bank**" |
| You have limited compute and can't run a transformer | You need state-of-the-art accuracy on downstream NLP tasks |
| Building a recommendation system adapting the item2vec pattern | Your corpus is tiny (< 1 million tokens) — vectors won't converge meaningfully |
| Initializing transformer embedding layers (pretrained weights) | You need multilingual or cross-lingual representations |
| Semantic search with FAISS over millions of documents at low cost | Fairness is a critical concern and bias auditing infrastructure isn't in place |

---

## Implementation Overview

| | From Scratch (NumPy) | Library (Gensim / PyTorch) |
|---|---|---|
| **Vocab building** | Manual token counting, `word2idx` dict | Automatic, configurable `min_count` |
| **Data pipeline** | Write context-window generator by hand | Built-in corpus iterators |
| **Weight matrices** | Two NumPy arrays, random init | `nn.Embedding` or Gensim handles internals |
| **Training loop** | Manual gradient computation, in-place SGD updates | `model.train()` call; Gensim streams corpus |
| **Negative sampling** | Sample from `freq^0.75` distribution manually | Controlled by `negative` parameter |
| **Best for** | Learning the mechanics, interviews, educational settings | Production, fine-tuning, large corpora |
| **Gotcha** | Slow on large corpora without batching | Gensim's `min_count` silently drops rare words |

```python
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Tokenize corpus (list of sentences, each a list of tokens)
corpus = [simple_preprocess(sentence) for sentence in raw_sentences]

# Train skip-gram with negative sampling
model = Word2Vec(
    sentences=corpus,
    vector_size=100,      # embedding dimension
    window=5,             # context window size
    min_count=5,          # ignore words with fewer occurrences
    sg=1,                 # 1 = skip-gram, 0 = CBOW
    negative=10,          # number of negative samples
    epochs=10,
    workers=4
)

# Retrieve a vector
king_vec = model.wv["king"]                        # shape: (100,)

# Find most similar words
model.wv.most_similar("king", topn=5)

# Analogy: king - man + woman
model.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=1)

# Save and load
model.wv.save_word2vec_format("embeddings.bin", binary=True)
```

---

## Top 5 Interview Questions

**Q1. Walk me through what the skip-gram model is actually learning. What are the two weight matrices?**
- Ideal answer structure:
  - The model learns two matrices: $V_{in}$ (input/center embeddings) and $V_{out}$ (context embeddings)
  - The objective pushes $V_{in}[w_c] \cdot V_{out}[w_o]$ high for real pairs, low for noise pairs
  - At the end, we typically use $V_{in}$ as the embedding; $V_{out}$ is discarded
  - Both matrices start random; gradient descent brings semantically related rows closer together

**Q2. Why can't you use a full softmax over the vocabulary, and how does negative sampling fix this?**
- Ideal answer structure:
  - Full softmax requires summing over all $V$ words in the denominator — $O(V)$ per update, $V$ can be millions
  - Negative sampling converts the problem to binary classification: real pair vs noise pair
  - Loss is logistic: maximize score for positives, minimize for $K$ negatives per positive
  - Complexity drops to $O(K)$ per step; $K$ is typically 5–20
  - Noise distribution: unigram frequency raised to 3/4 (flattens the distribution toward rare words)

**Q3. How does GloVe differ from Word2Vec in its training objective?**
- Ideal answer structure:
  - Word2Vec: local objective, processes one context window at a time, implicitly factorizes the PMI matrix
  - GloVe: global objective, builds full co-occurrence matrix first, then minimizes weighted squared error between dot products and log co-occurrence counts
  - GloVe trains faster to convergence on large corpora; Word2Vec handles streaming data more naturally
  - Both produce vectors of comparable quality; GloVe is easier to reason about mathematically

**Q4. A production system is built on Word2Vec but performs poorly on medical notes. What's your remediation strategy?**
- Ideal answer structure:
  - Diagnose: OOV rate, embedding similarity sanity checks for domain terms
  - Option 1: Fine-tune existing embeddings on medical corpus
  - Option 2: Train FastText from scratch on medical corpus (handles morphological variants like "hepatomegaly" via subword)
  - Option 3: Switch to domain-pretrained BERT (BioBERT, ClinicalBERT) — contextual embeddings eliminate OOV entirely
  - Consider: data availability, latency budget, whether single-word or sentence-level representations are needed

**Q5. How do you go from word embeddings to sentence-level embeddings for semantic search?**
- Ideal answer structure:
  - Naive: average all word vectors in the sentence — fast but ignores word order and weights all words equally
  - Better: SIF (Smooth Inverse Frequency) weighting — downweight common words, remove top principal component
  - Production: bi-encoder architecture (SBERT) — fine-tuned transformer produces sentence vectors directly
  - Asymmetric retrieval: query encoder and document encoder can differ (DPR for open-domain QA)
  - Failure modes of averaging: negation ("not good" ≈ "good"), stopword dominance, length sensitivity

---

## Quick Reference Table

| Item | Detail |
|---|---|
| **Algorithm type** | Self-supervised representation learning (neural language model) |
| **Training time complexity** | $O(T \cdot C \cdot K \cdot d)$ — $T$ tokens, $C$ window size, $K$ negatives, $d$ dimensions |
| **Inference time complexity** | $O(1)$ — table lookup per word |
| **Space complexity** | $O(V \cdot d)$ — vocabulary size × embedding dimension |
| **Key hyperparameters** | `vector_size` (50–300), `window` (2–10), `negative` (5–20), `min_count` (2–10), `epochs` (5–20) |
| **Intrinsic eval metrics** | Word similarity (SimLex-999, WordSim-353), analogy accuracy (Google Analogy Dataset) |
| **Extrinsic eval metrics** | NER F1, sentiment accuracy, QA EM — performance on downstream tasks |
| **Common vocab size** | 50K–3M tokens |
| **Typical embedding dim** | 100 (fast), 200 (balanced), 300 (quality) |

---

## References & Further Reading

| Resource | Why it matters |
|---|---|
| [Mikolov et al. (2013) — Distributed Representations of Words](https://arxiv.org/abs/1301.3781) | Original Word2Vec paper — read the abstract and Section 4 |
| [Pennington et al. (2014) — GloVe](https://nlp.stanford.edu/pubs/glove.pdf) | GloVe paper; Section 3 explains the objective clearly |
| [Jay Alammar — The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) | Best visual explainer available; ideal before going into math |
| [Levy & Goldberg (2014) — Neural Word Embedding as Implicit Matrix Factorization](https://papers.nips.cc/paper/2014/hash/feab05aa91085b7a8012516bc3533958-Abstract.html) | Proves skip-gram + NS implicitly factorizes PMI matrix — critical for interviews |
| [Stanford CS224N Lecture Notes — Word Vectors](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes01-wordvecs1.pdf) | Best structured lecture notes; covers math and intuition cleanly |