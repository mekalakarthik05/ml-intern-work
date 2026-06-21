# Tokenization

> Turning raw text into the numeric building blocks a model can actually understand.

**What you will learn:** You will understand why text must be broken into tokens before a model can process it, and how algorithms like BPE, WordPiece, and Unigram LM decide where to draw those boundaries. You will also be able to implement a simple tokenizer from scratch and use production libraries like HuggingFace `tokenizers` or `tiktoken` confidently in real projects.

---

## 2. What is Tokenization?

Tokenization is the process of splitting a piece of text into smaller units — called **tokens** — that a machine learning model can map to numbers. A model never sees the word "unbelievable." It sees a sequence of IDs like `[412, 891]`, where each ID points to a row in an embedding table. Tokenization is the translator that sits between human language and that numeric world.

Think of it like a vending machine that only accepts a fixed set of coins. You can't insert a $20 bill directly — you need to break it down into coins the machine recognizes (quarters, dimes, nickels). Words work the same way: a model has a fixed "coin set" called its **vocabulary**, and tokenization is the process of breaking any incoming sentence into pieces that exist in that vocabulary. If you only had quarters (whole words), you couldn't represent a $0.10 word (a rare or misspelled word) — you'd be stuck. If you only had pennies (characters), you could represent anything, but you'd need a huge pile of coins for every transaction (very long sequences). Subword tokenization is the practical middle ground: common words stay whole, rare ones get broken into reusable pieces like prefixes and suffixes.

This trade-off is the entire story of modern tokenization. Word-level tokenizers are intuitive but explode in vocabulary size and can't handle words they've never seen. Character-level tokenizers handle anything but produce very long sequences that are computationally expensive and harder for models to learn patterns from. Subword methods — BPE, WordPiece, Unigram — sit in between, learned directly from data so that frequent words stay intact while rare or unseen words decompose into familiar fragments.

---

## 3. Mathematical Formulation

**BPE merge rule** (greedy, frequency-based):

$$\text{merge}(a, b) = \arg\max_{(a,b)} \; \text{count}(a, b)$$

*Significance:* at every step, BPE looks at the corpus and merges whichever adjacent symbol pair appears most often. This is a pure popularity contest — no probability involved.

**WordPiece score** (likelihood-ratio based):

$$\text{score}(a, b) = \frac{\text{count}(a, b)}{\text{count}(a) \times \text{count}(b)}$$

*Significance:* this favors pairs that co-occur *more than expected by chance*, not just pairs that are simply frequent individually. It rewards genuine statistical association.

**Unigram LM objective:**

$$\mathcal{L} = \sum_{x} \log P(x), \quad P(x) = \sum_{s \in \text{Seg}(x)} \prod_{i} P(\text{token}_i)$$

*Significance:* every word can be split in multiple valid ways, and the model picks (or weights) the segmentation that maximizes overall corpus likelihood. Tokenization here is treated as a tiny probabilistic language model.

---

## 4. How It Works — Step by Step (BPE example)

1. **Start with characters.** Split every word into individual characters plus an end-of-word marker. *Example:* `"low"` → `l o w </w>`.
2. **Count adjacent pairs.** Tally how often every pair of neighboring symbols appears across the whole corpus. *Example:* `(l, o)` appears 30 times, `(o, w)` appears 25 times.
3. **Merge the most frequent pair.** Combine that pair into a single new symbol everywhere it appears. *Example:* merge `(l, o)` → `lo`, so `"low"` becomes `lo w </w>`.
4. **Repeat.** Recount pairs with the new symbol set and merge again, growing the vocabulary one merge at a time, like building bigger LEGO blocks from smaller ones.
5. **Stop at target vocabulary size.** Once you hit a predefined number of merges (e.g., 32,000), freeze the merge list — this *is* your trained tokenizer.
6. **Encode new text.** For a new word, apply the learned merges in the exact order they were learned, just like re-running the LEGO assembly instructions on new pieces.

---

## 5. Key Assumptions

| Assumption | If Violated |
|---|---|
| Training corpus is representative of deployment text | Vocabulary is mismatched; deployment text fragments excessively (e.g., training on English, deploying on code) |
| Frequency in training data implies relevance | Rare-but-important terms (medical, legal jargon) get over-fragmented |
| Tokens are roughly independent (Unigram LM) | Loses syntactic/semantic dependency information; segmentation may be locally optimal but globally odd |
| Whitespace is a meaningful boundary signal | Breaks for languages without spaces (Chinese, Japanese, Thai) |
| Merge order learned in training generalizes to all future text | Out-of-distribution scripts or symbols fall back to byte-level/unknown tokens |

---

## 6. When to Use / When Not to Use

| Use Subword Tokenization (BPE/WordPiece/Unigram) | Avoid / Reconsider |
|---|---|
| Building or fine-tuning a transformer-based LLM | Tiny vocab, fixed-domain classification with very limited terms (simple word-level may suffice) |
| Multilingual or open-domain text | Languages without clear word-boundaries without using SentencePiece variant |
| Need balance between sequence length and vocab size | Extremely latency-sensitive edge deployment where even tokenization overhead matters (consider char-level) |
| Domain has many rare/compound words (biomedical, code) | You control the entire closed vocabulary already (e.g., fixed enum-like categorical text) |
| Need to share tokenizer across many languages | Pure single-language, well-segmented data, where simple whitespace + stemming may be enough |

---

## 7. Implementation Overview

| Aspect | From Scratch (Python) | Library (HuggingFace `tokenizers` / `tiktoken`) |
|---|---|---|
| Vocabulary building | Manually count pairs, iteratively merge, store merge list | `BpeTrainer` or `tiktoken` precomputed merge ranks — single call |
| Speed | Slow (pure Python loops over corpus) | Fast (Rust/C backend, parallelized) |
| Encoding new text | Apply merges sequentially in a loop | Optimized trie/regex-based matching, near-instant |
| Special tokens | Manually inserted and tracked | Built-in handling (`<bos>`, `<eos>`, `<pad>`) |
| Best for | Learning the mechanics deeply | Any real production system |

```python
# Library-based tokenizer training (conceptual equivalent of "fit")
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE(unk_token="<unk>"))
trainer = BpeTrainer(vocab_size=32000, special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"])

tokenizer.train(files=["corpus.txt"], trainer=trainer)   # equivalent to .fit()
output = tokenizer.encode("unbelievable scaling laws")    # equivalent to .transform()
print(output.tokens)
```

---

## 8. Top 5 Interview Questions

1. **Walk through how BPE builds a vocabulary and why it merges the most frequent pair.**
   - Mention character-level start, pair counting, greedy merge, iteration, stopping criterion.
2. **Why does WordPiece use a likelihood-ratio score instead of raw frequency?**
   - Contrast frequency vs. statistical association; mention avoiding bias toward already-common symbols.
3. **Trace tokenization of an unseen word through BPE vs. byte-level BPE.**
   - Discuss OOV fallback, byte-level guarantee of full coverage, no `<unk>` needed.
4. **Why might an LLM struggle with large-number arithmetic, and how does tokenization contribute?**
   - Discuss inconsistent digit grouping, fix via digit-level or fixed-width number tokenization.
5. **Design a tokenizer for a low-resource language with no existing tokenizer.**
   - Discuss Unigram/SentencePiece choice, vocab size trade-offs, multilingual vocabulary sharing.

---

## 9. Quick Reference Table

| Item | Detail |
|---|---|
| Algorithm Type | Unsupervised, data-driven vocabulary construction (greedy or probabilistic) |
| Time Complexity | Training: ~O(n·k) for n corpus size, k merges; Encoding: near O(length) with optimized libraries |
| Space Complexity | O(vocab size) for merge table / embedding lookup |
| Key Hyperparameters | Vocabulary size, special tokens, pre-tokenization rules (whitespace/punctuation splitting) |
| Evaluation Metrics | Tokens-per-word ratio, compression rate (bytes/token), OOV rate, downstream model perplexity |

---

## 10. References & Further Reading

- **Original Paper (BPE for NLP):** [Sennrich et al., 2016 — Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- **WordPiece origin:** [Schuster & Nakajima, 2012 — Japanese and Korean Voice Search](https://research.google/pubs/japanese-and-korean-voice-search/)
- **SentencePiece:** [Kudo & Richardson, 2018 — SentencePiece Library](https://github.com/google/sentencepiece)
- **Best Practical Tutorial:** [HuggingFace — Tokenizers Documentation](https://huggingface.co/docs/tokenizers/index)
- **Hands-on Notebook:** [HuggingFace Course — Building a Tokenizer from Scratch](https://huggingface.co/learn/nlp-course/chapter6/8)

---

*Note: This topic does not involve a neural network layer, so a batch normalization diagram is not applicable here — see the Implementation Overview section above for the closest equivalent comparison (from-scratch vs. library tokenizer training).*