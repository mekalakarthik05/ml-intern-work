# Text Summarization

**Tagline:** *Teaching machines to read a lot and say a little — without losing what matters.*

**What you will learn:** By the end of this README, you will understand the two dominant paradigms of automatic text summarization — extractive (selecting) and abstractive (generating) — along with the math, mechanics, and failure modes behind each. You will also know how to build a simple extractive summarizer using `scikit-learn` and how to evaluate summary quality using ROUGE.

---

## What is Text Summarization?

Text summarization is the task of automatically condensing a long piece of text into a shorter version that retains its most important information. Think of it as the machine-learning equivalent of a news editor writing a headline-and-byline preview of a 2,000-word article — the reader gets the gist in seconds instead of minutes.

There are two fundamentally different ways to do this. **Extractive summarization** works like highlighting a textbook with a yellow marker: you don't write anything new, you just pick out the sentences that already exist in the document and that seem most important. **Abstractive summarization** is closer to how a human actually summarizes — you read the whole passage, understand it, and then write a brand-new sentence in your own words that captures the meaning, even if those exact words never appeared in the source.

A useful analogy: extractive summarization is like building a "best-of" highlight reel from raw match footage — every clip is real, but the editing choices determine quality. Abstractive summarization is more like a sports commentator describing the match from memory — more fluent and natural, but with the real risk of misremembering a detail. This is precisely why abstractive models are more powerful but also more dangerous: they can "hallucinate" facts that were never in the original text, the way an overconfident commentator might describe a goal that didn't happen.

---

## Mathematical Formulation

**1. TF-IDF Sentence Scoring (Extractive)**

$$\text{Score}(s) = \sum_{w \in s} \text{TF-IDF}(w)$$

*Significance:* A sentence is considered important if it contains many words that are frequent in that document but rare across the whole corpus. This is a crude but effective proxy for "this sentence carries unique, document-specific information."

**2. TextRank / LexRank (Graph-Based Extractive)**

$$WS(V_i) = (1-d) + d \sum_{j} \frac{\text{sim}(V_i, V_j)}{\sum_k \text{sim}(V_j, V_k)} \, WS(V_j)$$

*Significance:* This is PageRank applied to sentences instead of web pages. A sentence is important if it is similar to many other important sentences — importance is *relational*, not just based on word frequency.

**3. Sequence-to-Sequence Generation (Abstractive)**

$$P(y_t \mid y_{<t}, x) = \text{softmax}\big(W \cdot \text{Attention}(\text{decoder}_t, \text{encoder}_{1:n})\big)$$

*Significance:* Each output word is generated one at a time, conditioned on everything generated so far *and* on an attention-weighted view of the full source document. This is what allows the model to "look back" at relevant parts of the input as it writes.

**4. ROUGE-N (Evaluation)**

$$\text{ROUGE-N} = \frac{\sum \text{matching n-grams}}{\sum \text{total n-grams in reference}}$$

*Significance:* It measures word-overlap recall against a human-written reference summary — useful but blind to meaning, paraphrasing, or factual correctness.

---

## How It Works — Step by Step

**Extractive pipeline:**
1. **Segment the document into sentences.** *Analogy: cutting a cake into slices before deciding which slices to serve.*
2. **Convert each sentence into a vector** (TF-IDF or embeddings). *Analogy: assigning each slice a "flavor profile" so we can compare them.*
3. **Score each sentence** using TF-IDF sum or TextRank's graph-based importance. *Analogy: ranking the slices by how much everyone at the table seems to want them.*
4. **Select the top-K sentences** under a length budget, then reorder by original position. *Analogy: plating the best slices in the order they came off the cake.*

**Abstractive pipeline:**
1. **Tokenize the source text** into subword units the model understands.
2. **Encode** the full input into contextual vectors — the model "reads" the whole document at once.
3. **Decode** one token at a time, attending back to the encoded input at each step — like writing a summary while glancing back at your notes.
4. **Apply beam search** to keep multiple candidate sentences alive simultaneously and pick the most probable overall sequence, rather than greedily committing to the first plausible word each time.

---

## Key Assumptions

| Assumption | If Violated |
|---|---|
| Important information is lexically salient (high-frequency, distinctive words) | Sentences with subtle but critical information (e.g., a single negation) get under-ranked — extractive methods miss it entirely. |
| Sentence similarity ≈ semantic importance (TextRank) | In domains with naturally repetitive structure (e.g., legal boilerplate), repeated-but-trivial sentences get inflated scores. |
| The reference summary is the "ground truth" (ROUGE-based evaluation) | Valid paraphrases that don't lexically match the reference get penalized — the metric punishes good summaries that simply use different words. |
| Training data and target domain match | A model fine-tuned on news headlines will summarize scientific papers poorly — register and structure mismatch causes degraded fluency and faithfulness. |
| The full input fits within the model's context window | Information beyond the truncation point is silently dropped, with no warning — a major production risk for long documents. |

---

## When to Use / When Not to Use

| ✅ Use Summarization | ❌ Avoid / Use Caution |
|---|---|
| Condensing long reports, emails, or articles for quick triage | When zero tolerance for factual error exists (e.g., medical, legal) without a verification layer |
| Generating previews/snippets for search or recommendation UIs | When the input exceeds the model's context window without a chunking strategy |
| Summarizing customer support transcripts for agent handoff | When the document has highly technical/symbolic content (math, code) that generation may corrupt |
| Multi-document aggregation (e.g., daily news digest) | When the audience requires verbatim quotes or exact figures preserved |

---

## Implementation Overview

| Aspect | From Scratch (NumPy) | Library (scikit-learn) |
|---|---|---|
| Vectorization | Manually compute term frequency and inverse document frequency matrices | `TfidfVectorizer` handles tokenization, IDF, and normalization in one call |
| Similarity | Manually compute cosine similarity via dot products | `cosine_similarity` from `sklearn.metrics.pairwise` |
| Ranking | Hand-roll the PageRank iteration loop for TextRank | Use `networkx.pagerank` on a similarity graph, or skip straight to sentence-score sums |
| Control | Full visibility into every computation — ideal for learning | Production-ready, optimized, far less code, fewer bugs |

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 'sentences' is a list of sentence strings from the source document
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(sentences)

# Score each sentence by its similarity to the overall document vector
doc_vector = tfidf_matrix.mean(axis=0)
scores = cosine_similarity(tfidf_matrix, doc_vector).flatten()

top_k_indices = scores.argsort()[-3:][::-1]
summary = [sentences[i] for i in sorted(top_k_indices)]
```

---

## Top 5 Interview Questions

1. **High ROUGE but users complain about invented facts — diagnose and fix?**
   - Distinguish lexical overlap (ROUGE) from factual consistency
   - Mention reference-free faithfulness checkers (e.g., QA-based consistency, NLI entailment scoring)
   - Propose mitigation: copy mechanisms, faithfulness-aware fine-tuning

2. **Design a summarizer for legal documents requiring high accuracy.**
   - Favor extractive or hybrid over pure abstractive
   - Justify via risk profile: hallucination cost > fluency benefit in this domain

3. **Explain TextRank and where it fails on multi-speaker transcripts.**
   - Walk through the PageRank-style equation
   - Identify failure: turn-taking and speaker attribution break sentence-similarity assumptions

4. **Evaluate summarization without reference summaries.**
   - LLM-as-judge scoring
   - Factual consistency models (SummaC, QAGS)
   - Redundancy/coverage heuristics

5. **Summarize a 200-page document with a 4k-token context window model.**
   - Chunking + map-reduce summarization
   - Hierarchical summarization (summarize chunks, then summarize the summaries)
   - Retrieval-augmented selection of relevant chunks first

---

## Quick Reference Table

| Item | Detail |
|---|---|
| Algorithm Type | Extractive: unsupervised ranking — Abstractive: supervised sequence-to-sequence generation |
| Time Complexity | Extractive (TF-IDF): O(n·v); TextRank: O(n²·iterations) — Abstractive: O(n²) per decoding step (attention) |
| Space Complexity | Extractive: O(n·v) for the TF-IDF matrix — Abstractive: O(n²) for attention weights per layer |
| Key Hyperparameters | Top-K sentence count, damping factor *d* (TextRank); beam width, length penalty, no-repeat n-gram size (abstractive) |
| Evaluation Metrics | ROUGE-1/2/L, BERTScore, factual consistency scores (SummaC/QAGS), human fluency/faithfulness ratings |

---

## References & Further Reading

1. **Original Paper (Abstractive):** See, A., Liu, P. J., & Manning, C. D. (2017). *Get To The Point: Summarization with Pointer-Generator Networks.* — [arXiv:1704.04368](https://arxiv.org/abs/1704.04368)
2. **PEGASUS Paper:** Zhang, J. et al. (2020). *PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization.* — [arXiv:1912.08777](https://arxiv.org/abs/1912.08777)
3. **Best Tutorial:** Hugging Face Course — [Summarization Chapter](https://huggingface.co/learn/nlp-course/chapter7/5)
4. **Kaggle Notebook:** [News Summarization with TextRank & TF-IDF](https://www.kaggle.com/code) (search: "extractive text summarization TextRank")
5. **TextRank Original Paper:** Mihalcea, R., & Tarau, P. (2004). *TextRank: Bringing Order into Texts.* — [ACL Anthology](https://aclanthology.org/W04-3252/)

---

*Architecture diagram: `text_summarization_pipeline.svg` (included in this folder) — original diagram created for this study guide.*
