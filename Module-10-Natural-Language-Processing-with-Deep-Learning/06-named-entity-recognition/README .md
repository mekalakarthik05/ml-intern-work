# Named Entity Recognition (NER)

**Tagline:** Teaching machines to spot names, places, and things hiding inside plain text.

**What you will learn:** You'll understand how NER turns raw sentences into structured entities by treating the problem as sequence labeling rather than simple classification. You'll also learn the math, mechanics, failure modes, and implementation choices behind both classic (CRF) and modern (transformer-based) NER systems.

---

## 2. What is Named Entity Recognition?

Named Entity Recognition is the task of automatically finding specific "things" in text — people, organizations, locations, dates, products — and labeling what kind of thing each one is. Given a sentence like "Tim Cook visited Berlin in March," an NER system should recognize "Tim Cook" as a person, "Berlin" as a location, and "March" as a date. It's one of the foundational building blocks of natural language processing, sitting underneath search engines, chatbots, resume parsers, and medical record systems.

A helpful analogy: imagine a customs officer scanning a stream of luggage on a conveyor belt. The officer doesn't just look at one bag in isolation — they look at context, the order bags appear in, and patterns (a tagged suitcase next to a stroller probably belongs to a family). NER models work the same way. They don't label each word completely independently; they look at neighboring words and the broader sentence to decide where an entity starts, where it ends, and what type it is. A word like "Washington" could be a person, a city, or a state — context is everything.

What makes NER tricky compared to ordinary classification is that entities span multiple words, and the boundaries matter as much as the type. Getting "Bank of America" half right (e.g., just tagging "America" as a location) is a meaningful failure even if the type was sort of correct. This is why NER is treated as a *structured prediction* problem: the labels assigned to the words in a sentence are not independent decisions, they are a connected sequence that must make sense together.

---

## 3. Mathematical Formulation

**Linear-Chain CRF (Conditional Random Field) — the core classic equation:**

$$
P(y \mid x) = \frac{1}{Z(x)} \exp\left( \sum_{t} \sum_{k} \lambda_k f_k(y_{t-1}, y_t, x, t) \right)
$$

*Significance:* This says the probability of an entire label sequence `y` given the sentence `x` is computed jointly, not word-by-word. The model considers transitions between adjacent labels (`y_{t-1} → y_t`) alongside features from the input. `Z(x)` is a normalizing constant that ensures probabilities across **all possible label sequences** sum to 1 — this global view is what prevents nonsensical sequences like an "Inside-Person" tag following an "Outside" tag with no "Beginning" tag first.

**Viterbi Decoding (finding the best label sequence):**

$$
\delta_t(j) = \max_i \big[ \delta_{t-1}(i) + \text{score}(i \to j) \big] + \text{emission}(j, x_t)
$$

*Significance:* This is the equation that actually picks the winning sequence at prediction time. Instead of checking every possible label sequence (which explodes combinatorially), it builds up the best path one token at a time, carrying forward only the best score for each possible "current state" — a classic dynamic programming trick.

**Entity-Level F1 (the metric, not the model):**

$$
F1 = \frac{2 \cdot P \cdot R}{P + R}, \quad P = \frac{TP}{TP+FP}, \quad R = \frac{TP}{TP+FN}
$$

*Significance:* A predicted entity only counts as a "true positive" if **both** its span boundaries and its type match the gold label exactly. This is stricter than token accuracy and is the standard way NER systems are actually graded in practice.

---

## 4. How It Works — Step by Step

1. **Tokenize the sentence.** "Tim Cook visited Berlin" → `[Tim, Cook, visited, Berlin]`. Think of this as cutting a sentence into its individual luggage items on the conveyor belt.
2. **Assign a tag scheme to each token.** Using BIO tagging: `Tim=B-PER, Cook=I-PER, visited=O, Berlin=B-LOC`. "B" means beginning of an entity, "I" means inside, "O" means outside any entity — like stamping each bag with "start of family group," "part of family group," or "unrelated bag."
3. **Extract features or embeddings per token.** Classic systems used hand-crafted features (capitalization, prefixes); modern systems use contextual embeddings from a transformer (BERT) that already "know" Berlin is usually a city based on the surrounding words.
4. **Score each possible label given the input.** The model (CRF emissions or a neural classifier head) computes a score for each candidate tag at each position — like the customs officer rating how confident they are about each item.
5. **Score transitions between adjacent labels.** The CRF also scores how plausible it is to go from one tag to the next (e.g., `B-PER → I-PER` is plausible, `B-PER → I-LOC` is not).
6. **Decode the best overall sequence (Viterbi).** Rather than picking the best tag per word independently, the algorithm finds the single best-scoring full sequence across the entire sentence.
7. **Reconstruct entity spans from tags.** Consecutive `B-PER, I-PER` tags get merged back into the single entity "Tim Cook."
8. **Evaluate using entity-level F1.** Compare predicted spans+types against the gold-labeled spans+types exactly.

---

## 5. Key Assumptions

| Assumption | If Violated |
|---|---|
| Entities are flat (non-overlapping, non-nested) | Nested entities (e.g., "Bank of America" inside a larger org name) get mislabeled or merged incorrectly |
| Labels form a coherent first-order Markov chain (CRF) | Long-range dependencies beyond adjacent tags get missed, causing inconsistent tagging across a long sentence |
| Tokenization aligns cleanly with entity boundaries | Subword tokenizers (BPE/WordPiece) splitting a word mid-entity causes incorrect label propagation |
| Training and inference data come from similar domains | Severe performance drop under domain shift (e.g., news-trained model on legal/clinical text) |
| Sufficient labeled data exists for each entity type | Rare entity types get systematically under-predicted (class imbalance) |

---

## 6. When to Use / When Not to Use

| ✅ Use NER When... | ❌ Avoid / Reconsider When... |
|---|---|
| You need structured fields extracted from free text (invoices, resumes, support tickets) | The "entities" overlap heavily or are nested — consider span-based or seq2seq models instead |
| You have labeled training data or can use weak supervision/gazetteers | You have zero labeled data and no time to annotate — consider few-shot LLM prompting first |
| Entity types are well-defined and relatively stable | Entity types are fuzzy, subjective, or constantly evolving (hard to define a tagging scheme) |
| Downstream tasks (relation extraction, KG building) depend on accurate spans | You only need rough keyword presence, not exact boundaries — simpler keyword matching may suffice |
| Domain has consistent formatting/style (e.g., financial filings) | Text is highly informal/noisy (social media) without domain-adapted training data |

---

## 7. Implementation Overview

| Aspect | From Scratch (NumPy) | Library (sklearn-crfsuite / spaCy) |
|---|---|---|
| Feature extraction | Manually engineer capitalization, prefix/suffix, POS features | Built-in feature templates or pretrained embeddings handled internally |
| Model training | Implement forward-backward algorithm and gradient updates manually | `.fit(X_train, y_train)` handles optimization internally |
| Decoding | Implement Viterbi recursion by hand | `.predict()` calls Viterbi internally |
| Evaluation | Manually compute span-level TP/FP/FN | Use `seqeval` or built-in classification reports |
| Best for | Deep understanding, exam/interview prep | Production speed, reliability, less boilerplate |

```python
import sklearn_crfsuite
from sklearn_crfsuite import metrics

# X_train: list of sentences, each a list of per-token feature dicts
# y_train: list of sentences, each a list of BIO tags (e.g., "B-PER", "O")

crf = sklearn_crfsuite.CRF(
    algorithm="lbfgs",
    c1=0.1,        # L1 regularization strength
    c2=0.1,        # L2 regularization strength
    max_iterations=100,
    all_possible_transitions=True
)

crf.fit(X_train, y_train)
y_pred = crf.predict(X_test)

print(metrics.flat_f1_score(y_test, y_pred, average="weighted"))
```

---

## 8. Top 5 Interview Questions

1. **Why use a CRF/BiLSTM-CRF instead of plain per-token softmax classification?**
   - Mention label dependencies, global normalization, label bias problem in MEMMs.
2. **Your model has 95% token accuracy but poor real-world NER quality — why?**
   - Talk about "O" tag dominance, token vs entity-level F1, exact-match strictness.
3. **How do you handle subword tokenization when fine-tuning BERT for NER?**
   - Discuss label propagation to subtokens, masking non-first-subtoken loss.
4. **How would you handle nested/overlapping entities?**
   - Bring up span-based scoring, seq2seq extraction, layered tagging schemes.
5. **You have very little labeled data in a new domain — what's your approach?**
   - Cover weak supervision/gazetteers, few-shot LLM prompting, active learning, transfer learning from a pretrained model.

---

## 9. Quick Reference Table

| Item | Detail |
|---|---|
| Algorithm Type | Structured sequence labeling (discriminative, supervised) |
| Time Complexity | O(T · S²) for CRF training/decoding (T = sequence length, S = number of label states) |
| Space Complexity | O(T · S) for Viterbi trellis storage |
| Key Hyperparameters | Regularization (c1/c2 for CRF), learning rate & epochs (neural), tagging scheme (BIO/BIOES) |
| Evaluation Metrics | Entity-level Precision, Recall, F1 (exact span + type match) |

---

## 10. References & Further Reading

- Lafferty, McCallum, Pereira (2001) — *Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data* (the original CRF paper)
- HuggingFace Token Classification Tutorial — practical guide to fine-tuning transformers for NER
- spaCy official documentation on Named Entity Recognition pipelines
- Kaggle: "Named Entity Recognition (NER) using LSTMs" notebook — hands-on BiLSTM-CRF walkthrough
- `seqeval` library documentation — for correct entity-level evaluation in Python
