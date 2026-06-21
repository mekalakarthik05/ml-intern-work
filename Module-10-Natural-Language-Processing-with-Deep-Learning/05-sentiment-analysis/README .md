# Sentiment Analysis

> *Teaching machines to read between the lines of human opinion.*

**What you will learn:** This module takes you from the core intuition of sentiment analysis through its mathematical backbone, classical and neural implementations, and the failure modes that separate a toy model from a production-grade system. By the end, you'll be able to explain, implement, and defend a sentiment classification pipeline at an interview-ready level.

---

## Learning Objectives

| Level | Objective |
|---|---|
| **Explain** | What sentiment analysis is, why it's non-trivial (negation, sarcasm, domain shift), and the tradeoff between lexicon-based vs. ML-based vs. deep learning approaches |
| **Derive (conceptually)** | TF-IDF weighting intuition, Naive Bayes via Bayes' rule, logistic regression via sigmoid + cross-entropy, attention as a soft dictionary lookup |
| **Implement** | A full pipeline from raw text -> cleaned tokens -> vectorized features -> classifier -> evaluation, both from scratch (NumPy) and via sklearn/HuggingFace |

By the end, you must be able to defend design choices (why logistic regression over NB? why n-grams over unigrams? why fine-tune BERT vs. train from scratch?) in an interview setting.

---

## What is Sentiment Analysis?

Sentiment analysis is the task of automatically determining the emotional tone behind a piece of text -- is the writer happy, angry, disappointed, or neutral? It sits at the intersection of Natural Language Processing (NLP, the field of teaching computers to understand human language) and standard machine learning classification, because once we've converted words into numbers, sentiment analysis becomes "just" a classification problem: positive, negative, or neutral.

Think of it like a restaurant critic reading thousands of customer comment cards. A human critic doesn't just look for the word "good" -- they notice "not good," sarcasm like "oh, fantastic, another cold meal," and tone shifts mid-sentence ("the food was great but the wait was unbearable"). A machine has to learn these same instincts purely from patterns in labeled examples, without ever truly "feeling" anything. This is precisely why sentiment analysis is deceptively hard: the *words* are easy to count, but the *meaning* depends on word order, context, negation, and sometimes cultural or domain-specific knowledge.

There isn't one single way to do this. Some systems use sentiment dictionaries (a word like "terrible" has a fixed negative score) -- fast but rigid. Others use trained classifiers that learn statistical patterns from labeled data. The most powerful modern systems use deep contextual models that understand a word's meaning *in relation to* the words around it. Sentiment analysis is one of the best topics to learn ML on, because it forces you to confront the gap between "the model technically works" and "the model actually understands."

---

## Subtopic Map

### Core Concepts & Intuition
- **What sentiment analysis is and why it's hard** -- human language carries meaning beyond word presence; negation, sarcasm, tone, and domain context all break naive approaches.
- **Bag-of-Words vs. sequence-aware models** -- the fundamental choice between ignoring word order (fast, cheap) and modeling it (expensive, more accurate).
- **Lexicon-based vs. learned approaches** -- using fixed dictionaries (VADER, TextBlob) vs. training a model on labeled data.

### Mathematical Formulation
| Equation | Significance |
|---|---|
| **Naive Bayes:** P(class \| text) ∝ P(class) · ∏ P(wᵢ \| class) | Tells us the most probable sentiment class by multiplying how common each word is within that class -- simple, fast, surprisingly strong baseline. |
| **TF-IDF:** TF-IDF(t, d) = TF(t, d) × log(N / DF(t)) | Tells us how *important* a word is to a specific document relative to the whole corpus -- common words like "the" get down-weighted automatically. |
| **Logistic Regression:** P(positive \| x) = σ(wᵀx + b) | Converts a weighted combination of word features into a clean probability between 0 and 1 -- the workhorse of classical sentiment classifiers. |
| **Cross-Entropy Loss:** L = −[y·log(ŷ) + (1−y)·log(1−ŷ)] | Tells the model exactly how "wrong" its probability prediction was -- large penalty for being confidently wrong. |
| **RNN hidden state:** hₜ = tanh(Wₓₓₜ + Wₕₕhₜ₋₁ + b) | Tells us how the model "remembers" prior words while reading a sentence left to right -- this is how negation and context get captured. |
| **Attention:** Attention(Q,K,V) = softmax(QKᵀ / √dₖ) · V | Tells us which words in a sentence the model should "pay attention to" when forming its sentiment judgment -- the core mechanic behind transformers like BERT. |

### Algorithm Variants & Extensions
- **Naive Bayes (Multinomial / Bernoulli)** -- baseline classifiers differing in how they model word presence vs. counts.
- **Logistic Regression with TF-IDF** -- the classical industry workhorse; simple, interpretable, strong on well-structured data.
- **LSTM / GRU for Sentiment** -- first step into sequence modeling; captures long-range negation ("not... bad at all").
- **Transformer-based (BERT, RoBERTa, DistilBERT)** -- state-of-the-art; bidirectional context + fine-tuning yields best accuracy but highest cost.
- **Aspect-Based Sentiment Analysis (ABSA)** -- split into (1) aspect term extraction (NER-style), (2) sentiment classification per aspect.
- **Zero / Few-Shot Sentiment** -- using LLM prompts or cross-task transfer when labeled data is scarce.

### Common Failure Modes, Edge Cases & Misconceptions

| Failure Mode | Why It Happens | How to Mitigate |
|---|---|---|
| **Accuracy paradox** | 95% acc on 90% positive data = useless | Always check class-wise precision/recall/F1 |
| **Negation blind spot** | BoW treats "not bad" as combination of positive + negative | Use n-grams (2/3-grams) or sequence models |
| **Sarcasm misclassification** | Literal words positive, intended meaning negative | Requires context, tone cues, or large pretrained models |
| **Domain shift** | Model trained on reviews fails on tweets | Domain adaptation, fine-tune on target domain |
| **Neutral class collapse** | Model rarely predicts minority class | Class weighting, oversampling, or threshold tuning |
| **Label noise** | Sentiment annotation is subjective | Use majority vote across multiple annotators, drop low-agreement samples |

---

## How It Works -- Step by Step

1. **Collect & label data.** Gather text with known sentiment labels (e.g., movie reviews with star ratings). *Analogy: training a new intern by showing them thousands of pre-graded comment cards.*
2. **Clean the text.** Lowercase, strip punctuation, remove stopwords ("the," "is," "and"). *Analogy: erasing background noise before listening to the actual message.*
3. **Convert text to numbers.** Use Bag-of-Words/TF-IDF (classical) or embeddings (neural) to turn words into vectors. *Analogy: translating a sentence into a numerical fingerprint a calculator can work with.*
4. **Feed it to a model.** A classifier (Naive Bayes, Logistic Regression) or a neural network (LSTM, Transformer) learns the relationship between the numeric representation and the sentiment label. *Analogy: the intern starts noticing patterns -- "words like 'terrible' usually mean a 1-star review."*
5. **Predict.** Given new, unseen text, the model outputs a probability for each sentiment class. *Analogy: the now-trained intern reads a brand-new comment card and confidently grades it.*
6. **Evaluate & iterate.** Check precision/recall/F1 per class, find where the model fails (sarcasm, negation), and refine. *Analogy: a manager reviewing the intern's mistakes and giving targeted feedback.*

---

## Key Assumptions

- **Independence of words (Naive Bayes specifically):** Assumes each word contributes to sentiment independently of others. *If violated:* the model misses phrase-level meaning (e.g., "not bad" ≠ "not" + "bad" separately).
- **Labeled data reflects true sentiment:** Assumes human-annotated labels are consistent and correct. *If violated:* the model learns noisy or biased patterns, since sentiment labeling itself is subjective.
- **Training and deployment domains match:** Assumes vocabulary/tone in production resembles the training data. *If violated:* expect significant accuracy drops (a model trained on movie reviews struggles on tweets).
- **Bag-of-words/TF-IDF: order doesn't matter:** Assumes word presence/frequency is enough, ignoring sequence. *If violated (it usually is):* negation and sarcasm get misclassified.
- **Fixed, balanced classes:** Many models implicitly assume reasonably balanced class distribution. *If violated:* accuracy becomes misleading, and minority-class (often "neutral") recall suffers badly.

---

## When to Use / When Not to Use

| ✅ Use Sentiment Analysis When | ❌ Avoid / Be Cautious When |
|---|---|
| You need scalable, automated tone tracking (reviews, social media, support tickets) | The text is highly sarcastic or culturally nuanced without domain-specific training data |
| You have (or can get) reasonably labeled training data | You need precise *reasoning* about *why* a customer feels a certain way (sentiment alone won't explain root cause) |
| Real-time or near-real-time monitoring is valuable (brand monitoring, alerting) | The cost of misclassification is very high and unverified (e.g., automated moderation decisions without human review) |
| Aspect-level feedback is acceptable as a future enhancement, not a day-1 requirement | You need fine-grained aspect-based insight immediately -- that requires a more complex ABSA setup, not vanilla sentiment classification |
| You can tolerate occasional errors and have a feedback loop to retrain | Your domain language shifts rapidly (slang-heavy, fast-evolving platforms) without a retraining pipeline in place |

---

## Implementation Overview

### Approach Comparison

| Aspect | From Scratch (NumPy) | Library (scikit-learn) | Neural (HuggingFace) |
|---|---|---|---|
| Text vectorization | Manually build vocabulary, compute term/document frequencies | `TfidfVectorizer` handles this in one line | `AutoTokenizer.from_pretrained(...)` |
| Model training | Hand-code gradient descent for logistic regression weights | `LogisticRegression().fit()` | `Trainer` API with `.train()` |
| Probability output | Manually apply sigmoid/softmax to raw scores | Built-in `.predict_proba()` | `.predict()` with softmax logits |
| Speed of development | Slow -- but builds deep intuition | Fast -- production-ready in minutes | Moderate -- configuration overhead |
| Best use case | Learning/teaching, debugging model internals | Real projects, prototyping, baselines | SOTA accuracy, research, complex text |

### Quick Start (sklearn Baseline)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_vec, y_train)

accuracy = model.score(X_test_vec, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

---

## Top 5 Interview Questions

1. **Your classifier hits 95% accuracy on a 90%-positive dataset -- is it good?**
   - Hint: question accuracy as a metric -> discuss precision/recall/F1 per class -> mention confusion matrix.

2. **Design a sentiment analysis pipeline for product reviews end-to-end.**
   - Hint: walk through ingestion -> preprocessing -> representation choice -> model choice tradeoffs -> monitoring/retraining.

3. **Why does "not good" break bag-of-words models, and how would you fix it?**
   - Hint: explain independence assumption -> propose n-grams (classical fix) -> propose sequence models/RNNs (neural fix).

4. **How would you handle sarcasm with limited labeled sarcasm data?**
   - Hint: mention transfer learning, weak supervision, multi-task learning, or synthetic data augmentation.

5. **How is aspect-based sentiment analysis different from standard sentiment classification, and how would you architect it?**
   - Hint: distinguish extraction (what aspect?) from classification (what sentiment?) -> mention joint sequence-labeling approaches.

---

## Prerequisite Knowledge

| Topic | Why It's Needed |
|---|---|
| **Python proficiency** | All implementations and interviews use Python |
| **Basic probability & Bayes' rule** | Naive Bayes derivation, class priors, conditional independence |
| **Linear algebra (vectors, dot products, matrices)** | TF-IDF vectors, word embeddings, attention scores |
| **Binary classification fundamentals** | Confusion matrix, precision, recall, F1, ROC-AUC |
| **Gradient descent intuition** | Understanding how logistic regression or neural models learn weights |
| **Basic NLP concepts** | Tokenization, stopwords, stemming/lemmatization (can be co-learned but helpful) |

**Nice-to-have:** Familiarity with RNNs / LSTMs (covered in the module) and Transformer architecture basics (covered in the module).

---

## Connections to Other ML Topics

| Related Topic | How It Connects |
|---|---|
| **Text Classification (general)** | Sentiment is a special case of text classification -- same tools (TF-IDF + linear model), same evaluation framework |
| **Word Embeddings (Word2Vec, GloVe)** | The bridge from sparse BoW to dense semantic representations; sentiment benefits from pre-trained semantic spaces |
| **Sequence Models (RNN, LSTM, GRU)** | Sentiment is often the introductory application for sequence models due to intuitive interpretability (negation, phrase-level meaning) |
| **Transformers & Self-Attention** | BERT fine-tuned for sentiment is the canonical "off-the-shelf" SOTA; attention weights provide interpretability (which words drove the prediction) |
| **Transfer Learning & Fine-Tuning** | Sentiment is the classic case study: take a pretrained LM, replace the head, fine-tune on domain -- minimal data, strong results |
| **Data Imbalance** | Sentiment datasets are often skewed (neutral underrepresented) -- teaches resampling, cost-sensitive learning, threshold tuning |
| **Model Interpretability (LIME, SHAP)** | Sentiment is the most demo-friendly domain for explaining predictions ("this word pushed sentiment toward negative by X points") |
| **Multi-Task Learning** | Jointly learning sentiment + emotion + sarcasm from shared representations |
| **Active Learning** | Sentiment annotation is cheap and high-volume -- an ideal playground for uncertainty sampling and annotation efficiency strategies |

---

## Quick Reference Table

| Item | Detail |
|---|---|
| Algorithm Type | Supervised classification (binary/multi-class); classical or neural |
| Time Complexity | Training: O(n·d) for classical linear models; O(n·d·T) for sequence models (T = sequence length) |
| Space Complexity | O(d) for classical model weights; O(d·h) for RNN/embedding-based models (h = hidden size) |
| Key Hyperparameters | Vectorizer max_features/ngram_range, regularization strength (C), learning rate, hidden size, number of layers |
| Evaluation Metrics | Accuracy, Precision, Recall, F1-score (macro/weighted), Confusion Matrix, ROC-AUC |

---

## References & Further Reading

- Pang & Lee, *"Opinion Mining and Sentiment Analysis"* -- foundational survey paper on the field.
- [Stanford NLP Course Materials on Sentiment Analysis](https://web.stanford.edu/class/cs224n/) -- strong conceptual and neural NLP grounding.
- [Hugging Face Sentiment Analysis Tutorial](https://huggingface.co/docs/transformers/tasks/sequence_classification) -- practical transformer fine-tuning walkthrough.
- [Kaggle: Sentiment Analysis on Movie Reviews](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews) -- hands-on dataset and community notebooks.
- [scikit-learn Text Feature Extraction Documentation](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) -- official reference for TF-IDF/BoW implementation details.
