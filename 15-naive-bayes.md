# Day 15 — Naive Bayes — Probabilistic Classification

> **In a nutshell:**
> Naive Bayes predicts by calculating the probability of each class given the evidence — assuming all features are independent. Unrealistic assumption. Surprisingly effective in practice — especially for text.

---

## Concept

Naive Bayes is one of the fastest and simplest classification algorithms. Despite being simple — it performs surprisingly well, especially for text classification problems like spam detection.

---

**The core idea:**

Given what you know about something — what is the probability it belongs to each class?

A new email arrives containing the word "free". What is the probability it is spam?

Naive Bayes looks at historical data and asks:
```
How many spam emails contained "free"?       → 80%
How many legitimate emails contained "free"? → 10%
```

"free" is a strong spam signal. Naive Bayes uses this kind of probability to make its decision.

---

## Why Is It Called "Naive"?

Because it makes a very bold assumption — that all features are **completely independent of each other.**

In an email:
- The word "free" appearing does not affect whether "money" appears
- The word "click" appearing does not affect whether "here" appears

In reality this is almost never true. Words absolutely influence each other. But the algorithm assumes they do not — hence "naive."

Despite this unrealistic assumption — Naive Bayes works remarkably well in practice. Especially for text.

---

## How It Works — Bayes Theorem

The formula behind Naive Bayes is Bayes Theorem:

```
P(Spam | Email) = P(Email | Spam) × P(Spam) / P(Email)
```

What this is doing in plain English:

```
Probability this is spam given what the email contains
=
How likely is this email if it IS spam
×
How common is spam overall
/
How common is this type of email overall
```

**Three ingredients:**

| Ingredient | Meaning | Example |
|---|---|---|
| Prior | How common is this class overall? | 30% of all emails are spam |
| Likelihood | If it IS this class, how likely is this evidence? | 80% of spam emails contain "free" |
| Posterior | Combining both — the final probability | The probability this email is spam |

You do not need to memorise the formula. Just know: it combines how common a class is overall with how likely the evidence is given that class.

---

## A Concrete Example

Training data:
```
"free money now"      → Spam
"free gift for you"   → Spam
"win cash prize"      → Spam
"meeting at 3pm"      → Not Spam
"project update"      → Not Spam
"lunch tomorrow"      → Not Spam
```

New email: "free lunch tomorrow"

Naive Bayes calculates:
```
P(Spam | "free lunch tomorrow")     → combines word probabilities given spam
P(Not Spam | "free lunch tomorrow") → combines word probabilities given not spam
```

"free" strongly points to spam. "lunch" and "tomorrow" point to not spam. The algorithm weighs them all — whichever probability is higher wins.

---

## Three Variants

| Variant | Best for |
|---|---|
| Multinomial Naive Bayes | Text classification — word counts |
| Gaussian Naive Bayes | Continuous numerical features |
| Bernoulli Naive Bayes | Binary features — word present or not |

For text — Multinomial is the standard choice.

---

## Laplace Smoothing

If a word never appeared in training — its probability becomes zero. Zero multiplied by anything is still zero — breaking the entire calculation.

**Fix: Laplace Smoothing** — add a tiny count (usually 1) to every word so nothing is ever truly zero.

```
Without smoothing: P("unknown word" | Spam) = 0 → breaks model
With smoothing:    P("unknown word" | Spam) = tiny value → model survives
```

---

## Strengths and Weaknesses

**Strengths:**
- Extremely fast to train and predict
- Works well with small amounts of training data
- Handles high-dimensional data well — great for text
- Easy to update with new data
- Naturally handles multi-class problems

**Weaknesses:**
- The independence assumption is almost always wrong
- Poor at capturing relationships between features
- Not great for numerical features with complex relationships
- Requires Laplace smoothing to handle unseen words

---

## Real World Examples

| Use Case | Why Naive Bayes works well |
|---|---|
| Spam detection | Fast, accurate, handles many words well |
| Sentiment analysis | Classify reviews as positive or negative |
| Document categorisation | News articles into topics |
| Medical diagnosis | Symptoms treated as independent signals |

---

## Code

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Training data — emails and labels
emails = [
    "free money now",
    "free gift for you",
    "win cash prize",
    "meeting at 3pm",
    "project update tomorrow",
    "lunch with the team",
    "click here to claim",
    "quarterly report attached"
]
labels = [1, 1, 1, 0, 0, 0, 1, 0]  # 1 = spam, 0 = not spam

# Convert text to word counts — Naive Bayes needs numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Train
model = MultinomialNB()
model.fit(X, labels)

# Predict on new emails
new_emails = ["free cash for you", "team lunch tomorrow"]
X_new = vectorizer.transform(new_emails)
predictions = model.predict(X_new)

for email, pred in zip(new_emails, predictions):
    print(f"'{email}' → {'Spam' if pred == 1 else 'Not Spam'}")

# Probabilities
probs = model.predict_proba(X_new)
for email, prob in zip(new_emails, probs):
    print(f"'{email}' → Spam probability: {prob[1]:.2f}")
```

---

## Summary

| Concept | Meaning |
|---|---|
| Naive Bayes | Probabilistic classifier based on Bayes Theorem |
| Naive assumption | All features are independent — almost never true but works anyway |
| Prior | How common is this class overall |
| Likelihood | How likely is the evidence given this class |
| Posterior | Final probability — combining prior and likelihood |
| Laplace Smoothing | Adds a tiny count to avoid zero probabilities |
| Multinomial NB | Best variant for text classification |
| Gaussian NB | Best variant for continuous numerical features |

---

*Part of my AI/ML notes series — one concept documented per day.*
*LinkedIn post → [Day 15 — Naive Bayes](https://www.linkedin.com/posts/soumya-dodamani_ai-machinelearning-deeplearning-share-7452259898987376640-GKW_?utm_source=share&utm_medium=member_desktop&rcm=ACoAADIqRfgB9q57kOF3oEPWKkRR61-VN5IONck)*
