# Day 07 — Train, Validation and Test Split — Why it Matters

> **One line to remember:**
> Training teaches. Validation guides. Test judges — but only once.

---

## Concept

You already know from Day 03 that you split data into training and test sets. Today we go one level deeper — and add a third split: **validation.**

---

**Why is a simple train/test split sometimes not enough?**

Imagine you are building a model. You train it, check it on the test set, tweak it, train again, check the test set again — and repeat until it performs well on the test set.

The problem: you have now indirectly used the test set to make decisions about your model. It is no longer a truly unseen dataset. Your model may perform well on that specific test set but fail in the real world.

This is called **data leakage** — one of the most common mistakes in ML.

---

## The Three Splits

```
Full Dataset
├── Training Set    (~70%) — model learns from this
├── Validation Set  (~15%) — used during development to tune the model
└── Test Set        (~15%) — touched only once at the very end
```

---

**Training Set**

The model sees this data and learns from it. Weights are updated based on this data only.

**Validation Set**

Used during training to check how the model is doing — without touching the test set. You use this to:
- Compare different models
- Tune hyperparameters
- Decide when to stop training

Think of it as a practice exam. You use it to prepare — but it is not the final exam.

**Test Set**

Touched exactly once — at the very end, after all decisions are made. This gives you an honest, unbiased measure of how your model will perform on real world data.

Think of it as the final exam. You sit it once. No retakes, no peeking beforehand.

---

## The Most Common Mistake

Many beginners use the test set multiple times during development — tweaking the model each time based on test results. This makes the test set effectively part of the training process. Reported accuracy looks great but the model fails in production.

**Rule of thumb:** if you have touched the test set more than once — it is no longer a test set.

---

## K-Fold Cross Validation

When your dataset is small, a single validation split wastes data. K-Fold Cross Validation solves this:

- Split data into K equal parts (folds)
- Train on K-1 folds, validate on the remaining fold
- Repeat K times — each fold gets a turn as the validation set
- Average the results

```
K=5 example:
Fold 1: [VAL] [TRN] [TRN] [TRN] [TRN]
Fold 2: [TRN] [VAL] [TRN] [TRN] [TRN]
Fold 3: [TRN] [TRN] [VAL] [TRN] [TRN]
Fold 4: [TRN] [TRN] [TRN] [VAL] [TRN]
Fold 5: [TRN] [TRN] [TRN] [TRN] [VAL]
```

Every example gets to be in the validation set exactly once. More reliable than a single split.

---

## Real World Example

You are building a fraud detection model for a bank.

| Split | Data | Purpose |
|---|---|---|
| Training set | Transactions Jan–Oct | Model learns what fraud looks like |
| Validation set | Transactions Nov | Tune the model, adjust thresholds, compare approaches |
| Test set | Transactions Dec | Run once at the end — report final performance to stakeholders |

If November transactions influenced your decisions and you kept tweaking — December is your honest reality check.

---

## Code

```python
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

# Sample dataset
X = np.array([[i] for i in range(100)])
y = np.array([0 if i < 50 else 1 for i in range(100)])

# Three-way split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.18, random_state=42)

print(f"Train size:      {len(X_train)}")  # ~70%
print(f"Validation size: {len(X_val)}")    # ~15%
print(f"Test size:       {len(X_test)}")   # ~15%

# K-Fold Cross Validation
model = LogisticRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf)

print(f"\nK-Fold scores:   {scores}")
print(f"Average score:   {scores.mean():.2f}")
```

---

## Summary

| Split | Size | Purpose | How many times used |
|---|---|---|---|
| Training | ~70% | Model learns | Many times |
| Validation | ~15% | Tune and compare | As many as needed |
| Test | ~15% | Final honest evaluation | Exactly once |

---

*Part of my AI/ML notes series — one concept documented per day.*
*LinkedIn post → [Day 07 — Train, Validation and Test Split](https://www.linkedin.com/posts/soumya-dodamani_dotnet-ai-rag-share-7447949947146158081-T9yR?utm_source=share&utm_medium=member_desktop&rcm=ACoAADIqRfgB9q57kOF3oEPWKkRR61-VN5IONck)*
