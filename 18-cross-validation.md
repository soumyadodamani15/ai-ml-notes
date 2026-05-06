# Day 18 — Cross-Validation — Making Evaluation Reliable

> **In a nutshell:**
> Cross-validation evaluates your model on multiple different splits of the data and averages the results. It gives a more reliable picture of true performance than a single train/test split — and tells you how stable your model is across different data.

---

## Concept

You already know from Day 07 that you split data into training, validation and test sets. Cross-validation takes this idea further — making your model evaluation more reliable, especially when data is limited.

---

**The problem with a single train/test split:**

You split your data 80/20. You get 85% accuracy. But what if that 20% happened to be unusually easy examples? Or unusually hard ones?

A single split gives you one number. That number might not represent how your model truly performs.

**Cross-validation gives you many numbers — and averages them.**

---

## K-Fold Cross-Validation

Split your data into K equal parts called folds.

**Example with K=5 and 100 examples:**

```
Fold 1: examples  1– 20
Fold 2: examples 21– 40
Fold 3: examples 41– 60
Fold 4: examples 61– 80
Fold 5: examples 81–100
```

**Round 1:**
```
Training → Fold 2 + Fold 3 + Fold 4 + Fold 5  (80 examples)
Test     → Fold 1                               (20 examples)
Result   → 83%
```

**Round 2:**
```
Training → Fold 1 + Fold 3 + Fold 4 + Fold 5  (80 examples)
Test     → Fold 2                               (20 examples)
Result   → 87%
```

**All 5 rounds:**
```
Round 1 → test on fold 1 → 83%
Round 2 → test on fold 2 → 87%
Round 3 → test on fold 3 → 85%
Round 4 → test on fold 4 → 82%
Round 5 → test on fold 5 → 88%
```

Every example gets to be in the test set exactly once. No data is wasted.

---

## Calculating the Result — Average ± Standard Deviation

**Step 1 — Average:**
```
(83 + 87 + 85 + 82 + 88) / 5 = 85%
```

**Step 2 — Deviation from average:**
```
83 - 85 = -2
87 - 85 = +2
85 - 85 =  0
82 - 85 = -3
88 - 85 = +3
```

**Step 3 — Square each deviation:**
```
(-2)² =  4
(+2)² =  4
( 0)² =  0
(-3)² =  9
(+3)² =  9
```
Squaring removes negatives so they do not cancel each other out.

**Step 4 — Average the squared deviations (variance):**
```
(4 + 4 + 0 + 9 + 9) / 5 = 5.2
```

**Step 5 — Square root (standard deviation):**
```
√5.2 ≈ 2.2
```
Square root undoes the squaring from Step 3 — brings the result back to percentage units.

**Final result: 85% ± 2.2%**

Most of the time the model scores between 82.8% and 87.2%.

---

## Why the ± Matters

```
Model A: 85% ± 1%   → very stable — consistent across different data slices
Model B: 85% ± 10%  → very unstable — getting lucky on some folds, failing on others
```

Both models have the same average accuracy. But Model A is far more trustworthy.

```
Low std  → reliable  → deploy with confidence
High std → unreliable → investigate before deploying
```

---

## Common Variants

**Stratified K-Fold**
Ensures each fold has the same proportion of each class as the full dataset.
Critical when classes are imbalanced.
e.g. 90% not fraud, 10% fraud — a random split might put all fraud in one fold.
Always use Stratified K-Fold for classification problems.

**Leave-One-Out Cross-Validation (LOOCV)**
K = number of examples. Each round uses one example as the test set.
Very thorough but extremely slow. Use only for very small datasets.

**Time Series Cross-Validation**
Never shuffle time series data — future cannot train a model that predicts the past.
Always train on past, test on future:
```
Train: months 1–6 → Test: month 7
Train: months 1–7 → Test: month 8
Train: months 1–8 → Test: month 9
```

---

## When to Use Which

| Situation | Use |
|---|---|
| Large dataset, fast model | Simple train/test split |
| Small dataset | K-Fold cross-validation |
| Imbalanced classes | Stratified K-Fold |
| Time series data | Time series cross-validation |
| Final model evaluation | Always use a separate held-out test set |

**Important rule:** cross-validation is for model selection and tuning. Your final evaluation must always be on a separate held-out test set — never touched during cross-validation.

---

## Real World Example

Building a fraud detection model with 1000 examples — too small for a reliable single split.

With 5-Fold cross-validation:
- Each fold has 200 examples
- Model trained and tested 5 times
- Average score — reliable estimate of true performance
- Standard deviation — is the model consistent?
- Compare different algorithms using their cross-validation scores
- Pick the best — train on all 1000 examples — evaluate on final test set

---

## Code

```python
import numpy as np
from sklearn.model_selection import (
    cross_val_score,
    KFold,
    StratifiedKFold,
    cross_validate
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate a sample dataset
X, y = make_classification(
    n_samples=200,
    n_features=10,
    n_classes=2,
    random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)

# Basic K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

print(f"Scores per fold: {scores}")
print(f"Mean accuracy:   {scores.mean():.2f}")
print(f"Std deviation:   {scores.std():.2f}")

# Stratified K-Fold — better for classification
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_strat = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

print(f"\nStratified scores: {scores_strat}")
print(f"Mean accuracy:     {scores_strat.mean():.2f}")

# Multiple metrics at once
results = cross_validate(
    model, X, y, cv=skf,
    scoring=['accuracy', 'precision', 'recall', 'f1']
)

print(f"\nAccuracy:  {results['test_accuracy'].mean():.2f}")
print(f"Precision: {results['test_precision'].mean():.2f}")
print(f"Recall:    {results['test_recall'].mean():.2f}")
print(f"F1 Score:  {results['test_f1'].mean():.2f}")
```

---

## Connection to Earlier Concepts

Standard deviation uses the same core idea as MSE from Day 09:

```
MSE = mean((predicted - actual)²)          → measures prediction error
Std = √(mean((score - average score)²))    → measures score consistency
```

Same pattern — measure distances, square them, average them. Standard deviation adds a square root to return to original units.

---

## Summary

| Concept | Meaning |
|---|---|
| K-Fold Cross-Validation | Split into K folds, rotate test fold, average results |
| Standard Deviation (±) | How much scores varied across folds |
| Low ± | Model is stable and consistent |
| High ± | Model is unstable — investigate before deploying |
| Stratified K-Fold | Keeps class proportions equal across folds |
| LOOCV | One example per test fold — thorough but slow |
| Time Series CV | Always train on past, test on future |

---

*Part of my AI/ML notes series — one concept documented per day.*
*LinkedIn post → [Day 18 — Cross-Validation](https://www.linkedin.com/posts/soumya-dodamani_ai-machinelearning-deeplearning-share-7457712768557760512-EXqm?utm_source=share&utm_medium=member_desktop&rcm=ACoAADIqRfgB9q57kOF3oEPWKkRR61-VN5IONck)*
