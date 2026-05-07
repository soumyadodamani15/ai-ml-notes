# Day 19 — Hyperparameter Tuning — Grid Search vs Random Search

> **In a nutshell:**
> Hyperparameter tuning finds the best model settings before training. Grid Search tries every combination — thorough but slow. Random Search samples randomly — faster and often equally good. Always evaluate with cross-validation and keep a final test set untouched.

---

## Concept

Every ML algorithm has settings you choose before training begins. These are called **hyperparameters.**

Unlike regular parameters — which the model learns from data — hyperparameters are set by you. The model cannot figure them out on its own.

---

## Hyperparameters You Have Already Seen

| Algorithm | Hyperparameters |
|---|---|
| Decision Tree | max_depth, criterion, min_samples_split |
| Random Forest | n_estimators, max_depth, max_features |
| KNN | n_neighbors, metric, weights |
| SVM | C, kernel, gamma |
| K-Means | n_clusters, n_init |
| Gradient Descent | learning_rate, batch_size, epochs |

Choosing the right values can make the difference between a mediocre model and a great one.

---

## Three Approaches to Tuning

---

**1. Manual Search — just guessing**

You try a few values based on intuition or experience.

```
Try max_depth=3 → accuracy 82%
Try max_depth=5 → accuracy 85%
Try max_depth=7 → accuracy 84%
→ Pick max_depth=5
```

Fast but unreliable. Easy to miss the best combination. Biased by your own assumptions.

---

**2. Grid Search — try every combination**

You define a grid of values. Grid Search tries every single combination.

```python
param_grid = {
    'max_depth':    [3, 5, 7, 10],
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2']
}
```

Total combinations: 4 × 3 × 2 = 24.
Each evaluated using cross-validation.

Advantage: thorough — guaranteed to find the best within your grid.
Disadvantage: slow — grows exponentially. 5 parameters × 5 values = 5⁵ = 3125 combinations.

---

**3. Random Search — sample random combinations**

Instead of every combination — randomly sample N combinations from the grid.

Advantage: much faster. Research shows it finds equally good results as Grid Search in a fraction of the time.

Why? Not all hyperparameters matter equally. Random Search naturally explores the important dimensions more efficiently than Grid Search, which wastes time systematically varying unimportant ones.

---

**4. Bayesian Optimisation — the smart approach**

Grid Search and Random Search do not learn from previous results.

Bayesian Optimisation builds a model of which hyperparameter regions are promising — and focuses the search there.

```
Try combination 1 → poor result
Try combination 2 → good result
→ Focus next tries near combination 2
→ Converges to the best region faster
```

Used in tools like Optuna and Hyperopt. Most efficient when each training run is expensive.

---

## Always Use Cross-Validation During Tuning

Every combination should be evaluated using cross-validation — not a single train/test split. Otherwise you risk picking hyperparameters that got lucky on one particular split.

```
GridSearchCV       → Grid Search + cross-validation built in
RandomizedSearchCV → Random Search + cross-validation built in
```

---

## The Risk — Overfitting to the Validation Set

If you keep checking the same validation set and adjusting — you eventually overfit to it. The model looks great on validation but fails on truly new data.

This is why your final evaluation must always be on a completely separate test set — never used during tuning.

---

## Real World Example

Tuning a Random Forest for fraud detection:

```
Grid Search over:
  n_estimators: [50, 100, 200]
  max_depth:    [3, 5, 10, None]
  max_features: ['sqrt', 'log2']

Total combinations: 3 × 4 × 2 = 24
Each evaluated with 5-fold cross-validation
= 24 × 5 = 120 training runs

Best found:
  n_estimators=200, max_depth=10, max_features='sqrt'
  Cross-validation F1: 0.91
```

---

## Code

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.datasets import make_classification

# Generate dataset
X, y = make_classification(
    n_samples=500,
    n_features=10,
    n_classes=2,
    random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid Search — try every combination
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth':    [3, 5, 10],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X, y)

print(f"Grid Search best params: {grid_search.best_params_}")
print(f"Grid Search best score:  {grid_search.best_score_:.2f}")

# Random Search — try random combinations
param_dist = {
    'n_estimators': [50, 100, 150, 200, 300],
    'max_depth':    [3, 5, 7, 10, None],
    'max_features': ['sqrt', 'log2']
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=10,
    cv=cv,
    scoring='f1',
    random_state=42,
    n_jobs=-1
)
random_search.fit(X, y)

print(f"\nRandom Search best params: {random_search.best_params_}")
print(f"Random Search best score:  {random_search.best_score_:.2f}")
```

---

## Summary

| Approach | How it works | Best for |
|---|---|---|
| Manual Search | Guess based on intuition | Quick experiments |
| Grid Search | Try every combination | Small parameter spaces |
| Random Search | Sample random combinations | Larger parameter spaces |
| Bayesian Optimisation | Learn from previous results | Expensive training runs |

| Rule | Why |
|---|---|
| Always use cross-validation | Avoid getting lucky on one split |
| Keep test set separate | Avoid overfitting to validation data |
| Start with Random Search | Faster and often equally good as Grid Search |
| Use Optuna for large spaces | Smarter than both Grid and Random Search |

---

*Part of my AI/ML notes series — one concept documented per day.*
*LinkedIn post → [Day 19 — Hyperparameter Tuning](https://www.linkedin.com/posts/soumya-dodamani_ai-machinelearning-deeplearning-share-7458148489349361664-ykx-?utm_source=share&utm_medium=member_desktop&rcm=ACoAADIqRfgB9q57kOF3oEPWKkRR61-VN5IONck)*
