# Day 06 — Overfitting vs Underfitting — Bias-Variance Tradeoff

> **One line to remember:**
> Underfitting is too simple to learn. Overfitting is too specific to generalise. The goal is a model that learns the pattern — not the noise.

---

## Concept

Every ML model you build will face this problem. You want your model to learn the **general pattern** in the data — not memorise the training data, and not be too simple to learn anything useful.

There are three scenarios:

---

**Underfitting — the model is too simple**

The model has not learned enough. It performs poorly on both training data and new data. It missed the pattern entirely.

Imagine drawing a straight horizontal line through data that clearly curves upward. The line is too simple — it does not capture what is actually happening.

- High error on training data
- High error on test data
- Cause: model is too simple, not enough training, too few features

---

**Overfitting — the model is too complex**

The model has learned the training data too well — including the noise and random quirks that are not part of the real pattern. It performs great on training data but fails on new data.

Imagine a line that wiggles through every single training point perfectly — but when you give it a new point it goes wildly wrong because it memorised noise instead of learning the pattern.

- Very low error on training data
- High error on test data
- Cause: model is too complex, too many features, too little data, trained for too long

---

**Just right — good generalisation**

The model captures the real pattern without memorising noise. It performs well on both training data and new unseen data. This is always the goal.

---

## The Bias-Variance Tradeoff

| | Meaning | Causes |
|---|---|---|
| Bias | Error from wrong assumptions — model too simple | Underfitting |
| Variance | Error from being too sensitive to training data | Overfitting |

```
High Bias   + Low Variance  =  Underfitting
Low Bias    + High Variance =  Overfitting
Low Bias    + Low Variance  =  Just right ← this is the goal
```

The tradeoff: making a model more complex reduces bias but increases variance. Making it simpler reduces variance but increases bias. You are always balancing the two.

---

## How to Fix

**Overfitting:**
- Get more training data
- Simplify the model
- Use regularisation (L1/L2 — covered later)
- Use dropout (for neural networks — covered later)

**Underfitting:**
- Use a more complex model
- Add more relevant features
- Train for longer

---

## Real World Example

You build a model to predict house prices.

| Scenario | What happened |
|---|---|
| Underfitting | Only used number of rooms — too simple, missed location, size, condition |
| Overfitting | Used 500 features including seller name and day of week — memorised training quirks, failed on new listings |
| Just right | Used size, location, rooms, age of building — enough signal, no noise |

---

## Code

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Simple dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5, 7, 8, 9])

# Underfitting — degree 1 (straight line, too simple)
poly1 = PolynomialFeatures(degree=1)
X1 = poly1.fit_transform(X)
model1 = LinearRegression().fit(X1, y)
print(f"Underfit MSE: {mean_squared_error(y, model1.predict(X1)):.2f}")

# Just right — degree 2
poly2 = PolynomialFeatures(degree=2)
X2 = poly2.fit_transform(X)
model2 = LinearRegression().fit(X2, y)
print(f"Good fit MSE: {mean_squared_error(y, model2.predict(X2)):.2f}")

# Overfitting — degree 7 (too complex, memorises every point)
poly7 = PolynomialFeatures(degree=7)
X7 = poly7.fit_transform(X)
model7 = LinearRegression().fit(X7, y)
print(f"Overfit MSE:  {mean_squared_error(y, model7.predict(X7)):.2f}")
```

---

## Summary

| Concept | Meaning | Fix |
|---|---|---|
| Underfitting | Too simple — misses the pattern | More complexity, more features |
| Overfitting | Too complex — memorises noise | More data, simpler model, regularisation |
| Bias | Error from being too simple | Increase model complexity |
| Variance | Error from being too sensitive | Reduce complexity, get more data |
| Generalisation | Performs well on unseen data | The goal of every ML model |

---

*Part of my AI/ML notes series — one concept documented per day.*
*LinkedIn post → [Day 06 — Overfitting vs Underfitting](https://www.linkedin.com/posts/soumya-dodamani_dotnet-ai-rag-share-7447569298639798272-GDRB?utm_source=share&utm_medium=member_desktop&rcm=ACoAADIqRfgB9q57kOF3oEPWKkRR61-VN5IONck)*
