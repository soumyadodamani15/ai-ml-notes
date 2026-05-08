# Day 20 — Regularisation: L1 (Lasso) and L2 (Ridge)

> **In a nutshell:**
> L2 shrinks all weights. L1 zeroes out the unimportant ones. Lambda controls how hard the penalty hits. Together they keep your model generalising instead of memorising.

---

## Concept

You learned about overfitting in Day 06. A model that is too complex memorises the training data and fails on new data.

Regularisation is one of the most effective ways to fight overfitting — it penalises the model for being too complex during training itself.

---

## The Core Idea

During training the model tries to minimise loss — how wrong its predictions are.

Regularisation adds an extra penalty to that loss — a punishment for having large weights.

```
Without regularisation:
Loss = prediction error only

With regularisation:
Loss = prediction error + penalty for large weights
```

The model now has to balance two things — fitting the data well AND keeping weights small. This prevents it from going overboard trying to perfectly fit every training example.

---

## Why Do Large Weights Cause Overfitting?

Large weights mean the model is very sensitive to small changes in input. A tiny change in a feature causes a massive swing in the prediction. That is a sign the model has memorised specific quirks of the training data rather than learned the general pattern.

Small weights = smoother, more generalised predictions.

---

## L2 Regularisation — Ridge

Adds the sum of squared weights to the loss.

```
Loss = prediction error + λ × (sum of all weights²)
```

What the formula is doing:
- Square each weight — makes the penalty grow quickly for large weights
- Sum them all up — total complexity of the model
- Multiply by λ — controls how strong the penalty is
- Add to prediction error — model minimises both together

Effect: pushes all weights toward zero — but never exactly to zero. Every feature stays in the model — just with smaller influence.

Think of it as: "shrink everyone a little."

Best for: when you believe most features are relevant and you just want to reduce their influence.

---

## L1 Regularisation — Lasso

Adds the sum of absolute weights to the loss.

```
Loss = prediction error + λ × (sum of |all weights|)
```

What the formula is doing:
- Take the absolute value of each weight — removes the sign
- Sum them all up
- Multiply by λ
- Add to prediction error

Effect: pushes some weights all the way to exactly zero — effectively removing those features from the model entirely. Performs automatic feature selection.

Think of it as: "pick the important ones and drop the rest."

Best for: when you have many features and suspect only a few actually matter. L1 finds them for you.

---

## The λ (Lambda) Parameter

Lambda controls how strong the regularisation penalty is.

```
λ too high → penalty too strong → model too simple → underfitting
λ too low  → penalty too weak  → model still overfits
λ just right → balanced → good generalisation
```

Tune lambda using cross-validation. In sklearn it is called `alpha` instead of `lambda`.

---

## L1 vs L2 — When to Use Which

| | L1 (Lasso) | L2 (Ridge) |
|---|---|---|
| Effect on weights | Some go to exactly zero | All shrink but stay non-zero |
| Feature selection | Yes — automatic | No — keeps all features |
| Best when | Many features, few matter | Most features are relevant |
| Sensitive to outliers | More sensitive | Less sensitive |

**ElasticNet** — combines both L1 and L2. Useful when you want some feature selection but not as aggressive as pure L1.

---

## Where Regularisation Appears

Regularisation is not just for linear models. It appears everywhere:

| Model | Regularisation form |
|---|---|
| Linear / Logistic Regression | L1, L2, ElasticNet |
| Neural Networks | L2 weight decay, Dropout |
| SVM | C parameter (inverse regularisation) |
| Decision Trees | max_depth, min_samples_split |

---

## Real World Example

You are building a house price model with 100 features — size, location, age, number of windows, colour of front door, seller's name...

Without regularisation: model uses all 100 features. It memorises that houses with blue doors sold for slightly more in training data. This is noise — not signal.

With L1: model zeroes out door colour, seller's name, and other irrelevant features. Only truly important ones survive.

With L2: model keeps all features but reduces the influence of noisy ones. No feature dominates unfairly.

---

## Code

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Dataset — 20 features but only 2 actually matter
np.random.seed(42)
X = np.random.randn(100, 20)
y = 3*X[:,0] + 2*X[:,1] + np.random.randn(100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Always normalise before regularisation — scale affects penalty
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# No regularisation — overfits
lr = LinearRegression().fit(X_train, y_train)
print(f"No reg     MSE: {mean_squared_error(y_test, lr.predict(X_test)):.2f}")

# L2 Ridge — shrinks all weights
ridge = Ridge(alpha=1.0).fit(X_train, y_train)
print(f"Ridge      MSE: {mean_squared_error(y_test, ridge.predict(X_test)):.2f}")

# L1 Lasso — zeroes out irrelevant weights
lasso = Lasso(alpha=0.1).fit(X_train, y_train)
print(f"Lasso      MSE: {mean_squared_error(y_test, lasso.predict(X_test)):.2f}")
print(f"Lasso zeroed:  {sum(lasso.coef_ == 0)} of 20 features removed")

# ElasticNet — mix of L1 and L2
enet = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X_train, y_train)
print(f"ElasticNet MSE: {mean_squared_error(y_test, enet.predict(X_test)):.2f}")
```

---

## Summary

| Concept | Meaning |
|---|---|
| Regularisation | Adds a penalty for large weights to prevent overfitting |
| L1 (Lasso) | Penalty = sum of absolute weights — zeroes out some features |
| L2 (Ridge) | Penalty = sum of squared weights — shrinks all features |
| λ (lambda / alpha) | Controls penalty strength — tune with cross-validation |
| ElasticNet | Combines L1 and L2 — best of both |
| Feature selection | L1 does this automatically — L2 does not |
| Normalise first | Always scale features before regularisation — scale affects penalty |

---

*Part of my AI/ML notes series — one concept documented per day.*
*LinkedIn post → [Day 20 — Regularisation: L1 and L2](https://www.linkedin.com/posts/soumya-dodamani_ai-machinelearning-deeplearning-share-7458445462967177216-g57F?utm_source=share&utm_medium=member_desktop&rcm=ACoAADIqRfgB9q57kOF3oEPWKkRR61-VN5IONck)*
