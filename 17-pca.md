# Day 17 — PCA — Dimensionality Reduction Explained

> **In a nutshell:**
> PCA reduces the number of features by finding the directions where data varies the most. You keep the top few and discard the rest. Less data. Faster training. Less overfitting. Small loss of information.

---

## Concept

As datasets grow — they often have hundreds or thousands of features. More features sounds better. But it creates real problems.

This is called the **Curse of Dimensionality** — and PCA is one of the most important tools to fight it.

---

**The problem with too many features:**

- Models become slow to train
- Models overfit more easily — too many dimensions, not enough data
- It becomes impossible to visualise the data
- Many features are often redundant — they carry similar information

**Principal Component Analysis (PCA)** reduces the number of features while keeping as much useful information as possible.

---

## The Core Idea

Imagine you have data with 100 features. But many of those features are correlated — they move together. Height and shoe size tend to increase together.

PCA finds the directions in the data where the most variation exists — called **principal components** — and projects the data onto those directions.

The first principal component captures the most variation. The second captures the second most. And so on.

You keep only the top few components — and discard the rest. You lose a little information but gain a much simpler dataset.

---

**A simple visual example:**

Imagine data points scattered in 2D — x and y axes. Most of the variation runs diagonally — along one main direction.

```
        y
        |    • •
        |  • • •
        | • • •
        |• •
        |_________ x
```

PCA finds that diagonal direction — the axis where the data is most spread out. It rotates the data so that axis becomes the new x axis. Now most of the information sits in one dimension instead of two.

You have gone from 2 features to 1 — without losing much information.

---

## Principal Components

| Component | Meaning |
|---|---|
| PC1 | Direction of greatest variation in the data |
| PC2 | Direction of second greatest variation — perpendicular to PC1 |
| PC3 | Third greatest — perpendicular to both PC1 and PC2 |
| ... | And so on |

Each principal component is a combination of the original features — not a single feature. A blend of all of them weighted by importance.

---

## Explained Variance

Each principal component explains a certain percentage of the total variation in the data.

```
PC1 → explains 60% of variation
PC2 → explains 25% of variation
PC3 → explains 10% of variation
PC4 → explains  5% of variation
```

PC1 + PC2 together explain 85% of all variation. If you keep just these two — you have reduced from 4 features to 2 while retaining 85% of the information.

**Rule of thumb:** keep enough components to explain 95% of the variance.

---

## When to Use PCA

**Use PCA when:**
- Training a slow algorithm on high-dimensional data
- Many features are highly correlated and redundant
- You want to visualise high-dimensional data in 2D or 3D
- Reducing overfitting caused by too many features

**Do NOT use PCA when:**
- Interpretability matters — PCA components are blends of features and lose their original meaning
- Your features are already independent and meaningful on their own
- Using tree-based models — they handle high dimensions well and do not need PCA

---

## Important — PCA Needs Normalisation

PCA finds directions of maximum variance. If one feature has a much larger scale — it will dominate the variance and PCA will focus on it unfairly.

Always standardise your features before applying PCA.

---

## Real World Examples

| Use Case | How PCA helps |
|---|---|
| Image recognition | Images have thousands of pixels — PCA reduces to key features |
| Gene expression | Thousands of genes — PCA finds the key patterns |
| Finance | Many correlated stock features — PCA reduces redundancy |
| Visualisation | Reduce any dataset to 2D or 3D for plotting |

---

## Code

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Sample dataset — 4 features
X = np.array([
    [2.5, 2.4, 3.1, 2.8],
    [0.5, 0.7, 0.9, 0.6],
    [2.2, 2.9, 2.7, 3.0],
    [1.9, 2.2, 2.0, 2.1],
    [3.1, 3.0, 3.5, 3.2],
    [2.3, 2.7, 2.4, 2.6],
    [2.0, 1.6, 1.8, 1.9],
    [1.0, 1.1, 1.2, 1.0],
    [1.5, 1.6, 1.4, 1.7],
    [1.1, 0.9, 1.0, 1.2]
])

# Always normalise before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA — keep top 2 components
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

print(f"Original shape:  {X.shape}")         # (10, 4)
print(f"Reduced shape:   {X_reduced.shape}") # (10, 2)

# How much variance does each component explain?
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance retained:  {sum(pca.explained_variance_ratio_):.2%}")

# Find how many components explain 95% of variance
pca_95 = PCA(n_components=0.95)
X_95 = pca_95.fit_transform(X_scaled)
print(f"Components needed for 95% variance: {pca_95.n_components_}")
```

---

## Summary

| Concept | Meaning |
|---|---|
| PCA | Reduces features by finding directions of maximum variance |
| Principal Component | A direction of variation — a blend of original features |
| PC1 | Captures the most variation |
| Explained Variance | How much information each component retains |
| Curse of Dimensionality | Too many features causes slow training and overfitting |
| 95% rule | Keep enough components to retain 95% of variance |
| Normalisation required | Scale affects variance — always standardise before PCA |
| Not for tree models | Decision Trees and Random Forests handle dimensions well already |

---

*Part of my AI/ML notes series — one concept documented per day.*
*LinkedIn post → [Day 17 — PCA — Dimensionality Reduction](https://www.linkedin.com/posts/soumya-dodamani_ai-machinelearning-deeplearning-share-7455525371489398787-aJaj?utm_source=share&utm_medium=member_desktop&rcm=ACoAADIqRfgB9q57kOF3oEPWKkRR61-VN5IONck)*
