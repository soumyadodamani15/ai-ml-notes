# Day 13 — Support Vector Machines (SVM) — Margin and Kernels

> **In a nutshell:**
> SVM finds the boundary between classes with the widest possible margin. The kernel trick handles data that cannot be separated by a straight line — by lifting it into a higher dimension where separation becomes possible.

---

## Concept

Support Vector Machines — SVM — is one of the most powerful and elegant algorithms in classical ML. It works brilliantly for classification, especially when your data is clean and well structured.

---

**The core idea:**

Imagine apples on the left and lemons on the right on a table. Many lines can separate them. But which line is the safest choice?

```
🍎 🍎 🍎  |  🍋 🍋 🍋   ← line too close to lemons
🍎 🍎 🍎     |     🍋 🍋 🍋   ← line right in the middle ✅
🍎 🍎 🍎        |  🍋 🍋 🍋   ← line too close to apples
```

The middle line is safest — it has the most breathing room on both sides. If a new fruit arrives slightly out of place — the middle line is least likely to misclassify it.

SVM always finds that middle line — the one with **maximum breathing room on both sides.**

---

## Key Terms

**Margin**
The breathing room on both sides of the boundary line. SVM maximises this.
Wider margin = more robust = better generalisation to new data.

**Support Vectors**
The data points sitting right at the edge of the margin — the ones closest to the line.
They are the only points that define where the line goes.
Remove any other point and the line stays the same.

**Hyperplane**
The technical name for the boundary line.
In 2D it is a line. In 3D it is a plane. In higher dimensions it is a hyperplane.

---

## The Kernel Trick

Sometimes the groups are completely mixed and no straight line can separate them:

```
🍎 🍋 🍎 🍋 🍎 🍋   ← impossible to draw a straight line here
```

**The solution — lift the data into a higher dimension:**

Imagine all the fruits are on a flat table — a 2D surface. You cannot draw a straight line between them because they are mixed.

Now imagine you pick up some fruits and lift them into the air — into 3D space. Suddenly you can slide a flat sheet of paper between the apples in the air and the lemons on the table.

That sheet of paper is the separation boundary. Impossible in 2D. Easy in 3D.

The kernel function does this lifting mathematically — without actually computing every high-dimensional coordinate explicitly. It is a clever shortcut.

---

## Common Kernels

| Kernel | When to use |
|---|---|
| Linear | Data is linearly separable — a straight line works |
| RBF (Radial Basis Function) | Most common — works well for non-linear data |
| Polynomial | When relationship between features is polynomial |
| Sigmoid | Rarely used — similar to neural network activation |

RBF is the default and works well in most cases. Start here.

---

## The C Parameter

C controls how strict the model is about making mistakes during training.

```
High C  →  refuses to allow mistakes → tight boundary → risk of overfitting
Low C   →  allows some mistakes → wider margin → better generalisation
Right C →  sweet spot — generalises well to new data
```

Think of C as a dial between memorising and generalising:
- Turn it up → more variance → overfitting
- Turn it down → more bias → underfitting
- Start with C=1 and adjust based on validation performance

This is the same bias-variance tradeoff from Day 06 — applied as a single number.

---

## Important — SVM Needs Normalisation

Unlike Decision Trees and Random Forests — SVM is sensitive to scale.

If one feature ranges from 1 to 10 and another from 1 to 100,000 — SVM will be heavily influenced by the larger one. Always use StandardScaler before training an SVM.

---

## Strengths and Weaknesses

**Strengths:**
- Works very well in high-dimensional spaces
- Effective when features outnumber examples
- Memory efficient — only uses support vectors
- Powerful with the right kernel

**Weaknesses:**
- Slow on very large datasets
- Requires careful tuning of C and kernel
- Does not directly give probabilities — only a class decision
- Must normalise features before training

---

## Real World Examples

| Use Case | Why SVM works well |
|---|---|
| Image classification | Works well in high-dimensional spaces |
| Text classification | Many features (words) — SVM handles this well |
| Bioinformatics | Small datasets with many features — SVM's sweet spot |
| Handwriting recognition | Clean separation of character classes |

---

## Code

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

# Dataset — age, income → will buy?
X = np.array([
    [25, 30000], [35, 70000], [45, 90000],
    [22, 20000], [40, 60000], [30, 45000],
    [50, 80000], [28, 35000], [33, 55000],
    [48, 85000], [27, 32000], [42, 75000]
])
y = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# SVM needs normalisation — always scale first
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train — RBF kernel, C=1 is a good starting point
model = SVC(kernel='rbf', C=1.0, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")

# How many support vectors were used?
print(f"Support vectors per class: {model.n_support_}")
```

---

## Summary

| Concept | Meaning |
|---|---|
| SVM | Finds the boundary with the widest possible margin |
| Margin | Breathing room between the boundary and the nearest points |
| Support Vectors | Points at the edge of the margin — define the boundary |
| Kernel Trick | Transforms data into higher dimensions where separation is possible |
| RBF Kernel | Default kernel — handles non-linear data well |
| C Parameter | Controls strictness — high C = tight boundary, low C = wide margin |
| Normalisation required | SVM is sensitive to scale — always scale before training |

---

*Part of my AI/ML notes series — one concept documented per day.*
*LinkedIn post → [Day 13 — Support Vector Machines](https://www.linkedin.com/posts/soumya-dodamani_ai-machinelearning-deeplearning-share-7450986627017023488-kJ_V?utm_source=share&utm_medium=member_desktop&rcm=ACoAADIqRfgB9q57kOF3oEPWKkRR61-VN5IONck)*
