# Day 14 — K-Nearest Neighbors (KNN) — Lazy Learning

> **In a nutshell:**
> KNN makes predictions by finding the K most similar examples in the training data and taking a vote. No training happens — all the work is done at prediction time. Simple, intuitive, but slow on large datasets.

---

## Concept

KNN is one of the simplest and most intuitive algorithms in ML. Unlike every algorithm covered so far — KNN does not learn anything during training.

Instead it memorises the entire training dataset and only does the work when you ask it to make a prediction.

This is why it is called **Lazy Learning.**

---

**The core idea:**

To classify a new data point — KNN looks at the K nearest points in the training data and takes a vote.

```
K=3 → find the 3 nearest neighbours → majority vote → prediction
K=5 → find the 5 nearest neighbours → majority vote → prediction
```

No training. No weights. No gradient descent. Just distance and voting.

---

**A simple example:**

You want to predict whether a new customer will buy a product.

New customer → age 35, income €55,000.

KNN finds the 3 nearest customers in historical data:
```
Neighbour 1 → age 33, income €52,000 → Bought ✅
Neighbour 2 → age 37, income €58,000 → Bought ✅
Neighbour 3 → age 34, income €48,000 → Did not buy ❌

Majority vote → 2 bought, 1 did not → Prediction: Will buy ✅
```

---

## How It Measures Distance

The most common way is **Euclidean distance** — the straight line distance between two points.

```
Distance = √((x2-x1)² + (y2-y1)²)
```

What this formula is doing:
- Measure the gap in each feature
- Square each gap — removes negatives, penalises large gaps more
- Add them all up
- Take the square root — brings the result back to original units

Result: the straight line distance between two points. Smaller = more similar.

You do not need to memorise this. Just know: smaller distance = more similar points.

---

## Choosing K

K is the number of neighbours to consider. Getting this right matters a lot.

```
K too small (K=1) → very sensitive to noise
                    one wrong neighbour = wrong prediction
                    overfitting

K too large       → considers too many neighbours
                    prediction dominated by the majority class
                    underfitting

K just right      → smooth, reliable predictions
```

**Rules of thumb:**
- Start with K = √(number of training examples)
- Always use an odd number — avoids ties in binary classification
- Tune K using cross-validation

---

## KNN for Regression

KNN works for regression too — not just classification.

Instead of majority vote — take the **average** of the K nearest neighbours' values.

```
Neighbour 1 → house price €280,000
Neighbour 2 → house price €310,000
Neighbour 3 → house price €295,000

Prediction → average = €295,000
```

---

## Why KNN Needs Normalisation

KNN measures distance between points. If one feature has a much larger scale — it dominates the distance calculation unfairly.

```
Age:    25 to 60        → range of 35
Income: 20,000 to 100,000 → range of 80,000
```

Without normalisation — income completely dominates. Age barely contributes. Always use StandardScaler or MinMaxScaler before KNN.

---

## Strengths and Weaknesses

**Strengths:**
- Dead simple to understand and implement
- No training time — the model is just the data
- Naturally handles multi-class problems
- Works well for small, clean datasets
- No assumptions about data distribution

**Weaknesses:**
- Prediction is slow — must calculate distance to every training point
- Does not scale to large datasets
- Very sensitive to irrelevant features
- Must normalise — scale matters heavily
- High memory usage — stores entire training set

---

## Real World Examples

| Use Case | Why KNN works well |
|---|---|
| Recommendation systems | Find users similar to you — recommend what they liked |
| Medical diagnosis | Find patients with similar symptoms — use their outcomes |
| Anomaly detection | Points far from all neighbours are likely outliers |
| Image recognition | Find the most similar image in the training set |

---

## Code

```python
from sklearn.neighbors import KNeighborsClassifier
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

# KNN needs normalisation — always scale first
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train — K=3, start small and tune
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)  # just memorises the data

# Predict — distance calculation happens here not during training
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")

# Try different values of K to find the best one
for k in [1, 3, 5, 7]:
    m = KNeighborsClassifier(n_neighbors=k)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    print(f"K={k}: Accuracy = {acc:.2f}")
```

---

## Summary

| Concept | Meaning |
|---|---|
| KNN | Finds K nearest neighbours and votes |
| Lazy Learning | No training — all work done at prediction time |
| Euclidean Distance | Straight line distance between two points |
| K | Number of neighbours to consider |
| Small K | Sensitive to noise — overfitting risk |
| Large K | Dominated by majority class — underfitting risk |
| Normalisation required | Scale affects distance — always scale before KNN |
| KNN Regression | Average of K nearest neighbours instead of vote |

---

*Part of my AI/ML notes series — one concept documented per day.*
*LinkedIn post → [Day 14 — K-Nearest Neighbors](https://www.linkedin.com/posts/soumya-dodamani_ai-machinelearning-deeplearning-share-7451900866581905408-AQTg?utm_source=share&utm_medium=member_desktop&rcm=ACoAADIqRfgB9q57kOF3oEPWKkRR61-VN5IONck)*
