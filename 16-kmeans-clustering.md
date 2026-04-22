# Day 16 — K-Means Clustering — Unsupervised Grouping

> **In a nutshell:**
> K-Means groups data into K clusters by repeatedly assigning each point to its nearest centre and moving the centre to the average of its points. No labels needed — the algorithm discovers the natural structure in your data.

---

## Concept

Every algorithm covered so far has been supervised — you provide labels and the model learns from them.

K-Means is different. It is **unsupervised** — no labels, no correct answers. You just give it data and it finds natural groups on its own.

---

**The core idea:**

You have a dataset of customers. No labels. You want to group them into segments based on their behaviour — without knowing the groups in advance.

K-Means does exactly this. You tell it how many groups you want — K — and it figures out the best way to divide the data into K clusters.

---

## How It Works — Step by Step

```
Step 1 — Pick K random points as starting cluster centres (centroids)
Step 2 — Assign every data point to its nearest centroid
Step 3 — Move each centroid to the average position of all points assigned to it
Step 4 — Repeat steps 2 and 3 until the centroids stop moving
```

That is the entire algorithm. Assign. Update. Repeat.

---

**A simple example:**

You have customer data — age and spending score. You want 3 segments.

```
Iteration 1:
- Place 3 random centroids
- Assign each customer to nearest centroid
- Move centroids to the average position of their customers

Iteration 2:
- Some customers now closer to a different centroid
- Reassign them
- Move centroids again

Iteration 10:
- Centroids barely move
- Clusters have stabilised → Done
```

End result — three customer segments discovered automatically:
- Young, low spenders
- Middle-aged, high spenders
- Older, medium spenders

---

## Key Terms

**Centroid**
The centre point of a cluster — the average position of all data points in that cluster.
Think of it as the point where everyone in the group is equally close on average.

**Inertia**
The total distance of all points from their assigned centroid.
Lower inertia = more compact clusters = better fit.
Used in the Elbow Method to find the best K.

**Convergence**
When the centroids stop moving between iterations — the algorithm has found stable clusters.

---

## Choosing K — The Elbow Method

You must tell K-Means how many clusters you want. But how do you know the right number?

Run K-Means for different values of K and plot inertia against K:

```
K=1 → inertia very high
K=2 → inertia drops a lot
K=3 → inertia drops again
K=4 → barely drops ← elbow here → K=3 is the right choice
K=5 → barely drops
K=6 → barely drops
```

The "elbow" — where the curve bends and adding more clusters gives diminishing returns — is usually the best K.

---

## Limitations

- You must choose K in advance — not always obvious
- Sensitive to initial centroid placement — different runs can give different results
  Fix: run multiple times with different starts — sklearn does this automatically with n_init
- Assumes clusters are roughly circular and similar in size
- Sensitive to outliers — one extreme point can pull a centroid far off
- Must normalise — distance based, scale matters

---

## Why Normalisation Is Required

K-Means measures distance between points. If one feature has a much larger scale — it dominates the distance calculation.

```
Age:            25 to 65     → range of 40
Annual Spend:   100 to 10000 → range of 9900
```

Without normalisation — spend completely dominates. Age barely matters. Always scale before clustering.

---

## Real World Examples

| Use Case | What K-Means finds |
|---|---|
| Customer segmentation | Groups of customers with similar behaviour |
| Document clustering | Topics in a collection of articles |
| Image compression | Groups of similar pixel colours |
| Anomaly detection | Points far from any cluster are outliers |
| Market segmentation | Groups of products with similar sales patterns |

---

## Code

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Customer data — age and spending score
X = np.array([
    [25, 20], [30, 25], [22, 15],  # young, low spenders
    [45, 80], [50, 90], [48, 85],  # middle-aged, high spenders
    [60, 50], [65, 45], [58, 55]   # older, medium spenders
])

# Always normalise before K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train — K=3 clusters
model = KMeans(n_clusters=3, random_state=42, n_init=10)
model.fit(X_scaled)

# Cluster assignments
print("Cluster labels:", model.labels_)
print("Inertia:", model.inertia_)

# Elbow method — find the best K
inertias = []
K_range = range(1, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Plot elbow curve
plt.plot(K_range, inertias, marker='o')
plt.xlabel('Number of clusters K')
plt.ylabel('Inertia')
plt.title('Elbow Method — Finding the Best K')
plt.show()
```

---

## Summary

| Concept | Meaning |
|---|---|
| K-Means | Groups data into K clusters — no labels needed |
| Centroid | Centre point of a cluster — average of all its points |
| Inertia | Total distance of points from their centroid — lower is better |
| Elbow Method | Plot inertia vs K — find where the curve bends |
| Convergence | Centroids stop moving — algorithm is done |
| n_init | Number of times to run with different starts — pick the best |
| Normalisation required | Distance based — always scale before clustering |

---

*Part of my AI/ML notes series — one concept documented per day.*
*LinkedIn post → [Day 16 — K-Means Clustering](https://www.linkedin.com/posts/soumya-dodamani_ai-machinelearning-deeplearning-share-7452607988147568640-Ou2y?utm_source=share&utm_medium=member_desktop&rcm=ACoAADIqRfgB9q57kOF3oEPWKkRR61-VN5IONck)*
