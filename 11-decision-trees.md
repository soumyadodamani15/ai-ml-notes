# Day 11 — Decision Trees — How They Split Data

> **In a nutshell:**
> A Decision Tree learns by asking the best yes/no questions — splitting data into groups as pure as possible. It is the most explainable ML algorithm. Its weakness is overfitting — controlled by limiting depth.

---

## Concept

A Decision Tree is one of the most intuitive ML algorithms. It makes predictions by asking a series of yes/no questions — exactly like a flowchart.

---

**The core idea:**

You want to predict whether someone will buy a product. The tree asks:

```
Is age > 30?
├── Yes → Is income > 50,000?
│         ├── Yes → Will buy ✅
│         └── No  → Will not buy ❌
└── No  → Will not buy ❌
```

At each step the tree asks one question about one feature. Based on the answer it goes left or right. At the end of every branch it makes a prediction.

The model builds this tree itself from data — you do not write the conditions.

---

## How It Decides Which Question to Ask

The tree does not ask random questions. It asks the question that splits the data most cleanly — separating classes as clearly as possible.

The measure of "how clean is this split" is called **impurity.**

**Gini Impurity**

Measures how often a randomly chosen example would be incorrectly classified.
- 0 = perfectly pure — all examples in this group are the same class
- 0.5 = completely mixed — classes are evenly split

Job: find the split that brings Gini closest to 0.

**Information Gain (Entropy)**

Measures how much a split reduces uncertainty — how much information we gain by asking this question. Higher information gain = better question to ask.

Job: find the split that gives us the most information.

You do not need to memorise these formulas. Just know: **the tree always picks the question that makes the groups as pure as possible.**

---

## Key Terms

| Term | Meaning |
|---|---|
| Root Node | The very first question — the most important split |
| Internal Node | Any question in the middle of the tree |
| Leaf Node | The final answer — a prediction |
| Branch | The path taken based on yes or no |
| Depth | How many levels of questions the tree has |

---

## Overfitting in Decision Trees

A tree that keeps asking questions until every single training example is correctly classified will memorise the training data. It will have very high depth and fail on new data.

This is called a **fully grown tree** — and it overfits badly.

The fix: **pruning** — limiting how deep the tree can grow. You set a maximum depth so it learns the general pattern rather than memorising every detail.

```python
# max_depth=3 means the tree can only ask 3 levels of questions
model = DecisionTreeClassifier(max_depth=3)
```

---

## Strengths and Weaknesses

**Strengths:**
- Very easy to understand and explain — you can literally draw it
- No normalisation needed — scale does not affect splits
- Handles both numerical and categorical features
- Works well out of the box with minimal tuning

**Weaknesses:**
- Overfits easily if not pruned
- Small changes in data can produce very different trees — unstable
- Not the most accurate algorithm on its own

This is why Decision Trees are often used as building blocks for more powerful algorithms — like Random Forests — which we cover next.

---

## Real World Examples

| Use Case | Features | Prediction |
|---|---|---|
| Loan approval | Income, credit score, age | Approve / Reject |
| Medical diagnosis | Symptoms, test results, age | Disease / No disease |
| Customer churn | Usage, complaints, tenure | Churns / Stays |
| Email filtering | Word frequency, sender | Spam / Not spam |

---

## Code

```python
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Dataset — age, income → will buy?
X = np.array([
    [25, 30000], [35, 70000], [45, 90000],
    [22, 20000], [40, 60000], [30, 45000],
    [50, 80000], [28, 35000]
])
y = np.array([0, 1, 1, 0, 1, 0, 1, 0])  # 1 = will buy

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train — max_depth prevents overfitting
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")

# Print the tree — you can literally read every decision it made
print(export_text(model, feature_names=['Age', 'Income']))
```

---

## Summary

| Concept | Meaning |
|---|---|
| Decision Tree | Learns by asking yes/no questions about features |
| Impurity | How mixed the classes are in a group — lower is better |
| Gini Impurity | Measures misclassification rate — tree minimises this |
| Information Gain | Measures reduction in uncertainty — tree maximises this |
| Pruning | Limiting tree depth to prevent overfitting |
| Leaf Node | The end of a branch — where the prediction lives |
| No scaling needed | Trees split on thresholds — scale does not affect them |

---

*Part of my AI/ML notes series — one concept documented per day.*
*LinkedIn post → [Day 11 — Decision Trees](https://www.linkedin.com/posts/soumya-dodamani_ai-machinelearning-deeplearning-share-7449387826783174657-Qh5l?utm_source=share&utm_medium=member_desktop&rcm=ACoAADIqRfgB9q57kOF3oEPWKkRR61-VN5IONck)*
