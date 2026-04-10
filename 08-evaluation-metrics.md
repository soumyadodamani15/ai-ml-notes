# Day 08 — Evaluation Metrics: Accuracy, Precision, Recall, F1

> **One line to remember:**
> Accuracy tells you how often you are right. Precision tells you how trustworthy your yes is. Recall tells you how many real cases you caught. F1 balances both.

---

## Concept

You have built a model. It makes predictions. But how do you know if it is actually good?

Accuracy alone is not enough. Here is why.

---

**The problem with Accuracy:**

Imagine you build a model to detect a rare disease that affects 1% of the population. Your model predicts "no disease" for every single person.

Accuracy = 99%. Sounds great. But the model is completely useless — it never detects the disease at all.

This is why you need more than one metric.

---

## The Confusion Matrix

Before the metrics, you need to understand four outcomes:

```
                    Predicted: YES    Predicted: NO
Actual: YES         True Positive     False Negative
Actual: NO          False Positive    True Negative
```

| Term | Meaning |
|---|---|
| True Positive (TP) | Model said YES, actually YES. Correct. |
| True Negative (TN) | Model said NO, actually NO. Correct. |
| False Positive (FP) | Model said YES, actually NO. Wrong. Type I error. |
| False Negative (FN) | Model said NO, actually YES. Wrong. Type II error. |

---

## The Metrics

**Accuracy:**
```
Accuracy = (TP + TN) / Total
```
How often the model is correct overall. Misleading when classes are imbalanced.

---

**Precision:**
```
Precision = TP / (TP + FP)
```
Of all the times the model said YES — how many were actually YES?

- High precision = when the model says yes, you can trust it
- Use when **false positives are costly**
- e.g. Spam filter — you do not want legitimate emails flagged as spam

---

**Recall (Sensitivity):**
```
Recall = TP / (TP + FN)
```
Of all the actual YES cases — how many did the model catch?

- High recall = the model misses very few real positives
- Use when **false negatives are costly**
- e.g. Cancer detection — missing a real case is far worse than a false alarm

---

**F1 Score:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
The harmonic mean of precision and recall. A single number that balances both.

Use when you need to balance precision and recall — especially with imbalanced datasets.

---

## Which Metric to Use When

| Situation | Metric to prioritise |
|---|---|
| Balanced classes, general performance | Accuracy |
| False positives are costly (spam filter) | Precision |
| False negatives are costly (cancer, disease) | Recall |
| Need balance between both (fraud detection) | F1 Score |

---

## Real World Examples

| Use Case | Priority | Why |
|---|---|---|
| Spam filter | Precision | Do not block legitimate emails |
| Cancer detection | Recall | Never miss a real case |
| Fraud detection | F1 | Balance catching fraud and not annoying customers |
| Image classification | Accuracy | Classes are usually balanced |

---

## Code

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Actual vs predicted
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

print(f"Accuracy:  {accuracy_score(y_true, y_pred):.2f}")
print(f"Precision: {precision_score(y_true, y_pred):.2f}")
print(f"Recall:    {recall_score(y_true, y_pred):.2f}")
print(f"F1 Score:  {f1_score(y_true, y_pred):.2f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
```

---

## Summary

| Metric | Formula | Best used when |
|---|---|---|
| Accuracy | (TP + TN) / Total | Balanced classes |
| Precision | TP / (TP + FP) | False positives are costly |
| Recall | TP / (TP + FN) | False negatives are costly |
| F1 Score | 2 × (P × R) / (P + R) | Need to balance both |

---

*Part of my AI/ML notes series — one concept documented per day.*
*LinkedIn post → [Day 08 — Evaluation Metrics]()*
