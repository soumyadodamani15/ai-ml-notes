# Day 05 — Logistic Regression

> **One line to remember:**
> Logistic regression predicts the probability that something belongs to a category — then uses a threshold to make a yes or no decision.

---

## Concept

Despite the name — logistic regression is not used for predicting numbers. It is used for **classification** — predicting which category something belongs to.

The most common use case: yes or no questions.

- Is this email spam or not?
- Will this customer churn or stay?
- Is this transaction fraudulent or legitimate?

---

**How is it different from Linear Regression?**

| | Linear Regression | Logistic Regression |
|---|---|---|
| Output | A number — price, salary, temperature | A probability between 0 and 1 |
| Goal | Predict a value | Predict a category |
| Example | Predict house price | Predict spam or not spam |

---

**The key ingredient — the Sigmoid function:**

Linear regression draws a straight line. The problem is a straight line can give you values like -3 or 1.7 — which make no sense as probabilities.

Logistic regression applies a **sigmoid function** to squash any value into the range 0 to 1.

```
Sigmoid:  f(x) = 1 / (1 + e^(-x))
```

No matter what number goes in — the output is always between 0 and 1. That output is a probability.

```
Output > 0.5  →  Yes (spam, fraud, churn)
Output < 0.5  →  No (not spam, legitimate, stays)
```

---

**A simple example:**

You want to predict whether a student passes an exam based on hours studied:

```
Hours Studied   Passed?
1               No
2               No
3               No
5               Yes
6               Yes
8               Yes
```

Logistic regression learns the boundary — the point where the probability crosses 0.5. Below that boundary — fail. Above it — pass.

---

**Classification types:**

- **Binary classification** — two possible outputs. Spam/not spam. Pass/fail. This is what logistic regression handles natively.
- **Multi-class classification** — more than two outputs. Cat/dog/bird. Logistic regression can be extended for this but other algorithms handle it better.

---

## Real World Examples

| Use Case | Features | Label |
|---|---|---|
| Spam detection | Word frequency, sender | Spam / Not spam |
| Fraud detection | Amount, location, time | Fraud / Legitimate |
| Medical diagnosis | Test results, age | Disease / No disease |
| Customer churn | Usage, complaints, plan | Churns / Stays |

---

## Code

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Hours studied vs pass/fail
X = np.array([[1], [2], [3], [5], [6], [8]])  # hours studied
y = np.array([0, 0, 0, 1, 1, 1])              # 0 = fail, 1 = pass

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions)}")

# Predict probability for a student who studied 4 hours
prob = model.predict_proba([[4]])
print(f"Probability of passing: {prob[0][1]:.2f}")
```

---

## Summary

| Concept | Meaning |
|---|---|
| Logistic Regression | Classification algorithm — predicts a category |
| Sigmoid | Squashes any number into a probability between 0 and 1 |
| Threshold | Usually 0.5 — above it means Yes, below means No |
| Binary classification | Two possible outputs — the core use case |
| Probability output | Unlike linear regression, output is always between 0 and 1 |

---

*Part of my AI/ML notes series — one concept documented per day.*
*LinkedIn post → [Day 05 — Logistic Regression](https://www.linkedin.com/posts/activity-7447369367287291904-8p4d?utm_source=share&utm_medium=member_desktop&rcm=ACoAADIqRfgB9q57kOF3oEPWKkRR61-VN5IONck)*
