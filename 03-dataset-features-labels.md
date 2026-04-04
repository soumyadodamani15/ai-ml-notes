# Day 03 — What is a Dataset? Features, Labels, Training vs Test Sets

> **One line to remember:**
> Dataset = your raw material. Features = the inputs. Label = the answer you want to predict. Training data teaches. Test data honestly checks.

---

## Concept

Before a machine can learn anything, it needs data — organised in a specific way.

---

**What is a Dataset?**

A dataset is a collection of examples the machine learns from. Think of it as a spreadsheet — rows are examples, columns are properties.

```
Age   Salary   Purchased?
25    30000    No
35    60000    Yes
45    80000    Yes
22    25000    No
```

Every row is one example. Every column is one property.

---

**What are Features?**

Features are the input columns — the information you feed into the model to help it make a prediction. In the table above, `Age` and `Salary` are features.

Also called: independent variables, inputs, or predictors.

---

**What is a Label?**

The label is the output column — the answer you want the model to predict. In the table above, `Purchased?` is the label.

Labels only exist in supervised learning. In unsupervised learning there are no labels.

---

**What is Training Data?**

The portion of your dataset you use to teach the model. The model sees these examples, finds patterns, and adjusts itself.

---

**What is Test Data?**

The portion of your dataset you hide from the model during training and only use at the end to check how well it learned. The model has never seen these examples — so the result is an honest measure of performance.

---

**Why split at all?**

A student who memorises exam questions scores 100% on that exam — but fails a slightly different question. They did not learn. They memorised.

ML models can do the same thing — called **overfitting**. The train/test split catches this. If a model scores well on training data but poorly on test data, it memorised instead of learning.

---

**Typical splits:**

```
Full Dataset  →  80% Training  +  20% Test

# With validation:
Full Dataset  →  70% Training  +  15% Validation  +  15% Test
```

Validation is used during training to tune the model. Test is only touched once at the very end.

---

## Real World Example

You want to build a model that predicts whether a loan applicant will default.

| Concept | Example |
|---|---|
| Dataset | 50,000 past loan applications |
| Features | Age, income, credit score, loan amount, employment status |
| Label | Defaulted — Yes or No |
| Training data | 40,000 applications the model learns from |
| Test data | 10,000 applications used to check if the model actually works |

---

## Code

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Create a simple dataset
data = {
    'Age':       [25, 35, 45, 22, 40, 30],
    'Salary':    [30000, 60000, 80000, 25000, 70000, 50000],
    'Purchased': [0, 1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Separate features and label
X = df[['Age', 'Salary']]  # features
y = df['Purchased']         # label

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training size: {len(X_train)}")  # → 4
print(f"Test size:     {len(X_test)}")   # → 2
```

---

## Summary

| Concept | What it is | Analogy |
|---|---|---|
| Dataset | Collection of examples | A spreadsheet of past records |
| Features | Input columns | The questions on an exam |
| Label | Output column | The answer key |
| Training data | Data model learns from | Student studying with examples |
| Test data | Data used to evaluate model | The actual exam — unseen until the end |

---

*Part of my AI/ML notes series — one concept documented per day.*
*LinkedIn post → [Day 03 — What is a Dataset?]()*
