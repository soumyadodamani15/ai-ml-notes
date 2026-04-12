# Day 10 — Feature Engineering and Normalisation

> **In a nutshell:**
> Feature engineering transforms raw data into meaningful inputs. Normalisation ensures no feature dominates because of its scale. Together they are often the difference between a mediocre model and a great one.

---

## Concept

You have data. You have an algorithm. But the quality of your model depends heavily on one thing most beginners underestimate — the quality of the features you feed into it.

Feature engineering and normalisation are about preparing your data so the model can learn as effectively as possible.

---

## Feature Engineering

Feature engineering is the process of using domain knowledge to create, transform, or select the best input features for your model.

Raw data is rarely in the perfect shape for a model to learn from. Feature engineering bridges that gap.

---

**1. Creating new features from existing ones**

Raw data often hides useful information. You extract it.

```
Raw data:     date = "2024-03-15"
New features: year = 2024, month = 3, day_of_week = Friday, is_weekend = False
```

A model cannot learn from a raw date string. But it can learn a lot from month, day of week, and whether it is a weekend.

---

**2. Combining features**

```
Raw data:    height = 1.80m, weight = 80kg
New feature: BMI = weight / height² = 24.7
```

BMI is more meaningful to a health model than raw height and weight separately.

---

**3. Encoding categorical variables**

Models work with numbers — not text. You need to convert categories into numbers.

```
Raw data:  colour = ["red", "blue", "green"]
Encoded:   red=[1,0,0]  blue=[0,1,0]  green=[0,0,1]
```

This is called one-hot encoding. Each category becomes its own column with a 1 or 0.

---

**4. Handling missing values**

Real world data is messy. Missing values break models.

Options:
- Fill with the mean or median of the column
- Fill with the most common value
- Drop the row entirely
- Create a new feature: "was this value missing?" — sometimes missingness itself is informative

---

**5. Removing irrelevant features**

More features is not always better. Irrelevant features add noise and slow training.
A customer's name or ID number tells the model nothing useful about whether they will churn — drop them.

---

## Normalisation

Normalisation is about scaling your numerical features so they are on a comparable scale.

**Why does scale matter?**

Imagine predicting house prices with two features:
```
Size:  50 to 500 sqm   → range of 450
Rooms: 1 to 10         → range of 9
```

The model will pay far more attention to size simply because its numbers are larger — not because it is more important. This unfairly biases the model.

Think of it like grading two students on Maths (out of 100) and Attendance (out of 5). If you just add the scores, Maths dominates — not because it matters more, but because its scale is bigger. Normalisation levels the playing field.

---

**Min-Max Normalisation:**

```
scaled = (value - min) / (max - min)
```

What it does: takes every value and squashes it between 0 and 1.
- Smallest value → becomes 0
- Largest value → becomes 1
- Everything else → somewhere in between

When to use: when you know the bounds of your data and there are no extreme outliers.
Good for: neural networks, KNN.

---

**Standardisation (Z-score scaling):**

```
scaled = (value - mean) / standard deviation
```

What it does: centres data around zero. Most values end up between -3 and +3.
- Zero means average
- Positive means above average
- Negative means below average

Why better for outliers: uses mean and spread — not min and max. One extreme outlier does not push everything else toward zero.

When to use: when your data has outliers or follows a roughly normal distribution.
Good for: linear models, SVM.

---

**When no scaling is needed:**

Tree-based models — Decision Trees and Random Forests — make decisions by asking threshold questions:
```
Is house size > 200 sqm? Yes → go left. No → go right.
```

They never add or multiply features together. Scale does not affect their decisions. No normalisation needed.

---

## When to Use Which

| Technique | Use when |
|---|---|
| Min-Max Normalisation | Bounded data, neural networks, KNN |
| Standardisation | Data with outliers, linear models, SVM |
| No scaling needed | Decision Trees, Random Forests |

---

## Real World Example

Building a model to predict customer churn:

| Raw Feature | Problem | Engineered Feature |
|---|---|---|
| signup_date | Model cannot use a date | days_since_signup |
| country = "Germany" | Model cannot use text | is_germany = 1, is_other = 0 |
| last_login = null | Missing value breaks model | days_since_login = 999, has_logged_in = 0 |
| monthly_spend | Large scale biases model | normalised_spend (0 to 1) |
| customer_id | Meaningless to the model | Drop it entirely |

---

## Code

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Raw dataset
data = {
    'signup_date':      ['2023-01-15', '2022-06-20', '2024-03-01'],
    'country':          ['Germany', 'France', 'Germany'],
    'monthly_spend':    [120, 450, 80],
    'last_login_days':  [2, None, 15]
}

df = pd.DataFrame(data)

# 1. Create new features from date
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['days_since_signup'] = (pd.Timestamp.now() - df['signup_date']).dt.days

# 2. One-hot encode categorical variable
df = pd.get_dummies(df, columns=['country'])

# 3. Handle missing values
df['last_login_days'] = df['last_login_days'].fillna(999)

# 4. Drop original date column — no longer needed
df = df.drop(columns=['signup_date'])

# 5. Min-Max normalisation
scaler = MinMaxScaler()
df[['monthly_spend', 'days_since_signup']] = scaler.fit_transform(
    df[['monthly_spend', 'days_since_signup']]
)

print(df)
```

---

## Summary

| Concept | What it does | Why it matters |
|---|---|---|
| Feature engineering | Transforms raw data into useful inputs | Raw data is rarely model-ready |
| One-hot encoding | Converts categories to numbers | Models only understand numbers |
| Missing value handling | Fills or flags missing data | Missing values break models |
| Min-Max normalisation | Scales values to 0–1 | Prevents large-scale features dominating |
| Standardisation | Centres data around zero | More robust to outliers than Min-Max |

---

*Part of my AI/ML notes series — one concept documented per day.*
*LinkedIn post → [Day 10 — Feature Engineering and Normalisation]()*
