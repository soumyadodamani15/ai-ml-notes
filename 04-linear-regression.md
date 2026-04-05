# Day 04 — Linear Regression

> **One line to remember:**
> Linear regression finds the best straight line through your data so you can predict a number — like price, salary, or temperature — from one or more inputs.

---

## Concept

Linear Regression is the simplest and most fundamental ML algorithm. Almost every advanced concept builds on it.

The core idea: find a straight line that best fits your data. Use that line to predict an output for any new input.

---

**A simple example:**

You have data on house sizes and their prices:

```
Size (sqm)   Price (€)
50           150,000
80           220,000
100          280,000
120          340,000
150          400,000
```

Linear regression finds the line that best fits these points. Once you have that line — give it any house size and it predicts the price.

---

**The formula:**

```
y = mx + b
```

| Symbol | Meaning |
|---|---|
| `y` | The output you want to predict (price) |
| `x` | The input feature (size) |
| `m` | The slope — how much y changes for every unit increase in x |
| `b` | The intercept — the value of y when x is zero |

The model's job during training is to find the best values of `m` and `b` that make the line fit the data as closely as possible. You do not write these values — the model discovers them from data.

---

**How does it know what "best fit" means?**

It minimises the error — the difference between what the line predicts and the actual value. These errors are called **residuals.**

The most common way to measure this is **Mean Squared Error (MSE)**:
- Calculate the difference between predicted and actual value for each example
- Square each difference
- Take the average

The model adjusts `m` and `b` to make MSE as small as possible. This process is called **minimising the loss.**

---

**When to use it:**

Linear regression works when:
- Your output is a continuous number (not a category)
- The relationship between input and output is roughly linear

It does not work well when the relationship is curved or complex. For that you need other algorithms.

---

## Real World Examples

| Use Case | Features | Label |
|---|---|---|
| House price prediction | Size, location, rooms | Price |
| Salary prediction | Years of experience, role | Salary |
| Sales forecasting | Ad spend, season | Revenue |
| Temperature forecasting | Month, location | Temperature |

---

## Code

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Dataset — house size vs price
X = np.array([[50], [80], [100], [120], [150]])   # size in sqm
y = np.array([150000, 220000, 280000, 340000, 400000])  # price in €

# Train the model
model = LinearRegression()
model.fit(X, y)

# What did the model learn?
print(f"Slope (m):     {model.coef_[0]:.2f}")      # how much price increases per sqm
print(f"Intercept (b): {model.intercept_:.2f}")     # base price

# Predict price for a 110 sqm house
prediction = model.predict([[110]])
print(f"Predicted price for 110sqm: €{prediction[0]:,.0f}")
```

---

## Summary

| Concept | Meaning |
|---|---|
| Linear Regression | Finds the best straight line through data |
| Features (x) | The inputs — what you know |
| Label (y) | The output — what you want to predict |
| Slope (m) | How much the output changes per unit of input |
| Intercept (b) | The baseline output when input is zero |
| Loss (MSE) | How wrong the model is — minimised during training |

---

*Part of my AI/ML notes series — one concept documented per day.*
*LinkedIn post → [Day 04 — Linear Regression]()*
