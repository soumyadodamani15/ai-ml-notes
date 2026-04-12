# Day 09 — Gradient Descent — How Models Actually Learn

> **One line to remember:**
> Measure how wrong you are. Figure out which direction makes it worse. Step the opposite way. Repeat until you reach the minimum.

---

## Concept

You have seen models "learn" from data in previous lessons. But what does learning actually mean under the hood? How does a model adjust itself to get better?

The answer is Gradient Descent.

---

**The intuition:**

Imagine you are blindfolded on a hilly landscape. Your goal is to find the lowest point — the valley. You cannot see anything.

You feel the slope under your feet. You take a small step in the direction that goes downhill. You repeat. Step by step, you get closer to the bottom.

That is gradient descent. The hill is the loss function — a measure of how wrong the model is. The lowest point is the minimum loss. The steps are parameter updates.

---

## The Three Components

**The Loss Function — the hill**

Measures how wrong the model's predictions are. For linear regression this is Mean Squared Error (MSE):

```
Loss = mean((predicted - actual)²)
```

Why square the error?
- Makes all errors positive — no cancellation between positive and negative errors
- Punishes large errors more heavily than small ones

Why take the mean?
- So the loss does not grow just because you have more examples

The higher the loss — the worse the model. Gradient descent minimises this number.

---

**The Gradient — the slope**

The gradient answers: "if I increase this parameter slightly, does the loss go up or down — and by how much?"

```
dm = mean(2 × (predicted - actual) × x)   # gradient for slope
db = mean(2 × (predicted - actual))        # gradient for intercept
```

- Positive gradient → increasing the parameter makes loss go up → decrease the parameter
- Negative gradient → increasing the parameter makes loss go down → increase the parameter

The gradient always points uphill. You want to go downhill — so you subtract it.

---

**The Learning Rate — the step size**

Controls how big each step is.

```
Too large  → overshoot the valley, bounce around, never converge
Too small  → take forever to reach the bottom
Just right → converge smoothly to the minimum
```

Typical values: 0.1, 0.01, 0.001. Finding the right one is one of the key skills in ML.

---

## The Parameter Update

```
new m = old m - (learning_rate × dm)
new b = old b - (learning_rate × db)
```

Subtract because the gradient points uphill. You want downhill. Each update makes m and b slightly more accurate.

---

## Step by Step — How a Model Learns

```
1. Start with random values for m and b
2. Make predictions using current m and b
3. Calculate loss — how wrong are the predictions?
4. Calculate gradients — which direction is uphill?
5. Update m and b — take a small step downhill
6. Repeat thousands of times
7. Loss converges — model has learned
```

---

## Three Variants

| Variant | Uses | Speed | Noise |
|---|---|---|---|
| Batch Gradient Descent | Entire dataset per step | Slow | Smooth |
| Stochastic GD (SGD) | One example per step | Fast | Very noisy |
| Mini-Batch GD | Small batch (32–64) per step | Balanced | Moderate |

Mini-Batch is the standard in practice — best of both worlds.

---

## Understanding the Formulas

**Why does MSE use (predicted - actual)²?**

Simple difference has a cancellation problem:
```
errors = [+1, -1, -2]
sum    = -2  ← looks small but model is wrong on everything
```

Squaring removes the cancellation and penalises large errors more.

**Where does the gradient formula come from?**

It is the derivative of MSE with respect to m and b — from calculus. You do not need to derive it manually. The key insight is:

```
dm = how much does loss change as m changes?
   = how wrong were we × how much did x contribute to that error
```

**Why subtract the gradient in the update?**

The gradient points uphill. Subtracting it means stepping downhill — toward lower loss.

---

## The Three Questions to Ask Any Formula

When you encounter any ML formula, always ask:

1. What is it measuring or adjusting?
2. Why this approach and not something simpler?
3. What breaks if I change it?

Applied to MSE:
1. Measuring how wrong the model is
2. Because simple difference has cancellation problems
3. Remove the square → errors cancel → model never improves properly

Applied to gradient:
1. Measuring which direction makes loss worse
2. So we know which way to step
3. Step in the same direction → climb uphill → model gets worse

Applied to learning rate:
1. Controls the step size
2. Full gradient step is too large and overshoots
3. Too large → never converge. Too small → takes forever.

---

## Real World Example

Training a house price model:

| Round | m | b | Loss |
|---|---|---|---|
| 1 | 0.0 | 0.0 | Huge — predicts €0 for everything |
| 100 | 1200 | 30000 | Getting better |
| 500 | 1900 | 48000 | Very close |
| 1000 | 2000 | 50000 | Converged — accurate predictions |

---

## Code

```python
import numpy as np

# Dataset — y = 2x
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

# Start with random parameters
m, b = 0.0, 0.0
learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    # Step 1 — make predictions
    y_pred = m * X + b

    # Step 2 — calculate loss (MSE)
    loss = np.mean((y_pred - y) ** 2)

    # Step 3 — calculate gradients
    dm = np.mean(2 * (y_pred - y) * X)
    db = np.mean(2 * (y_pred - y))

    # Step 4 — update parameters (step downhill)
    m -= learning_rate * dm
    b -= learning_rate * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d}: Loss = {loss:.4f}, m = {m:.4f}, b = {b:.4f}")

print(f"\nFinal: m = {m:.2f}, b = {b:.2f}")
# Converges to m ≈ 2.0, b ≈ 0.0
```

---

## Summary

| Concept | Meaning |
|---|---|
| Loss Function | Measures how wrong the model is |
| MSE | Average squared difference between predicted and actual |
| Gradient | Direction of steepest climb — we step the opposite way |
| Learning Rate | Controls step size — too large overshoots, too small is slow |
| Parameter Update | new = old - (learning_rate × gradient) |
| Epoch | One full pass through the training loop |
| Convergence | Loss stops improving — model has learned |

---

*Part of my AI/ML notes series — one concept documented per day.*
*LinkedIn post → [Day 09 — Gradient Descent](https://www.linkedin.com/posts/soumya-dodamani_ai-machinelearning-deeplearning-share-7448800687653740544-ChOH?utm_source=share&utm_medium=member_desktop&rcm=ACoAADIqRfgB9q57kOF3oEPWKkRR61-VN5IONck)*
