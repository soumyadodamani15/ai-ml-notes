# Day 02 — Types of Machine Learning

> **One line to remember:**
> Supervised = learn from labelled examples. Unsupervised = find hidden patterns. Reinforcement = learn by trial, reward and penalty.

---

## Concept

Every ML problem falls into one of three core categories depending on what kind of data you have and what you want the machine to do.

> Note: There are also two modern variations — Semi-Supervised and Self-Supervised Learning — built on top of these three foundations. Those are covered separately in a later note.

---

**Supervised Learning**

You give the machine data that is already labelled. It learns by seeing examples with correct answers, then predicts answers for new data it has never seen.

Think of it like a student studying with an answer key. You show it 1000 emails labelled spam/not-spam. It learns the pattern. Now give it a new email — it predicts.

**Unsupervised Learning**

You give the machine data with no labels at all. No answer key. It has to find hidden patterns and groupings on its own.

Think of it like dropping someone into a library with no index. They start organising books by noticing similarities — same author, same topic, same size — without being told the categories.

**Reinforcement Learning**

No dataset at all. Instead, an agent learns by taking actions in an environment and receiving rewards or penalties. It figures out the best strategy over time through trial and error.

Think of it like training a dog. You do not explain the rules. It tries things — sits, gets a treat. Jumps on you, gets told off. Over thousands of tries it learns what works.

---

## How They Differ

| | Supervised | Unsupervised | Reinforcement |
|---|---|---|---|
| Data needed | Labelled | Unlabelled | No dataset |
| Learns from | Examples with answers | Hidden patterns | Rewards and penalties |
| Output | Prediction or classification | Groups or structure | A strategy or policy |

---

## Real World Examples

| Type | Real World Example |
|---|---|
| Supervised | Email spam filter, house price prediction, cancer detection from scans |
| Unsupervised | Customer segmentation, anomaly detection, topic clustering in documents |
| Reinforcement | AlphaGo mastering chess, self-driving cars, game-playing AI |

---

## Code

**Supervised — you provide labels:**
```python
from sklearn.linear_model import LogisticRegression

X = [[1], [2], [3], [4]]   # features
y = [0, 0, 1, 1]           # labels — YOU provide these

model = LogisticRegression()
model.fit(X, y)
print(model.predict([[2.5]]))  # → [1]
```

**Unsupervised — no labels, machine finds groups:**
```python
from sklearn.cluster import KMeans

X = [[1], [1.5], [3], [3.5]]  # no labels — machine figures it out

model = KMeans(n_clusters=2)
model.fit(X)
print(model.labels_)  # → [0 0 1 1] — it found 2 groups itself
```

**Reinforcement — reward-based learning (concept):**
```python
# Simplified idea — agent gets +1 for correct action, -1 for wrong action
def get_reward(action, correct_action):
    return +1 if action == correct_action else -1

# Agent tries actions repeatedly, learns which ones maximise total reward over time
```

---

## Summary

| Type | You provide | Machine does |
|---|---|---|
| Supervised | Labelled data | Learns to predict |
| Unsupervised | Unlabelled data | Finds hidden structure |
| Reinforcement | An environment + reward signal | Learns optimal behaviour |

---

*Part of my AI/ML notes series — one concept documented per day.*
*LinkedIn post → [Day 02 — Types of Machine Learning](https://www.linkedin.com/posts/activity-7445764885583552512-TzSx?utm_source=share&utm_medium=member_desktop&rcm=ACoAADIqRfgB9q57kOF3oEPWKkRR61-VN5IONck)*
