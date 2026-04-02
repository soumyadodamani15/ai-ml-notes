# Day 01 — What is AI? What is ML? What is the difference?

> **One line to remember:**
> AI is the goal. ML is one method to reach it. Deep Learning is one technique within ML. Data Science is the profession that uses all of them.

---

## Concept

These four terms are often used interchangeably. They are not the same thing.

**Artificial Intelligence (AI)** is the broadest idea — making machines behave smartly. Any technique that allows a computer to do something that would normally require human intelligence falls under AI. Chess engines, voice assistants, spam filters — all AI.

**Machine Learning (ML)** is one way to achieve AI. Instead of a programmer writing rules manually, you give the machine data and let it figure out the rules itself. The key shift: you provide examples, not instructions.

**Deep Learning (DL)** is a subset of ML — a specific technique using neural networks with many layers. It is what powers image recognition, large language models like ChatGPT, and voice recognition systems.

**Data Science** is different from the above three. It is not just about building intelligent systems — it is about extracting insights from data using statistics, visualisation, and ML as tools. A Data Scientist may use ML, or may simply analyse and present findings. It overlaps but is not the same thing.

---

## How They Relate

```
AI
└── ML
    └── Deep Learning

Data Science — intersects with all of the above
```

---

## Real World Examples

| Term | Real World Example |
|---|---|
| AI | Google Maps rerouting you around traffic in real time |
| ML | Netflix recommending your next show based on watch history |
| Deep Learning | FaceID recognising your face. ChatGPT generating text. |
| Data Science | Analysing why sales dropped in Q3 and presenting findings to leadership |

---

## The Key Insight — Rules vs Examples

**Traditional rule-based approach (not ML):**
```python
# You write every rule manually
def is_spam(email):
    if "free money" in email or "click here" in email:
        return True
    return False
```

**ML approach — machine learns the rules from data:**
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training data — examples with labels, not rules
emails = ["free money now", "meeting at 3pm", "click here to win", "project update"]
labels = [1, 0, 1, 0]  # 1 = spam, 0 = not spam

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

model = MultinomialNB()
model.fit(X, labels)  # model learns the rules itself

# Predict on new unseen data
test = vectorizer.transform(["win free cash"])
print(model.predict(test))  # → [1] spam
```

You gave it examples — not rules. That is the entire soul of ML.

---

## Summary

| Term | What it is | Goal |
|---|---|---|
| AI | The big idea | Make machines behave intelligently |
| ML | One method to achieve AI | Learn rules from data |
| Deep Learning | One technique within ML | Learn complex patterns via neural networks |
| Data Science | A profession that uses all of the above | Extract insights from data |

---

*Part of my AI/ML notes series — one concept documented per day.*
*LinkedIn post → [Day 01 — What is AI? What is ML?](https://www.linkedin.com/posts/soumya-dodamani_dotnet-ai-rag-share-7445497006179123200-ToZK)*
