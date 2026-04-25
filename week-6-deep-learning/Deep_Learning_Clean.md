# Machine Learning & Deep Learning — Master Reference
### For Engineers with a Java Background

---

## Concept Guide Index

This document is the **high-level master reference** covering all ML and Deep Learning concepts. For deep dives with interview questions on individual architectures, see:

| Concept | Detailed Guide | Notebook |
|---|---|---|
| ANN / MLP (Feedforward Networks) | [ANN_Comprehensive_Guide.md](ANN_Comprehensive_Guide.md) | [ANN_Neural_Network.ipynb](ANN_Neural_Network.ipynb) |
| RNN / LSTM / GRU (Sequences) | [RNN_Comprehensive_Guide.md](RNN_Comprehensive_Guide.md) | [RNN_SBI_Stock prediction.ipynb](RNN_SBI_Stock%20prediction.ipynb) |
| CNN (Images) | *Coming soon* | — |
| Transformers / LLMs | [Transformers_LLM_Comprehensive_Guide.md](../week-7-transformers-llm/Transformers_LLM_Comprehensive_Guide.md) | *Coming soon* |

---

> **How to read this document**
> Every concept includes a **Java Analogy** box. If a concept feels abstract,
> jump to the analogy first — it will anchor the idea in something familiar.
> This document covers all ML types AND deep learning in one place.

---

## Table of Contents

**Part 1 — Machine Learning Foundations**
1. [Introduction & History](#1-introduction--history)
2. [Types of Machine Learning](#2-types-of-machine-learning)
3. [Supervised Learning](#3-supervised-learning)
4. [Unsupervised Learning & K-Means](#4-unsupervised-learning--k-means)
5. [Semi-Supervised, Reinforcement & Self-Supervised Learning](#5-semi-supervised-reinforcement--self-supervised-learning)

**Part 2 — Deep Learning**
6. [Why Deep Learning Now?](#6-why-deep-learning-now)
7. [Neural Network Basics — Perceptron](#7-neural-network-basics--perceptron)
8. [How the Brain Analogy Works](#8-how-the-brain-analogy-works)
9. [Forward Propagation](#9-forward-propagation)
10. [Activation Functions](#10-activation-functions)
11. [Loss Function](#11-loss-function)
12. [Backward Propagation](#12-backward-propagation)
13. [Optimizers & Gradient Descent](#13-optimizers--gradient-descent)
14. [Training Terminology — Epochs, Batches, Iterations](#14-training-terminology--epochs-batches-iterations)
15. [Full Training Loop](#15-full-training-loop)
16. [Multi-Layered Neural Network](#16-multi-layered-neural-network)
17. [Vanishing & Exploding Gradients](#17-vanishing--exploding-gradients)
18. [Weight Initialization](#18-weight-initialization)
19. [Overfitting & Underfitting](#19-overfitting--underfitting)
20. [Regularization — Dropout, L1, L2](#20-regularization--dropout-l1-l2)
21. [Batch Normalization](#21-batch-normalization)
22. [Convolutional Neural Networks (CNN)](#22-convolutional-neural-networks-cnn)
23. [Recurrent Neural Networks (RNN), LSTM & GRU](#23-recurrent-neural-networks-rnn-lstm--gru)
24. [Transfer Learning](#24-transfer-learning)
25. [Model Evaluation Metrics](#25-model-evaluation-metrics)
26. [Data Preprocessing for Deep Learning](#26-data-preprocessing-for-deep-learning)
27. [Popular Frameworks](#27-popular-frameworks)
28. [Quick Reference Tables](#28-quick-reference-tables)

---

# PART 1 — MACHINE LEARNING FOUNDATIONS

---

## 1. Introduction & History

Machine Learning is a sub-field of AI where computers learn patterns from data rather than following hand-written rules. Deep Learning is a further sub-field of ML that uses multi-layered neural networks.

**The AI family tree:**
```
Artificial Intelligence (AI)
└── Machine Learning (ML)     ← algorithms that learn from data
    └── Deep Learning (DL)    ← multi-layer neural networks
        └── LLMs / GenAI      ← transformers trained at massive scale
```

**Timeline at a glance:**

| Year | Milestone |
|------|-----------|
| 1958 | Perceptron invented (Rosenblatt) |
| 1986 | Backpropagation popularised (Rumelhart, Hinton) |
| 1998 | LeNet — first practical CNN for digit recognition (LeCun) |
| 2012 | AlexNet wins ImageNet — deep learning era begins |
| 2014 | GANs invented (Goodfellow) |
| 2017 | Transformer architecture introduced ("Attention is All You Need") |
| 2022 | ChatGPT — large language models go mainstream |

---

## 2. Types of Machine Learning

ML is not one technique — it is a family of approaches. Choose based on what data you have, whether you have labels, and what you are predicting.

```
MACHINE LEARNING
│
├── 1. Supervised Learning        → Learns from labeled examples
│   ├── Classification            → Predicts a category
│   └── Regression                → Predicts a number
│
├── 2. Unsupervised Learning      → Finds hidden patterns, no labels
│   ├── Clustering                → Groups similar items
│   ├── Dimensionality Reduction  → Simplifies complex data
│   └── Anomaly Detection         → Finds unusual items
│
├── 3. Semi-Supervised Learning   → Mix of labeled + unlabeled data
│
├── 4. Reinforcement Learning     → Learns by reward and penalty
│
└── 5. Self-Supervised Learning   → Generates its own labels from data
```

**Side-by-side comparison:**

| Feature | Supervised | Unsupervised | Semi-Supervised | Reinforcement | Self-Supervised |
|---|---|---|---|---|---|
| Labels required | Yes, all data | None | Some | None | None (self-generated) |
| Human involvement | High | Low | Medium | Low | Low |
| Output type | Prediction | Patterns/Groups | Prediction | Actions/Policy | Representations |
| Cost of data | High | Low | Medium | Low | Low |
| Example | Spam detection | Customer grouping | Medical imaging | Game AI | GPT, Claude, Gemini |

**Decision tree for choosing ML type:**
```
START
  │
  ├── Do you have labeled training data?
  │     ├── YES → Is it ALL labeled?
  │     │           ├── YES → SUPERVISED LEARNING
  │     │           └── NO  → SEMI-SUPERVISED LEARNING
  │     └── NO  → Are you making sequential decisions?
  │                   ├── YES → REINFORCEMENT LEARNING
  │                   └── NO  → UNSUPERVISED or SELF-SUPERVISED
```

---

## 3. Supervised Learning

### The Core Idea

Every training example has both an **input** and a **correct answer (label)**. The model learns the input→answer mapping, then applies it to new unseen data.

> **Java Analogy:** Supervised learning is like studying with an answer key — you see the question AND the correct answer repeatedly until you can answer new questions on your own.

### The Two Tasks

#### Classification — Predict a Category

| Example | Input | Output |
|---|---|---|
| Email spam filter | Email text | Spam / Not Spam |
| Medical diagnosis | Patient symptoms | Disease / No Disease |
| Image recognition | Photo | Cat / Dog / Car |
| Sentiment analysis | Review text | Positive / Negative |

#### Regression — Predict a Number

| Example | Input | Output |
|---|---|---|
| House price prediction | Size, location, rooms | Price in dollars |
| Stock price prediction | Historical prices | Tomorrow's price |
| Temperature forecast | Weather patterns | Temperature (°C) |
| Sales forecasting | Past sales data | Next month's revenue |

### Common Supervised Learning Algorithms

| Algorithm | Best For | Complexity |
|---|---|---|
| Linear Regression | Numbers with linear relationships | Simple |
| Logistic Regression | Binary classification | Simple |
| Decision Tree | Classification and regression | Medium |
| Random Forest | Robust classification and regression | Medium |
| Support Vector Machine (SVM) | High-dimensional classification | Medium |
| Neural Network | Complex patterns, images, language | Complex |
| K-Nearest Neighbors (KNN) | Simple classification | Simple |

### Strengths and Weaknesses

| Strengths | Weaknesses |
|---|---|
| High accuracy with good labels | Requires labeled data (expensive) |
| Clear, measurable performance | Model only as good as the labels |
| Well-understood mathematically | Can overfit on small datasets |

---

## 4. Unsupervised Learning & K-Means

### The Core Idea

There are no labels. The algorithm receives raw data and discovers hidden patterns, structures, or groupings on its own.

> **Java Analogy:**
> ```
> Supervised   = a student with a textbook that has answers at the back
> Unsupervised = a student given 1000 documents with no answers — must
>                figure out which ones are similar and group them
> ```

### The Three Tasks

#### Clustering — Group Similar Items

The algorithm groups data points so that items within a group are more similar to each other than to items in other groups.

| Example | Data | Groups Found |
|---|---|---|
| Customer segmentation | Purchase behavior | Budget / Regular / Premium |
| Document grouping | News articles | Sports / Politics / Tech |
| Gene analysis | DNA sequences | Related genetic groups |

#### Dimensionality Reduction — Simplify Complex Data

Compresses many features into fewer while preserving the important information.

| Algorithm | Use Case |
|---|---|
| PCA (Principal Component Analysis) | Feature compression, noise removal |
| t-SNE | Data visualization in 2D/3D |
| Autoencoder | Feature learning via neural network |

#### Anomaly Detection — Find What Does Not Fit

Learns what "normal" looks like, then flags anything unusual.

| Example | Anomaly |
|---|---|
| Credit card fraud | Purchase in 3 countries in 1 hour |
| Network security | Unusual data transfer at 3 AM |
| Manufacturing | Defective item dimensions |

### Common Unsupervised Algorithms

| Algorithm | Task | Use Case |
|---|---|---|
| K-Means | Clustering | Customer segmentation |
| DBSCAN | Clustering | Geographic data, irregular shapes |
| Hierarchical Clustering | Clustering | Gene analysis, dendrograms |
| PCA | Dimensionality Reduction | Feature compression |
| t-SNE | Dimensionality Reduction | Visualization |
| Isolation Forest | Anomaly Detection | Fraud detection |

---

### K-Means Clustering — Deep Dive

**What K-Means Does:**
Automatically groups N data points into **K clusters** where each cluster contains the most similar points.

**The Shopping Mall Example:**
```
Given: income and spending data for 200 customers (no labels)
Ask K-Means: "Find me 5 natural groups"

K-Means returns:
  Group 1 → Low income,    Low spending   → Budget shoppers
  Group 2 → Low income,    High spending  → Impulsive shoppers
  Group 3 → Medium income, Medium spend   → Average shoppers
  Group 4 → High income,   Low spending   → Wealthy but careful
  Group 5 → High income,   High spending  → Premium customers

No one told it these groups existed — it found them from raw numbers.
```

**How K-Means Works Step by Step:**
```
Step 1 — Place K centroids randomly in the data space

Step 2 — Assign every data point to the nearest centroid
         (Euclidean distance: d = √( (x₂-x₁)² + (y₂-y₁)² ))

Step 3 — Move each centroid to the mean position of its members
         new_centroid = (mean of all x values, mean of all y values)

Step 4 — Repeat Steps 2 and 3 until no point changes cluster
         (convergence: inertia stops decreasing)
```

**Inertia / WCSS (Within Cluster Sum of Squares):**
```
Inertia = Σ (distance from each point to its centroid)²
```
Lower inertia = tighter clusters = better fit. Algorithm stops when inertia converges.

**The Elbow Method — Choosing K:**
```
Run K-Means with K = 1, 2, 3 ... 10. Plot inertia:

Inertia
  |*
  |  *
  |    *
  |      *  ← ELBOW here (optimal K)
  |         * * * * *
  +─────────────────── K
    1  2  3  4  5  6

Pick K at the elbow — where the curve bends and flattens.
After the elbow, adding more clusters gives diminishing returns.
```

**K-Means in Code:**
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)  # ALWAYS scale first

# Find best K
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Train with chosen K
model = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = model.fit_predict(X_scaled)
```

**K-Means Limitations:**

| Limitation | Workaround |
|---|---|
| You must choose K | Use Elbow Method or Silhouette Score |
| Sensitive to outliers | Remove outliers before clustering |
| Assumes round/spherical clusters | Use DBSCAN for irregular shapes |
| Random initialisation varies results | Set `random_state` for reproducibility |
| Scale-sensitive | Always apply `StandardScaler` first |

**K-Means vs Other Clustering:**

| Algorithm | How It Works | Use When |
|---|---|---|
| K-Means | Groups by distance to centroid | Clusters are spherical, K is known |
| DBSCAN | Groups by density | Clusters are irregular, K is unknown |
| Hierarchical | Builds a tree of clusters | You want to see all possible groupings |

> **Java Analogy:** K-Means is like a load balancer that partitions requests across K servers. It continuously reassigns requests (data points) to the nearest server (centroid) and moves servers to the centre of their traffic zone until nothing changes.

---

## 5. Semi-Supervised, Reinforcement & Self-Supervised Learning

### Semi-Supervised Learning

**The Core Idea:** A small amount of labeled data + a large amount of unlabeled data. The model learns from the labeled examples first, then uses that knowledge to make sense of the unlabeled data.

**When to use:** Labeling all data is too costly, but having no labels would lose too much accuracy.

| Domain | Labeled Data | Unlabeled Data |
|---|---|---|
| Medical imaging | 200 scans labeled by doctors | 50,000 unlabeled scans |
| Web content moderation | 1,000 reviewed posts | 10 million posts |
| Speech recognition | 100 hours transcribed | 10,000 hours of audio |

**How It Works:**
```
Step 1: Train on labeled data → Initial model
Step 2: Apply to unlabeled data → Get predictions with confidence scores
Step 3: Add high-confidence predictions as new labels (> 95% confidence)
Step 4: Retrain with expanded dataset → Better model
```

---

### Reinforcement Learning (RL)

**The Core Idea:** An **agent** takes actions in an **environment**. Good actions get a **reward**. Bad actions get a **penalty**. Through trial and error, the agent learns a **policy** that maximizes total reward.

**Key Components:**

| Component | What It Is | Example |
|---|---|---|
| Agent | The AI making decisions | The game player |
| Environment | The world the agent acts in | The game itself |
| State | What the agent currently observes | Current game screen |
| Action | What the agent chooses | Move left, jump, shoot |
| Reward | Score for an action outcome | +10 win, -5 lose |
| Policy | The learned strategy | "In this state, do this action" |

**The Learning Loop:**
```
Observe State → Choose Action → Environment Changes
      ↑                                │
      │                                ↓
Update Policy ← Learn from Reward ← Receive Reward/Penalty
```

**Real-World Examples:**

| Domain | Reward |
|---|---|
| Games (AlphaGo, Atari) | Game score |
| Self-driving cars | Safe driving |
| ChatGPT (RLHF) | Human approval rating |
| Trading bots | Profit |

**Key Algorithms:**

| Algorithm | Description |
|---|---|
| Q-Learning | Learns value of each action in each state |
| Deep Q-Network (DQN) | Q-Learning with a neural network |
| PPO (Proximal Policy Optimization) | Used to train ChatGPT with human feedback (RLHF) |
| AlphaZero | Mastered Go, Chess, Shogi through self-play |

> **Java Analogy:** RL is like a JVM JIT compiler — it starts with no knowledge of which code paths are hot, but repeatedly profiles execution (takes actions, gets rewards from the OS/hardware) and adapts its optimization policy to maximize throughput.

---

### Self-Supervised Learning

**The Core Idea:** The algorithm creates its own labels from the input data. No human labeling needed. The model is trained on tasks where the answer is hidden within the data itself.

**How It Works:**
```
Original Text:
  "The quick brown fox jumps over the lazy dog"

Self-Supervised Task (Masked Language Modeling):
  "The quick [MASK] fox jumps over the lazy dog"
  → Model predicts: "brown" ✓

Self-Supervised Task (Next Word Prediction):
  "The quick brown fox"
  → Model predicts: "jumps" ✓

After billions of such tasks:
  → Model deeply understands grammar, facts, and reasoning
```

**Why It Is Powerful:**
- Uses the entire internet as training data — no labeling cost
- Foundation for GPT, Claude, Gemini, BERT, LLaMA
- Produces models that generalize to almost any downstream task

| Model | Self-Supervised Task | Result |
|---|---|---|
| GPT series | Predict next word | ChatGPT |
| BERT | Predict masked words | Google Search improvements |
| Claude | Predict next word + RLHF | Conversational AI |
| SimCLR | Predict if two image crops are from same image | Image features without labels |

---

# PART 2 — DEEP LEARNING

---

## 6. Why Deep Learning Now?

Three forces converged to make Deep Learning practical:

1. **Data** — Hadoop, Spark, and cloud storage made it possible to collect and process massive datasets. Neural networks need large data to learn from.

2. **Hardware** — GPUs (originally built for gaming graphics) are ideal for the parallel matrix operations in neural networks, reducing training from weeks to hours.

3. **Algorithms** — Improvements in activation functions (ReLU), optimizers (Adam), and regularization techniques (Dropout) resolved many training problems from earlier decades.

> **Java Analogy:** Think of it like trying to run a Java EE application in 1995 — the language existed but the hardware, infrastructure (cloud), and frameworks (Spring) weren't mature. The same code that would have taken months now runs in minutes.

---

## 7. Neural Network Basics — Perceptron

The **Perceptron** is the simplest neural network — a single layer of neurons.

It takes input features, multiplies each by a **weight**, sums them, adds a **bias**, and passes the result through an **activation function** to produce output.

It is a **binary classifier** — the foundational building block of all deep networks.

```
┌─────────────────────────────────────────────────────────────┐
│                    PERCEPTRON STRUCTURE                      │
│                                                             │
│   x1 ──w1──►                                               │
│   x2 ──w2──► [Σ xiWi + bias] ──► Activation ──► ŷ (0/1)  │
│   x3 ──w3──►                                               │
│                                                             │
│   Inputs  →  Weighted Sum  →  Non-linearity  →  Prediction │
└─────────────────────────────────────────────────────────────┘
```

> **Java Analogy:** A Perceptron is like a single `if` statement:
> ```java
> // Hand-written rule (old way):
> if (glucose > 140 && bmi > 30) return "diabetic";
>
> // Perceptron (learned rule):
> double score = w1*glucose + w2*bmi + w3*age + bias;
> return sigmoid(score) > 0.5 ? "diabetic" : "healthy";
> ```
> The difference: you don't write the weights — the network **learns** them from data.

---

## 8. How the Brain Analogy Works

A newborn has no concept of "car" — it learns through repeated exposure and feedback. Neural networks are identical: they start with **random weights** and learn through training on labeled data.

**Biological vs Artificial Neuron:**

| Biological Neuron | Artificial Neuron |
|---|---|
| Dendrites (receive signals) | Input values (x1, x2, x3) |
| Synapse strength | Weight (w1, w2, w3) |
| Cell body (sums signals) | Weighted sum Σ(xᵢwᵢ) + bias |
| Fires or doesn't | Activation function output |
| Axon (sends signal forward) | Output to next layer |

---

## 9. Forward Propagation

Forward propagation feeds input data through the network from input layer to output layer to produce a prediction.

**At each neuron:**
1. Weighted sum: `z = Σ(xᵢwᵢ) + bias`
2. Apply activation: `ŷ = activation(z)`

The **bias** ensures the neuron can fire even when all inputs are zero — like the y-intercept in `y = mx + c`.

```
┌────────────────────────────────────────────────────────────────────┐
│                    FORWARD PROPAGATION FLOW                        │
│                                                                    │
│  Step 1          Step 2              Step 3          Step 4        │
│  Input         Weighted Sum          Activation      Output        │
│  Features ──► Σ(xᵢwᵢ) + bias  ──►  f(z)       ──►  ŷ             │
│  (x1,x2,x3)   = w·x + b           sigmoid/relu    (0 or 1)        │
└────────────────────────────────────────────────────────────────────┘
```

> **Java Analogy:** Forward propagation is a chain of method calls where each transforms data and passes it to the next:
> ```java
> double z      = weightedSum(inputs, weights, bias);
> double output = sigmoid(z);
> String label  = output > 0.5 ? "Yes" : "No";
> ```

---

## 10. Activation Functions

An activation function introduces **non-linearity**. Without it, stacking layers is mathematically equivalent to a single linear layer — no matter how deep, it could only learn straight lines.

```
┌────────────────────────────────────────────────────────────────────┐
│                   ACTIVATION FUNCTIONS COMPARISON                  │
│                                                                    │
│  SIGMOID              RELU               TANH                      │
│  f(x) = 1/(1+e⁻ˣ)    f(x) = max(0,x)   f(x) = (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)│
│  Range: (0, 1)        Range: [0, ∞)     Range: (-1, 1)            │
│                                                                    │
│  USE: Binary output   USE: Hidden layers  USE: Hidden layers, RNNs │
│  CON: Vanishing grad  CON: Dying ReLU     CON: Vanishing grad      │
│                                                                    │
│  LEAKY RELU: f(x) = max(0.01x, x)  → fixes dying ReLU             │
│  SOFTMAX:    converts raw scores → probabilities that sum to 1     │
│              USE: Multi-class output layer                         │
└────────────────────────────────────────────────────────────────────┘
```

**When to use which:**

| Layer | Recommended Activation | Reason |
|---|---|---|
| Hidden layers | ReLU (default) | Fast, avoids vanishing gradient |
| Hidden layers (alternative) | Leaky ReLU | If neurons are dying |
| Binary output | Sigmoid | Outputs probability 0–1 |
| Multi-class output | Softmax | Probabilities per class, sum to 1 |
| RNN hidden state | Tanh | Zero-centered, good for sequences |

**Softmax example:**
```
Raw scores:    [2.0, 1.0, 0.1]
After Softmax: [0.70, 0.26, 0.04]  ← probabilities, sum = 1.0
Prediction: Class 0 (70% confident)
```

> **Java Analogy:**
> - ReLU = `Math.max(0, value)`
> - Sigmoid = normalize to a 0–1 probability score
> - Softmax = `normalize()` that converts scores to percentage shares adding up to 100%

---

## 11. Loss Function

The loss function is the **report card** — it measures how far predictions are from the true labels. Training goal: **minimize this number**.

| Loss Function | Use Case | Formula |
|---|---|---|
| **MSE** (Mean Squared Error) | Regression | `½(y - ŷ)²` |
| **MAE** (Mean Absolute Error) | Regression, robust to outliers | `|y - ŷ|` |
| **Binary Cross-Entropy** | Binary classification | `-[y·log(ŷ) + (1-y)·log(1-ŷ)]` |
| **Categorical Cross-Entropy** | Multi-class classification | `-Σ yᵢ·log(ŷᵢ)` |
| **Sparse Categorical CE** | Multi-class (integer labels) | Same as above, labels as ints |

**Why Cross-Entropy instead of MSE for classification?**
MSE penalizes wrong predictions too gently near 0 and 1. Cross-Entropy penalizes **confident wrong answers** much more severely.

> **Java Analogy:**
> ```java
> double loss = Math.pow(actual - predicted, 2);  // MSE
> // loss == 0 → test passes; loss is high → fix the model (backprop)
> ```

---

## 12. Backward Propagation

Backpropagation is the learning mechanism. When loss is large, it figures out **which weights caused the error** and by exactly how much — then adjusts them.

It flows error **backwards** from output to input, computing the **gradient** (partial derivative) of the loss w.r.t. each weight using the **chain rule**.

**Chain Rule:**
```
∂Loss/∂w1 = (∂Loss/∂w3) × (∂w3/∂w2) × (∂w2/∂w1)
```

```
┌────────────────────────────────────────────────────────────────────┐
│                    BACKPROPAGATION FLOW                            │
│                                                                    │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐        │
│  │  Input  │    │Hidden 1 │    │Hidden 2 │    │ Output  │        │
│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘        │
│       ◄──── ∂L/∂w1 ──◄──── ∂L/∂w2 ──◄──── ∂L/∂w3 ──◄── Loss     │
│                                                                    │
│  Weight Update: W_new = W_old − η × (∂L/∂W_old)                  │
└────────────────────────────────────────────────────────────────────┘
```

> **Java Analogy:** Backpropagation is like reading a stack trace after an exception — the error is thrown at the output, and you trace it back layer by layer to find which weight caused the problem.

---

## 13. Optimizers & Gradient Descent

An optimizer **executes** the weight updates calculated by backprop.

**Gradient Descent** moves weights in the direction of the **negative gradient** — like a ball rolling toward the lowest valley (minimum loss).

```
Loss
  │  ● ← Start (high loss)
  │   \
  │    ●
  │     \
  │      ★ ← Global Minimum

Formula: W_new = W_old − η × (∂Loss / ∂W_old)

η too large → Overshoots minimum (diverges)
η too small → Painfully slow convergence
η ≈ 0.001  → Balanced convergence ✅
```

**Types of Gradient Descent:**

| Type | Data per Update | Speed | Stability |
|---|---|---|---|
| Batch GD | Full dataset | Slow | Very stable |
| Stochastic GD (SGD) | 1 sample | Fast but noisy | Unstable |
| Mini-Batch GD | Small batch (32–256) | Balanced | Balanced ✅ |

**Advanced Optimizers:**

| Optimizer | Key Idea | When to Use |
|---|---|---|
| SGD | Basic gradient descent | Simple baseline |
| SGD + Momentum | Adds velocity — rolls past small bumps | General use |
| RMSprop | Adapts learning rate per parameter | RNNs |
| Adam | Momentum + adaptive rate combined | **Default — use this** |
| AdaGrad | Large LR for rare features | Sparse data, NLP |

> **Java Analogy:**
> - **SGD** = take stairs, check every single step
> - **SGD + Momentum** = run downhill, carry speed from previous steps
> - **Adam** = elevator that auto-adjusts speed per floor — the go-to strategy

---

## 14. Training Terminology — Epochs, Batches, Iterations

| Term | Definition | Example |
|---|---|---|
| **Sample** | One training example | 1 image |
| **Batch (mini-batch)** | A small group of samples processed together | 32 images at once |
| **Iteration** | One forward + backward pass on one batch | Process batch → update weights once |
| **Epoch** | One full pass through the entire training dataset | All batches processed once |

**Worked example:**
```
Dataset:    10,000 training samples
Batch size: 100

Iterations per epoch = 10,000 / 100 = 100 iterations
After 1 epoch  → weights updated 100 times
After 50 epochs → weights updated 5,000 times
```

**How to choose epochs?**
- Too few → underfitting (model hasn't learned enough)
- Too many → overfitting (model memorizes training data)
- Use **early stopping**: monitor validation loss, stop when it stops improving

> **Java Analogy:**
> ```java
> for (int epoch = 0; epoch < 50; epoch++) {          // Epoch
>     for (List<Sample> batch : trainingBatches) {    // Iteration
>         double loss = forwardPass(batch);            // Forward prop
>         backwardPass(loss);                          // Backward prop
>         optimizer.updateWeights();                   // Weight update
>     }
> }
> ```

---

## 15. Full Training Loop

```
┌────────────────────────────────────────────────────────────────────┐
│                  COMPLETE NEURAL NETWORK TRAINING LOOP             │
│                                                                    │
│               FORWARD PROPAGATION                                  │
│  ① Input (x1, x2, x3)                                             │
│       ▼                                                            │
│  ② Multiply by Weights → Σ(xᵢwᵢ) + b                             │
│       ▼                                                            │
│  ③ Activation Function → ŷ                                        │
│                          ▼                                         │
│               BACKWARD PROPAGATION                                 │
│  ④ Loss = f(y, ŷ)  — how wrong is the prediction?                 │
│       ▼                                                            │
│  ⑤ Compute gradients via chain rule                               │
│       ▼                                                            │
│  ⑥ W_new = W_old − η·(∂L/∂W_old)   → Repeat                      │
└────────────────────────────────────────────────────────────────────┘
```

---

## 16. Multi-Layered Neural Network

A **Multi-Layer Perceptron (MLP)** stacks multiple hidden layers. Each layer learns progressively **abstract representations**:

- **Layer 1** — learns simple features (edges, common word pairs)
- **Layer 2** — combines simple features into shapes, phrases
- **Layer 3+** — combines shapes into objects, sentences into meaning

```
┌────────────────────────────────────────────────────────────────────┐
│                   MULTI-LAYERED NEURAL NETWORK                     │
│                                                                    │
│  Input        Hidden Layer 1   Hidden Layer 2      Output          │
│  Layer             (HL1)           (HL2)           Layer           │
│                                                                    │
│  x1 ──○──────────○──────────────○                                  │
│         \       / \             / \                                │
│  x2 ──○──\──── /   \───────── /   \──────○──── ŷ                  │
│           \  ○       ○───────○     ○                               │
│  x3 ──○────○                                                       │
│                                                                    │
│  Each arrow = a weight (learned during training)                   │
│  Each ○     = a neuron (applies activation function)               │
└────────────────────────────────────────────────────────────────────┘
```

> **Java Analogy:**
> ```java
> Object result = inputLayer
>     .transform(hiddenLayer1)   // learn low-level features
>     .transform(hiddenLayer2)   // combine into higher-level features
>     .transform(outputLayer);   // produce final prediction
> ```

---

## 17. Vanishing & Exploding Gradients

During backprop, gradients are multiplied layer by layer. In very deep networks:
- Gradients < 1 → multiply repeatedly → shrink to **zero** → early layers don't learn (vanishing)
- Gradients > 1 → multiply repeatedly → grow to **infinity** → NaN / training instability (exploding)

```
VANISHING:
  Layer 4 gradient: 0.5
  Layer 3 gradient: 0.5 × 0.5 = 0.25
  Layer 2 gradient: 0.25 × 0.5 = 0.125
  Layer 1 gradient: 0.125 × 0.5 = 0.0625  ← barely any update

EXPLODING:
  Layer 4 gradient: 2.0
  Layer 3 gradient: 2.0 × 2.0 = 4.0
  Layer 2 gradient: 4.0 × 2.0 = 8.0
  Layer 1 gradient: 8.0 × 2.0 = 16.0  ← NaN / inf
```

**Solutions:**

| Problem | Solution |
|---|---|
| Vanishing gradient | Use **ReLU** (gradient is 1 for positives) |
| Vanishing gradient | **Batch Normalization** |
| Vanishing gradient | **Residual connections** (skip connections, ResNet) |
| Exploding gradient | **Gradient clipping** — cap gradients at a threshold |
| Both | **LSTM** for recurrent networks |
| Both | Careful **weight initialization** |

> **Java Analogy:** Vanishing gradient = a message passed through 20 layers of management, each summarizing it shorter, until the original instruction is lost. Exploding = each layer exaggerates the message until it becomes noise.

---

## 18. Weight Initialization

Weights cannot start at **zero** — all neurons learn identically (symmetry problem). Weights cannot start too **large** — causes exploding gradients.

| Method | Formula | Use With |
|---|---|---|
| Random small values | N(0, 0.01) | Simple baseline (not great for deep nets) |
| **Xavier / Glorot** | N(0, 1/√n_in) | Tanh, Sigmoid activations |
| **He initialization** | N(0, 2/√n_in) | ReLU, Leaky ReLU activations |

**Why He for ReLU?** ReLU kills ~half the neurons (outputs 0 for negatives). He uses larger variance (`2/n` instead of `1/n`) so the remaining active neurons carry enough signal.

```python
# Keras handles this automatically:
Dense(128, activation='relu', kernel_initializer='he_normal')
```

> **Java Analogy:** Weight initialization is like choosing the starting seed for `Random`. Bad seed (all zeros = same seed for everyone) → every worker does identical work — useless. He/Xavier are carefully chosen seeds ensuring diverse, stable starting values.

---

## 19. Overfitting & Underfitting

**Underfitting** — model too simple. High training AND validation loss.
**Overfitting** — model memorizes training data. Low training loss, HIGH validation loss.

```
Training loss:     High         Low          Very Low
Validation loss:   High         Low          Very High
                Underfitting  Good Fit     Overfitting

Loss │
     │  ●●●●
     │       ●●●●
     │           ●●● ← Train loss still falling
     │               ●● ← Valid loss starts rising = overfitting
     └───────────── Epochs
                 ↑
       Best stopping point (early stopping)
```

**Solutions to Overfitting:**

| Technique | Description |
|---|---|
| Get more data | Most effective solution |
| **Dropout** | Randomly disable neurons during training |
| **L1 / L2 Regularization** | Penalize large weights |
| **Batch Normalization** | Stabilizes training, mild regularization |
| **Early Stopping** | Stop when validation loss stops improving |
| Reduce model complexity | Fewer layers / neurons |
| Data augmentation | Create synthetic variations of training data |

> **Java Analogy:** Overfitting = a developer who memorized every question from the company's interview bank but fails on any new question. Underfitting = studied only chapter titles — knows nothing in depth.

---

## 20. Regularization — Dropout, L1, L2

### Dropout

Randomly sets a fraction of neuron outputs to **zero** on each forward pass, forcing the network to learn redundant representations.

```
Without Dropout:              With Dropout (p=0.5):
○──○──○──○──○                 ○──○──X──○──X
All neurons active            ~50% randomly zeroed each batch
→ neurons co-adapt            → neurons learn independently
→ overfitting risk            → generalizes better
```

- `p = 0.5` → 50% dropped per training step
- **Only active during training** — turned off at inference
- At inference, outputs are scaled by `(1 - p)`

> **Java Analogy:** Dropout is like randomly removing team members from a sprint. Remaining members cover all tasks themselves — every member becomes capable of any job, rather than depending on specific partners.

### L2 Regularization (Weight Decay)

```
Total Loss = Original Loss + λ × Σ(w²)
```
Pushes all weights toward zero — prevents any single connection from dominating.

### L1 Regularization

```
Total Loss = Original Loss + λ × Σ|w|
```
Tends to produce **sparse models** — many weights become exactly zero. Useful for feature selection.

| | L1 | L2 |
|---|---|---|
| Penalty | Σ\|w\| | Σw² |
| Effect | Drives many weights to exactly 0 | Drives all weights toward 0 |
| Result | Sparse model (feature selection) | Smaller, spread-out weights |
| Use case | High-dimensional, few relevant features | General deep learning |

---

## 21. Batch Normalization

**The Problem:** As training progresses, the distribution of inputs to each layer shifts (**internal covariate shift**). Each layer is chasing a moving target.

**The Solution:** After computing weighted sum `z`, normalize to mean ≈ 0 and variance ≈ 1:
```
z_norm = (z - mean) / sqrt(variance + ε)
output = γ × z_norm + β     ← γ and β are learned scaling parameters
```

**Benefits:**
- Allows much **higher learning rates** → faster training
- Reduces sensitivity to weight initialization
- Acts as mild regularization
- Helps with vanishing gradients

> **Java Analogy:** Batch Normalization is like normalization middleware in a data pipeline — before passing data to the next stage, standardize it to a known range. Without it, each stage deals with wildly varying input distributions.

---

## 22. Convolutional Neural Networks (CNN)

**Used for:** Images, video, any grid-structured data.

**The Problem with MLP for images:**
A 224×224 RGB image has `224 × 224 × 3 = 150,528` input features. Connecting every pixel to every neuron is infeasible and ignores spatial structure (nearby pixels are related).

**CNN Solution — Convolution:**
A small **filter** (kernel), typically 3×3 or 5×5, slides across the image computing dot products — detecting specific local patterns (edges, corners, textures).

```
┌────────────────────────────────────────────────────────────────────┐
│                  CONVOLUTIONAL NEURAL NETWORK                      │
│                                                                    │
│  Input Image      Conv Layer        Pooling         Fully          │
│  (28×28×1)    →   (26×26×32)    →   (13×13×32)  →  Connected  →ŷ │
│                   32 filters        Max Pooling      (MLP)         │
│                   3×3 kernel        2×2 window                     │
│                                                                    │
│  Filter sliding over image:                                        │
│  Image patch:  1 0 1    Filter:  1 0 1                             │
│                0 1 0      ×      0 1 0   = dot product output      │
│                1 0 1             1 0 1   → "this pattern exists"   │
│                                                                    │
│  Architecture:                                                     │
│  INPUT → [CONV → ReLU → POOL] × N → FLATTEN → FC → SOFTMAX → ŷ  │
└────────────────────────────────────────────────────────────────────┘
```

**Key CNN Components:**

| Component | What it does |
|---|---|
| **Convolution layer** | Learns local patterns using filters |
| **ReLU** | Adds non-linearity after each conv |
| **Max Pooling** | Downsamples — keeps dominant feature, reduces spatial size |
| **Flatten** | Converts 2D feature maps to 1D vector for FC layer |
| **Fully Connected (FC)** | Final classification — standard MLP layer |

**What each layer learns (image example):**
```
Layer 1: Detects edges, corners, gradients
Layer 2: Detects shapes — circles, rectangles
Layer 3: Detects object parts — wheels, eyes
Layer 4: Detects full objects — cars, faces
```

**Popular CNN Architectures:**

| Architecture | Year | Key Innovation |
|---|---|---|
| LeNet | 1998 | First CNN for digit recognition |
| AlexNet | 2012 | Deep CNN, won ImageNet |
| VGGNet | 2014 | Very deep (16–19 layers) |
| ResNet | 2015 | Residual / skip connections (solved vanishing gradient) |
| EfficientNet | 2019 | Scaled width, depth, resolution together |

> **Java Analogy:** A CNN filter is like a custom `java.awt.image.ConvolveOp` kernel — a small matrix that slides over the image and detects specific patterns. Training a CNN means automatically *learning* the best filter values instead of hand-coding them.

---

## 23. Recurrent Neural Networks (RNN), LSTM & GRU

### Why RNN? — The Sequence Problem

**MLPs and CNNs have no memory** — each input is processed independently. For sequences, context matters:
- "I went to the **bank** to fish" vs "I went to the **bank** to withdraw" — same word, different meaning based on context
- A sentence of 5 words and a sentence of 50 words cannot share the same fixed-size MLP input
- Traditional NNs require fixed-size inputs — variable-length sequences require padding/truncation

**Why not just use a bigger MLP?**
- A sentence of 100 words × embedding size 300 = 30,000 input neurons — **parameter explosion**
- Standard NNs don't **share parameters across positions** — what learned at position 1 doesn't transfer to position 5. The network re-learns the same concept at every position
- Moving a word from position 3 to position 5 fools the network completely

**RNN Solution:**
- Processes **one token at a time**, regardless of sequence length
- Uses the **same weights at every step** (weight sharing across time)
- Passes a **hidden state** (memory) from each step to the next

---

### RNN Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         RNN UNROLLED                               │
│                                                                    │
│  Input:  "The   market  crashed  today"                            │
│                                                                    │
│   x₁        x₂        x₃        x₄                               │
│   "The"    "market"  "crashed"  "today"                           │
│    ↓          ↓         ↓         ↓                                │
│   [RNN] ──► [RNN] ──► [RNN] ──► [RNN] ──► ŷ (sentiment)         │
│    ↑           ↑         ↑         ↑                               │
│   h₀          h₁         h₂        h₃   ← hidden state carries   │
│                                           memory from past steps   │
└────────────────────────────────────────────────────────────────────┘
```

**Hidden State formula:**
```
hₜ = tanh(W_xh · xₜ + W_hh · hₜ₋₁ + b)
```
- `W_xh` = input-to-hidden weights (same at every step)
- `W_hh` = hidden-to-hidden weights (same at every step)
- `hₜ₋₁` = previous hidden state (memory from past)
- `b` = bias

The same `W_xh`, `W_hh`, and `b` are reused at every position — this is **weight sharing across time**, analogous to how a CNN filter slides across an image.

---

### RNN Sequence Architecture Types

| Type | Input → Output | Use Case |
|---|---|---|
| **Many-to-One** | Sequence → Single value | Sentiment analysis, document classification |
| **One-to-Many** | Single value → Sequence | Image captioning, music generation |
| **Many-to-Many (aligned)** | Sequence → Same-length sequence | POS tagging, NER, translation token-by-token |
| **Many-to-Many (unaligned)** | Sequence → Different-length sequence | Machine translation, time series forecasting |

**Many-to-One (Seq → Vec) — Sentiment Analysis:**
```
"The movie was absolutely terrible" (5 words)
       ↓ ↓ ↓ ↓ ↓
   [RNN RNN RNN RNN RNN] → final hidden state → [Classifier] → "Negative"

After reading all words, the RNN distills the entire sequence into one hidden
state vector, which feeds a classifier for a global judgment.
```

**One-to-Many (Vec → Seq) — Image Captioning:**
```
[CNN encodes image → feature vector]
           ↓ (seeds the hidden state)
      [RNN] → "A" → [RNN] → "dog" → [RNN] → "running" → ...

The vector seeds the RNN decoder, which generates tokens one at a time.
Each output word becomes the next input (autoregressive generation).
```

**Many-to-Many Aligned — POS Tagging:**
```
Input:   "The     cat     sat     on      the     mat"
          ↓        ↓       ↓       ↓       ↓       ↓
        [RNN] → [RNN] → [RNN] → [RNN] → [RNN] → [RNN]
          ↓        ↓       ↓       ↓       ↓       ↓
Output:  DET     NOUN    VERB    PREP    DET     NOUN

One output per input, positions correspond directly.
Also used for Named Entity Recognition (NER), Chunking.
```

**Many-to-Many Unaligned — Encoder-Decoder (Seq2Seq):**
```
Encoder:            Decoder:
"Bonjour monde"  →  context vector  →  "Hello world"
  (French, 2 words)                     (English, 2 words)

The encoder reads the full input and compresses it into a context vector.
The decoder reads the context vector and generates output word by word.
Input and output can have different lengths and different word orders.
```

Real-world applications:
- **Google Translate**: Encoder-Decoder Seq2Seq
- **Gmail Smart Compose**: predicts next words as you type
- **Time series forecasting**: 90 days in → predict 30 days out
- **Stock/sales forecasting**: RNN learns weekly cycles, momentum trends
- **Named Entity Recognition**: label each word as Person/Org/Location/Other

---

### RNN Problem — Vanishing Gradient + Short Memory

For long sequences, the hidden state "forgets" early context. The vanishing gradient problem hits RNNs especially hard — gradients multiply through many timesteps and shrink to zero, making early timesteps invisible to the loss.

---

### LSTM (Long Short-Term Memory)

LSTM solves the forgetting problem by introducing **gates** that explicitly control what information to remember, update, or forget:

```
┌────────────────────────────────────────────────────────────────────┐
│                      LSTM CELL INTERNALS                           │
│                                                                    │
│  Three gates (each is a sigmoid layer outputting 0–1):            │
│                                                                    │
│  FORGET GATE  : "What fraction of old memory to erase?"           │
│  INPUT GATE   : "What new information to add to memory?"           │
│  OUTPUT GATE  : "What part of memory to pass to next step?"        │
│                                                                    │
│  Cell State (Cₜ)  — the long-term memory "conveyor belt"          │
│  Hidden State (hₜ) — the short-term working memory                │
│                                                                    │
│  Forget: fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)                           │
│  Input:  iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)                           │
│  Output: oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)                           │
│  Cell:   Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙tanh(Wc·[hₜ₋₁,xₜ]+bc)              │
│  Hidden: hₜ = oₜ ⊙ tanh(Cₜ)                                      │
└────────────────────────────────────────────────────────────────────┘
```

**The Cell State** is a highway that runs through the entire sequence with only minor linear interactions — gradients can flow through many timesteps without vanishing, enabling learning of long-range dependencies.

---

### GRU (Gated Recurrent Unit)

A simplified LSTM with only 2 gates (forget + update merged into one update gate). Fewer parameters, trains faster, often comparable accuracy.

| Model | Gates | Memory Type | Speed | When to use |
|---|---|---|---|---|
| RNN | None | Short-term only | Fast | Short sequences only |
| LSTM | 3 gates | Long + short term | Slower | Long sequences, complex dependencies |
| GRU | 2 gates | Combined | Medium | Good balance, often preferred over LSTM |

> **Java Analogy:**
> - **RNN** = a `LinkedList` iterator that only remembers the previous node
> - **LSTM** = a `HashMap` with read/write/evict operations — you explicitly decide what to cache (remember) and what to evict (forget) at each step
> - **GRU** = a simplified cache with merged read/write policy

**Keras code:**
```python
from tensorflow import keras

# Simple RNN
model = keras.Sequential([
    keras.layers.SimpleRNN(64, return_sequences=True, input_shape=(timesteps, features)),
    keras.layers.SimpleRNN(32),
    keras.layers.Dense(1)
])

# LSTM
model = keras.Sequential([
    keras.layers.LSTM(128, return_sequences=True, input_shape=(timesteps, features)),
    keras.layers.LSTM(64),
    keras.layers.Dense(num_classes, activation='softmax')
])

# GRU
model = keras.Sequential([
    keras.layers.GRU(64, input_shape=(timesteps, features)),
    keras.layers.Dense(1, activation='sigmoid')
])
```

---

## 24. Transfer Learning

**The Idea:**
Train a large model on a massive dataset. The lower layers learn general features. Then **reuse those learned weights** for a different but related task — only retrain the last few layers.

```
┌────────────────────────────────────────────────────────────────────┐
│                     TRANSFER LEARNING                              │
│                                                                    │
│  Pre-trained model (e.g. ResNet-50 trained on ImageNet):          │
│  ┌──────────────────────────────────────┬──────────────┐          │
│  │  Conv layers (frozen — general       │  FC Head     │          │
│  │  features: edges, shapes, textures)  │  (retrained  │          │
│  │  ← These weights stay fixed          │  for your    │          │
│  │                                      │  task)       │          │
│  └──────────────────────────────────────┴──────────────┘          │
│                                                                    │
│  Your task: Classify X-ray images (only 500 images)               │
│  Solution:  Freeze conv layers, retrain only the output head      │
└────────────────────────────────────────────────────────────────────┘
```

**Strategies:**

| Strategy | When | What to Retrain |
|---|---|---|
| **Feature extraction** | Very small dataset, similar domain | Only output head (last layer) |
| **Fine-tuning** | Moderate dataset, different domain | Last few layers + output head |
| **Full retraining** | Large dataset, very different domain | Entire network |

**Popular Pre-trained Models:**

| Domain | Model | Pre-trained On |
|---|---|---|
| Images | ResNet-50, EfficientNet | ImageNet (1.2M images) |
| Text | BERT, GPT, RoBERTa | Wikipedia + BooksCorpus |
| Multimodal | CLIP, GPT-4V | Image-text pairs |

> **Java Analogy:**
> ```java
> // Instead of implementing authentication from scratch:
> public class MyApp extends SpringSecurityApp {  // = pretrained weights
>     @Override
>     public void configureForMyDomain() { ... }  // = retrain output head
> }
> ```

---

## 25. Model Evaluation Metrics

A single accuracy number is often misleading — especially for imbalanced datasets.

### Confusion Matrix

```
              Predicted +     Predicted -
  Actual +    TP (True Pos)   FN (False Neg)
  Actual -    FP (False Pos)  TN (True Neg)

TP = correctly predicted positive
TN = correctly predicted negative
FP = predicted positive but actually negative  (Type I error)
FN = predicted negative but actually positive  (Type II error)
```

### Key Metrics

| Metric | Formula | When it matters |
|---|---|---|
| **Accuracy** | (TP+TN) / Total | Balanced datasets |
| **Precision** | TP / (TP+FP) | When false positives are costly (spam filter) |
| **Recall** | TP / (TP+FN) | When false negatives are costly (cancer detection) |
| **F1 Score** | 2 × (P×R)/(P+R) | Imbalanced datasets |
| **AUC-ROC** | Area under ROC curve | Binary classification, threshold-independent |

**Which to optimize?**
```
Medical diagnosis (cancer):  → Maximize RECALL (missing a case = dangerous)
Spam filter:                 → Maximize PRECISION (blocking real email = bad)
Balanced use:                → Maximize F1 Score
```

### Validation Strategy

| Method | Description | When to use |
|---|---|---|
| **Train/Val/Test split** | 70/15/15 split | Large datasets |
| **k-Fold Cross Validation** | k=5 or 10 folds | Small/medium datasets |
| **Early Stopping** | Stop when val loss stops improving | Prevent overfitting |

> **Java Analogy:**
> - Precision = test suite pass rate (of tests that passed, how many were genuinely correct?)
> - Recall = test coverage (of all actual bugs, how many tests caught them?)
> - F1 = balanced coverage + correctness score

---

## 26. Data Preprocessing for Deep Learning

Neural networks are sensitive to input scale. Without preprocessing, a feature with range 0–10,000 (salary) will dominate a feature with range 0–1 (age fraction).

### Normalization vs Standardization

| Method | Formula | Result Range | When to use |
|---|---|---|---|
| **Min-Max Normalization** | (x - min) / (max - min) | [0, 1] | When you know the bounds |
| **Standardization (Z-score)** | (x - mean) / std | mean=0, std=1 | **Most DL cases — preferred** |

### Encoding Categorical Variables

| Method | Example | When |
|---|---|---|
| **Label Encoding** | `red=0, blue=1, green=2` | Ordinal categories (small, medium, large) |
| **One-Hot Encoding** | `red=[1,0,0], blue=[0,1,0]` | Nominal categories (no order) |
| **Embedding** | Dense vector learned during training | NLP (words), high-cardinality categories |

### Handling Images

```python
# Standard preprocessing for CNN input:
# 1. Resize to fixed size (e.g. 224×224)
# 2. Normalize pixel values from [0,255] to [0,1]
# 3. Subtract ImageNet mean per channel (for transfer learning)
image = image / 255.0
image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
```

### Data Augmentation

Artificially expand training set by creating variations:
```
Original image → flip → rotate → crop → brightness shift → add noise
```
All augmented versions are still the same class — teaches the model to be invariant to these transformations and reduces overfitting.

> **Java Analogy:** Normalization is like converting all monetary values to a single currency before doing any math. Without it, some values are in pence and some in billions — the math produces meaningless results.

---

## 27. Popular Frameworks

| Framework | By | Best For |
|---|---|---|
| **TensorFlow** | Google | Production, mobile deployment |
| **Keras** | Google | High-level API on top of TensorFlow, rapid prototyping |
| **PyTorch** | Meta | Research, flexible dynamic graphs |
| **JAX** | Google | Research, custom gradient computation |
| **ONNX** | Community | Export/import models across frameworks |

**TensorFlow/Keras example:**
```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(10,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
```

**PyTorch example:**
```python
import torch
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))
```

> **Java Analogy:**
> - **Keras** = Spring Boot — convention over configuration, fast to start
> - **PyTorch** = plain Java with full control — verbose but flexible
> - **TensorFlow** = Java EE — enterprise-grade, more boilerplate

---

## 28. Quick Reference Tables

### Activation Functions

| Function | Formula | Range | Use For | Problem |
|---|---|---|---|---|
| Sigmoid | 1/(1+e⁻ˣ) | (0, 1) | Binary output | Vanishing gradient |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | (-1, 1) | Hidden layers, RNNs | Vanishing gradient |
| ReLU | max(0, x) | [0, ∞) | Hidden layers (default) | Dying ReLU |
| Leaky ReLU | max(0.01x, x) | (-∞, ∞) | Hidden layers | — |
| Softmax | eˣᵢ/Σeˣⱼ | (0,1) sum=1 | Multi-class output | — |

### Loss Functions

| Loss | Task | Notes |
|---|---|---|
| MSE | Regression | Penalizes large errors heavily |
| MAE | Regression | More robust to outliers than MSE |
| Binary Cross-Entropy | Binary classification | Use with Sigmoid output |
| Categorical Cross-Entropy | Multi-class classification | Use with Softmax output |
| Sparse Categorical CE | Multi-class (integer labels) | Labels are ints, not one-hot |

### Optimizers

| Optimizer | Key Idea | When to Use |
|---|---|---|
| SGD | Basic gradient descent | Simple baseline |
| SGD + Momentum | Adds velocity | General training |
| RMSprop | Adaptive per-parameter LR | RNNs |
| Adam | Momentum + adaptive LR | **Default — use unless you have a reason not to** |
| AdaGrad | Large LR for rare features | Sparse / NLP |

### Architecture Selection Guide

| Data Type | Architecture | Why |
|---|---|---|
| Tabular / structured | MLP (Dense layers) | Simple, effective |
| Images | CNN | Exploits spatial locality |
| Text / sequences | RNN, LSTM, GRU, Transformer | Exploits sequential structure |
| Both image + text | Multimodal (CNN + Transformer) | — |
| Pre-existing large task | Transfer Learning | Save compute, need less data |

### Common Hyperparameters

| Hyperparameter | Typical Range | Effect |
|---|---|---|
| Learning rate | 0.0001 – 0.01 | Step size during gradient descent |
| Batch size | 16 – 512 | Samples per weight update |
| Epochs | 10 – 1000 | How long to train |
| Dropout rate | 0.2 – 0.5 | Fraction of neurons to randomly drop |
| Hidden layer size | 32 – 2048 | Model capacity |
| Number of layers | 1 – 100+ | Model depth |

### RNN Sequence Type Quick Reference

| Architecture | Input | Output | Example |
|---|---|---|---|
| Many-to-One | Sequence | Single value | Sentiment analysis, spam detection |
| One-to-Many | Single value | Sequence | Image captioning, music generation |
| Many-to-Many (aligned) | Sequence | Same-length sequence | POS tagging, NER |
| Many-to-Many (unaligned) | Sequence | Different-length sequence | Machine translation, forecasting |

### ML Type Quick Reference

| Situation | Recommended Type |
|---|---|
| Labeled historical data, want to predict outcomes | Supervised |
| Explore data and find hidden groups | Unsupervised |
| Lots of data but only some labeled | Semi-Supervised |
| AI learns a strategy through trial and error | Reinforcement |
| Train on raw text/images without human labels | Self-Supervised |

---

*Document covers: All 5 ML types, K-Means clustering, Perceptron, Forward/Back propagation, Activation functions, Loss functions, Optimizers, Training loop, Vanishing gradients, Weight initialization, Overfitting, Regularization, Batch Normalization, CNNs, RNNs/LSTMs/GRUs with all sequence types, Encoder-Decoder, Transfer Learning, Model Evaluation, Data Preprocessing, Frameworks — with Java analogies throughout.*
