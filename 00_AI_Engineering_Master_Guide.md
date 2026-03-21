# AI Engineering — Master Reference Guide
### From Zero to Production-Ready Agentic AI Systems

> **Aligned to:** Credo Systemz — Generative AI Program (12 sections) + Agentic AI Program (36 sections)
> **Who this is for:** Anyone building toward an AI Engineering role — use this as your complete course companion.

---

## Table of Contents

### Part 1 — Foundations
| # | Section |
|---|---|
| 1 | [Why AI Matters](#1-why-ai-matters) |
| 2 | [History of AI](#2-history-of-ai) |
| 3 | [Core Concepts — AI vs ML vs DL vs GenAI](#3-core-concepts--ai-vs-ml-vs-dl-vs-genai) |
| 4 | [Types of Machine Learning](#4-types-of-machine-learning) |
| 5 | [NLP Fundamentals](#5-nlp-fundamentals) |
| 6 | [Python for AI Toolkit](#6-python-for-ai-toolkit) |

### Part 2 — Deep Learning & Models
| # | Section |
|---|---|
| 7 | [Neural Networks & Deep Learning](#7-neural-networks--deep-learning) |
| 8 | [Transformers & Attention](#8-transformers--attention) |
| 9 | [Generative Models — LLMs, Diffusion, GANs](#9-generative-models--llms-diffusion-gans) |
| 10 | [Large Language Models (LLMs)](#10-large-language-models-llms) |
| 11 | [Multimodal AI — Vision, Voice & Images](#11-multimodal-ai--vision-voice--images) |

### Part 3 — LLM Engineering
| # | Section |
|---|---|
| 12 | [Prompt Engineering](#12-prompt-engineering) |
| 13 | [OpenAI & LLM APIs](#13-openai--llm-apis) |
| 14 | [Hugging Face Ecosystem](#14-hugging-face-ecosystem) |
| 15 | [RAG — Retrieval-Augmented Generation](#15-rag--retrieval-augmented-generation) |
| 16 | [Fine-Tuning & RLHF](#16-fine-tuning--rlhf) |

### Part 4 — Agentic Systems
| # | Section |
|---|---|
| 17 | [AI Agents & Agentic Systems](#17-ai-agents--agentic-systems) |
| 18 | [LangChain](#18-langchain) |
| 19 | [LangGraph](#19-langgraph) |
| 20 | [AutoGen](#20-autogen) |
| 21 | [Multi-Agent Orchestration](#21-multi-agent-orchestration) |

### Part 5 — Data & ML Engineering
| # | Section |
|---|---|
| 22 | [Data Engineering Fundamentals](#22-data-engineering-fundamentals) |
| 23 | [Feature Engineering](#23-feature-engineering) |
| 24 | [Model Evaluation & Metrics](#24-model-evaluation--metrics) |
| 25 | [Vector Databases & Embeddings](#25-vector-databases--embeddings) |

### Part 6 — Production & MLOps
| # | Section |
|---|---|
| 26 | [MLOps & Production AI](#26-mlops--production-ai) |
| 27 | [Containerization with Docker](#27-containerization-with-docker) |
| 28 | [Kubernetes for AI](#28-kubernetes-for-ai) |
| 29 | [CI/CD for AI Pipelines](#29-cicd-for-ai-pipelines) |
| 30 | [Logging, Observability & Monitoring](#30-logging-observability--monitoring) |
| 31 | [Cloud Deployment — Azure & AWS](#31-cloud-deployment--azure--aws) |

### Part 7 — Toolkit, Ethics & Career
| # | Section |
|---|---|
| 32 | [Python Environment Setup](#32-python-environment-setup) |
| 33 | [AI Frameworks & Tools Ecosystem](#33-ai-frameworks--tools-ecosystem) |
| 34 | [Responsible AI & Ethics](#34-responsible-ai--ethics) |
| 35 | [AI Engineer Career Map](#35-ai-engineer-career-map) |
| 36 | [Master Vocabulary Reference](#36-master-vocabulary-reference) |

---

## 1. Why AI Matters

You use AI dozens of times every day without realising it:

| Your Daily Activity | AI Task | ML Type |
|---|---|---|
| Gmail auto-suggests a reply | Next-word prediction | Self-supervised (LLM) |
| Netflix recommends a show | Collaborative filtering | Supervised / Unsupervised |
| Spam goes to spam folder | Binary classification | Supervised |
| Phone unlocks with your face | Image recognition | Deep Learning (CNN) |
| Google Translate | Seq-to-seq translation | Transformers |
| ChatGPT answers questions | Language generation | LLM + RLHF |
| Google Maps reroutes you | Graph optimization | Reinforcement Learning |
| Credit card fraud blocked | Anomaly detection | Unsupervised |
| Siri / Alexa understands you | Speech-to-text | Deep Learning (RNN/Transformer) |
| Instagram curates your feed | Ranking & personalization | Supervised + RL |

---

## 2. History of AI

```
1950 ──── Turing asks "Can machines think?" — seeds the entire field
1956 ──── "Artificial Intelligence" coined at Dartmouth Conference
1960s-70s ── Rule-based Expert Systems: IF x THEN y
1980s ──── Expert Systems commercially deployed (narrow, rigid)
1986 ──── Backpropagation re-discovered → neural nets become trainable
1997 ──── Deep Blue defeats Kasparov at chess (brute force, not learning)
2006 ──── Hinton proves deep neural nets learn layered representations
2012 ──── AlexNet wins ImageNet — error rate halved — GPU era begins
2014 ──── GANs invented (Goodfellow) — AI can generate realistic images
2017 ──── "Attention Is All You Need" — Transformer architecture born
2018 ──── BERT (Google) — bidirectional language understanding
2019 ──── GPT-2 — large-scale coherent text generation
2020 ──── GPT-3 (175B params) — in-context few-shot learning emerges
2022 ──── ChatGPT — 100M users in 60 days, fastest tech adoption ever
2023 ──── GPT-4, Claude 2, Gemini — multimodal AI mainstream
2024 ──── Claude 3, GPT-4o — real-time voice + vision + 1M token context
2025 ──── Agentic AI, reasoning models (o3), AI coding assistants at scale
```

### Key Paradigm Shifts

| Era | Approach | Why It Broke |
|---|---|---|
| 1960s–1980s | Hand-written rules: `IF X THEN Y` | Can't write a rule for everything |
| 1980s–2000s | Statistical ML — patterns from data | Required manual feature engineering |
| 2012–2016 | Deep Learning — auto-extract features | Needed massive labeled datasets |
| 2017–2020 | Transformers — understand full context | Very expensive to train |
| 2020–Now | Foundation Models — train once, use everywhere | Alignment, hallucination, cost |

---

## 3. Core Concepts — AI vs ML vs DL vs GenAI

```
┌──────────────────────────────────────────────────────────────┐
│                   ARTIFICIAL INTELLIGENCE                     │
│          Any machine simulating human intelligence            │
│                                                              │
│   ┌──────────────────────────────────────────────────────┐  │
│   │                 MACHINE LEARNING                      │  │
│   │          Learns patterns from data                    │  │
│   │                                                       │  │
│   │   ┌────────────────────────────────────────────┐    │  │
│   │   │             DEEP LEARNING                   │    │  │
│   │   │        Uses neural networks                 │    │  │
│   │   │                                             │    │  │
│   │   │   ┌──────────────────────────────────┐    │    │  │
│   │   │   │         GENERATIVE AI             │    │    │  │
│   │   │   │     Creates new content           │    │    │  │
│   │   │   │                                   │    │    │  │
│   │   │   │  ┌────────────────────────────┐  │    │    │  │
│   │   │   │  │      AGENTIC AI            │  │    │    │  │
│   │   │   │  │  Plans & takes actions     │  │    │    │  │
│   │   │   │  └────────────────────────────┘  │    │    │  │
│   │   │   └──────────────────────────────────┘    │    │  │
│   │   └────────────────────────────────────────────┘    │  │
│   └──────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

| Term | Core Idea | Real Example |
|---|---|---|
| **AI** | Machine doing human-like tasks | Any smart software |
| **Machine Learning** | Learns rules from data | Spam filter, recommendation engine |
| **Deep Learning** | Multi-layer neural networks | Face ID, speech recognition |
| **Generative AI** | Creates new content | ChatGPT, DALL-E, GitHub Copilot |
| **Agentic AI** | Plans and executes multi-step tasks | AI travel agent, AutoGen workflows |

### Old Programming vs Machine Learning

```
Traditional Programming:    Rules + Data ──→ Answers
Machine Learning:           Data + Answers ──→ Rules (model)
                            Then: Rules + New Data ──→ New Answers
```

### AI vs GenAI — Key Differences

| Dimension | Traditional AI / ML | Generative AI |
|---|---|---|
| **Output** | Prediction, classification, number | New content (text, image, code, audio) |
| **Training** | Task-specific labeled data | Massive unlabeled data (self-supervised) |
| **Flexibility** | One task per model | One model, many tasks |
| **Examples** | Spam filter, fraud detection | ChatGPT, DALL-E, Copilot, Sora |
| **Evaluation** | Accuracy, F1, RMSE | Human preference, BLEU, perplexity |

---

## 4. Types of Machine Learning

```
MACHINE LEARNING
│
├── 1. Supervised Learning       ← Labeled examples: input + correct answer
│   ├── Classification           ← Predicts a category (spam / not spam)
│   └── Regression               ← Predicts a number (house price)
│
├── 2. Unsupervised Learning     ← No labels, finds hidden patterns
│   ├── Clustering               ← Groups similar items
│   ├── Dimensionality Reduction ← Compresses data while keeping meaning
│   └── Anomaly Detection        ← Finds unusual outliers
│
├── 3. Semi-Supervised           ← Small labeled + large unlabeled data
│
├── 4. Reinforcement Learning    ← Agent learns through reward / penalty
│
└── 5. Self-Supervised           ← Creates its own training signal from data
```

### 4.1 Supervised Learning

```
Training:   Input + Label ──→ Algorithm ──→ Learns mapping
Inference:  New Input ──→ Trained Model ──→ Prediction
```

**Classification Examples**

| Use Case | Input | Output |
|---|---|---|
| Email spam | Email text | Spam / Not Spam |
| Medical diagnosis | Patient data | Disease / Healthy |
| Sentiment analysis | Review text | Positive / Negative / Neutral |
| Image recognition | Photo pixels | Cat / Dog / Car |

**Regression Examples**

| Use Case | Input | Output |
|---|---|---|
| House price | Size, location, rooms | Dollar amount |
| Sales forecast | Past 12 months data | Next month revenue |
| Temperature | Weather sensor data | °C value |

**Common Algorithms**

| Algorithm | Complexity | Best For |
|---|---|---|
| Linear / Logistic Regression | Low | Baselines, interpretable |
| Decision Tree | Low–Med | Categorical data |
| Random Forest | Medium | Robust, general purpose |
| XGBoost / LightGBM | Medium–High | Tabular data, competitions |
| Neural Network | High | Images, text, complex patterns |

### 4.2 Unsupervised Learning

No labels. The algorithm finds structure on its own.

| Task | Algorithm | Example |
|---|---|---|
| Clustering | K-Means, DBSCAN | Customer segments |
| Dimensionality Reduction | PCA, t-SNE, UMAP | Visualize high-dim data |
| Anomaly Detection | Isolation Forest, Autoencoder | Fraud, defect detection |

### 4.3 Semi-Supervised Learning

```
500 labeled examples ──→ Train initial model
50,000 unlabeled examples ──→ Model labels with confidence
High-confidence labels ──→ Added to training set
Retrain ──→ Better model (effectively more labeled data)
```

### 4.4 Reinforcement Learning

```
Observe State ──→ Choose Action ──→ Environment Changes
      ↑                                      │
      └────── Update Policy ←── Reward/Penalty ◄──┘
```

| Component | Meaning | Game Example |
|---|---|---|
| Agent | The AI decision-maker | Game player |
| Environment | The world | The game engine |
| State | Current observation | Game screen |
| Action | Agent's choice | Move left, jump |
| Reward | Feedback signal | +10 win, -5 die |
| Policy | Learned strategy | "In state X, do Y" |

**Real Applications:** Self-driving cars · Trading bots · ChatGPT (RLHF phase) · AlphaGo

### 4.5 Self-Supervised Learning

```
Input:   "The quick brown fox jumps over the lazy dog"

Task 1 — Masked (BERT-style):
  "The quick [MASK] fox jumps" → predicts "brown"

Task 2 — Next Token (GPT-style):
  "The quick brown fox" → predicts "jumps"

After billions of predictions across the internet:
  → Model understands language, facts, code, reasoning
```

### 4.6 Comparison Table

| | Supervised | Unsupervised | Semi | Reinforcement | Self-Supervised |
|---|---|---|---|---|---|
| **Labels** | All | None | Some | None | Self-generated |
| **Human effort** | High | Low | Medium | Low | Low |
| **Example** | Spam filter | Clustering | Medical imaging | Game AI | GPT, Claude |

---

## 5. NLP Fundamentals

> **Natural Language Processing (NLP)** is the subfield of AI that enables computers to understand, process, and generate human language.

### The NLP Pipeline

```
Raw Text
  │
  ▼
Tokenization       → Split text into tokens (words, subwords)
  │
  ▼
Normalization      → Lowercase, remove punctuation, fix encoding
  │
  ▼
Stop Word Removal  → Remove "the", "is", "a" (low-info words)
  │
  ▼
Stemming / Lemmatization → Reduce to root form
  │
  ▼
Feature Extraction → Convert to numbers the model can use
  │
  ▼
Model / Task       → Classification, generation, translation
```

### Tokenization

```python
# Word tokenization
"Hello, world!" → ["Hello", ",", "world", "!"]

# Subword tokenization (used by GPT, BERT)
"unhappiness" → ["un", "happiness"]
"ChatGPT"     → ["Chat", "G", "PT"]

# Why subword? Handles rare words, new words, typos
```

### Stemming vs Lemmatization

| | Stemming | Lemmatization |
|---|---|---|
| **Approach** | Chop suffix (crude) | Use dictionary root |
| **"running"** | "run" | "run" |
| **"better"** | "better" | "good" |
| **"studies"** | "studi" | "study" |
| **Speed** | Fast | Slower |
| **Use case** | Search engines | NLP tasks needing correctness |

### Text Representations

#### Bag of Words (BoW)
```
Vocabulary: [cat, dog, sat, mat]
"The cat sat"  → [1, 0, 1, 0]
"The dog sat"  → [0, 1, 1, 0]

Problem: No word order, no meaning
```

#### TF-IDF (Term Frequency — Inverse Document Frequency)
```
TF  = (occurrences of word in document) / (total words in document)
IDF = log(total documents / documents containing word)
TF-IDF = TF × IDF

High TF-IDF = word is frequent in THIS doc but rare across all docs
→ Highlights document-specific important words
```

#### Word Embeddings
```
"king"   → [0.25, 0.84, -0.41, ...]  (300 dimensions)
"queen"  → [0.23, 0.82, -0.39, ...]  (similar vector)
"apple"  → [-0.71, 0.12, 0.55, ...]  (very different vector)

Famous relation: king − man + woman ≈ queen
```

| Embedding Model | Dimensions | Notes |
|---|---|---|
| Word2Vec | 100–300 | Static embeddings, 2013 |
| GloVe | 50–300 | Global co-occurrence statistics |
| FastText | 100–300 | Handles subwords, OOV words |
| BERT embeddings | 768 | Contextual — same word, different meaning |
| Sentence-BERT | 384–1024 | Sentence-level semantic similarity |

### Core NLP Tasks

| Task | Description | Example |
|---|---|---|
| **Sentiment Analysis** | Positive / Negative / Neutral | Product reviews |
| **Named Entity Recognition** | Find people, places, dates | "Apple announced in Cupertino" |
| **Text Classification** | Assign category to text | Spam filter, topic labeling |
| **Machine Translation** | Translate between languages | Google Translate |
| **Summarization** | Shorten while keeping key info | News headlines |
| **Question Answering** | Answer from a passage | RAG, chatbots |
| **Text Generation** | Generate coherent text | ChatGPT |
| **POS Tagging** | Label each word's grammar role | Noun, Verb, Adjective |

### Classical NLP vs Modern NLP

| Approach | Tools | Limitation |
|---|---|---|
| **Rule-based** | Regex, hand-crafted rules | Brittle, can't scale |
| **Classical ML** | TF-IDF + Logistic Regression | Feature engineering required |
| **Deep Learning** | RNNs, LSTMs | Struggles with long context |
| **Transformers** | BERT, GPT, T5 | Current state-of-the-art |

---

## 6. Python for AI Toolkit

### NumPy — Numerical Computing

```python
import numpy as np

# Create arrays
a = np.array([1, 2, 3, 4, 5])
matrix = np.zeros((3, 4))           # 3×4 matrix of zeros
rand = np.random.randn(100, 10)     # 100 samples, 10 features

# Operations (vectorized — much faster than Python loops)
a * 2           # [2, 4, 6, 8, 10]
np.dot(A, B)    # Matrix multiplication
a.mean()        # Average
a.std()         # Standard deviation
a.reshape(5,1)  # Change shape without changing data
```

### Pandas — Data Manipulation

```python
import pandas as pd

# Load data
df = pd.read_csv("data.csv")

# Explore
df.head()            # First 5 rows
df.info()            # Column types and nulls
df.describe()        # Statistical summary
df.shape             # (rows, columns)

# Select
df["column"]         # Single column (Series)
df[["col1","col2"]]  # Multiple columns (DataFrame)
df[df["age"] > 30]   # Filter rows

# Clean
df.dropna()                    # Remove rows with nulls
df.fillna(0)                   # Fill nulls with 0
df["col"].str.lower()          # Lowercase text
df.drop_duplicates()           # Remove duplicate rows

# Group
df.groupby("category")["price"].mean()  # Average price per category

# Export
df.to_csv("output.csv", index=False)
```

### Regex — Pattern Matching for Text

```python
import re

text = "Contact us at info@example.com or call +91-9884412301"

# Find all email addresses
emails = re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', text)

# Find phone numbers
phones = re.findall(r'\+?\d[\d\s-]{8,}', text)

# Replace patterns
clean = re.sub(r'http\S+', '', text)   # Remove URLs
clean = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters

# Common patterns
r'\d+'          # One or more digits
r'\w+'          # One or more word characters
r'\s+'          # One or more whitespace
r'^Hello'       # Starts with "Hello"
r'end$'         # Ends with "end"
```

### REST APIs & JSON

```python
import requests
import json

# GET request
response = requests.get("https://api.example.com/data",
                         headers={"Authorization": "Bearer YOUR_TOKEN"})

# Check status
if response.status_code == 200:
    data = response.json()      # Parse JSON response
    print(data["result"])

# POST request (e.g., calling an LLM API)
payload = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello"}]
}
response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json=payload
)

# Working with JSON
json_string = json.dumps(data, indent=2)   # dict → JSON string
data = json.loads(json_string)             # JSON string → dict
```

### Web Scraping

```python
import requests
from bs4 import BeautifulSoup

# Fetch a webpage
url = "https://example.com/articles"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Find elements
title = soup.find("h1").text
all_links = soup.find_all("a", href=True)
articles = soup.select(".article-title")   # CSS selector

# Extract text
for article in articles:
    print(article.get_text(strip=True))

# Save results
data = [a.text for a in articles]
pd.DataFrame(data, columns=["title"]).to_csv("articles.csv")
```

---

## 7. Neural Networks & Deep Learning

### The Neuron

```
Inputs: x1, x2, x3
          │
          ▼
  Weighted Sum: z = w1·x1 + w2·x2 + w3·x3 + bias
          │
          ▼
  Activation Function: output = f(z)
```

### Activation Functions

| Function | Formula | Use Case |
|---|---|---|
| **ReLU** | max(0, x) | Hidden layers — default choice |
| **Sigmoid** | 1/(1+e⁻ˣ) | Binary output (0 to 1) |
| **Softmax** | eˣⁱ / Σeˣʲ | Multi-class output (probabilities) |
| **Tanh** | (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ) | Centered, zero-mean outputs |
| **GELU** | x·Φ(x) | Transformers (BERT, GPT) |

### Neural Network Architectures

| Architecture | Abbreviation | Best For |
|---|---|---|
| Feedforward / Dense | MLP | Tabular data |
| Convolutional Neural Net | CNN | Images, spatial patterns |
| Recurrent Neural Net | RNN / LSTM / GRU | Sequences, time series |
| Transformer | — | Language, vision, multimodal |
| Autoencoder | AE | Compression, anomaly detection |
| GAN | GAN | Image generation |
| Diffusion Model | — | High-quality image/video generation |

### Training Loop

```
Step 1: Forward Pass
  Input → Layers → Prediction (ŷ)

Step 2: Compute Loss
  Loss = compare ŷ to true label y
  Classification: Cross-Entropy Loss
  Regression: MSE / MAE

Step 3: Backpropagation
  Compute gradient of loss w.r.t. every weight
  Chain rule applied layer by layer

Step 4: Weight Update (Gradient Descent)
  weight = weight − learning_rate × gradient

Repeat for all batches × all epochs → model converges
```

### Key Hyperparameters

| Hyperparameter | Effect | Typical Range |
|---|---|---|
| **Learning Rate** | Step size in gradient descent | 1e-5 to 1e-1 |
| **Batch Size** | Samples per update step | 16–2048 |
| **Epochs** | Passes through training data | 10–1000 |
| **Dropout Rate** | Regularization — prevents overfitting | 0.1–0.5 |
| **Weight Decay** | L2 regularization | 1e-5 to 1e-2 |

### Overfitting vs Underfitting

```
              Training Acc    Validation Acc    Problem
Underfitting:    Low              Low           Model too simple
Good fit:        High             High          Just right
Overfitting:     Very High        Low           Model memorized data

Fix Underfitting: More features, more complex model, more epochs
Fix Overfitting:  Dropout, more data, L2 regularization, early stopping
```

---

## 8. Transformers & Attention

### The Problem Transformers Solved

RNNs processed text sequentially (word by word) — early context was forgotten by the end.

Transformers process the **entire sequence at once** and learn which parts to pay attention to.

### Self-Attention

```
Input: "The bank can guarantee deposits will cover future tuition costs"

For the word "bank":
  → Attends heavily to "deposits", "guarantee", "cover"
  → Result: "bank" = financial institution (not a river bank)

Each word attends to every other word — full context captured
```

**Math:**
```
Q (Query)  = "What am I looking for?"
K (Key)    = "What does each word offer?"
V (Value)  = "What information does each word carry?"

Attention(Q,K,V) = softmax(QKᵀ / √d_k) × V
```

### Transformer Architecture

```
Input Tokens
     │
     ▼
Embeddings + Positional Encoding
     │
     ▼
┌────────────────────────────────┐
│  Encoder (BERT-style)          │  ← Sees full sequence
│  Multi-Head Self-Attention     │
│  Feed-Forward Network          │
│  Layer Norm + Residuals        │
└────────────────────────────────┘
     │
     ▼
┌────────────────────────────────┐
│  Decoder (GPT-style)           │  ← Left-to-right only
│  Masked Self-Attention         │
│  Cross-Attention (optional)    │
│  Feed-Forward Network          │
└────────────────────────────────┘
     │
     ▼
Output Probabilities → next token
```

### Model Type Comparison

| Type | Examples | Training | Best For |
|---|---|---|---|
| **Encoder-only** | BERT, RoBERTa | Masked LM | Classification, embeddings, search |
| **Decoder-only** | GPT, Claude, LLaMA | Next token | Text generation, chat |
| **Encoder-Decoder** | T5, BART, mT5 | Seq2Seq | Translation, summarization |

---

## 9. Generative Models — LLMs, Diffusion, GANs

### Taxonomy of Generative Models

| Model | What It Generates | How |
|---|---|---|
| **LLM (Transformer)** | Text, code, reasoning | Predict next token auto-regressively |
| **Diffusion Model** | Images, video, audio | Learn to reverse a noise-adding process |
| **GAN** | Images, video | Generator vs Discriminator adversarial game |
| **VAE** | Images, latent representations | Encode to distribution, decode samples |

### Diffusion Models

```
Forward Process (training):
  Real image ──→ Add noise step by step ──→ Pure noise

Reverse Process (generation):
  Pure noise ──→ Predict & remove noise step by step ──→ Generated image

The model learns to denoise — trained to predict the noise added at each step
```

| Model | Company | Use |
|---|---|---|
| Stable Diffusion | Stability AI | Open-source image generation |
| DALL-E 3 | OpenAI | Text-to-image via API |
| Midjourney | Midjourney | High-quality artistic images |
| Sora | OpenAI | Text-to-video generation |
| AudioCraft | Meta | Music and audio generation |

### GANs (Generative Adversarial Networks)

```
Generator          Discriminator
    │                   │
Creates fake  ──→  Tries to tell real
 images             from fake

Generator trains to fool Discriminator
Discriminator trains to detect fakes
Both improve together through competition

Result: Generator creates increasingly realistic outputs
```

**GAN Applications:** Deepfakes · Style transfer · Image-to-image translation · Super-resolution · Data augmentation

### Diffusion vs GAN vs LLM

| | GAN | Diffusion | LLM |
|---|---|---|---|
| **Training** | Adversarial (unstable) | Stable, simpler | Auto-regressive |
| **Quality** | Good, can have artifacts | Very high quality | N/A (text) |
| **Diversity** | Mode collapse risk | High diversity | High diversity |
| **Speed** | Fast inference | Slow (many steps) | Medium |
| **Best for** | Face generation | Photorealistic images | Text, code |

---

## 10. Large Language Models (LLMs)

### Scale

```
Model Size      Parameters    Capabilities
Tiny            7M            Simple tasks
Small           7B            Many general tasks
Medium          70B           Strong reasoning
Large           400B+         Frontier capabilities

Training Cost:   $10M – $100M+ for frontier models
Training Data:   Trillions of tokens from web, books, code
```

### LLM Training Stages

```
Stage 1 — Pre-training (Self-Supervised)
  Dataset: Entire internet + books + code
  Objective: Predict next token
  Result: Base model with broad knowledge

Stage 2 — Supervised Fine-Tuning (SFT)
  Dataset: (instruction, ideal response) pairs
  Result: Model follows instructions

Stage 3 — RLHF
  Dataset: Human preference comparisons
  Result: Helpful, harmless, honest assistant
```

### Tokenization

```
Text:    "Hello, world!"
Tokens:  ["Hello", ",", " world", "!"]
IDs:     [15496, 11, 995, 0]

~1 token ≈ 0.75 words in English
~1 token ≈ 0.5 words in code (more tokens per word)
```

### Context Window

| Context Size | Approx Pages | Use Case |
|---|---|---|
| 4K tokens | ~3 pages | Basic chat |
| 32K tokens | ~25 pages | Long documents |
| 128K tokens | ~100 pages | Full reports, codebases |
| 200K+ tokens | ~150+ pages | Claude 3.5, Gemini 1.5 |

### Key Capabilities

| Capability | Description | Example |
|---|---|---|
| **Zero-shot** | No examples needed | "Translate: Bonjour" |
| **Few-shot** | 1–5 examples given | Q: ... A: ... pattern |
| **Chain-of-Thought** | Step-by-step reasoning | "Let's think step by step..." |
| **Function Calling** | Model outputs structured JSON | Call a weather API |
| **Streaming** | Returns tokens as generated | Live typing in chat |

### Major LLM Families (2025)

| Model | Company | Strengths |
|---|---|---|
| GPT-4o | OpenAI | Multimodal, fast, widely integrated |
| Claude 3.5 / 4.x | Anthropic | Long context, safety, coding |
| Gemini 1.5 / 2.0 | Google | 1M+ token context, multimodal |
| LLaMA 3.x | Meta | Open-source, self-hostable |
| Mistral / Mixtral | Mistral AI | Efficient open-source |
| DeepSeek-R1 | DeepSeek | Strong reasoning, efficient |

---

## 11. Multimodal AI — Vision, Voice & Images

### What is Multimodal AI?

AI that processes and generates across multiple modalities — text, images, audio, video — in a single model.

```
Input types:   Text · Images · Audio · Video · Code
Output types:  Text · Images · Audio · Video · Code

GPT-4o:   Text + Image in → Text + Image out
Gemini:   Text + Image + Audio + Video in → Text out
Claude:   Text + Image in → Text out
Whisper:  Audio in → Text out (transcription)
DALL-E:   Text in → Image out
Sora:     Text in → Video out
```

### OpenAI DALL-E — Text to Image

```python
from openai import OpenAI
client = OpenAI()

response = client.images.generate(
    model="dall-e-3",
    prompt="A futuristic city skyline at sunset, digital art",
    size="1024x1024",
    quality="standard",
    n=1,
)
image_url = response.data[0].url
```

### OpenAI Whisper — Speech to Text

```python
from openai import OpenAI
client = OpenAI()

audio_file = open("speech.mp3", "rb")
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file
)
print(transcript.text)
```

### Vision — Image Understanding

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url",
             "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }]
)
print(response.choices[0].message.content)
```

---

## 12. Prompt Engineering

### Anatomy of a Good Prompt

```
┌──────────────────────────────────────────────────────────┐
│  SYSTEM PROMPT                                           │
│  "You are a senior data scientist. Be precise and        │
│  use Python examples where relevant."                    │
├──────────────────────────────────────────────────────────┤
│  CONTEXT                                                 │
│  "We have 10,000 customer transactions from 2024..."     │
├──────────────────────────────────────────────────────────┤
│  INSTRUCTION                                             │
│  "Identify the top 3 features correlated with churn."    │
├──────────────────────────────────────────────────────────┤
│  OUTPUT FORMAT                                           │
│  "Return a numbered list with one-sentence rationale."   │
└──────────────────────────────────────────────────────────┘
```

### Core Techniques

**Zero-Shot**
```
Prompt: "Classify this as Positive or Negative:
         'The battery dies after 2 hours.'"
Output: "Negative"
```

**Few-Shot**
```
Prompt: "Examples:
         'Great product!' → Positive
         'Stopped working.' → Negative
         'Decent but overpriced.' → ?"
Output: "Neutral"
```

**Chain-of-Thought (CoT)**
```
Prompt: "A store has 15 apples. They sell 7 and get 12 more.
         How many apples? Let's think step by step."
Output: "15 - 7 = 8. Then 8 + 12 = 20. Answer: 20 apples."
```

**ReAct (Reason + Act)**
```
Thought: I need current population of Tokyo.
Action: web_search("Tokyo population 2025")
Observation: 37.4 million
Answer: Tokyo's population is approximately 37.4 million.
```

### Best Practices

| Practice | Example |
|---|---|
| Be specific | "Summarize in 3 bullet points" not "summarize" |
| Assign a role | "You are an expert Python developer..." |
| Specify format | "Return as JSON with keys: name, score, reason" |
| Use delimiters | Wrap user content in `<input>...</input>` |
| Avoid negatives | "Write formally" not "Don't write casually" |
| Iterate | Treat prompts like code — version and test them |

### Advanced Patterns

| Pattern | When to Use |
|---|---|
| **System prompt persona** | Consistent tone and domain expertise |
| **Step-back prompting** | Ask for principles before solving |
| **Self-consistency** | Sample N outputs, take majority vote |
| **Least-to-most** | Break complex problem into sub-problems |
| **Prompt chaining** | Output of one prompt feeds into the next |
| **Structured output** | Force JSON schema via function calling |

---

## 13. OpenAI & LLM APIs

### Chat Completions API

```python
from openai import OpenAI
client = OpenAI(api_key="YOUR_API_KEY")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain transformers simply."}
    ],
    max_tokens=500,
    temperature=0.7     # 0 = deterministic, 1+ = creative
)
print(response.choices[0].message.content)
```

### Key Parameters

| Parameter | What It Controls | Range |
|---|---|---|
| **temperature** | Randomness of output | 0 (deterministic) to 2 (very random) |
| **max_tokens** | Max length of response | 1 to model limit |
| **top_p** | Nucleus sampling cutoff | 0 to 1 |
| **stream** | Return tokens as generated | True / False |
| **n** | Number of responses to generate | 1 to N |

### Function Calling (Structured Output)

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Chennai?"}],
    tools=tools,
    tool_choice="auto"
)
# Model returns structured JSON to call get_weather(city="Chennai")
```

### Token Management

```python
import tiktoken

# Count tokens before sending
enc = tiktoken.encoding_for_model("gpt-4o")
tokens = enc.encode("Hello, how are you?")
print(f"Token count: {len(tokens)}")

# Estimate cost
# gpt-4o: ~$5 per 1M input tokens, ~$15 per 1M output tokens
cost = (input_tokens / 1_000_000 * 5) + (output_tokens / 1_000_000 * 15)
```

### Anthropic Claude API

```python
import anthropic
client = anthropic.Anthropic(api_key="YOUR_KEY")

message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system="You are a helpful AI assistant.",
    messages=[{"role": "user", "content": "Explain RAG in simple terms."}]
)
print(message.content[0].text)
```

---

## 14. Hugging Face Ecosystem

> Hugging Face is the GitHub of AI — a hub for models, datasets, and spaces. It hosts 500,000+ models.

### Core Libraries

| Library | Purpose |
|---|---|
| `transformers` | Load and run pre-trained models |
| `datasets` | Access 70,000+ datasets |
| `tokenizers` | Fast tokenization |
| `peft` | Efficient fine-tuning (LoRA, QLoRA) |
| `accelerate` | Multi-GPU and distributed training |
| `diffusers` | Diffusion model inference |

### Loading and Using Models

```python
from transformers import pipeline

# Sentiment analysis
classifier = pipeline("sentiment-analysis")
result = classifier("I love building AI systems!")
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Text generation
generator = pipeline("text-generation", model="gpt2")
output = generator("Once upon a time", max_length=50)

# Summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary = summarizer(long_article, max_length=130, min_length=30)

# Translation
translator = pipeline("translation_en_to_fr",
                       model="Helsinki-NLP/opus-mt-en-fr")
result = translator("Hello, how are you?")
```

### Tokenization

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Tokenize
inputs = tokenizer("Hello world", return_tensors="pt")
# {'input_ids': tensor([[101, 7592, 2088, 102]]),
#  'attention_mask': tensor([[1, 1, 1, 1]])}

# Get embeddings
outputs = model(**inputs)
embeddings = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
```

### Fine-Tuning with PEFT (LoRA)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8b")

# Configure LoRA
lora_config = LoraConfig(
    r=16,               # Rank — lower = fewer params
    lora_alpha=32,      # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.1
)

# Wrap model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 8,388,608 || all params: 8,038,440,960 || ~0.1%
```

### Deploying to Hugging Face Spaces

```bash
# Create a Streamlit app
# app.py → deploys as a live demo on spaces.huggingface.co

# Gradio (simplest)
import gradio as gr

def predict(text):
    result = classifier(text)
    return result[0]["label"]

gr.Interface(fn=predict, inputs="text", outputs="text").launch()
```

---

## 15. RAG — Retrieval-Augmented Generation

### The Problem RAG Solves

```
Pure LLM Problems:           RAG Solutions:
  ✗ Knowledge cutoff           ✓ Retrieves fresh info at query time
  ✗ Hallucination              ✓ Grounds answer in real documents
  ✗ No private data access     ✓ Works with your own data
  ✗ Can't cite sources         ✓ Returns source references
```

### RAG Pipeline

```
INDEXING (one-time):
Documents → Chunking → Embedding → Vector Database

QUERY (real-time):
User Question
     │
     ▼
Embed the question
     │
     ▼
Search Vector DB → Top-K similar chunks
     │
     ▼
Build prompt: System + Context chunks + User question
     │
     ▼
LLM generates grounded answer
```

### Chunking Strategies

| Strategy | How | Best For |
|---|---|---|
| Fixed-size | Split every N characters | Baseline, simple |
| Sentence-level | Split at sentence boundaries | Preserves grammar |
| Paragraph-level | Split at paragraph breaks | Preserves topic |
| Semantic | Split when topic changes | Best retrieval quality |
| Hierarchical | Store both summary + detail chunks | Multi-level search |

### LangChain RAG Example

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader

# Load documents
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)

# Embed and store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# Create RAG chain
llm = ChatOpenAI(model="gpt-4o")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# Query
answer = qa_chain.invoke("What is the main conclusion?")
```

### RAG Evaluation Metrics

| Metric | Measures |
|---|---|
| **Context Recall** | Did retrieval find the relevant chunks? |
| **Context Precision** | Were retrieved chunks relevant (not noisy)? |
| **Answer Faithfulness** | Is the answer grounded in the context? |
| **Answer Relevance** | Does the answer address the actual question? |

### RAG vs Fine-Tuning

```
Use RAG when:                   Use Fine-Tuning when:
  ✓ Data changes frequently       ✓ Consistent style/format needed
  ✓ Need to cite sources          ✓ Have 1000+ quality examples
  ✓ Large document collections    ✓ Task requires domain behavior
  ✓ Want to control knowledge     ✓ Latency is critical
```

---

## 16. Fine-Tuning & RLHF

### Types of Fine-Tuning

| Type | Params Trained | VRAM | Data Needed |
|---|---|---|---|
| Full Fine-Tuning | All | Very high (80GB+) | 10K–100K+ examples |
| LoRA | ~0.1–1% | Medium (16–24GB) | 1K–10K examples |
| QLoRA | ~0.1–1% (4-bit quantized) | Low (8–12GB) | 500–5K examples |
| Prompt Tuning | Soft prompts only | Very low | 100–1K examples |

### LoRA Explained

```
Original weight W: frozen (millions of params)

LoRA adds: ΔW = A × B  where rank(A,B) << rank(W)

Update: W_effective = W + α × A × B

Training only updates A and B → ~0.1% of original parameters
At inference: merge → zero overhead
```

### RLHF Pipeline

```
Step 1 — SFT (Supervised Fine-Tuning)
  Data: (prompt, ideal response) pairs written by humans
  Result: Base instruction-following model

Step 2 — Reward Model
  Data: Pairs of responses with human preference label
  Result: Model that scores "how good is this response?"

Step 3 — RL with PPO
  Policy: The SFT model being improved
  Reward: Reward model score
  Constraint: KL divergence penalty (don't drift too far from SFT)
  Result: Helpful, harmless, honest assistant
```

---

## 17. AI Agents & Agentic Systems

### What Makes an Agent

```
Traditional LLM:
  Input → LLM → One Response → Done

Agentic Loop:
  Goal
   │
   ▼
  Think: what action should I take?
   │
   ▼
  Act: call a tool (search, code, API)
   │
   ▼
  Observe: get result
   │
   ▼
  Reflect: did this help? What next?
   │
   └──→ Repeat until goal is achieved
```

### Agent Components

| Component | Description |
|---|---|
| **LLM (Brain)** | Reasons, plans, decides next action |
| **Tools** | Functions the agent can call (web search, Python REPL, APIs) |
| **Memory** | Short-term (conversation) + Long-term (vector store) |
| **Planning** | Breaking goals into sub-tasks |
| **Feedback** | Observing results and adapting |

### Tool / Function Calling

```python
# Define tools
tools = [
    {
        "name": "search_web",
        "description": "Search internet for current information",
        "parameters": {"query": {"type": "string"}}
    },
    {
        "name": "run_python",
        "description": "Execute Python code and return result",
        "parameters": {"code": {"type": "string"}}
    },
    {
        "name": "read_file",
        "description": "Read a file from disk",
        "parameters": {"path": {"type": "string"}}
    }
]
```

### Agent Challenges

| Challenge | Cause | Mitigation |
|---|---|---|
| Hallucination | Wrong tool calls or fabricated results | Verify outputs, use RAG |
| Infinite loops | No stopping condition | Max steps limit |
| Context overflow | Long conversations fill context | Summarize history |
| Cost explosion | Many LLM calls per task | Cache, smaller models for sub-tasks |
| Unsafe actions | Agent takes irreversible actions | Human-in-the-loop for risky steps |

---

## 18. LangChain

> LangChain is the most popular framework for building LLM-powered applications — it provides abstractions for chains, agents, memory, and retrieval.

### Core Abstractions

```
LangChain Building Blocks:

LLMs / Chat Models   → Wrappers for any LLM API
Prompts              → PromptTemplates, system/human messages
Chains               → Connect multiple components in sequence
Agents               → LLM + Tools + Planning loop
Memory               → Conversation history management
Retrievers           → Fetch relevant documents from vector store
Output Parsers       → Structure LLM output as JSON, lists, etc.
```

### Basic Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()

response = chain.invoke({"question": "What is LangChain?"})
```

### Memory

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

conversation.predict(input="My name is John")
conversation.predict(input="What is my name?")
# → "Your name is John."  (remembers context)
```

### LangChain Agent with Tools

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub

tools = [DuckDuckGoSearchRun()]
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = executor.invoke({"input": "What is the latest news on AI?"})
```

---

## 19. LangGraph

> LangGraph extends LangChain with **stateful, graph-based workflows** — enabling complex multi-step agentic applications with loops, branching, and persistence.

### Why LangGraph?

```
LangChain Chains:          Linear — A → B → C
LangGraph:                 Graph — any topology, cycles allowed
                            A → B → C
                                ↑   ↓
                                D ←─┘  (loops, conditionals)
```

### Core Concepts

| Concept | Description |
|---|---|
| **Node** | A function that takes state and returns updated state |
| **Edge** | Connection between nodes (defines execution flow) |
| **State** | Shared data structure that flows through the graph |
| **Conditional Edge** | Branch based on current state value |
| **StateGraph** | The graph object that orchestrates everything |
| **Checkpointer** | Persists state — enables pause/resume |

### LangGraph Structure

```
State Schema (TypedDict)
         │
         ▼
┌─────────────────────┐
│    StateGraph        │
│                     │
│  START              │
│    │                │
│    ▼                │
│  [Node A]           │
│    │                │
│    ├──(condition)──→ [Node B]
│    │                │
│    └──────────────→ [Node C]
│                     │
│  END                │
└─────────────────────┘
```

### Basic LangGraph Example

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

# Define state shape
class AgentState(TypedDict):
    question: str
    answer: str
    steps: list

# Define nodes
def think(state: AgentState) -> AgentState:
    # LLM reasons about the question
    thought = llm.invoke(state["question"])
    return {**state, "steps": state["steps"] + [thought.content]}

def answer(state: AgentState) -> AgentState:
    # Generate final answer
    final = llm.invoke(f"Based on reasoning: {state['steps']}, answer: {state['question']}")
    return {**state, "answer": final.content}

def should_continue(state: AgentState) -> str:
    if len(state["steps"]) < 3:
        return "think"   # Continue reasoning
    return "answer"      # Enough — give final answer

# Build graph
graph = StateGraph(AgentState)
graph.add_node("think", think)
graph.add_node("answer", answer)

graph.set_entry_point("think")
graph.add_conditional_edges("think", should_continue)
graph.add_edge("answer", END)

app = graph.compile()
result = app.invoke({"question": "What is 2+2?", "steps": []})
```

### LangGraph Key Features

| Feature | Description |
|---|---|
| **Cycles / Loops** | Agent can revisit nodes — enables retry/reflection |
| **Conditional branching** | Route to different nodes based on state |
| **Human-in-the-loop** | Pause graph, get human input, resume |
| **Streaming** | Stream intermediate steps to UI |
| **Persistence** | Save/load graph state — resume interrupted workflows |
| **Parallel nodes** | Fan-out to multiple nodes simultaneously |

---

## 20. AutoGen

> AutoGen (Microsoft) is a framework for building **multi-agent conversations** — agents talk to each other to solve tasks.

### AutoGen vs LangGraph

| | AutoGen | LangGraph |
|---|---|---|
| **Model** | Conversation-based | Graph-based state machine |
| **Agents talk** | To each other | Through shared state |
| **Control flow** | Emergent from conversation | Explicit graph structure |
| **Best for** | Collaborative problem solving | Deterministic pipelines |
| **Termination** | Condition on message content | Explicit END node |

### Core Agent Types

| Agent Type | Description |
|---|---|
| **AssistantAgent** | LLM-powered — reasons and generates responses |
| **UserProxyAgent** | Executes code, talks to human, relays messages |
| **GroupChatManager** | Orchestrates multi-agent conversations |
| **ConversableAgent** | Base class — fully configurable |

### Basic AutoGen Example

```python
import autogen

config_list = [{"model": "gpt-4o", "api_key": "YOUR_KEY"}]

# Create agents
assistant = autogen.AssistantAgent(
    name="Assistant",
    llm_config={"config_list": config_list},
    system_message="You are a helpful AI assistant."
)

user_proxy = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",          # No human input in loop
    max_consecutive_auto_reply=5,
    code_execution_config={"work_dir": "code_output"}
)

# Start conversation
user_proxy.initiate_chat(
    assistant,
    message="Write a Python function to sort a list and test it."
)
```

### Multi-Agent Group Chat

```python
# Multiple specialized agents
researcher = autogen.AssistantAgent("Researcher", ...)
coder = autogen.AssistantAgent("Coder", ...)
reviewer = autogen.AssistantAgent("Reviewer", ...)

# Group chat
groupchat = autogen.GroupChat(
    agents=[researcher, coder, reviewer],
    messages=[],
    max_round=10,
    speaker_selection_method="auto"   # LLM decides who speaks next
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=...)

# Initiate
user_proxy.initiate_chat(manager, message="Build an invoice parser.")
```

### AutoGen Patterns

| Pattern | Description |
|---|---|
| **Supervisor + Workers** | One agent delegates to specialized agents |
| **Round-robin** | Each agent takes turns |
| **Debate** | Agents argue positions to reach consensus |
| **Code + Review** | Coder writes, Reviewer checks, Coder fixes |
| **Research + Write** | Researcher finds info, Writer synthesizes |

---

## 21. Multi-Agent Orchestration

### Architecture Patterns

```
1. Sequential Pipeline
   Agent A → Agent B → Agent C → Result

2. Parallel Fan-out
   ┌→ Agent A ─┐
   ├→ Agent B ─┤ → Aggregator → Result
   └→ Agent C ─┘

3. Hierarchical (Supervisor)
   Supervisor
   ├── Research Agent
   ├── Code Agent
   └── Review Agent

4. Debate / Critique
   Agent A proposes → Agent B critiques → Agent A revises
```

### Framework Comparison

| Framework | Paradigm | Key Strength |
|---|---|---|
| **LangChain** | Chain + Agent | Wide ecosystem, tools |
| **LangGraph** | Stateful graph | Complex workflows, loops |
| **AutoGen** | Conversational | Multi-agent discussion |
| **CrewAI** | Role-based crew | Easy role assignment |
| **LlamaIndex** | Data-centric | Document intelligence |
| **MS Semantic Kernel** | Enterprise plugin | Azure / Microsoft stack |

### Context vs Memory vs History

```
Conversation History:  Every message in this session (grows unbounded)
Context Window:        What the LLM can actually "see" (limited tokens)
Memory:                Compressed/selective recall across sessions
                       Types: Buffer | Summary | Entity | Vector (semantic)

Strategy:
  Short convos    → Buffer memory (keep all)
  Long convos     → Summary memory (compress old turns)
  Knowledge facts → Entity memory (remember names, facts)
  Semantic search → Vector memory (retrieve by meaning)
```

---

## 22. Data Engineering Fundamentals

### Data Pipeline

```
Raw Sources (DB, APIs, Files, Streams)
     │
     ▼
Ingestion Layer  → Collect and transport data
     │
     ▼
Storage Layer    → Data lake, warehouse, feature store
     │
     ▼
Processing Layer → Clean, transform, feature engineer
     │
     ▼
Feature Store    → Reusable, versioned features for training
     │
     ▼
Model Training / Inference
```

### Data Quality Dimensions

| Dimension | Check |
|---|---|
| **Completeness** | Missing values? Null rates? |
| **Accuracy** | Values match real-world truth? |
| **Consistency** | Same entity represented uniformly? |
| **Timeliness** | Data fresh enough for use case? |
| **Validity** | Values conform to expected schema? |
| **Uniqueness** | No duplicate records? |

### Train / Validation / Test Split

```
Full Dataset (100%)
  ├── Training Set   (70–80%)  ← Model learns from this
  ├── Validation Set (10–15%)  ← Tune hyperparameters here
  └── Test Set       (10–15%)  ← Final evaluation — TOUCH ONLY ONCE

Golden rule: Test set data must NEVER influence training decisions
             Leakage → model appears better than it is
```

---

## 23. Feature Engineering

### What Is a Feature?

```
Raw: email = "Congratulations! You won $1,000,000!"

Features:
  word_count = 6
  has_dollar_sign = 1
  exclamation_count = 1
  sender_domain = "unknown.ru"
  is_reply = 0
```

### Techniques

| Technique | When | Example |
|---|---|---|
| **Normalization** (0–1) | Neural nets, distance models | price / max_price |
| **Standardization** (z-score) | Linear models, SVMs | (x − mean) / std |
| **One-Hot Encoding** | Categorical variables | `red` → [1,0,0] |
| **Log Transform** | Skewed distribution | log(salary) |
| **Binning** | Continuous → groups | age → child/adult/senior |
| **Interaction Terms** | Capture combined effects | rooms × size |
| **TF-IDF** | Text features | Word importance scores |
| **Embeddings** | Semantic text features | sentence-transformers |
| **Date Parts** | Timestamp signals | hour, day_of_week, is_holiday |

---

## 24. Model Evaluation & Metrics

### Classification Metrics

```
Confusion Matrix:
                   Predicted Positive  Predicted Negative
Actual Positive         TP                   FN
Actual Negative         FP                   TN
```

| Metric | Formula | When to Use |
|---|---|---|
| **Accuracy** | (TP+TN)/Total | Balanced classes |
| **Precision** | TP/(TP+FP) | When FP is costly (spam filter) |
| **Recall** | TP/(TP+FN) | When FN is costly (cancer screening) |
| **F1 Score** | 2·(P·R)/(P+R) | Imbalanced classes |
| **ROC-AUC** | Area under ROC | Ranking quality |

### Regression Metrics

| Metric | Meaning |
|---|---|
| **MAE** | Average absolute error — easy to interpret |
| **RMSE** | Penalizes large errors more |
| **R²** | 1.0 = perfect, 0 = no better than predicting mean |
| **MAPE** | Percentage error — easy to communicate |

### LLM Evaluation

| Metric | Measures | Used For |
|---|---|---|
| **Perplexity** | Model surprise on held-out text | LM quality |
| **BLEU** | N-gram overlap with reference | Translation |
| **ROUGE** | Recall-based n-gram overlap | Summarization |
| **BERTScore** | Semantic similarity via embeddings | Generation quality |
| **LLM-as-Judge** | Strong LLM scores outputs | Scalable quality eval |
| **RAGAs** | Context recall, faithfulness, relevance | RAG evaluation |

---

## 25. Vector Databases & Embeddings

### What Are Embeddings?

```
Text: "The cat sat on the mat"
       ↓ Embedding Model
Vector: [0.12, -0.45, 0.89, ..., 0.67]  ← 1536 dimensions

Semantic property:
  "dog" and "puppy"  → vectors are CLOSE  (similar meaning)
  "dog" and "cloud"  → vectors are FAR    (different meaning)
```

### Similarity Metrics

| Metric | Best For |
|---|---|
| **Cosine Similarity** | Text (ignores magnitude) — most common |
| **Dot Product** | When vectors are normalized |
| **Euclidean Distance** | Spatial distance tasks |

### Vector DB Comparison

| DB | Deployment | Highlights |
|---|---|---|
| **Pinecone** | Managed cloud | Simple API, production-ready |
| **Weaviate** | Cloud + self-host | Multi-modal, GraphQL |
| **Qdrant** | Cloud + self-host | Fast, filtering support |
| **Chroma** | Local + cloud | Great for prototyping |
| **Milvus** | Self-host | High scale, Kubernetes-native |
| **pgvector** | PostgreSQL extension | No new infra if using Postgres |
| **FAISS** | In-memory library | Facebook's library, no persistence |

### Embedding Models

| Model | Dimensions | Best For |
|---|---|---|
| `text-embedding-3-small` (OpenAI) | 1536 | Cost-efficient general use |
| `text-embedding-3-large` (OpenAI) | 3072 | Higher accuracy |
| `all-MiniLM-L6-v2` (HuggingFace) | 384 | Fast, local, free |
| `bge-large-en-v1.5` (BAAI) | 1024 | Strong open-source |

---

## 26. MLOps & Production AI

### MLOps Maturity Levels

```
Level 0 — Manual:         Jupyter notebooks, no automation
Level 1 — ML Pipeline:    Automated training, manual deploy
Level 2 — CI/CD:          Auto train + test + deploy on merge
Level 3 — Full MLOps:     Feature store, registry, monitoring, auto-retrain
```

### Production ML System

```
Data Sources ──→ Feature Store ──→ Training Pipeline ──→ Model Registry
                                                               │
                                                               ▼
                                                      Serving Layer (API)
                                                               │
                                                               ▼
                                                      Monitoring & Alerts
                                                               │
                                                      (drift detected)
                                                               │
                                                        Auto-Retrain
```

### Model Serving Patterns

| Pattern | Use Case |
|---|---|
| **REST API** | Most web applications |
| **Batch Inference** | Overnight processing, reports |
| **Streaming** | Real-time fraud detection, monitoring |
| **Edge** | Offline, privacy-sensitive, mobile |

### Monitoring

| Metric | What to Watch |
|---|---|
| **Model performance** | Accuracy on live data vs baseline |
| **Data drift** | Input distribution shifts |
| **Concept drift** | X→y relationship changes |
| **Latency (P99)** | Response time under load |
| **Cost** | Tokens used, compute cost per request |

### Experiment Tracking

```
Always record:
  - Experiment name + date
  - Dataset version and split
  - Hyperparameters
  - Training loss curve
  - Validation metrics
  - Inference time
  - Git commit hash
```

Tools: **MLflow** · **Weights & Biases** · **Comet ML**

---

## 27. Containerization with Docker

### Why Docker for AI?

```
Problem: "It works on my machine"
Solution: Package app + dependencies + runtime into a container

Container = Your code + Libraries + Python version + OS libs
           Everything needed → runs identically everywhere
```

### Core Concepts

| Concept | Description |
|---|---|
| **Image** | Blueprint — read-only template |
| **Container** | Running instance of an image |
| **Dockerfile** | Instructions to build an image |
| **Registry** | Store and share images (Docker Hub, ECR, ACR) |
| **Volume** | Persistent storage mounted into container |
| **Network** | Communication between containers |

### Dockerfile for AI App

```dockerfile
# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Common Docker Commands

```bash
# Build image
docker build -t my-ai-app:v1 .

# Run container
docker run -p 8000:8000 my-ai-app:v1

# Run with environment variables
docker run -e OPENAI_API_KEY=sk-... -p 8000:8000 my-ai-app:v1

# Run in background
docker run -d --name ai-app my-ai-app:v1

# View logs
docker logs ai-app

# Stop / remove
docker stop ai-app && docker rm ai-app

# Push to registry
docker tag my-ai-app:v1 myregistry.azurecr.io/my-ai-app:v1
docker push myregistry.azurecr.io/my-ai-app:v1
```

### Docker Compose for Multi-Service AI

```yaml
# docker-compose.yml
version: "3.8"
services:

  api:
    build: .
    ports: ["8000:8000"]
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - VECTOR_DB_URL=http://vectordb:6333
    depends_on: [vectordb]

  vectordb:
    image: qdrant/qdrant
    ports: ["6333:6333"]
    volumes:
      - qdrant_data:/qdrant/storage

  redis:
    image: redis:alpine
    ports: ["6379:6379"]

volumes:
  qdrant_data:
```

```bash
docker-compose up -d          # Start all services
docker-compose logs -f api    # Follow logs
docker-compose down           # Stop all
```

---

## 28. Kubernetes for AI

### Why Kubernetes?

```
Docker runs one container on one machine.
Kubernetes orchestrates many containers across many machines.

K8s handles:
  ✓ Scaling (add more replicas under load)
  ✓ Self-healing (restart failed containers)
  ✓ Rolling deployments (zero-downtime updates)
  ✓ Load balancing (distribute traffic)
  ✓ Secret management
  ✓ Persistent storage
```

### Core K8s Objects

| Object | Description |
|---|---|
| **Pod** | Smallest unit — one or more containers |
| **Deployment** | Manages pod replicas and rolling updates |
| **Service** | Exposes pods to network (load balancer) |
| **ConfigMap** | Non-secret configuration data |
| **Secret** | Sensitive data (API keys, passwords) |
| **Ingress** | HTTP routing rules (maps URLs to services) |
| **PersistentVolume** | Durable storage for stateful apps |
| **HPA** | Horizontal Pod Autoscaler — scale on CPU/memory |

### LangGraph Deployment Manifest

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langgraph-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: langgraph-api
  template:
    metadata:
      labels:
        app: langgraph-api
    spec:
      containers:
      - name: langgraph-api
        image: myregistry.azurecr.io/langgraph-api:v1
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

```bash
# Essential kubectl commands
kubectl apply -f deployment.yaml    # Deploy
kubectl get pods                     # List pods
kubectl logs pod-name               # View logs
kubectl exec -it pod-name -- bash   # Shell into container
kubectl scale deployment langgraph-api --replicas=5
kubectl rollout status deployment/langgraph-api
kubectl delete deployment langgraph-api
```

### K8s vs Docker Compose

| | Docker Compose | Kubernetes |
|---|---|---|
| **Scale** | Single machine | Multi-machine cluster |
| **Use case** | Development, small apps | Production at scale |
| **Self-healing** | No | Yes |
| **Load balancing** | Basic | Advanced |
| **Complexity** | Low | High |

---

## 29. CI/CD for AI Pipelines

### What is CI/CD?

```
CI — Continuous Integration:
  Every code push triggers: build → test → validate

CD — Continuous Delivery:
  Tested code is automatically deployed to staging/production
```

### GitHub Actions for AI

```yaml
# .github/workflows/deploy-ai.yml
name: AI App CI/CD

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run tests
        run: pytest tests/ -v

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Login to Azure Container Registry
        uses: azure/docker-login@v1
        with:
          login-server: myregistry.azurecr.io
          username: ${{ secrets.ACR_USERNAME }}
          password: ${{ secrets.ACR_PASSWORD }}

      - name: Build and push Docker image
        run: |
          docker build -t myregistry.azurecr.io/ai-app:${{ github.sha }} .
          docker push myregistry.azurecr.io/ai-app:${{ github.sha }}

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/ai-app \
            ai-app=myregistry.azurecr.io/ai-app:${{ github.sha }}
```

### AI-Specific CI Checks

```
Standard:
  ✓ Unit tests pass
  ✓ Linting (black, flake8)
  ✓ Docker build succeeds

AI-Specific:
  ✓ Prompt regression tests (outputs haven't degraded)
  ✓ Token cost budget check (not exceeding limits)
  ✓ Embedding model version pinned
  ✓ Vector DB schema migration tested
  ✓ LLM API key valid (smoke test)
```

---

## 30. Logging, Observability & Monitoring

### The Three Pillars of Observability

```
Logs       → What happened (events, errors, debug info)
Metrics    → How the system is performing (numbers over time)
Traces     → How a request flows through the system end-to-end
```

### LLM-Specific Observability

```python
# Log every LLM call
import logging

def call_llm(prompt: str, model: str) -> str:
    start = time.time()
    response = client.chat.completions.create(...)
    latency = time.time() - start

    logging.info({
        "event": "llm_call",
        "model": model,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "latency_ms": round(latency * 1000),
        "cost_usd": calculate_cost(response.usage)
    })
    return response.choices[0].message.content
```

### OpenTelemetry for Agent Tracing

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

tracer = trace.get_tracer(__name__)

def run_agent(query: str):
    with tracer.start_as_current_span("agent_run") as span:
        span.set_attribute("query", query)

        with tracer.start_as_current_span("retrieval"):
            docs = retriever.get_relevant_documents(query)
            span.set_attribute("docs_retrieved", len(docs))

        with tracer.start_as_current_span("llm_call"):
            answer = llm.invoke(query)

    return answer
```

### LangGraph / AutoGen Internal Logging

```python
# LangGraph — stream intermediate steps
for step in app.stream({"question": "What is AI?"}):
    for key, value in step.items():
        print(f"Node: {key}")
        print(f"State: {value}")

# AutoGen — verbose mode
user_proxy = autogen.UserProxyAgent(
    ...,
    is_termination_msg=lambda x: "DONE" in x["content"],
)
# All messages automatically logged to console
```

### Monitoring Stack

| Tool | Purpose |
|---|---|
| **Prometheus** | Collect and store metrics |
| **Grafana** | Visualize metrics as dashboards |
| **OpenTelemetry** | Distributed tracing standard |
| **Jaeger / Zipkin** | Trace visualization |
| **Evidently AI** | ML model monitoring, drift detection |
| **WhyLabs** | Data + model quality monitoring |

---

## 31. Cloud Deployment — Azure & AWS

### Azure Deployment

```
Key Azure Services for AI:
  Azure OpenAI Service    → Host GPT-4 in your own Azure subscription
  Azure Container Registry → Store Docker images
  Azure App Service       → Deploy containerized apps (no K8s needed)
  Azure Kubernetes Service (AKS) → Managed K8s
  Azure Cognitive Services → Vision, Speech, OCR (Textract equivalent)
  Azure AI Foundry        → Build and deploy AI apps
```

**Deploy to Azure App Service:**
```bash
# Login
az login

# Create resource group
az group create --name ai-rg --location eastus

# Create container registry
az acr create --resource-group ai-rg --name myregistry --sku Basic

# Build and push image
az acr build --registry myregistry --image ai-app:v1 .

# Deploy to App Service
az webapp create \
  --resource-group ai-rg \
  --plan my-plan \
  --name my-ai-app \
  --deployment-container-image-name myregistry.azurecr.io/ai-app:v1
```

### AWS Deployment

```
Key AWS Services for AI:
  Amazon Bedrock          → Managed LLM APIs (Claude, Llama, etc.)
  Amazon ECS + Fargate    → Serverless container hosting
  Amazon ECR              → Elastic Container Registry (store images)
  AWS Lambda              → Serverless functions for light AI tasks
  Amazon Textract         → OCR — extract text from documents/images
  CloudWatch              → Logging and monitoring
  IAM                     → Identity and access management
```

**Deploy to ECS Fargate:**
```bash
# Push image to ECR
aws ecr create-repository --repository-name ai-app
docker tag ai-app:v1 <account>.dkr.ecr.<region>.amazonaws.com/ai-app:v1
docker push <account>.dkr.ecr.<region>.amazonaws.com/ai-app:v1

# Create ECS task definition + service via AWS Console or CLI
# Fargate handles server provisioning — fully serverless
```

### AWS Bedrock

```python
import boto3

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

response = bedrock.invoke_model(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello, Claude!"}]
    })
)
result = json.loads(response["body"].read())
print(result["content"][0]["text"])
```

### Azure vs AWS for AI

| | Azure | AWS |
|---|---|---|
| **LLM API** | Azure OpenAI (GPT-4) | Bedrock (Claude, Llama, etc.) |
| **Container service** | AKS (K8s) / App Service | ECS Fargate / EKS |
| **OCR** | Azure Cognitive Vision | Amazon Textract |
| **Monitoring** | Azure Monitor | CloudWatch |
| **Serverless** | Azure Functions | AWS Lambda |

---

## 32. Python Environment Setup

### The Problem

```
Without virtual environments:
  pip install django==3.2  ✓
  pip install django==5.0  ← Overwrites 3.2 → Project A breaks!

With virtual environments:
  project-a/venv/ → Django 3.2  ✓  isolated
  project-b/venv/ → Django 5.0  ✓  isolated
```

### venv Commands

| Action | macOS / Linux | Windows |
|---|---|---|
| Create | `python3 -m venv venv` | `python -m venv venv` |
| Activate | `source venv/bin/activate` | `venv\Scripts\activate` |
| Deactivate | `deactivate` | `deactivate` |
| Install | `pip install package` | same |
| Save deps | `pip freeze > requirements.txt` | same |
| Restore | `pip install -r requirements.txt` | same |

### conda Commands

```bash
conda create --name ml_project python=3.11 numpy pandas scikit-learn jupyter
conda activate ml_project
conda env export > environment.yml
conda env create -f environment.yml
```

### AI Project Structure

```
my_ai_project/
├── venv/                   ← NEVER commit to Git
├── src/
│   ├── data/               ← Data loading & preprocessing
│   ├── models/             ← Model definitions
│   ├── agents/             ← Agent logic
│   └── api/                ← FastAPI endpoints
├── notebooks/              ← Exploration only
├── tests/                  ← Unit + integration tests
├── configs/                ← Hyperparameters, prompts
├── requirements.txt        ← ALWAYS commit
├── Dockerfile
├── docker-compose.yml
├── .env.example            ← Template (never commit .env)
├── .gitignore
└── README.md
```

---

## 33. AI Frameworks & Tools Ecosystem

### The Full Stack

```
┌───────────────────────────────────────────────────────────────┐
│                      USER / APPLICATION                        │
│              Streamlit · Gradio · Next.js · FastAPI            │
├───────────────────────────────────────────────────────────────┤
│                      AGENTIC LAYER                             │
│    LangGraph · AutoGen · CrewAI · Microsoft Semantic Kernel    │
├───────────────────────────────────────────────────────────────┤
│                      LLM LAYER                                 │
│  LangChain · LlamaIndex · Haystack · OpenAI SDK · Anthropic SDK│
├───────────────────────────────────────────────────────────────┤
│                      MODEL LAYER                               │
│         PyTorch · TensorFlow · Hugging Face · ONNX             │
├───────────────────────────────────────────────────────────────┤
│                      DATA LAYER                                │
│         Pandas · Polars · DVC · Airflow · dbt · Spark          │
├───────────────────────────────────────────────────────────────┤
│                    INFRASTRUCTURE LAYER                        │
│    Docker · Kubernetes · Azure · AWS · GitHub Actions          │
└───────────────────────────────────────────────────────────────┘
```

### Tool Cheat Sheet

| Category | Top Tools |
|---|---|
| **LLM APIs** | OpenAI, Anthropic, Google Gemini, AWS Bedrock |
| **Orchestration** | LangChain, LlamaIndex, LangGraph, AutoGen |
| **Vector DBs** | Pinecone, Qdrant, Weaviate, Chroma, Milvus, pgvector |
| **ML Frameworks** | PyTorch, TensorFlow, scikit-learn, XGBoost |
| **Fine-tuning** | Hugging Face PEFT, Unsloth, Axolotl |
| **UI / Demo** | Streamlit, Gradio, Chainlit |
| **Experiment Tracking** | MLflow, Weights & Biases, Comet ML |
| **Serving** | FastAPI, BentoML, Triton, Ray Serve |
| **Monitoring** | Evidently AI, WhyLabs, Prometheus + Grafana |
| **Workflow** | Airflow, Prefect, Dagster |
| **CI/CD** | GitHub Actions, GitLab CI |
| **Containers** | Docker, Docker Compose |
| **Orchestration** | Kubernetes, Helm, Minikube |
| **Cloud** | Azure ML, AWS SageMaker, GCP Vertex AI |

---

## 34. Responsible AI & Ethics

### Core Principles

| Principle | Meaning | Failure Example |
|---|---|---|
| **Fairness** | Equal performance across groups | Loan model rejects one gender unfairly |
| **Transparency** | Explainable decisions | Black-box medical diagnosis |
| **Privacy** | Protect personal data | Model trained on private records |
| **Safety** | Prevent harmful outputs | Chatbot gives dangerous advice |
| **Reliability** | Consistent, predictable behavior | Self-driving fails on edge cases |
| **Accountability** | Clear ownership when AI causes harm | Who's responsible when AI is wrong? |

### Types of Bias

| Bias | Description |
|---|---|
| **Historical** | Training data reflects past human prejudice |
| **Representation** | Certain groups under-represented |
| **Measurement** | Proxy labels don't capture true outcome |
| **Aggregation** | One model for groups with different patterns |
| **Deployment** | Model used in different context than trained |

### LLM Safety Concerns

| Risk | Mitigation |
|---|---|
| **Hallucination** | RAG, output verification, citations |
| **Prompt injection** | Input sanitization, strict templates |
| **Jailbreaking** | Constitutional AI, RLHF, content filters |
| **Data leakage** | Differential privacy, data hygiene |
| **Overreliance** | Communicate uncertainty, human review |

### Pre-Deployment Checklist

```
□ Tested on diverse demographic groups?
□ Edge cases and failure modes documented?
□ Human review for high-stakes decisions?
□ Privacy impact assessment completed?
□ Model outputs logged for audit?
□ Rollback plan in place?
□ Rate limiting and abuse prevention?
□ Terms of service for AI usage communicated?
```

---

## 35. AI Engineer Career Map

### AI Engineer vs AI Researcher

| | AI Researcher | AI Engineer |
|---|---|---|
| Focus | Invents new models | Deploys and scales models |
| Output | Papers, algorithms | Production systems |
| Skills | Math, theory, PyTorch | APIs, DevOps, system design |
| Metric | SOTA benchmark results | Reliability, latency, cost |

### Core Skill Stack

```
Foundation:
  ✅ Python programming
  ✅ Statistics and probability
  ✅ Linear algebra (vectors, matrices)
  ✅ SQL + pandas (data manipulation)

Machine Learning:
  ✅ Supervised/Unsupervised/RL fundamentals
  ✅ scikit-learn, XGBoost
  ✅ Feature engineering + evaluation

Deep Learning:
  ✅ PyTorch (preferred)
  ✅ Neural network architectures
  ✅ Training, regularization, optimization

LLM Engineering:
  ✅ Prompt engineering (zero/few-shot, CoT, ReAct)
  ✅ LangChain + LangGraph
  ✅ RAG system design
  ✅ LLM APIs (OpenAI, Anthropic)
  ✅ Fine-tuning (LoRA/QLoRA)
  ✅ Hugging Face ecosystem

Agentic AI:
  ✅ Tool / function calling
  ✅ Multi-agent systems (AutoGen, CrewAI)
  ✅ Agent memory and state management

Production:
  ✅ FastAPI for model serving
  ✅ Docker + Docker Compose
  ✅ Kubernetes basics
  ✅ GitHub Actions CI/CD
  ✅ Azure / AWS fundamentals
  ✅ MLflow / W&B experiment tracking
  ✅ Monitoring and observability
```

### Portfolio Projects by Level

| Level | Project | Skills |
|---|---|---|
| Beginner | House price predictor + EDA | ML fundamentals |
| Beginner | Sentiment classifier on reviews | NLP, classification |
| Intermediate | RAG Q&A over your own PDFs | LLMs, embeddings, vector DB |
| Intermediate | AI REST API with FastAPI + Docker | Model serving |
| Advanced | LangGraph agent with web search + code | Agents, tools, orchestration |
| Advanced | Multi-agent invoice parser (AutoGen) | Multi-agent, document AI |
| Expert | Fine-tuned domain LLM + deployment | LoRA, Hugging Face, cloud |

---

## 36. Master Vocabulary Reference

| Term | Definition |
|---|---|
| **AI** | Machines performing tasks requiring human intelligence |
| **Machine Learning** | Machines that learn patterns from data |
| **Deep Learning** | ML with multi-layered neural networks |
| **Generative AI** | AI that creates new content — text, images, code |
| **Agentic AI** | AI that plans and takes autonomous multi-step actions |
| **Foundation Model** | Pre-trained on broad data; fine-tuned for many tasks |
| **LLM** | Large Language Model — transformer trained on text at scale |
| **Transformer** | Neural architecture using self-attention; basis of all LLMs |
| **Self-Attention** | Mechanism letting each token attend to all other tokens |
| **Tokenization** | Splitting text into units an LLM can process |
| **Token** | Smallest text unit an LLM processes (~0.75 words) |
| **Context Window** | Max tokens LLM can process at once |
| **Embedding** | Dense vector representation of data |
| **Vector Database** | Database for storing and searching embeddings |
| **Cosine Similarity** | Measure of angle between two vectors — used for semantic search |
| **Training** | Teaching the model using labeled examples |
| **Inference** | Using a trained model to make predictions |
| **Fine-Tuning** | Additional training on a pre-trained model for a specific task |
| **LoRA** | Low-Rank Adaptation — efficient fine-tuning (0.1% of params) |
| **QLoRA** | LoRA with 4-bit quantization — fine-tune on a single GPU |
| **RLHF** | Reinforcement Learning from Human Feedback |
| **Reward Model** | Model trained to predict human preference scores |
| **SFT** | Supervised Fine-Tuning — train on (instruction, response) pairs |
| **Prompt** | Input text sent to an LLM |
| **System Prompt** | Instructions defining AI behavior and persona |
| **Zero-Shot** | Model performs task with no examples given |
| **Few-Shot** | Model given 1–5 examples before the actual task |
| **Chain-of-Thought** | Prompting technique eliciting step-by-step reasoning |
| **ReAct** | Reason + Act — interleave thinking and tool calls |
| **RAG** | Retrieval-Augmented Generation — ground LLM with retrieved docs |
| **Chunking** | Splitting documents into smaller pieces for embedding |
| **Hallucination** | LLM confidently generating false information |
| **Temperature** | Controls randomness: 0 = deterministic, 1+ = creative |
| **Top-p** | Nucleus sampling — sample from smallest probable token set |
| **Function Calling** | LLM outputs structured JSON to invoke external tools |
| **Streaming** | Return tokens as generated rather than waiting for full response |
| **AI Agent** | LLM that plans, uses tools, and executes multi-step tasks |
| **Tool** | External function an agent can call (search, code, API) |
| **LangChain** | Framework for building LLM-powered apps with chains and agents |
| **LangGraph** | LangChain extension for stateful, graph-based agentic workflows |
| **Node** | A function in a LangGraph that processes state |
| **Edge** | Connection between nodes — defines execution flow |
| **State** | Shared data structure flowing through a LangGraph |
| **Conditional Edge** | Route to different nodes based on state value |
| **AutoGen** | Microsoft framework for multi-agent conversational AI |
| **AssistantAgent** | AutoGen LLM-powered reasoning agent |
| **UserProxyAgent** | AutoGen agent that executes code and relays human input |
| **GroupChat** | AutoGen multi-agent conversation manager |
| **CrewAI** | Framework for role-based multi-agent teams |
| **DAG** | Directed Acyclic Graph — graph with no cycles |
| **Overfitting** | Model memorizes training data; fails on new data |
| **Underfitting** | Model too simple to capture the pattern |
| **Regularization** | Techniques to prevent overfitting (L1, L2, Dropout) |
| **Loss Function** | Measures how wrong the model's predictions are |
| **Backpropagation** | Algorithm to compute gradients through a neural network |
| **Gradient Descent** | Optimization that minimizes loss by updating weights |
| **Learning Rate** | Step size in gradient descent |
| **Batch Size** | Number of samples processed per gradient update |
| **Epoch** | One complete pass through the training dataset |
| **Accuracy** | Fraction of correct predictions |
| **Precision** | Of predicted positives, what fraction are correct |
| **Recall** | Of actual positives, what fraction were caught |
| **F1 Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Ranking quality of a classifier |
| **Confusion Matrix** | Table of TP, FP, TN, FN counts |
| **Data Drift** | Input distribution shifts over time |
| **Concept Drift** | Relationship between input and output changes |
| **MLOps** | DevOps practices applied to ML systems |
| **Feature Store** | Central repository for reusable, versioned ML features |
| **Model Registry** | Versioned catalog of trained models |
| **Docker** | Platform for packaging apps in portable containers |
| **Dockerfile** | Instructions to build a Docker image |
| **Container** | Running instance of a Docker image |
| **Kubernetes (K8s)** | Container orchestration at scale |
| **Pod** | Smallest K8s unit — one or more containers |
| **Deployment** | K8s object managing pod replicas and updates |
| **HPA** | Horizontal Pod Autoscaler — scales based on load |
| **CI/CD** | Automated build, test, and deploy pipelines |
| **GitHub Actions** | CI/CD platform built into GitHub |
| **OpenTelemetry** | Standard for distributed tracing and observability |
| **Prometheus** | Metrics collection and alerting |
| **Grafana** | Dashboard visualization for metrics |
| **Azure OpenAI** | GPT-4 hosted in your Azure subscription |
| **AWS Bedrock** | Managed access to Claude, Llama, Titan via AWS |
| **Amazon Textract** | AWS OCR service for documents |
| **ECS Fargate** | Serverless container hosting on AWS |
| **Diffusion Model** | Generates images by learning to reverse noise |
| **GAN** | Generator + Discriminator trained adversarially |
| **DALL-E** | OpenAI text-to-image model |
| **Whisper** | OpenAI speech-to-text model |
| **OCR** | Optical Character Recognition — extract text from images |
| **TF-IDF** | Term frequency weighting for text features |
| **Word2Vec** | Static word embedding model |
| **Sentence-BERT** | Model for sentence-level semantic embeddings |
| **Perplexity** | How surprised a language model is by held-out text |
| **BLEU** | Translation quality metric (n-gram overlap) |
| **ROUGE** | Summarization quality metric |
| **Quantization** | Reduce model precision (float32 → int8) to save memory |
| **Distillation** | Train a small model to mimic a large model |
| **Constitutional AI** | AI evaluates and revises its own outputs against principles |
| **Alignment** | Ensuring AI behaves according to human values |
| **Prompt Injection** | Attack where user input hijacks the system prompt |
| **REST API** | HTTP-based interface — standard way to call AI services |
| **FastAPI** | Python framework for building fast REST APIs |
| **Streamlit** | Python library for building AI web UIs in minutes |
| **Gradio** | Python library for ML model demos |

---

## Quick Decision Guides

### Which ML Type?

```
All examples labeled?               → Supervised Learning
Some examples labeled?              → Semi-Supervised Learning
No labels, explore patterns?        → Unsupervised Learning
Sequential decisions + reward?      → Reinforcement Learning
Raw text/images at scale?           → Self-Supervised (pre-training)
```

### RAG vs Fine-Tuning vs Prompting?

```
Start here → Prompt Engineering (fast, no cost)
Need your own data / fresh info?    → RAG
Need consistent style/format?       → Fine-Tuning
Have 1000+ quality examples?        → Fine-Tuning
Data changes frequently?            → RAG
```

### LangChain vs LangGraph vs AutoGen?

```
Simple LLM chain or chatbot?                        → LangChain
Complex workflow with loops, state, branching?       → LangGraph
Multiple agents talking to solve a problem?          → AutoGen
Role-based team (researcher, writer, reviewer)?      → CrewAI
```

### Which Metric?

```
Classification, balanced?           → Accuracy
Classification, imbalanced?         → F1, ROC-AUC
FP is costly (spam filter)?         → Precision
FN is costly (cancer detection)?    → Recall
Regression?                         → RMSE or MAE
LLM output quality?                 → LLM-as-Judge or Human eval
RAG system?                         → RAGAs (faithfulness, relevance)
```

### Docker vs Kubernetes?

```
Local development?                  → Docker / Docker Compose
One machine, small app?             → Docker Compose
Multiple machines, high traffic?    → Kubernetes
Need auto-scaling?                  → Kubernetes + HPA
Serverless containers?              → AWS Fargate / Azure Container Apps
```

---

*Last updated: 2026-03-21 | Credo Systemz — Generative AI + Agentic AI Programs*
*Notes added weekly as course progresses. Sections will be expanded with lab examples.*
