# Transformers & Large Language Models: The Complete Guide
### From Java Developer to AI Engineer

---

> **Who this is for:** Java developers with solid programming fundamentals who want to deeply
> understand how Transformer-based LLMs work — from first principles to production — and
> transition into an AI engineering role.
>
> **Java analogies are used throughout** because you already think in types, classes, and
> systems — we just map those mental models to Python and deep learning.

---

## Table of Contents

1. [Chapter 1: The Big Picture — Why Transformers?](#chapter-1-the-big-picture)
2. [Chapter 2: Java → Python Quick-Reference](#chapter-2-java-to-python)
3. [Chapter 3: The Math You Actually Need](#chapter-3-math)
4. [Chapter 4: Tokenization — From Text to Numbers](#chapter-4-tokenization)
5. [Chapter 5: Embeddings — Giving Meaning to Numbers](#chapter-5-embeddings)
6. [Chapter 6: Positional Encoding — Teaching Order](#chapter-6-positional-encoding)
7. [Chapter 7: Self-Attention — The Heart of Transformers](#chapter-7-self-attention)
8. [Chapter 8: Multi-Head Attention — Parallel Perspectives](#chapter-8-multi-head-attention)
9. [Chapter 9: Feed-Forward Networks & Normalization](#chapter-9-ffn-norm)
10. [Chapter 10: The Full Transformer Architecture](#chapter-10-full-architecture)
11. [Chapter 11: Training LLMs — Loss, Optimization & Scale](#chapter-11-training)
12. [Chapter 12: Decoding Strategies — How LLMs Generate Text](#chapter-12-decoding)
13. [Chapter 13: Modern LLM Architectures — GPT, LLaMA, and Beyond](#chapter-13-modern-architectures)
14. [Chapter 14: Fine-Tuning, RLHF & Alignment](#chapter-14-fine-tuning)
15. [Chapter 15: Hugging Face Ecosystem — The #1 Job-Market Toolkit](#chapter-15-huggingface)
16. [Chapter 16: LangChain & RAG — Building LLM Applications](#chapter-16-langchain)
17. [Chapter 17: Build a Transformer from Scratch (PyTorch)](#chapter-17-pytorch)
18. [Chapter 18: Production & Inference Optimization](#chapter-18-production)
19. [Chapter 19: Career Transition Roadmap — Java Dev to AI Engineer](#chapter-19-career)
20. [Chapter 20: Interview Questions — Beginner to Advanced](#chapter-20-interviews)
21. [Appendix A: Glossary](#appendix-a-glossary)
22. [Appendix B: Recommended Resources](#appendix-b-resources)

---

## Chapter 1: The Big Picture

### What is a Transformer?

A **Transformer** is a neural network architecture introduced in the 2017 paper
*"Attention Is All You Need"* by Vaswani et al. at Google.

Before Transformers, sequence models (RNNs, LSTMs) processed text **one token at a time**,
like a `for` loop you cannot parallelize. Transformers process **all tokens in parallel**
using a mechanism called **self-attention**, where every token can "look at" every other
token simultaneously.

**Result:** Transformers power virtually every modern LLM — GPT-4, Claude, LLaMA, Gemini,
Mistral, Falcon, and hundreds more.

### Why Transformers Beat RNNs

| Problem | RNN | Transformer |
|---|---|---|
| Speed | Sequential — like `for` loop | Parallel — like `parallelStream()` |
| Long-range memory | Forgets early tokens | Direct connection any token→any token |
| GPU utilization | ~20-30% (sequential bottleneck) | ~80-90% (fully parallel) |
| Scaling | Diminishing returns | Reliable gains with more data + compute |

### The 30-Second Mental Model

Think of a Transformer as a processing pipeline (like a Java stream chain):

```
Raw Text
  → Tokenizer         (split into sub-words, like String.split() but smarter)
  → Embedding Layer   (each token → float[] of 768/4096 dimensions)
  → Positional Encoding (inject "I am token #3" signal)
  → N × Transformer Blocks:
      → Multi-Head Self-Attention  (tokens look at each other)
      → Feed-Forward Network       (each token processed independently)
      → Layer Normalization + Residual Connections
  → Output Head       (predict next token probabilities — softmax over vocab)
```

Every stage is a **matrix operation** (like matrix math in Java, but GPU-accelerated).
The entire model is differentiable, so it trains end-to-end with gradient descent.

### Interview Questions — Chapter 1

> **Beginner:** What problem did Transformers solve that RNNs couldn't?
> → Sequential bottleneck (can't parallelize), vanishing gradients over long sequences,
> poor GPU utilization.

> **Intermediate:** Why is the attention mechanism called "self"-attention?
> → Because the Q, K, V vectors all come from the SAME sequence (the model attends to itself),
> unlike cross-attention where Q comes from one sequence and K/V from another.

> **Advanced:** What is the computational complexity of self-attention and why does it matter?
> → O(n² · d) where n = sequence length. Doubling context length quadruples attention cost.
> This is why long-context models (128K tokens) are expensive and why optimizations like
> Flash Attention and linear attention variants were invented.

---

## Chapter 2: Java → Python Quick-Reference

Before diving into code, here is a translation table so Python feels familiar.

### Types & Variables

```java
// Java
int x = 5;
double y = 3.14;
String s = "hello";
boolean flag = true;
```

```python
# Python — dynamic typing, no declaration needed
x = 5
y = 3.14
s = "hello"
flag = True
```

### Collections

```java
// Java
List<Integer> list = new ArrayList<>(Arrays.asList(1, 2, 3));
Map<String, Integer> map = new HashMap<>();
map.put("cat", 1);
int[] arr = {1, 2, 3};
```

```python
# Python
lst = [1, 2, 3]           # list — like ArrayList
dct = {"cat": 1}          # dict — like HashMap
tup = (1, 2, 3)           # tuple — like immutable list
arr = [1, 2, 3]           # list doubles as array
```

### Classes

```java
// Java
public class Animal {
    private String name;
    public Animal(String name) { this.name = name; }
    public String speak() { return "..."; }
}
public class Dog extends Animal {
    public Dog(String name) { super(name); }
    @Override public String speak() { return "Woof"; }
}
```

```python
# Python
class Animal:
    def __init__(self, name):      # __init__ = constructor
        self.name = name           # no private — just convention _name

    def speak(self):
        return "..."

class Dog(Animal):                 # Dog extends Animal
    def speak(self):               # @Override is implicit
        return "Woof"
```

### Key Python Idioms You'll See Everywhere

```python
# List comprehension — like Java stream().map().collect()
squares = [x**2 for x in range(10)]         # [0, 1, 4, 9, ...]

# f-strings — like String.format() but nicer
name = "LLaMA"
print(f"Model: {name}, params: {7e9:.0f}")  # Model: LLaMA, params: 7000000000

# Unpacking — like destructuring
a, b, c = [1, 2, 3]

# *args and **kwargs — like varargs + Map<String,Object>
def func(*args, **kwargs):
    print(args)    # tuple of positional args
    print(kwargs)  # dict of keyword args

# with statement — like try-with-resources
with open("file.txt") as f:
    content = f.read()

# Decorators — like Java annotations that add behavior
@torch.no_grad()    # equivalent to: disable gradient tracking for this function
def inference(model, input):
    return model(input)
```

### NumPy Arrays ≈ Java float[][]

NumPy is the foundation of all ML in Python. Think of `np.ndarray` as a
multi-dimensional `float[]` with GPU-like bulk operations.

```python
import numpy as np

# Create arrays
a = np.array([1.0, 2.0, 3.0])          # 1D — like float[]
b = np.array([[1, 2], [3, 4]])          # 2D — like float[][]
c = np.zeros((3, 4))                    # 3×4 matrix of zeros
d = np.ones((2, 3, 4))                  # 3D tensor — like float[][][]

# Shape — always check this first when debugging
print(b.shape)   # (2, 2) — rows, cols

# Operations — applied element-wise (no loops needed)
a * 2            # → [2.0, 4.0, 6.0]
a + a            # → [2.0, 4.0, 6.0]

# Matrix multiplication — @ operator (like matrix.multiply() in Java)
A = np.ones((3, 4))
B = np.ones((4, 5))
C = A @ B        # → shape (3, 5)

# Slicing
b[0, :]          # first row — like b[0][]
b[:, 1]          # second column — no Java equivalent without a loop
b[1:3, 0:2]      # sub-matrix
```

---

## Chapter 3: The Math You Actually Need

### Linear Algebra Essentials

**Matrix Multiplication** — the #1 operation in deep learning:

```
C = A @ B
If A is (m, n) and B is (n, p) then C is (m, p)
C[i][j] = sum(A[i][k] * B[k][j] for k in range(n))
```

In Java terms: `A` is a 2D array of `m` rows and `n` cols. `B` is `n` rows and `p` cols.
`C[i][j]` is the dot product of row `i` of A with column `j` of B.

In a Transformer, **a linear layer is just a matrix multiply**:
```
output = input @ W + b      # W = weight matrix, b = bias vector
```

**Dot Product** — measures similarity between two vectors:

```
a · b = sum(a[i] * b[i] for all i) = |a| * |b| * cos(θ)
```

- High dot product → vectors point same direction → tokens are related
- Zero dot product → vectors perpendicular → tokens unrelated

This is the core of attention — we measure how "similar" tokens are to each other.

**Softmax** — converts raw scores into a probability distribution:

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)
```

Properties:
- All outputs between 0 and 1
- All outputs sum to 1.0
- The largest input dominates (exponential amplification)

```python
import numpy as np

def softmax(z):
    e = np.exp(z - z.max())   # subtract max for numerical stability
    return e / e.sum()

softmax([2.0, 1.0, 0.1])
# → [0.659, 0.242, 0.099]
```

### Calculus: Just One Concept

**The chain rule** is how neural networks learn:

```
∂Loss/∂w = ∂Loss/∂output × ∂output/∂w
```

PyTorch's `autograd` computes this for you automatically — you never hand-derive gradients.
But you need to understand: *training = computing how each weight affects the loss,
then nudging weights to reduce loss.*

### Probability: Cross-Entropy Loss

LLMs are trained to predict the next token. The loss measures how wrong the prediction is:

```
Loss = -log(probability assigned to the correct token)
```

- Model assigns 0.9 to correct token → Loss = -log(0.9) = 0.105 ✅ (good)
- Model assigns 0.01 to correct token → Loss = -log(0.01) = 4.605 ❌ (bad)

In Java terms: imagine a function `cost(prediction, truth)` that returns 0 when perfect
and grows to infinity when very wrong. Training minimizes this across all examples.

### Interview Questions — Chapter 3

> **Beginner:** Why do we divide attention scores by √d_k?
> → As d_k grows, dot products grow large in magnitude, pushing softmax into saturation
> (tiny gradients). Dividing by √d_k keeps variance ≈ 1, stabilizing training.

> **Intermediate:** What is the difference between cross-entropy loss and MSE loss,
> and why does LLM training use cross-entropy?
> → MSE penalizes proportionally to error². Cross-entropy is better for classification
> (predicting a token from vocabulary = classification). LLMs predict next token from
> 50K+ classes, making cross-entropy the natural choice.

---

## Chapter 4: Tokenization — From Text to Numbers

Neural networks process numbers, not text. Tokenization is the translation layer.

### Why Not Characters or Words?

| Approach | Problem |
|---|---|
| **Characters** | Long sequences, carry little meaning individually |
| **Words** | Can't handle typos, rare words, or new compound words |
| **Sub-words (BPE)** | ✅ Best of both: common words stay whole, rare words split |

Think of BPE like Java's class hierarchy: common concepts get their own token,
rare concepts are built from component tokens.

### Byte-Pair Encoding (BPE)

**How it works (training, run once):**

```
Corpus: "low low low lower newest newest"

Step 0: Vocabulary = individual characters {l, o, w, e, r, n, s, t, _}
Step 1: Most frequent pair = (l, o) → merge → new token 'lo'
Step 2: Most frequent pair = (lo, w) → merge → new token 'low'
Step 3: Most frequent pair = (low, _) → merge → new token 'low_'
Step 4: Most frequent pair = (e, s) → merge → new token 'es'
... repeat until vocab reaches target size (e.g., 50,000)
```

**At inference:**

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "Hello, I am learning Transformers!"
tokens = tokenizer.encode(text)
print(tokens)       # [15496, 11, 314, 716, 4673, 3602, 364, 0]
print(tokenizer.decode(tokens))  # "Hello, I am learning Transformers!"

# See token strings
print(tokenizer.convert_ids_to_tokens(tokens))
# ['Hello', ',', 'Ġ I', 'Ġam', 'Ġlearning', 'ĠTransformers', '!']
# Ġ = space prefix
```

### Vocabulary Size Trade-offs

| Vocab Size | Sequence Length | Embedding Table | Semantic Density |
|---|---|---|---|
| Small (8K) | Long sequences (expensive) | Small | Each token = small piece |
| Medium (32K) | Balanced | Moderate | ✅ Good balance |
| Large (100K+) | Short sequences (cheaper) | Large | More whole words |

Modern LLMs: GPT-4 uses ~100K, LLaMA uses ~32K, Mistral uses ~32K.

### Special Tokens

Every tokenizer adds special tokens for structure:

```python
# Common special tokens
[BOS]  # Beginning of Sequence — <s> or <|endoftext|>
[EOS]  # End of Sequence — </s> or <|endoftext|>
[PAD]  # Padding — makes variable-length sequences same size in a batch
[UNK]  # Unknown — for characters not in vocabulary (rare with BPE)
[SEP]  # Separator — used in BERT between sentence pairs
[MASK] # Mask — used in BERT for masked language modeling
```

In Java terms: these are like `null`, `EOF`, or sentinel values — markers the model
learns to recognize as control signals.

### Interview Questions — Chapter 4

> **Beginner:** What is a token and how is it different from a word?
> → A token is a sub-word unit. Common words like "cat" are one token. Rare words like
> "uncharacteristically" might split into ["un", "character", "istically"]. On average,
> 1 word ≈ 1.3 tokens in English.

> **Intermediate:** Why does tokenization affect model performance?
> → A poorly designed tokenizer creates unnecessarily long sequences (more compute) or
> fails to capture meaningful sub-units. Domain-specific text (code, math, non-English)
> can be tokenized inefficiently if the tokenizer was trained on mostly English text.

> **Advanced:** What is the "fertility" problem in multilingual tokenization?
> → Tokenizers trained mostly on English assign fewer tokens to English than to other
> languages for the same information content. A sentence in Thai might need 5× more tokens
> than the same meaning in English, giving the model less "budget" for non-English content.

---

## Chapter 5: Embeddings — Giving Meaning to Numbers

### From Token IDs to Vectors

Each token ID maps to a learned vector of dimension `d_model` (768, 1024, 4096, etc.).

```
Token "cat" → ID 2368 → Vector [0.12, -0.45, 0.89, ..., 0.03]  (d_model floats)
```

In Java terms: the embedding layer is a `HashMap<Integer, float[]>` — a lookup table.
Except it is a `float[vocab_size][d_model]` matrix, and the values are learned during training.

```python
import torch
import torch.nn as nn

# vocab_size = 32,000 tokens, each as a 768-dimensional vector
embedding = nn.Embedding(32000, 768)

# Look up token IDs — like map.get(key) for each ID in the list
token_ids = torch.tensor([2368, 415, 9021])   # "cat sat mat"
vectors = embedding(token_ids)                 # shape: (3, 768)
```

### What Do These Vectors Capture?

After training on billions of tokens, embedding vectors encode semantic relationships:

```
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
vector("Paris") - vector("France") + vector("Italy") ≈ vector("Rome")
vector("Java") - vector("programming") + vector("music") ≈ vector("guitar")
```

These relationships **emerge purely** from predicting the next token — no one programs
them in. The model discovers that words with similar contexts have similar vectors.

### Hugging Face: Access Real Embeddings

```python
from transformers import AutoModel, AutoTokenizer
import torch

model_name = "sentence-transformers/all-MiniLM-L6-v2"  # popular embedding model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**inputs)
    # Mean pool across token dimension
    return output.last_hidden_state.mean(dim=1)

emb1 = get_embedding("cat")
emb2 = get_embedding("dog")
emb3 = get_embedding("airplane")

# Cosine similarity: cat-dog should be higher than cat-airplane
cos = nn.CosineSimilarity(dim=1)
print(cos(emb1, emb2))      # ~0.85 (similar)
print(cos(emb1, emb3))      # ~0.30 (dissimilar)
```

### Interview Questions — Chapter 5

> **Beginner:** What is an embedding and why do we need it?
> → Tokens are discrete IDs (integers). Neural networks need continuous vectors to compute
> with. An embedding layer maps each ID to a learned float vector.

> **Intermediate:** What is weight tying and why does it save memory?
> → Modern LLMs share the same weight matrix for the input embedding and the output
> projection (lm_head). Since both have shape (vocab_size × d_model), tying halves the
> memory for these parameters (~600MB saved for a 32K vocab / 4096 d_model model).

> **Advanced:** What is the difference between token embeddings and sentence embeddings,
> and which does a Transformer output?
> → Token embeddings: one vector per token (what the Transformer outputs at each position).
> Sentence embeddings: one vector for the whole text (created by pooling token embeddings,
> e.g., mean pooling or using the [CLS] token). Tasks like RAG retrieval need sentence
> embeddings; tasks like next-token prediction use token embeddings.

---

## Chapter 6: Positional Encoding — Teaching Order

### The Problem

Self-attention is **permutation-invariant** — it treats "cat bites dog" and "dog bites cat"
identically because it has no concept of position. We must inject order information explicitly.

In Java terms: self-attention sees a `Set<Token>`, but we need it to see a `List<Token>`.
Positional encoding turns the Set back into a List.

### Sinusoidal Positional Encoding (Original Paper)

```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

These are **added** to the token embeddings:

```
input_to_transformer = token_embedding + positional_encoding
```

**Why sines and cosines?**
- Each position gets a unique pattern (like a fingerprint)
- Different frequencies: low-frequency dims = coarse position, high-frequency = fine
- The model can compute relative position: `PE(pos + k)` is a linear function of `PE(pos)`

### Learned Positional Embeddings

GPT-2, BERT: learned position vectors — another lookup table of shape `[max_seq_len, d_model]`.
Simpler but cannot generalize to longer sequences than seen during training.

### Rotary Position Embeddings (RoPE)

Used by **LLaMA, Mistral, Qwen, Phi** — most modern open-source LLMs.

Instead of adding position info, RoPE **rotates** the Q and K vectors by an angle
proportional to position. Key property: the dot product Q·K depends only on the
**relative distance** `m - n`, not absolute positions.

```python
# Conceptual illustration
def rope(x, position):
    # Rotate pairs of dimensions by position-dependent angle
    theta = position * frequencies      # frequencies learned per dimension
    x_rotated = rotate(x, theta)        # rotation in 2D subspaces
    return x_rotated
```

**Why RoPE is better:**
- No extra parameters (unlike learned embeddings)
- Naturally encodes relative position (what the model actually needs)
- Can be extended to longer contexts than trained on (with tricks like YaRN)

### Interview Questions — Chapter 6

> **Beginner:** Why does a Transformer need positional encoding?
> → Self-attention has no built-in sense of order — it's a set operation.
> Without positional encoding, "I love AI" and "AI love I" would produce
> identical outputs.

> **Intermediate:** What advantage does RoPE have over learned positional embeddings?
> → RoPE encodes relative position (how far apart two tokens are) rather than
> absolute position (which position in the sequence). Relative position is usually
> what matters linguistically. Also, RoPE can be scaled to longer sequences
> without retraining.

> **Advanced:** What is "context length extension" and how does it work?
> → Models trained on 4K context can't handle 32K out of the box because
> positional patterns at positions > 4K were never seen. Techniques like
> RoPE scaling (YaRN, LongRoPE) re-scale the position frequencies so the
> model sees familiar angles even at new positions. Fine-tuning on longer
> sequences then teaches the model to use this extended range.

---

## Chapter 7: Self-Attention — The Heart of Transformers

### The Core Idea

Self-attention lets each token compute a **weighted average of all tokens**, where the
weights are learned based on content.

**Analogy:** Imagine you are reading "The bank was flooded after the rain." When you
process "bank", you look back at all words and decide "flooded" and "rain" are more
relevant than "the". Self-attention is the mechanism that learns to do this.

In Java terms: for each token, self-attention runs a "search" across all tokens, scores
each by relevance, then returns a weighted blend of their information.

### Query, Key, Value — The Search Engine Analogy

| Concept | Search Engine | Self-Attention |
|---|---|---|
| Query (Q) | Your search query | "What information am I looking for?" |
| Key (K) | Page title/metadata | "What does this token contain?" |
| Value (V) | Page content | "What information do I provide?" |

The model scores Q·K for relevance, then retrieves a blend of V vectors.

### Step-by-Step Calculation

Given input matrix **X** of shape `(seq_len, d_model)`:

**Step 1: Create Q, K, V**
```
Q = X @ W_Q    shape: (seq_len, d_k)
K = X @ W_K    shape: (seq_len, d_k)
V = X @ W_V    shape: (seq_len, d_v)
```
W_Q, W_K, W_V are learned weight matrices (like Java class fields that get trained).

**Step 2: Compute Attention Scores**
```
scores = Q @ K^T    shape: (seq_len, seq_len)
```
`scores[i][j]` = how much token `i` should attend to token `j`.

**Step 3: Scale**
```
scaled_scores = scores / √d_k
```
Without this, large `d_k` causes overflow in softmax → vanishing gradients.

**Step 4: Apply Causal Mask (decoder/GPT-style)**
```
mask = upper_triangular matrix of -∞
masked_scores = scaled_scores + mask
```
After softmax, `-∞` → `0`, so future tokens are invisible. This is what makes
language models autoregressive (can't peek at future tokens while training).

**Step 5: Softmax**
```
attention_weights = softmax(masked_scores)    shape: (seq_len, seq_len)
```
Each row sums to 1.0 — a probability distribution over which tokens to attend to.

**Step 6: Weighted Sum**
```
output = attention_weights @ V    shape: (seq_len, d_v)
```

### The Complete Formula

```
Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
```

This is the most important equation in modern AI. Memorize it.

### Worked Numerical Example

```
Sequence: ["I", "love", "AI"]   d_model=4, d_k=2

Token embeddings X:
  I    = [1.0, 0.5, 0.3, 0.8]
  love = [0.2, 0.9, 0.7, 0.1]
  AI   = [0.8, 0.3, 0.6, 0.9]

W_Q = [[0.1, 0.3],   (4×2 matrix)
       [0.4, 0.2],
       [0.5, 0.1],
       [0.2, 0.6]]

Q = X @ W_Q:
  Q_I    = [1×0.1 + 0.5×0.4 + 0.3×0.5 + 0.8×0.2, ...] = [0.61, 0.96]
  Q_love = [0.75, 0.53]
  Q_AI   = [0.68, 0.92]

scores = Q @ K^T / √2  (computed similarly)
attention_weights = softmax(scores)   each row sums to 1.0
output = attention_weights @ V        each output token = blend of all V vectors
```

The key: W_Q, W_K, W_V are **learned** so attention patterns capture real linguistic
relationships — not random blends.

### Python Code: Pure Attention

```python
import torch
import torch.nn.functional as F

def self_attention(X, W_Q, W_K, W_V, causal=True):
    """
    X:    (seq_len, d_model)
    W_Q, W_K, W_V:  (d_model, d_k)
    """
    Q = X @ W_Q          # (seq_len, d_k)
    K = X @ W_K
    V = X @ W_V

    d_k = Q.shape[-1]
    scores = Q @ K.T / d_k**0.5   # (seq_len, seq_len)

    if causal:
        seq_len = scores.shape[0]
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

    weights = F.softmax(scores, dim=-1)    # (seq_len, seq_len)
    return weights @ V                     # (seq_len, d_k)
```

### Interview Questions — Chapter 7

> **Beginner:** In plain English, what does the attention mechanism do?
> → For each token, it looks at all other tokens, scores how relevant each one is,
> then blends the information from all tokens weighted by those scores.
> "the bank flooded" → when processing "bank", attention learns to weight
> "flooded" highly, giving the word its aquatic meaning, not financial meaning.

> **Intermediate:** What is the difference between encoder attention and decoder attention?
> → Encoder: bidirectional — every token can attend to every other token (past and future).
> Decoder: causal/unidirectional — token at position i can only attend to positions 0..i.
> The causal mask (upper triangle of -∞) enforces this during training.

> **Advanced:** What is the KV cache and why is it critical for inference efficiency?
> → During generation, at each step, K and V vectors for all previous tokens are recomputed.
> KV cache stores them — new tokens only compute their own K, V and append to the cache.
> Reduces generation from O(n²) to O(n) per step. For a 70B model with 8K context,
> the KV cache can be 10-20 GB per request — a major engineering challenge.

---

## Chapter 8: Multi-Head Attention — Parallel Perspectives

### Why Multiple Heads?

A single attention head captures one type of relationship. Multiple heads capture many simultaneously.

In Java terms: imagine running multiple `Comparator`s on the same list simultaneously —
each finds different orderings, and you combine all results.

```
Head 1 → syntactic: subject-verb agreement ("she runs" not "she run")
Head 2 → coreference: which noun a pronoun refers to ("John...he")
Head 3 → semantic: "bank" means river or financial based on context
Head 4 → positional: tokens near each other are related
Head 5 → entity: person names attend to each other
...
```

### Calculation

Split `d_model` into `h` heads, each with dimension `d_k = d_model / h`:

```
For each head i:
    Q_i = X @ W_Q_i    shape: (seq_len, d_k)
    K_i = X @ W_K_i
    V_i = X @ W_V_i
    head_i = Attention(Q_i, K_i, V_i)    shape: (seq_len, d_k)

Concatenate:
    concat = [head_1 | head_2 | ... | head_h]   shape: (seq_len, d_model)

Final projection:
    output = concat @ W_O                        shape: (seq_len, d_model)
```

### Grouped Query Attention (GQA)

Used by **LLaMA 2/3, Mistral, Gemma**. Key optimization for inference:

```
Standard MHA:   h Q heads, h K heads, h V heads   → large KV cache
GQA:            h Q heads, h/g K heads, h/g V heads  → g = group size, smaller KV cache
MQA:            h Q heads, 1 K head,  1 V head    → smallest KV cache
```

LLaMA 3 8B: 32 Q heads, 8 KV heads (GQA with g=4).
Memory saved: 4× smaller KV cache vs standard MHA.

### Computational Complexity

Self-attention: **O(n² · d)** where n = sequence length.

This is why long context is expensive:
- 4K context: baseline
- 8K context: 4× more attention compute
- 128K context: 1024× more attention compute

Flash Attention (next chapter) reduces the *memory* from O(n²) to O(n) but does not
change the computational complexity (still O(n²) FLOPs).

### Interview Questions — Chapter 8

> **Beginner:** Why do we need multiple attention heads?
> → Different linguistic relationships need to be captured simultaneously.
> A single head can only focus on one pattern. Multiple heads learn diverse
> patterns in parallel, then combine results.

> **Intermediate:** What is GQA and why did LLaMA adopt it?
> → Grouped Query Attention shares K/V heads across groups of Q heads.
> This reduces KV cache size (critical for serving many concurrent requests)
> with minimal quality loss. LLaMA 2 65B uses 8 KV heads for 64 Q heads.

> **Advanced:** How does Flash Attention improve on standard attention?
> → Standard attention materializes the full n×n attention matrix in HBM (slow GPU memory).
> Flash Attention tiles the computation into blocks that fit in SRAM (fast on-chip memory),
> using the online softmax trick to never materialize the full matrix.
> Result: same output, O(n) memory instead of O(n²), 2-4× faster wall-clock time.

---

## Chapter 9: Feed-Forward Networks & Normalization

### Position-wise FFN

After attention, each token's representation passes through a two-layer neural network
**independently** (no cross-token interaction):

```
FFN(x) = activation(x @ W1 + b1) @ W2 + b2
```

The FFN is interpreted as a **memory store**: attention decides *which information to retrieve*,
the FFN decides *what to do with it*.

Dimensions:
- Input/Output: `d_model` (e.g., 4096)
- Hidden: `4 × d_model` (e.g., 16384) — most parameters live here

### Activation Functions

```python
# GELU — used in GPT-2, BERT
# Smooth version of ReLU — allows small negative values through
import torch.nn.functional as F
F.gelu(x)   # ≈ x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

# SwiGLU — used in LLaMA, Mistral, PaLM (state of the art)
# Adds a learned gate that controls how much signal flows through
def swiglu(x, W1, W_gate, W2):
    return (F.silu(x @ W_gate) * (x @ W1)) @ W2
# silu(x) = x * sigmoid(x) — "Swish" activation
```

SwiGLU consistently outperforms GELU on benchmarks. Nearly all modern LLMs use SwiGLU
or its variants.

### Layer Normalization

Normalizes each token's representation to have mean ≈ 0 and variance ≈ 1:

```
LayerNorm(x) = γ ⊙ (x - μ) / √(σ² + ε) + β
```

- `μ` = mean of x across features (not across the batch!)
- `σ²` = variance
- `γ, β` = learned scale and shift
- `ε = 1e-5` for numerical stability

**Why it matters:** Without normalization, activations can explode or vanish across
100+ layers. LayerNorm keeps them in a stable range for consistent gradient flow.

### RMSNorm (Modern Variant)

Used by **LLaMA, Mistral, Qwen** — simpler and faster than LayerNorm:

```
RMSNorm(x) = γ ⊙ x / √(mean(x²) + ε)
```

Skips the mean-centering step. Empirically performs as well as LayerNorm, ~10% faster.

### Residual Connections (Skip Connections)

Every sub-layer uses a residual connection — like a bypass pipe:

```python
# Post-norm (original 2017 paper)
output = LayerNorm(x + SubLayer(x))

# Pre-norm (modern — more stable for deep networks)
output = x + SubLayer(LayerNorm(x))
```

**Why residuals are critical:**
- Gradients flow directly through skip paths → deep networks train without vanishing gradients
- At initialization, output ≈ input (SubLayer ≈ 0) → stable training start
- In Java terms: like a default return value `x` that the sub-layer *modifies*, not replaces

### Interview Questions — Chapter 9

> **Beginner:** What role does the FFN play in a Transformer block?
> → Attention handles token-to-token communication. The FFN then processes each token
> independently, applying learned transformations. It's the "thinking" step after the
> "looking around" step.

> **Intermediate:** Why does modern LLMs use Pre-Norm instead of Post-Norm?
> → Pre-Norm (normalize before sub-layer) is more stable for very deep networks (100+ layers)
> because it ensures inputs to each sub-layer have controlled scale regardless of depth.
> Post-Norm can cause exploding/vanishing gradients in very deep models.

> **Advanced:** Some researchers say FFN layers are "key-value memory stores." What does this mean?
> → Geva et al. (2021) showed that each FFN neuron acts as a key-value pair: the first layer
> (W1) acts as keys that match certain input patterns, and the second layer (W2) acts as values
> that contribute factual information when matched. E.g., a neuron might activate for
> "capital of France" inputs and contribute "Paris" to the output.

---

## Chapter 10: The Full Transformer Architecture

### One Transformer Block (Pre-Norm Style)

```
Input x  (shape: batch × seq_len × d_model)
│
├── x_attn = LayerNorm(x)
│       ↓
│   Multi-Head Self-Attention
│       ↓
├── x = x + attention_output       ← Residual #1
│
├── x_ffn = LayerNorm(x)
│       ↓
│   Feed-Forward Network (SwiGLU)
│       ↓
└── x = x + ffn_output             ← Residual #2

Output x  (shape: batch × seq_len × d_model)  — same shape as input
```

### Full Model: Stack of N Blocks

```
Token IDs (batch × seq_len)
    ↓ Embedding Layer
    ↓ (+ Positional Encoding or RoPE)
    ↓ Block 1
    ↓ Block 2
    ↓ ...
    ↓ Block N
    ↓ Final LayerNorm
    ↓ Linear (d_model → vocab_size)
    ↓ Softmax
Next-Token Probabilities (batch × seq_len × vocab_size)
```

### Encoder vs Decoder vs Encoder-Decoder

| Architecture | Attention Type | Use Cases | Examples |
|---|---|---|---|
| **Encoder-only** | Bidirectional (see all) | Classification, embeddings | BERT, RoBERTa |
| **Decoder-only** | Causal (see past only) | Text generation | GPT, LLaMA, Claude |
| **Encoder-Decoder** | Both + cross-attention | Translation, summarization | T5, BART |

**Why modern LLMs are all decoder-only:**
- One architecture handles all tasks via prompting
- Scales predictably — just add more layers
- Autoregressive generation maps naturally to most use cases

### Parameter Count Formula

For one block with `d_model = d`, FFN hidden = `4d`, `h` heads:

```
Self-Attention:
  W_Q, W_K, W_V:  3 × (d × d) = 3d²
  W_O:             d × d = d²
  Total Attention: 4d²

FFN (standard):
  W1: d × 4d = 4d²
  W2: 4d × d = 4d²
  Total FFN:     8d²

Per block:   12d²
N blocks:    12Nd²
Embedding:   vocab × d
lm_head:     d × vocab (usually tied with embedding)
```

**Example — LLaMA 3 8B:**

```
d = 4096, N = 32 blocks, vocab = 32,000
Per block: 12 × 4096² = 201 million
32 blocks: 6.44 billion
Embedding: 32,000 × 4096 = 131 million
Total: ≈ 8 billion parameters ✓
```

### Interview Questions — Chapter 10

> **Beginner:** What is the shape of a Transformer's input and output?
> → Input: (batch_size, seq_len) — token IDs (integers).
> After embedding: (batch_size, seq_len, d_model).
> After final linear: (batch_size, seq_len, vocab_size) — probability over each vocab token
> at each position.

> **Intermediate:** Why is weight tying (sharing embedding and lm_head weights) used?
> → The input embedding maps token_id → vector, and lm_head maps vector → token_id scores.
> These are inverse operations, so sharing weights (transposing) reduces parameters by
> ~vocab_size × d_model (130M+ params for a 32K vocab) and often improves performance.

> **Advanced:** What is the difference between cross-attention and self-attention?
> → Self-attention: Q, K, V all come from the same sequence.
> Cross-attention: Q comes from the decoder, K and V come from the encoder.
> Cross-attention lets the decoder "look at" the encoded source sequence — used in
> translation and seq2seq tasks but not in decoder-only LLMs.

---

## Chapter 11: Training LLMs

### The Objective: Causal Language Modeling

Given tokens `[t₁, t₂, ..., tₙ]`, predict each token from all previous:

```
Loss = -(1/n) × Σ log P(tᵢ | t₁, ..., tᵢ₋₁)
```

Every sentence in the training corpus creates supervised pairs automatically — no labels
needed. This is **self-supervised learning**.

### Training Data Scale

| Model | Training Tokens | Parameters |
|---|---|---|
| GPT-2 | ~10 billion | 1.5 billion |
| GPT-3 | 300 billion | 175 billion |
| LLaMA 2 | 2 trillion | 7B – 70B |
| LLaMA 3 | 15 trillion | 8B – 405B |
| Gemini Ultra | ~13 trillion | ~1.5 trillion (est.) |

**Chinchilla Law** (Hoffmann et al., 2022): optimal training uses ~20 tokens per parameter.
Modern models over-train (more tokens than Chinchilla-optimal) for better inference efficiency.

### Optimizer: AdamW

The standard optimizer for all modern Transformers:

```
m_t = β₁ × m_{t-1} + (1 - β₁) × g_t        # first moment (momentum)
v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²       # second moment (variance)
m̂_t = m_t / (1 - β₁ᵗ)                       # bias correction
v̂_t = v_t / (1 - β₂ᵗ)
θ_t = θ_{t-1} - lr × (m̂_t / (√v̂_t + ε) + λ × θ_{t-1})   # weight decay
```

Typical settings: β₁=0.9, β₂=0.95, ε=1e-8, λ=0.1.

In Java terms: AdamW is like a smart gradient descent that adapts the step size per
parameter based on its historical gradient magnitudes.

### Learning Rate Schedule

```python
def get_lr(step, warmup_steps, max_lr, min_lr, total_steps):
    if step < warmup_steps:
        return max_lr * step / warmup_steps          # linear warmup
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
    # cosine decay
```

Warmup avoids early training instability (random weights → large gradients → divergence).

### Distributed Training Techniques

| Technique | What it splits | When to use |
|---|---|---|
| Data Parallelism | Batches across GPUs | Always — standard |
| Tensor Parallelism | Individual matrix ops | Large models (70B+) |
| Pipeline Parallelism | Layers across GPUs | Very large models |
| ZeRO / FSDP | Optimizer states + params | Memory bottleneck |

### Mixed Precision Training

```python
# Modern LLM training uses BF16:
# - Same exponent range as FP32 (no overflow)
# - Half the memory, 2× faster matmuls
# - Critical: loss and gradient accumulation still in FP32

from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast(dtype=torch.bfloat16):
    logits, loss = model(inputs, targets)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
scaler.step(optimizer)
scaler.update()
```

### Interview Questions — Chapter 11

> **Beginner:** What does "pre-training" mean for an LLM?
> → Training from scratch on a massive text corpus (trillions of tokens) to predict
> the next token. This gives the model broad language knowledge before any task-specific
> fine-tuning.

> **Intermediate:** Why do we clip gradients during LLM training?
> → Occasionally a batch produces a very large gradient that would take a huge step and
> destroy learned weights. Gradient clipping (typically max_norm=1.0) caps the gradient
> magnitude, preventing these "gradient explosions" that destabilize training.

> **Advanced:** What is the Chinchilla scaling law and how did it change LLM training?
> → Hoffmann et al. showed that for a given compute budget, the optimal strategy is
> to train a model half the size of GPT-3 on twice as many tokens. Before Chinchilla,
> the field over-parameterized (huge models, few tokens). After: LLaMA 7B on 1T tokens
> beats GPT-3 175B on many benchmarks, using far less compute.

---

## Chapter 12: Decoding Strategies

After training, the model outputs a probability distribution over vocabulary at each step.
How we sample determines output quality.

### Greedy Decoding

```python
next_token = logits.argmax(dim=-1)
```

Fast, deterministic, but repetitive and boring. Gets stuck in loops.

### Temperature Sampling

```python
temperature = 0.8    # < 1.0 = more focused; > 1.0 = more creative
adjusted_logits = logits / temperature
probs = F.softmax(adjusted_logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

- T=0 → greedy
- T=1 → original distribution
- T=2 → very random (almost uniform)

### Top-k Sampling

```python
k = 50
top_k_logits, _ = torch.topk(logits, k)
# Zero out all tokens below the k-th largest
logits[logits < top_k_logits[..., -1:]] = float('-inf')
probs = F.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

### Top-p (Nucleus) Sampling

```python
p = 0.9
sorted_logits, sorted_idx = torch.sort(logits, descending=True)
cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
# Remove tokens once cumulative probability exceeds p
remove_mask = cumulative_probs - sorted_logits.softmax(dim=-1) >= p
sorted_logits[remove_mask] = float('-inf')
logits.scatter_(1, sorted_idx, sorted_logits)
next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
```

Top-p adapts dynamically: sometimes 5 tokens cover 90% probability (focused), sometimes 500.

### Typical Defaults for Different Tasks

| Task | Temperature | Top-p | Top-k |
|---|---|---|---|
| Code generation | 0.2 | 0.95 | 50 |
| Chat / Q&A | 0.7 | 0.9 | 50 |
| Creative writing | 1.0–1.2 | 0.95 | 100 |
| Factual retrieval | 0.0–0.3 | — | — |

### Beam Search

Maintain `b` candidate sequences at each step, expand all. Used in translation but
rarely in chat LLMs (tends to produce generic, safe text).

### Speculative Decoding

1. A small "draft" model generates K candidate tokens fast
2. The large model verifies all K tokens in one parallel forward pass
3. If draft was wrong at position i, accept tokens 0..i-1, reject rest

Result: 2-4× throughput improvement with identical output distribution.

### Interview Questions — Chapter 12

> **Beginner:** What does temperature do in LLM generation?
> → Divides all logits before softmax. Low temperature (0.2) makes the model
> more deterministic and focused. High temperature (1.5) makes it more random
> and creative. Temperature 0 = always pick the most likely token (greedy).

> **Intermediate:** Why is top-p usually preferred over top-k for LLM generation?
> → Top-k always considers exactly k tokens regardless of the probability distribution.
> If the model is very confident (one token has 99% prob), top-k=50 still includes
> 49 low-quality options. Top-p adapts: in confident cases it narrows to fewer tokens;
> in uncertain cases it expands to more.

> **Advanced:** Explain speculative decoding and why it works without changing output quality.
> → The draft model proposes tokens; the large model rejects inconsistent ones using
> a rejection sampling procedure that maintains the exact distribution of the large model.
> It works because the large model can evaluate K draft tokens in one forward pass
> (via the attention mask trick), and most draft tokens for common continuations are accepted.

---

## Chapter 13: Modern LLM Architectures

### GPT Family (OpenAI)

- Decoder-only, causal attention, learned positional embeddings
- GPT-2 (2019): 1.5B params, open weights — great for learning
- GPT-3 (2020): 175B params, demonstrated in-context learning
- GPT-4 (2023): Likely MoE, multimodal, architecture not disclosed

### LLaMA Family (Meta) — Most Important for AI Engineers

Open-weight models that power most enterprise and research deployments.

**LLaMA architectural innovations over GPT-2:**

| Component | GPT-2 | LLaMA |
|---|---|---|
| Normalization | Post-LayerNorm | Pre-RMSNorm |
| Activation | GELU | SwiGLU |
| Position | Learned embeddings | RoPE |
| KV sharing | MHA (h heads each) | GQA (h Q, h/g K/V) |
| Bias terms | Yes | No (removed for speed) |

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load LLaMA 3 8B (requires huggingface-cli login with Meta license accepted)
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,   # use BF16 for ~14GB instead of ~28GB
    device_map="auto"              # automatically split across available GPUs
)

# Format prompt for instruction-tuned model
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Explain RoPE positional encoding in 3 sentences."}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
```

### Mixture of Experts (MoE)

Used by Mixtral 8x7B, likely GPT-4, Grok, and more. Replace single FFN with:

```
Router(x) → select top-k experts out of N
output = Σ gating_score_i × Expert_i(x)   (only k experts compute)
```

Mixtral 8x7B: 8 experts, 2 active per token. Total params: 47B. Active per token: ~13B.
Cost of a 13B model, quality of a 47B model.

### Flash Attention

Not a new architecture — an **implementation optimization** used everywhere:

```
Standard attention:   materializes full n×n matrix in HBM → slow, O(n²) memory
Flash Attention:      tiles computation in SRAM, never materializes full matrix
                      → same output, O(n) memory, 2-4× faster
```

All modern training and inference frameworks (vLLM, Hugging Face, llama.cpp) use Flash Attention.

### Interview Questions — Chapter 13

> **Beginner:** What is the difference between LLaMA 3 8B and LLaMA 3 8B Instruct?
> → "8B" (base model) is pre-trained on raw text — it continues text but doesn't follow
> instructions well. "Instruct" is further fine-tuned with SFT + RLHF to follow instructions,
> engage in dialogue, and be helpful/safe. Always use Instruct for applications.

> **Intermediate:** What is Grouped Query Attention and why does LLaMA 3 use it?
> → GQA shares K/V heads across groups of Q heads. With 32 Q heads and 8 K/V heads,
> the KV cache is 4× smaller. This is critical for serving: each concurrent user's request
> needs its own KV cache. Smaller KV cache = more concurrent users per GPU.

> **Advanced:** What is the routing problem in MoE models and how is it solved?
> → The router must assign tokens to experts differentially (not all tokens go to the same expert).
> Naive training collapses — popular experts get most tokens, others never train.
> Solution: auxiliary load-balancing loss penalizes uneven expert utilization.
> Mixtral's router uses a top-2 softmax over expert logits with auxiliary loss.

---

## Chapter 14: Fine-Tuning, RLHF & Alignment

### The Three Training Stages of a Chat LLM

```
Stage 1: Pre-training
  Data: Trillions of tokens from the internet
  Goal: Learn language, facts, reasoning
  Output: Base model (text completer)

Stage 2: Supervised Fine-Tuning (SFT)
  Data: 10K–1M curated (instruction, response) pairs
  Goal: Learn to follow instructions
  Output: Instruction-tuned model

Stage 3: RLHF / DPO
  Data: Human preference rankings of model responses
  Goal: Be helpful, honest, harmless
  Output: Aligned model (Claude, ChatGPT)
```

### LoRA — The Essential Fine-Tuning Technique

Full fine-tuning updates all parameters. For a 7B model, that's 28GB of gradients alone.
**LoRA** adds tiny adapter matrices instead:

```
Original weight:  W  (frozen)  shape: (d_out, d_in)
LoRA update:      W + α × B @ A
  A: (r, d_in)    — r = rank, typically 8–64
  B: (d_out, r)   — initialized to zero (so initial output = W)
```

Only `A` and `B` are trained. For d=4096, r=16:
- Full fine-tuning: 4096² = 16.7M parameters
- LoRA: 2 × 4096 × 16 = 131K parameters → **128× fewer**

```python
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

lora_config = LoraConfig(
    r=16,                          # rank — higher = more capacity
    lora_alpha=32,                 # scaling factor α (usually 2×r)
    target_modules=[               # which layers to apply LoRA to
        "q_proj", "k_proj",        # attention projections
        "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"  # FFN projections
    ],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 20,971,520 || all params: 8,051,273,728 || trainable%: 0.2604%
```

### QLoRA — Fine-Tuning on Consumer GPUs

Quantize base model to 4-bit (NF4), then apply LoRA:

```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",       # NormalFloat4 — best quality for 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,   # quantize quantization constants too
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)   # enable gradient checkpointing

# Then apply LoRA as above
model = get_peft_model(model, lora_config)
# 7B model now fits in ~6GB GPU RAM instead of ~28GB
```

### RLHF vs DPO

**RLHF (Reinforcement Learning from Human Feedback):**
1. Train a reward model on human preference data
2. Use PPO to optimize LLM to maximize reward while staying close to SFT model

**DPO (Direct Preference Optimization):**
Skips the reward model — directly trains on preference pairs:

```
Loss = -log σ(β × (log π(y_win|x)/π_ref(y_win|x) - log π(y_lose|x)/π_ref(y_lose|x)))
```

DPO is simpler, cheaper, and increasingly preferred. Mistral, Zephyr, and many
open-source aligned models use DPO.

### Full Fine-Tuning Example with Hugging Face Trainer

```python
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

# Load dataset — format: {"instruction": "...", "output": "..."}
dataset = load_dataset("tatsu-lab/alpaca", split="train")

def format_prompt(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    }

dataset = dataset.map(format_prompt)

training_args = TrainingArguments(
    output_dir="./lora-llama",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,     # effective batch = 4×4 = 16
    learning_rate=2e-4,
    bf16=True,
    logging_steps=50,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=100,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="wandb",                 # Weights & Biases tracking
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

### Interview Questions — Chapter 14

> **Beginner:** What is the difference between pre-training and fine-tuning?
> → Pre-training: train on massive raw text to learn general language knowledge (expensive,
> done once by large labs). Fine-tuning: adapt the pre-trained model to a specific task
> using smaller curated data (affordable for most companies).

> **Intermediate:** Why does LoRA work? What is the low-rank hypothesis?
> → The hypothesis: the weight updates needed for fine-tuning have low intrinsic
> dimensionality — they lie in a small subspace of the full parameter space.
> LoRA approximates this subspace with two small matrices (rank r), capturing
> the essential update without touching the full matrix.

> **Advanced:** What is catastrophic forgetting and how does RLHF's KL penalty prevent it?
> → After SFT, fine-tuning with RL can "forget" base capabilities as the model
> over-optimizes for the reward. The KL penalty `β × KL(π_new || π_ref)` penalizes
> the policy from drifting too far from the SFT model, acting as a regularizer
> that preserves base capabilities while improving alignment.

---

## Chapter 15: Hugging Face Ecosystem — The #1 Job-Market Toolkit

Hugging Face is **the most important library in the AI engineering job market**.
Every AI engineering job listing mentions it. Master this ecosystem.

### Core Libraries

| Library | Purpose | Job Market Importance |
|---|---|---|
| `transformers` | Load/run 300K+ pre-trained models | ⭐⭐⭐⭐⭐ Essential |
| `datasets` | Load and process NLP datasets | ⭐⭐⭐⭐⭐ Essential |
| `peft` | LoRA, QLoRA, adapter fine-tuning | ⭐⭐⭐⭐⭐ Essential |
| `accelerate` | Distributed training / multi-GPU | ⭐⭐⭐⭐ Important |
| `trl` | SFT, RLHF, DPO training | ⭐⭐⭐⭐ Important |
| `evaluate` | Metrics (BLEU, ROUGE, etc.) | ⭐⭐⭐ Good to know |
| `tokenizers` | Fast tokenization (Rust-backed) | ⭐⭐⭐ Good to know |

### Transformers Quickstart — The 3 Core APIs

```python
# ─── API 1: Pipeline — highest level, for quick tasks ────────────────────

from transformers import pipeline

# Text generation
generator = pipeline("text-generation", model="gpt2")
result = generator("The capital of India is", max_new_tokens=30)
print(result[0]["generated_text"])

# Sentiment analysis
sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
print(sentiment("This movie was absolutely fantastic!"))
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
text = "India has the world's largest population... [long article]"
print(summarizer(text, max_length=130, min_length=30))

# Named Entity Recognition
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english",
               aggregation_strategy="simple")
print(ner("Apple Inc. was founded by Steve Jobs in Cupertino, California."))
```

```python
# ─── API 2: AutoModel + AutoTokenizer — mid-level, for custom logic ──────

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Tokenize input
prompt = "Explain what gradient descent is in one paragraph."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print(f"Input token IDs: {inputs['input_ids']}")
print(f"Input shape: {inputs['input_ids'].shape}")     # (1, seq_len)

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

# Decode — skip the input tokens, show only generated part
new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
```

```python
# ─── API 3: Model internals — for research and debugging ─────────────────

from transformers import AutoModel
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)

text = "The bank by the river was flooded."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Hidden states — shape: (batch, seq_len, d_model)
last_hidden = outputs.last_hidden_state
print(f"Hidden state shape: {last_hidden.shape}")

# Attention weights — list of tensors, one per layer
# Each: (batch, n_heads, seq_len, seq_len)
attention = outputs.attentions
print(f"Attention shape (layer 0): {attention[0].shape}")

# Plot what "bank" attends to
import matplotlib.pyplot as plt
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
bank_idx = tokens.index("bank")
head_attn = attention[0][0, 0, bank_idx].detach().numpy()
plt.bar(tokens, head_attn)
plt.title("What 'bank' attends to (layer 0, head 0)")
plt.xticks(rotation=45)
plt.show()
```

### Datasets Library

```python
from datasets import load_dataset, Dataset
import pandas as pd

# Load from Hugging Face Hub
dataset = load_dataset("squad")      # Stanford QA dataset
print(dataset)
# DatasetDict({'train': Dataset(10570 rows), 'validation': Dataset(1182 rows)})

# Access splits
train = dataset["train"]
print(train[0])        # first example
print(train.column_names)   # ['id', 'title', 'context', 'question', 'answers']

# Filter and map
filtered = train.filter(lambda x: len(x["context"]) < 500)
tokenized = filtered.map(
    lambda x: tokenizer(x["question"], x["context"],
                        truncation=True, max_length=512),
    batched=True
)

# Load from your own files
df = pd.DataFrame({"text": ["Hello AI", "Learning transformers"], "label": [0, 1]})
custom_dataset = Dataset.from_pandas(df)
```

### Accelerate — Multi-GPU Made Easy

```python
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="bf16")   # handles mixed precision + distribution

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for batch in dataloader:
    with accelerator.accumulate(model):            # gradient accumulation
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)                 # handles distributed backward
        optimizer.step()
        optimizer.zero_grad()
```

### TRL — SFT and RLHF Training

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

trainer = SFTTrainer(
    model=model,                    # your LoRA-patched model
    args=SFTConfig(
        output_dir="./sft-output",
        max_seq_length=2048,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        bf16=True,
    ),
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

### Pushing Models to Hugging Face Hub

```python
from huggingface_hub import HfApi

# Save and upload your fine-tuned model
model.save_pretrained("./my-fine-tuned-model")
tokenizer.save_pretrained("./my-fine-tuned-model")

api = HfApi()
api.upload_folder(
    folder_path="./my-fine-tuned-model",
    repo_id="your-username/my-model-name",
    repo_type="model",
)
```

### Interview Questions — Chapter 15

> **Beginner:** What is the difference between `pipeline()` and `AutoModelForCausalLM`?
> → `pipeline()` is the highest-level API — it handles tokenization, model loading,
> and decoding in one call. `AutoModelForCausalLM` gives you raw access to the model
> for custom training loops, custom decoding, or inspecting internals.

> **Intermediate:** What is `device_map="auto"` and when do you need it?
> → It tells Transformers to automatically distribute the model across all available GPUs
> and CPU RAM using a device map. Essential for models too large for a single GPU
> (e.g., 70B model with 4× A100 GPUs or even CPU offloading).

> **Advanced:** What is the difference between SFTTrainer and the standard Hugging Face Trainer?
> → SFTTrainer (from TRL) adds supervised fine-tuning utilities: response-only masking
> (compute loss only on response tokens, not instruction tokens), packing (concatenate
> multiple short examples to fill context window), and easier chat template handling.

---

## Chapter 16: LangChain & RAG — Building LLM Applications

### Why RAG? The Core Problem

LLMs have:
- **Knowledge cutoff** — don't know about events after training
- **Hallucination** — confidently make up facts
- **No access to private data** — your company's documents are unknown to the model

**RAG (Retrieval-Augmented Generation)** solves this:
```
User question
    → Embed question as vector
    → Search vector DB for relevant document chunks
    → Inject retrieved chunks into LLM prompt
    → LLM answers with real context
```

### RAG Pipeline — End to End

```python
# pip install langchain langchain-openai langchain-community chromadb sentence-transformers

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── Step 1: Load documents ────────────────────────────────────────────────
loader = TextLoader("your_document.txt")
documents = loader.load()

# ── Step 2: Split into chunks ─────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,          # characters per chunk
    chunk_overlap=50,        # overlap prevents losing context at boundaries
)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# ── Step 3: Embed chunks + store in vector DB ─────────────────────────────
# Use a local embedding model (free, runs on CPU)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_db = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"    # saved to disk
)
print(f"Stored {vector_db._collection.count()} chunks in ChromaDB")

# ── Step 4: Create retriever ──────────────────────────────────────────────
retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}             # retrieve top-5 most similar chunks
)

# ── Step 5: Set up LLM + QA chain ────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # or use local model

prompt_template = """Use the following context to answer the question.
If you don't know the answer from the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",             # "stuff" = put all chunks in prompt
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True    # shows which chunks were used
)

# ── Step 6: Query ─────────────────────────────────────────────────────────
result = qa_chain.invoke({"query": "What is the main topic of this document?"})
print("Answer:", result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print(f"  - {doc.page_content[:100]}...")
```

### LangChain Chains and Agents

```python
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Simple chain: prompt → LLM → output
llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Java developer transitioning to AI engineering."),
    ("user", "Explain {concept} using a Java analogy.")
])

chain = prompt | llm    # LangChain Expression Language (LCEL)

result = chain.invoke({"concept": "attention mechanism"})
print(result.content)

# ── Chain of chains ───────────────────────────────────────────────────────
from langchain_core.output_parsers import StrOutputParser

# Step 1: Summarize → Step 2: Translate
summarize_prompt = ChatPromptTemplate.from_template("Summarize this in 2 sentences: {text}")
translate_prompt = ChatPromptTemplate.from_template("Translate to Hindi: {summary}")

full_chain = (
    summarize_prompt
    | llm
    | StrOutputParser()
    | (lambda summary: {"summary": summary})
    | translate_prompt
    | llm
    | StrOutputParser()
)

result = full_chain.invoke({"text": "The Transformer architecture..."})
print(result)
```

### Using OpenAI API Directly (Common in Jobs)

```python
from openai import OpenAI

client = OpenAI()   # uses OPENAI_API_KEY env variable

# ── Chat completion (most common) ─────────────────────────────────────────
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful AI engineer."},
        {"role": "user", "content": "What is Flash Attention?"},
    ],
    temperature=0.7,
    max_tokens=500,
)
print(response.choices[0].message.content)
print(f"Tokens used: {response.usage.total_tokens}")

# ── Streaming ─────────────────────────────────────────────────────────────
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me about LLaMA"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)

# ── Embeddings ────────────────────────────────────────────────────────────
emb_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=["Hello world", "Bonjour monde"]
)
vectors = [item.embedding for item in emb_response.data]
print(f"Embedding dimension: {len(vectors[0])}")   # 1536

# ── Structured output (JSON mode) ─────────────────────────────────────────
from pydantic import BaseModel

class BookReview(BaseModel):
    title: str
    rating: int
    summary: str

completion = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Review 'Attention is All You Need'"}],
    response_format=BookReview,
)
review = completion.choices[0].message.parsed
print(f"Title: {review.title}, Rating: {review.rating}/10")
```

### Interview Questions — Chapter 16

> **Beginner:** What is RAG and why is it better than just asking the LLM directly?
> → Retrieval-Augmented Generation retrieves relevant documents from a knowledge base
> and injects them into the prompt. This grounds the LLM in real facts, reduces
> hallucination, keeps knowledge current without retraining, and lets you use
> private/proprietary data.

> **Intermediate:** What is the chunk size trade-off in RAG?
> → Small chunks (100-200 chars): precise retrieval but may lack context.
> Large chunks (1000-2000 chars): more context but may retrieve irrelevant info.
> Typical sweet spot: 500-800 characters with 10-15% overlap.
> Semantic chunking (split on meaning, not character count) is even better.

> **Advanced:** What is the difference between "stuff", "map-reduce", and "refine" chains in RAG?
> → Stuff: put all retrieved chunks into one prompt (simple, but limited by context window).
> Map-reduce: run LLM on each chunk separately, then combine outputs (handles large docs).
> Refine: iteratively update answer with each chunk (better quality but many LLM calls).

---

## Chapter 17: Build a Transformer from Scratch (PyTorch)

This is the most important exercise for AI engineering interviews. Interviewers will ask
you to explain every component — this code will help you do that.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """Root Mean Square Normalization — simpler and faster than LayerNorm."""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class RotaryEmbedding(nn.Module):
    """RoPE — encodes relative position via rotation of Q/K vectors."""
    def __init__(self, d_head: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        # Compute inverse frequencies for each pair of dimensions
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        positions = torch.arange(max_seq_len).float()
        angles = torch.outer(positions, inv_freq)    # (max_seq_len, d_head/2)
        self.register_buffer("cos_cache", angles.cos())
        self.register_buffer("sin_cache", angles.sin())

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        # x: (batch, n_heads, seq_len, d_head)
        cos = self.cos_cache[:seq_len]   # (seq_len, d_head/2)
        sin = self.sin_cache[:seq_len]

        # Split x into even and odd dimensions (rotate in 2D pairs)
        x_even = x[..., ::2]            # dimensions 0, 2, 4, ...
        x_odd  = x[..., 1::2]           # dimensions 1, 3, 5, ...

        # Apply rotation: [x_even, x_odd] → [x_even*cos - x_odd*sin, x_even*sin + x_odd*cos]
        rotated = torch.stack([
            x_even * cos - x_odd * sin,
            x_even * sin + x_odd * cos,
        ], dim=-1).flatten(-2)          # interleave back to original order

        return rotated


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention with RoPE and optional GQA."""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int = None,         # GQA: fewer K/V heads than Q heads
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads or n_heads    # default: standard MHA
        self.d_head     = d_model // n_heads
        self.n_groups   = n_heads // self.n_kv_heads   # how many Q heads per KV head

        # Projection matrices (no bias — modern convention)
        self.W_q = nn.Linear(d_model, n_heads * self.d_head,         bias=False)
        self.W_k = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.W_v = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.W_o = nn.Linear(n_heads * self.d_head, d_model,         bias=False)

        self.rope = RotaryEmbedding(self.d_head, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape   # batch, seq_len, d_model

        # Project to Q, K, V
        q = self.W_q(x).view(B, T, self.n_heads,    self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        # Apply RoPE to Q and K
        q = self.rope(q, T)
        k = self.rope(k, T)

        # Expand K, V for GQA (repeat KV heads to match Q heads)
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        # Scaled dot-product attention with causal mask
        # F.scaled_dot_product_attention uses Flash Attention in PyTorch 2.0+
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Merge heads: (B, n_heads, T, d_head) → (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network — standard in LLaMA, Mistral."""
    def __init__(self, d_model: int):
        super().__init__()
        # LLaMA uses 8/3 * d_model for FFN hidden dim, rounded up to multiple of 256
        d_ff = int(d_model * 8 / 3)
        d_ff = 256 * math.ceil(d_ff / 256)

        self.gate = nn.Linear(d_model, d_ff, bias=False)   # gating path
        self.up   = nn.Linear(d_model, d_ff, bias=False)   # value path
        self.down = nn.Linear(d_ff, d_model, bias=False)   # output projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: element-wise gate × up, then project down
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    """One Transformer block with Pre-Norm (norm before sub-layer)."""
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int = None):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads, n_kv_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn   = FeedForward(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))   # attention with residual
        x = x + self.ffn(self.norm2(x))    # FFN with residual
        return x


class MiniLLM(nn.Module):
    """
    Minimal decoder-only Transformer LLM (LLaMA-style).
    Architecturally identical to LLaMA 3 — just smaller.
    """
    def __init__(
        self,
        vocab_size: int  = 32000,
        d_model: int     = 512,
        n_layers: int    = 8,
        n_heads: int     = 8,
        n_kv_heads: int  = 4,           # GQA: 4 KV heads for 8 Q heads
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads)
            for _ in range(n_layers)
        ])
        self.norm    = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: embedding and lm_head share weights (saves memory)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, ids: torch.Tensor, targets: torch.Tensor = None):
        x = self.token_emb(ids)         # (B, T) → (B, T, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)        # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten for cross_entropy: (B*T, vocab_size) vs (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, max_new: int = 100,
                 temp: float = 0.8, top_p: float = 0.9) -> torch.Tensor:
        """Autoregressive generation with nucleus sampling."""
        for _ in range(max_new):
            ctx = ids[:, -2048:]                    # crop to max_seq_len
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temp        # last token, apply temperature

            # Top-p filtering
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = sorted_logits.softmax(-1).cumsum(-1)
            mask = cum_probs - sorted_logits.softmax(-1) >= top_p
            sorted_logits[mask] = float('-inf')
            logits.scatter_(1, sorted_idx, sorted_logits)

            next_id = torch.multinomial(F.softmax(logits, -1), 1)
            ids = torch.cat([ids, next_id], dim=1)
        return ids


# ── Quick test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = MiniLLM()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")       # ~50 million

    # Forward pass
    ids     = torch.randint(0, 32000, (2, 64))   # batch=2, seq_len=64
    targets = torch.randint(0, 32000, (2, 64))
    logits, loss = model(ids, targets)
    print(f"Logits: {logits.shape}")             # (2, 64, 32000)
    print(f"Loss: {loss.item():.4f}")            # ~10.3 (random = log(32000))

    # Generation
    prompt = torch.randint(0, 32000, (1, 10))
    output = model.generate(prompt, max_new=20)
    print(f"Generated: {output.shape[1]} tokens")
```

### Training Loop

```python
from torch.optim import AdamW

model = MiniLLM().cuda()
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))

for step, batch in enumerate(dataloader):
    ids     = batch["input_ids"].cuda()
    targets = batch["labels"].cuda()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss = model(ids, targets)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)    # prevent explosion
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)      # set_to_none saves memory vs zero_

    if step % 100 == 0:
        print(f"Step {step}: loss={loss.item():.4f}")
```

---

## Chapter 18: Production & Inference Optimization

### KV Cache — The Most Important Inference Optimization

```
Without KV cache:
  Token 1: compute K₁, V₁ → attend over [K₁, V₁]
  Token 2: recompute K₁, V₁, K₂, V₂ → attend over all 4
  Token n: recompute all K, V from scratch → O(n²) total

With KV cache:
  Token 1: compute K₁, V₁ → cache [K₁, V₁]
  Token 2: compute K₂, V₂ only → append to cache
  Token n: compute Kₙ, Vₙ only → O(n) total
```

KV cache memory per request:
```
2 × n_layers × n_kv_heads × d_head × seq_len × bytes_per_element
```

For LLaMA 3 8B (32 layers, 8 KV heads, 128 d_head, BF16, 8K context):
```
2 × 32 × 8 × 128 × 8192 × 2 bytes ≈ 1.07 GB per request
```

### Quantization

```python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM

# 4-bit quantization with AWQ (best quality)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto"
)
# 7B model: FP16 = 14GB → INT4 = ~4GB
```

| Format | Bits | 7B Model | Quality |
|---|---|---|---|
| FP32 | 32 | 28 GB | Baseline |
| BF16/FP16 | 16 | 14 GB | Same as FP32 |
| INT8 | 8 | 7 GB | Minimal loss |
| INT4 (GPTQ/AWQ) | 4 | 3.5 GB | Small loss |
| GGUF Q4_K_M | ~4.5 | 4 GB | Good balance |

### vLLM — Production Serving

```python
from vllm import LLM, SamplingParams

# Load model once, serve many requests
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    dtype="bfloat16",
    gpu_memory_utilization=0.90,
    max_model_len=8192,
)

params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=200)

# Continuous batching: handle multiple prompts efficiently
prompts = [
    "What is machine learning?",
    "Explain attention mechanisms.",
    "What is fine-tuning?",
]

outputs = llm.generate(prompts, params)
for output in outputs:
    print(output.outputs[0].text)
```

**vLLM advantages:**
- **PagedAttention**: Manages KV cache like OS virtual memory → no fragmentation
- **Continuous batching**: New requests join mid-batch, finished requests leave
- **Up to 24× higher throughput** than naive Hugging Face inference

### Ollama — Local Deployment

```bash
# Install and run a model locally (great for development)
ollama pull llama3
ollama run llama3 "What is a Transformer?"

# Start API server (OpenAI-compatible)
ollama serve    # starts on localhost:11434
```

```python
# Use Ollama with OpenAI client (they have compatible APIs)
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
response = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "Explain RoPE in 3 sentences"}]
)
print(response.choices[0].message.content)
```

### Interview Questions — Chapter 18

> **Beginner:** What is quantization and why do we use it?
> → Quantization reduces model weight precision (e.g., from FP16 to INT4).
> This shrinks model size (7B FP16 = 14GB → 4-bit ≈ 3.5GB), fits models on
> consumer GPUs, and speeds up inference — with small quality loss.

> **Intermediate:** What is continuous batching and why does it matter for LLM serving?
> → Traditional batching waits for all requests in a batch to finish before starting new ones.
> Continuous batching allows finished requests to leave mid-batch and new requests to join.
> This dramatically improves GPU utilization since requests finish at different lengths.

> **Advanced:** Explain PagedAttention and why vLLM uses it.
> → Standard attention servers pre-allocate contiguous KV cache blocks per request.
> This wastes memory (over-allocation) and limits batch size.
> PagedAttention manages KV cache in non-contiguous fixed-size pages (like OS virtual memory),
> enabling fine-grained allocation, copy-on-write for parallel sampling, and up to 24×
> better throughput vs Hugging Face inference.

---

## Chapter 19: Career Transition Roadmap — Java Dev to AI Engineer

### Your Java Advantage

Most AI engineers come from data science backgrounds and struggle with:
- Production systems design
- API design and reliability
- Distributed systems
- Debugging complex pipelines
- Testing and CI/CD

You already have these. Your gap is ML theory + Python ecosystem.
**You don't need to close the whole gap — just the ML part.**

### 12-Month Roadmap

```
Month 1-2:   Python + NumPy + Math foundations
Month 3-4:   Deep Learning fundamentals (Karpathy Zero to Hero)
Month 5-6:   Transformer deep-dive + build from scratch
Month 7-8:   Hugging Face ecosystem + fine-tuning
Month 9-10:  RAG + LangChain + production serving (vLLM)
Month 11-12: Specialization + portfolio + job search
```

### Month 1-2: Python & Math

| Topic | Resource | Time |
|---|---|---|
| Python basics | Official tutorial + Codecademy | 2 weeks |
| NumPy | NumPy 100 exercises | 1 week |
| Pandas | 10 minutes to pandas | 3 days |
| Linear algebra | 3Blue1Brown Essence of Linear Algebra (YouTube) | 1 week |
| Calculus review | 3Blue1Brown Essence of Calculus (YouTube) | 3 days |

**Java developer tip:** Python feels loose after Java. Embrace it — use type hints to feel
at home and mypy for type checking.

```python
# Type hints — makes Python feel more like Java
def train_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    lr: float = 3e-4,
    epochs: int = 10
) -> dict[str, list[float]]:
    ...
```

### Month 3-4: Deep Learning Fundamentals

1. **Andrej Karpathy — Zero to Hero** (YouTube, free): Build a neural network from scratch
2. **Fast.ai Part 1** (free): Top-down practical approach
3. Build: image classifier, sentiment analyzer, character-level language model

### Month 5-6: Transformer Deep-Dive

1. Read "Attention Is All You Need" (use this guide as companion)
2. **Karpathy — "Let's build GPT"** (2 hours, YouTube): The single best video in AI education
3. Implement the model in Chapter 17 of this guide — every line
4. Train on Shakespeare (small corpus, fast to run)
5. Read "The Illustrated Transformer" (Jay Alammar blog)

### Month 7-8: Hugging Face + Fine-Tuning

1. Complete the Hugging Face NLP course (free at huggingface.co/learn)
2. Fine-tune LLaMA 3 8B with LoRA/QLoRA on a custom task
3. Push model to Hugging Face Hub
4. Run evaluations (perplexity, ROUGE, task-specific metrics)

### Month 9-10: RAG & Production

1. Build a RAG pipeline (end-to-end, like Chapter 16)
2. Set up vLLM server, benchmark throughput and latency
3. Quantize a 7B model and compare quality vs size vs speed
4. Monitor with Weights & Biases or LangSmith

### Month 11-12: Portfolio & Job Search

**Portfolio projects (pick 2-3):**
- RAG chatbot over a domain-specific corpus (medical, legal, code)
- Fine-tuned LLM for a specific task (text-to-SQL, classification, etc.)
- Inference optimization: quantize a model, measure quality/speed trade-offs
- Custom evaluation framework for an LLM task

**Job titles to target:**
- AI Engineer — build LLM-powered products (closest to SWE)
- ML Engineer — train and deploy models
- LLM Engineer — specialize in LLM infrastructure and fine-tuning
- Platform/MLOps Engineer — infrastructure for training and serving (SWE skills = high value)

**Keywords for job search:**
LLM, RAG, fine-tuning, LoRA, Hugging Face, LangChain, vLLM, PyTorch, vector databases

### Salary Ranges (India & US, 2025)

**India:**

| Role | Experience | Range (LPA) |
|---|---|---|
| AI/ML Engineer (Mid) | 2-4 yrs ML | ₹18L – ₹40L |
| Senior AI Engineer | 4-7 yrs | ₹35L – ₹70L |
| Staff AI Engineer | 7+ yrs | ₹60L – ₹1.5Cr+ |

**USA:**

| Role | Experience | Range (USD) |
|---|---|---|
| AI Engineer (Mid) | 2-4 yrs ML | $180K – $280K |
| Senior AI Engineer | 4-7 yrs | $250K – $400K |
| ML Research Engineer | 3-6 yrs | $220K – $350K |

*Ranges include base + equity + bonus at tech companies.*

---

## Chapter 20: Interview Questions — Beginner to Advanced

### Tier 1: Conceptual (All AI Engineering Interviews)

1. **What is self-attention? Explain in plain English.**
   → Each token computes how relevant every other token is to it, then creates a
   weighted blend of information from all tokens based on those relevance scores.

2. **Why do Transformers scale better than RNNs?**
   → Full parallelism during training — all tokens processed simultaneously, enabling
   GPU utilization of 80-90%. RNNs are sequential loops with no parallelism.

3. **What is the purpose of Layer Normalization?**
   → Normalizes each token's feature vector to prevent activation explosion/vanishing
   across many layers. Keeps gradients stable during training of deep networks (100+ layers).

4. **What is tokenization and why do we use sub-word tokenization?**
   → Splitting text into model-readable units. Sub-word (BPE) handles rare words,
   typos, and morphology better than words, without sequences as long as characters.

5. **What is the KV cache and why is it critical?**
   → Stores K and V tensors computed for previous tokens so they don't need recomputation.
   Reduces generation from O(n²) to O(n) per step — essential for low-latency inference.

6. **What is temperature in LLM generation?**
   → Divides logits before softmax. Low temperature → focused, deterministic.
   High temperature → creative, random. Zero temperature → greedy (always pick max prob).

7. **What is fine-tuning vs pre-training?**
   → Pre-training: learn general language from trillions of tokens (very expensive).
   Fine-tuning: adapt pre-trained model to specific task using small curated data (affordable).

8. **What is RAG?**
   → Retrieval-Augmented Generation: retrieve relevant documents from a knowledge base,
   inject into LLM prompt. Solves knowledge cutoff, hallucination, and private data access.

### Tier 2: Technical (Engineering Role Interviews)

9. **Explain the attention formula Q@K^T/√d_k and why each part exists.**
   → Q@K^T: compute dot-product similarity between all query-key pairs → (n,n) scores.
   /√d_k: scale to prevent softmax saturation for large d_k. Then softmax → probability
   weights. Then @V → weighted blend of value vectors.

10. **What is the computational complexity of attention? Why does it matter?**
    → O(n² · d) where n = sequence length. Quadratic in sequence length: doubling context
    quadruples attention compute. Critical for long-context models — Flash Attention reduces
    memory from O(n²) to O(n) by avoiding materializing the full attention matrix.

11. **Explain LoRA — how does it work and why does it reduce parameters?**
    → Instead of updating W (d×d), learn W + αBA where B is (d×r) and A is (r×d), r<<d.
    Only A and B are trained. Parameters: 2dr vs d² — for d=4096, r=16: 128× fewer.

12. **What is the difference between MHA, GQA, and MQA?**
    → MHA: h Q, h K, h V heads. GQA: h Q heads, h/g K/V heads (shared per group).
    MQA: h Q heads, 1 K head, 1 V head. GQA/MQA reduce KV cache size, critical for serving.

13. **Why does LLaMA use RMSNorm instead of LayerNorm?**
    → RMSNorm skips mean-centering (only normalizes by RMS). Empirically matches LayerNorm
    quality, runs ~10% faster. Modern LLMs prioritize compute efficiency at this scale.

14. **What is the difference between Pre-Norm and Post-Norm?**
    → Pre-norm: normalize before sub-layer (x + SubLayer(Norm(x))) — more stable for
    very deep networks. Post-norm: normalize after (Norm(x + SubLayer(x))) — original paper.
    Pre-norm is standard in modern LLMs (LLaMA, GPT-4, Mistral).

### Tier 3: Deep / Research (Senior/Staff Interviews)

15. **Explain the Chinchilla scaling law and its practical implications.**
    → Optimal compute = train model on 20× as many tokens as parameters. Before Chinchilla,
    labs over-parameterized (bigger models, fewer tokens). After: smaller models trained
    on more data (LLaMA 7B on 1T tokens) beat larger undertrained models on benchmarks.

16. **What is speculative decoding? Under what conditions does it help?**
    → Small draft model generates K tokens; large model verifies all K in one parallel pass.
    Accepted tokens are identical in distribution to the large model (rejection sampling).
    Helps when: draft model's accuracy is high (common text patterns) and K/V cache
    computation dominates (memory-bound inference at small batch sizes).

17. **Explain Flash Attention's IO-aware algorithm.**
    → Standard attention computes S=Q@K^T (writes to HBM), then P=softmax(S) (reads+writes HBM),
    then O=P@V (reads HBM). Each HBM write/read takes ~10× longer than SRAM compute.
    Flash Attention tiles QKV into SRAM blocks and uses online softmax to compute
    attention without writing S or P to HBM. Result: same output, O(n) HBM reads/writes.

18. **What is reward hacking in RLHF and how does the KL penalty address it?**
    → The LLM learns to maximize reward score by gaming the reward model — producing text
    that scores high but isn't actually helpful (e.g., unusual formatting that fools the
    reward model). KL(π_new || π_ref) penalizes deviation from the SFT model, regularizing
    against extreme policy shifts while allowing gradual alignment improvements.

19. **What is a mixture-of-experts model and what is the load balancing problem?**
    → MoE replaces FFN with N experts + router. Router picks top-k experts per token.
    Load balancing problem: without intervention, the router collapses — a few popular
    experts receive all tokens, others never train. Solution: auxiliary loss that penalizes
    uneven expert utilization across the batch.

20. **How does continuous batching differ from traditional batching in LLM serving?**
    → Traditional: fill batch, run until all finish, fill next batch. Wasteful when
    requests have different output lengths (short requests wait for long ones).
    Continuous batching (iteration-level batching): at each decode step, finished sequences
    are evicted and new requests join. PagedAttention enables this without KV cache
    fragmentation. vLLM implements both — enabling 24× throughput improvement.

---

## Appendix A: Glossary

| Term | Definition |
|---|---|
| **Attention** | Mechanism computing weighted relationships between all tokens in a sequence |
| **Autoregressive** | Generating one token at a time, each conditioned on all previous tokens |
| **BPE** | Byte-Pair Encoding — sub-word tokenization algorithm |
| **Causal mask** | Upper-triangular -∞ mask preventing attention to future tokens |
| **Chinchilla law** | Optimal training uses ~20 tokens per parameter |
| **d_model** | Main hidden dimension of the Transformer (vector size per token) |
| **DPO** | Direct Preference Optimization — simpler alternative to RLHF |
| **Embedding** | Learned mapping from discrete token ID to continuous vector |
| **FFN** | Feed-Forward Network — two-layer MLP per token |
| **Flash Attention** | IO-aware attention avoiding n×n matrix materialization |
| **GQA** | Grouped-Query Attention — shares KV heads across Q groups |
| **HBM** | High Bandwidth Memory — main GPU memory (slow, large) |
| **KV Cache** | Stored K/V tensors from previous tokens to avoid recomputation |
| **LoRA** | Low-Rank Adaptation — PEFT technique: W + αBA |
| **MoE** | Mixture of Experts — router selects k of N expert FFNs per token |
| **MQA** | Multi-Query Attention — all Q heads share 1 K head and 1 V head |
| **PEFT** | Parameter-Efficient Fine-Tuning — LoRA, QLoRA, adapters |
| **Perplexity** | exp(cross-entropy loss) — how surprised the model is by test data |
| **QLoRA** | LoRA on a 4-bit quantized base model |
| **RAG** | Retrieval-Augmented Generation — inject retrieved docs into LLM prompt |
| **RLHF** | Reinforcement Learning from Human Feedback |
| **RMSNorm** | Root Mean Square Normalization — skip mean-centering vs LayerNorm |
| **RoPE** | Rotary Position Embedding — encodes relative position via rotation |
| **SFT** | Supervised Fine-Tuning — train on (instruction, response) pairs |
| **SwiGLU** | Gated activation: Swish(x@Wgate) ⊙ (x@W1) |
| **Temperature** | Divides logits before softmax — controls generation randomness |
| **Token** | Sub-word unit — atomic element models process |
| **Top-p** | Nucleus sampling — keep smallest token set covering cumulative prob p |
| **vLLM** | High-throughput LLM serving with PagedAttention |
| **Weight tying** | Share embedding and lm_head weights (saves vocab×d_model params) |

---

## Appendix B: Recommended Resources

### Papers (Read in This Order)

1. **Attention Is All You Need** (Vaswani et al., 2017) — The foundation
2. **BERT** (Devlin et al., 2019) — Encoder-only, bidirectional training
3. **Language Models are Few-Shot Learners** (Brown et al., 2020) — GPT-3, in-context learning
4. **Training Compute-Optimal LLMs** (Hoffmann et al., 2022) — Chinchilla scaling law
5. **LLaMA** (Touvron et al., 2023) — Modern open-source architecture
6. **LoRA** (Hu et al., 2021) — Parameter-efficient fine-tuning
7. **InstructGPT** (Ouyang et al., 2022) — RLHF in practice
8. **Direct Preference Optimization** (Rafailov et al., 2023) — Simpler RLHF alternative
9. **Flash Attention** (Dao et al., 2022) — IO-aware attention algorithm
10. **Mixtral** (Jiang et al., 2024) — Open-source MoE LLM

### Videos (Watch in This Order)

1. **Andrej Karpathy — Neural Networks: Zero to Hero** (YouTube, free, 16 hours)
   → Build everything from scratch: backprop, MLP, RNN, GPT
2. **Andrej Karpathy — Let's build GPT** (YouTube, free, 2 hours)
   → The single best 2-hour investment in AI education
3. **3Blue1Brown — Attention in transformers** (YouTube, free, 30 min)
   → Visual intuition for attention
4. **Stanford CS224N** (YouTube, free)
   → Academic depth on NLP and Transformers
5. **Fast.ai Practical Deep Learning** (fast.ai, free)
   → Practical, top-down, code-first

### Written Tutorials

- **The Illustrated Transformer** — Jay Alammar (jalammar.github.io) — best visual guide
- **Hugging Face NLP Course** — huggingface.co/learn/nlp-course — hands-on, complete
- **LLM Visualization** — bbycroft.net/llm — interactive 3D walkthrough
- **Dive into Deep Learning** — d2l.ai — free, interactive, PyTorch code

### Libraries to Master (Job Market Priority)

| Library | Priority | What it does |
|---|---|---|
| `torch` (PyTorch) | ⭐⭐⭐⭐⭐ | Tensors, autograd, model training |
| `transformers` (HF) | ⭐⭐⭐⭐⭐ | Load 300K+ pre-trained models |
| `datasets` (HF) | ⭐⭐⭐⭐⭐ | Load and process ML datasets |
| `peft` (HF) | ⭐⭐⭐⭐⭐ | LoRA, QLoRA fine-tuning |
| `trl` (HF) | ⭐⭐⭐⭐ | SFT, RLHF, DPO training |
| `langchain` | ⭐⭐⭐⭐ | LLM application framework |
| `openai` (OpenAI SDK) | ⭐⭐⭐⭐ | GPT-4 API access |
| `vllm` | ⭐⭐⭐⭐ | High-throughput LLM serving |
| `accelerate` (HF) | ⭐⭐⭐ | Multi-GPU / distributed training |
| `wandb` | ⭐⭐⭐ | Experiment tracking |
| `chromadb` | ⭐⭐⭐ | Local vector database for RAG |
| `sentence-transformers` | ⭐⭐⭐ | Sentence-level embeddings |

### Communities

- **Hugging Face Discord** — largest open ML community
- **r/LocalLLaMA** — running open models, quantization, fine-tuning
- **r/MachineLearning** — research discussion
- **EleutherAI Discord** — open-source LLM research

---

*This guide is designed for Java developers making the transition to AI engineering.
The best way to learn is to build — start with Chapter 17's code, run it, modify it,
break it, and understand every line. The math matters, but the intuition matters more.
The Java mindset (systems thinking, production quality, type safety) is your biggest
advantage over data scientists who've never built production software.*

*Good luck on your journey from Java to AI engineering.*
