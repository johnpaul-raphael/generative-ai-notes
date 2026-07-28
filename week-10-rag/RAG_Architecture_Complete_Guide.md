# 🚀 RAG Architecture: The Complete Senior Engineer's Guide

> **A comprehensive interview preparation document covering Retrieval-Augmented Generation (RAG) from fundamentals to production-grade implementation.**

---

## 📖 Table of Contents

1. [What is RAG and Why Do We Need It?](#1-what-is-rag-and-why-do-we-need-it)
2. [The Two Phases of RAG](#2-the-two-phases-of-rag)
3. [Step 1: Document Loading](#3-step-1-document-loading)
4. [Step 2: Chunking Strategies (Deep Dive)](#4-step-2-chunking-strategies-deep-dive)
5. [Step 3: Embeddings (Deep Dive)](#5-step-3-embeddings-deep-dive)
6. [Step 4: Vector Database Storage](#6-step-4-vector-database-storage)
7. [Step 5: The Retrieval Phase](#7-step-5-the-retrieval-phase)
8. [Step 6: Augmentation & Generation](#8-step-6-augmentation--generation)
9. [Production Architecture Patterns](#9-production-architecture-patterns)
10. [Evaluation & Metrics](#10-evaluation--metrics)
11. [Common Failure Modes](#11-common-failure-modes)
12. [End-to-End Example](#12-end-to-end-example)
13. [Interview Q&A Cheat Sheet](#13-interview-qa-cheat-sheet)

---

# 1. What is RAG and Why Do We Need It?

## 🤔 The Problem

Imagine you have a very smart friend (an LLM like GPT or Claude) who has read millions of books **up to a certain date**. Now you ask them:

> "What does my company's internal HR policy say about remote work?"

Your friend has two problems:
1. **They never read your company's documents** (private data).
2. **Their knowledge has a cutoff date** (they don't know recent events).

So they'll either say "I don't know" or worse, **hallucinate** (make up a confident-sounding wrong answer).

## ✅ The Solution: RAG

**RAG (Retrieval-Augmented Generation)** solves this by giving the LLM a "cheat sheet" right before it answers.

> Instead of relying only on what the LLM memorized during training, we **retrieve** relevant information from an external knowledge base and **augment** the prompt with it, so the LLM can **generate** an accurate, grounded answer.

That's literally where the name comes from:
- **R**etrieval → Search and fetch relevant documents
- **A**ugmented → Add those documents to the prompt
- **G**eneration → LLM generates the final answer

### 🧮 Core Formula:

$$
\text{Answer} = \text{LLM}(\text{Query} + \text{Retrieved Context})
$$

### 🎯 Tiny Example:

| Without RAG | With RAG |
|------------|----------|
| **Q:** What is our company's remote work policy? | **Q:** What is our company's remote work policy? |
| **A:** "I don't have access to your company's internal policies." | **A:** "Employees can work remotely up to 3 days a week..." (grounded in retrieved HR doc) |

## 💡 Why RAG is Better Than Fine-Tuning

| Aspect | Fine-Tuning | RAG |
|--------|------------|-----|
| **Cost** | Expensive (training) | Cheap (only inference) |
| **Update Frequency** | Need to retrain | Just add new docs |
| **Hallucination Risk** | High | Low (grounded in facts) |
| **Source Attribution** | Impossible | Easy (cite chunks) |
| **Data Privacy** | Data baked into model | Data stays in your DB |
| **Setup Time** | Days/weeks | Hours |

---

# 2. The Two Phases of RAG

RAG has **two phases**:

## 🔵 OFFLINE Phase (Indexing)
- Happens **once** (or periodically when adding new docs)
- No user involved
- Prepares the knowledge base
- Like a librarian organizing books on shelves

## 🟢 ONLINE Phase (Retrieval + Generation)
- Happens **every time** a user asks a question
- Real-time
- Uses the prepared knowledge base
- Like the librarian helping a reader find a book

### 📊 Visual Overview:

```
┌─────────────────────────────────────────────────────────┐
│                  OFFLINE PHASE (One-time)               │
│                                                         │
│  Documents → Chunking → Embeddings → Vector DB          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              ONLINE PHASE (Per User Query)              │
│                                                         │
│  User Query → Embed Query → Search Vector DB →          │
│  Retrieve Top-K Chunks → Build Prompt → LLM → Answer    │
└─────────────────────────────────────────────────────────┘
```

---

# 3. Step 1: Document Loading

The first step is gathering all the documents your RAG system needs to know about.

## 📁 Common Document Sources:

| Source Type | Examples | Tools/Libraries |
|------------|----------|----------------|
| Files | PDFs, Word, PPT, Excel, CSV | `PyPDF2`, `python-docx`, `unstructured` |
| Web Pages | Blog posts, documentation | `BeautifulSoup`, `Playwright`, `Selenium` |
| Databases | SQL tables, NoSQL collections | Direct DB connectors |
| APIs | REST endpoints, GraphQL | `requests`, custom connectors |
| Knowledge Bases | Notion, Confluence, SharePoint | Official APIs/connectors |
| Email | Gmail, Outlook | IMAP, Gmail API |
| Chat | Slack, Teams transcripts | Slack API |

## 🛠️ Loading Best Practices:

1. **Preserve metadata** — Always keep source filename, page numbers, authors, dates
2. **Handle encoding issues** — UTF-8 by default, but watch for old files
3. **OCR for scanned PDFs** — Use Tesseract or AWS Textract
4. **Parse tables carefully** — Tables often lose structure during extraction
5. **Strip noise** — Remove headers, footers, page numbers from PDFs

---

# 4. Step 2: Chunking Strategies (Deep Dive)

## 🤷 Why Chunk Documents?

You **cannot** dump a 50-page PDF into the LLM. Why?
- LLMs have a **context window limit** (e.g., 8k, 32k, 128k tokens)
- Even if it fits, the LLM gets distracted by irrelevant content
- Retrieval works better on smaller, focused pieces
- Embedding models also have token limits

**Solution:** Split documents into **chunks** — typically 200 to 500 words each, often with **overlap**.

---

## 🧩 Strategy 1: Fixed-Size Chunking

**How it works:** Split text into chunks of a fixed number of characters or tokens (e.g., every 500 tokens), optionally with overlap.

### Formula:

$$
\text{Chunk}_i = \text{Text}[i \cdot (S - O) : i \cdot (S - O) + S]
$$

Where:
- $S$ = chunk size (e.g., 500 tokens)
- $O$ = overlap size (e.g., 50 tokens)
- $i$ = chunk index (0, 1, 2, ...)

### Example:
```
Original text (1200 tokens) with S=500, O=50:
Chunk 1: tokens [0   → 500]
Chunk 2: tokens [450 → 950]   ← 50 token overlap with Chunk 1
Chunk 3: tokens [900 → 1200]
```

### ✅ When to use:
- Quick prototyping or POCs
- Uniform text (logs, transcripts, plain articles)
- When document structure doesn't matter much

### ❌ When NOT to use:
- Structured documents (code, markdown, legal contracts)
- Can cut sentences/paragraphs awkwardly, hurting meaning

---

## 🧩 Strategy 2: Recursive Character Splitting

**How it works:** Try to split on **natural boundaries** in order of priority: paragraphs → sentences → words → characters. Stop as soon as chunks fit the size limit.

### Example Priority Order:
```
["\n\n", "\n", ". ", " ", ""]
```

It first tries splitting by double newlines (paragraphs). If a chunk is still too big, it splits that chunk by single newlines, then by sentences, and so on.

### Example Output:
```
Chunk 1: "Remote work is allowed up to 3 days per week. 
         Manager approval is needed for exceptions."  ← split at paragraph
         
Chunk 2: "Employees must maintain core working hours of 
         10 AM to 4 PM regardless of location."        ← split at paragraph
```

### ✅ When to use:
- **Default choice for most RAG systems** — great balance
- General-purpose documents (articles, blogs, books, reports)
- When you don't know the document structure upfront

### ❌ When NOT to use:
- Highly structured data (code, tables) — use specialized splitters

---

## 🧩 Strategy 3: Document-Structure-Aware Chunking

**How it works:** Use the document's natural structure — markdown headers, HTML tags, code functions, JSON keys — as chunk boundaries.

### Example (Markdown):
```markdown
# Remote Work Policy           ← becomes Chunk 1
## Eligibility                 ← becomes Chunk 2
## Approval Process            ← becomes Chunk 3
```

Each section becomes its own chunk, preserving the **logical hierarchy**.

### ✅ When to use:
- Markdown documentation, technical docs
- HTML pages, Confluence/Notion exports
- Source code (split by function/class)
- Legal documents (split by clause)

### ❌ When NOT to use:
- Unstructured text (raw transcripts, plain prose)

---

## 🧩 Strategy 4: Semantic Chunking

**How it works:** Use embeddings to detect **topic changes**. Split where adjacent sentences become semantically different.

### Algorithm:
1. Split text into sentences
2. Embed each sentence
3. Compute similarity between adjacent sentences
4. Where similarity drops below a threshold → split there

### Formula:

$$
\text{Split if: } \text{sim}(\vec{s_i}, \vec{s_{i+1}}) < \tau
$$

Where $\tau$ is your threshold (e.g., 0.7) and $\text{sim}$ is cosine similarity.

### Example:
```
Sentence 1: "Remote work is allowed 3 days a week."
Sentence 2: "Manager approval is required for more."
Sentence 3: "Annual leave totals 20 days per year."   ← topic shift!
```
Similarity between S2 and S3 is low → split here.

### ✅ When to use:
- High-quality production systems
- Documents covering multiple topics
- When retrieval accuracy matters more than indexing speed

### ❌ When NOT to use:
- Real-time/low-budget systems (it's expensive — needs embeddings during indexing)

---

## 🧩 Strategy 5: Agentic / LLM-Based Chunking

**How it works:** Use an LLM itself to decide where to chunk — almost like asking "where would a human split this?"

### ✅ When to use:
- Complex documents (research papers, legal contracts)
- When you have budget for high quality

### ❌ When NOT to use:
- Large-scale indexing (very slow and expensive)

---

## 📊 Chunking Decision Table:

| Document Type | Recommended Strategy |
|---------------|---------------------|
| Plain articles, blogs | **Recursive** |
| Markdown / HTML docs | **Structure-aware** |
| Source code | **Structure-aware** (by function) |
| Legal / Medical | **Semantic** or **Agentic** |
| Chat logs / Transcripts | **Fixed-size** with overlap |
| Mixed/Unknown | **Recursive** (safe default) |

## 💡 Senior Engineer's Chunking Tips:

1. **Start with Recursive Character Splitting** — safe default
2. **Common chunk sizes:** 256, 512, or 1024 tokens
3. **Overlap:** 10-20% of chunk size (e.g., 50 tokens overlap for 500-token chunks)
4. **Why overlap matters:** Preserves context across chunk boundaries
5. **Measure first, optimize later** — don't over-engineer day one

---

# 5. Step 3: Embeddings (Deep Dive)

## 🔢 What Are Embeddings?

Computers don't understand text — they understand **numbers**. We convert each chunk into a **vector** (a list of numbers, typically 384, 768, or 1536 dimensions) using an **embedding model**.

### Formula:

$$
\vec{v} = \text{EmbeddingModel}(\text{chunk})
$$

Where $\vec{v} \in \mathbb{R}^d$ (a vector of $d$ dimensions, e.g., $d=768$).

### 🪄 The Magic of Embeddings:

Embeddings capture **semantic meaning**. Texts with similar meaning end up close together in vector space.

**Example:**
```
"remote work policy"   → [0.12, -0.45, 0.78, ..., 0.33]
"work from home rules" → [0.14, -0.43, 0.76, ..., 0.31]  ← very similar!
"company holiday list" → [0.89, 0.21, -0.55, ..., -0.12] ← very different
```

Even though "remote work" and "work from home" use different words, their **vectors are close** because the meaning is similar.

---

## ⚠️ Critical Concept: Tokenization vs Embedding

Beginners often confuse these. They are **two different things**.

### 🔹 Tokenization
**What it does:** Breaks text into smaller pieces called **tokens** (words, sub-words, or characters).

```
Input:  "Remote work policy"
Output: ["Remote", " work", " policy"]   ← 3 tokens
```

Tokenization just **splits text** — it doesn't capture meaning.

### 🔹 Embedding
**What it does:** Converts those tokens into a **vector of numbers** that captures meaning.

```
Input:  "Remote work policy"
Output: [0.12, -0.45, 0.78, ..., 0.33]   ← 1536 numbers (a vector)
```

### 🔄 The Full Pipeline:
```
Text → [Tokenizer] → Tokens → [Embedding Model] → Vector
```

---

## 🚨 The "Same Model" Rule

> **The embedding model used during OFFLINE indexing MUST be the same as the embedding model used during ONLINE query.**

### 🗺️ Why? Map Analogy:

Imagine two people drawing maps of the **same city**:
- **Person A** uses English labels and kilometers
- **Person B** uses French labels and miles

Both maps show the same city, but the **coordinate systems are different**. If you take a location from Person A's map and try to find it on Person B's map, you'll end up in the wrong place — even though the city is the same!

That's exactly what happens with embedding models.

### ❌ WRONG Approach:
```
Documents → BGE Embeddings  → Vector DB
Query     → OpenAI Embeddings → Search DB
                                   ↓
                          GARBAGE RESULTS!
```

### ✅ CORRECT Approach:
```
Documents → BGE Embeddings → Vector DB
Query     → BGE Embeddings → Search DB
                                ↓
                       ACCURATE RESULTS!
```

### 🎯 Key Insight: Embedding Model ≠ LLM

These are **two completely independent** model choices:

| Aspect | Embedding Model | LLM |
|--------|----------------|-----|
| **Purpose** | Convert text → vectors | Generate answers |
| **Must be same** | Yes (offline = online) | No constraint |
| **Cost** | Free if self-hosted | Pay per token |
| **Hosting** | Easy (small model) | Hard (huge model) |

**You can mix providers freely:**
- Local BGE embeddings + OpenAI GPT-4o LLM ✅
- OpenAI embeddings + Claude LLM ✅
- Cohere embeddings + Anthropic LLM ✅

---

## 🏆 How to Choose an Embedding Model

### 5 Key Factors:

#### 1️⃣ Quality (MTEB Benchmark)
**MTEB (Massive Text Embedding Benchmark)** is the industry-standard leaderboard. Higher MTEB score = better retrieval quality.

#### 2️⃣ Dimension Size
- Small: 384 dims (e.g., `all-MiniLM-L6-v2`)
- Medium: 768 dims (e.g., `BGE-base`)
- Large: 1536–3072 dims (e.g., OpenAI `text-embedding-3-large`)

### Storage Trade-off:

$$
\text{Storage Cost} \propto N \times d
$$

Where $N$ = number of chunks, $d$ = dimensions.

#### 3️⃣ Context Window (Max Tokens)
- `all-MiniLM-L6-v2`: 256 tokens (small!)
- `OpenAI text-embedding-3-large`: 8191 tokens (huge!)

If your chunks exceed the model's limit, it **truncates** and you lose information.

#### 4️⃣ Cost & Hosting
- **API-based** (OpenAI, Cohere, Voyage) — pay per token, no infra
- **Open-source** (BGE, E5, MiniLM) — free, but self-host

#### 5️⃣ Domain Specificity
General-purpose models work for general text. For specialized domains (medical, legal, code), use domain-specific models.

---

## 📊 Top Embedding Models (2026):

| Model | Dims | Max Tokens | Type | Best For |
|-------|------|-----------|------|----------|
| **OpenAI text-embedding-3-small** | 1536 | 8191 | API | Production, general use, budget |
| **OpenAI text-embedding-3-large** | 3072 | 8191 | API | Highest quality, willing to pay |
| **Cohere embed-v3** | 1024 | 512 | API | Multilingual |
| **Voyage AI voyage-3** | 1024 | 32000 | API | Long documents, top MTEB |
| **BGE-large-en-v1.5** | 1024 | 512 | Open-source | Self-hosted, high quality |
| **E5-large-v2** | 1024 | 512 | Open-source | Self-hosted alternative |
| **all-MiniLM-L6-v2** | 384 | 256 | Open-source | Cheap, fast prototypes |
| **CodeBERT** | 768 | 512 | Open-source | Code search |
| **BioBERT** | 768 | 512 | Open-source | Medical text |

## 🎯 Decision Framework:

| Your Need | Recommended Model |
|-----------|------------------|
| Best price/performance (API) | OpenAI `text-embedding-3-small` |
| Highest quality (API) | OpenAI `text-embedding-3-large` or Voyage AI |
| Self-hosted, high quality | `BGE-large-en-v1.5` |
| Self-hosted, lightweight | `all-MiniLM-L6-v2` |
| Multilingual | Cohere `embed-v3` |
| Long documents (>512 tokens) | OpenAI or Voyage AI |
| Code search | CodeBERT |
| Medical | BioBERT / ClinicalBERT |
| Legal | Legal-BERT |

## ⚠️ Common Mistakes:

1. **Picking the biggest model blindly** — overkill for small datasets
2. **Mixing embedding models** between offline and online
3. **Not matching chunk size to model's max tokens** — causes truncation
4. **Ignoring MTEB scores for your domain**

---

# 6. Step 4: Vector Database Storage

After embedding, we store all `(chunk, vector, metadata)` pairs in a **vector database**.

## 📦 What Gets Stored:

| ID | Chunk Text | Vector (Embedding) | Metadata |
|----|-----------|--------------------|----------|
| 1 | "Employees can work remotely 3 days..." | [0.12, -0.45, ...] | source: HR.pdf, page: 5 |
| 2 | "Manager approval required for..." | [0.14, -0.43, ...] | source: HR.pdf, page: 6 |
| 3 | "Annual leave is 20 days..." | [0.55, 0.22, ...] | source: leave.pdf, page: 2 |

## 🏪 Popular Vector Databases:

| Database | Type | Best For |
|----------|------|----------|
| **Pinecone** | Managed cloud | Production, no infra worry |
| **Weaviate** | Open-source / cloud | Hybrid search, GraphQL |
| **ChromaDB** | Open-source / local | Prototypes, small projects |
| **FAISS** | Library (Meta) | Fast in-memory, research |
| **Milvus** | Open-source / cloud | Large-scale (billions of vectors) |
| **Qdrant** | Open-source / cloud | Performance-focused |
| **pgvector** | Postgres extension | Already using Postgres |
| **Elasticsearch** | Search engine | Already using ES, hybrid search |

## 🎯 How to Choose:

| Scale | Recommendation |
|-------|---------------|
| < 10k chunks | **ChromaDB** or **FAISS** (local, free) |
| 10k – 1M chunks | **Qdrant**, **Weaviate** (self-hosted or cloud) |
| 1M – 100M chunks | **Pinecone**, **Milvus** |
| > 100M chunks | **Milvus**, **Vespa** |

## 💡 Best Practice: Store Embedding Model Metadata

```json
{
  "chunk": "Remote work is allowed 3 days...",
  "vector": [0.12, -0.45, ...],
  "metadata": {
    "source": "hr_policy.pdf",
    "page": 5,
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "model_version": "v1.5",
    "indexed_at": "2026-05-17"
  }
}
```

This prevents silent failures if someone uses a different model later.

---

# 7. Step 5: The Retrieval Phase

This is the **R** in RAG — the online phase. Given a user query, find the most relevant chunks **as fast and accurately as possible**.

## 7.1 Embed the User Query

Convert the user's query into a vector — using the **same embedding model** as indexing.

### Formula:

$$
\vec{q} = \text{EmbeddingModel}(\text{user\_query})
$$

### Example:
```
User Query: "What is our remote work policy?"
       ↓ BGE-small
Vector: [0.14, -0.43, 0.76, ..., 0.31]   ← 384 dimensions
```

### 💡 Query Preprocessing Tips:
- **Modern models handle case/punctuation** — no aggressive cleaning needed
- **Spell correction** — helpful if users make typos
- **Query expansion** — rewrite to improve recall
- **Caching** — cache common queries' embeddings to save compute

---

## 7.2 Similarity Search — The Heart of Retrieval

### 📐 Metric 1: Cosine Similarity (Most Popular)

Measures the **angle** between two vectors, ignoring magnitude.

### Formula:

$$
\text{cosine\_sim}(\vec{q}, \vec{v}) = \frac{\vec{q} \cdot \vec{v}}{\|\vec{q}\| \cdot \|\vec{v}\|} = \frac{\sum_{i=1}^{d} q_i v_i}{\sqrt{\sum_{i=1}^{d} q_i^2} \cdot \sqrt{\sum_{i=1}^{d} v_i^2}}
$$

Range: $[-1, 1]$
- $1$ = identical direction (very similar)
- $0$ = orthogonal (unrelated)
- $-1$ = opposite direction (very dissimilar)

### Concrete Example:
```
Query vector:   q = [1, 2, 3]
Chunk A vector: a = [2, 4, 6]    ← same direction, just scaled
Chunk B vector: b = [-1, -2, -3] ← opposite direction
```

**Cosine sim(q, a):**

$$
= \frac{(1)(2) + (2)(4) + (3)(6)}{\sqrt{1^2+2^2+3^2} \cdot \sqrt{2^2+4^2+6^2}} = \frac{28}{\sqrt{14} \cdot \sqrt{56}} = \frac{28}{28} = 1.0
$$

**Cosine sim(q, b) = -1.0** (perfectly opposite)

---

### 📐 Metric 2: Dot Product

Same as numerator of cosine, without normalization.

### Formula:

$$
\text{dot}(\vec{q}, \vec{v}) = \vec{q} \cdot \vec{v} = \sum_{i=1}^{d} q_i v_i
$$

### ✅ When to use:
- Vectors are already **normalized** (length = 1) → faster than cosine
- OpenAI embeddings are normalized → dot product is optimal

---

### 📐 Metric 3: Euclidean Distance (L2)

Measures **straight-line distance** between two points.

### Formula:

$$
\text{euclidean}(\vec{q}, \vec{v}) = \sqrt{\sum_{i=1}^{d} (q_i - v_i)^2}
$$

Range: $[0, \infty)$ — smaller = more similar.

### ❌ When NOT to use:
- High-dimensional text embeddings (curse of dimensionality)

---

### 🎯 Metric Selection Matrix:

| Embedding Model | Best Metric |
|-----------------|-------------|
| OpenAI `text-embedding-3-*` | **Cosine** or dot (normalized) |
| BGE / E5 | **Cosine** |
| Sentence-Transformers | **Cosine** |
| Custom unnormalized | Euclidean or cosine, test both |

**Default rule:** Use **cosine similarity** unless you have a specific reason not to.

---

## 7.3 Top-K Retrieval

After computing similarity with all chunks, sort and pick the top **K**.

### Formula:

$$
\text{TopK}(\vec{q}, K) = \arg\max_{S \subseteq D, |S|=K} \sum_{\vec{v} \in S} \text{sim}(\vec{q}, \vec{v})
$$

In plain English: *"Return the K chunks with the highest similarity scores."*

### Example:
```
Chunk 47:  similarity = 0.89  ← top 1
Chunk 12:  similarity = 0.85  ← top 2
Chunk 153: similarity = 0.82  ← top 3
Chunk 88:  similarity = 0.81  ← top 4
Chunk 199: similarity = 0.79  ← top 5
```

### 💡 Choosing K:

| K Value | Pros | Cons |
|---------|------|------|
| **1–2** | Fast, low cost | May miss relevant info |
| **3–5** | Sweet spot for most use cases | — |
| **10+** | High recall | More tokens = higher LLM cost + noise |

### ⚠️ Watch Out for "Context Pollution":
Increasing K doesn't always help — irrelevant chunks **distract** the LLM. Research calls this **"lost in the middle"** — LLMs pay less attention to chunks in the middle of long contexts.

---

## 7.4 Brute-Force vs Approximate Nearest Neighbor (ANN)

### ❌ Naive Approach: Brute-Force Search

Compute similarity between query and **every single chunk**.

### Complexity:

$$
O(N \cdot d)
$$

Where $N$ = number of chunks, $d$ = dimensions.

**For 200 chunks:** trivially fast (few milliseconds)
**For 10M chunks with 1536 dims:** painfully slow (seconds per query)

### ✅ Optimized Approach: ANN Algorithms

#### 1️⃣ HNSW (Hierarchical Navigable Small World) — Most Popular
- Multi-layer graph structure
- Search starts at top (sparse) layer, navigates down
- **Complexity:** $O(\log N)$
- **Trade-off:** Slightly approximate, but >95% recall

#### 2️⃣ IVF (Inverted File Index)
- Clusters vectors into groups (k-means)
- At query time, only search closest few clusters

#### 3️⃣ PQ (Product Quantization)
- Compresses vectors (e.g., 1536-dim float → 32 bytes)
- Tiny accuracy hit, massive memory savings
- Often combined as **IVF-PQ**

### 💡 When to Use ANN:

| Dataset Size | Approach |
|-------------|----------|
| < 100k chunks | **Brute-force** is fine |
| 100k – 1M | Consider **HNSW** |
| > 1M | **ANN mandatory** (HNSW or IVF-PQ) |

---

## 7.5 Advanced Retrieval Techniques

### 🔹 Technique 1: Hybrid Search (Vector + Keyword)

**Problem:** Pure vector search misses exact keyword matches (e.g., "CVE-2024-12345").

**Solution:** Combine vector search with traditional keyword search (BM25).

### Formula:

$$
\text{score}(\vec{q}, \vec{v}) = \alpha \cdot \text{cosine\_sim}(\vec{q}, \vec{v}) + (1 - \alpha) \cdot \text{BM25}(\text{query}, \text{chunk})
$$

Where $\alpha \in [0, 1]$ is the weight (e.g., 0.7 = 70% vector, 30% keyword).

### When to use:
- Domain has technical terms, codes, names, IDs
- Mix of semantic + exact-match needs

---

### 🔹 Technique 2: Re-ranking

After retrieving top-K with vector search, use a **more powerful (but slower) model** to re-rank.

### Pipeline:
```
Vector Search → Top 20 chunks (fast, approximate)
       ↓
Cross-Encoder Re-ranker → Top 5 chunks (slow, accurate)
       ↓
Send to LLM
```

### Bi-Encoder vs Cross-Encoder:

| Aspect | Bi-Encoder (embeddings) | Cross-Encoder (re-ranker) |
|--------|------------------------|---------------------------|
| Input | Query and chunk separately | Query and chunk **together** |
| Speed | Fast | Slow (~10-100x slower) |
| Accuracy | Good | Better |
| Use | Initial retrieval | Re-rank top candidates |

**Popular re-rankers:** `ms-marco-MiniLM-L-12-v2`, Cohere Rerank API.

---

### 🔹 Technique 3: Metadata Filtering

Add structured filters alongside vector search.

```python
results = vector_db.search(
    query_vector=q,
    filter={"department": "HR", "year": 2025},
    top_k=5
)
```

### When to use:
- Documents have natural categories
- Multi-tenant systems (per-user filtering)

---

### 🔹 Technique 4: Query Transformation

#### a) HyDE (Hypothetical Document Embeddings)
Ask an LLM to generate a hypothetical answer, embed THAT, then search.

```
User Query: "What is the leave policy?"
HyDE: LLM generates "Our company offers 20 days of paid leave..."
Embed the hypothetical answer → search DB
```

**Why?** Document chunks look more like answers than questions. Embedding a hypothetical answer matches better.

#### b) Multi-Query
Generate 3-5 variations of the query, search with each, merge results.

```
Original: "Can I work from home?"
Variations:
  1. "Remote work policy"
  2. "Work from home eligibility"
  3. "Telecommuting rules"
```

---

## 7.6 Full Retrieval Pipeline (Production)

```
User Query
   │
   ▼
[Query Preprocessing]
   - Spell check
   - Optional: Query expansion / HyDE
   │
   ▼
[Embed Query] → Same embedding model as indexing
   │
   ▼
[Vector Search] (ANN or brute-force)
   - Optional: Metadata filters
   - Retrieve top 20-50 candidates
   │
   ▼
[Hybrid Score] (optional)
   - Combine with BM25 keyword score
   │
   ▼
[Re-ranking] (optional)
   - Cross-encoder narrows to top 3-5
   │
   ▼
[Final Top-K Chunks]
   │
   ▼
Pass to LLM (Step 6)
```

---

# 8. Step 6: Augmentation & Generation

The **A** and **G** in RAG. This is where retrieved chunks meet the LLM.

## 8.1 The Core Formula

$$
\text{Answer} = \text{LLM}\left(\text{Prompt}(\text{Query}, \text{Retrieved\_Chunks}, \text{Instructions})\right)
$$

Where `Prompt(...)` combines:
1. **System instructions** — Rules for the LLM
2. **Retrieved context** — Top-K chunks
3. **User query** — Original question

## 8.2 Anatomy of a Production-Grade RAG Prompt

```
You are a helpful HR assistant for ACME Corp. Answer the user's question 
based ONLY on the provided context below. 

Rules:
1. If the answer is not in the context, say "I don't have information about that."
2. Do not make up information.
3. Cite the source document for each fact (e.g., [Source: HR_Policy.pdf]).
4. Keep your answer concise and professional.

---

CONTEXT:
[Source: remote_work_policy.pdf, Chunk 1]
Employees can work remotely up to 3 days per week. Manager approval is 
required for any exceptions.

[Source: remote_work_policy.pdf, Chunk 2]
Core working hours are 10 AM to 4 PM regardless of work location.

[Source: hr_handbook.pdf, Chunk 3]
All remote work arrangements must be documented in the HR system 
within 5 business days.

---

USER QUESTION:
What is the remote work policy at ACME Corp?

ANSWER:
```

### 🔬 Breakdown of Each Component:

| Component | Purpose |
|-----------|---------|
| **Role/Persona** | Shapes tone, vocabulary, behavior |
| **Grounding Instruction** | "ONLY use context" — prevents hallucination |
| **Behavioral Rules** | Each rule prevents a specific failure mode |
| **Context Block** | Retrieved chunks with source metadata |
| **User Query** | Placed AFTER context (matters for attention) |

---

## 8.3 Critical Prompt Engineering Decisions

### 🔹 Decision 1: Order of Components

**"Lost in the Middle" research:** LLMs pay most attention to the beginning and end, least to the middle.

### Best Practice:
```
[System Instructions]   ← Beginning (high attention)
[Context Chunks]        ← Middle (lower attention)
[User Query]            ← End (high attention)
```

---

### 🔹 Decision 2: How to Format Chunks

#### ✅ Option C: XML-style Tags (Recommended)
```xml
<context>
  <chunk source="remote_work_policy.pdf" page="5">
    Employees can work remotely up to 3 days per week.
  </chunk>
  <chunk source="remote_work_policy.pdf" page="6">
    Manager approval is required for exceptions.
  </chunk>
</context>
```

**Why XML works best:**
- LLMs trained to recognize structured tags
- Easy to include metadata
- Enables citations
- Clear chunk boundaries

---

### 🔹 Decision 3: Token Budget Management

Every LLM has a context window limit:

$$
T_{\text{total}} = T_{\text{system}} + T_{\text{context}} + T_{\text{query}} + T_{\text{answer}} \leq T_{\text{max}}
$$

### Context Windows (2026):

| Model | Context Window |
|-------|---------------|
| GPT-4o | 128k tokens |
| GPT-4o-mini | 128k tokens |
| Claude Opus 4 | 200k tokens |
| Gemini 1.5 Pro | 2M tokens |

### Practical Guideline:
- Aim for **2,000–8,000 tokens of context** (3–10 chunks)
- **Quality > Quantity**
- Don't max out — costs money, slows latency, pollutes context

---

## 8.4 LLM Generation Parameters

### 🎛️ Temperature
Controls randomness/creativity.

### Formula:

$$
P(\text{token}_i) = \frac{\exp(\text{logit}_i / T)}{\sum_j \exp(\text{logit}_j / T)}
$$

- $T \to 0$: Deterministic (picks most likely)
- $T = 1$: Standard
- $T > 1$: More creative

### RAG Recommendation:
**Temperature = 0 to 0.3** — you want factual, grounded answers.

### 🎛️ Top-p (Nucleus Sampling)
Considers tokens whose cumulative probability ≤ p.
- **Default p = 0.9** is fine for RAG

### 🎛️ Max Tokens
Hard cap on output length.
- FAQ: 200-500 tokens
- Detailed: 1000-2000 tokens

### 🎛️ Stop Sequences
Tell LLM where to stop. E.g., stop at `\n\nQuestion:` to prevent fake Q&A continuation.

---

## 8.5 Sample API Call (Production)

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": augmented_prompt}
    ],
    temperature=0.1,        # Low for factual accuracy
    max_tokens=800,         # Limit response length
    top_p=0.9
)

answer = response.choices[0].message.content
```

---

## 8.6 Advanced Generation Techniques

### 🔹 1. Citation Enforcement
Force LLM to cite chunks:
```
For every fact, cite the chunk: [chunk_id].
Example: "Employees can work remotely 3 days [chunk_1]."
```

### 🔹 2. Confidence Calibration
Make LLM express uncertainty:
```
If context partially answers: "Based on available info, X. May not cover all cases."
If context doesn't answer: "I don't have enough information."
```

### 🔹 3. Chain-of-Thought
For complex questions:
```
Before answering, reason through the context step by step, 
then provide your final answer.

Reasoning: [...]
Final Answer: [...]
```

### 🔹 4. Structured Output (JSON Mode)
```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    response_format={"type": "json_object"}
)
```

### 🔹 5. Streaming Responses
```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content, end="", flush=True)
```

Better UX — users see tokens appearing live.

---

# 9. Production Architecture Patterns

## 🏗️ Pattern 1: Budget-Conscious (Local + API LLM)

**Best for:** Small teams, low traffic, cost-sensitive projects

```
┌─────────────────────────────────────────────────────┐
│  OFFLINE (One-time, Free)                           │
│                                                     │
│  Documents → Chunking → BGE Embeddings → ChromaDB   │
│                          (local)         (local)    │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  ONLINE (Per query)                                 │
│                                                     │
│  User Query → BGE Embed → ChromaDB Search →         │
│  Top-K Chunks + Query → OpenAI API → Answer         │
│                                                     │
│  Cost per query: ~$0.0004                           │
└─────────────────────────────────────────────────────┘
```

### Cost Estimate (100 queries/day):
- Embedding (local): **$0**
- Vector DB (local): **$0**
- LLM API (GPT-4o-mini): **~$1-3/month**
- **Total: ~$3/month** 🎉

---

## 🏗️ Pattern 2: Full Cloud (Managed Services)

**Best for:** Production, scale, no infra team

```
Documents → AWS S3
       ↓
LangChain/LlamaIndex Pipeline
       ↓
OpenAI Embeddings → Pinecone
       ↓
User Query → API Gateway → Lambda
       ↓
Pinecone Search → Top-K → OpenAI GPT-4o → Response
```

### Components:
- **Storage:** AWS S3 / Azure Blob
- **Embeddings:** OpenAI / Cohere
- **Vector DB:** Pinecone / Weaviate Cloud
- **LLM:** OpenAI / Anthropic / Azure OpenAI
- **Orchestration:** LangChain / LlamaIndex

---

## 🏗️ Pattern 3: Self-Hosted (Privacy-First)

**Best for:** Healthcare, finance, regulated industries

```
Documents → On-prem storage
       ↓
Open-source embedding (BGE) → Self-hosted Qdrant
       ↓
User Query → Internal API → BGE Embed → Qdrant
       ↓
Self-hosted LLM (Llama 3 / Mistral) → Response
```

### Components:
- **Storage:** On-premises
- **Embeddings:** BGE / E5 (self-hosted)
- **Vector DB:** Qdrant / Milvus (self-hosted)
- **LLM:** Llama 3 / Mixtral (self-hosted on GPUs)

---

# 10. Evaluation & Metrics

## 📏 Why Evaluate?

You can't improve what you don't measure. RAG has multiple failure points — bad chunking, bad retrieval, bad generation. Metrics tell you **where** to fix.

## 🎯 Retrieval Metrics

### 1️⃣ Precision@K
What fraction of retrieved chunks are actually relevant?

$$
\text{Precision@K} = \frac{|\text{Relevant chunks in top-K}|}{K}
$$

### 2️⃣ Recall@K
What fraction of all relevant chunks did we retrieve?

$$
\text{Recall@K} = \frac{|\text{Relevant chunks in top-K}|}{|\text{All relevant chunks}|}
$$

### 3️⃣ MRR (Mean Reciprocal Rank)
How high (in rank order) does the first relevant chunk appear?

$$
\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}
$$

### 4️⃣ NDCG (Normalized Discounted Cumulative Gain)
Rewards relevant chunks appearing higher in the ranking.

---

## 🎯 Generation Metrics

### 1️⃣ Faithfulness
Does the answer stick to the retrieved context (no hallucination)?

### 2️⃣ Answer Relevance
Does the answer actually address the question?

### 3️⃣ Context Precision
Are retrieved chunks relevant to the question?

### 4️⃣ Context Recall
Are all needed chunks retrieved?

---

## 🛠️ Evaluation Frameworks

### RAGAS (RAG Assessment)
Open-source framework for RAG evaluation.

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

results = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
```

### Other Tools:
- **TruLens** — Trace-based evaluation
- **DeepEval** — Pytest-style unit tests for LLMs
- **LangSmith** — LangChain's eval platform
- **Phoenix (Arize)** — Open-source observability

---

# 11. Common Failure Modes

| Failure Mode | Symptom | Root Cause | Fix |
|--------------|---------|-----------|-----|
| **Hallucination** | LLM invents facts | Weak grounding | Strengthen "based ONLY on context"; lower temperature |
| **Ignoring context** | LLM uses training data | Bad prompt structure | Put query after context; repeat grounding |
| **Context overflow** | Prompt exceeds limit | Too many chunks | Reduce K; summarize chunks |
| **Wrong chunks retrieved** | Irrelevant context | Bad embedding/chunking | Improve retrieval; re-rank; hybrid search |
| **Verbose answers** | Long, unfocused | No length control | Add "Keep under N words" |
| **No "I don't know"** | LLM guesses | Missing instruction | Explicit "say I don't know" rule |
| **Mixed embeddings** | Garbage retrieval | Different models offline/online | Always use same model |
| **Stale data** | Outdated answers | No re-indexing | Add periodic re-indexing pipeline |
| **Citation errors** | Wrong sources cited | LLM confusion | Use structured chunk IDs |
| **Lost in middle** | Missing key info | Too many chunks | Reduce K or re-rank top |

---

# 12. End-to-End Example

Let's trace a real query through a complete RAG system.

## 🎬 Scenario:
**Setup:** HR chatbot with 200 indexed chunks from company policies.

**Stack:**
- Embeddings: `BAAI/bge-small-en-v1.5` (local)
- Vector DB: ChromaDB (local)
- LLM: OpenAI `gpt-4o-mini` (API)

## 📥 User Query:
> "Can I work from home 4 days a week?"

---

## 🔄 Step-by-Step Trace:

### Step 1: Query Embedding (Local BGE)
```python
query = "Can I work from home 4 days a week?"
query_vector = bge_model.encode(query)
# Output: [0.14, -0.43, 0.76, ..., 0.31]  (384 dimensions)
# Latency: ~20ms
```

### Step 2: Vector Search (Local ChromaDB)
```python
results = chroma_db.query(
    query_embeddings=[query_vector],
    n_results=5
)
# Latency: ~5ms (brute-force on 200 chunks)
```

### Step 3: Top-5 Retrieved Chunks
```
Chunk 1 (sim=0.91): "Employees can work remotely up to 3 days per week..."
Chunk 2 (sim=0.87): "Manager approval is required for any exceptions..."
Chunk 3 (sim=0.72): "Core working hours are 10 AM to 4 PM..."
Chunk 4 (sim=0.68): "Remote work tools include Slack, Zoom..."
Chunk 5 (sim=0.65): "Employees must document arrangements in HR system..."
```

### Step 4: Build Augmented Prompt
```python
system_prompt = """You are a helpful HR assistant for ACME Corp.
Answer based ONLY on the provided context.

Rules:
1. If not in context, say "I don't have information about that."
2. Do not make up information.
3. Cite sources using [chunk_id].
4. Keep answers concise."""

user_prompt = """
<context>
  <chunk id="1" source="remote_work.pdf">
    Employees can work remotely up to 3 days per week.
  </chunk>
  <chunk id="2" source="remote_work.pdf">
    Manager approval is required for any exceptions.
  </chunk>
  <chunk id="3" source="remote_work.pdf">
    Core working hours are 10 AM to 4 PM.
  </chunk>
</context>

<question>
Can I work from home 4 days a week?
</question>

Answer:"""
```

### Step 5: LLM Generation (OpenAI GPT-4o-mini)
```python
response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.1,
    max_tokens=300
)
# Latency: ~1.5s
# Cost: ~$0.0004
```

### Step 6: Final Answer
> "According to company policy, employees can work remotely up to 3 days per week [chunk_1]. To work 4 days from home, you would need manager approval for this exception [chunk_2]. Please consult with your manager to discuss this arrangement."

## ✅ What Makes This a Good RAG Response:
- **Grounded** in retrieved context (no hallucination)
- **Cites sources** [chunk_1], [chunk_2]
- **Nuanced** — doesn't blindly say yes/no
- **Actionable** — suggests next step (consult manager)
- **Concise** — no fluff

## 📊 Total Performance:
- **Total latency:** ~1.5s (dominated by LLM)
- **Cost per query:** ~$0.0004
- **For 100 queries/day:** ~$1.20/month

---

# 13. Interview Q&A Cheat Sheet

## 🎤 Common Interview Questions

### Q1: What is RAG and why use it?
**A:** RAG (Retrieval-Augmented Generation) combines information retrieval with LLM generation. Instead of relying solely on the LLM's training data, we retrieve relevant chunks from an external knowledge base and feed them to the LLM as context. This solves three problems: (1) outdated knowledge, (2) private/proprietary data access, (3) hallucinations.

### Q2: RAG vs Fine-tuning — which is better?
**A:** It depends on the use case. RAG is better for:
- Frequently changing data
- Source attribution requirements
- Lower cost and setup time
- Privacy (data stays in your DB)

Fine-tuning is better for:
- Teaching the model new skills or styles
- Specialized domain language

Often, **the best systems combine both** — fine-tune for domain language + RAG for facts.

### Q3: How do you choose a chunk size?
**A:** Trade-offs:
- **Smaller chunks** (256 tokens): More precise retrieval, less context per chunk
- **Larger chunks** (1024 tokens): More context, but may include irrelevant info

Default: **500 tokens with 50-token overlap**. Measure retrieval quality and adjust.

### Q4: Why must the embedding model be the same for indexing and querying?
**A:** Each embedding model creates a unique vector space. Different models produce vectors in **different coordinate systems** — even if dimensions match, the meaning of each dimension differs. Comparing vectors from different models is like comparing distances on maps with different scales — meaningless.

### Q5: Cosine similarity vs dot product vs Euclidean — when to use which?
**A:**
- **Cosine similarity:** Default for text (ignores magnitude, focuses on direction = meaning)
- **Dot product:** Faster when vectors are normalized (e.g., OpenAI embeddings)
- **Euclidean:** Rarely used for text in high dimensions due to curse of dimensionality

### Q6: What is chunking overlap and why use it?
**A:** Overlap means consecutive chunks share some text (e.g., 50 tokens). This prevents context loss at chunk boundaries. Without overlap, a sentence split between chunks would lose meaning.

### Q7: What is HyDE?
**A:** **Hypothetical Document Embeddings** — Use an LLM to generate a hypothetical answer to the query, then embed and search using that answer instead of the query. Works because document chunks look more like answers than questions, so hypothetical answers match better in vector space.

### Q8: What is re-ranking and why use it?
**A:** A two-stage retrieval approach:
1. **Stage 1:** Use fast bi-encoder embeddings to get top 20-50 candidates
2. **Stage 2:** Use slower but more accurate cross-encoder to re-rank top 3-5

Bi-encoders embed query and chunk separately; cross-encoders process them together for better relevance scoring.

### Q9: How do you prevent hallucinations in RAG?
**A:** Multiple strategies:
1. **Strong grounding instruction:** "Answer ONLY from context"
2. **Low temperature** (0-0.3)
3. **Force citations** for each claim
4. **Explicit "I don't know" instruction**
5. **Confidence calibration**
6. **Faithfulness evaluation** with RAGAS

### Q10: How would you scale RAG to 100M chunks?
**A:**
- **Vector DB:** Use distributed solutions (Milvus, Pinecone)
- **Indexing:** Use ANN algorithms (HNSW, IVF-PQ)
- **Sharding:** Partition by metadata (e.g., department, date)
- **Caching:** Cache common queries
- **Async indexing:** Use queue-based pipelines for new docs
- **Hybrid storage:** Hot data in memory, cold data on disk

### Q11: What is "Lost in the Middle" and how to mitigate?
**A:** A 2023 research finding that LLMs pay most attention to the beginning and end of long contexts, ignoring the middle. Mitigations:
1. Reduce K (fewer chunks = no middle)
2. Re-rank to put best chunks first/last
3. Use models with better long-context handling (Claude, Gemini)

### Q12: What metrics do you use to evaluate RAG?
**A:** Two categories:
1. **Retrieval metrics:** Precision@K, Recall@K, MRR, NDCG
2. **Generation metrics:** Faithfulness, answer relevance, context precision/recall

Frameworks: **RAGAS**, TruLens, DeepEval.

### Q13: How do you handle multimodal RAG (images, tables)?
**A:**
- **Images:** Use multimodal embeddings (CLIP, OpenAI vision)
- **Tables:** Convert to markdown, use table-aware models
- **PDFs with mixed content:** Use tools like `unstructured` or LlamaParse

### Q14: When would you NOT use RAG?
**A:**
- Queries don't need external knowledge (general chitchat)
- Latency-critical applications (RAG adds 1-2s)
- Very small, static knowledge bases (just put in system prompt)
- Highly creative tasks (RAG constrains the LLM)

### Q15: How do you handle data updates in RAG?
**A:**
1. **Incremental indexing:** Add new chunks without re-embedding everything
2. **Delete + re-add:** For updated documents
3. **Versioning:** Tag chunks with version numbers
4. **Scheduled re-indexing:** Nightly or weekly for high-volume systems
5. **Webhook-based:** Real-time updates from CMS/databases

---

## 🧠 Key Concepts to Remember

| Concept | One-Line Summary |
|---------|-----------------|
| **RAG** | Retrieve relevant docs, augment prompt, LLM generates grounded answer |
| **Chunking** | Split docs into smaller pieces (200-1000 tokens) with overlap |
| **Embedding** | Convert text → vector capturing semantic meaning |
| **Same Model Rule** | Must use same embedding model for indexing and querying |
| **Cosine Similarity** | Measures angle between vectors (default for text) |
| **Top-K Retrieval** | Get K most similar chunks to query |
| **ANN** | Approximate Nearest Neighbor (HNSW, IVF) for fast search at scale |
| **Hybrid Search** | Combine vector + keyword (BM25) search |
| **Re-ranking** | Two-stage: fast retrieval + slow accurate re-rank |
| **HyDE** | Embed a hypothetical answer instead of the query |
| **Prompt Engineering** | System rules + context + query, structured carefully |
| **Temperature** | 0-0.3 for RAG (factual, not creative) |
| **Lost in Middle** | LLMs ignore middle of long contexts |
| **RAGAS** | Framework for measuring RAG quality |

---

## 🚀 Final Tips for Interviews

1. **Always discuss trade-offs** — there's no "best" choice, only "best for the context"
2. **Use concrete numbers** — "I'd start with 500-token chunks and 5 top-K"
3. **Mention failure modes** — shows production thinking
4. **Reference evaluation** — "How would you measure improvement?"
5. **Draw the architecture** — visual explanations impress interviewers
6. **Talk about cost** — engineering means budget awareness
7. **Show iterative thinking** — "Start simple, measure, improve"

---

## 📚 Recommended Further Reading

- **Papers:**
  - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
  - "Lost in the Middle: How Language Models Use Long Contexts" (Liu et al., 2023)
  - "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE, 2022)

- **Tools to Master:**
  - LangChain, LlamaIndex (orchestration)
  - Pinecone, Weaviate, Qdrant, ChromaDB (vector DBs)
  - RAGAS (evaluation)
  - sentence-transformers (open-source embeddings)

- **Benchmarks:**
  - MTEB Leaderboard (embedding models)
  - BEIR benchmark (retrieval)

---

# 🎯 Summary: The RAG Mental Model

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   OFFLINE (One-time):                                   │
│   Documents → [Chunk] → [Embed] → [Store in Vector DB]  │
│                                                         │
│   ONLINE (Per Query):                                   │
│   Query → [Embed] → [Search] → [Re-rank] → [Top-K]      │
│            ↓                                            │
│   [System Prompt + Context + Query] → [LLM] → Answer    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Remember:** RAG is just enhanced prompting with dynamic context retrieval. Master the fundamentals, understand trade-offs, and you'll ace any RAG interview. 🚀

---

*Document created for senior AI engineer interview preparation. Good luck! 🎓*
