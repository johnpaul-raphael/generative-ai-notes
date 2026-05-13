# Encoder, Decoder & Encoder-Decoder Transformers: The Complete Week 8 Guide
### The Three Architecture Families — From Theory to Practice

---

> **Scope of this document:**
> Week 7 covered Transformer internals deeply — self-attention, multi-head attention, FFN,
> positional encoding, tokenization, training, BERT, GPT/LLaMA fine-tuning, and RLHF.
>
> This document builds on that foundation. It does NOT repeat it. Every chapter here covers
> something Week 7 left out:
> - **Cross-attention** (mentioned once as an interview answer in Week 7 — explained fully here)
> - **T5 and BART** (the encoder-decoder family — completely absent from Week 7)
> - **Span corruption and denoising pre-training** (T5/BART's training objectives)
> - **Seq2seq fine-tuning pattern** (the third fine-tuning paradigm Week 7 skipped)
> - **Causal LM fine-tuning specifics** (from the GPT-2 recipe notebook)
> - **Architecture selection guide** (practical decision tree for real jobs)
>
> **Pointer to Week 7:** Whenever you see `→ Week 7 Ch.X`, stop here and re-read that
> chapter if the concept feels unfamiliar. Don't read both in parallel.

---

## Table of Contents

1. [Chapter 1: The Three Architecture Families — Quick Map](#chapter-1-three-families)
2. [Chapter 2: Cross-Attention — The Missing Piece](#chapter-2-cross-attention)
3. [Chapter 3: Encoder-Decoder Models — T5 Deep Dive](#chapter-3-t5)
4. [Chapter 4: BART — Denoising the Encoder-Decoder](#chapter-4-bart)
5. [Chapter 5: Causal LM Fine-Tuning — GPT-2 Recipe Generator Explained](#chapter-5-causal-lm-finetuning)
6. [Chapter 6: Fine-Tuning Patterns per Family](#chapter-6-finetuning-patterns)
7. [Chapter 7: Architecture Selection Guide](#chapter-7-selection-guide)
8. [Chapter 8: Lab Notebooks — Theory to Practice](#chapter-8-lab-notebooks)
9. [Chapter 9: Model Taxonomy — Every Major Model Placed](#chapter-9-model-taxonomy)
10. [Chapter 10: Interview Questions — Week 8 Focus](#chapter-10-interviews)

---

## Chapter 1: The Three Architecture Families — Quick Map

> This chapter is a reference map, not a deep dive. The internals of self-attention,
> BERT, and GPT/LLaMA are already in Week 7 (Chapters 7–13B). Read those first.

### Why Three Families Exist

The original 2017 Transformer paper ("Attention Is All You Need") introduced an
**encoder-decoder** model for machine translation. The encoder reads the source sentence;
the decoder generates the target sentence. Both use self-attention, but the decoder adds
an extra layer — **cross-attention** — to look at the encoder's output.

Researchers then discovered: what if you only use the encoder? Or only the decoder?

```
Original (2017):  Encoder + Decoder  →  "Attention Is All You Need" (translation)
         2018:    Encoder only       →  BERT  (understanding tasks)
         2018:    Decoder only       →  GPT-1 (generation tasks)
         2020:    Encoder + Decoder  →  T5 renames and unifies all tasks
```

Each family emerged from a different question:
- **Encoder-only:** "Can we pre-train a model that deeply understands language?"
- **Decoder-only:** "Can we pre-train a model that generates fluent language at any scale?"
- **Encoder-Decoder:** "Can we unify ALL NLP tasks into one seq2seq framework?"

### The Three Families Side-by-Side

```
┌──────────────────┬──────────────────────┬──────────────────────┬──────────────────────┐
│                  │  ENCODER-ONLY        │  DECODER-ONLY        │  ENCODER-DECODER     │
├──────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ Attention type   │ Bidirectional        │ Causal (unidirect.)  │ Both + cross-attn    │
│                  │ (all tokens see all) │ (only past tokens)   │                      │
├──────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ Pre-training     │ MLM (mask & predict) │ CLM (predict next)   │ Span corruption /    │
│ objective        │ + NSP (BERT)         │                      │ denoising (T5/BART)  │
├──────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ Output           │ Rich contextual      │ Next token           │ Full output sequence │
│                  │ embedding per token  │ probabilities        │ (variable length)    │
├──────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ Can it generate? │ No (no causal LM)   │ Yes (autoregressive) │ Yes (via decoder)    │
├──────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ Best tasks       │ Classification, NER, │ Chat, completion,    │ Translation, summar- │
│                  │ extractive QA,       │ generation, code,    │ ization, abstractive │
│                  │ semantic search      │ reasoning            │ QA, data-to-text     │
├──────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ Key models       │ BERT, RoBERTa,       │ GPT-2, GPT-3/4,      │ T5, mT5, BART,       │
│                  │ DistilBERT, ALBERT   │ LLaMA, Mistral       │ mBART, PEGASUS       │
└──────────────────┴──────────────────────┴──────────────────────┴──────────────────────┘
```

### Java Analogy: Three Different Interface Implementations

Think of the three families as three implementations of a common `NLPModel` interface,
each optimised for a different use case:

```java
// The common interface
interface NLPModel {
    float[] encode(String text);       // produce representations
    String  generate(String prompt);   // produce output text
}

// Encoder-only: excels at encode(), cannot generate
class BERTModel implements NLPModel {
    float[] encode(String text)    { /* bidirectional attention — sees full context */ }
    String  generate(String prompt){ throw new UnsupportedOperationException(); }
    // → Specialised for tasks that classify or extract from existing text
}

// Decoder-only: excels at generate(), encode() is "accidentally good"
class GPTModel implements NLPModel {
    float[] encode(String text)    { /* causal attention — one-directional */ }
    String  generate(String prompt){ /* autoregressive — one token at a time */ }
    // → Specialised for tasks that produce new text
}

// Encoder-Decoder: proper implementation of both
class T5Model implements NLPModel {
    float[] encode(String text)    { /* bidirectional encoder */ }
    String  generate(String prompt){ /* decoder reads encoder output via cross-attention */ }
    // → Best when input structure ≠ output structure (translation, summarization)
}
```

---

## Chapter 2: Cross-Attention — The Missing Piece

> **Week 7 Ch.10 interview answer:** "Cross-attention: Q comes from the decoder,
> K and V come from the encoder." That's what Week 7 said. This chapter explains WHY,
> HOW it works mechanically, and what happens without it.

### Why Encoder-Decoder Models Need a Bridge

In a pure decoder (GPT), the model generates token-by-token using only its own
history. When you want to TRANSLATE a sentence, that's not enough — the decoder
must have access to the encoded source sentence at every generation step.

Cross-attention is that access mechanism.

```
Without cross-attention (pure decoder trying to translate):

Encoder: "Je suis étudiant"  →  [some encodings]
                                         ↓  No way to pass this in
Decoder: generates "I am a ___"  ← guessing with no source access

With cross-attention:

Encoder: "Je suis étudiant"  →  [e₁, e₂, e₃]  ← 3 context vectors, one per token
                                         ↓  STORED, passed to every decoder layer
Decoder at each step: "what in the encoder is relevant to what I'm generating right now?"
         → Q (from decoder) × K (from encoder) → scores
         → softmax(scores) × V (from encoder) → context vector
         → use context vector + decoder's own state to predict next token
```

### The Mechanics: Three-Matrix Operation with Mixed Sources

Self-attention (covered in Week 7 Ch.7): Q, K, V all come from the **same** sequence.
Cross-attention: Q comes from the **decoder**, K and V come from the **encoder**.

```
Self-Attention (encoder block):
  Input x: (batch, src_len, d_model)
  Q = x @ W_Q    shape: (batch, src_len, d_k)
  K = x @ W_K    shape: (batch, src_len, d_k)   ← same x for Q and K
  V = x @ W_V    shape: (batch, src_len, d_v)   ← same x for V

Cross-Attention (decoder block, second sub-layer):
  Decoder state h: (batch, tgt_len, d_model)   ← what the decoder has built so far
  Encoder output e: (batch, src_len, d_model)  ← frozen output from encoder
  Q = h @ W_Q    shape: (batch, tgt_len, d_k)  ← from DECODER
  K = e @ W_K    shape: (batch, src_len, d_k)  ← from ENCODER
  V = e @ W_V    shape: (batch, src_len, d_v)  ← from ENCODER

Attention scores: Q @ K.T → (batch, tgt_len, src_len)
                  ↑ this is a (target length × source length) matrix
                  Each decoder position asks: "which encoder position is most relevant?"

Softmax over src_len dimension:
  (batch, tgt_len, src_len)  →  attention weights summing to 1 over src positions

Output: weights @ V → (batch, tgt_len, d_v)
  ↑ each decoder position gets a weighted sum of encoder values
```

### ASCII Diagram: Cross-Attention in a Translation Step

```
Source: "Je  suis étudiant"     Target so far: "I  am"
         e₁   e₂    e₃                           h₁  h₂

Cross-attention when predicting 3rd target token ("a"):

Decoder query (for position 3):   q₃ = h₂ @ W_Q    ← "what am I looking for?"

Encoder keys:
  k₁ = e₁ @ W_K   (for "Je")
  k₂ = e₂ @ W_K   (for "suis")
  k₃ = e₃ @ W_K   (for "étudiant")

Scores = q₃ · k₁, q₃ · k₂, q₃ · k₃
       = [0.1,     0.1,     0.8]    ← high score on "étudiant" makes sense: "a" ≈ article before noun

Weights = softmax([0.1, 0.1, 0.8]) = [0.07, 0.07, 0.86]

Context = 0.07 × v₁ + 0.07 × v₂ + 0.86 × v₃
       → heavily weighted toward the "étudiant" encoding
       → model uses this to predict "student"

Next predicted token: "student"   ✓
```

### Where Cross-Attention Sits in a Decoder Block

Each decoder block has **three** sub-layers (vs encoder's two):

```
Decoder Block N:
  ┌────────────────────────────────────────────────┐
  │  Sub-layer 1: Masked Self-Attention             │
  │    → decoder attends to its own previous tokens │
  │    → causal mask: cannot see future positions   │
  │                                                 │
  │  Sub-layer 2: CROSS-ATTENTION  ← the new one    │
  │    → Q: from decoder's self-attention output    │
  │    → K, V: from encoder's final hidden states   │
  │    → "bridge" from source to target             │
  │                                                 │
  │  Sub-layer 3: Feed-Forward Network              │
  │    → same as encoder FFN                        │
  └────────────────────────────────────────────────┘

Each sub-layer has its own LayerNorm and residual connection.

Encoder block has: Self-Attention + FFN  (2 sub-layers)
Decoder block has: Masked Self-Attn + Cross-Attn + FFN  (3 sub-layers)
```

### Key Properties of Cross-Attention

**1. Encoder output is computed once, reused at every decoder step:**

```python
# During inference (generation loop):
encoder_output = model.encoder(input_ids, attention_mask)  # run once

for step in range(max_new_tokens):
    # Decoder runs at every step, but cross-attention
    # always reads the SAME encoder_output
    decoder_output = model.decoder(
        decoder_input_ids,
        encoder_hidden_states=encoder_output,   # ← reused every step
        encoder_attention_mask=attention_mask
    )
    next_token = decoder_output.logits[:, -1, :].argmax()
    decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
```

**2. The attention weight matrix reveals translation alignment:**

```
Cross-attention weight matrix for "Je suis étudiant" → "I am a student":

             Je    suis  étudiant
        I   [0.90  0.05   0.05  ]   ← "I" attends mostly to "Je"
        am  [0.05  0.90   0.05  ]   ← "am" attends mostly to "suis"
        a   [0.05  0.10   0.85  ]   ← "a" attends mostly to "étudiant"
       student[0.05 0.05  0.90  ]   ← "student" attends mostly to "étudiant"

This is soft alignment — the model learned translation alignment
from training data alone, without explicit alignment labels.
```

**3. Cross-attention is NOT present in decoder-only models (GPT, LLaMA):**

```
GPT/LLaMA decoder block: Causal Self-Attention + FFN  (no cross-attention)
  → No encoder, so nothing to cross-attend to
  → The "context" comes entirely from the input prompt,
    which is prepended to the generation (prefix lm)

T5/BART decoder block:   Masked Self-Attn + Cross-Attn + FFN
  → Cross-attention reads the encoded source sequence
```

### Interview Questions — Cross-Attention

> **Beginner:** What is the difference between self-attention and cross-attention?
> → Self-attention: Q, K, V all come from the same sequence (model attends to itself).
> Cross-attention: Q comes from the decoder's current state, K and V come from the
> encoder's output. Cross-attention is how the decoder "reads" the source sentence
> while generating the target.

> **Intermediate:** Why is the cross-attention weight matrix useful for interpretability?
> → The (tgt_len × src_len) weight matrix shows which source tokens each target token
> attended to. For translation this reveals learned alignment. For summarization it shows
> which source sentences the summary is based on — without any explicit annotation.

> **Advanced:** In an encoder-decoder model, why is the encoder output passed to EVERY
> decoder layer rather than just the first?
> → Each decoder layer has its own cross-attention weights (W_Q, W_K, W_V), allowing
> different layers to attend to different aspects of the source. Lower layers may attend
> to syntactic structure (word order, grammar), while upper layers attend to semantic
> content (meaning, entities). Sharing the same encoder output across all layers is
> computationally cheap since it's computed once and cached — passing it to every layer
> is free in memory terms.

---

## Chapter 3: Encoder-Decoder Models — T5 Deep Dive

> T5 (Text-to-Text Transfer Transformer) is NOT covered in Week 7. This chapter is new.

### What is T5?

T5 (Raffel et al., Google, 2019) stands for "Text-to-Text Transfer Transformer." Its central
idea is radical: **every NLP task is reformulated as a text-to-text problem.**

```
Task                 Input (text)                          Output (text)
─────────────────────────────────────────────────────────────────────────────
Translation          "translate English to German: Hello"  "Hallo"
Summarization        "summarize: [article...]"             "[summary]"
Classification       "sentiment: This movie is great."     "positive"
Question Answering   "question: Who? context: [passage]"   "Marie Curie"
Regression (STS-B)   "stsb sentence1: ... sentence2: ..."  "3.8"  ← even a number!
Entailment           "mnli hypothesis: ... premise: ..."   "entailment"
```

Every single task outputs a text string. The model doesn't need different output heads —
**the decoder always predicts tokens.** This unification is T5's most important innovation.

### T5 Architecture

T5 uses the standard encoder-decoder architecture with a few specific choices:

```
┌──────────────────────────────────────────────────────────────────────┐
│  ENCODER (bidirectional self-attention)                               │
│                                                                      │
│  Input: "summarize: Scientists discover new species in Amazon..."    │
│    ↓ Tokenize (SentencePiece, vocab 32,000)                          │
│    ↓ Token Embedding                                                 │
│    ↓ Block 1: Self-Attention + FFN + LayerNorm + Residual            │
│    ↓ Block 2: ...                                                     │
│    ↓ Block N: ...                                                     │
│    → Encoder hidden states: (batch, src_len, d_model)                │
└──────────────────────────────────────────────────────────────────────┘
                         ↓ encoder_output (passed to every decoder block)
┌──────────────────────────────────────────────────────────────────────┐
│  DECODER (masked self-attention + cross-attention)                   │
│                                                                      │
│  Input: starts with <pad> token (decoder start token)               │
│    ↓ Block 1: Masked Self-Attn + Cross-Attn + FFN                   │
│    ↓ Block 2: ...                                                     │
│    ↓ Block N: ...                                                     │
│    ↓ Linear (d_model → vocab_size)                                   │
│    → Next token logits                                               │
│                                                                      │
│  Generates: "Researchers identify previously unknown species in       │
│              the Brazilian Amazon rainforest."                        │
└──────────────────────────────────────────────────────────────────────┘
```

### T5 Architecture Choices (What Makes T5 Different)

| Component | T5 Choice | Why |
|---|---|---|
| Positional encoding | Relative position bias | More robust to sequences longer than training |
| Normalization | Pre-LayerNorm (before attention) | More stable training |
| Activation | ReLU in FFN | Simpler than GELU (T5 was designed before SwiGLU) |
| Tokenizer | SentencePiece (unigram LM) | Language-agnostic, good for multilingual |
| Decoder start | `<pad>` token | Unlike BERT's `[CLS]`, unlike GPT's BOS |
| Task prefix | Natural language prefix | "translate English to French:" |
| Weight tying | Encoder embedding = Decoder embedding = LM head | Reduces parameters |

**T5 Model Sizes:**

```
T5-Small  :  60M parameters   (6 encoder + 6 decoder layers, d=512)
T5-Base   : 220M parameters   (12+12 layers, d=768)
T5-Large  : 770M parameters   (24+24 layers, d=1024)
T5-XL     :  3B parameters    (24+24 layers, d=2048)
T5-XXL    : 11B parameters    (24+24 layers, d=4096)
Flan-T5   : fine-tuned T5 on 1800+ tasks with natural language instructions
```

### T5's Pre-training Objective: Span Corruption (Denoising)

This is the most important concept unique to T5. Instead of masking individual tokens
(like BERT's MLM), T5 masks **consecutive spans** and trains the model to reconstruct them.

```
Original text:
  "The quick brown fox jumps over the lazy dog near the river bank."

Step 1: Select random spans to corrupt (15% of tokens, avg span length = 3):
  Spans selected: ["quick brown"] and ["near the river"]

Step 2: Replace each span with a unique sentinel token:
  "The <extra_id_0> fox jumps over the lazy dog <extra_id_1> bank."
  (sentinel tokens are special vocab entries: <extra_id_0> to <extra_id_99>)

Step 3: Create the TARGET — just the corrupted spans + sentinels:
  "<extra_id_0> quick brown <extra_id_1> near the river <extra_id_2>"
  (ends with a final sentinel to signal end of output)

T5 INPUT  (to encoder): "The <extra_id_0> fox jumps over the lazy dog <extra_id_1> bank."
T5 TARGET (from decoder): "<extra_id_0> quick brown <extra_id_1> near the river <extra_id_2>"
```

**Why span corruption is better than MLM for seq2seq:**

```
BERT MLM problem for seq2seq:
  → Masks individual tokens, predicts them IN-PLACE
  → Input and output are the same length
  → Cannot learn to generate VARIABLE-LENGTH outputs
  → Forces bidirectional encoder, no generative decoder training

T5 Span Corruption advantages:
  → Output is a SHORT sequence (only the missing spans)
  → Trains both encoder (to understand corrupted input) AND
    decoder (to generate the missing text autoregressively)
  → Models longer-range dependencies within spans
  → More compute-efficient (decoder sees fewer tokens than full target)
```

### T5 Relative Position Bias

Week 7 covered sinusoidal (original Transformer) and RoPE (LLaMA) positional encoding.
T5 uses a third approach: **learnable relative position biases**.

```
Absolute position (BERT style):
  Each token gets embedding for position 0, 1, 2, ...
  Problem: "token at position 5 in training" ≠ "token at position 5 in a longer test sequence"

Relative position (T5 style):
  Each pair of tokens (i, j) gets a scalar bias based on their DISTANCE (j - i)
  These biases are added to attention scores before softmax:
  score(i,j) = (q_i · k_j) / √d_k + bias(j - i)

Benefits:
  ✓ Model learns "token 2 positions away" not "token at absolute position 7"
  ✓ Better generalisation to longer sequences
  ✓ Shared across all attention heads (few parameters)
  ✓ The biases are bucketed: exact for small distances, bucketed for large distances
```

### Fine-tuning T5

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset

# ── Load T5 ────────────────────────────────────────────────────────────────
model_name = "t5-base"
tokenizer  = T5Tokenizer.from_pretrained(model_name)
model      = T5ForConditionalGeneration.from_pretrained(model_name)

# ── Format data as text-to-text pairs ──────────────────────────────────────
def preprocess(examples):
    # T5 expects task prefix in the INPUT
    inputs = ["summarize: " + doc for doc in examples["article"]]
    targets = examples["summary"]

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    # Tokenize targets — use as_target_tokenizer context (older HF) or just tokenize
    labels = tokenizer(
        targets,
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# ── Key difference from BERT fine-tuning ───────────────────────────────────
# BERT: model(input_ids, attention_mask, labels) → ClassificationOutput
# T5:   model(input_ids, attention_mask, labels) → Seq2SeqLMOutput
#   → labels here are the TARGET SEQUENCE (decoder output), not a class integer
#   → T5 computes cross-entropy over each target token position
#   → Padding (-100) is automatically ignored in loss computation

# ── T5 automatically handles teacher forcing during training ───────────────
# Teacher forcing: at each decoder step, feed the GROUND TRUTH previous token
# (not the model's own prediction) as input. This is standard for seq2seq training.
# The Trainer handles this automatically when you pass "labels".

training_args = TrainingArguments(
    output_dir="./t5-summarizer",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=3e-4,           # T5 uses higher LR than BERT (3e-4 to 1e-3)
    warmup_steps=500,
    weight_decay=0.01,
    predict_with_generate=True,   # use model.generate() for eval, not greedy logits
    fp16=True
)

# ── Generation at inference ────────────────────────────────────────────────
def summarize(text, max_new_tokens=128):
    input_ids = tokenizer(
        "summarize: " + text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).input_ids

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        num_beams=4,          # beam search (week 7 ch.12) — better than greedy
        early_stopping=True,  # stop when all beams produce EOS
        no_repeat_ngram_size=3  # prevent repetitive output
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**BERT fine-tuning vs T5 fine-tuning side-by-side:**

```
BERT (encoder-only) fine-tuning:
  1. Load BertForSequenceClassification (adds Linear(768 → num_classes) head)
  2. Pass input_ids + attention_mask + labels (integer class IDs)
  3. Model forward pass → [CLS] vector → linear → logits → CrossEntropyLoss(logits, labels)
  4. labels dtype: torch.long (integers)

T5 (encoder-decoder) fine-tuning:
  1. Load T5ForConditionalGeneration (uses decoder as the output head)
  2. Pass input_ids + attention_mask + labels (token ID sequences)
  3. Model forward pass → encoder hidden states → decoder → logits → CrossEntropyLoss per token
  4. labels dtype: list of token IDs (sequences, not integers)
  5. Replace padding in labels with -100 so loss ignores padding positions
```

### T5 Variants

| Model | Description | When to use |
|---|---|---|
| **T5-base** | Standard 220M model | Most fine-tuning tasks |
| **Flan-T5** | T5 fine-tuned on 1800+ instruction-following tasks | Zero-shot and few-shot tasks |
| **mT5** | Multilingual T5, pre-trained on 101 languages | Cross-lingual tasks |
| **mT0** | mT5 fine-tuned on multilingual instruction following | Multilingual instruction following |
| **UL2** | T5 with improved pre-training (mixture of denoisers) | Better generalisation |
| **CodeT5** | T5 pre-trained on code (GitHub) | Code generation, code summarization |
| **Flan-UL2** | UL2 with instruction fine-tuning | State-of-the-art instruction following (open) |

---

## Chapter 4: BART — Denoising the Encoder-Decoder

> BART (Lewis et al., Facebook AI, 2019) is NOT covered in Week 7. This chapter is new.

### What is BART?

BART (Bidirectional and Auto-Regressive Transformers) is another encoder-decoder model,
similar in structure to T5 but with a very different pre-training approach.

**Key insight:** BART is pre-trained as a **denoising autoencoder** — corrupt text in many
different ways, then train the model to reconstruct the original uncorrupted text.

```
BART pre-training flow:

Original: "The scientists discovered a new species in the Amazon rainforest."

Apply noise (one or more corruption functions):
  → "The scientists <MASK> in <MASK> discovered new a rainforest Amazon species."
     (tokens deleted, sentences shuffled, spans masked)

Encoder reads NOISY input.
Decoder reconstructs the ORIGINAL clean text.

This is exactly the translation task structure:
  "noisy version of text" → "clean version of text"
```

### BART's Five Noise Functions

BART's pre-training is unique because it applies FIVE different types of noise,
not just masking. Each type teaches the model a different reconstruction skill.

**Noise 1: Token Masking (same as BERT MLM)**

```
Original : "The scientists discovered a new species"
Masked   : "The [MASK] discovered a [MASK] species"
Teaches  : in-filling individual tokens (same as BERT)
```

**Noise 2: Token Deletion**

```
Original : "The scientists discovered a new species"
Deleted  : "The discovered new species"    ← "scientists" and "a" deleted entirely
Teaches  : model must DETECT what's missing (harder than masking — no [MASK] hint)
```

**Noise 3: Text Infilling (Span Masking — similar to T5)**

```
Original : "The scientists discovered a new species in Amazon"
Masked   : "The scientists [MASK] in Amazon"
             ↑ "discovered a new species" → single [MASK] token
Teaches  : predict a variable-length span from a single mask token
           The decoder must figure out HOW MANY tokens to generate
```

**Noise 4: Sentence Permutation**

```
Original (2 sentences): "Scientists found a new species. It was in the Amazon."
Shuffled               : "It was in the Amazon. Scientists found a new species."
Teaches  : discourse-level ordering — important for multi-sentence generation
```

**Noise 5: Document Rotation**

```
Original : "Scientists found a new species. It was in the Amazon. Research continues."
Rotated  : "It was in the Amazon. Research continues. Scientists found a new species."
            ↑ start token is now the 2nd sentence
Teaches  : identify the start of a document (useful for summarization)
```

**Why multiple noise functions?**

```
T5 span corruption → excellent at generation, general text-to-text
BART denoising     → excellent at generation WITH SPECIFIC STRUCTURAL SKILLS:
  - Sentence permutation → teaches discourse/coherence → good for summarization
  - Token deletion       → harder reconstruction → stronger representations
  - Document rotation    → teaches document structure → good for abstractive QA

Empirically: BART outperforms T5 on summarization (CNN/DailyMail, XSum)
             T5 outperforms BART on translation and classification
```

### BART Architecture vs T5 Architecture

```
Both: Encoder-Decoder with bidirectional encoder + autoregressive decoder

BART differences from T5:
  - Uses learned absolute position embeddings (like original Transformer)
    T5 uses relative position biases
  - Uses standard GELU activation
    T5 uses ReLU
  - Pre-training: denoising autoencoder (multiple noise functions)
    T5 pre-training: span corruption only
  - Final layer norm position: post-norm (original paper style)
    T5: pre-norm
  - Tokenizer: BPE (same as GPT-2, vocab=50,265)
    T5: SentencePiece (vocab=32,000)
  - Initialization: parameters initialized from GPT-2 (decoder) and BERT (encoder)
    T5: random initialization
```

### BART for Summarization

BART is the standard baseline for abstractive summarization. Here's the key intuition:

```
News article → ENCODER (reads full article bidirectionally)
                    ↓ Cross-attention
             DECODER (generates summary autoregressively)

The denoising pre-training "document rotation" noise specifically
trained BART to understand full document structure — which token
should come first in a coherent output. This is exactly summarization.
```

```python
from transformers import BartForConditionalGeneration, BartTokenizer

model_name = "facebook/bart-large-cnn"  # fine-tuned on CNN/DailyMail
tokenizer  = BartTokenizer.from_pretrained(model_name)
model      = BartForConditionalGeneration.from_pretrained(model_name)

article = """
Scientists at the University of São Paulo have discovered a new species of
orchid in the Brazilian Amazon. The species, named Cattleya amazonica,
features striking purple flowers with yellow centers. Researchers believe
the species has existed for thousands of years but was hidden deep in the
forest. The discovery adds to the growing catalog of Amazon biodiversity.
"""

inputs = tokenizer(article, return_tensors="pt", max_length=1024, truncation=True)

summary_ids = model.generate(
    inputs["input_ids"],
    max_new_tokens=60,
    min_length=20,
    length_penalty=2.0,   # penalise short summaries (>1 = prefer longer output)
    num_beams=4,
    early_stopping=True
)

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary)
# → "Scientists have discovered a new orchid species in the Brazilian Amazon.
#    Cattleya amazonica features purple flowers with yellow centers and was
#    hidden deep in the forest."
```

### BART Variants

| Model | Specialisation | HuggingFace path |
|---|---|---|
| **bart-base** | General pre-trained | `facebook/bart-base` |
| **bart-large-cnn** | Summarization (CNN/DailyMail fine-tuned) | `facebook/bart-large-cnn` |
| **bart-large-xsum** | Summarization (XSum — extreme, one sentence) | `facebook/bart-large-xsum` |
| **mBART** | Multilingual BART (50 languages) | `facebook/mbart-large-50` |
| **mBART-50-many-to-many** | Multilingual translation, 50→50 languages | `facebook/mbart-large-50-many-to-many-mmt` |
| **PEGASUS** | Summarization specialist (gap sentence pre-training) | `google/pegasus-large` |

### Interview Questions — T5 and BART

> **Beginner:** What does "text-to-text" mean in T5?
> → Every NLP task is reframed as: input = text string, output = text string.
> Classification becomes "predict the class label as text" (e.g., "positive").
> Translation is "translate English to French: Hello" → "Bonjour".
> The model never needs different output heads — the decoder always generates tokens.

> **Intermediate:** What is span corruption in T5 and how is it different from BERT's MLM?
> → BERT MLM: randomly replaces 15% of individual tokens with [MASK], predicts them
> in-place (same-length output). T5 span corruption: replaces consecutive spans with
> sentinel tokens (e.g., <extra_id_0>), and the TARGET is only the missing spans — a
> shorter sequence. This trains the full encoder-decoder: encoder sees corrupted input,
> decoder generates the missing spans autoregressively.

> **Advanced:** Why does BART outperform T5 on summarization despite T5 being trained
> on more data?
> → BART's sentence permutation and document rotation noise functions specifically train
> the model to understand document-level structure — ordering, coherence, and what
> constitutes a good beginning of a sequence. These are exactly the skills needed for
> abstractive summarization. T5's span corruption is more general-purpose. Additionally,
> BART's GPT-2-initialized decoder was already exposed to a wide range of fluent English
> generation patterns before pre-training even began.

---

## Chapter 5: Causal LM Fine-Tuning — GPT-2 Recipe Generator Explained

> This chapter explains the concepts behind `recipe_generator_gpt2_finetune.ipynb`.
> Week 7 covered BERT fine-tuning in detail (Ch.13B). This covers the DECODER fine-tuning
> counterpart — what's different, what's the same, and when to use it.

### The Core Difference: CLM vs Classification

```
BERT fine-tuning (encoder):
  Goal: teach BERT to classify text it ALREADY READS bidirectionally
  Loss: CrossEntropyLoss(predicted_class, true_class) — scalar per sample
  Output head: Linear(768 → num_classes) bolted onto [CLS] token
  Labels: integer class IDs [0, 1, 2, 4, 3, ...]

GPT-2 fine-tuning (decoder):
  Goal: teach GPT-2 to generate text IN A SPECIFIC DOMAIN (recipes here)
  Loss: CrossEntropyLoss(predicted_token, actual_next_token) — per position, per batch
  Output head: lm_head (d_model → vocab_size) — already part of GPT-2
  Labels: the input_ids themselves, shifted by 1
```

**The "labels = input_ids shifted" pattern (causal LM):**

```
Input sequence:  [Recipe: Butter Chicken <sep> Ingredients: chicken butter cream ...]

At each position, predict the NEXT token:

Position 0 ("Recipe"):        predict → ":"
Position 1 (":"):             predict → "Butter"
Position 2 ("Butter"):        predict → "Chicken"
Position 3 ("Chicken"):       predict → "<sep>"
...

So input_ids = labels, but labels are shifted left by 1:
  input_ids : ["Recipe",  ":",      "Butter", "Chicken", "<sep>", ...]
  labels    : [":",       "Butter", "Chicken","<sep>",   "Ingr.", ...]
                                                    ↑ predict next from current

HuggingFace's DataCollatorForLanguageModeling (mlm=False) does this shift FOR you.
You just pass input_ids — it creates the shifted labels automatically.
```

### DataCollatorForLanguageModeling Deep Dive

```python
from transformers import DataCollatorForLanguageModeling

# mlm=False → Causal LM (decoder-style, GPT-style)
# mlm=True  → Masked LM (BERT-style, randomly masks 15% of tokens)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# What this does internally:
# 1. Takes a batch of tokenized sequences (variable length after truncation)
# 2. Pads them to the same length within the batch
# 3. Sets labels = input_ids
# 4. Sets labels[padding_positions] = -100
#    (CrossEntropyLoss ignores positions where label == -100)
# 5. Returns {"input_ids": ..., "attention_mask": ..., "labels": ...}

# Why -100 for padding? CrossEntropyLoss in PyTorch ignores index -100 by default.
# This means the model is NOT penalized for predicting tokens at padding positions.
```

### Why Fine-tune a Decoder vs Just Prompting?

This is a practical question you'll face in every job:

```
┌──────────────────┬──────────────────────────┬────────────────────────────┐
│                  │  PROMPTING (no training) │  FINE-TUNING               │
├──────────────────┼──────────────────────────┼────────────────────────────┤
│ Data needed      │ 0–50 examples (few-shot) │ 100–10,000+ examples       │
│ Compute          │ Just inference           │ Training cost upfront       │
│ Latency          │ Longer prompts = slower  │ Shorter prompts = faster    │
│ Domain control   │ Limited by base model    │ Strong domain adaptation    │
│ Format control   │ Inconsistent             │ Consistent output format    │
│ Cost per call    │ Higher (long prompts)    │ Lower (short prompts)       │
│ Example (recipe) │ "Write a recipe for..."  │ Model always starts with    │
│                  │   (unreliable format)    │   "Recipe: ... Ingredients: │
│                  │                          │   ... Instructions: ..."    │
└──────────────────┴──────────────────────────┴────────────────────────────┘

Rule of thumb:
  - Prompting first: always try few-shot prompting before fine-tuning
  - Fine-tune when: format matters AND you have domain-specific data
  - The recipe notebook is a textbook domain adaptation use case:
    Indian recipes have a specific text format the base GPT-2 has never seen
```

### What GPT-2 Fine-tuning Teaches (and What It Doesn't)

```
Fine-tuning GPT-2 on recipes:
  DOES teach:
    ✓ Recipe text format (Recipe: ... Ingredients: ... Instructions: ...)
    ✓ Common Indian ingredient names and their collocations
    ✓ Instruction sequence style ("Heat oil. Add onions. Sauté until golden.")
    ✓ Domain vocabulary (masala, tadka, ghee, toor dal)

  DOES NOT teach (requires larger models or different training):
    ✗ Factual correctness of ingredient combinations
    ✗ Cooking science (temperature, timing relationships)
    ✗ Instruction following from a prompt ("make this spicier")
    ✗ Multi-turn dialogue about recipes

The model learns STYLE and DISTRIBUTION, not KNOWLEDGE or REASONING.
```

### GPT-2 Fine-tuning Hyperparameters in the Notebook

```python
# From recipe_generator_gpt2_finetune.ipynb:
training_args = TrainingArguments(
    output_dir="./recipe-gpt2",
    num_train_epochs=3,              # 3 epochs — enough for domain adaptation
                                     # too many → memorise recipes, can't generalise
    per_device_train_batch_size=4,   # small batch — GPT-2 is 768M+ activation memory
    per_device_eval_batch_size=4,
    warmup_steps=100,                # warmup over first 100 steps
    weight_decay=0.01,               # L2 regularisation (same as BERT fine-tuning)
    logging_steps=50,
    eval_strategy="epoch",
    fp16=True                        # half precision — saves memory, speeds up training
)

# Why no gradient clipping here?
# Trainer sets max_grad_norm=1.0 by default — it IS doing gradient clipping,
# just not shown explicitly (unlike the BERT notebook which called
# torch.nn.utils.clip_grad_norm_ manually)

# Why fp16=True?
# GPT-2's activations are large. fp16 halves memory usage.
# Loss scaling is handled automatically by Trainer.
```

### The Causal LM Training Loop (What Trainer Does Internally)

```
For each batch of recipe texts:

1. Tokenize + pad to max_length (512 for GPT-2)
2. DataCollator creates labels = input_ids, with padding → -100

3. Forward pass:
   input_ids → GPT-2 (12 decoder blocks) → logits (batch × 512 × 50257)

4. Loss:
   CrossEntropy(logits[:, :-1, :], labels[:, 1:])
   ↑ Compare prediction at each position (shifted left by 1)
   ↑ -100 positions in labels are ignored

5. Backward pass: gradients through all 12 layers

6. AdamW update (LR=5e-5 default in Trainer)

7. Scheduler step (linear decay with warmup)

Total training: 3 epochs × 4750 samples / batch_size=4 = 3562 steps
```

### Text Generation After Fine-tuning

```python
from transformers import pipeline

generator = pipeline("text-generation", model="./recipe-gpt2-final")

result = generator(
    "Recipe: milk-based pudding\nIngredients:",
    max_new_tokens=400,
    temperature=0.7,     # < 1.0 = more focused/predictable
                         # > 1.0 = more random/creative
    do_sample=True,      # sample from distribution (not greedy)
    pad_token_id=tokenizer.eos_token_id   # GPT-2 has no pad token; use EOS
)
```

**Why `pad_token_id=eos_token_id`?**

```
GPT-2 was not pre-trained with a dedicated pad token.
Its tokenizer.pad_token is None.
During generation, if the batch contains multiple sequences of different lengths,
HuggingFace needs a padding token to fill shorter sequences.
Using EOS as pad is a common workaround — the model already knows to stop
generating when it sees EOS, so the padding doesn't affect output quality.
```

---

## Chapter 6: Fine-Tuning Patterns per Family

> This chapter synthesizes the fine-tuning approaches across all three families.
> Week 7 Ch.14 covered SFT, RLHF, and LoRA for decoder models.
> This chapter adds encoder and encoder-decoder patterns.

### The Three Fine-tuning Paradigms

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PARADIGM 1: HEAD-BASED (Encoder-only — BERT family)                        │
│                                                                             │
│  Pre-trained BERT (frozen or fine-tuned body)                               │
│         ↓                                                                   │
│  Add task-specific head on [CLS] or token embeddings                        │
│         ↓                                                                   │
│  Fine-tune (full or LoRA)                                                   │
│                                                                             │
│  Heads:                                                                     │
│    Classification  → Linear([CLS] 768, num_classes)                        │
│    Token labeling  → Linear(each_token 768, num_labels) for NER, POS       │
│    Span extraction → Linear(each_token 768, 2) — start/end logits for QA   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  PARADIGM 2: LANGUAGE MODEL (Decoder-only — GPT/LLaMA family)               │
│                                                                             │
│  Pre-trained GPT / LLaMA (existing lm_head)                                 │
│         ↓                                                                   │
│  Continue training on domain-specific text (recipe fine-tuning)            │
│  OR instruction fine-tuning (supervised fine-tuning with prompts)           │
│  OR RLHF (add reward model, PPO update — see Week 7 Ch.14)                 │
│                                                                             │
│  No new head needed — lm_head (d_model → vocab_size) already exists        │
│  Labels = input_ids shifted (causal LM objective)                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  PARADIGM 3: SEQ2SEQ (Encoder-Decoder — T5/BART family)                    │
│                                                                             │
│  Pre-trained T5 / BART (encoder + decoder + cross-attention)                │
│         ↓                                                                   │
│  Fine-tune on (input_text, target_text) pairs                              │
│                                                                             │
│  Labels = tokenized target sequence (not a class integer)                  │
│  Loss computed over each target token (teacher forcing during training)     │
│  predict_with_generate=True for evaluation (use model.generate())          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Fine-Tuning Each Family: Practical Checklist

```
ENCODER-ONLY (BERT) fine-tuning checklist:
  □ Use BertForSequenceClassification (or TokenClassification, QuestionAnswering)
  □ Learning rate: 2e-5 to 5e-5 (higher destroys pre-trained weights)
  □ Epochs: 2–4 (easy overfitting on small datasets)
  □ Batch size: 16 or 32
  □ Warmup: 10% of steps
  □ Gradient clipping: max_norm=1.0
  □ Label type: integer (class IDs)
  □ Evaluation: accuracy, F1 (no generation needed)

DECODER-ONLY (GPT/LLaMA) fine-tuning checklist:
  □ Use AutoModelForCausalLM
  □ DataCollatorForLanguageModeling(mlm=False)
  □ Learning rate: 1e-5 to 5e-5 (even lower for large models: 1e-5)
  □ Use LoRA for large models (LLaMA) — full fine-tuning needs 16+ GB VRAM
  □ Set tokenizer.pad_token = tokenizer.eos_token (for GPT-2)
  □ Label type: token ID sequences (auto-shifted by DataCollator)
  □ Evaluation: perplexity, or qualitative generation samples

ENCODER-DECODER (T5/BART) fine-tuning checklist:
  □ Use T5ForConditionalGeneration or BartForConditionalGeneration
  □ Task prefix in input text ("summarize: ", "translate French to English: ")
  □ Tokenize inputs and targets SEPARATELY
  □ Set padding tokens in labels to -100 (ignored by loss)
  □ Use predict_with_generate=True in TrainingArguments for eval
  □ Evaluation: ROUGE (summarization), BLEU (translation), exact match (QA)
  □ Label type: token ID sequences (full target text)
  □ Beam search at inference (num_beams=4 is standard)
```

### LoRA on Each Architecture

LoRA (Low-Rank Adaptation) works on all three families. The target modules differ:

```python
from peft import LoraConfig, get_peft_model, TaskType

# BERT (encoder classification):
lora_config_bert = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    target_modules=["query", "value"],   # attention projection matrices in BERT
    lora_dropout=0.1
)

# GPT-2 / LLaMA (causal LM):
lora_config_gpt = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"],  # GPT-2 uses c_attn for Q/K/V combined
    # For LLaMA: ["q_proj", "v_proj"] or ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_dropout=0.1
)

# T5 (seq2seq):
lora_config_t5 = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],   # T5 attention projections
    lora_dropout=0.1
)

# Apply to any model:
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output example for LLaMA 7B with LoRA r=16:
# trainable params: 4,194,304 || all params: 6,738,415,616 || trainable%: 0.06%
```

### The LoRA Math (Why It Works)

```
Standard fine-tuning: update W (d × d matrix) — d² parameters
LoRA: freeze W, add low-rank perturbation: ΔW = A × B
  A: (d × r) matrix, r << d (e.g., r=16, d=4096)
  B: (r × d) matrix

Full fine-tuning: W + ΔW   — all d² parameters updated
LoRA:             W + A×B   — only r×d + d×r = 2rd parameters updated

For LLaMA 7B with d=4096, r=16:
  Full Q matrix:  4096 × 4096 = 16.7M parameters
  LoRA matrices:  4096×16 + 16×4096 = 131K parameters
  Reduction: 128× fewer parameters for this matrix

The low-rank assumption: the USEFUL updates during fine-tuning lie
in a low-dimensional subspace. LoRA exploits this observation.
```

---

## Chapter 7: Architecture Selection Guide

### The Decision Tree

```
START: What is your task?
       ↓
┌──────────────────────────────────────────────────────────────────────┐
│ Does the task require GENERATING new text that isn't in the input?   │
└──────────────────────────────────────────────────────────────────────┘
    │                              │
   YES                             NO
    │                              │
    ↓                              ↓
┌──────────────────────┐    ┌──────────────────────────────────────┐
│ Is the OUTPUT length │    │ → ENCODER-ONLY (BERT family)          │
│ closely tied to the  │    │   Classification, NER, extractive QA, │
│ INPUT structure?     │    │   semantic similarity, embeddings      │
└──────────────────────┘    └──────────────────────────────────────┘
    │              │
   YES             NO
    │              │
    ↓              ↓
┌──────────────┐  ┌──────────────────────────────────────────────┐
│ Translation, │  │ Chat, open-ended generation, code, creative  │
│ summarization│  │ writing, reasoning, instruction following    │
│ paraphrase,  │  │                                              │
│ abstractive  │  │ → DECODER-ONLY (GPT/LLaMA family)            │
│ QA           │  └──────────────────────────────────────────────┘
│              │
│ → ENCODER-   │
│   DECODER    │
│   (T5/BART)  │
└──────────────┘
```

### Task-to-Architecture Mapping Table

| Task | Best Family | Best Model Choice | Why |
|---|---|---|---|
| Text classification | Encoder | BERT / RoBERTa | [CLS] vector → classification head |
| Named Entity Recognition | Encoder | BERT / RoBERTa | Token-level outputs |
| Sentiment analysis | Encoder | DistilBERT | Fast, 97% of BERT quality |
| Extractive QA (find answer in passage) | Encoder | BERT-large | Span extraction head |
| Semantic search (dense retrieval) | Encoder | bi-encoder BERT (SBERT) | Embedding quality |
| Abstractive QA (generate answer) | Enc-Dec | T5-base / Flan-T5 | Generates free-form answer |
| Translation | Enc-Dec | mBART / Helsinki-NLP/opus | Cross-lingual + encoder-decoder |
| Summarization | Enc-Dec | BART-large-cnn / PEGASUS | Decoder trained for fluency |
| Data-to-text | Enc-Dec | T5 | Transforms structured data to text |
| Open-ended chat | Decoder | LLaMA-3-8B-Instruct | Scale + instruction fine-tuning |
| Code generation | Decoder | Code Llama / Codestral | Pre-trained on code |
| Domain text generation | Decoder | GPT-2 fine-tuned | Causal LM domain adaptation |
| Instruction following | Decoder | Flan-T5 (smaller) / LLaMA (larger) | Depends on compute budget |
| Few-shot NLP tasks | Decoder | LLaMA-3 / Mistral | Strong in-context learning |
| Clinical NLP | Encoder | BioBERT / ClinicalBERT | Domain pre-training |
| Multilingual classification | Encoder | mBERT / XLM-RoBERTa | 100-language pre-training |

### Resource vs. Quality Trade-offs

```
Compute Budget Guide (for fine-tuning):

VERY LIMITED (free Colab T4, 16GB GPU):
  → DistilBERT for classification (fine-tune fully in < 5 min)
  → T5-Small or T5-Base for seq2seq (fine-tune in < 30 min)
  → GPT-2 (124M or 355M) for generation (fine-tune in < 1 hour)

MODERATE (A100 40GB or equivalent):
  → BERT-large / RoBERTa-large for classification
  → T5-Large or BART-Large for summarization
  → LLaMA-3-8B with LoRA r=16 for instruction fine-tuning

LARGE (8× A100 80GB or cloud TPU):
  → Full fine-tuning of LLaMA-3-8B / 13B
  → T5-XL (3B) full fine-tuning
  → Training from scratch (not recommended unless you have a domain corpus)
```

### When NOT to Use Each Family

```
Do NOT use encoder-only (BERT) when:
  ✗ Task requires generating new text not present in input
  ✗ Output length varies significantly (summarization, translation)
  ✗ Task needs multi-turn conversation
  ✗ You want a single model for 10+ different tasks without task-specific heads

Do NOT use decoder-only (GPT/LLaMA) when:
  ✗ You need a fixed output format with high consistency (structured extraction)
  ✗ Task is well-defined classification with enough data → fine-tuned BERT is cheaper
  ✗ You're running at the edge / mobile / embedded → 7B+ params is impractical
  ✗ Output must be grounded in a specific source text (use RAG or enc-dec instead)

Do NOT use encoder-decoder (T5/BART) when:
  ✗ Input and output are not structurally different (classification → use BERT)
  ✗ Generation is open-ended and creative → decoder-only scales better
  ✗ Task needs reasoning over very long contexts → modern decoder-only handles better
  ✗ You need a conversational agent → use an instruction-tuned decoder model
```

---

## Chapter 8: Lab Notebooks — Theory to Practice

### Notebook 1: `healthcare_bert_classifier.ipynb` — BERT (Encoder-only)

```
What the notebook does:
  Scrapes MedlinePlus health news → 5 categories (Heart Disease, Diabetes,
  Mental Health, Cancer, Nutrition) → fine-tunes bert-base-uncased for classification

Where it fits in the architecture picture:
  ┌─────────────────────────────────────────────────────────┐
  │  Encoder-only — Paradigm 1 (head-based fine-tuning)    │
  │                                                         │
  │  Input: "High blood pressure increases cardiac risk"    │
  │    ↓ BertTokenizer → [CLS] input [SEP] [PAD] [PAD]    │
  │    ↓ bert-base-uncased (12 bidirectional layers)        │
  │    ↓ Extract [CLS] vector (768 dims)                    │
  │    ↓ Dropout(0.1)                                       │
  │    ↓ Linear(768 → 5)   ← the new head                  │
  │    → Logits [0.02, 0.85, 0.03, 0.05, 0.05]            │
  │    → Prediction: "Heart Disease" (index 2)             │
  └─────────────────────────────────────────────────────────┘

Key concepts demonstrated in notebook → where they appear in notes:
  - WordPiece tokenization               → Week 7 Ch.4, Week 7 Ch.13B
  - [CLS], [SEP], [PAD], attention mask  → Week 7 Ch.13B
  - BertForSequenceClassification        → Week 7 Ch.13B
  - Full fine-tuning (all 110M params)   → Week 7 Ch.13B
  - AdamW + warmup + gradient clipping   → Week 7 Ch.11, Ch.13B
  - MLM + NSP pre-training (why BERT works)→ Week 7 Ch.13B
```

**What the notebook does NOT cover (gaps to know for interviews):**

```
  ✗ Does not show DistilBERT / RoBERTa comparison
     → DistilBERT: 6 layers (half BERT), 40% smaller, 60% faster, 97% accuracy
     → RoBERTa: removed NSP, larger batches, better than BERT on almost everything

  ✗ Does not show token-level tasks (NER, POS tagging)
     → BertForTokenClassification → Linear(768, num_labels) at every token
     → Labels are sequences, not a single integer

  ✗ Does not show frozen backbone strategy
     → for param in model.bert.parameters(): param.requires_grad = False
     → Only train the classification head (~4K params instead of 110M)
     → Much faster prototyping but lower accuracy

  ✗ Does not show semantic similarity / sentence embeddings
     → mean-pool all token embeddings → cosine similarity
     → Used in dense retrieval (RAG pipelines, search)
```

### Notebook 2: `recipe_generator_gpt2_finetune.ipynb` — GPT-2 (Decoder-only)

```
What the notebook does:
  Loads 5938 Indian recipes from HuggingFace → formats as text strings with
  Recipe: / Ingredients: / Instructions: structure → fine-tunes GPT-2 for
  domain-specific recipe generation

Where it fits in the architecture picture:
  ┌─────────────────────────────────────────────────────────────────┐
  │  Decoder-only — Paradigm 2 (causal LM fine-tuning)              │
  │                                                                 │
  │  Training: recipe texts → GPT-2 predicts each token from past  │
  │    input:  ["Recipe:", ":","Butter","Chicken","<sep>","Ingr:"] │
  │    target: [":",  "Butter","Chicken","<sep>","Ingr:","chicken"]│
  │    loss: CrossEntropy at every non-padding position             │
  │                                                                 │
  │  Inference: prompt = "Recipe: milk pudding\nIngredients:"       │
  │    → GPT-2 autoregressively generates next tokens              │
  │    → Stops at <|endoftext|> or max_new_tokens                   │
  └─────────────────────────────────────────────────────────────────┘

Key concepts demonstrated → where they appear in notes:
  - AutoModelForCausalLM                    → Week 7 Ch.13, this doc Ch.5
  - DataCollatorForLanguageModeling(mlm=F)  → this doc Ch.5
  - HuggingFace Trainer API                 → Week 7 Ch.15
  - pipeline("text-generation")             → Week 7 Ch.15
  - temperature, do_sample                  → Week 7 Ch.12 (decoding strategies)
```

**What the notebook does NOT cover:**

```
  ✗ Does not explain WHY GPT-2 has no pad token (and why eos is used as pad)
     → GPT-2 was never trained with a pad token (it saw continuous text, no padding)
     → Fix: tokenizer.pad_token = tokenizer.eos_token

  ✗ Does not show instruction fine-tuning
     → Domain fine-tuning: train on "Recipe: ... Ingredients: ..."
     → Instruction fine-tuning: train on "User: make a recipe for X\nAssistant: ..."
     → Instruction style creates a chat-capable model; domain style creates a text-completer

  ✗ Does not show LoRA for memory efficiency
     → GPT-2 (117M) is small enough for full fine-tuning
     → For LLaMA-7B: must use LoRA (see Ch.6 of this document)

  ✗ Does not show perplexity evaluation
     → Perplexity = exp(avg cross-entropy loss) — the standard metric for CLM
     → eval_loss from Trainer can be converted: perplexity = math.exp(eval_loss)
     → Lower perplexity = model assigns higher probability to real text
```

### Notebook 3 (Missing): What a T5 Encoder-Decoder Notebook Would Look Like

The current week-8 folder has two notebooks (encoder-only and decoder-only) but
NO encoder-decoder notebook. Here is the template you can build when ready:

```python
# Hypothetical: t5_summarizer.ipynb
# Task: Abstractive summarization of research paper abstracts

from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Dataset: scientific paper abstracts (input) → one-sentence summaries (target)
dataset = load_dataset("scientific_papers", "arxiv", split="train[:5000]")

model_name = "t5-base"
tokenizer  = T5Tokenizer.from_pretrained(model_name)
model      = T5ForConditionalGeneration.from_pretrained(model_name)

# CRITICAL difference from BERT: labels are TEXT sequences, not integers
def preprocess(examples):
    inputs = ["summarize: " + a for a in examples["abstract"]]
    targets = examples["section_names"]  # title as one-line summary proxy

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=64, truncation=True, padding="max_length")

    # Replace padding ID (0) with -100 so loss ignores it
    label_ids = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = label_ids
    return model_inputs

# Evaluation uses ROUGE (not accuracy — we're evaluating generated text)
import evaluate
rouge = evaluate.load("rouge")

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    decoded_preds  = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    return {k: round(v, 4) for k, v in result.items()}

training_args = TrainingArguments(
    output_dir="./t5-summarizer",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    predict_with_generate=True,   # CRITICAL for seq2seq — use model.generate() for eval
    generation_max_length=64,
    fp16=True
)
```

---

## Chapter 9: Model Taxonomy — Every Major Model Placed

### Encoder-Only Family (Understanding → BERT lineage)

```
BERT (2018, Google)
  - 12 layers, 768 hidden, 110M params (base)
  - Pre-trained: MLM + NSP on BooksCorpus + Wikipedia
  - Tokenizer: WordPiece, 30,522 vocab, uncased/cased variants
  - Breakthrough: bidirectional pre-training for NLP understanding

RoBERTa (2019, Facebook)
  - Same architecture as BERT-base/large
  - Key changes: removed NSP, larger batches (256→8192), 10× more data
  - Dynamic masking: new masks each epoch (BERT uses static masks)
  - Result: outperforms BERT on nearly all GLUE benchmarks
  - Use instead of BERT for best encoder-only performance

DistilBERT (2019, HuggingFace)
  - 6 layers (half of BERT), 66M params
  - Trained via knowledge distillation from BERT-base (teacher)
  - 60% faster, 40% smaller, 97% of BERT performance
  - Production choice when latency matters

ALBERT (2019, Google)
  - Parameter sharing: all encoder layers share weights → very small
  - Factorized embeddings: embedding size << hidden size
  - 12M params (ALBERT-base) vs 110M (BERT-base)
  - Used when memory is severely constrained

DeBERTa (2020, Microsoft)
  - Disentangled attention: content and position attention separate
  - Enhanced mask decoder for MLM
  - Best encoder-only on most benchmarks (especially NLU)
  - Use for best classification/NER results if you have compute

Domain-specific variants:
  BioBERT      → PubMed + PMC  → biomedical NLP
  ClinicalBERT → MIMIC-III     → clinical notes, EHR
  SciBERT      → Semantic Scholar → scientific text
  LegalBERT    → legal text    → contract analysis
  FinBERT      → financial news → sentiment in finance
```

### Decoder-Only Family (Generation → GPT lineage)

```
GPT-2 (2019, OpenAI)
  - 4 sizes: 117M, 345M, 762M, 1.5B
  - Pre-trained on WebText (Reddit outbound links, 40GB text)
  - Tokenizer: BPE, 50,257 vocab
  - Open weights — the learning model in the recipe notebook

GPT-3 (2020, OpenAI)
  - 175B parameters, API-only (not open)
  - Demonstrated in-context learning with few-shot prompts
  - Foundation of the "large language models can do anything" era

ChatGPT / GPT-3.5 (2022, OpenAI)
  - GPT-3.5 base + SFT + RLHF (see Week 7 Ch.14)
  - First mainstream chat LLM, 100M users in 2 months

GPT-4 (2023, OpenAI)
  - Architecture not disclosed (likely MoE)
  - Multimodal (text + images)

LLaMA (2023, Meta)
  - 7B, 13B, 33B, 65B — open weights
  - Uses RoPE, SwiGLU, RMSNorm, GQA (see Week 7 Ch.13)
  - Foundation of open-source LLM ecosystem

LLaMA 2 (2023, Meta)
  - 7B, 13B, 70B — more data (2T tokens), RLHF for chat variants
  - First Meta release with commercial license

LLaMA 3 / 3.1 / 3.2 (2024, Meta)
  - 8B, 70B, 405B
  - 15T tokens, 128K context (3.1), multimodal (3.2)
  - State-of-the-art open-weight LLM as of 2024-2025

Mistral 7B (2023, Mistral AI)
  - 7B params but outperforms LLaMA 2 13B on many benchmarks
  - Sliding window attention for long context
  - Grouped query attention (GQA)

Mixtral 8×7B (2024, Mistral AI)
  - Mixture of experts: 8 experts, 2 active per token
  - Effective: 13B active params, total 47B stored params
  - Outperforms LLaMA 2 70B at 13B inference cost

Gemini (2023-2024, Google DeepMind)
  - Multimodal from the start (text, image, audio, video)
  - Ultra, Pro, Nano sizes
  - Nano designed for on-device inference

Claude (Anthropic)
  - Constitutional AI training (RLHF variant with AI feedback)
  - Strong reasoning, long-context, safety focus
```

### Encoder-Decoder Family (Seq2Seq → T5/BART lineage)

```
Original Transformer (2017, Google)
  - 6 encoder + 6 decoder layers, d=512
  - Introduced for English-German translation
  - The paper that started everything

T5 (2019, Google)
  - Unified text-to-text framework
  - span corruption pre-training
  - Sizes: 60M to 11B
  - Pre-trained on C4 (750GB cleaned web text)

Flan-T5 (2022, Google)
  - T5 + instruction fine-tuning on 1800+ tasks
  - Best open seq2seq model for instruction following
  - Outperforms GPT-3 (175B) on many benchmarks using just 11B params
  - Use for: zero-shot classification, QA, summarization with prompts

mT5 (2020, Google)
  - T5 pre-trained on 101 languages (mC4 corpus)
  - Use for multilingual summarization and translation

BART (2019, Facebook)
  - Denoising autoencoder pre-training (5 noise functions)
  - Best for: abstractive summarization, data-to-text
  - bart-large-cnn: fine-tuned on CNN/DailyMail news summarization

PEGASUS (2020, Google)
  - Gap sentence generation: randomly remove full sentences from document,
    train decoder to reconstruct them
  - Pre-training matches summarization task structure exactly
  - Best ROUGE scores on news summarization (XSum, CNN/DailyMail)

mBART (2020, Facebook)
  - BART pre-trained on 25 languages
  - Used for multilingual translation and summarization

NLLB (2022, Meta)
  - No Language Left Behind
  - 200 languages, specifically designed for low-resource translation
  - facebook/nllb-200-distilled-600M
```

---

## Chapter 10: Interview Questions — Week 8 Focus

These questions specifically test week-8 material (cross-attention, T5/BART, encoder-decoder).
For BERT-specific and GPT-specific questions see Week 7 Ch.13B and Ch.13.

---

### Encoder vs Decoder vs Encoder-Decoder

> **Beginner:** Name one model from each of the three Transformer architecture families
> and one task each is best suited for.
> → Encoder-only: BERT — text classification. Decoder-only: GPT-2 — text generation.
> Encoder-Decoder: T5 — text summarization.

> **Intermediate:** Why can encoder-only models NOT generate text?
> → Because they use bidirectional attention — every token sees every other token,
> including future tokens. Autoregressive generation requires generating one token at
> a time, where each new token can only see the tokens generated so far (causal attention).
> BERT was never trained with a causal LM objective, so there is no output head for
> next-token prediction (lm_head). You would need to add and train one from scratch,
> which defeats the purpose of pre-training.

> **Advanced:** Is it possible to use an encoder-only model for generation? Explain the
> limitations.
> → Yes, with modifications. BERT-gen (Dong et al., 2019) showed you can use BERT for
> generation by applying a causal mask during inference. However, the pre-training
> objective (MLM) did not train BERT to predict tokens autoregressively, so generation
> quality is poor compared to decoder-only models of similar size. The representation
> quality for understanding tasks also degrades if you apply a causal mask during
> fine-tuning. This is why separate encoder and decoder models exist rather than
> reusing one for both.

---

### Cross-Attention Specifics

> **Beginner:** In an encoder-decoder model, what are Q, K, V in cross-attention?
> → Q (Query) comes from the decoder's current position.
> K (Key) and V (Value) come from the encoder's output.
> Each decoder position asks "which encoder positions are relevant to me right now?"
> by computing Q·K scores, then weighing the encoder's V (values) by those scores.

> **Intermediate:** How is cross-attention different from the masked self-attention
> in the decoder?
> → Masked self-attention: Q, K, V all from the same (decoder) sequence. Causal mask
> prevents attending to future positions. Used to build up context from the target
> sequence generated so far.
> Cross-attention: Q from decoder, K/V from encoder. No mask — all encoder positions
> are visible. Used to read relevant information from the source sequence.

> **Advanced:** What happens to cross-attention at inference time when generating
> long sequences?
> → The encoder output is computed once and cached before generation starts. Cross-attention
> K and V matrices are derived from this fixed encoder output — they do NOT change as
> generation proceeds. Only the decoder's self-attention KV cache grows (one entry per
> new token). This means long-sequence generation does NOT increase cross-attention
> computation — it's O(1) relative to output length. Only the causal self-attention
> is O(n²) with output length, making long encoder-decoder generation more efficient
> than a pure decoder with a very long prefix.

---

### T5 and Span Corruption

> **Beginner:** What does the T5 prefix "summarize:" do?
> → T5 was pre-trained on many tasks simultaneously, each with a natural language prefix
> that tells the model which task to perform. "summarize:" activates the summarization
> behavior learned during multi-task pre-training. Without the prefix, the model has
> no signal about which task to run and generates poor output.

> **Intermediate:** How does T5's span corruption differ from BERT's MLM?
> → BERT MLM: masks individual tokens (one at a time), predicts them in-place in the
> encoder. Output is same-length as input. T5 span corruption: replaces consecutive
> spans with single sentinel tokens (e.g., <extra_id_0>), then trains the DECODER to
> generate only the missing spans. Output is shorter than input, and the model learns
> autoregressive generation — a skill BERT never develops.

> **Advanced:** Why does Flan-T5 outperform T5 on zero-shot tasks despite being the
> same model?
> → Flan-T5 is T5 fine-tuned on a mixture of 1800+ NLP tasks formatted as natural
> language instructions ("Answer the following question:", "Classify the sentiment:",
> etc.). This instruction tuning teaches the model to follow diverse natural language
> commands without task-specific prefixes or examples. Zero-shot performance improves
> because the model has seen many examples of how to follow instructions — not because
> it gained new knowledge, but because it learned to apply existing knowledge when asked
> in natural language. This is the same principle behind ChatGPT's SFT stage.

---

### BART Pre-training

> **Beginner:** What is denoising pre-training in BART?
> → BART is pre-trained as a denoising autoencoder: take clean text, corrupt it with
> one or more noise functions (masking, deleting, shuffling sentences), then train the
> model to reconstruct the original clean text. The encoder reads the noisy version;
> the decoder generates the clean version. This gives BART strong generation ability.

> **Intermediate:** Which BART noise function is most responsible for its summarization
> performance, and why?
> → Sentence permutation — it shuffles the order of sentences in the training document.
> The decoder must reconstruct the CORRECT order. This teaches BART to model discourse-
> level coherence and understand what constitutes a logical document beginning — which
> is directly the skill needed for abstractive summarization (generating a coherent
> summary that starts with the most important point).

> **Advanced:** When would you choose BART over T5 for a seq2seq task?
> → BART is preferred for tasks requiring high-fluency generation with strong discourse
> structure: abstractive summarization (bart-large-cnn, bart-large-xsum are benchmarks),
> dialogue response generation, story generation. BART's decoder was initialized from
> GPT-2 weights, giving it strong priors for fluent English text.
> T5 is preferred when: (1) the task benefits from the unified text-to-text framing,
> (2) multi-task or zero-shot performance matters (Flan-T5), (3) multilingual tasks
> (mT5), (4) structured/classification output formats (T5 can output "positive" as text).
> In practice: start with Flan-T5 for instruction-following; start with BART-large-cnn
> for summarization.

---

### Fine-Tuning Patterns

> **Beginner:** What is the difference between the "labels" format for BERT vs T5 fine-tuning?
> → BERT: labels are a single integer per sample (e.g., label=2 for "Heart Disease").
> T5: labels are a sequence of token IDs representing the full target text
> (e.g., [1893, 3456, 42, 1] for "Heart Disease" tokenized). This is because BERT
> adds a linear classification head with one output per class, while T5's decoder
> generates tokens one at a time using the full sequence as its target.

> **Intermediate:** What is teacher forcing in seq2seq training, and what problem
> does it solve?
> → During seq2seq training, at each decoder step, instead of feeding the model's
> own previous prediction as input (which may be wrong early in training), we feed the
> GROUND TRUTH previous token. This prevents error accumulation — a wrong token early
> would corrupt all subsequent steps. During inference, the model's own predictions
> are used (no ground truth available), creating a small train/inference gap called
> "exposure bias." Teacher forcing is handled automatically by HuggingFace's Trainer
> when you pass labeled sequences.

> **Advanced:** How do you apply LoRA to a T5 model and which modules should you target?
> → T5's attention weights are named "q", "k", "v", "o" (for output projection) inside
> each T5Block. LoraConfig(target_modules=["q", "v"]) targets these. The "q" and "v"
> projections are most commonly targeted because research (Hu et al., LoRA paper) found
> they carry the most task-relevant information. "k" (key) projections influence which
> tokens get attended to but are less sensitive to the task specifics. For T5, using
> TaskType.SEQ_2_SEQ_LM correctly applies LoRA adapters to both encoder and decoder
> attention, since both need to be fine-tuned for seq2seq tasks.

---

## Appendix A: Pre-training Objectives Cheat Sheet

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Objective          │ Model  │ Description                                   │
├────────────────────┼────────┼───────────────────────────────────────────────┤
│ MLM                │ BERT   │ Mask 15% of tokens, predict them in-place    │
│ NSP                │ BERT   │ Predict if sentence B follows sentence A      │
│ CLM                │ GPT    │ Predict next token from all previous tokens   │
│ Span Corruption    │ T5     │ Replace spans with sentinels, reconstruct     │
│ Denoising AE       │ BART   │ Corrupt text 5 ways, reconstruct original    │
│ Gap Sentence Gen.  │ PEGASUS│ Remove whole sentences, generate them        │
│ Contrastive        │ SimCSE │ Similar sentences closer in embedding space  │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Appendix B: Attention Mask Types Cheat Sheet

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Mask Type           │ Where Used         │ What It Masks                   │
├─────────────────────┼────────────────────┼─────────────────────────────────┤
│ Padding mask        │ All models         │ [PAD] tokens (value = 0)        │
│ Causal mask         │ Decoder self-attn  │ Future positions (lower triangle)│
│ Combined mask       │ Decoder self-attn  │ Causal + padding together        │
│ Encoder mask        │ Cross-attention K/V│ Encoder padding positions        │
│                     │ (in enc-dec model) │                                 │
│ MLM mask            │ BERT pre-training  │ 15% randomly selected tokens    │
│ Span corruption mask│ T5 pre-training    │ Consecutive spans → sentinels   │
└─────────────────────────────────────────────────────────────────────────────┘

Note: "attention_mask" in HuggingFace tokenizers ALWAYS refers to the PADDING mask
(1 = real token, 0 = pad). The causal mask is applied INSIDE the model automatically
for decoder models — you never pass it manually.
```

## Appendix C: HuggingFace Model Class Quick Reference

```python
# Encoder-only (BERT family):
from transformers import (
    BertForSequenceClassification,      # single or multi-label classification
    BertForTokenClassification,         # NER, POS tagging
    BertForQuestionAnswering,           # extractive QA (start/end span)
    BertForMaskedLM,                    # fine-tune for MLM (domain adaptation)
    RobertaForSequenceClassification,   # RoBERTa variants — same pattern
    DistilBertForSequenceClassification # DistilBERT — same pattern
)

# Decoder-only (GPT family):
from transformers import (
    GPT2LMHeadModel,                    # GPT-2 text generation (recipe notebook)
    AutoModelForCausalLM,               # any decoder: GPT-2, LLaMA, Mistral
)

# Encoder-Decoder (T5/BART family):
from transformers import (
    T5ForConditionalGeneration,         # T5 for any seq2seq task
    BartForConditionalGeneration,       # BART (summarization, translation)
    AutoModelForSeq2SeqLM,              # any enc-dec: T5, BART, PEGASUS, mBART
)

# Universal auto classes (when you don't know the architecture):
from transformers import (
    AutoModel,                          # base model (no task head)
    AutoModelForSequenceClassification, # detects and loads right class
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,                      # loads correct tokenizer automatically
)
```

## Appendix D: Cross-Week Reading Map

```
┌────────────────────────────────────────────────────────────────────────────┐
│  Concept                          │  Week 7 Location  │  Week 8 Location   │
├───────────────────────────────────┼───────────────────┼────────────────────┤
│  Tokenization (BPE, WordPiece)    │  Ch.4             │  —                 │
│  Embeddings                       │  Ch.5             │  —                 │
│  Positional encoding (sin/cos,RoPE│  Ch.6             │  —                 │
│  T5 relative position bias        │  —                │  Ch.3 here         │
│  Self-attention mechanics         │  Ch.7             │  —                 │
│  Cross-attention mechanics        │  Ch.10 (1 line)   │  Ch.2 here (full)  │
│  Multi-head attention             │  Ch.8             │  —                 │
│  FFN + LayerNorm                  │  Ch.9             │  —                 │
│  Full architecture (enc/dec table)│  Ch.10 (brief)    │  Ch.1 here (map)   │
│  Causal LM training objective     │  Ch.11            │  —                 │
│  Decoding strategies              │  Ch.12            │  —                 │
│  GPT, LLaMA architectures        │  Ch.13            │  —                 │
│  BERT (MLM, NSP, variants)        │  Ch.13B           │  —                 │
│  T5 (span corruption, fine-tune)  │  —                │  Ch.3 here (full)  │
│  BART (denoising, noise funcs)    │  —                │  Ch.4 here (full)  │
│  Fine-tuning (SFT, RLHF, LoRA)   │  Ch.14            │  Ch.6 here (ext.)  │
│  Causal LM fine-tuning specifics  │  —                │  Ch.5 here         │
│  Architecture selection guide     │  —                │  Ch.7 here         │
│  HuggingFace Trainer, pipelines   │  Ch.15            │  —                 │
│  Lab notebook: BERT classifier    │  (notebook only)  │  Ch.8 here         │
│  Lab notebook: GPT-2 recipes      │  (notebook only)  │  Ch.8 here         │
└───────────────────────────────────┴───────────────────┴────────────────────┘
```

---

*This document covers Week 8 material only. All cross-references to Week 7 chapters
point to `week-7-transformers-llm/Transformers_LLM_Comprehensive_Guide.md`.*
