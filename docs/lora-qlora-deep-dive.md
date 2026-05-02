# LoRA & QLoRA — Complete Deep Dive

> A from-scratch teaching guide that builds intuition, math, and practical understanding of LoRA fine-tuning. Specifically grounded in the Kiki SLM context (Qwen3-4B-Thinking-2507, rank 32 → 64 for v2).

---

## Table of Contents

- [Part 1 — Why LoRA exists in the first place](#part-1)
- [Part 2 — The key insight that makes LoRA work](#part-2)
- [Part 3 — The matrix factorization (the heart of LoRA)](#part-3)
- [Part 4 — What "rank" actually means (intuition)](#part-4)
- [Part 5 — Why rank 32 → 64 gives "headroom"](#part-5)
- [Part 6 — Which layers of the LLM get LoRA, and how](#part-6)
- [Part 7 — How the forward pass works (with LoRA, at inference)](#part-7)
- [Part 8 — QLoRA: how 4-bit quantization works](#part-8)
- [Part 9 — Why this all matters for your v2 retrain](#part-9)
- [Mental model recap (the things to remember forever)](#recap)

---

<a id="part-1"></a>
# Part 1 — Why LoRA exists in the first place

A neural network is just a chain of matrix multiplications. When you fine-tune, you're nudging those matrices to behave differently on your task. The naive way:

```
new_weights = old_weights + change
W'         = W           + ΔW
```

For Qwen3-4B, the model has ~4 billion parameters spread across weight matrices. To "fine-tune" the traditional way, you'd need to compute and store `ΔW` for every weight matrix.

**Here's where the memory disaster starts:**

```
Base model weights (W):             ~8 GB (4B params × 2 bytes for fp16)
Gradient for each param:            ~8 GB (same shape as W)
Adam optimizer state (m, v):        ~16 GB (TWO tensors same shape as W)
Activations cached for backprop:    ~8 GB (depends on batch size)
─────────────────────────────────────────
Total memory needed for full FT:    ~40 GB
```

That doesn't fit on any single consumer GPU. An A100 (40 GB) might JUST fit it with tiny batches. A Colab L4 (24 GB)? Forget it.

The problem isn't computation — it's that we're treating EVERY parameter as a learnable variable when most of them probably don't need to change much.

---

<a id="part-2"></a>
# Part 2 — The key insight that makes LoRA work

In 2021, the LoRA paper made a beautifully simple observation:

> "When you fine-tune for a specific task, the change `ΔW` doesn't actually need 4 billion degrees of freedom. The meaningful change has a **low intrinsic rank**."

Translation: **the way the model's behavior changes during fine-tuning lies in a small subspace** of all possible changes. You don't need to update every direction in 2560-dimensional space — you only need to update maybe 32 important directions.

This is the same idea as image compression. A 4K photograph has 8 million pixels, but it usually compresses to a few hundred KB because the actual *information* in the image is much smaller than the raw pixel count suggests. Most of those pixels are predictable from neighboring ones.

LoRA does the same thing for fine-tuning updates. Instead of storing the full `ΔW`, it stores a **compressed representation** of it.

---

<a id="part-3"></a>
# Part 3 — The matrix factorization (this is the heart of LoRA)

Let's pick one specific weight matrix to work with. In Qwen3-4B, the attention layer has a "query projection" matrix called `W_q`. It has shape `(2560, 2560)` — that's **6.5 million parameters** in just this one matrix.

To fully fine-tune `W_q`, we'd need to learn:

```
ΔW_q : shape (2560, 2560) = 6.5M trainable parameters
```

LoRA's trick: instead of learning `ΔW_q` directly, factor it into two thin matrices:

```
        2560               2560 → →
       ┌────┐             ┌────────┐
       │    │             │   B    │   ← 32 × 2560 = 81,920 params
       │ ΔW │   ≈    A    │        │
2560 ↓ │    │   2560 × 32 └────────┘
       │    │   = 81,920
       │    │   params
       └────┘
       6.5M params
       (don't learn this)
```

Where:
- `A` has shape `(2560, 32)` — a tall, skinny matrix
- `B` has shape `(32, 2560)` — a short, wide matrix
- Their product `A × B` has shape `(2560, 2560)` — same as the original
- Their product can approximate any rank-32 matrix exactly

**Parameter count comparison:**

```
Full ΔW:        2560 × 2560 = 6,553,600 trainable params per matrix
LoRA r=32:      2 × 2560 × 32 = 163,840 trainable params per matrix

Reduction:      6,553,600 / 163,840 = 40× fewer parameters
```

You get 40× fewer parameters per matrix while still being able to express any rank-32 update to that matrix.

The forward pass becomes:

```
                 ↓ frozen, not learning
y = W_q · x  +  α/r · (A · B) · x
        ↑              ↑
    base model         LoRA adapter (the only thing that learns)
```

`α/r` is a scaling factor (we'll get to this). The base model contribution is just standard inference — those weights never change. Only `A` and `B` get updated by gradients during training.

---

<a id="part-4"></a>
# Part 4 — What "rank" actually means (intuition)

The number `r` (rank) is the most important LoRA parameter. Let me build intuition four different ways:

## Way 1: The "expressiveness budget" view

Think of `r` as the number of **independent directions** the adapter can push the model in.

- `r = 1`: the adapter can only express ONE specific change. Every input gets the same kind of nudge.
- `r = 8`: 8 independent ways to modify the model's behavior. Enough for simple classification (e.g., sentiment).
- `r = 32`: 32 directions. Enough for moderate complexity (e.g., domain-specific instruction following).
- `r = 64`: 64 directions. Enough for complex multi-skill tasks (Kiki — 11-way classification + tool calling + structured generation).
- `r = 256`: starts approaching full fine-tuning capacity. Diminishing returns set in.

## Way 2: The "knob" analogy

Imagine the base model has 6.5 million tiny knobs (one per parameter in `W_q`). Full fine-tuning lets you turn every knob independently. LoRA says: "you don't have 6.5M independent decisions to make. Group these knobs into `r` clusters, and only let the user turn each cluster as a unit."

- `r = 32`: you have 32 cluster-level knobs to turn. You can express any pattern that's consistent across the original 6.5M knobs in one of these 32 patterns.
- `r = 64`: 64 cluster-level knobs. You can express twice as many patterns simultaneously.

## Way 3: The subspace view (technically accurate)

A 2560×2560 matrix represents a transformation in 2560-dimensional space. Updating it with full fine-tuning could rotate, stretch, or skew along ANY direction in that 2560-dim space.

LoRA with rank `r` says: "your update only acts in `r` of those 2560 dimensions. In the other 2560 - `r` dimensions, the model is unchanged."

So:
- `r = 32`: your fine-tuning operates within a 32-dimensional subspace of all possible changes
- `r = 64`: 64-dimensional subspace — can express changes in twice as many independent directions
- `r = 2560`: equivalent to full fine-tuning (no compression)

## Way 4: The image-compression analogy

A 1080p video is 2 million pixels per frame. But you can record it in different "qualities":
- 240p: very small, captures only the gross outlines
- 480p: medium, captures most details
- 1080p: full, captures everything
- 4K: more pixels than the source — pure waste

LoRA rank works the same way. Your fine-tuning task has some "true rank" — the actual complexity of behavior you're teaching the model. If true rank is ~50 and you train at:
- `r = 8`: the adapter is too compressed to capture the task. Underfits.
- `r = 32`: most of the task fits, but some nuance is lost.
- `r = 64`: full task fits with room to spare.
- `r = 256`: wastes parameters; might overfit small details of training data.

The challenge is you don't know the true rank in advance. You experiment.

---

<a id="part-5"></a>
# Part 5 — Why rank 32 → 64 gives "headroom"

Most explanations skip this and just say "more capacity = better." That's lazy. Let me show you what's actually happening.

## 5.1 What capacity actually is

Imagine you're trying to teach a friend to recognize 11 types of customer support tickets. They have a notebook with limited pages. Each page can hold the description of one "pattern."

- 1-page notebook: friend can only learn ONE pattern. They'll lump everything into that pattern.
- 11-page notebook: each ticket type gets its own page. Perfect.
- 100-page notebook: lots of room. Each ticket type can get multiple pages with sub-patterns.
- 1,000,000-page notebook: most pages are wasted, but no harm done if you can afford it.

The notebook pages = the model's **capacity**. It's the number of independent patterns the model can store and recall.

In a LoRA adapter at rank `r`, the "notebook" has `r` pages.

## 5.2 How patterns actually get stored in a LoRA adapter

This is the part most explanations skip. Let me show you mechanically.

When you train a LoRA adapter, what's happening is gradient descent is finding values for the matrices `A` and `B` that minimize loss on your training data. Let me trace through what `A` and `B` end up looking like.

Recall:
```
A: shape (d, r)   ← d rows, r columns
B: shape (r, d)   ← r rows, d columns
```

Where `d = 2560` (hidden dim of Qwen3-4B) and `r` is your rank (32 in v1, 64 in v2).

**Each column of `A`** is a 2560-dimensional vector. There are `r` columns. Think of each column as a "pattern detector" — a direction in the input space that, when matched, activates that column.

**Each row of `B`** is also a 2560-dimensional vector. There are `r` rows. Think of each row as a "pattern emitter" — what to output when the corresponding column of `A` is activated.

So columns of `A` and rows of `B` come in pairs. **You have `r` pairs of (detector, emitter).**

```
Adapter pair 1:  Column 1 of A detects → Row 1 of B emits
Adapter pair 2:  Column 2 of A detects → Row 2 of B emits
...
Adapter pair r:  Column r of A detects → Row r of B emits
```

When an input vector `x` flows through the adapter:

```
1. A^T · x    →   r-dim vector (how much each detector matched)
2. then B^T   →   weighted sum of emitters based on detection strength
```

So the adapter has `r` slots, where each slot stores ONE (input pattern → output transformation) rule.

**This is what rank means physically.** The adapter has `r` slots. Each slot can hold one input/output pattern. That's it.

## 5.3 What the model learns to put in those slots — for Kiki

Now let's think about what patterns the Kiki adapter needs to learn:

For each transformer layer's adapter (and there are 252 of them), the model needs to learn:

| Need | Approximate # of slots needed |
|---|---|
| Detect "this is a delivery_issue ticket" pattern | 2-3 slots |
| Detect "this is a design_update ticket" pattern | 2-3 slots |
| Detect "this is a refund_request ticket" pattern | 2-3 slots |
| (... 11 categories × ~2-3 slots each ...) | 22-33 slots |
| Detect language signals (French, German, Spanish, etc.) | 6-8 slots |
| Detect urgency markers (delayed, urgent, deadline) | 2-3 slots |
| Detect rejection patterns (spam, newsletter) | 3-5 slots |
| Detect "should call rag_search" patterns | 2-3 slots |
| Patterns for tool-call argument generation | 4-6 slots |
| Patterns for JSON structure generation | 3-5 slots |
| Patterns for tone/empathy on complaints | 3-5 slots |
| **Total ideal slots needed** | **~50-70** |

So the model **wants** about 50-70 slots to do its job well. With `r = 32`, you only have 32 slots. The model has to compromise.

## 5.4 What gradient descent does when there aren't enough slots

This is where it gets interesting. Gradient descent doesn't say "OK I'll spread the slots evenly across the 11 categories." It allocates slots based on **gradient signal strength** — which is proportional to **how often a pattern shows up in training**.

Your v1 training data:
```
design_update:        1,030 examples (25.1%)  ← strong gradient signal
delivery_issue:         873 examples (21.3%)  ← strong gradient signal
new_order_inquiry:      677 examples (16.5%)  ← moderate
quality_complaint:      168 examples ( 4.1%)  ← weak
order_cancellation:     147 examples ( 3.6%)  ← weak
payment_confirmation:   140 examples ( 3.4%)  ← weak
sample_request:          78 examples ( 1.9%)  ← very weak
price_negotiation:       44 examples ( 1.1%)  ← very weak
customer_feedback:       33 examples ( 0.8%)  ← negligible
refund_request:          30 examples ( 0.7%)  ← negligible
```

When you run gradient descent on this distribution, the slots get allocated roughly proportionally to the gradient signal. So with 32 slots, the allocation looks something like:

```
With r=32 (v1):

Slots 1-3:   design_update detection + variations    (3 slots taken)
Slots 4-6:   delivery_issue detection + variations   (3 slots)
Slots 7-9:   new_order_inquiry                       (3 slots)
Slots 10:    other (catch-all)                       (1 slot)
Slots 11-13: tool-calling patterns                   (3 slots)
Slots 14-15: JSON structure patterns                 (2 slots)
Slots 16-19: language detection                      (4 slots)
Slots 20-22: rejection patterns                      (3 slots)
Slots 23-24: quality_complaint (split with delivery) (2 slots)
Slots 25-26: payment_confirmation                    (2 slots)
Slots 27:    sample_request (compressed)             (1 slot)
Slots 28:    order_cancellation                      (1 slot)
Slots 29:    urgency markers                         (1 slot)
Slots 30-32: tone/format patterns                    (3 slots)

Slots needed but not allocated:
- price_negotiation        ← model has to confuse it with new_order_inquiry
- customer_feedback        ← model has to confuse it with "other"
- refund_request           ← model has to confuse it with delivery_issue
- empathy patterns         ← model can't develop dedicated empathy
- per-language closings    ← English-style closings leak into French/German
- complex tool-call patterns ← only simple patterns get learned
```

When refund_request training examples show up, gradient descent tries to push the adapter in a "refund-like direction." But there's no dedicated slot for it. The gradient ends up partially overwriting the delivery_issue slots (because refunds also mention delivery delays sometimes) and the customer-feedback area. The result: the model "kinda" learns refund but always confuses it with delivery_issue at inference.

This is what "starvation" of rare classes looks like at the LoRA level.

## 5.5 What rank 64 actually changes

With `r = 64`, you have 64 slots. The dominant categories still claim their fair share (because their gradient signal hasn't changed), but now there's headroom:

```
With r=64 (v2):

Slots 1-6:   design_update + variations               (6 slots — luxury)
Slots 7-12:  delivery_issue + variations              (6 slots)
Slots 13-17: new_order_inquiry                        (5 slots)
Slots 18-22: quality_complaint                        (5 slots) ← previously 2
Slots 23-26: payment_confirmation                     (4 slots)
Slots 27-30: order_cancellation                       (4 slots)
Slots 31-33: sample_request                           (3 slots) ← previously 1
Slots 34-36: price_negotiation                        (3 slots) ← previously 0!
Slots 37-39: customer_feedback                        (3 slots) ← previously 0!
Slots 40-42: refund_request                           (3 slots) ← previously 0!
Slots 43-45: tool-calling patterns                    (3 slots)
Slots 46-48: JSON structure                           (3 slots)
Slots 49-54: language detection (6 languages)         (6 slots)
Slots 55-57: rejection patterns                       (3 slots)
Slots 58-60: tone/empathy patterns                    (3 slots)
Slots 61-62: urgency including critical               (2 slots)
Slots 63-64: edge cases / adaptation                  (2 slots)
```

Notice the rare categories (refund_request, customer_feedback, price_negotiation) now get their own dedicated slots. They're no longer fighting the dominant categories for representation.

**Key insight:** rank 64 doesn't make refund_request data more abundant. It just stops the abundant categories from STARVING the rare ones at the architectural level.

The 30 refund examples can now develop their own pattern in slots 40-42 without being overwritten by the 1,030 design_update examples occupying slots 1-6. Each cluster of slots is somewhat independent because gradient descent can find different `A` and `B` columns for different patterns.

## 5.6 The "headroom" metaphor made concrete

So when I said "headroom" in v2, what I meant mechanically:

- v1 (rank 32): adapter is at capacity. Adding new categories or refining existing ones REQUIRES sacrificing something else.
- v2 (rank 64): adapter has spare slots. New patterns can be learned without overwriting existing ones.

This is why you can't fix the v1 problems just by adding more rare-category data without ALSO increasing rank. If you keep rank at 32 and add more refund examples, gradient descent will eventually overfit on the dominant classes and still under-allocate slots to refund. You need both more data AND more capacity.

## 5.7 Why not rank 128 or higher?

The diminishing returns curve says: at some point, adding slots doesn't help because there ISN'T 128 distinct patterns to learn. You'd just be paying for empty notebook pages.

Empirically, for an 11-class structured-output task at the 4B parameter scale with ~8K training examples:
- Rank 16: too tight, even dominant categories get squeezed
- Rank 32: covers dominant classes well, starves rare classes (your v1)
- Rank 64: covers all classes including rare ones (v2)
- Rank 128: marginal improvement (~1-2% on rare class accuracy, doubles training time)
- Rank 256: no measurable improvement, 4× training cost

There's also an OVERFITTING risk at high rank. With 4× the parameters but the same amount of data, the adapter can start memorizing specific training examples instead of learning general patterns.

So you pick the smallest rank that covers your task complexity. For Kiki, that's 64.

## 5.8 About `alpha` (the often-misunderstood parameter)

Alpha is a scaling factor applied to the LoRA update:

```
y = W·x + (α / r) · (A·B)·x
            ↑
        scaling
```

The convention is to set `alpha = 2 × r` (so `α/r = 2`). This was empirically found to work well across many setups.

Why scale at all? When `A` and `B` are randomly initialized to small values, their product `A·B` is tiny. As they grow during training, the LoRA contribution naturally grows. The `α/r` scale ensures that a "fully grown" LoRA adapter has roughly the same magnitude of effect across different rank choices.

Without scaling, switching from rank 32 to 64 would also DOUBLE the effective magnitude of LoRA updates, which would force you to also halve the learning rate. With `α = 2r`, switching ranks is hyperparameter-stable: `α/r = 2` regardless of `r`.

So in v2 we keep `α/r = 2`:
```
v1:  r=32, α=64,  α/r=2
v2:  r=64, α=128, α/r=2  ← same effective magnitude, more capacity
```

---

<a id="part-6"></a>
# Part 6 — Which layers of the LLM get LoRA, and how

Let me teach you transformer architecture from scratch so you understand WHERE LoRA gets injected and WHY it matters.

## 6.1 What a transformer is, fundamentally

A transformer is a stack of identical "layers" that take a sequence of tokens and progressively transform them. Each layer makes the tokens "smarter" — they understand more about the surrounding context.

```
Input text:  "My order is delayed by 3 days"

Step 1: Tokenize and embed
        ┌──────┬───────┬─────┬─────────┬─────┬───┬──────┐
        │ "My" │"order"│"is" │"delayed"│"by" │"3"│"days"│
        └──────┴───────┴─────┴─────────┴─────┴───┴──────┘
        Each token becomes a vector (in Qwen3-4B, a 2560-dim vector)

        Tokens after embedding (showing as small vectors for clarity):
        [v_My, v_order, v_is, v_delayed, v_by, v_3, v_days]

Step 2: Pass through Layer 1
        Each vector gets updated based on the others.
        After layer 1, v_order "knows" it's an order being talked about.
        After layer 1, v_delayed "knows" it modifies "order" in this context.

Step 3: Pass through Layer 2, 3, ..., 36
        Each layer adds more context-aware understanding.
        By layer 36, the final hidden state of the LAST token contains
        a representation of the entire sentence.

Step 4: Final projection
        Use the last hidden state to predict the next token.
        Most likely next token: "."  or "Can"  or "Could"
```

OK that's the high-level. Now let's open up ONE layer.

## 6.2 Inside one transformer layer

A single Qwen3 transformer layer has this structure:

```
                    Input: 7 vectors of dim 2560
                    [v_My, v_order, v_is, v_delayed, v_by, v_3, v_days]
                              │
                              ▼
                ┌──────────────────────────┐
                │    LayerNorm (RMSNorm)    │
                │    Normalizes each vector │
                └──────────────────────────┘
                              │
                              ▼
            ┌─────────────────────────────────────┐
            │   SELF-ATTENTION BLOCK              │
            │                                     │
            │  Each token "looks at" all others   │
            │  and decides how much to attend     │
            │  to each. Then gathers info.        │
            │                                     │
            │  Uses 4 weight matrices:            │
            │   - W_Q (query)                     │
            │   - W_K (key)                       │
            │   - W_V (value)                     │
            │   - W_O (output)                    │
            └─────────────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │ Residual: add input back  │
                │  output = output + input  │  ← skip connection
                └──────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │    LayerNorm (RMSNorm)    │
                └──────────────────────────┘
                              │
                              ▼
            ┌─────────────────────────────────────┐
            │   MLP (FEED-FORWARD) BLOCK          │
            │                                     │
            │  Each token gets independently      │
            │  transformed in higher-dim space    │
            │  to "think more deeply" about it.   │
            │                                     │
            │  Uses 3 weight matrices:            │
            │   - W_gate                          │
            │   - W_up                            │
            │   - W_down                          │
            └─────────────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │ Residual: add input back  │
                └──────────────────────────┘
                              │
                              ▼
                Output: 7 updated vectors of dim 2560
```

So each layer has TWO sub-blocks:
1. **Self-attention**: tokens communicate with each other
2. **MLP** (also called Feed-Forward, or FFN): each token is independently transformed

Each sub-block has its own weight matrices. **These are the matrices LoRA adapts.**

The 4 attention matrices + 3 MLP matrices = 7 matrices per layer. Across 36 layers = 252 matrices. LoRA adds an adapter to each.

## 6.3 Self-attention explained mechanically

This is the most important part. Let me walk through what self-attention DOES with a tiny example.

Imagine our tokens after embedding are 4-dimensional (instead of 2560-dim) for clarity:

```
v_My      = [0.2, 0.1, 0.0, 0.3]
v_order   = [0.1, 0.5, 0.4, 0.2]
v_delayed = [0.0, 0.3, 0.6, 0.1]
```

(Pretend this is the whole sentence — just 3 tokens.)

### Step 1: Compute Queries, Keys, Values

The attention block has three weight matrices: W_Q, W_K, W_V. Each is shape `(d, d)` = `(4, 4)` in our toy example, `(2560, 2560)` in Qwen3.

For each token, we compute:

```
query_i  = W_Q · v_i      (this token's "question")
key_i    = W_K · v_i      (this token's "label")
value_i  = W_V · v_i      (this token's "content")
```

Conceptually:
- **Query**: "what am I looking for?"
- **Key**: "what do I have to offer?"
- **Value**: "the actual information I'd contribute"

It's like a search engine. Each token formulates a query about what context it needs. Other tokens publish keys (advertisements). The token finds the best matches and pulls in their values.

### Step 2: Compute attention scores

For each token, we compare its query against all keys:

```
score(i, j) = query_i · key_j  ÷ √d

score("delayed", "My")     = 0.1   (low — "My" doesn't help "delayed")
score("delayed", "order")  = 0.8   (high — "order" tells us what's delayed)
score("delayed", "delayed")= 0.5   (medium — self-attention)
```

The score says "how relevant is token j to token i?"

### Step 3: Softmax the scores

Convert scores to probabilities (so they sum to 1.0):

```
weights_for_delayed = softmax([0.1, 0.8, 0.5])
                    = [0.18, 0.55, 0.27]

So when computing the new "delayed" representation:
- 18% from "My"
- 55% from "order"
- 27% from itself
```

### Step 4: Weighted sum of values

```
new_v_delayed = 0.18 · value_My  +  0.55 · value_order  +  0.27 · value_delayed
```

This new vector for "delayed" is a blend of the most relevant tokens. Now "delayed" has absorbed information about "order" — it KNOWS what's delayed.

### Step 5: Output projection W_O

The blended vector goes through one more matrix multiplication: `output = W_O · blended_value`. This re-projects it back into the residual stream so it can be added to the input.

```
final_v_delayed = W_O · new_v_delayed
```

So the 4 attention matrices each play a specific role:
- **W_Q**: shapes how each token formulates its question
- **W_K**: shapes how each token advertises itself
- **W_V**: shapes the actual content each token contributes
- **W_O**: shapes how the blended result gets re-injected into the stream

## 6.4 What LoRA does to each attention matrix

Now imagine you fine-tune for Kiki's customer support task. Each of the 4 attention matrices needs to be modified slightly to adapt the model's behavior.

### LoRA on W_Q ("change what tokens ask about")

Adding `A_Q · B_Q` to W_Q means the queries get adjusted. The model can learn:

> "When I see `<think>` tokens, formulate queries that look for category-related signals in the ticket. When I see tool result tokens, formulate queries that look for policy-citation phrases."

So the adapter on W_Q changes what each token in the sequence is "asking about" depending on its position and content.

### LoRA on W_K ("change what tokens advertise")

Adding `A_K · B_K` to W_K means the keys (ads) get adjusted. The model can learn:

> "When the token is from the customer's message text, advertise it as relevant to triage. When the token is from a retrieved KB chunk, advertise it as relevant to grounding."

### LoRA on W_V ("change what content gets shared")

Adding `A_V · B_V` to W_V means the values (actual content blended) get adjusted. The model can learn:

> "When a KB chunk is retrieved, contribute its policy details verbatim. When a customer's order number appears, contribute it as a structured token."

### LoRA on W_O ("change how blended info gets used")

Adding `A_O · B_O` to W_O means the output projection gets adjusted. The model can learn:

> "After mixing in retrieved context, project the result in a way that the next layer can interpret as 'ready for JSON formatting'."

### Why all 4 matter for Kiki

If you only LoRA-tune W_Q and W_V (a common subset), you change what to look for and what to share, but not what's advertised (W_K) or how to use the result (W_O). For a task as complex as Kiki — multilingual, structured output, tool calling, multi-turn — you need all four to coordinate.

## 6.5 The MLP (Feed-Forward) block explained

After attention, each token's vector goes through the MLP independently. This is where the model does its "deep thinking" per token.

In Qwen3 (using SwiGLU activation), the MLP has 3 matrices:

- **W_gate**: shape `(2560, 10240)` — expands token to 10240-dim "thinking space" (gate path)
- **W_up**: shape `(2560, 10240)` — expands token to 10240-dim "thinking space" (value path)
- **W_down**: shape `(10240, 2560)` — compresses 10240-dim back down to 2560

The math:
```
gate_signal = SiLU(W_gate · token)         shape (10240,)
up_signal   = W_up · token                 shape (10240,)
combined    = gate_signal ⊙ up_signal      elementwise multiply, shape (10240,)
output      = W_down · combined            shape (2560,)
```

Conceptually:
- **W_up** says "expand this token to 10240 dimensions where each dimension represents some specific concept the model knows about"
- **W_gate** says "for each of those 10240 concepts, decide whether to ACTIVATE it for this specific token"
- The element-wise multiply means: only the gated-on concepts contribute
- **W_down** says "compress the 10240-dim activated representation back to 2560 dimensions"

The 10240-dim space is like a vast "concept dictionary" the model has built up during pretraining. Each of the 10240 dimensions represents some specific concept ("delivery", "refund", "polite", "French", "policy citation", etc.). For each token, gate decides which concepts apply.

This is the part of the model that holds most of its **knowledge**. It's where facts and patterns are stored. Attention is about routing information; MLP is about processing it.

## 6.6 What LoRA does to each MLP matrix

### LoRA on W_up ("teach new concepts")

Adding `A_up · B_up` to W_up adjusts how tokens get expanded into the concept dictionary. The adapter can learn:

> "When the token represents 'refund' (in any language), expand it to activate the 'refund-policy-needed' concept dimension. When it represents 'delivery', activate 'delivery-policy-needed'."

This is how the model learns to recognize new domain-specific concepts.

### LoRA on W_gate ("teach when to activate concepts")

Adding `A_gate · B_gate` to W_gate adjusts the gating decisions. The adapter can learn:

> "When the surrounding context mentions 'urgent' or 'événement' or 'event', activate the 'high-urgency-context' concept. When the context mentions 'thanks' alone, deactivate most concepts."

This is the model's "context-awareness" for which concepts to use.

### LoRA on W_down ("teach how concepts combine into token meaning")

Adding `A_down · B_down` to W_down adjusts how the 10240-dim activated state gets compressed back. The adapter can learn:

> "When the 'refund-policy-needed' AND 'rag-context-not-yet-retrieved' concepts are both active, project the result as 'I should call rag_search next'."

This is how the model learns ROUTING decisions — what action to take based on the active concepts.

## 6.7 Why "all 7 projections" wins for Kiki

Different fine-tuning tasks need different LoRA target subsets:

| Task type | Best LoRA targets | Why |
|---|---|---|
| Sentiment classification | `q_proj`, `v_proj` only | Just need to focus on emotional words |
| Style transfer (formal→casual) | All attention | Change how tokens are blended |
| Adding domain knowledge | All MLP | Need to teach new concepts and routings |
| **Multi-skill (your case)** | **All 7** | **Need attention + concepts + routing** |

Kiki needs to:
1. Learn new attention patterns (which parts of the ticket matter for triage) → attention LoRA
2. Learn new concepts (Loopper-specific terminology, refund policies, design workflows) → MLP up/gate LoRA
3. Learn new routing decisions (when to call rag_search, which collection, when to emit JSON) → MLP down LoRA

Skipping any of these subsets would handicap one of the skills. So you target all 7.

Concretely: if you skipped the MLP and only adapted attention, the model would learn to FOCUS on the right tokens but wouldn't know what to DO with them. It would still produce generic non-Loopper-aware responses.

## 6.8 Why all 36 layers (not just some)

You also LoRA every one of the 36 transformer layers. Why not just the last few?

Different layers learn different things:
- **Early layers** (1-12): syntax, grammar, basic word relationships
- **Middle layers** (12-24): sentence-level meaning, basic reasoning
- **Late layers** (24-36): task-specific patterns, output formatting

For Kiki, you need:
- Early-layer LoRA: handle multilingual input (French/German/Spanish word patterns)
- Middle-layer LoRA: build customer-support-specific representations
- Late-layer LoRA: output the 11-field JSON structure correctly

If you only LoRA-tuned the last 12 layers (a common shortcut), you'd save params but lose the ability to handle multilingual customer text properly, since language-specific features are encoded in early layers.

So you target ALL 36 layers × ALL 7 projections = 252 LoRA adapters. Total param budget at rank 32: ~33-66M params (depending on GQA config). At rank 64: ~66-132M.

---

<a id="part-7"></a>
# Part 7 — How the forward pass works (with LoRA, at inference)

Let me walk you through what happens when you send a ticket to the deployed Kiki SLM. We'll trace through ONE token through ONE layer in detail.

## 7.1 Setup

Suppose the ticket is `"My order is delayed"`. After tokenization, we get:

```
Token IDs: [2387, 1843, 374, 22119]
            ↑    ↑    ↑   ↑
            "My" "order" "is" "delayed"
```

The model has an embedding table that maps each token ID to a 2560-dim vector. After embedding:

```
v_My      = [0.142, -0.087, 0.023, ..., 0.156]  ← 2560 numbers
v_order   = [0.231,  0.044, -0.118, ..., 0.077]
v_is      = [...]
v_delayed = [...]
```

These 4 vectors enter Layer 1.

## 7.2 Forward pass through ONE attention block (with LoRA)

Let's trace what happens for ONE matrix, say W_Q, in Layer 1's attention block.

**Without LoRA (base model only):**
```
For each token's vector x:
  query = W_Q · x          # shape (2560,)
```

Concretely for `v_My`:
```
query_My = W_Q [shape (2560, 2560)] · v_My [shape (2560,)]
         = a new 2560-dim vector
```

That's a single matmul. 2560 × 2560 = 6,553,600 multiplications, then sums.

**With LoRA adapter (rank 64):**
```
For each token's vector x:
  base_part   = W_Q · x                       # shape (2560,)
  lora_part   = (alpha/r) · A_Q · B_Q · x     # shape (2560,)
  query       = base_part + lora_part
```

Notice: `B_Q · x` first, giving a 64-dim vector. Then `A_Q · (B_Q · x)`, giving a 2560-dim vector. Then scaled by `alpha/r = 2`.

The adapter computation:
```
B_Q has shape (64, 2560)
B_Q · x has shape (64,)            ← compress to 64 dims

A_Q has shape (2560, 64)
A_Q · (B_Q · x) has shape (2560,)  ← expand back to 2560
```

So you do two smaller matmuls instead of one big one PLUS the original matmul. The total cost for the LoRA-adapted matrix:

```
Base:   2560 × 2560        = 6,553,600 multiplications
LoRA:   2560 × 64 + 64 × 2560 = 327,680 multiplications
        ─────────────────
Total:  6,881,280 multiplications

Overhead: 327,680 / 6,553,600 = 5%
```

So inference with rank-64 LoRA is ~5% slower per matrix. Across 252 LoRA-adapted matrices, total inference overhead is ~5-10%. Not huge.

## 7.3 The math, step-by-step, for one full layer

Let me show you the COMPLETE forward pass through one layer with LoRA. I'll use simplified shapes (d=2560, r=64, mlp=10240, 4 tokens):

```
INPUT: x (shape: 4 × 2560)  — 4 tokens, each 2560-dim
        │
        ▼
RMSNorm: x_normed = normalize each token  (shape: 4 × 2560)
        │
        ▼
ATTENTION:
  ┌─ Q-projection (with LoRA) ──────────────────────┐
  │ Q = x_normed · W_Q^T                  (4 × 2560) │
  │     + (alpha/r) · x_normed · B_Q^T · A_Q^T       │
  │                                       (4 × 2560) │
  └──────────────────────────────────────────────────┘
  ┌─ K-projection (with LoRA) ──────────────────────┐
  │ K = x_normed · W_K^T + LoRA_K          (4 × 2560)│
  └──────────────────────────────────────────────────┘
  ┌─ V-projection (with LoRA) ──────────────────────┐
  │ V = x_normed · W_V^T + LoRA_V          (4 × 2560)│
  └──────────────────────────────────────────────────┘

  Attention scores: A = softmax(Q · K^T / sqrt(d))   (4 × 4)
  Mixed values:     M = A · V                        (4 × 2560)

  ┌─ O-projection (with LoRA) ──────────────────────┐
  │ attn_out = M · W_O^T + LoRA_O          (4 × 2560)│
  └──────────────────────────────────────────────────┘

x_after_attn = x + attn_out  (shape: 4 × 2560)  — residual connection
        │
        ▼
RMSNorm: x_normed2 = normalize each token  (shape: 4 × 2560)
        │
        ▼
MLP:
  ┌─ Gate-projection (with LoRA) ───────────────────┐
  │ gate = x_normed2 · W_gate^T + LoRA_gate         │
  │                                       (4 × 10240)│
  └──────────────────────────────────────────────────┘
  ┌─ Up-projection (with LoRA) ─────────────────────┐
  │ up = x_normed2 · W_up^T + LoRA_up               │
  │                                       (4 × 10240)│
  └──────────────────────────────────────────────────┘

  combined = SiLU(gate) ⊙ up                      (4 × 10240)

  ┌─ Down-projection (with LoRA) ───────────────────┐
  │ mlp_out = combined · W_down^T + LoRA_down       │
  │                                        (4 × 2560)│
  └──────────────────────────────────────────────────┘

x_after_mlp = x_after_attn + mlp_out  (shape: 4 × 2560)  — residual

OUTPUT: x_after_mlp goes to next layer
```

That's ONE layer. The full Qwen3 has 36 of these stacked. The output of layer 36 goes to the final output projection (lm_head) which produces logits over the vocabulary, which become probabilities for the next token.

## 7.4 What's actually happening to the LoRA weights at inference

Here's a cool detail: at inference, the LoRA adapter weights are READ-ONLY just like the base weights. They got their values during training and now they're fixed. Inference just multiplies and adds.

The adapter weights live in GPU memory alongside the base weights. For rank 64 across 252 matrices, that's about 130 MB of additional memory. Compared to the 8 GB base model in fp16, it's a rounding error.

## 7.5 Merging — what it does and why you do it

You can collapse the LoRA adapter into the base weights to remove the inference overhead. Mathematically:

```
Original (with LoRA):
  query = W_Q · x + (alpha/r) · A_Q · B_Q · x
        = W_Q · x + (A_Q' · B_Q) · x         where A_Q' = (alpha/r) · A_Q
        = (W_Q + A_Q' · B_Q) · x
        = W_Q_merged · x

So define W_Q_merged = W_Q + (alpha/r) · A_Q · B_Q
Now you compute: query = W_Q_merged · x
```

`W_Q_merged` has the same shape as `W_Q` (2560 × 2560). It's a single matrix with the LoRA changes baked in. Inference becomes a regular matmul, no adapter overhead.

The merge is a one-time operation — you do it once after training, and the resulting "merged" model is just a regular fine-tuned Qwen3 with no LoRA-specific machinery needed at inference.

```python
# Pseudocode for merging
for each LoRA-adapted matrix W in the model:
    A, B = the LoRA adapter matrices for W
    alpha, r = the LoRA hyperparameters
    W_merged = W + (alpha / r) * A @ B    # @ means matmul
    replace W with W_merged
    discard A and B
```

After merging, you save the model as if it were a standard fp16 Qwen3 fine-tune. That's what your `model.save_pretrained_merged("merged/", ...)` call does.

### Why merge for production?

1. **Inference speed**: ~5-10% faster (no adapter overhead)
2. **Compatibility**: vLLM, TGI, ollama, llama.cpp can serve a merged model without LoRA-specific code
3. **Distribution**: a single safetensors file is easier to share than base + adapter

### Why NOT merge during development?

1. **Memory**: keeping base + adapter separate lets you swap adapters without reloading the base model
2. **Multi-task serving**: vLLM can hot-swap LoRA adapters per request, useful if you have multiple fine-tuned variants
3. **Iterative training**: merging is one-way; you can't "unmerge" if you want to continue training

For Kiki you merge at deploy time. The Modal endpoint serves the merged model.

---

<a id="part-8"></a>
# Part 8 — QLoRA: how 4-bit quantization works

LoRA solves the "trainable parameter count" problem. But during training, you still need to LOAD the base model into GPU memory to do forward passes. Qwen3-4B in fp16 is 8 GB. On a 24 GB L4, that leaves only 16 GB for everything else (activations, gradients for adapters, optimizer states).

QLoRA is a way to compress the base model to 2 GB while keeping it functional. Let me explain how.

## 8.1 What "precision" means for floating-point numbers

Numbers in computers are stored as floats with a specific precision:

```
fp32 (single precision): 32 bits per number
  - Range: ±10^38 (huge)
  - Precision: ~7 significant decimal digits
  - Memory: 4 bytes per number

fp16 (half precision): 16 bits per number
  - Range: ±65,504
  - Precision: ~3 significant decimal digits
  - Memory: 2 bytes per number

bf16 (brain float): 16 bits per number
  - Range: ±10^38 (same as fp32)
  - Precision: ~2 significant decimal digits
  - Memory: 2 bytes per number
```

Neural network weights typically have values between -1 and +1, with most close to zero. Both fp16 and bf16 work fine for this range. fp16 is more precise; bf16 has bigger range for safety.

For 4 billion parameters:
- fp32: 16 GB
- fp16/bf16: 8 GB (this is what your training script uses by default)

We need to go smaller.

## 8.2 What 4-bit quantization is

Instead of storing each weight as a 16-bit float, what if we stored each weight using only 4 bits?

4 bits can represent 2^4 = 16 different values. So we map each fp16 weight to one of 16 "bins."

**Naive 4-bit quantization (don't do this):**

```
For each weight w in matrix W:
  if w < -0.875:  bin = 0
  if w < -0.75:   bin = 1
  if w < -0.625:  bin = 2
  ... etc, 16 evenly-spaced bins from -1 to +1

Store: 4-bit bin index (0-15)
Recover at inference: bin number → midpoint of the bin range
```

The problem: weights aren't uniformly distributed between -1 and +1. They cluster near zero. With evenly-spaced bins, most weights map to the central few bins, wasting precision on rare extreme weights.

## 8.3 NF4 (NormalFloat 4-bit) — the QLoRA scheme

QLoRA uses a smarter scheme called **NF4** that's specifically designed for normally-distributed data (which neural network weights approximately are).

Key insight: instead of evenly-spaced bins, use bins that are closer together near zero (where most weights live) and farther apart for extreme values.

Here are the actual NF4 quantization levels (approximate):

```
Bin  -8: -1.000
Bin  -7: -0.696  ← bins are closer together
Bin  -6: -0.525  ← than uniform spacing
Bin  -5: -0.395
Bin  -4: -0.284
Bin  -3: -0.184
Bin  -2: -0.091
Bin  -1: 0.000
Bin   0: 0.080
Bin   1: 0.161
Bin   2: 0.246
Bin   3: 0.337
Bin   4: 0.441
Bin   5: 0.563
Bin   6: 0.723
Bin   7: 1.000
```

These 16 levels are chosen so that the quantization error is minimized for normally-distributed weights. Most weights fall close to one of these levels.

## 8.4 The actual storage scheme

For each weight matrix in the model:

1. Find the maximum absolute value: `scale = max(|W|)`
2. Normalize: `W_normalized = W / scale`  (now in [-1, 1] range)
3. For each weight, find the closest NF4 level → store as 4-bit bin index
4. Save the scale separately as a fp16 number

Per-weight storage:
- Bin index: 4 bits = 0.5 bytes
- Plus a small per-block overhead for the scale: ~0.06 extra bytes per weight

Total: ~0.56 bytes per weight (compared to 2 bytes for fp16).

For Qwen3-4B's ~4 billion parameters:
- fp16: 4B × 2 bytes = 8 GB
- NF4: 4B × 0.56 bytes ≈ 2.2 GB

**4× memory reduction** with minimal accuracy loss.

## 8.5 What happens during a forward pass

When we need to use a quantized weight, we dequantize it on-the-fly back to bf16:

```
At forward pass time, for each weight w in matrix W:
  bin_index = stored 4-bit value
  w_normalized = NF4_LEVELS[bin_index]      # lookup in precomputed table
  w_recovered = w_normalized × scale        # multiply by per-block scale
  w_bf16 = w_recovered (now in bf16)
```

The dequantization happens INSIDE the matmul kernel, not as a separate step. CUDA libraries (bitsandbytes) implement this efficiently. The compute-time overhead is small (~5-10%) but you save 4× memory.

## 8.6 Why QUANTIZE the base but NOT the adapter

This is the part that confuses most people. The base model is in NF4 (4-bit, frozen). The LoRA adapter (A and B matrices) stays in bf16 (16-bit, trainable). Why?

Three reasons:

### Reason 1: Gradients need precision

During training, gradients flow back through the model and update the adapter weights. Gradient values are often very small (1e-5, 1e-7). To accumulate these tiny updates correctly, you need high precision. Quantizing the trainable parameters would corrupt gradient updates and prevent learning.

### Reason 2: The base is frozen anyway

Since the base weights don't update during training, their precision only matters for the FORWARD pass. NF4 is "good enough" for forward passes because the precision loss (~1%) is negligible for inference.

### Reason 3: Adapter is small

The LoRA adapter is only ~66M parameters (rank 64). At bf16, that's 132 MB. Quantizing it to 4-bit would save just ~100 MB — not worth the gradient precision loss.

## 8.7 The complete memory budget for QLoRA training

Putting it all together for training Kiki at rank 64:

```
Component                           Size       Why
─────────────────────────────────────────────────────────────────
Base model (Qwen3-4B in NF4):      ~2.2 GB    4-bit quantized, frozen
LoRA adapters A, B (bf16):         ~132 MB    rank 64, trainable
Adapter gradients (bf16):          ~132 MB    same shape as adapters
Adam optimizer state m (bf16):     ~132 MB    momentum tensor
Adam optimizer state v (bf16):     ~132 MB    variance tensor
Activations cached for backprop:   ~3-6 GB    depends on batch + seq len
Gradient checkpointing buffers:    ~1-2 GB    if enabled
─────────────────────────────────────────────────────────────────
Total during training:             ~8-12 GB
```

Fits comfortably on a 24 GB L4. On a free T4 (16 GB), you'd reduce batch size and use gradient checkpointing to fit.

For comparison, full fine-tuning would need:
```
Base model fp16:                    8 GB
Gradients fp16:                     8 GB
Optimizer state (Adam, fp32):      32 GB
Activations:                        4-8 GB
─────────────────────────────────────
Total:                             52-56 GB

Doesn't fit on anything smaller than an A100 80GB. Not feasible for most people.
```

## 8.8 Inference: do you still need NF4?

For inference (after training), you have a choice:

**Option 1: Keep using QLoRA at inference**
- Base in NF4, adapter in bf16
- Saves memory (good for cheap GPUs)
- Slightly slower (~10% overhead from dequantization)

**Option 2: Merge and use bf16/fp16 fully**
- Base + adapter merged into a single bf16/fp16 model
- Faster inference (no dequantization)
- More memory needed (8 GB instead of 2 GB)

**Option 3: Convert to GGUF Q4_K_M (similar to NF4 but different scheme)**
- Used by llama.cpp / Ollama
- Smaller still (~2.6 GB)
- Slowest (CPU inference)
- Best for local Mac dev

For your Kiki deployment:
- **Modal vLLM**: Option 2 (merged bf16, ~8 GB on A10G) — fastest
- **Local Ollama dev**: Option 3 (GGUF Q4_K_M, ~2.6 GB) — works on Mac
- **Edge deployment** (if you ever need it): Option 3 GGUF — smallest

You picked Option 2 for prod. Good choice — speed matters more than memory on the A10G.

---

<a id="part-9"></a>
# Part 9 — Why this all matters for your v2 retrain

Now I can connect everything. Let me summarize what's changing and why each change is justified by the architecture we just walked through.

## 9.1 The v1 → v2 hyperparameter changes, explained

```
                    v1          v2          Why
─────────────────────────────────────────────────────────────────────
LoRA rank           32          64          More slots for rare classes (Part 5)
LoRA alpha          64          128         Keep alpha/r=2 (consistent magnitude)
LoRA dropout        0           0.05        Mild reg vs overfitting on oversampled rare data
Target modules      All 7       All 7       Same — multi-skill task needs everything (Part 6)
Layers              All 36      All 36      Same — every depth matters (Part 6)
Base quantization   NF4         NF4         Same — fits on L4 (Part 8)
Learning rate       2e-4        1.5e-4      Slightly lower — more data, less per-example signal needed
Epochs              3           3           Same — 3 epochs on 8K = enough exposures
Effective batch     ~16-32      ~16-32      Same — tuned for L4
```

## 9.2 Why rank 64 specifically — the math

Going back to the slot allocation argument:

**Patterns Kiki needs to learn (rough estimate):**
- 11 intent categories × 2-3 dedicated slots each = 22-33 slots
- 6 languages × 1-2 slots each = 6-12 slots
- Tool-calling decision logic = 3-5 slots
- JSON output structure = 3-5 slots
- Tone/empathy patterns × 6 languages = 6-12 slots
- Urgency including critical = 2-4 slots
- Rejection patterns = 3-5 slots
- **Total: ~45-76 slots**

At rank 32, you're way under capacity. At rank 64, you're at or just above capacity. At rank 128, you're paying for empty slots.

## 9.3 Why dropout 0.05

LoRA dropout randomly zeroes out parts of the adapter during training. With v1's tight rank, dropout would risk losing important signal. With v2's higher rank and oversampled rare classes, dropout helps prevent the adapter from memorizing the duplicated rare-class examples.

If you oversample refund_request from 30 → 300 by duplicating each example 10 times, the adapter would naturally try to memorize those 30 specific tickets. Dropout 0.05 forces the adapter to find more general patterns instead of memorizing.

## 9.4 Why learning rate goes down slightly

With more data (8-10K vs 4K examples), each step has more diverse signal. A slightly lower learning rate (1.5e-4 vs 2e-4) means smaller weight updates per step, which is appropriate when:
- You have more examples per epoch
- You want to avoid overshooting on rare-class signals
- You want smoother convergence

It's a small tweak — not critical, but matches the new data shape.

## 9.5 What stays the same

- **Base model**: Qwen3-4B-Thinking-2507 (you wouldn't switch base mid-iteration)
- **Layer count**: all 36 (you can't change this without changing the base)
- **Target modules**: all 7 projections per layer (Kiki needs the multi-skill capacity)
- **Quantization**: NF4 for the base, bf16 for adapters (this is the QLoRA standard)
- **Sequence length**: 4096 (matches training context, fits 86% of examples)

These don't change because they're already correct for the task.

## 9.6 What you should expect after v2

Based on what we discussed:

| Metric | v1 | v2 expected |
|---|---|---|
| Intent accuracy on rare classes | ~10-30% | 60-80% |
| Tool-call rate on valid tickets | ~70% | >90% |
| Response includes specific KB facts | ~30-40% | >70% |
| Language-appropriate closings | ~50% | >85% |
| Overall intent accuracy | 51% | 75%+ |

The gap closes because:
1. Rank 64 gives capacity to learn rare classes (Part 5)
2. More balanced training data feeds the gradient signal to those slots
3. Better-quality response examples teach proper warmth/specificity
4. Validation gates catch malformed outputs before they reach training

## 9.7 What WON'T fix the v1 problems

If you only changed ONE thing:

- **Just rank 64, same v1 data**: marginal improvement. The data imbalance still starves rare classes regardless of capacity.
- **Just balanced data, rank 32**: better than v1, but capacity still constrained.
- **Just better responses, rank 32 + v1 distribution**: response quality improves slightly but the model still skips tools and misclassifies rare intents.

You need ALL THREE: more capacity (rank 64), better-distributed data (300+ per category), and higher-quality individual examples (warmth, specifics, proper tool calls).

---

<a id="recap"></a>
# Mental model recap (the things to remember forever)

If you remember these 12 facts you'll never forget LoRA/QLoRA:

1. **Full fine-tuning needs ~40 GB for a 4B model.** LoRA + QLoRA brings it to ~10 GB.

2. **`ΔW` (the change from fine-tuning) has low intrinsic rank.** That's the key insight LoRA exploits.

3. **`A × B` factorizes any rank-`r` matrix update** with `2dr` parameters instead of `d²`.

4. **A LoRA adapter has `r` slots** where `r` is the rank. Each slot stores one (input pattern → output transformation) rule.

5. **Gradient descent allocates slots based on training signal frequency**. Common categories claim more slots. Rare categories starve.

6. **Increasing rank = more slots**, which lets rare classes coexist with dominant ones without being overwritten.

7. **A transformer layer has 7 weight matrices** that get LoRA adapters: 4 attention (Q, K, V, O) + 3 MLP (gate, up, down).

8. **Attention controls "what tokens look at"; MLP controls "what they think about it."** You need both for multi-skill tasks.

9. **You apply LoRA to ALL 36 layers** because different layers learn different things (syntax → meaning → output format).

10. **At inference, LoRA adds one extra small matmul per matrix**. ~5-10% overhead. Merge for production to remove this.

11. **Merging means baking LoRA into the base** as `W_merged = W + (α/r) · A · B`. One-way operation.

12. **QLoRA = LoRA + base model in 4-bit (NF4 scheme).** Base is read-only and quantized; adapters stay bf16 and learn.

13. **NF4 levels are not uniform** — they're closer together near zero where most weights live. This minimizes quantization error for normally-distributed weights.

14. **For v2, rank 64 is the sweet spot**: enough slots for all 11 categories + tool patterns + language patterns + tone patterns, without diminishing returns from going higher.

15. **`alpha = 2 × r`** keeps the LoRA update magnitude consistent across rank changes. Always change them together.

You now understand LoRA mechanically. The math, the intuition, the tradeoffs, the practical implications. When you sit down to retrain v2, every hyperparameter choice will make sense to you.
