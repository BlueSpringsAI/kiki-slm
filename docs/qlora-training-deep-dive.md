# QLoRA Training Deep Dive — From First Principles

> Everything you need to understand about how your Kiki SLM training actually works, from the forward pass to gradient updates, and why every hyperparameter is set the way it is.

---

## Table of Contents

1. [The Big Picture: What Happens When You Train](#1-the-big-picture)
2. [LoRA: Training 2.5% Instead of 100%](#2-lora)
3. [QLoRA: The 4-bit Trick](#3-qlora)
4. [The Forward Pass: Input to Loss](#4-forward-pass)
5. [The Loss Function: Cross-Entropy](#5-loss-function)
6. [The Backward Pass: Computing Gradients](#6-backward-pass)
7. [Gradient Accumulation: Simulating Large Batches](#7-gradient-accumulation)
8. [The Optimizer: AdamW 8-bit](#8-optimizer)
9. [Learning Rate: Why 2e-4 and the Cosine Schedule](#9-learning-rate)
10. [Batch Size: Why It Matters](#10-batch-size)
11. [Packing: Eliminating Waste](#11-packing)
12. [Gradient Checkpointing: Trading Compute for Memory](#12-gradient-checkpointing)
13. [Precision: bf16 vs fp16](#13-precision)
14. [Putting It All Together: One Training Step](#14-one-training-step)
15. [What to Tune and When](#15-what-to-tune)

---

## 1. The Big Picture

Training is a loop that repeats thousands of times:

```
For each batch of examples:
    1. FORWARD PASS:  Feed input through model → get predicted tokens
    2. COMPUTE LOSS:  Compare predictions to actual answers → get a number (loss)
    3. BACKWARD PASS: Calculate how each weight contributed to the error (gradients)
    4. UPDATE WEIGHTS: Nudge weights in the direction that reduces the error
    5. REPEAT
```

After thousands of iterations, the model's predictions get closer to the training data's answers. That's it. Everything else is optimization of this loop.

```
Step 0:    Model outputs garbage     loss = 2.8
Step 100:  Model outputs rough JSON  loss = 1.9
Step 500:  Model outputs valid JSON  loss = 1.2
Step 1000: Model outputs correct     loss = 0.8
Step 4000: Model outputs accurate    loss = 0.5  ← where you stopped
```

---

## 2. LoRA: Training 2.5% Instead of 100%

### The Problem

Qwen3-4B has **4 billion parameters**. Training all of them requires:
- Storing all 4B parameters in memory (8GB in fp16)
- Storing gradients for all 4B parameters (another 8GB)
- Storing optimizer states for all 4B (another 16GB for Adam)
- **Total: ~32GB minimum** — and that's before data and activations

### The Insight

Research found that when fine-tuning large models, the weight updates have **low rank** — meaning you don't need to update all dimensions of every weight matrix. Most of the update information can be captured by two much smaller matrices.

### How LoRA Works

Every transformer layer has weight matrices like `W_q` (query), `W_k` (key), `W_v` (value), etc. Each is a large matrix, e.g., `[4096 × 4096]` = 16M parameters.

LoRA **freezes** the original weight and adds a small trainable detour:

```
Original (frozen):          LoRA addition (trainable):

x ──→ [W_original] ──→ y   x ──→ [A] ──→ [B] ──→ Δy
       4096 × 4096              4096×32  32×4096
       16M params               131K + 131K = 262K params
       FROZEN                   TRAINABLE

Final output: y + Δy

The "rank" r=32 means A and B are thin: 4096×32 and 32×4096
Instead of 16M trainable params, you have 262K (60x reduction)
```

### Your LoRA Config

```python
r = 32              # Rank — how many dimensions in the detour
lora_alpha = 64     # Scaling factor (alpha/r = 2.0 scaling)
target_modules = [  # Which weight matrices get LoRA adapters
    "q_proj",       # Query projection (attention)
    "k_proj",       # Key projection (attention)
    "v_proj",       # Value projection (attention)
    "o_proj",       # Output projection (attention)
    "gate_proj",    # MLP gate
    "up_proj",      # MLP up projection
    "down_proj",    # MLP down projection
]
# = 7 modules × 36 layers = 252 LoRA pairs
# = 66M trainable parameters out of 4B total (2.57%)
```

### Why r=32?

```
r=8:   Very small adapter. Fast, low memory.
       Good for simple tasks (classification only).
       May underfit complex tasks.

r=16:  Good for intent classification.
       Our intent_classifier config uses this.

r=32:  Our current setting. Good balance for multi-task
       (intent + workflow + tools + response all at once).

r=64:  Large adapter. Best for response generation quality.
       Our response_generator config uses this.
       More parameters = more expressive = more memory.

r=128: Diminishing returns. Almost full fine-tuning cost
       for marginal quality improvement.
```

### Why alpha=64 (2x rank)?

`alpha` scales the LoRA output: `Δy = (alpha/r) * B @ A @ x`

With r=32 and alpha=64: scaling = 64/32 = **2.0**

This means the LoRA update is amplified 2x. Why?
- Too small (alpha=r=32, scaling=1.0): LoRA updates are tiny, learning is slow
- Too large (alpha=128, scaling=4.0): LoRA updates are huge, training is unstable
- **2x is the empirically best default** for QLoRA fine-tuning

---

## 3. QLoRA: The 4-bit Trick

### What 4-bit Quantization Means

Normal fp16 uses 16 bits per parameter. 4-bit uses 4 bits.

```
fp16:  Each weight stored as 16-bit float
       4B params × 2 bytes = 8GB

4-bit: Each weight stored as 4-bit integer (NF4 format)
       4B params × 0.5 bytes = 2GB

Savings: 8GB → 2GB for the base model (4x compression)
```

NF4 (Normal Float 4-bit) is a special quantization format that:
- Maps the 16 possible 4-bit values to match the typical weight distribution
- Groups weights into blocks of 64 and stores a scale factor per block
- Is optimized for the bell-curve-shaped weight distributions in transformers

### The QLoRA Training Flow

```
BASE MODEL (4-bit, FROZEN)              LoRA ADAPTERS (fp16, TRAINABLE)
┌─────────────────────────┐             ┌─────────────────────┐
│ W_q: 4-bit quantized    │             │ A_q: fp16 (4096×32) │
│ W_k: 4-bit quantized    │             │ B_q: fp16 (32×4096) │
│ W_v: 4-bit quantized    │             │ A_k: fp16           │
│ ...                     │             │ B_k: fp16           │
│ 4B params, ~2.5GB       │             │ ...                 │
│ NO gradients computed   │             │ 66M params, ~0.5GB  │
│ NEVER updated           │             │ Gradients computed  │
└─────────────────────────┘             │ Updated every step  │
                                        └─────────────────────┘

During forward pass:
  1. Dequantize W (4-bit → fp16) on-the-fly
  2. Compute y = W @ x  (using dequantized weights)
  3. Compute Δy = B @ A @ x  (using LoRA weights)
  4. Output = y + Δy

During backward pass:
  Gradients flow through Δy path only (A and B get gradients)
  W is frozen — no gradients needed for it
```

### Memory Breakdown (Your Training)

```
Component                    VRAM
───────────────────────────────────
Base model (4-bit)           2.5 GB
LoRA adapters (fp16)         0.5 GB
Optimizer states (8-bit)     0.5 GB    ← AdamW 8-bit (see section 8)
Gradients (fp16)             0.5 GB
Activations (checkpointed)   1.0 GB    ← Gradient checkpointing (see section 12)
Data batch                   0.1 GB
───────────────────────────────────
TOTAL                       ~5.1 GB

Your H100 has 80GB → massive headroom
T4 has 16GB → still fits comfortably
```

---

## 4. The Forward Pass: Input to Loss

### Step by step for ONE training example

Your training example is a ChatML conversation:

```
<|im_start|>system
You are Kiki, an AI customer service agent...
<|im_end|>
<|im_start|>user
I was charged twice for my subscription
<|im_end|>
<|im_start|>assistant
{"intent": "billing_inquiry", "urgency": "high", ...}
<|im_end|>
```

### Step 1: Tokenization

The tokenizer converts text to token IDs:

```
"I was charged twice" → [40, 574, 18923, 11192]

Full example → [151644, 8948, 198, 2610, 527, ...] (maybe 400-800 tokens)
```

### Step 2: Embedding

Each token ID is converted to a 2048-dimensional vector:

```
Token 40 → [0.023, -0.156, 0.891, ..., 0.034]  (2048 numbers)
Token 574 → [0.451, 0.023, -0.234, ..., 0.187]  (2048 numbers)
...

Input shape: [batch_size, seq_length, 2048]
For batch_size=4, seq_length=500: [4, 500, 2048]
```

### Step 3: Through 36 Transformer Layers

Each layer transforms the embeddings:

```
Layer 0:
  Attention:  Tokens look at each other
              Q = (W_q + B_q @ A_q) @ x    ← frozen base + LoRA
              K = (W_k + B_k @ A_k) @ x
              V = (W_v + B_v @ A_v) @ x
              Attention_scores = softmax(Q @ K^T / sqrt(d))
              Attention_output = Attention_scores @ V
              Output = (W_o + B_o @ A_o) @ Attention_output

  MLP:        Non-linear transformation
              gate = (W_gate + B_gate @ A_gate) @ x
              up = (W_up + B_up @ A_up) @ x
              mlp_output = (W_down + B_down @ A_down) @ (gate * SiLU(up))

Layer 1: Same operations, different weights
...
Layer 35: Same operations, different weights

Output shape: [4, 500, 2048]
```

### Step 4: Language Model Head

The final layer converts 2048-dim vectors back to vocabulary probabilities:

```
logits = LM_head @ hidden_states
Shape: [4, 500, 151936]  ← probability for each of 151,936 vocab tokens
                            at each position

For position 100 (predicting next token):
  logits[0, 100] = [0.01, 0.003, ..., 0.85, ..., 0.002]
                                        ↑
                                   token "billing" has highest probability
```

---

## 5. The Loss Function: Cross-Entropy

### What Cross-Entropy Loss Does

At each position in the sequence, the model predicts a probability distribution over all 151,936 tokens. Cross-entropy measures how far this prediction is from the actual next token.

```
Position 100: Model predicts next token probabilities
  "billing":     0.85   ← model's prediction
  "payment":     0.08
  "account":     0.03
  "refund":      0.02
  ...

Actual next token: "billing" (from training data)

Cross-entropy = -log(0.85) = 0.163  ← low loss (good prediction)

If model had predicted:
  "billing": 0.02  ← wrong prediction

Cross-entropy = -log(0.02) = 3.912  ← high loss (bad prediction)
```

### Loss Over Full Sequence

```
Position:  [system][prompt][tokens...]   [user][message][tokens...]   [assistant][response][tokens...]
Loss:      computed on ALL tokens (with packing, no masking)

Total loss = average of cross-entropy at every token position

Your training loss progression:
  Step 1:     loss = 2.00  (model mostly guessing)
  Step 100:   loss = 1.50  (learning the JSON structure)
  Step 500:   loss = 0.90  (getting intents right)
  Step 1000:  loss = 0.65  (refining responses)
  Step 4428:  loss = ~0.50 (converged)
```

### Why Cross-Entropy? Alternatives?

| Loss Function | What It Does | Used For |
|:-------------|:-------------|:---------|
| **Cross-entropy** (what we use) | Penalizes wrong token predictions logarithmically | Standard for language model training. Penalizes confident wrong predictions heavily. |
| Mean Squared Error (MSE) | Penalizes squared difference | Bad for classification/token prediction — treats all errors equally |
| Focal Loss | Down-weights easy examples, focuses on hard ones | Useful if some tokens are much harder than others. Not standard for LLMs. |
| DPO Loss | Maximizes margin between preferred and rejected outputs | Used in Phase 2 alignment — needs preference pairs, not just correct outputs |
| GRPO Loss | Reward-weighted policy gradient | Used in Phase 2 with reward functions — optimizes for specific metrics |

**Cross-entropy is the right choice for SFT.** It's what Qwen3 was pre-trained with, so fine-tuning with the same loss maintains consistency.

---

## 6. The Backward Pass: Computing Gradients

### What Gradients Are

A gradient tells you: "If I increase this weight by a tiny amount, how does the loss change?"

```
Weight w = 0.5, loss = 1.2
Gradient ∂loss/∂w = -0.03

Meaning: increasing w by 0.001 would decrease loss by 0.00003
→ We should increase w (move in direction of -gradient)
```

### Backpropagation Through the Network

Gradients flow backward from the loss, through every layer:

```
Loss (scalar)
  ↓ gradient flows backward
LM Head: ∂loss/∂logits
  ↓
Layer 35: ∂loss/∂hidden_35
  ↓
  Attention: ∂loss/∂A_q, ∂loss/∂B_q, ∂loss/∂A_k, ...  ← LoRA gradients
             ∂loss/∂W_q = NOT COMPUTED (frozen!)         ← saves memory + compute
  ↓
  MLP: ∂loss/∂A_gate, ∂loss/∂B_gate, ...               ← LoRA gradients
       ∂loss/∂W_gate = NOT COMPUTED (frozen!)
  ↓
Layer 34: same
  ↓
... (36 layers)
  ↓
Layer 0: ∂loss/∂A_q_0, ∂loss/∂B_q_0, ...

Result: gradients for all 66M LoRA parameters
        (NOT for the 4B frozen parameters)
```

### Why Gradients Are Only Computed for LoRA

```
Full fine-tuning:
  Compute gradients for 4B parameters
  Store 4B gradients in memory (8GB)
  Update all 4B parameters

QLoRA:
  Compute gradients for 66M LoRA parameters only
  Store 66M gradients in memory (0.13GB)
  Update only 66M parameters

Memory savings: 8GB → 0.13GB (60x reduction)
Compute savings: ~40x fewer gradient computations
```

---

## 7. Gradient Accumulation: Simulating Large Batches

### The Problem

Larger batches = more stable gradients = better training. But larger batches = more VRAM.

```
Batch size 32 on a 4B model:
  32 sequences × 2048 tokens × 2048 dimensions = huge memory for activations
  Might not fit even on H100
```

### The Solution: Accumulate Over Multiple Mini-Batches

Instead of one batch of 32, process 8 mini-batches of 4 and average the gradients:

```
Your config: batch_size=4, gradient_accumulation_steps=8
Effective batch size: 4 × 8 = 32

Mini-batch 1 (4 examples): compute gradients, STORE (don't update yet)
Mini-batch 2 (4 examples): compute gradients, ADD to stored gradients
Mini-batch 3 (4 examples): compute gradients, ADD to stored gradients
Mini-batch 4 (4 examples): compute gradients, ADD to stored gradients
Mini-batch 5 (4 examples): compute gradients, ADD to stored gradients
Mini-batch 6 (4 examples): compute gradients, ADD to stored gradients
Mini-batch 7 (4 examples): compute gradients, ADD to stored gradients
Mini-batch 8 (4 examples): compute gradients, ADD to stored gradients
                                     ↓
                           Average all 8 gradient sets
                                     ↓
                           ONE optimizer step (update weights)
                                     ↓
                           This counts as 1 "step" in training

So 8 forward+backward passes = 1 training step
4,428 steps × 8 accumulations = 35,424 forward passes
35,424 × 4 batch = 141,696 examples processed per epoch
× 3 epochs = ~425K example passes (close to your 200K × 3 = 600K
minus packing compression)
```

### Why Not Just Use batch_size=32?

```
batch_size=4:  Activations for 4 sequences fit in ~1GB
batch_size=32: Activations for 32 sequences need ~8GB
               May not fit on smaller GPUs

Gradient accumulation gets the same math with less memory.
The ONLY cost: 8x more forward passes before each weight update.
But each forward pass is fast, so it's a good tradeoff.
```

---

## 8. The Optimizer: AdamW 8-bit

### What the Optimizer Does

After computing gradients, the optimizer decides HOW MUCH to adjust each weight. It's not just `weight -= learning_rate * gradient`. Adam is smarter:

```
For each weight w:
  m = moving average of gradients (momentum)       "which direction?"
  v = moving average of squared gradients (variance) "how noisy?"

  update = m / (sqrt(v) + epsilon)

  w = w - learning_rate * update

Why this is better than plain gradient descent:
  - m (momentum): smooths out noisy gradients, prevents zigzagging
  - v (variance): adapts learning rate per-weight
    - Weights with consistent gradients → larger updates
    - Weights with noisy gradients → smaller updates (be cautious)
```

### Why AdamW (not Adam)?

AdamW adds **weight decay** — a tiny penalty for large weights:

```
w = w - learning_rate * update - weight_decay * w
                                  ↑
                     "Shrink large weights slightly"
                     Prevents overfitting
                     Your config: weight_decay=0.01
```

### Why 8-bit?

Standard Adam stores `m` and `v` for every parameter in fp32 (4 bytes each):
```
66M LoRA params × 4 bytes × 2 states = 528MB
```

8-bit Adam quantizes `m` and `v` to 8-bit:
```
66M LoRA params × 1 byte × 2 states = 132MB

Savings: 528MB → 132MB (4x less optimizer memory)
Quality loss: negligible (< 0.1% accuracy difference)
```

---

## 9. Learning Rate: Why 2e-4 and the Cosine Schedule

### Why 2e-4 (0.0002)?

The learning rate controls the step size: how much weights change per update.

```
Too high (1e-2):    Weights jump around wildly
                    Loss goes up and down, training diverges
                    "Running too fast, tripping over everything"

Too low (1e-6):     Weights barely change
                    Need 100x more steps to converge
                    "Walking so slowly you never arrive"

Just right (2e-4):  Weights change enough to learn quickly
                    But not so much that training is unstable
                    "Jogging at a comfortable pace"
```

**2e-4 is the empirically validated default for QLoRA SFT.** It comes from the QLoRA paper (Dettmers et al. 2023) and has been confirmed across hundreds of fine-tuning experiments.

**CRITICAL: Different training stages need different learning rates:**

```
SFT (learning new behavior):   2e-4   ← aggressive, model needs to learn
DPO (refining preferences):    5e-6   ← gentle, model already knows the task
GRPO (reward optimization):    1e-6   ← very gentle, small adjustments
KTO (safety alignment):        1e-6   ← conservative, don't break anything
```

Using SFT's learning rate for DPO would **destroy** what the model learned. DPO nudges the model 40x more gently.

### The Cosine Schedule

The learning rate isn't constant — it follows a cosine curve:

```
LR
0.0002 |  *****
       |        ***
       |           ***
       |              **
       |                **
       |                  **
       |                    *
       |                     **
       |                       **
       |                         ***
0      |                            ****
       +──────────────────────────────── Steps
       0      1000     2000    3000   4428

Phase 1 (steps 0-133): WARMUP
  LR ramps from 0 → 0.0002 linearly
  warmup_ratio=0.03 → 3% of 4428 = ~133 steps
  "Ease into it — don't shock the model with large updates immediately"

Phase 2 (steps 133-4428): COSINE DECAY
  LR follows cos(π × progress) from 0.0002 → 0
  "Gradually slow down as you approach convergence"
  Large steps early (when far from optimum), tiny steps late (fine-tuning)
```

### Why Cosine (Not Linear or Constant)?

```
Constant LR:     Same step size throughout
                 Works but overshoots near convergence
                 Final loss: ~0.55

Linear decay:    Steps get smaller linearly
                 OK but not optimal
                 Final loss: ~0.52

Cosine decay:    Steps get smaller following a smooth curve
                 Spends more time in the "sweet spot" of medium LR
                 Final loss: ~0.50

The difference is small but consistent across experiments.
Cosine is the standard for transformer fine-tuning.
```

---

## 10. Batch Size: Why It Matters

### Effective Batch Size = per_device × grad_accum × num_gpus

```
Your config:
  per_device_train_batch_size = 4
  gradient_accumulation_steps = 8
  num_gpus = 1

  Effective batch size = 4 × 8 × 1 = 32
```

### What Batch Size Affects

```
Small batch (4):
  - Each gradient estimate is noisy (based on 4 examples)
  - Training is more random, explores more of the loss landscape
  - Can find sharper minima (sometimes better generalization)
  - More gradient updates per epoch
  - Uses less memory

Large batch (128):
  - Each gradient estimate is stable (averaged over 128 examples)
  - Training is smoother, more predictable
  - Converges faster in wall-clock time (fewer updates needed)
  - But can converge to flatter minima (sometimes worse generalization)
  - Uses more memory

Sweet spot for QLoRA SFT: 16-32 effective batch size
Your setting (32) is in the sweet spot.
```

### Batch Size and Learning Rate Scaling

There's a rule of thumb: **if you double batch size, increase LR by sqrt(2)**:

```
Batch 16, LR 1.4e-4
Batch 32, LR 2.0e-4   ← your config
Batch 64, LR 2.8e-4
```

This compensates for the fact that larger batches make each update more "confident," so you can take bigger steps. Your config follows this convention.

---

## 11. Packing: Eliminating Waste

### The Problem Without Packing

Different examples have different lengths. Without packing, short examples are padded:

```
Example 1: [tokens tokens tokens PAD PAD PAD PAD PAD PAD PAD]  ← 300 tokens + 1748 padding
Example 2: [tokens tokens tokens tokens PAD PAD PAD PAD PAD]   ← 500 tokens + 1548 padding
Example 3: [tokens tokens PAD PAD PAD PAD PAD PAD PAD PAD]     ← 200 tokens + 1848 padding
Example 4: [tokens tokens tokens tokens tokens tokens PAD PAD] ← 1200 tokens + 848 padding

Total: 4 × 2048 = 8192 token positions
Actual content: 2200 tokens
Padding waste: 5992 tokens (73% wasted!)
```

### With Packing (what we use)

Multiple examples are concatenated into one sequence:

```
Packed sequence: [Ex1 tokens | Ex2 tokens | Ex3 tokens | Ex4 tokens | Ex5 tokens | PAD]
                  300          500          200          400          600         48

Total: 2048 token positions
Actual content: 2000 tokens
Padding waste: 48 tokens (2% wasted!)
```

### Impact on Training

```
Without packing:
  200K examples, avg 400 tokens each
  Each sequence is 2048 tokens (73% padding)
  200K sequences / 32 batch = 6,250 steps/epoch

With packing:
  200K examples packed into ~39K sequences (avg 5 examples per sequence)
  Each sequence is 2048 tokens (~2% padding)
  39K sequences / 32 batch = ~1,219 steps/epoch

Packing gives: ~5x fewer steps, ~5x faster training
Same number of examples seen, just packed efficiently.
```

### The Tradeoff

With packing, loss is computed on ALL tokens (system + user + assistant). Without packing, you can use `DataCollatorForCompletionOnlyLM` to mask loss on system/user tokens and only train on assistant tokens.

```
Packing:       Trains on everything. ~5x faster. Standard practice.
No packing:    Can mask non-assistant tokens. 5x slower. Slightly more precise.

For SFT, packing is the right choice. The model learns the format
AND the responses, which is what we want.
```

---

## 12. Gradient Checkpointing: Trading Compute for Memory

### The Problem

During the forward pass, every layer's intermediate results (activations) are stored in memory because the backward pass needs them to compute gradients:

```
Without checkpointing:
  Forward: Store activations at every layer (36 layers)
  Memory: 36 × activation_size ≈ 4-8GB for a 4B model

  Backward: Use stored activations to compute gradients
  Speed: Fast (everything is in memory)
```

### The Solution

Don't store all activations. Recompute them during the backward pass:

```
With gradient checkpointing:
  Forward: Store activations at every 6th layer only (6 checkpoints)
  Memory: 6 × activation_size ≈ 0.7-1.4GB (6x reduction)

  Backward: At each checkpoint, recompute the activations for
            the 5 layers between checkpoints, then compute gradients
  Speed: ~30% slower (recomputation cost)
```

### Unsloth's Optimization

Standard checkpointing: recompute every N layers (30% slower)
Unsloth's `use_gradient_checkpointing="unsloth"`: smarter checkpointing that:
- Checkpoints only the attention layers (most memory-heavy)
- Doesn't checkpoint the MLP layers (cheaper to recompute)
- Results in ~20% less memory than standard, ~10% overhead instead of 30%

```
Your config:
  gradient_checkpointing=True
  gradient_checkpointing_kwargs={"use_reentrant": False}
  + Unsloth's optimized checkpointing

Memory savings: ~2-3GB
Speed cost: ~10-15% slower (worth it for the memory savings)
```

---

## 13. Precision: bf16 vs fp16

### What They Are

```
fp32 (full precision):  32 bits per number. Maximum accuracy.
                        1 sign + 8 exponent + 23 mantissa
                        Range: ±3.4 × 10^38

fp16 (half precision):  16 bits per number. 2x faster, half memory.
                        1 sign + 5 exponent + 10 mantissa
                        Range: ±65,504  ← SMALL range, can overflow!

bf16 (brain float 16):  16 bits per number. 2x faster, half memory.
                        1 sign + 8 exponent + 7 mantissa
                        Range: ±3.4 × 10^38  ← SAME range as fp32!
                        Less precision than fp16, but much safer.
```

### Why bf16 Is Better for Training

```
fp16 risk:
  During training, gradient values can be very large (>65,504)
  fp16 can't represent these → overflow → NaN → training crashes
  Needs "loss scaling" to work around this (hacky)

bf16 safety:
  Same range as fp32 (handles any gradient magnitude)
  Slightly less precise (7-bit mantissa vs 10-bit)
  But the precision loss is negligible for training
  No loss scaling needed

Your config:
  bf16=True on A100/H100 (they support bf16 natively)
  fp16=True as fallback on T4 (T4 doesn't have bf16)
```

---

## 14. Putting It All Together: One Training Step

Here's exactly what happens in one of your 4,428 training steps:

```
STEP 1,000 of 4,428:

1. DATA LOADING
   Dataloader serves a packed batch: [4 packed sequences, each 2048 tokens]
   Total tokens this mini-batch: 4 × 2048 = 8,192

2. FORWARD PASS (this happens 8 times for gradient accumulation)

   Mini-batch 1 of 8:
   ├── Tokenized input → Embeddings [4, 2048, 2048]
   ├── Layer 0: Attention (Q,K,V with LoRA) → MLP (with LoRA) → Residual
   ├── Layer 1: Same
   ├── ...
   ├── Layer 35: Same
   ├── LM Head → Logits [4, 2048, 151936]
   └── Cross-entropy loss = 0.847

3. BACKWARD PASS

   ├── Gradients flow backward through all 36 layers
   ├── Gradient checkpointing: recompute activations where needed
   ├── Compute ∂loss/∂A and ∂loss/∂B for all 252 LoRA pairs
   └── ACCUMULATE gradients (add to running sum, don't update yet)

4. REPEAT steps 2-3 for mini-batches 2 through 8

5. AVERAGE accumulated gradients (divide by 8)

6. OPTIMIZER UPDATE (AdamW 8-bit)
   For each of the 66M LoRA parameters:
   ├── Update momentum: m = 0.9 * m + 0.1 * gradient
   ├── Update variance: v = 0.999 * v + 0.001 * gradient²
   ├── Compute update: Δw = m / (√v + 1e-8)
   ├── Apply weight decay: w = w × (1 - 0.01 × lr)
   └── Update weight: w = w - lr × Δw

   Current lr at step 1000: ~0.000185 (cosine decay from 0.0002)

7. LOGGING
   ├── Loss: 0.847
   ├── Learning rate: 1.85e-4
   ├── Grad norm: 2.3
   └── Logged to W&B

8. Total tokens processed this step: 8 × 4 × 2048 = 65,536 tokens
   Total examples (packed): ~65,536 / 400 avg ≈ 164 examples

   Wall clock time: ~5 seconds on H100
```

---

## 15. What to Tune and When

### Your Accuracy Is Stuck — What to Change?

```
DIAGNOSIS → FIX MAP:

"Loss is still decreasing"
  → Train more epochs (you might be underfitting)
  → Increase to 5 epochs

"Loss plateaued but accuracy is low"
  → Your training data doesn't match your eval data
  → Fix the data, not the hyperparameters
  → THIS IS YOUR SITUATION (81% accuracy, loss ~0.5)

"Loss is very low but eval is bad"
  → Overfitting. Reduce epochs, increase dropout,
    add weight decay, or get more diverse data.

"Training loss is unstable (jumping up and down)"
  → Learning rate too high. Reduce to 1e-4.
  → Or batch size too small. Increase grad_accum.

"Loss explodes to NaN"
  → Learning rate way too high
  → Or bf16/fp16 precision issue
  → Reduce LR to 5e-5, check for data corruption
```

### Hyperparameter Priority (what to change first)

```
HIGHEST IMPACT (fix these first):
  1. Training data quality ← YOUR #1 PRIORITY
     - Wrong intent labels → wrong predictions
     - Template responses → generic outputs
     - Missing intent categories → 0% on those categories

  2. Data mix / proportions
     - Too much CLINC (63% general_inquiry) → model defaults to general
     - Too little customer_support_tickets → no technical_support examples

  3. Number of epochs
     - Too few (1): underfitting
     - Too many (10): overfitting
     - Sweet spot: 2-4 for most SFT tasks

MEDIUM IMPACT:
  4. Learning rate
     - Only if loss curve looks wrong (unstable or not decreasing)
     - Default 2e-4 is almost always right for QLoRA SFT

  5. LoRA rank (r)
     - r=16 if intent accuracy is fine but you want less VRAM
     - r=64 if response quality needs improvement
     - r=32 (current) is good for multi-task

  6. Batch size
     - Only matters if training is unstable
     - 16-32 effective is the sweet spot

LOW IMPACT (don't touch unless everything else is right):
  7. Warmup ratio (0.03 is fine)
  8. Weight decay (0.01 is fine)
  9. Alpha/rank ratio (2x is fine)
  10. Target modules (all attention+MLP is fine)
```

### Quick Reference Card

```
┌────────────────────────┬──────────────┬──────────────────────────────┐
│ Parameter              │ Your Value   │ When to Change               │
├────────────────────────┼──────────────┼──────────────────────────────┤
│ learning_rate          │ 2e-4         │ Only if loss is unstable     │
│ epochs                 │ 3            │ ↑ if underfitting, ↓ if over │
│ batch_size (effective) │ 32           │ ↑ if training is noisy       │
│ lora_r                 │ 32           │ ↑ for better quality (VRAM)  │
│ lora_alpha             │ 64 (2×r)     │ Keep at 2×r always           │
│ warmup_ratio           │ 0.03         │ Almost never                 │
│ weight_decay           │ 0.01         │ ↑ to 0.05 if overfitting     │
│ lr_scheduler           │ cosine       │ Never (cosine is optimal)    │
│ packing                │ True         │ Never (always use packing)   │
│ bf16                   │ True         │ fp16 only if GPU lacks bf16  │
│ optim                  │ adamw_8bit   │ adamw_torch if memory allows │
│ max_seq_length         │ 2048         │ ↑ for longer conversations   │
│ grad_checkpointing     │ True         │ False only if VRAM is huge   │
└────────────────────────┴──────────────┴──────────────────────────────┘
```
