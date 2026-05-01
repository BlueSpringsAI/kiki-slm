# Kiki SLM — Complete Architecture & Training Guide

A chapter-by-chapter technical deep-dive into building, training, evaluating, and deploying a fine-tuned Small Language Model that replaces a multi-agent LLM pipeline for B2B customer support.

---

## Chapter 1 — The Goal: Why Fine-tune at All?

### The problem

Loopper runs a customer support agent on LangGraph. Every incoming Freshdesk ticket flows through 3 LLM calls:

1. **Triage + classify** (`o4-mini`) — is this a valid support ticket? What category? Cost: ~$0.005/ticket
2. **RAG retrieval** (`gpt-5-mini` in a ReACT loop) — search the knowledge base 1-3 times. Cost: ~$0.005/ticket
3. **Response generation** (`o4-mini`) — produce a structured JSON with intent, urgency, resolution, team, actions, summary, and customer response. Cost: ~$0.014/ticket

Total: **~$0.028/ticket**. At 200 tickets/day, that's $168/month in OpenAI costs alone, plus latency (3 sequential LLM calls = 5-10 seconds per ticket).

### The hypothesis

A single fine-tuned 4B parameter model can learn to do all three steps in one call:
1. **Reason** about the ticket (what's the intent? how urgent?)
2. **Search** the knowledge base (call `rag_search` 1-3 times)
3. **Generate** the structured 11-field JSON response

If it works, we replace 3 LLM calls with 1 self-hosted model. Cost drops to ~$0.0001/ticket (infra only). Latency drops because there's no API round-trip to OpenAI — inference runs on our own GPU.

### The approach: teacher-student distillation

We don't hand-label 5,000 examples. Instead:
1. Run the **existing LangGraph agent** (the "teacher") on 4,544 real Freshdesk tickets
2. Record every step: the reasoning, the tool calls, the tool results, the final JSON
3. Convert these traces into **ChatML training examples** — the student sees exactly what the teacher did
4. Fine-tune a small model on these examples
5. At inference time, the student replicates the teacher's behavior — reasoning, tool calls, structured output — but in a single model

This is **behavioral cloning** via supervised fine-tuning (SFT). The student model learns the teacher's decision-making process from examples, not from a reward signal.

---

## Chapter 2 — SFT Data Curation

### Where the data comes from

Source: **4,544 real customer tickets** from Freshdesk, filtered for English language. These are Loopper's actual production tickets — orders, design changes, delivery complaints, refund requests, etc.

The sampling is stratified across three dimensions to ensure the model sees the full distribution:

| Dimension | Buckets | Why |
|---|---|---|
| **Depth** | single_turn (35%), customer_reply (5%), with_agent (30%), multi_turn (20%), deep_thread (10%) | Model must handle everything from 1-message tickets to 20-message threads |
| **Sentiment** | negative (20%), neutral (40%), positive (40%) | Real-world skew: most tickets are neutral/positive, but negative ones are harder |
| **Category proxy** | reply (30%), eproof (10%), tracking (10%), attachment (5%), promo (5%), none (40%) | Keyword-based proxy for ticket type before LLM classification |

### The teacher agent trace

For each ticket, the teacher agent (LangGraph with o4-mini + gpt-5-mini) produces a multi-turn conversation:

```
[system]  You are a Loopper support agent...
[user]    Ticket ID: 448226. Message Count: 8...
[assistant, reasoning_content="Let me analyze...", tool_calls=[rag_search(faq, "delivery delay")]]
[tool]    {"results": [{"text": "Delivery policy says...", "collection": "faq"}]}
[assistant, reasoning_content="Context shows...", tool_calls=[rag_search(guidelines, "tone")]]
[tool]    {"results": [{"text": "Use empathetic tone...", "collection": "communication_guidelines"}]}
[assistant, reasoning_content="Final reasoning...", content={"intent":"delivery_issue","urgency":"high",...}]
```

This is the full trace: reasoning, tool call, tool result, more reasoning, another tool call, final structured output. The student model sees ALL of this during training.

### The 11-field output schema

Every training example ends with the assistant producing a single JSON object:

```json
{
  "intent": "delivery_issue",
  "urgency": "high",
  "confidence": 0.92,
  "is_valid": true,
  "rejection_type": null,
  "resolution_type": "requires_human_action",
  "team": "logistics",
  "actions": ["Check with courier"],
  "summary": ["Delivery delayed 3 days"],
  "reasoning": {
    "intent_basis": "Customer reports late delivery",
    "urgency_basis": "3-day delay, express shipment",
    "resolution_basis": "Needs logistics team",
    "policy_used": "Delivery policy from KB"
  },
  "response": "Dear [NAME], I apologize..."
}
```

**11 categories**: new_order_inquiry, design_update, payment_confirmation, delivery_issue, refund_request, order_cancellation, quality_complaint, sample_request, price_negotiation, customer_feedback, other

**4 urgencies**: low, medium, high, critical

**4 resolution types**: direct_resolve, requires_human_action, needs_escalation, needs_more_info

**5 teams**: design, logistics, finance, account_manager, none

**5 rejection types**: spam, misdirected, newsletter, auto_reply, unrelated

### The `reasoning_content` field (critical design decision)

Qwen3-Thinking-2507 has a native `reasoning_content` field that renders as `<think>...</think>` blocks in the chat template. This is **separate** from the `content` field.

We use this separation deliberately:
- `reasoning_content`: the model's chain-of-thought (intent analysis, search planning, context evaluation)
- `content`: the actual output (empty string on tool-call turns, JSON on the final turn)

**Why not put reasoning in `content`?** On small models (4B), mixing reasoning text with structured output in the same field causes format degradation. The model starts leaking reasoning into its JSON output or vice versa. The separate field keeps them architecturally isolated.

Each training example has 3 types of synthesized reasoning:
1. **First turn**: intent analysis + category confidence + search plan
2. **Between tool calls**: prior search evaluation + rationale for next search
3. **Final turn**: retrieved context summary + resolution reasoning + guardrail check

### Tool results trimming

Raw RAG results from the teacher agent are verbose — each result has `text`, `collection`, `score`, `rerank_score`, `chunk_index`, `doc_type`, `source`. Most of these are metadata the student doesn't need.

`scripts/trim_chatml.py` strips 5 fields, keeping only `text` and `collection`:
- Text truncated to 800 chars per result
- Max 5 results per tool call
- **Result**: 28% token reduction, 86% of examples fit in 4096 tokens (up from 37%)

### Final dataset statistics

| Split | Examples | Size |
|---|---|---|
| train_trimmed.jsonl | 4,099 (90%) | 47 MB |
| eval_trimmed.jsonl | 445 (10%) | 4.8 MB |
| gold_100.jsonl | 100 (hand-verified) | covers all 11 intents, 4 urgencies, 4 resolution types, 5 teams |

---

## Chapter 3 — Model Selection: Why Qwen3-4B-Thinking-2507

### Requirements that narrowed the field

1. **Tool calling support** — the model must generate `<tool_call>` blocks that inference servers can parse
2. **Reasoning/thinking mode** — separate `<think>` blocks for chain-of-thought without polluting the output
3. **Small enough for self-hosting** — must run on a single T4/A10G GPU (16-24 GB VRAM)
4. **Good base quality** — 4B parameters is small; the base model needs to be strong at structured JSON output

### Why Qwen3, not Llama/Mistral/Phi

| Model | Size | Tool calling | Thinking mode | JSON quality | Verdict |
|---|---|---|---|---|---|
| Llama 3.2 3B | 3B | via fine-tuning | no native | decent | No thinking mode — reasoning leaks into output |
| Phi-3.5 Mini | 3.8B | via fine-tuning | no native | good | Same problem |
| Mistral 7B | 7B | native | no | good | Too large for T4 in fp16, no thinking |
| **Qwen3-4B-Thinking-2507** | **4B** | **native** | **yes** | **excellent** | **Only model with both at 4B scale** |

### Why `unsloth/` prefix, not `Qwen/`

The `unsloth/` version has a patched chat template that correctly handles `reasoning_content` + `tool_calls` together. The official `Qwen/` version has template bugs where `reasoning_content` gets silently dropped during training.

### Why NOT `-Instruct-2507`

The `-Instruct` variant has no thinking mode. No `reasoning_content` field. Fine-tuning it would require putting reasoning in `content` alongside the JSON output, which causes format degradation on a 4B model.

---

## Chapter 4 — LoRA & QLoRA: What, Why, How

### Full fine-tuning vs LoRA

Full fine-tuning updates all ~4 billion parameters. Memory: ~8 GB weights + ~24 GB optimizer states = **~32 GB minimum**. Doesn't fit on any single consumer GPU.

**LoRA** freezes the base model and adds small trainable matrices. Instead of updating `W` (shape `d × d`), it learns `A` (shape `d × r`) and `B` (shape `r × d`) where `r << d`. The update is `W' = W + A × B`.

For our config:
- Base model: ~4B frozen parameters
- LoRA adapter: ~33M trainable parameters (**0.8% of total**)

### QLoRA = LoRA + 4-bit base

The base model is loaded in 4-bit NF4 quantization (~2.5 GB instead of ~8 GB). Only the LoRA adapters train in fp16.

```
Base model weights:  4-bit NF4  (~2.5 GB)  — frozen
LoRA adapters:       fp16       (~66 MB)   — trainable
Optimizer states:    fp32       (~264 MB)  — for adapters only
Total training:      ~10-15 GB → fits on L4 (24 GB)
```

### Which layers get LoRA

```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # attention (4)
    "gate_proj", "up_proj", "down_proj",       # MLP (3)
]
```

All 7 projection matrices in every transformer layer. Attention-only would save memory but gives worse quality for structured output tasks. We need both attention (learns which parts of the ticket matter) and MLP (learns the JSON schema patterns).

### Rank = 32 and Alpha = 64 — why these numbers

**Rank 32**: each adapter matrix has 32 columns. Lower (8-16) can't capture the full 11-field schema + reasoning + tool calling. Higher (64+) risks overfitting on 4K examples.

**Alpha 64**: the scaling factor. `alpha / rank = 64 / 32 = 2.0`. This controls how much the adapter modifies the base model.
- Ratio 1.0: conservative — preserves base behavior
- **Ratio 2.0: moderate — good for domain adaptation** (our choice)
- Ratio 4.0: aggressive — risk of catastrophic forgetting

Think of it this way: rank determines the adapter's **capacity** (how complex a behavior it can learn), alpha determines its **influence** (how much that learned behavior overrides the base model).

---

## Chapter 5 — Training Hyperparameters Deep Dive

### Learning rate: 2e-4

Standard for LoRA. Higher than full fine-tuning (1e-5 to 5e-5) because only ~33M parameters are being updated — the learning signal needs to be proportionally larger to have effect through the low-rank bottleneck.

### Scheduler: cosine with warmup (warmup_ratio = 0.03)

```
Steps 0-30:     ramp from 0 → 2e-4      (warmup — prevent destabilizing random LoRA init)
Steps 30-1000:  cosine decay 2e-4 → ~0  (smooth convergence)
```

### Epochs: 3

Each example seen ~3 times. For SFT on domain data:
- 1 epoch: underfitting — model hasn't memorized the schema
- **3 epochs: good generalization**
- 5+: overfitting — model memorizes specific ticket→response pairs

### Packing: True

Without packing: 500-token example padded to 4096 = 3,596 wasted tokens per example.
With packing: multiple examples concatenated into one 4096-token sequence.
**Result**: ~3× more examples per GPU step. Training finishes in 1/3 the time.

### train_on_responses_only: True (loss masking)

Only assistant turns contribute to the loss. The model sees system prompts, user messages, and tool results during the forward pass (learns to condition on them), but gradients only come from predicting the assistant's reasoning, tool calls, and JSON output.

**Without this**: model wastes capacity learning to predict ticket text and RAG results it will never generate. **With this**: every gradient step teaches "given this input, what should I output?"

### Optimizer: AdamW 8-bit

Standard AdamW with 8-bit quantized optimizer states. Cuts optimizer memory by ~60%. Essential for fitting on 24 GB GPUs.

### Effective batch size: auto-scaled

| GPU | Batch | Grad accum | Effective batch |
|---|---|---|---|
| A100 (80 GB) | 4 | 8 | 32 |
| L4 (24 GB) | 2 | 8 | 16 |
| T4 (16 GB) | 1 | 8 | 8 |

`grad_accum` simulates a larger batch by accumulating gradients before updating. Same gradient quality as a large batch, fraction of the memory.

---

## Chapter 6 — How the Model Learned Reasoning

### The key insight

This is NOT RLHF. The model learns to reason by **imitating** the teacher's reasoning traces. The training data literally shows the model what to think at each step.

### What the model sees during training

```
<|im_start|>assistant
<think>
Let me analyze this ticket.
Primary intent: Customer reports a delivery delay
Key signals: 'delayed by 3 days', 'express shipment'
This fits delivery_issue with high urgency.
I'll need to search:
- Delivery policy (faq)
- Response tone for complaints (communication_guidelines)
</think>

<tool_call>
{"name": "rag_search", "arguments": {"collection": "faq", "query": "delivery delay"}}
</tool_call>
<|im_end|>
```

The model learns the **pattern** from hundreds of these traces:
1. Start with `<think>` when reasoning is needed
2. Identify intent and urgency from key signals
3. Plan which KB collections to search
4. Close `</think>`, emit `<tool_call>`
5. After receiving tool results, reason about them
6. Emit more `<tool_call>` if needed, or output final JSON

### The multi-turn structure teaches the tool loop

A typical 7-message training example:

```
msg[0]: system prompt (with tool definitions)
msg[1]: user ticket text
msg[2]: assistant (reasoning + tool_call #1)       ← learns WHEN to search
msg[3]: tool result #1                              ← learns to READ results
msg[4]: assistant (more reasoning + tool_call #2)   ← learns to search AGAIN
msg[5]: tool result #2
msg[6]: assistant (final reasoning + JSON output)   ← learns WHEN TO STOP
```

Three distinct skills emerge from this structure:
1. **When to search**: first assistant turn almost always calls `rag_search`
2. **When to search again**: if first result isn't sufficient, model has seen examples of follow-up searches
3. **When to stop**: final turn has no `tool_calls` and outputs JSON instead

### Category-specific search strategies (learned, not hardcoded)

The model learns which KB collections to search for each ticket type:

| Category | Typical search pattern |
|---|---|
| delivery_issue | faq (delivery policy) → communication_guidelines (empathetic tone) |
| refund_request | faq (refund policy) → operations (refund process) → communication_guidelines |
| quality_complaint | faq (quality handling) → operations (inspection) → communication_guidelines |
| sample_request | faq (pricing/MOQ) → operations (lead times) |

These patterns emerge from the training data — the model generalizes from seeing hundreds of examples per category.

### SFT vs RLHF: why SFT was the right call for v1

| | SFT (what we did) | RLHF/DPO |
|---|---|---|
| Signal | "Here's what the teacher did" | "This output is better than that one" |
| Data | Teacher traces (single path) | Preference pairs (better vs worse) |
| Cost | ~$0.03/example (teacher inference) | ~$0.10/example (human labeling) |
| Quality ceiling | As good as the teacher | Can exceed the teacher |
| When to use | You have a working teacher + real data | Teacher isn't good enough |

For v1: we have a working teacher, 4,544 real tickets, and need to ship fast. If quality ceiling proves too low, v2 adds a DPO stage where humans compare student vs teacher outputs.

### How to replicate this for a different domain

The pattern generalizes:

1. **Build a teacher** — any multi-step LLM pipeline that solves your task
2. **Run on real data** — 3,000-10,000 examples from production
3. **Record traces** — every LLM call, tool call, tool result, final output
4. **Convert to ChatML** — system + user + assistant(reasoning + tools) + tool_results + assistant(output)
5. **Use `reasoning_content`** — keep reasoning separate from output
6. **Fine-tune with loss masking** — only train on assistant turns
7. **Evaluate on a gold set** — hand-verify 50-100 examples

The critical ingredient: **teacher quality > data quantity > hyperparameters**. If the teacher makes mistakes, the student inherits them.

---

## Chapter 7 — Evaluation

### Why a gold dataset

The 4,099 training examples are teacher-generated — they reflect the teacher's biases. The `gold_100.jsonl` is **human-verified**: 100 tickets where a domain expert confirmed the correct intent, urgency, resolution, team, and expected tool calls.

### The eval challenge: multi-turn inference

The model doesn't output a single JSON. It outputs `<think>` + `<tool_call>`, then **stops and waits** for tool results. A naive eval script gets 0% parse rate.

The eval script (`colab_eval.py`) simulates the tool loop:

```python
for turn in range(4):
    output = generate(messages)
    
    if output has tool_calls:
        messages.append(assistant_with_tool_calls)
        messages.append({"role": "tool", "content": '{"results": []}'})
        continue
    else:
        return parse_json(output)  # final turn
```

Empty tool results are a compromise — the eval doesn't have a real RAG server. The model handles them gracefully (escalates instead of inventing).

### Three-layer parsing stack

**Layer 1 — Server-side** (vLLM `--reasoning-parser qwen3 --tool-call-parser hermes`):
vLLM strips `<think>` into reasoning field, extracts `<tool_call>` into structured `tool_calls`. Works when output is clean.

**Layer 2 — Client-side `_parse_tool_calls_from_content()`**:
If Layer 1 fails (trailing comma, parser confusion), regex-extracts `<tool_call>` blocks from raw content. Strips commas, handles malformed JSON.

**Layer 3 — `parse_model_json()` with balanced-brace extraction**:
For the final JSON. Strips remaining `<think>`/`<tool_call>` blocks. Tries `json.loads()`, falls back to brace-matching (finds `{`, counts depth to find matching `}`).

### v1 results on gold_100

| Metric | Result | What it means |
|---|---|---|
| intent_accuracy | 51% | Correct 11-way classification |
| urgency_accuracy | 59% | SLA prioritization |
| is_valid_accuracy | 88% | Spam filtering |
| tool_name_f1 | 87% | Called the right tools |
| json_parse_rate | 100% | Always produces valid JSON |
| avg_turns | 3.05 | Multi-turn loop converges |

**Strong**: 100% parse rate, 87% tool F1, 88% spam detection.
**Weak**: 51% intent — the 4B model hasn't fully learned category boundaries.
**Production will be better**: eval uses empty fake RAG results; prod has real context.

---

## Chapter 8 — Inference Scaffolding: GGUF, Ollama, vLLM

### From LoRA adapter to deployable model

Training output: LoRA adapter (~66 MB). Two merge paths:

**fp16 safetensors (~8 GB)** — for vLLM, HF TGI, Modal:
```python
model.save_pretrained_merged("merged/", tokenizer, save_method="merged_16bit")
```

**GGUF Q4_K_M (~2.6 GB)** — for Ollama, llama.cpp:
```python
model.save_pretrained_gguf("gguf/", tokenizer, quantization_method="q4_k_m")
```

Quantization reduces quality ~1% but makes the model 3× smaller and runnable on CPU.

### The chat template: why it matters for inference

The chat template converts structured messages into the token sequence the model was trained on. If the serving template doesn't match training, output degrades.

Qwen3-Thinking template handles:
- `<think>...</think>` for reasoning
- `<tool_call>...</tool_call>` for function calls
- `<tool_response>...</tool_response>` for tool results
- `<|im_start|>`/`<|im_end|>` for turn boundaries

When exporting to GGUF, Unsloth embeds this template in the GGUF metadata. Ollama reads it. vLLM reads it from `tokenizer_config.json`.

### Ollama (local dev)

```bash
ollama create kiki-v0 -f Modelfile   # loads GGUF + template
ollama run kiki-v0 "test"             # interactive
# API: http://localhost:11434/v1/chat/completions
```

Good for: developer iteration, quick testing, zero cost.

### vLLM (production)

```bash
python -m vllm.entrypoints.openai.api_server \
    --model bluespringsAI/qiki-v0 \
    --reasoning-parser qwen3 \           # strips <think> before tool parsing
    --enable-auto-tool-choice \
    --tool-call-parser hermes \          # extracts <tool_call> into structured field
    --max-model-len 4096
# API: http://host:8000/v1/chat/completions
```

The `--reasoning-parser qwen3` flag is the key discovery. Without it, the hermes tool parser chokes on the `<think>` block that appears before `<tool_call>`. With it, vLLM strips `<think>` into a separate field first, then hermes sees clean tool call output.

### Modal.com (managed production — current deployment)

```python
# 50 lines of Python — no Docker, no ECS, no IAM
@app.function(gpu="A10G", ...)
@modal.web_server(port=8000)
def serve():
    subprocess.Popen(["python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "bluespringsAI/qiki-v0",
        "--reasoning-parser", "qwen3",
        "--tool-call-parser", "hermes", ...])
```

Scales to zero when idle. First cold start: ~90s (download + compile). Subsequent: ~30s.

### Hosting comparison

| | Ollama | vLLM on ECS | Modal.com |
|---|---|---|---|
| Format | GGUF Q4_K_M | fp16 safetensors | fp16 safetensors |
| GPU | CPU/Metal | Fargate CPU or EC2 | A10G (24 GB) |
| Tool calling | Template-native | Server-side (hermes) | Server-side (hermes) |
| Scale to zero | N/A | No | Yes |
| Cost | $0 | $60-140/mo | ~$1.10/hr (idle = $0) |
| Best for | Dev | Self-hosted prod | Managed prod |

---

## Chapter 9 — Production Integration

### The env-flag architecture

One flag switches the entire inference path:

```python
# graph.py — compile-time branch
if settings.USE_KIKI_SLM:
    # 1 node: kiki_slm_inference (SLM + multi-turn tool loop)
else:
    # 3 nodes: triage_and_classify → retrieve_context → compose_response (OpenAI)
```

Switch: `USE_KIKI_SLM=true` + redeploy. Rollback: flip to `false` (60 seconds).

### The client speaks OpenAI format

`kiki_client.py` hits `/v1/chat/completions` — works against Ollama, vLLM, Modal, or any OpenAI-compatible server. One code path for all backends. The only config that changes is `KIKI_SLM_URL`.

### The multi-turn loop in production

```
invoke_kiki(ticket_text, rag_tools):
  1. POST /v1/chat/completions (system prompt + user ticket + tool schema)
  2. Response has tool_calls? → dispatch REAL rag_search via MCP → append results → loop
  3. Fallback: tool_calls empty but <tool_call> in content? → client-side parsing → loop
  4. No tool calls? → parse final 11-field JSON → return to agent
```

The critical difference from eval: production dispatches **real** `rag_search` calls against the Loopper Qdrant knowledge base. The model gets actual context, which significantly improves output quality.

---

## Chapter 10 — Known Issues & v2 Roadmap

### Known issues

1. **Trailing comma in `<tool_call>` JSON** — model emits `},` instead of `}`. Breaks vLLM hermes parser. Fix: validate in `build_chatml.py` before retraining.

2. **Invalid RAG collection names** — model calls `customer_policy_faq` instead of `faq`. Fix: validate enum in training data.

3. **English-only** — translation still uses OpenAI gpt-5-mini ($0.0045/ticket). Fix: multilingual training data in v2.

4. **51% intent accuracy** — below teacher's ~80%. Fix: more data, higher rank, or DPO stage.

### v2 roadmap

- [ ] Fix trailing comma + collection name issues in `build_chatml.py`
- [ ] Expand to 8-10K training examples
- [ ] Evaluate with real RAG (not empty results) for accurate baseline
- [ ] Consider LoRA rank 64 (more capacity for 11-way classification)
- [ ] Consider DPO stage for intent/resolution accuracy
- [ ] Test multilingual training to eliminate translation nodes
- [ ] Benchmark Qwen3-8B-Thinking if available (more capacity if 4B plateaus)
