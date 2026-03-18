# Strategy: 81% → 95-99% Accuracy

## Current State

| Metric | Score | Target |
|:-------|:------|:-------|
| Intent Accuracy | 81% | 95-99% |
| Urgency Accuracy | 59% | 90%+ |
| JSON Parse Rate | 100% | 100% |
| Workflow Accuracy | Not measured | 90%+ |

## The 5-Phase Plan

```
Phase 1: Data quality (NOW)          81% → 88%     ~2 days work
Phase 2: Synthetic augmentation      88% → 92%     ~1 day work
Phase 3: DPO on wrong predictions    92% → 95%     ~1 day work
Phase 4: GRPO with reward functions  95% → 97%     ~2 days work
Phase 5: Multi-adapter specialists   97% → 99%     ~1 week work
```

---

## Phase 1: Data Quality (81% → 88%)

**Status: IN PROGRESS** — converter fixes + weight rebalancing + CLINC filtering + urgency escalation already done.

**Remaining items:**
1. Regenerate training data with `python scripts/prepare_sft_chatml.py --total-examples 100000`
2. Retrain on Colab
3. Evaluate

**Expected gain:** +7% from fixing the data distribution that caused 0% on 6 intent categories.

---

## Phase 2: Synthetic Data Augmentation (88% → 92%)

**What:** Use GPT-4o to generate high-quality training examples for the intents and edge cases where the model fails.

**Step 1: Analyze failure cases from 100-gold eval**

From the 19 wrong predictions at 81%, identify patterns:
- Which intents get confused with which?
- What message patterns cause failures?
- Where does urgency go wrong?

**Step 2: Generate synthetic examples with GPT-4o**

Create a script `scripts/generate_synthetic_data.py` that:

```python
# For each underperforming intent, generate 500 diverse examples
SYNTHETIC_TARGETS = {
    "order_status": 500,       # Often confused with billing_inquiry
    "shipping_issue": 500,     # Often confused with account_management
    "complaint": 300,          # Only source is bitext FEEDBACK
    "fraud_report": 300,       # Often confused with account_management
    "return_request": 300,     # Very few training examples
    "refund_request": 300,     # Often confused with billing_inquiry
    "cancellation": 300,       # Often confused with billing_inquiry
    "technical_support": 500,  # Only source is customer_support_tickets
}

# For each, prompt GPT-4o:
prompt = f"""Generate a realistic customer service message where the intent is {intent}.
Include varied language, different urgency levels, and different scenarios.
Output JSON: {{"customer_message": "...", "intent": "{intent}", "urgency": "...",
"workflow_steps": [...], "tools_required": [...], "reasoning": "...", "response": "..."}}"""
```

Cost: ~$10-20 for 3,500 high-quality synthetic examples.

**Step 3: Generate ambiguous/borderline examples**

The hardest cases are intent boundaries:
```
"I want to cancel my order and get a refund"
  → Is this cancellation or refund_request?
  → Generate 200 examples for EACH side of the boundary
  → Teach the model the decision criteria

"My account seems to have unauthorized charges"
  → Is this fraud_report or billing_inquiry?
  → Generate 200 examples with clear fraud signals vs billing errors
```

**Step 4: Generate urgency-varied examples**

For each intent, generate messages at ALL urgency levels:
```
order_status + low:      "Just wondering about my order status"
order_status + medium:   "My order hasn't arrived, can you check?"
order_status + high:     "I've been waiting 2 weeks, I need this urgently!"
order_status + critical: "My business depends on this shipment arriving TODAY"
```

500 examples × 4 urgency levels = 2,000 urgency-diverse examples.

**Expected gain:** +4% from filling gaps in underrepresented intents and teaching intent boundaries.

---

## Phase 3: DPO on Wrong Predictions (92% → 95%)

**What:** Direct Preference Optimization — teach the model "this intent is RIGHT, that intent is WRONG" using preference pairs from evaluation failures.

**How:**

1. Run eval on 500+ gold tickets (expand the gold set)
2. For every wrong prediction, create a preference pair:

```json
{
  "prompt": [
    {"role": "system", "content": "You are Kiki..."},
    {"role": "user", "content": "I want to cancel my order and get a refund"}
  ],
  "chosen": [
    {"role": "assistant", "content": "{\"intent\": \"refund_request\", \"urgency\": \"high\", ...}"}
  ],
  "rejected": [
    {"role": "assistant", "content": "{\"intent\": \"cancellation\", \"urgency\": \"medium\", ...}"}
  ]
}
```

3. Run DPO training with the KikiDPOTrainer (already built):

```bash
python scripts/train_alignment.py --config configs/alignment/dpo.yaml \
    --data-path data/processed/dpo_pairs.jsonl
```

**Key hyperparameters for DPO:**
- Learning rate: **5e-6** (NOT 2e-4 — 40x smaller than SFT)
- Beta: 0.1 (controls preference strength)
- Epochs: 1-3
- Start from SFT checkpoint

**Expected gain:** +3% from directly teaching the model its specific failure modes.

---

## Phase 4: GRPO with Reward Functions (95% → 97%)

**What:** Group Relative Policy Optimization — the model generates multiple responses, a reward function scores them, and the model learns to prefer high-scoring outputs.

**How:**

1. Create a prompt dataset (10K diverse customer messages)
2. For each prompt, the model generates 8 candidate responses
3. Four reward functions score each candidate:

```
Policy compliance (35%):  No PII, no scope violations, proper escalation
Tool accuracy (25%):      Correct tool names, valid parameters
Intent accuracy (25%):    Matches expected intent for the message
Response quality (15%):   Professional, empathetic, specific
```

4. GRPO optimizes the model to generate responses that score higher.

```bash
python scripts/train_alignment.py --config configs/alignment/grpo.yaml
```

**Key hyperparameters:**
- Learning rate: **1e-6** (100x smaller than SFT)
- Num generations: 8 per prompt
- Use vLLM for fast generation
- KL coefficient: 0.001 (stay close to SFT model)

**Expected gain:** +2% from reward-guided optimization on all metrics simultaneously.

---

## Phase 5: Multi-Adapter Specialists (97% → 99%)

**What:** Train 4 separate LoRA adapters, each specialized for one subtask:

| Adapter | Task | LoRA Rank | Trained On |
|:--------|:-----|:----------|:-----------|
| Intent classifier | Classify intent + urgency | r=16 | banking77 + clinc + bitext categories |
| Workflow planner | Plan resolution steps | r=32 | annotated workflows + arcee_agent |
| Tool caller | Select tools + parameters | r=32 | xlam + hermes + toolace |
| Response writer | Write customer response | r=64 | bitext responses + DPO-aligned |

**Why this gets to 99%:**
- Intent classifier with r=16 on intent-only data → 98%+ accuracy on just classification
- Each adapter is optimized for exactly one skill, not compromising across 4 tasks
- Response adapter is DPO-aligned independently from classifier

**How to serve:**
```
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
    --enable-lora --max-loras 4 \
    --lora-modules \
        intent-classifier=./adapters/intent/ \
        workflow-planner=./adapters/workflow/ \
        tool-caller=./adapters/tools/ \
        response-writer=./adapters/response/
```

---

## Training Speed Optimizations for H100/A100

### Current vs Optimized

| Setting | Before (T4/A100 safe) | After (H100 optimized) |
|:--------|:---------------------|:----------------------|
| batch_size | 4 | **16** |
| grad_accum | 8 | **2** |
| effective batch | 32 | 32 (same) |
| forward passes per step | 8 | **2** (4x fewer) |
| Training speed | ~5s/step | **~1.5s/step** |

**Why this is faster:** The H100 has 80GB VRAM. With QLoRA, the model uses ~5GB. There's 75GB free. Increasing batch_size from 4 to 16 uses more VRAM but does 4x more computation per forward pass, and reduces gradient accumulation from 8 to 2 — meaning only 2 forward passes per optimizer step instead of 8.

**Same math, 4x faster wall clock.**

### Additional H100 optimizations

1. **Flash Attention 2** — already enabled if available, 2x faster attention
2. **bf16** — H100 has dedicated bf16 tensor cores, native support
3. **Torch compile** (optional) — `torch.compile(model)` can give 10-20% speedup:
   ```python
   # Add to colab_train.py after model loading
   if hasattr(torch, "compile"):
       model = torch.compile(model)
   ```

### Training time estimates

| Dataset Size | H100 (optimized) | A100 80GB | T4 16GB |
|:-------------|:-----------------|:----------|:--------|
| 50K examples | ~30 min | ~1.5 hours | ~4 hours |
| 100K examples | ~1 hour | ~3 hours | ~8 hours |
| 200K examples | ~2 hours | ~6 hours | ~16 hours |
| 626K (--use-all) | ~6 hours | ~18 hours | ~48 hours |

---

## Eval Speed Optimizations

### Batched inference

| GPU | Batch Size | 100 tickets eval time |
|:----|:-----------|:---------------------|
| T4 | 1 (sequential) | ~8-10 min |
| A100 | 8 | ~2-3 min |
| H100 | 16 | ~1-2 min |

Auto-detected by `colab_eval.py` based on GPU VRAM.

---

## Recommended Execution Order

```
WEEK 1:
  [x] Fix converters (done)
  [x] Rebalance weights (done)
  [x] CLINC filtering (done)
  [x] Urgency escalation (done)
  [ ] Regenerate 100K training data
  [ ] Retrain → expect 88%

WEEK 2:
  [ ] Generate 3,500 synthetic examples with GPT-4o
  [ ] Expand gold set to 500 examples
  [ ] Retrain → expect 92%
  [ ] Run DPO on wrong predictions → expect 95%

WEEK 3:
  [ ] Build GRPO prompt dataset (10K prompts)
  [ ] Run GRPO training → expect 97%
  [ ] Begin multi-adapter training

WEEK 4:
  [ ] Complete multi-adapter specialists
  [ ] Full evaluation on 500+ gold set
  [ ] Target: 97-99% intent, 90%+ urgency, 90%+ workflow
```
