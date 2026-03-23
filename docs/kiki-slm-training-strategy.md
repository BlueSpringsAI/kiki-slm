# Kiki SLM — Training Strategy & Dataset Creation Plan

> Internal document for team discussion. Covers the complete strategy for building a customer service SLM that replaces expensive LLM APIs with a specialized, self-improving small language model.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State](#2-current-state)
3. [The Vision](#3-the-vision)
4. [Dataset Strategy](#4-dataset-strategy)
5. [Organic Data Pipeline](#5-organic-data-pipeline)
6. [Training Strategy](#6-training-strategy)
7. [Capability Roadmap](#7-capability-roadmap)
8. [Evaluation Strategy](#8-evaluation-strategy)
9. [Production Architecture](#9-production-architecture)
10. [Timeline & Resources](#10-timeline--resources)

---

## 1. Executive Summary

### What we're building

A fine-tuned 4B-parameter language model (Qwen3-4B) that replaces GPT-4o/Claude for customer service automation. The model outputs structured JSON containing intent classification, urgency assessment, workflow planning, tool selection, and customer-facing responses — enabling backend systems to execute ticket resolution with zero human intervention on 80-90% of volume.

### Why this matters

| | GPT-4o (prompted) | Kiki SLM (fine-tuned) |
|:--|:-------------------|:----------------------|
| **Cost per ticket** | ~$0.02 | ~$0.0001 |
| **Monthly cost at 100K/day** | ~$60,000 | ~$300 |
| **Latency** | 800ms-2s | 100-150ms |
| **Data privacy** | Sent to OpenAI | On-premise |
| **Improves from your data** | No | Yes (data flywheel) |
| **Custom workflows in weights** | No (prompt only) | Yes (trained) |

### Where we are today

- **Intent accuracy: 96.4%** on 336 gold test tickets (after gold label correction)
- **Urgency accuracy: 72.6%** (improving with keyword escalation)
- **JSON parse rate: 100%** (model always produces valid structured output)
- **Training pipeline: production-ready** — single YAML config drives everything
- **Model: Qwen3-4B + QLoRA** (r=32, 66M trainable params, 2.57% of total)
- **Training data: 270K examples** from 10 public HuggingFace datasets

### What's next

Incorporate **organic ticket data** (real Freshdesk tickets) into training to teach the model your specific business workflows, customer language patterns, and resolution procedures. This is the moat — no competitor can replicate your production data.

---

## 2. Current State

### Training data composition (Phase 0 — public datasets)

| Dataset | Examples | Weight | What it teaches |
|:--------|:---------|:-------|:----------------|
| bitext_cs | 24K | 20% | General CS intent + real responses |
| bitext_ecom | 40K | 20% | E-commerce customer service |
| customer_support_tickets | 21K | 15% | Helpdesk workflows, technical support |
| arcee_agent | 77K | 15% | Tool calling, multi-step reasoning |
| xlam_60k | 42K | 8% | Function calling with parameters |
| bitext_banking | 23K | 5% | Banking-specific intents |
| bitext_insurance | 27K | 5% | Insurance-specific intents |
| banking77 | 9K | 5% | Fine-grained intent classification (77 classes) |
| hermes_fc | 1.7K | 5% | Multi-turn tool calling |
| clinc_oos | 4K | 2% | Out-of-scope detection |
| **Total** | **~270K** | **100%** | |

### What the model currently does well

- **Intent classification**: 96.4% accuracy across 13 intent categories
- **JSON structure**: 100% valid structured output (never generates free text)
- **Response generation**: Produces professional customer-facing responses
- **Tool suggestion**: Selects from 12-tool registry (static mapping, needs enrichment)

### What the model currently does poorly

- **Urgency**: 72.6% — defaults to "medium" too often (training data imbalance)
- **Workflow steps**: Static per-intent templates, not message-specific
- **Tool parameters**: Suggests tool names but doesn't fill correct parameters
- **Multi-turn**: Not trained on conversation threads (only single-turn)
- **Your business specifics**: Doesn't know your products, policies, or workflows
- **Fraud detection**: Confuses fraud with billing disputes (5 of 12 remaining errors)

---

## 3. The Vision

### The end state

An SLM that can handle **any customer service ticket** by:

1. **Classifying** the customer's intent and urgency from any message format (email, chat, phone transcript, system event)
2. **Planning** the specific workflow steps needed for THIS ticket (not generic templates)
3. **Calling** the right tools with correct parameters based on the situation
4. **Generating** a professional, empathetic response in the customer's language
5. **Handling** multi-turn conversations with context awareness
6. **Accepting** runtime overrides via custom system prompts (different clients, different workflows, different tools)
7. **Learning** continuously from production corrections

### What makes this unique

```
GENERIC CS CHATBOT                      KIKI SLM
─────────────────                       ────────
Generates text responses                Generates executable JSON
Requires human to act on output         Backend executes autonomously
Fixed behavior per prompt               Configurable via system prompt overrides
Doesn't learn from corrections          Data flywheel improves weekly
Same model for every client             Custom adapter per vertical/client (future)
```

### The competitive moat

1. **Your organic ticket data** — real resolution patterns from your business
2. **The data flywheel** — every ticket processed improves the model
3. **Embedded workflows** — your business logic lives in model weights, not brittle prompts
4. **Multi-adapter architecture** — specialized adapters per task (intent, workflow, tools, response)
5. **Custom override capability** — same model adapts to different clients via system prompt

---

## 4. Dataset Strategy

### The three data sources

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING DATA MIX                             │
├────────────────────┬──────────────────┬─────────────────────────┤
│   PUBLIC DATASETS  │  ORGANIC TICKETS │  SYNTHETIC (GPT-4o)     │
│   (breadth)        │  (depth)         │  (edge cases)           │
│                    │                  │                         │
│   270K examples    │  5K-50K examples │  5K-10K examples        │
│   10 HF datasets   │  Your Freshdesk  │  Generated for gaps     │
│                    │                  │                         │
│   Teaches:         │  Teaches:        │  Teaches:               │
│   - 13 intents     │  - YOUR workflows│  - Urgency boundaries   │
│   - Tool calling   │  - YOUR language │  - Fraud vs billing     │
│   - JSON format    │  - YOUR policies │  - Multi-intent tickets │
│   - General CS     │  - Multi-turn    │  - Ambiguous cases      │
│                    │  - Resolution    │  - Custom overrides     │
│                    │    patterns      │                         │
│   Weight: 60-70%   │  Weight: 20-30%  │  Weight: 5-10%          │
└────────────────────┴──────────────────┴─────────────────────────┘
```

### Phase 1 data: Public datasets (DONE)

Already processed. 270K examples covering 13 intents, 12 tools, 4 urgency levels. Provides the foundation — the model knows what customer service IS.

### Phase 2 data: Organic tickets (NEXT)

Your Freshdesk tickets contain:
- **Real customer language** — how YOUR customers actually write
- **Real agent responses** — how YOUR agents actually resolve issues
- **Multi-turn threads** — the back-and-forth resolution flow
- **Real urgency signals** — from actual priority assignments
- **Real resolution patterns** — what worked (closed/resolved tickets)

#### Freshdesk data structure

```json
{
  "ticket": {
    "subject": "Angebot 5355779",
    "description_text": "ich wollt gerade die Bestellung auslösen...",
    "priority": 4,           // 1=low, 2=med, 3=high, 4=urgent
    "status": 5,             // 2=open, 3=pending, 4=resolved, 5=closed
    "source": 1,             // 1=email, 2=portal, 3=phone
    "detected_language": "de",
    "sentiment_score": 8,    // 0-100 (low = negative)
    "tags": ["REPLY"],
    "created_at": "2025-12-02T10:51:25Z"
  },
  "conversations": [
    {
      "incoming": true,       // true = customer, false = agent
      "body_text": "...",
      "created_at": "..."
    },
    {
      "incoming": false,      // agent reply
      "body_text": "Natürlich. Ich habe die Zahlungsmethode umgestellt..."
    }
  ]
}
```

#### What we extract from each ticket

| Freshdesk Field | Training Signal | How We Use It |
|:----------------|:----------------|:--------------|
| `description_text` | Customer's initial message | User message in ChatML |
| `conversations[incoming=true]` | Customer follow-ups | Multi-turn user messages |
| `conversations[incoming=false]` | Agent responses | Ground truth for response quality |
| `priority` (1-4) | Urgency label | Map to critical/high/medium/low |
| `status` (resolved/closed) | Resolution signal | Only train on successfully resolved tickets |
| `tags` | Category hints | Map to intent taxonomy |
| `sentiment_score` | Customer satisfaction proxy | Use for DPO (low sentiment = rejected, high = chosen) |
| `detected_language` | Language | Filter or include for multilingual training |
| Conversation thread order | Multi-turn flow | Preserve turn sequence in ChatML |

#### Ticket filtering criteria

Not all tickets are useful for training. Filter:

```
KEEP:
  ✓ Status = resolved (4) or closed (5)     → successful resolutions
  ✓ Has at least 1 agent response            → has a resolution pattern
  ✓ Customer message > 20 characters          → not empty/noise
  ✓ Language = supported (de, en, etc.)       → model can learn from it
  ✓ Not spam (spam=false)                     → real tickets

DISCARD:
  ✗ Automated notifications (0 conversations, template subjects)
  ✗ Spam tickets
  ✗ Unresolved/open tickets (status=2,3)    → no resolution to learn from
  ✗ No customer message (empty description)
  ✗ Duplicate tickets (same description hash)
```

### Phase 3 data: Synthetic (GPT-4o generated)

Fill gaps that neither public nor organic data covers:

| Gap | What to Generate | Count |
|:----|:-----------------|:------|
| Fraud vs billing boundary | "unauthorized charge" vs "wrong charge" examples | 500 |
| Urgency diversity | Same intent at critical/high/medium/low levels | 2,000 |
| Multi-intent tickets | "cancel + refund + complaint" in one message | 500 |
| Custom override training | Examples with varied system prompts/tools | 1,000 |
| Edge cases | Empty messages, very long messages, multi-language | 500 |
| Event-driven tickets | System notifications, automated alerts | 500 |
| **Total** | | **~5,000** |

Cost: ~$5-10 using GPT-4o-mini for annotation.

---

## 5. Organic Data Pipeline

### Step-by-step process

```
                FRESHDESK EXPORT
                     │
                     ▼
         ┌───────────────────────┐
         │  1. LOAD & FILTER     │
         │  - Remove spam        │
         │  - Remove automated   │
         │  - Keep resolved only │
         │  - Deduplicate        │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  2. ANNOTATE (GPT-4o) │
         │  - Intent (13 labels) │
         │  - Urgency (4 levels) │
         │  - Workflow steps     │
         │  - Tools required     │
         │  - Resolution summary │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  3. CONVERT TO ChatML │
         │  - Single-turn format │
         │  - Multi-turn format  │
         │  - With tool results  │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  4. QUALITY FILTER    │
         │  - Remove low-quality │
         │  - Dedup again        │
         │  - Validate JSON      │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  5. MIX WITH PUBLIC   │
         │  Public: 60-70%       │
         │  Organic: 20-30%      │
         │  Synthetic: 5-10%     │
         └───────────┬───────────┘
                     │
                     ▼
              TRAINING JSONL
```

### Multi-turn conversion example

A Freshdesk ticket with 4 conversations becomes:

```json
{"messages": [
    {"role": "system", "content": "You are Kiki, an AI customer service agent..."},

    {"role": "user", "content": "ich wollt gerade die Bestellung zu o.g. Angebot auslösen. Allerdings sind die Zahlungsmodi nicht wie vereinbart."},

    {"role": "assistant", "content": "{\"intent\": \"payment_issue\", \"urgency\": \"high\", \"workflow_steps\": [\"check_order_details\", \"verify_payment_methods\", \"update_payment_configuration\"], \"tools_required\": [\"order_lookup_api\", \"payment_gateway_api\"], \"reasoning\": \"Customer cannot complete order due to incorrect payment method options.\", \"response\": \"Ich schaue mir das sofort an. Können Sie mir bitte die Angebotsnummer bestätigen?\"}"},

    {"role": "user", "content": "Angebot 5355779, die Zahlungsmethode sollte auf Rechnung stehen."},

    {"role": "assistant", "content": "{\"intent\": \"payment_issue\", \"urgency\": \"high\", \"workflow_steps\": [\"update_payment_to_invoice\", \"resubmit_order\", \"send_confirmation\"], \"tools_required\": [\"payment_gateway_api\", \"order_lookup_api\"], \"reasoning\": \"Customer needs payment method changed to invoice for order 5355779.\", \"response\": \"Natürlich. Ich habe die Zahlungsmethode für Sie umgestellt und die Bestellung nun aufgegeben. Sie erhalten in Kürze eine Bestätigung.\"}"}
]}
```

This teaches the model:
- German customer language
- How payment issues are resolved in YOUR system
- Multi-turn context (referencing previous message)
- Specific tool selection for payment changes
- Your company's communication style

### Event-driven ticket conversion

Automated notifications (shipping, system events) become a different training pattern:

```json
{"messages": [
    {"role": "system", "content": "You are Kiki. Analyze this ticket and determine if it requires human attention or can be auto-processed."},

    {"role": "user", "content": "Transport notification: 7 packages from [ORG], reference [ADDRESS], pickup scheduled."},

    {"role": "assistant", "content": "{\"intent\": \"system_event\", \"urgency\": \"low\", \"workflow_steps\": [\"log_shipment_notification\", \"update_order_tracking\", \"auto_close_ticket\"], \"tools_required\": [\"shipment_tracking_api\", \"ticket_update_api\"], \"reasoning\": \"Automated shipping notification. No customer action required. Auto-process and close.\", \"response\": null, \"auto_close\": true}"}
]}
```

---

## 6. Training Strategy

### Training phases

```
PHASE 1 (DONE)                    PHASE 2 (NEXT)                    PHASE 3 (LATER)
───────────────                   ──────────────                    ─────────────────
SFT on public datasets            SFT on public + organic           DPO/GRPO alignment
270K examples                     + enriched workflows              + continuous learning
Single adapter                    + multi-turn data                 + multi-adapter
                                  + synthetic edge cases

Intent: 96%                       Intent: 97-98%                    Intent: 99%+
Urgency: 73%                      Urgency: 85-90%                   Urgency: 95%+
Workflow: static templates        Workflow: message-specific         Workflow: learned from data
Tools: name only                  Tools: name + params              Tools: full execution chain
Single-turn only                  Multi-turn supported              Multi-turn + context carry
English focus                     + German + multilingual           All supported languages
```

### Phase 2: SFT with organic data

**Training data composition:**

| Source | Examples | Weight | Purpose |
|:-------|:---------|:-------|:--------|
| Public datasets (current) | 270K | 60% | Breadth — 13 intents, diverse language |
| Organic Freshdesk tickets | 5K-50K | 25% | Depth — YOUR workflows, YOUR customers |
| Enriched workflows (GPT-4o) | 50K | 10% | Message-specific workflow steps |
| Synthetic edge cases | 5K | 5% | Urgency boundaries, fraud, overrides |
| **Total** | **~350K** | **100%** | |

**Hyperparameters (from configs/colab_config.yaml):**

```yaml
model:
  name: Qwen/Qwen3-4B-Instruct-2507
  max_seq_length: 2048
  load_in_4bit: true

lora:
  r: 32
  alpha: 64
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

training:
  epochs: 3
  learning_rate: 2.0e-4
  lr_scheduler: cosine
  packing: true
  optim: adamw_8bit
```

**Estimated training time:**

| GPU | 350K examples | Cost |
|:----|:-------------|:-----|
| RTX PRO 6000 (95GB) | ~8 hours | ~$10-15 (Colab) |
| A100 80GB | ~6 hours | ~$10 (cloud) |
| H100 80GB | ~4 hours | ~$8 (cloud) |

### Phase 3: DPO alignment

After SFT, use DPO to teach the model preference from real agent corrections:

```
Agent overrides SLM prediction:
  SLM said: intent=billing_inquiry
  Agent corrected to: intent=fraud_report

  → DPO pair:
    chosen:  {"intent": "fraud_report", ...}
    rejected: {"intent": "billing_inquiry", ...}

After 1000+ corrections accumulated → retrain with DPO
Learning rate: 5e-6 (40x smaller than SFT)
```

### Phase 3: GRPO with reward functions

After DPO, use GRPO to optimize against multiple objectives simultaneously:

```
Reward functions (already built in src/kiki/rewards/):
  Policy compliance:   35% weight — no PII, proper escalation, scope limits
  Tool accuracy:       25% weight — correct tool name + valid parameters
  Response quality:    25% weight — professional, empathetic, specific
  Format validity:     15% weight — valid JSON matching SLMOutput schema
```

---

## 7. Capability Roadmap

### Capability 1: Intent Classification (DONE — 96%)

The model classifies customer messages into 13 intent categories:

```
order_status, refund_request, billing_inquiry, technical_support,
complaint, shipping_issue, cancellation, return_request,
account_management, product_inquiry, payment_issue, fraud_report,
general_inquiry
```

**To reach 99%:** Add organic data with your specific intent patterns + DPO on remaining 12 errors.

### Capability 2: Urgency Assessment (IN PROGRESS — 73%)

The model assesses urgency as critical/high/medium/low.

**Current problem:** Training data is 80% "medium" — the model defaults to medium when unsure.

**Fix:**
1. Urgency keyword escalation in converters (DONE)
2. Urgency-diverse synthetic data (2K examples at all 4 levels)
3. Organic ticket priority mapping (Freshdesk priority 1-4 → urgency)
4. Sentiment-based urgency (low sentiment score → higher urgency)

### Capability 3: Workflow Planning (NEEDS WORK — 41%)

The model should output message-specific workflow steps, not static templates.

**Current state:** Static per-intent templates (`billing_inquiry` → always same 4 steps)

**Fix:**
1. GPT-4o workflow enrichment on training data (script ready: `scripts/enrich_workflows.py`)
2. Extract real workflows from organic multi-turn conversations (agent actions = workflow)
3. Train dedicated workflow adapter (r=32) on enriched data

### Capability 4: Tool Calling (NEEDS WORK — 24%)

The model should select the right tools AND fill correct parameters.

**Current state:** Suggests tool names from registry but parameters are generic

**Fix:**
1. Train on xlam/hermes/arcee_agent data (already in mix — 28% weight)
2. Add organic examples with your specific tool schemas and parameter formats
3. Train dedicated tool adapter (r=32) on function calling data

### Capability 5: Multi-turn Conversations (NOT YET)

The model should handle back-and-forth customer↔agent threads.

**How to train:**
1. Extract multi-turn conversations from organic Freshdesk tickets
2. Each conversation turn gets its own assistant response with updated intent/urgency
3. The model sees the full conversation history and adjusts its assessment

**Expected training data:** 2K-10K multi-turn examples from resolved Freshdesk tickets

### Capability 6: Custom Override (NOT YET)

The model should follow runtime instructions from the system prompt.

**How to train:** Add examples with varied system prompts:

```
System prompt A: "Use tools: [carrier_api, warehouse_api]. Policy: no refunds on sale items."
System prompt B: "Use tools: [crm_api, billing_api]. Policy: auto-refund under $50."
System prompt C: "Use tools: [helpdesk_api]. Policy: escalate all fraud to security team."
```

The model learns to READ and FOLLOW the system prompt, not just use its trained defaults. This enables per-client customization without retraining.

**Expected training data:** 1K-2K examples with diverse system prompts

### Capability 7: Event-Driven Processing (NOT YET)

The model should distinguish automated events from human tickets.

**New intent:** `system_event` — for automated notifications, webhooks, system alerts

**Training data:** Organic automated tickets (shipping notifications, system events) labeled as `system_event` with `auto_close: true`

### Capability 8: Multilingual (PARTIAL)

Qwen3 supports 119 languages. The model currently trains mostly on English.

**Your organic data is German** — adding it to training automatically makes the model bilingual.

**Future:** Add French, Spanish, etc. from organic data or translated synthetic examples.

---

## 8. Evaluation Strategy

### Gold test set (current: 336 examples)

Expand to **500-1000** examples covering:

| Dimension | Coverage Target |
|:----------|:---------------|
| Each of 13 intents | 30-50 examples each |
| Each of 4 urgency levels | 100-150 each |
| Multi-turn conversations | 50-100 |
| German language | 50-100 |
| Multi-intent tickets | 30-50 |
| System events | 20-30 |
| Custom override scenarios | 20-30 |

### Metrics tracked

| Metric | Current | Phase 2 Target | Phase 3 Target |
|:-------|:--------|:---------------|:---------------|
| Intent accuracy | 96.4% | 98% | 99% |
| Urgency accuracy | 72.6% | 88% | 95% |
| Workflow accuracy | 40.8% | 75% | 90% |
| Tool selection F1 | 23.6% | 60% | 85% |
| JSON parse rate | 100% | 100% | 100% |
| Multi-turn context | N/A | 80% | 95% |

### Evaluation automation

```bash
# After every training run:
python -u scripts/colab_eval.py --adapter-path {ADAPTER}

# Automated gold data audit:
python scripts/audit_gold.py --gold-file data/gold/gold_100.jsonl --eval-results eval_results.json --apply
```

All results tracked in W&B for experiment comparison across runs.

---

## 9. Production Architecture

### Phase 1: Single adapter (NOW)

```
Customer message → SLM (1 call, 120ms) → JSON → Backend executes → Response
```

Simple, fast, cheap. Handles 80% of tickets autonomously.

### Phase 2: Hybrid SLM + Agent (MONTH 3-4)

```
Customer message → SLM (100ms, classify + confidence score)
                      │
            ┌─────────┴──────────┐
            ▼                    ▼
      conf > 0.85           conf < 0.85
      (80% of tickets)      (20% of tickets)
            │                    │
            ▼                    ▼
      Execute SLM plan     LangGraph Agent
      directly             (multi-step reasoning)
```

### Phase 3: Multi-adapter (MONTH 5-6)

```
Customer message → Intent adapter (50ms) → route by intent
                                              │
                   ┌─────────────────────────┤
                   ▼                          ▼
            Workflow adapter (80ms)     Tool adapter (80ms)
                   │                          │
                   ▼                          ▼
            Backend executes tools      Tool results
                   │                          │
                   ▼                          ▼
            Response adapter (100ms) ← full context
                   │
                   ▼
            Customer response
```

### Serving infrastructure

```yaml
# docker-compose.yaml — already built
services:
  vllm:
    image: vllm/vllm-openai:latest
    command: >
      --model Qwen/Qwen3-4B-Instruct-2507
      --enable-lora --max-loras 4
      --lora-modules
        intent-classifier=./adapters/intent/
        workflow-planner=./adapters/workflow/
        tool-caller=./adapters/tools/
        response-writer=./adapters/response/
    ports: ["8000:8000"]

  prometheus:  # monitoring
  grafana:     # dashboards
```

---

## 10. Timeline & Resources

### Immediate next steps (Week 1)

| Task | Owner | Time | Output |
|:-----|:------|:-----|:-------|
| Export full Freshdesk ticket dataset | Team | 1 day | `raw_tickets/*.json` (1K-50K tickets) |
| Build Freshdesk → ChatML converter | Engineering | 1 day | `processors.py` update |
| GPT-4o batch annotation of organic tickets | Engineering | 1 day | Annotated JSONL |
| Run workflow enrichment on training data | Engineering | Overnight | Enriched JSONL ($5) |
| Regenerate combined training data | Engineering | 1 hour | 350K mixed JSONL |
| Retrain on Colab | Engineering | 4-8 hours | New adapter checkpoint |
| Evaluate on expanded gold set | Engineering | 30 min | Accuracy report |

### Phase 2 milestones (Month 1-2)

| Milestone | Target Metric | Dependency |
|:----------|:-------------|:-----------|
| Organic data integrated | 5K+ organic examples in training | Freshdesk export |
| Multi-turn training | Model handles 2+ turn conversations | Organic conversation data |
| Urgency 88%+ | From 73% to 88% | Urgency-diverse data |
| Workflow 75%+ | From 41% to 75% | GPT-4o enrichment |
| Tool F1 60%+ | From 24% to 60% | Organic tool usage data |

### Phase 3 milestones (Month 3-4)

| Milestone | Target Metric | Dependency |
|:----------|:-------------|:-----------|
| DPO alignment | Intent 99%, Urgency 92% | 1000+ production corrections |
| GRPO optimization | All metrics optimized simultaneously | Reward functions |
| Custom override | Model follows runtime system prompts | Override training data |
| Production deployment | Live traffic on real tickets | All above |

### Compute budget

| Activity | GPU Hours | Estimated Cost |
|:---------|:----------|:---------------|
| SFT training (3 epochs, 350K examples) | 8 hours H100 | $15 |
| DPO training (3 epochs, 5K pairs) | 2 hours H100 | $4 |
| GRPO training (1 epoch, 10K prompts) | 4 hours H100 | $8 |
| Evaluation (per run) | 0.5 hours | $1 |
| GPT-4o annotation (50K tickets) | N/A | $5-10 |
| GPT-4o workflow enrichment (100K) | N/A | $5-10 |
| **Total per training cycle** | | **~$45-50** |

### Data flywheel timeline

```
Month 1:  5K organic tickets → SFT → 97% intent
Month 2:  + 1K corrections → DPO → 98% intent, 88% urgency
Month 3:  + 5K more organic → GRPO → 99% intent, 92% urgency
Month 4:  + production data → continuous retraining → 99%+ all metrics
Month 6+: Self-improving weekly cycle, minimal human oversight
```

---

## Questions for Team Discussion

1. **How many Freshdesk tickets can we export?** Need at least 5K resolved tickets with conversations.
2. **What are the most common ticket categories in our system?** Need to map to the 13 Kiki intents.
3. **Which tools/APIs does our backend have?** Need real tool schemas to replace the generic 12-tool registry.
4. **What languages do our customers use?** German + English confirmed — any others?
5. **Do we have CSAT data linked to tickets?** CSAT scores become DPO signals (4-5 = chosen, 1-2 = rejected).
6. **Who reviews AI suggestions today?** Their corrections become training data for DPO.
7. **What's our target automation rate?** 80%? 90%? This determines how much we invest in the long tail.
8. **Timeline for production deployment?** This determines how aggressive the training schedule needs to be.
