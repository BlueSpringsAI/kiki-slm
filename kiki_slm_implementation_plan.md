# Kiki SLM: Complete Implementation Plan

## For use with Claude Code agent — build everything described below

---

## Table of contents

1. Project overview and goals
2. Phase 1: POC (2-day sprint) — 4 flat scripts
3. Phase 2: Production pipeline — modular architecture
4. Data schemas and contracts
5. Dataset registry and download instructions
6. Training configurations
7. RL alignment pipeline
8. Inference and serving
9. Evaluation framework
10. Continuous learning
11. Dependencies and environment setup

---

## 1. Project overview and goals

### What is Kiki

Kiki is an AI Service Operations Platform that uses a multi-agent architecture to automate customer service. The core reasoning engine is a fine-tuned Small Language Model (SLM) that replaces complex prompt engineering with workflow knowledge embedded directly into model weights.

### What the SLM must do

The SLM powers all reasoning within the Kiki platform. It must:

- Classify customer intent from ticket text (order_status, refund_request, billing_inquiry, complaint, shipping_issue, cancellation, return_request, account_management, product_inquiry, payment_issue, fraud_report, general_inquiry)
- Determine urgency level (critical, high, medium, low)
- Plan multi-step service workflows (e.g., refund: verify_identity → check_order → validate_return_policy → initiate_refund → send_confirmation)
- Invoke enterprise tools with correct parameters as structured JSON (order_lookup_api, shipment_tracking_api, refund_processing_api, payment_gateway_api, etc.)
- Evaluate evidence from documents, images, and backend data
- Make operational decisions (refund, replacement, repair, escalation, rejection)
- Detect missing data or unavailable tools and generate structured error messages
- Generate professional, empathetic customer-facing responses

### Two-phase approach

Phase 1 (POC, 2 days): 4 flat scripts proving SLM viability with 5K labeled tickets on a single A100. Deliverables: intent accuracy benchmarks, end-to-end reasoning demo, SLM vs GPT-4 comparison, cost/latency analysis.

Phase 2 (Production, 6 months): Full modular pipeline with multi-LoRA specialist adapters, RL alignment (SimPO → GRPO → KTO), vLLM serving, continuous learning. Supports 450K+ tickets across e-commerce, insurance, and banking verticals.

### Base model selection

Primary: `Qwen/Qwen3-4B-Instruct-2507` (Apache 2.0, 119 languages, native tool calling, 262K context, top-ranked SLM for fine-tuning)

Fallback: `Qwen/Qwen2.5-7B-Instruct` (most battle-tested, largest adapter community)

Intent classifier specialist: `meta-llama/Llama-3.2-3B-Instruct` (highest fine-tuning gains, good for lightweight routing)

### Compute

- Training: Single A100 80GB (rent from Lambda Labs or RunPod, ~$1.50-2.00/hr)
- Demo inference: Apple Silicon Mac with MLX framework
- Production inference: A100 with vLLM

---

## 2. Phase 1: POC (2-day sprint)

### Directory structure

```
kiki-poc/
├── scripts/
│   ├── 1_annotate.py          # GPT-4o-mini batch annotation
│   ├── 2_train.py             # QLoRA fine-tuning with Unsloth
│   ├── 3_evaluate.py          # Benchmarking + comparison
│   └── 4_demo.py              # Gradio demo on MLX
├── configs/
│   └── poc_config.yaml        # All hyperparameters
├── prompts/
│   └── annotator_system.txt   # Annotation system prompt
├── data/
│   ├── raw/                   # Your 200K raw tickets (CSV/JSON)
│   ├── sampled/               # 5K stratified sample
│   ├── annotated/             # GPT-4o-mini labeled output
│   ├── formatted/             # ChatML JSONL ready for training
│   └── gold/                  # 100 hand-labeled test tickets
├── outputs/
│   ├── models/                # Saved adapters
│   ├── results/               # Benchmark results JSON
│   └── exports/               # MLX exported model
├── requirements.txt
└── README.md
```

### Script 1: `scripts/1_annotate.py`

Purpose: Sample 5K tickets from the 200K raw dataset, send to GPT-4o-mini for structured annotation, output labeled JSONL.

Implementation requirements:

1. Load raw tickets from CSV/JSON files in `data/raw/`. Support both formats. Expect at minimum two columns: `customer_message` (or `body`, `text`, `message`) and `agent_response` (or `response`, `answer`, `reply`). Auto-detect column names with fallbacks.

2. Stratified sampling: Sample 5,000 tickets. If tickets have any existing category/type labels, use those for stratification. Otherwise, random sample. Save sample to `data/sampled/sample_5k.jsonl`.

3. Build the annotation prompt. The system prompt (stored in `prompts/annotator_system.txt`) must instruct GPT-4o-mini to output structured JSON matching this exact Pydantic schema:

```python
from pydantic import BaseModel
from typing import Literal

class TicketAnnotation(BaseModel):
    intent: Literal[
        "order_status", "refund_request", "billing_inquiry",
        "technical_support", "complaint", "shipping_issue",
        "cancellation", "return_request", "account_management",
        "product_inquiry", "payment_issue", "fraud_report",
        "general_inquiry"
    ]
    urgency: Literal["critical", "high", "medium", "low"]
    workflow_steps: list[str]
    tools_required: list[str]
    key_entities: dict  # order_id, amount, product, date, etc.
    confidence: float   # 0.0 to 1.0
```

4. Call OpenAI API with structured outputs (response_format). Use `gpt-4o-mini` model. Process in batches of 50 with asyncio + semaphore (max 20 concurrent). Include retry logic with exponential backoff. Log progress every 100 tickets.

5. For each ticket, send:
   - System: the annotation system prompt
   - User: `"Customer message: {customer_message}\nAgent response: {agent_response}"`

6. Save annotated output to `data/annotated/annotated_5k.jsonl`. Each line contains the original ticket fields plus the annotation fields.

7. Print summary statistics: intent distribution, urgency distribution, average confidence, processing time, estimated cost.

### The annotation system prompt (`prompts/annotator_system.txt`)

```
You are an expert customer service analyst. Analyze the following customer support ticket and provide a structured annotation.

You must classify:
1. Intent: The primary purpose of the customer's message
2. Urgency: How time-sensitive this issue is
3. Workflow steps: The ordered sequence of actions needed to resolve this
4. Tools required: Which enterprise APIs/tools would be needed
5. Key entities: Important data points extracted from the message
6. Confidence: Your confidence in this classification (0.0-1.0)

Available tools for the tools_required field:
- order_lookup_api: Look up order details by order ID
- shipment_tracking_api: Track shipment status
- customer_profile_api: Get customer account info
- refund_processing_api: Process monetary refunds
- payment_gateway_api: Handle payment operations
- invoice_verification_api: Validate invoice data
- warranty_check_api: Check warranty status
- ticket_update_api: Update ticket state
- notification_service: Send customer notifications
- policy_engine: Check business rules and policies
- vision_api: Analyze product images for damage
- document_verification: OCR and verify uploaded documents

Common workflow steps:
- verify_identity, check_order, check_shipment, verify_return_policy, validate_invoice, assess_damage, calculate_refund, initiate_refund, create_replacement, schedule_pickup, escalate_to_supervisor, notify_customer, update_ticket, request_additional_info

Respond ONLY with valid JSON matching the required schema. No additional text.
```

### Script 2: `scripts/2_train.py`

Purpose: Format annotated data into ChatML, fine-tune Qwen3-4B with QLoRA using Unsloth.

Implementation requirements:

1. Load `poc_config.yaml` for all hyperparameters.

2. Load annotated data from `data/annotated/annotated_5k.jsonl`.

3. Convert each record to ChatML messages format:

```python
def format_training_example(record: dict) -> dict:
    system_prompt = """You are Kiki, an AI customer service agent. When given a customer message, analyze it and respond with:
1. Your classification (intent, urgency)
2. The workflow steps needed to resolve this
3. Which tools to invoke with what parameters
4. A professional, empathetic response to the customer

Always respond in valid JSON with these fields:
- intent: string
- urgency: string (critical/high/medium/low)
- workflow_steps: list of strings
- tools_required: list of strings with parameters
- reasoning: brief explanation of your analysis
- response: the customer-facing reply"""

    assistant_output = json.dumps({
        "intent": record["intent"],
        "urgency": record["urgency"],
        "workflow_steps": record["workflow_steps"],
        "tools_required": record["tools_required"],
        "reasoning": f"Customer intent is {record['intent']} with {record['urgency']} urgency. "
                     f"Required workflow: {' → '.join(record['workflow_steps'])}.",
        "response": record["agent_response"]
    }, indent=2)

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": record["customer_message"]},
            {"role": "assistant", "content": assistant_output}
        ]
    }
```

4. Split: 90% train, 10% eval. Save formatted data to `data/formatted/`.

5. Load model with Unsloth:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,  # auto-detect
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
```

6. Train with TRL SFTTrainer:

```python
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="outputs/models/kiki-poc-v1",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    optim="adamw_8bit",
    max_seq_length=2048,
    packing=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    save_strategy="steps",
    save_steps=200,
    eval_strategy="steps",
    eval_steps=200,
    logging_steps=10,
    report_to="none",  # or "wandb" if configured
    seed=42,
)
```

7. After training, save the adapter and tokenizer. Print training stats (loss curve, duration, peak VRAM).

8. Export to GGUF for llama.cpp/MLX inference:

```python
model.save_pretrained_gguf(
    "outputs/exports/kiki-poc-q4",
    tokenizer,
    quantization_method="q4_k_m",
)
```

### Script 3: `scripts/3_evaluate.py`

Purpose: Run the fine-tuned SLM, GPT-4o, and Claude Sonnet on the same 100 gold-labeled test tickets. Produce comparison tables and metrics.

Implementation requirements:

1. Load 100 gold-labeled tickets from `data/gold/gold_100.jsonl`. These must have human-verified `intent`, `urgency`, `workflow_steps`, and `tools_required` labels. If this file doesn't exist, the script should create a template and instruct the user to fill it.

2. Load the fine-tuned model using Unsloth or vLLM for inference.

3. For each test ticket, run inference through three systems:
   - Fine-tuned SLM (local, via the loaded model)
   - GPT-4o (via OpenAI API)
   - Claude Sonnet (via Anthropic API)

   All three receive the SAME system prompt and user message. Parse the JSON output from each.

4. Compute metrics for each system:
   - Intent accuracy: exact match percentage
   - Intent F1: micro-averaged F1 across all intent classes
   - Urgency accuracy: exact match percentage
   - Workflow accuracy: ordered sequence match using edit distance (Levenshtein on step lists), normalized to 0-1
   - Tool selection F1: set-based F1 (predicted tools vs gold tools)
   - Response quality: use GPT-4o as judge, scoring 1-5 on helpfulness, correctness, professionalism, empathy. Average across dimensions.

5. Compute latency for each system:
   - SLM: measure time from input to complete output
   - GPT-4o: measure API round-trip time
   - Claude: measure API round-trip time
   Report p50 and p95 latencies.

6. Compute cost estimates:
   - SLM: GPU cost per hour / tickets processed per hour
   - GPT-4o: actual token costs from API responses
   - Claude: actual token costs from API responses
   Project to monthly cost at 100K tickets/day.

7. Save all results to `outputs/results/evaluation_results.json` with this structure:

```json
{
    "timestamp": "2026-03-14T...",
    "num_test_tickets": 100,
    "systems": {
        "kiki_slm": {
            "model": "Qwen3-4B-Instruct-2507 + QLoRA",
            "intent_accuracy": 0.94,
            "intent_f1": 0.93,
            "urgency_accuracy": 0.88,
            "workflow_accuracy": 0.85,
            "tool_selection_f1": 0.91,
            "response_quality": 4.1,
            "latency_p50_ms": 120,
            "latency_p95_ms": 280,
            "cost_per_1k_tickets": 0.002,
            "monthly_cost_100k_day": 1080
        },
        "gpt4o": { ... },
        "claude_sonnet": { ... }
    },
    "per_ticket_results": [ ... ]
}
```

8. Print a formatted comparison table to stdout.

### Script 4: `scripts/4_demo.py`

Purpose: Launch a Gradio web interface showing the SLM processing tickets in real-time with full pipeline visualization.

Implementation requirements:

1. Load the fine-tuned model. Try MLX first (if on Apple Silicon), fall back to transformers with the QLoRA adapter.

For MLX:
```python
from mlx_lm import load, generate
model, tokenizer = load("outputs/exports/kiki-poc-q4")
```

For transformers fallback:
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    "outputs/models/kiki-poc-v1",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

2. Build a Gradio interface with these components:

   - **Input**: Textbox for customer message, dropdown for channel (email/chat/phone), optional file upload (for future vision integration)
   - **Processing visualization**: Show each pipeline stage with timing:
     - Stage 1: Intent Classification → show intent + confidence + urgency badge
     - Stage 2: Workflow Planning → show ordered step list with tool mappings
     - Stage 3: Tool Invocation Plan → show JSON tool calls with parameters
     - Stage 4: Response Generation → show the customer-facing response
   - **Sidebar**: Model info (name, parameters, quantization), inference stats (tokens/sec, latency), cost comparison vs GPT-4o

3. Include 10 pre-loaded example tickets covering different intents:
   - "My order #ORD-48293 hasn't arrived yet. I placed it 5 days ago."
   - "I received a damaged laptop. The screen is cracked. I want a full refund."
   - "Can you help me change my delivery address for order #ORD-77102?"
   - "I've been charged twice for my subscription. Please fix this."
   - "I want to cancel my account and get a refund for the remaining months."
   - "What's your return policy for electronics?"
   - "Someone made unauthorized purchases on my account!"
   - "My warranty claim for product SN-847291 was denied. I want to appeal."
   - "I need to update my payment method to a new credit card."
   - "The product I received doesn't match the description on your website."

4. The interface should be launchable with: `python scripts/4_demo.py`

### POC config file (`configs/poc_config.yaml`)

```yaml
# Kiki SLM POC Configuration
# All hyperparameters for the 2-day experiment

project:
  name: kiki-slm-poc
  version: "0.1.0"

data:
  raw_dir: data/raw
  sample_size: 5000
  gold_test_size: 100
  train_split: 0.9

annotation:
  model: gpt-4o-mini
  max_concurrent: 20
  batch_size: 50
  temperature: 0.0
  max_retries: 3

model:
  name: Qwen/Qwen3-4B-Instruct-2507
  max_seq_length: 2048
  load_in_4bit: true

lora:
  r: 32
  lora_alpha: 64
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  lora_dropout: 0
  bias: none

training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  num_train_epochs: 3
  learning_rate: 2.0e-4
  lr_scheduler_type: cosine
  warmup_ratio: 0.03
  bf16: true
  optim: adamw_8bit
  packing: true
  gradient_checkpointing: true
  save_steps: 200
  eval_steps: 200
  seed: 42

evaluation:
  judge_model: gpt-4o
  comparison_models:
    - model: gpt-4o
      provider: openai
    - model: claude-sonnet-4-20250514
      provider: anthropic

export:
  format: gguf
  quantization: q4_k_m
  output_dir: outputs/exports
```

---

## 3. Phase 2: Production pipeline — modular architecture

### Directory structure

```
kiki/
├── pyproject.toml
├── Makefile
├── Dockerfile
├── docker-compose.yaml              # vLLM + Prometheus + Grafana
│
├── configs/
│   ├── base.yaml                    # Shared defaults
│   ├── sft/
│   │   ├── intent_classifier.yaml
│   │   ├── workflow_reasoner.yaml
│   │   ├── tool_caller.yaml
│   │   └── response_generator.yaml
│   ├── alignment/
│   │   ├── simpo.yaml
│   │   ├── dpo.yaml
│   │   ├── grpo.yaml
│   │   └── kto.yaml
│   ├── evaluation/
│   │   └── eval_suite.yaml
│   └── serving/
│       ├── vllm_multi_lora.yaml
│       └── mlx_local.yaml
│
├── src/kiki/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py               # Task 3.1
│   │   ├── processors.py            # Task 3.2
│   │   ├── pii_anonymizer.py        # Task 3.3
│   │   ├── annotators.py            # Task 3.4
│   │   ├── preference_builder.py    # Task 3.5
│   │   ├── quality_filter.py        # Task 3.6
│   │   ├── dataset_mixer.py         # Task 3.7
│   │   └── validators.py            # Task 3.8
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_loader.py          # Task 3.9
│   │   ├── peft_config.py           # Task 3.10
│   │   └── merge_adapter.py         # Task 3.11
│   │
│   ├── trainers/
│   │   ├── __init__.py
│   │   ├── base_trainer.py          # Task 3.12
│   │   ├── sft_trainer.py           # Task 3.13
│   │   ├── dpo_trainer.py           # Task 3.14
│   │   ├── grpo_trainer.py          # Task 3.15
│   │   └── kto_trainer.py           # Task 3.16
│   │
│   ├── rewards/
│   │   ├── __init__.py
│   │   ├── policy_compliance.py     # Task 3.17
│   │   ├── tool_accuracy.py         # Task 3.18
│   │   ├── response_quality.py      # Task 3.19
│   │   └── composite.py             # Task 3.20
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py             # Task 3.21
│   │   ├── metrics.py               # Task 3.22
│   │   ├── judges.py                # Task 3.23
│   │   └── test_suites/
│   │       ├── __init__.py
│   │       ├── test_intent.py       # Task 3.24
│   │       ├── test_tool_calling.py # Task 3.25
│   │       ├── test_workflow.py     # Task 3.26
│   │       └── test_safety.py       # Task 3.27
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── pipeline.py              # Task 3.28
│   │   ├── router.py                # Task 3.29
│   │   ├── tool_executor.py         # Task 3.30
│   │   ├── postprocessor.py         # Task 3.31
│   │   └── ab_testing.py            # Task 3.32
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                # Task 3.33
│       ├── experiment_tracker.py    # Task 3.34
│       └── gpu_utils.py             # Task 3.35
│
├── scripts/
│   ├── prepare_data.py              # Orchestrates full data pipeline
│   ├── download_datasets.py         # Downloads all open-source datasets
│   ├── train_sft.py                 # Entry point for SFT
│   ├── train_alignment.py           # Entry point for DPO/SimPO/GRPO/KTO
│   ├── generate_preferences.py      # Build preference pairs
│   ├── merge_and_export.py          # Merge adapters, export GGUF/MLX
│   ├── evaluate.py                  # Run evaluation suite
│   └── serve.py                     # Launch vLLM server
│
└── tests/
    ├── test_data/
    │   ├── test_loaders.py
    │   ├── test_processors.py
    │   ├── test_pii.py
    │   └── test_validators.py
    ├── test_trainers/
    │   ├── test_sft.py
    │   └── test_dpo.py
    └── test_inference/
        ├── test_pipeline.py
        └── test_router.py
```

### Task breakdown for each module

Below is the specification for every module. Build them in the numbered order.

---

#### Task 3.1: `src/kiki/data/loaders.py`

Abstract base class `BaseDataLoader` with method `load() -> datasets.Dataset` and `get_metadata() -> DatasetMetadata`.

Concrete implementations:
- `HuggingFaceLoader(dataset_id, split, subset)` — wraps `datasets.load_dataset`
- `CSVLoader(path, encoding, delimiter)` — handles CSV with auto-detection of common column name variants
- `JSONLLoader(path)` — loads line-delimited JSON
- `DatabaseLoader(connection_string, query)` — SQLAlchemy-based, loads from Zendesk/Freshdesk/Zammad databases
- `DatasetRegistry` — class-level registry mapping dataset names to loader configs, populated from YAML

All loaders must return HuggingFace `Dataset` objects with standardized column names. Include a `normalize_columns` method that maps variant column names to the canonical schema:
- `customer_message` ← body, text, message, input, instruction, query, content
- `agent_response` ← response, answer, reply, output, resolution
- `intent` ← intent, category, label, type, class
- `urgency` ← urgency, priority, severity

---

#### Task 3.2: `src/kiki/data/processors.py`

`ChatMLConverter` class with static methods for converting each dataset format to TRL's messages format.

Required converters:
- `from_bitext(example)` → messages with system prompt + user (instruction) + assistant (response), preserving intent/category metadata
- `from_ticket(example)` → messages with department/priority in system prompt
- `from_glaive_function_calling(example)` → parse SYSTEM/USER/ASSISTANT/FUNCTION_CALL/FUNCTION_RESPONSE format into proper tool_calls messages
- `from_xlam(example)` → parse xLAM's query/tools/answers format into messages with tool schemas
- `from_hermes(example)` → parse NousResearch Hermes XML-style tool calling
- `from_toolbench(example)` → parse Thought→Action→Observation chains
- `from_banking77(example)` → convert to intent classification format
- `from_clinc(example)` → convert to intent classification with OOS handling
- `from_kiki_annotated(example)` → convert our custom annotated format (from the POC annotator)
- `from_preference_pair(example)` → convert chosen/rejected format for DPO

Each converter returns `{"messages": [...]}` in the standard ChatML format. Include a `process_dataset(dataset, converter_name)` method that applies the correct converter and removes original columns.

The Kiki system prompt template used across converters:

```
You are Kiki, an AI customer service agent for {company_name}. You help customers by:
1. Understanding their request and classifying intent
2. Planning the resolution workflow
3. Invoking the correct enterprise tools
4. Generating professional, empathetic responses

Available tools: {tool_list}

Always respond in structured JSON with: intent, urgency, workflow_steps, tools_required, reasoning, response.
When you need to call a tool, use the function calling format.
If required data is missing, ask the customer for it.
If confidence is low (<0.7), recommend escalation to a human agent.
```

---

#### Task 3.3: `src/kiki/data/pii_anonymizer.py`

`PIIAnonymizer` class using Microsoft Presidio + spaCy for entity detection, Faker for synthetic replacement.

Entity types to detect and replace:
- PERSON → Faker name
- EMAIL_ADDRESS → Faker email
- PHONE_NUMBER → Faker phone
- CREDIT_CARD → "[CARD_XXXX]"
- US_SSN → "[SSN_REDACTED]"
- IBAN_CODE → "[IBAN_REDACTED]"
- IP_ADDRESS → Faker IPv4
- LOCATION → Faker city
- DATE_TIME → keep as-is (needed for context)
- Custom patterns via regex:
  - Order IDs (ORD-XXXXX) → keep (non-PII, needed for training)
  - Ticket IDs (TKT-XXXXX) → keep
  - Tracking numbers → replace with synthetic

Methods:
- `anonymize_text(text: str) -> tuple[str, list[DetectedEntity]]`
- `process_dataset(dataset: Dataset, text_columns: list[str]) -> Dataset` — parallel processing with `num_proc=4`
- `generate_audit_report(detections: list) -> dict` — summary of what was anonymized

Dependencies: `presidio-analyzer`, `presidio-anonymizer`, `spacy`, `faker`, `en_core_web_lg` spaCy model

---

#### Task 3.4: `src/kiki/data/annotators.py`

`LLMAnnotator` class for batched async annotation using GPT-4o-mini or Claude.

This is the production version of the POC `1_annotate.py` script, with:
- Async batching with configurable concurrency (semaphore)
- Structured output parsing via Pydantic models
- Retry with exponential backoff and jitter
- Quality scoring of responses (helpfulness, correctness, professionalism on 1-5 scale)
- Preference ranking (given two responses, which is better)
- Cost tracking per batch
- Resume from checkpoint (save progress every N items)

Key methods:
- `annotate_tickets(dataset, schema: type[BaseModel]) -> Dataset` — bulk annotation
- `score_responses(pairs: list[dict]) -> list[QualityScore]` — quality scoring
- `rank_preferences(prompt, response_a, response_b) -> str` — pairwise ranking
- `generate_synthetic_examples(intent, count, difficulty) -> list[dict]` — synthetic data generation for underrepresented categories

Supports both OpenAI and Anthropic as backends, selected via config.

---

#### Task 3.5: `src/kiki/data/preference_builder.py`

`PreferencePairBuilder` class for constructing DPO/SimPO training pairs.

Strategies:
1. `from_scored_responses(prompt, responses, scores, min_margin)` — pair highest vs lowest scored
2. `from_helpsteer(dataset)` — convert HelpSteer2/3 attribute scores to binary preferences
3. `from_ultrafeedback(dataset)` — use pre-computed chosen/rejected from Argilla cleaned version
4. `from_on_policy_generation(model, prompts, judge_model, n_samples)` — generate N responses per prompt using the SFT model, rank with judge, pair best vs worst
5. `from_human_corrections(original_response, corrected_response)` — agent edits as preference signal
6. `from_csat_scores(conversations, scores)` — CSAT 4-5 as chosen, 1-2 as rejected

Output format: `{"prompt": [...messages], "chosen": [...messages], "rejected": [...messages]}`

Include a `validate_pairs(pairs)` method that checks: chosen/rejected are different, prompt is non-empty, no data leakage between train/test.

---

#### Task 3.6: `src/kiki/data/quality_filter.py`

`QualityFilter` class for cleaning and filtering training data.

Filters:
- `dedup_exact(dataset)` — exact duplicate removal on customer_message
- `dedup_semantic(dataset, threshold=0.95)` — embedding-based near-duplicate removal using sentence-transformers
- `filter_length(dataset, min_tokens=10, max_tokens=2000)` — remove too-short or too-long examples
- `filter_language(dataset, allowed_languages=["en"])` — language detection with fasttext
- `filter_confidence(dataset, min_confidence=0.8)` — remove low-confidence annotations
- `filter_quality_score(dataset, min_score=3.0)` — remove low-quality responses
- `balance_intents(dataset, max_ratio=0.3)` — downsample any intent exceeding 30% of dataset
- `ensure_edge_cases(dataset, min_edge_ratio=0.1)` — verify at least 10% are edge cases

Returns filtered dataset with a report of what was removed and why.

---

#### Task 3.7: `src/kiki/data/dataset_mixer.py`

`DatasetMixer` class for combining multiple datasets with configurable weights.

```python
mixer = DatasetMixer(
    datasets={
        "bitext_cs": {"loader": "huggingface", "id": "bitext/Bitext-customer-support-llm-chatbot-training-dataset", "weight": 0.25},
        "bitext_ecom": {"loader": "huggingface", "id": "bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset", "weight": 0.15},
        "bitext_banking": {"loader": "huggingface", "id": "bitext/Bitext-retail-banking-llm-chatbot-training-dataset", "weight": 0.10},
        "kiki_tickets": {"loader": "jsonl", "path": "data/annotated/", "weight": 0.20},
        "arcee_agent": {"loader": "huggingface", "id": "arcee-ai/agent-data", "weight": 0.20},
        "safety": {"loader": "huggingface", "id": "gretelai/gretel-safety-alignment-en-v1", "weight": 0.10},
    },
    total_examples=100000,
    seed=42,
)
mixed = mixer.mix()  # Returns a single Dataset with proportional sampling
```

Methods:
- `mix() -> Dataset` — combine with proportional sampling
- `get_composition_report() -> dict` — show actual counts per source
- `validate_format_consistency()` — ensure all datasets have the same schema after conversion

---

#### Task 3.8: `src/kiki/data/validators.py`

Pydantic models for validating data at every pipeline stage.

```python
class RawTicket(BaseModel):
    customer_message: str = Field(min_length=5)
    agent_response: str = Field(min_length=5)

class AnnotatedTicket(RawTicket):
    intent: str
    urgency: Literal["critical", "high", "medium", "low"]
    workflow_steps: list[str] = Field(min_length=1)
    tools_required: list[str]
    confidence: float = Field(ge=0.0, le=1.0)

class ChatMLExample(BaseModel):
    messages: list[dict] = Field(min_length=2)
    # Each message must have "role" and "content"

class PreferencePair(BaseModel):
    prompt: list[dict]
    chosen: list[dict]
    rejected: list[dict]

class ToolCall(BaseModel):
    name: str
    parameters: dict

class SLMOutput(BaseModel):
    intent: str
    urgency: str
    workflow_steps: list[str]
    tools_required: list[str]
    reasoning: str
    response: str
```

Include a `validate_dataset(dataset, schema)` function that validates every row against the schema and reports errors.

---

#### Task 3.9: `src/kiki/models/model_loader.py`

Unified model loading that supports Unsloth (preferred for training), standard transformers, and vLLM (for GRPO generation).

```python
class ModelLoader:
    @staticmethod
    def load_for_training(config: ModelConfig) -> tuple[model, tokenizer]:
        """Load with Unsloth for QLoRA training."""
    
    @staticmethod
    def load_for_inference(model_path: str, quantize: bool = True) -> tuple[model, tokenizer]:
        """Load for local inference with adapter."""
    
    @staticmethod
    def load_for_merging(base_model: str, adapter_path: str) -> tuple[model, tokenizer]:
        """Load base + adapter for merging."""
```

Handle the Qwen3 chat template properly — it uses `<|im_start|>` / `<|im_end|>` tokens. Ensure `tokenizer.chat_template` is set correctly. Set `tokenizer.padding_side = "right"` for training.

---

#### Task 3.10: `src/kiki/models/peft_config.py`

Factory for creating task-appropriate LoRA configurations.

```python
class PEFTConfigFactory:
    PRESETS = {
        "intent_classifier": {
            "r": 16, "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
            "description": "Lightweight, classification-focused"
        },
        "workflow_reasoner": {
            "r": 32, "lora_alpha": 64,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
            "description": "Higher rank for multi-step reasoning"
        },
        "tool_caller": {
            "r": 32, "lora_alpha": 32,
            "target_modules": "all-linear",
            "description": "Structured output generation"
        },
        "response_generator": {
            "r": 64, "lora_alpha": 64,
            "target_modules": "all-linear",
            "description": "Full expressiveness for natural language"
        },
        "dpo_alignment": {
            "r": 16, "lora_alpha": 16,
            "target_modules": "all-linear",
            "modules_to_save": ["lm_head", "embed_tokens"],
            "description": "Lightweight for preference tuning"
        }
    }
```

---

#### Task 3.11: `src/kiki/models/merge_adapter.py`

Utilities for merging LoRA adapters into base models and exporting.

Methods:
- `merge_adapter(base_model, adapter_path) -> merged_model`
- `export_gguf(merged_model, output_path, quantization="q4_k_m")`
- `export_mlx(merged_model, output_path, quantize_bits=4)`
- `export_awq(merged_model, output_path)` — for vLLM serving
- `export_safetensors(merged_model, output_path)` — for HuggingFace Hub

---

#### Tasks 3.12-3.16: Trainers

All trainers inherit from `BaseTrainer` which provides:
- Config loading from YAML
- W&B / MLflow experiment tracking setup
- Checkpoint management
- Post-training evaluation trigger
- GPU memory monitoring and cleanup

**Task 3.12: `base_trainer.py`** — Abstract base with `train()`, `evaluate()`, `save()`, `load_config()`.

**Task 3.13: `sft_trainer.py`** — Wraps TRL `SFTTrainer`. Key: enable `packing=True`, use Unsloth's `get_peft_model`, gradient checkpointing with `use_reentrant=False`.

**Task 3.14: `dpo_trainer.py`** — Wraps TRL `DPOTrainer`. CRITICAL: learning rate must be 5e-6 (NOT 2e-4 like SFT). Monitor `rewards/margins` — if it doesn't grow, increase beta or decrease lr. Supports `loss_type`: "sigmoid" (standard DPO), "simpo" (reference-free SimPO via CPOTrainer).

**Task 3.15: `grpo_trainer.py`** — Wraps TRL `GRPOTrainer`. Enable `use_vllm=True` for fast generation. Accept a list of reward functions from `src/kiki/rewards/`. Configure `num_generations=8` (or 2 for efficiency).

**Task 3.16: `kto_trainer.py`** — Wraps TRL `KTOTrainer`. Binary labels only (good/bad). Asymmetric loss — penalizes bad outputs more than it rewards good ones. Use for final safety/compliance fine-tuning. Very conservative lr: 1e-6, single epoch.

---

#### Tasks 3.17-3.20: Reward functions for GRPO

**Task 3.17: `rewards/policy_compliance.py`**

Rule-based reward function checking enterprise policy compliance:
- Refund amount within authorized limits (auto-approve <$20, flag >$500)
- No PII exposure in responses (scan for SSN/card patterns)
- Escalation triggered when required (fraud, 3+ claims in 90 days)
- No fabricated information (claims must reference tool call results)
- Scope boundaries respected (no legal/medical/financial advice)
- Returns: float between -1.0 (critical violation) and +0.5 (fully compliant)

**Task 3.18: `rewards/tool_accuracy.py`**

Checks tool calling correctness:
- Correct tool name selected for the intent
- Valid JSON parameters
- Required parameters present
- Parameter types match schema
- Returns: 1.0 (perfect), 0.5 (right tool wrong params), 0.0 (wrong tool or invalid JSON)

**Task 3.19: `rewards/response_quality.py`**

Evaluates customer-facing response quality:
- Contains concrete next steps (not vague)
- Appropriate tone (professional, empathetic)
- Reasonable length (not too short, not too verbose)
- Addresses the customer's actual concern
- Returns: float 0.0 to 1.0

**Task 3.20: `rewards/composite.py`**

Combines reward functions with configurable weights:
```python
class CompositeReward:
    def __init__(self, rewards_config: dict):
        self.rewards = {
            "policy_compliance": (PolicyComplianceReward(), 0.35),
            "tool_accuracy": (ToolAccuracyReward(), 0.25),
            "response_quality": (ResponseQualityReward(), 0.25),
            "format_validity": (FormatValidityReward(), 0.15),
        }
    
    def __call__(self, completions, **kwargs) -> list[float]:
        # Weighted sum of all reward components
```

---

#### Tasks 3.21-3.27: Evaluation framework

**Task 3.21: `evaluation/evaluator.py`** — Orchestrator that runs all test suites, aggregates results, produces reports.

**Task 3.22: `evaluation/metrics.py`** — Custom metrics:
- `intent_f1(predictions, labels)` — micro/macro F1 with per-class breakdown
- `workflow_edit_distance(predicted_steps, gold_steps)` — normalized Levenshtein
- `tool_set_f1(predicted_tools, gold_tools)` — set-based precision/recall/F1
- `decision_accuracy(predicted_decisions, gold_decisions)` — multi-class accuracy
- `schema_validity_rate(outputs)` — percentage of outputs that parse as valid JSON matching SLMOutput schema

**Task 3.23: `evaluation/judges.py`** — LLM-as-judge implementation using GPT-4o or Claude. Score on dimensions: helpfulness (1-5), correctness (1-5), professionalism (1-5), empathy (1-5). Include calibration against human scores.

**Tasks 3.24-3.27: Test suites** — Each loads a gold test set and computes relevant metrics. `test_safety.py` checks for PII leakage, policy violations, and scope breaches in generated responses.

---

#### Tasks 3.28-3.32: Inference pipeline

**Task 3.28: `inference/pipeline.py`** — Multi-stage pipeline:
1. Intent classification (via intent LoRA adapter or guided_choice)
2. Workflow planning (via workflow LoRA)
3. Tool invocation (via tool LoRA, parse structured output)
4. Response generation (via response LoRA)

Uses OpenAI-compatible client pointing at vLLM server. Each stage specifies the `model` parameter to select the LoRA adapter.

**Task 3.29: `inference/router.py`** — Routes requests to the correct adapter. Supports A/B testing via deterministic user_id hashing. Dynamic adapter loading/unloading via vLLM's `/v1/load_lora_adapter` endpoint.

**Task 3.30: `inference/tool_executor.py`** — Parses tool calls from model output, validates parameters against tool schemas, executes against mock or real APIs, returns results. Include a `MockToolExecutor` for testing with predefined responses.

**Task 3.31: `inference/postprocessor.py`** — Safety filter (scan for PII in output, block policy violations), response formatting, confidence scoring, escalation logic (if confidence < 0.7 or policy violation detected, flag for human review).

**Task 3.32: `inference/ab_testing.py`** — Deterministic experiment assignment. Track metrics per variant. Statistical significance testing (chi-squared for rates, t-test for continuous metrics).

---

#### Tasks 3.33-3.35: Utilities

**Task 3.33: `utils/config.py`** — YAML config loading with OmegaConf. Merge base.yaml with task-specific configs. Environment variable interpolation. Validation against expected schema.

**Task 3.34: `utils/experiment_tracker.py`** — Thin wrapper around W&B and MLflow. Auto-log configs, metrics, artifacts. Support `report_to: ["wandb"]` or `report_to: ["mlflow"]` in configs.

**Task 3.35: `utils/gpu_utils.py`** — `get_gpu_memory()`, `clear_gpu_cache()`, `estimate_training_time(model_size, dataset_size, batch_size)`, `check_flash_attention_available()`.

---

## 4. Data schemas and contracts

### Canonical ticket schema (internal)

Every dataset, regardless of source, is normalized to this schema before processing:

```python
{
    "id": "TKT-394848",                    # unique identifier
    "customer_message": "My order...",      # required
    "agent_response": "Your order is...",   # required for SFT
    "intent": "order_status",              # from annotation
    "urgency": "medium",                   # from annotation
    "category": "shipping_issue",          # broader category
    "workflow_steps": ["check_order", "track_shipment", "notify_customer"],
    "tools_required": ["order_lookup_api", "shipment_tracking_api"],
    "key_entities": {"order_id": "ORD-48293"},
    "resolution": "shipment_tracking_provided",
    "escalated": false,
    "channel": "email",                    # email/chat/phone/social
    "language": "en",
    "source_dataset": "bitext_cs",         # provenance tracking
    "annotation_confidence": 0.92
}
```

### ChatML training format

```json
{
    "messages": [
        {"role": "system", "content": "You are Kiki..."},
        {"role": "user", "content": "My order #ORD-48293 hasn't arrived."},
        {"role": "assistant", "content": "{\"intent\": \"order_status\", ...}"}
    ]
}
```

### Tool calling format (matches Qwen3's native format)

```json
{
    "messages": [
        {"role": "system", "content": "You are Kiki...", "tools": [
            {"type": "function", "function": {
                "name": "order_lookup_api",
                "description": "Look up order details",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "The order ID"}
                    },
                    "required": ["order_id"]
                }
            }}
        ]},
        {"role": "user", "content": "Where is my order ORD-48293?"},
        {"role": "assistant", "content": null, "tool_calls": [
            {"type": "function", "function": {
                "name": "order_lookup_api",
                "arguments": "{\"order_id\": \"ORD-48293\"}"
            }}
        ]},
        {"role": "tool", "name": "order_lookup_api", "content": "{\"status\": \"in_transit\", \"eta\": \"2026-01-12\"}"},
        {"role": "assistant", "content": "Your order ORD-48293 is currently in transit and should arrive by January 12th."}
    ]
}
```

### DPO preference format

```json
{
    "prompt": [
        {"role": "system", "content": "You are Kiki..."},
        {"role": "user", "content": "I want a refund for my damaged laptop."}
    ],
    "chosen": [
        {"role": "assistant", "content": "I'm sorry about the damage. Let me look into this right away. I'll need your order number to process the refund..."}
    ],
    "rejected": [
        {"role": "assistant", "content": "Please provide your order number."}
    ]
}
```

---

## 5. Dataset registry

### SFT datasets to download

```yaml
# scripts/download_datasets.py should download all of these
sft_datasets:
  # Customer service conversations
  - id: bitext/Bitext-customer-support-llm-chatbot-training-dataset
    size: 27K
    converter: bitext
    use: general CS intent + response training
    license: CDLA-Sharing-1.0
    
  - id: bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset
    size: 50K+
    converter: bitext
    use: e-commerce specific intents (67 intents)
    license: CDLA-Sharing-1.0
    
  - id: bitext/Bitext-retail-banking-llm-chatbot-training-dataset
    size: 37K+
    converter: bitext
    use: banking intents (activate_card, apply_for_loan, etc.)
    license: CDLA-Sharing-1.0
    
  - id: bitext/Bitext-insurance-llm-chatbot-training-dataset
    size: 38K+
    converter: bitext
    use: insurance intents (claims, policy, coverage)
    license: CDLA-Sharing-1.0

  - id: Tobi-Bueck/customer-support-tickets
    size: 61.8K
    converter: ticket
    use: realistic helpdesk structure with priority, queue, tags
    license: CC-BY-NC-4.0

  - id: PolyAI/banking77
    size: 13K
    converter: banking77
    use: fine-grained banking intent classification (77 classes)
    license: CC-BY-4.0

  - id: clinc/clinc_oos
    size: 23.7K
    converter: clinc
    use: intent classification with out-of-scope detection (150+1 classes)
    license: CC-BY-3.0

  # Tool calling
  - id: arcee-ai/agent-data
    size: 486K
    converter: arcee_agent
    use: combined tool calling + chat (prevents catastrophic forgetting)
    license: MIT

  - id: Salesforce/xlam-function-calling-60k
    size: 60K
    converter: xlam
    use: high-precision function calling (>95% verified)
    license: CC-BY-4.0

  - id: NousResearch/hermes-function-calling-v1
    size: 11.6K
    converter: hermes
    use: multi-turn tool calling with XML tags
    license: Apache-2.0

  - id: Team-ACE/ToolACE
    size: 11.3K
    converter: toolace
    use: additional tool calling diversity
    license: Apache-2.0

  # Workflow reasoning
  - id: capitalone/T1
    size: 13.5K
    converter: t1
    use: multi-domain multi-turn tool conversations with caching
    license: CC-BY-4.0

  # Safety
  - id: gretelai/gretel-safety-alignment-en-v1
    size: varies
    converter: safety
    use: unsafe→safe response pairs for safety SFT
    license: Apache-2.0

preference_datasets:
  - id: argilla/ultrafeedback-binarized-preferences-cleaned
    size: 61K pairs
    use: primary DPO dataset (cleaned, plug-and-play)
    license: MIT

  - id: nvidia/HelpSteer3
    size: 40.5K pairs
    use: rich preference with annotator reasoning
    license: CC-BY-4.0

  - id: Anthropic/hh-rlhf
    size: 170K pairs
    use: helpfulness + harmlessness preferences
    license: MIT

  - id: berkeley-nest/Nectar
    size: 182K × 7 responses
    use: 7-way ranking for reward model or rich DPO pairs
    license: Apache-2.0
```

---

## 6. Training configurations

### SFT configs

All configs live in `configs/sft/`. Each config inherits from `configs/base.yaml`.

**`configs/base.yaml`:**
```yaml
model:
  name: Qwen/Qwen3-4B-Instruct-2507
  max_seq_length: 2048
  load_in_4bit: true
  use_unsloth: true

defaults:
  bf16: true
  gradient_checkpointing: true
  gradient_checkpointing_kwargs:
    use_reentrant: false
  optim: adamw_8bit
  report_to: [wandb]
  save_strategy: steps
  save_steps: 500
  eval_strategy: steps
  eval_steps: 500
  logging_steps: 10
  seed: 42
```

**`configs/sft/intent_classifier.yaml`:**
```yaml
inherits: base.yaml
task: intent_classifier
output_dir: runs/sft-intent-v1

lora:
  r: 16
  lora_alpha: 16
  target_modules: [q_proj, v_proj]

training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  num_train_epochs: 5
  learning_rate: 2.0e-4
  packing: true
  max_seq_length: 512

datasets:
  - id: bitext/Bitext-customer-support-llm-chatbot-training-dataset
    weight: 0.3
  - id: PolyAI/banking77
    weight: 0.3
  - id: clinc/clinc_oos
    weight: 0.2
  - id: kiki_annotated
    weight: 0.2
```

**`configs/sft/tool_caller.yaml`:**
```yaml
inherits: base.yaml
task: tool_caller
output_dir: runs/sft-tools-v1

lora:
  r: 32
  lora_alpha: 32
  target_modules: all-linear

training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  num_train_epochs: 3
  learning_rate: 2.0e-4
  packing: true
  max_seq_length: 2048

datasets:
  - id: arcee-ai/agent-data
    weight: 0.4
  - id: Salesforce/xlam-function-calling-60k
    weight: 0.3
  - id: NousResearch/hermes-function-calling-v1
    weight: 0.15
  - id: Team-ACE/ToolACE
    weight: 0.15
```

### Alignment configs

**`configs/alignment/dpo.yaml`:**
```yaml
inherits: base.yaml
task: dpo_alignment
output_dir: runs/dpo-v1
base_model_or_adapter: runs/sft-response-v1/checkpoint-final

alignment:
  method: dpo
  beta: 0.1
  loss_type: sigmoid
  learning_rate: 5.0e-6         # CRITICAL: 10-100x smaller than SFT
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  num_train_epochs: 3
  max_length: 1536
  max_prompt_length: 768

lora:
  r: 16
  lora_alpha: 16
  target_modules: all-linear
  modules_to_save: [lm_head, embed_tokens]

datasets:
  - id: argilla/ultrafeedback-binarized-preferences-cleaned
    weight: 0.6
  - id: nvidia/HelpSteer3
    weight: 0.3
  - id: kiki_preferences
    weight: 0.1
```

**`configs/alignment/grpo.yaml`:**
```yaml
inherits: base.yaml
task: grpo_alignment
output_dir: runs/grpo-v1
base_model_or_adapter: runs/dpo-v1/checkpoint-final

alignment:
  method: grpo
  learning_rate: 1.0e-6
  per_device_train_batch_size: 4
  num_train_epochs: 1
  use_vllm: true
  vllm_mode: colocate
  num_generations: 8
  max_completion_length: 512
  max_prompt_length: 1024
  kl_coef: 0.001

rewards:
  policy_compliance:
    weight: 0.35
    module: kiki.rewards.policy_compliance
  tool_accuracy:
    weight: 0.25
    module: kiki.rewards.tool_accuracy
  response_quality:
    weight: 0.25
    module: kiki.rewards.response_quality
  format_validity:
    weight: 0.15
    module: kiki.rewards.format_validity

datasets:
  - id: kiki_grpo_prompts
    path: data/processed/grpo_prompts.jsonl
```

---

## 7. RL alignment pipeline detail

### Training sequence (mandatory order)

```
Stage 1: SFT (foundation)
    ├── Intent classifier adapter (r=16)
    ├── Workflow reasoner adapter (r=32)
    ├── Tool caller adapter (r=32)
    └── Response generator adapter (r=64)
         │
Stage 2: SimPO (broad alignment, reference-free)
    └── Align response generator on CSAT-derived preferences
         │
Stage 3: DPO (preference alignment)
    └── Align on UltraFeedback + HelpSteer3 + Kiki preferences
         │
Stage 4: GRPO (rule-based RL with verifiable rewards)
    └── Optimize against 4 composite reward functions
         │
Stage 5: KTO (risk-weighted safety)
    └── Final pass on high-stakes edge cases (binary good/bad)
         │
Production deployment
```

### Building GRPO prompt dataset

The GRPO dataset is just prompts (no expected outputs — the model generates those during training). Build from:

1. Extract 10K diverse customer messages from the annotated dataset
2. Include edge cases: multi-issue tickets, ambiguous intents, policy boundary cases, potential fraud
3. Include scenarios requiring tool calls, escalation, and missing data detection
4. Format: `{"prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]}`

### Building KTO dataset

KTO needs binary labels. Source from:
- Good examples: resolved tickets with CSAT 4-5, compliant tool calls, accepted AI suggestions
- Bad examples: policy violations, incorrect tool calls, PII leaks, rejected AI suggestions, customer complaints about AI responses

Format: `{"prompt": [...], "completion": "...", "label": true/false}`

---

## 8. Inference and serving

### vLLM multi-LoRA server launch

```bash
# Production serving command
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --quantization awq \
  --enable-prefix-caching \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 64 \
  --max-cpu-loras 8 \
  --lora-modules \
    intent-classifier=./adapters/intent/ \
    workflow-reasoner=./adapters/workflow/ \
    tool-caller=./adapters/tools/ \
    response-gen=./adapters/response/ \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --enable-chunked-prefill \
  --port 8000 \
  --api-key kiki-internal
```

### docker-compose.yaml

```yaml
version: "3.8"
services:
  vllm:
    image: vllm/vllm-openai:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "8000:8000"
    volumes:
      - ./adapters:/adapters
      - ./models:/models
    command: >
      --model Qwen/Qwen3-4B-Instruct-2507
      --enable-lora --max-loras 4
      --enable-prefix-caching
      --quantization awq
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - ./monitoring/dashboards:/var/lib/grafana/dashboards
```

### MLX serving for Apple Silicon demos

```bash
pip install mlx-lm
mlx_lm.server \
  --model outputs/exports/kiki-mlx-4bit \
  --port 8080 \
  --chat-template chatml
```

---

## 9. Evaluation framework targets

### Metric targets (from Kiki spec)

| Metric | Target | Test set size | Method |
|--------|--------|---------------|--------|
| Intent accuracy | >95% micro-F1 | 500+ gold tickets | Exact match per class |
| Workflow accuracy | >90% | 200+ workflow cases | Edit distance scoring |
| Tool selection accuracy | >92% set F1 | 200+ tool cases | Set precision/recall |
| Decision accuracy | >88% | 200+ decision cases | Multi-class accuracy |
| Response quality | >4.2/5 | 100+ human eval | LLM-as-judge + human panel |
| Error detection rate | >90% | 100+ synthetic failures | Missing data/tool detection |
| Latency (p95) | <500ms | 1000 requests | vLLM benchmark |
| Schema validity | >98% | All test outputs | JSON parse + Pydantic validation |

### Evaluation command

```bash
python scripts/evaluate.py \
  --config configs/evaluation/eval_suite.yaml \
  --model runs/grpo-v1/checkpoint-final \
  --test-set data/gold/gold_500.jsonl \
  --output outputs/results/eval_report.json
```

---

## 10. Continuous learning

### Monthly cycle

1. Export production logs (SLM inferences + outcomes)
2. Sample 5-10% for human review
3. Capture agent overrides (human corrected AI suggestions)
4. Build new preference pairs from corrections
5. Retrain adapters if >500 new corrections accumulated
6. Run full evaluation suite
7. Deploy only if all metrics hold or improve

### Quarterly cycle

Week 1: Audit metrics, identify degraded categories, cluster emerging intents
Week 2: Curate new training data, update intent taxonomy, refresh gold test set
Week 3: Full retrain with hyperparameter sweep
Week 4: Staged rollout (shadow → 5% canary → 25% → 100%)

### Version control

Every adapter version stored with:
- Training data hash (SHA-256 of the dataset)
- All metric scores at evaluation time
- Config snapshot (full YAML)
- Git commit hash
- Date and responsible engineer

Maintain last 3 adapter versions for instant rollback.

---

## 11. Dependencies and environment setup

### `requirements.txt` (POC)

```
# Core
torch>=2.4.0
transformers>=4.46.0
datasets>=3.0.0
accelerate>=1.0.0
peft>=0.13.0
trl>=0.12.0
unsloth>=2024.12

# Inference
mlx-lm>=0.19.0          # Apple Silicon only
vllm>=0.6.0             # GPU serving

# Annotation
openai>=1.50.0
anthropic>=0.37.0
pydantic>=2.0

# Demo
gradio>=5.0.0

# Utilities
pyyaml
tqdm
pandas
numpy
```

### `requirements.txt` (Production — add to above)

```
# Data processing
presidio-analyzer>=2.2
presidio-anonymizer>=2.2
spacy>=3.7
faker>=30.0
fasttext-wheel>=0.9.2
sentence-transformers>=3.0

# Evaluation
deepeval>=1.0
scikit-learn>=1.5

# Experiment tracking
wandb>=0.18
mlflow>=2.17

# Monitoring
prometheus-client>=0.21

# Config
omegaconf>=2.3
```

### Environment setup commands

```bash
# Create environment
conda create -n kiki python=3.11 -y
conda activate kiki

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Unsloth (fastest single-GPU training)
pip install unsloth

# Install remaining dependencies
pip install -r requirements.txt

# Download spaCy model for PII detection
python -m spacy download en_core_web_lg

# Login to services
huggingface-cli login
wandb login
```

---

## Build order for Claude Code

### Priority 1: POC (build first, test immediately)

1. Create project structure for `kiki-poc/`
2. Write `configs/poc_config.yaml`
3. Write `prompts/annotator_system.txt`
4. Build `scripts/1_annotate.py` — test with 10 tickets first
5. Build `scripts/2_train.py` — test with 100 examples first
6. Build `scripts/3_evaluate.py`
7. Build `scripts/4_demo.py`

### Priority 2: Production data pipeline

8. Build `src/kiki/data/validators.py` (schemas first)
9. Build `src/kiki/data/loaders.py`
10. Build `src/kiki/data/pii_anonymizer.py`
11. Build `src/kiki/data/processors.py`
12. Build `src/kiki/data/annotators.py`
13. Build `src/kiki/data/quality_filter.py`
14. Build `src/kiki/data/dataset_mixer.py`
15. Build `src/kiki/data/preference_builder.py`
16. Write `scripts/download_datasets.py`
17. Write `scripts/prepare_data.py`

### Priority 3: Production training pipeline

18. Build `src/kiki/utils/config.py`
19. Build `src/kiki/utils/gpu_utils.py`
20. Build `src/kiki/utils/experiment_tracker.py`
21. Build `src/kiki/models/model_loader.py`
22. Build `src/kiki/models/peft_config.py`
23. Build `src/kiki/trainers/base_trainer.py`
24. Build `src/kiki/trainers/sft_trainer.py`
25. Write `scripts/train_sft.py`
26. Write all SFT config YAMLs
27. Build `src/kiki/trainers/dpo_trainer.py`
28. Build `src/kiki/rewards/` (all 4 modules)
29. Build `src/kiki/trainers/grpo_trainer.py`
30. Build `src/kiki/trainers/kto_trainer.py`
31. Write `scripts/train_alignment.py`
32. Write all alignment config YAMLs

### Priority 4: Production inference and evaluation

33. Build `src/kiki/evaluation/metrics.py`
34. Build `src/kiki/evaluation/judges.py`
35. Build `src/kiki/evaluation/evaluator.py`
36. Build all test suites (3.24-3.27)
37. Build `src/kiki/inference/pipeline.py`
38. Build `src/kiki/inference/router.py`
39. Build `src/kiki/inference/tool_executor.py`
40. Build `src/kiki/inference/postprocessor.py`
41. Build `src/kiki/inference/ab_testing.py`
42. Build `src/kiki/models/merge_adapter.py`
43. Write `scripts/merge_and_export.py`
44. Write `scripts/serve.py`
45. Write `docker-compose.yaml`
46. Write `Dockerfile`
47. Write `Makefile`
48. Write `pyproject.toml`

### Priority 5: Tests

49. Write unit tests for data modules
50. Write unit tests for trainers (mock model, test config loading)
51. Write integration tests for inference pipeline
52. Write end-to-end test: data → train → evaluate → serve

---

## Key decisions to remember

1. **Learning rates**: SFT = 2e-4, DPO = 5e-6, GRPO = 1e-6, KTO = 1e-6
2. **DPO diagnostic**: if `rewards/margins` doesn't grow, increase beta or decrease lr
3. **Packing = True**: always for SFT, eliminates padding waste
4. **Unsloth gradient checkpointing**: use `"unsloth"` mode, not standard
5. **Qwen3 chat template**: uses `<|im_start|>` / `<|im_end|>`, set `tokenizer.padding_side = "right"`
6. **No single intent > 30%** of dataset — enforce with stratified sampling
7. **10% edge cases minimum** in every training dataset
8. **PII anonymized BEFORE annotation** — never send raw PII to annotation APIs
9. **Gold test set NEVER used in training** — maintain strict separation
10. **Every adapter versioned** with data hash + metrics + config snapshot
