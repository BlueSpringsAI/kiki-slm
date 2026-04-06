# Loopper SLM Fine-Tuning Pipeline

## What This Is

Pipeline to fine-tune `unsloth/Qwen3-4B-Thinking-2507` to replace the Loopper LangGraph customer service agent. The SLM learns to: read a ticket, reason in `<think>` blocks, call `rag_search` 1-3 times, reason over results, and output an 11-field JSON response.

Teacher: the existing Loopper LangGraph agent (o4-mini triage + gpt-5-mini RAG ReACT). Cost ~$0.03/ticket.

## Key Decisions (DO NOT CHANGE without discussion)

- **Model**: `unsloth/Qwen3-4B-Thinking-2507` — NOT `Qwen/` prefix (Unsloth has patched chat template), NOT `-Instruct-2507` (no thinking mode)
- **Format**: `reasoning_content` field (separate from `content`) — Unsloth template renders it as `<think>` blocks
- **Rejection path**: `intent: "other"` + `is_valid: false` + `rejection_type` (5 values: spam/misdirected/newsletter/auto_reply/unrelated)
- **English-only**: first iteration is English only. `langdetect` library filters at both sample time and ChatML build time
- **Training**: LoRA rank 32, lr 1e-4, max_seq_length 4096, `train_on_responses_only` loss masking, `packing=True`

## Pipeline Scripts (run in order)

```bash
# 1. Sample tickets (already done — 4,642 English tickets)
uv run python -m scripts.loopper.sample_tickets

# 2. Generate traces via teacher agent (requires RAG server + OpenAI keys)
#    RAG server: cd ../rag-service && uv run uvicorn src.api.app:app --port 8010
uv run --project ../Looper-Support-Agent-Server python -m scripts.loopper.generate_traces \
  --sample 4642 --concurrency 5 --resume

# 3. Build ChatML dataset (has langdetect filter for non-English)
uv run python -m scripts.loopper.build_chatml

# 4. Validate dataset
uv run python -m scripts.loopper.validate_dataset

# 5. Verify format through tokenizer (sanity check)
uv run python -m scripts.loopper.verify_format

# 6. Train on Colab Pro (L4 24GB)
#    Upload train.jsonl + eval.jsonl, run scripts/colab_train.py
```

## Output Schema (11 fields)

The SLM outputs a single JSON with: `intent`, `urgency`, `confidence`, `is_valid`, `rejection_type`, `resolution_type`, `team`, `actions`, `summary`, `reasoning` (dict with 4 sub-fields), `response`.

See `configs/loopper_pipeline.yaml` validation section for allowed values.

## Common Pitfalls

- Don't strip `tool_calls` or `reasoning_content` from messages — `colab_train.py` had this bug, now fixed
- Don't mix `<think>` tags in content with `reasoning_content` field — pick one (we use the field)
- Don't use `system_notification` as a rejection type — not in the agent's Literal
- Don't forget `tools=` param when calling `apply_chat_template` — tools must appear in the system block
- The RAG MCP server must be running locally before `generate_traces.py`
- `content: null` (not `""`) on tool-call assistant turns — Python `None` handles this

## Workspace Context

This repo (`kiki-slm`) lives at `/Users/vishnu/Dev/bluesprings/slm/finetune-trials/curation/kiki-slm/`. Adjacent repos:
- `Looper-Support-Agent-Server/` — the teacher agent
- `rag-service/` — RAG MCP server (Qdrant + OpenAI embeddings)
- `finetune-dataset/data/` — sampled tickets, traces, ChatML output

Full handoff doc: `/Users/vishnu/Dev/bluesprings/slm/finetune-trials/curation/HANDOFF.md`
