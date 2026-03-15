# Kiki SLM POC — Priority 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the Phase 1 POC: 4 flat scripts (annotate, train, evaluate, demo) with configs, tested with 3 dummy tickets.

**Architecture:** Flat script-based POC in `kiki-poc/` style layout. GPT-4o-mini annotates raw tickets → QLoRA fine-tunes Qwen3-4B → evaluate against GPT-4o/Claude → Gradio demo. All config in one YAML.

**Tech Stack:** Python 3.11+, OpenAI API (annotation), Unsloth + TRL (training), Pydantic (validation), Gradio (demo), UV (package management)

---

### Task 1: Project Structure + Environment

**Files:**
- Create: directory tree (`scripts/`, `configs/`, `prompts/`, `data/{raw,sampled,annotated,formatted,gold}`, `outputs/{models,results,exports}`)
- Create: `pyproject.toml`
- Create: `.gitignore`

**Steps:**
1. Create all directories
2. Create `pyproject.toml` with UV-compatible project config and dependencies
3. Create `.gitignore` (data files, model outputs, venvs, __pycache__)
4. Init UV venv and install deps
5. Commit

---

### Task 2: `configs/poc_config.yaml`

**Files:**
- Create: `configs/poc_config.yaml`

**Steps:**
1. Write the full config YAML (as specified in plan Section 2)
2. Commit

---

### Task 3: `prompts/annotator_system.txt`

**Files:**
- Create: `prompts/annotator_system.txt`

**Steps:**
1. Write the annotation system prompt (as specified in plan Section 2)
2. Commit

---

### Task 4: `scripts/1_annotate.py`

**Files:**
- Create: `scripts/1_annotate.py`
- Create: `data/raw/sample_tickets.jsonl` (3 dummy tickets for testing)

**Steps:**
1. Create 3 dummy tickets in `data/raw/sample_tickets.jsonl`
2. Build `1_annotate.py` with: Pydantic schema, async OpenAI calls, stratified sampling, batch processing, retry logic, progress logging, summary stats
3. Test with dummy tickets (set sample_size=3 via CLI override)
4. Verify output in `data/annotated/`
5. Commit

---

### Task 5: `scripts/2_train.py`

**Files:**
- Create: `scripts/2_train.py`

**Steps:**
1. Build `2_train.py` with: config loading, ChatML formatting, Unsloth model loading, QLoRA setup, SFTTrainer, GGUF export
2. Commit (training runs on GPU, script is buildable locally)

---

### Task 6: `scripts/3_evaluate.py`

**Files:**
- Create: `scripts/3_evaluate.py`

**Steps:**
1. Build `3_evaluate.py` with: gold test loading, multi-system inference, metric computation (intent F1, workflow edit distance, tool F1), latency/cost tracking, comparison table output
2. Commit

---

### Task 7: `scripts/4_demo.py`

**Files:**
- Create: `scripts/4_demo.py`

**Steps:**
1. Build `4_demo.py` with: MLX/transformers model loading, Gradio UI with pipeline visualization, 10 example tickets, sidebar stats
2. Commit

---
