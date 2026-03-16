# Training Pipeline Production Refactor — Design Spec

**Date:** 2026-03-15
**Status:** Approved (post-review, v2)
**Scope:** Refactor Colab training notebook into moderate-thin executor backed by production scripts with W&B monitoring

---

## Problem Statement

The current Colab notebook has 30+ cells with inline processing logic, broken logging, incorrect step counts, and no training curve visibility. All processing should live in version-controlled scripts; the notebook should be a thin-to-moderate executor that calls scripts and displays results.

### Specific bugs to fix:
1. Step count shows 7577 but actual training does 945 (packing + gradient accumulation not accounted for)
2. LiveLossPlotCallback never fires `on_train_end` in Colab
3. tqdm progress bars invisible (stderr buffering in IPython display system)
4. Training curves never generated
5. No experiment tracking across runs

---

## Design

### Architecture: Notebook <-> Scripts

```
notebooks/kiki_sft_finetune.ipynb  (~15 cells, full rewrite of current 33-cell notebook)
    |
    |-- Cell: %cd {REPO_DIR}  (set CWD to repo root — required for script imports)
    |
    |-- Cell: !python -u scripts/colab_train.py --train-file ... --wandb
    |       +-- scripts/colab_train.py
    |               |-- loads data from Drive path
    |               |-- applies chat template (standalone helper, no kiki package import)
    |               |-- configures Unsloth + QLoRA + SFTTrainer
    |               |-- logs to W&B (real-time loss, lr, GPU, config)
    |               |-- saves adapter + metrics to Drive
    |               |-- patches saved tokenizer_config.json with original chat template
    |               +-- prints summary to stdout
    |
    |-- Cell: !python -u scripts/colab_eval.py --adapter ... --gold ... --base-model ...
    |       +-- scripts/colab_eval.py
    |               |-- loads base model, runs all gold tickets, frees VRAM
    |               |-- loads fine-tuned model, runs all gold tickets
    |               |-- strips <think> tokens, parses JSON
    |               |-- computes metrics (intent acc, urgency acc, JSON parse rate)
    |               |-- prints comparison table to stdout
    |               +-- saves results JSON to Drive
    |
    +-- Cell: display results / W&B link
```

### New Scripts

#### `scripts/colab_train.py`
Single entry point for Colab SFT training. Self-contained — does NOT import from `kiki` package to avoid editable-install complexity on Colab.

**Arguments:**
- `--train-file` — path to ChatML JSONL (on Drive)
- `--eval-file` — path to eval JSONL (on Drive)
- `--output-dir` — where to save adapter (on Drive)
- `--base-model` — HF model ID (default: Qwen/Qwen3-4B-Instruct-2507)
- `--max-seq-length` — default 2048
- `--lora-r` — default 32
- `--lora-alpha` — default 64
- `--epochs` — default 3
- `--lr` — default 2e-4
- `--batch-size` — auto-detect from GPU, overridable
- `--seed` — default 42
- `--wandb` — enable W&B logging (flag)
- `--wandb-project` — project name (default: kiki-slm)
- `--wandb-run-name` — run name (default: auto-generated)
- `--dry-run` — validate config, print expected settings, exit without loading model

**Responsibilities:**
1. Auto-detect GPU and set batch size / grad accum
2. Load data from JSONL using `datasets.load_dataset`
3. Apply chat template with standalone sanitization function (no `kiki` package dependency)
4. Save original chat template string, set `tokenizer.chat_template = None` for training
5. Configure SFTConfig with `dataset_text_field="text"`, `packing=True`
6. Set `report_to=["wandb"]` if --wandb flag; wrap `wandb.init()` in try/except, fallback to file logging if W&B fails
7. Print correct step count from `trainer.state.max_steps` inside `on_train_begin` callback (NOT from `len(dataloader) * epochs` which ignores packing and gradient accumulation)
8. Train with `disable_tqdm=False`; script invoked as `python -u` for unbuffered output
9. Save adapter + tokenizer to output-dir
10. Patch saved `tokenizer_config.json` on disk to restore original `chat_template` field (so the saved artifact works for inference without manual patching)
11. Save `training_metrics.json` with loss, duration, peak VRAM, W&B run URL, config snapshot
12. Print summary to stdout

**Step count fix detail:**
The old notebook calculated `len(dataloader) * epochs` which ignores:
- Packing (multiple examples per sequence, reducing dataloader length)
- Gradient accumulation (N batches per optimizer step)
Correct formula: `trainer.state.max_steps` which is set by HuggingFace Trainer as `ceil(len(dataloader) * epochs / gradient_accumulation_steps)`. This is only available after the training loop begins, so we read it from an `on_train_begin` callback and print it.

**tqdm fix detail:**
Running the script as `!python -u scripts/colab_train.py` (unbuffered) ensures tqdm's stderr output passes through to Colab's cell output in real-time, bypassing IPython's display buffering that caused invisible progress bars when running in-process.

**Chat template self-contained helper:**
The sanitization function is defined directly in `colab_train.py` (~20 lines). It strips messages to `role` + `content` only, applies `tokenizer.apply_chat_template(tokenize=False)`, and falls back to manual `<|im_start|>` formatting on error. This avoids importing from `kiki.data.processors` which would require `uv pip install -e .` on Colab.

**Loss masking note:**
`packing=True` trains on all tokens (system + user + assistant). Response-only masking requires `packing=False` which halves throughput. Current design uses packing for speed; response-only masking is a future optimization.

#### `scripts/colab_eval.py`
Base vs fine-tuned comparison for Colab. Also self-contained.

**Arguments:**
- `--adapter-path` — path to fine-tuned adapter
- `--base-model` — HF model ID
- `--gold-file` — path to gold JSONL
- `--output-file` — where to save results JSON
- `--max-seq-length` — default 2048

**Responsibilities:**
1. Load base model (Unsloth, 4bit), run inference on all gold tickets
2. Delete base model, clear CUDA cache (prevents OOM on T4 16GB)
3. Load fine-tuned model (Unsloth, 4bit with adapter), run inference on all gold tickets
4. Strip `<think>...</think>` tokens via regex, parse JSON
5. Compute: intent accuracy, urgency accuracy, JSON parse rate, avg latency
6. Print formatted comparison table + per-ticket detail to stdout
7. Save results JSON to output-file

**Sequential model loading:** Models are loaded one at a time with explicit `del model; torch.cuda.empty_cache()` between them. This ensures the script works on T4 (16GB) where loading both simultaneously would OOM.

### Modified Notebook Structure (~15 cells, full rewrite)

| # | Type | Content |
|---|------|---------|
| 1 | md | Title + prerequisites |
| 2 | code | Install uv + unsloth + wandb + datasets |
| 3 | code | Verify GPU |
| 4 | code | Mount Drive |
| 5 | code | Clone/pull repo + `%cd {REPO_DIR}` (sets CWD for all subsequent `!python` calls) |
| 6 | code | Configure paths (DRIVE_DATA_DIR, DRIVE_OUTPUT_DIR, etc.) + verify data files exist |
| 7 | code | `wandb.login()` (interactive, needs notebook) |
| 8 | code | Inspect data: load JSONL, show row count + source distribution + sample (~5 lines) |
| 9 | md | Training section header |
| 10 | code | `!python -u scripts/colab_train.py --train-file ... --wandb` |
| 11 | code | Display W&B dashboard link |
| 12 | md | Evaluation section header |
| 13 | code | `!python -u scripts/colab_eval.py --adapter-path ... --gold-file ...` |
| 14 | code | Quick 3-message inference test (small inline, display results) |
| 15 | md | Done + next steps |

### W&B Logging

What HuggingFace Trainer + W&B logs automatically:
- `train/loss` — every logging_steps
- `train/learning_rate` — every logging_steps
- `eval/loss` — every eval_steps
- `train/epoch` — current epoch
- System metrics: GPU utilization, VRAM, temperature
- Config: all hyperparameters captured

What we log additionally:
- `train/peak_vram_gb` — at end of training
- `dataset/train_examples`, `dataset/eval_examples` — counts
- `dataset/sources` — source distribution dict

**W&B error handling:** `wandb.init()` is wrapped in try/except. If it fails (network, auth), the script prints a warning and falls back to `report_to=["none"]` + file-based logging. Training continues regardless. HuggingFace Trainer already handles mid-training W&B errors gracefully (logs warning, continues training).

### File-based Fallback (when --wandb not passed)

- Save `{output_dir}/training_metrics.json` — final summary
- Print progress to stdout every logging_steps
- Always works, no account needed

---

## What stays in the notebook (inline)

1. **Path configuration** — user edits Drive paths
2. **Data inspection display** — load JSONL, show row count + source distribution (~5 lines read-only)
3. **W&B login** — `wandb.login()` needs interactive input
4. **Quick inference test** — 3 messages after eval, display results inline
5. **W&B link display** — show clickable URL after training

## What moves to scripts

1. Chat template application + sanitization
2. Model loading + LoRA config
3. SFTTrainer setup + training loop
4. Adapter saving + tokenizer patching
5. Base vs fine-tuned evaluation (sequential model loading)
6. JSON parsing with `<think>` stripping
7. Metrics computation + comparison table

---

## Files to create/modify

| Action | File | Description |
|--------|------|-------------|
| CREATE | `scripts/colab_train.py` | Training entry point for Colab (self-contained) |
| CREATE | `scripts/colab_eval.py` | Base vs fine-tuned eval for Colab (self-contained) |
| REWRITE | `notebooks/kiki_sft_finetune.ipynb` | Full rewrite: ~15 cells, moderate executor |
| VERIFY | `pyproject.toml` | wandb already in deps (no change needed) |
| VERIFY | `.gitignore` | wandb/ already ignored (no change needed) |

---

## Success Criteria

1. `colab_train.py` runs end-to-end on Colab with W&B logging — real-time loss curves visible in W&B dashboard
2. Correct step count displayed (matches actual training, accounts for packing + grad accum)
3. tqdm progress visible in Colab output (subprocess with unbuffered output)
4. Training curves in W&B dashboard (no matplotlib dependency)
5. `colab_eval.py` produces comparison table with `<think>` stripping, sequential model loading (no OOM on T4)
6. Saved adapter has correct chat template in `tokenizer_config.json` (works for inference)
7. Notebook is ~15 cells, no inline processing logic beyond display
8. `--dry-run` validates config without loading model
9. All scripts pass syntax check
10. W&B failure is non-fatal (falls back to file logging)
