# ML Engineer Handoff — Kiki SLM Training → Model Artifact

**Owner:** `kiki-train` repo owner (you).
**Goal:** produce a validated GGUF file + Modelfile in S3 that DevOps can point Ollama at.
**Input:** Freshdesk tickets + teacher agent running locally.
**Output:** `s3://<bucket>/kiki-sft-vN-Q4_K_M.gguf` + version metadata.

You do NOT need to touch AWS beyond the S3 upload. You do NOT need to touch the Loopper agent repo. Your handoff is a single S3 URI + version string.

---

## Prerequisites

- Access to Colab Pro (L4 24GB minimum, A100 or H100 preferred)
- `kiki-train` repo cloned and on branch `feat/loopper-organic-data-pipeline`
- Access to Google Drive folder `My Drive/kiki-slm/`
- AWS credentials with `s3:PutObject` on the bucket DevOps gives you

---

## Step 1 — Verify your training data is ready

```bash
# In kiki-train/ on your local machine
ls -lh data/sft-data/chatml/
# Expected:
#   train_trimmed.jsonl  (~47 MB, ~4099 examples)
#   eval_trimmed.jsonl   (~4.8 MB, ~445 examples)
ls data/sft-data/gold/gold_100.jsonl  # 100 gold tickets for eval
```

If any of these are missing, you're not ready — go back to the pipeline scripts (`sample_tickets.py` → `generate_traces.py` → `build_chatml.py` → `validate_dataset.py` → `trim_chatml.py`).

---

## Step 2 — Train on Colab

Upload `train_trimmed.jsonl`, `eval_trimmed.jsonl`, `gold_100.jsonl` to
`My Drive/kiki-slm/data/sft-data/` and `My Drive/kiki-slm/data/gold_100.jsonl`.

Open `notebooks/kiki_sft_finetune.ipynb` in Colab. Run cells 1 → 10 in order.

Watch for:
- Cell 2: GPU check should show ≥20 GB VRAM
- Cell 4: repo cloned + on the correct branch
- Cell 6: dataset loads, shows ~4k examples
- Cell 8 (train): loss drops below 0.5 within the first epoch; eval_loss improving each step
- Cell 10 (eval): intent_accuracy > 45%, json_parse_rate > 95% on the fine-tuned model

**If eval numbers are bad** (intent < 30%, parse < 80%), stop here and re-investigate training. Don't ship a bad model.

---

## Step 3 — Review eval results

```bash
# Download eval_results.json from Drive, then:
uv run python scripts/build_eval_report.py \
    --eval-file eval_results.json \
    --gold-file data/sft-data/gold/gold_100.jsonl \
    --out-dir reports/

open reports/eval_comparison.html
```

Gate criteria for proceeding to Step 4:

| Metric | Minimum | Red flag below |
|---|---:|---:|
| intent_accuracy | 45% | 30% |
| is_valid_accuracy | 80% | 60% |
| json_parse_rate | 95% | 80% |
| tool_name_f1 | 75% | 50% |
| avg_turns | 1.5–3.5 | >4 (never converges) |

If any metric is in the red zone, the model isn't ready. Go back to training (more epochs? higher rank? better data?).

---

## Step 4 — Export GGUF on Colab

Add a new cell at the end of the notebook:

```python
!python -u scripts/export_gguf.py \
    --adapter-dir {DRIVE_OUT} \
    --drive-out /content/drive/MyDrive/kiki-slm
```

Runtime: ~10–15 minutes total (5 min to load + merge, 5–10 min for GGUF conversion).

Outputs to `My Drive/kiki-slm/`:
- `merged/kiki-sft-v1/` — fp16 safetensors (~8 GB, recovery checkpoint)
- `gguf/kiki-sft-v1-Q4_K_M.gguf` — ~2.6 GB, this is what you ship
- `gguf/Modelfile` — ready to drop into Ollama

**If GGUF export fails** with a template error, you still have the fp16 merged checkpoint in `merged/kiki-sft-v1/`. Download it and convert locally with `llama.cpp/convert_hf_to_gguf.py` as a fallback. Don't ship the fp16 — it's 8 GB and too slow for CPU inference.

---

## Step 5 — Download GGUF + Modelfile to your Mac

```bash
# From Drive — whatever way you prefer (rclone, Drive app, web download)
# Target: ~/kiki-models/kiki-sft-v1/
ls -lh ~/kiki-models/kiki-sft-v1/
# Expected:
#   kiki-sft-v1-Q4_K_M.gguf  (~2.6 GB)
#   Modelfile
```

---

## Step 6 — Local smoke test with Ollama

This is **the** critical validation step. Do not skip.

```bash
# 1. Install + start Ollama
brew install ollama
ollama serve &

# 2. Create the model (one-time per model version)
cd ~/kiki-models/kiki-sft-v1
ollama create kiki-sft-v1 -f Modelfile
# Expected: "transferring model data ... success"

# 3. Quick sanity check
ollama run kiki-sft-v1 "Hello"
# Should respond (anything — just proves the model loaded)
# Press Ctrl+D to exit

# 4. Full smoke test with real gold tickets
cd ~/Desktop/kiki-train
uv run python scripts/test_kiki_local.py \
    --url http://localhost:11434 \
    --model kiki-sft-v1 \
    --gold-file data/sft-data/gold/gold_100.jsonl \
    --limit 5 \
    --verbose
```

**What good output looks like:**

```
  ── turn 1 (3.2s) ──
    content: <think>The customer is reporting a delivery delay...</think>
    tool: rag_search({"collection": "faq", "query": "delivery delay policy"})
  ── turn 2 (2.1s) ──
    content: <think>Retrieved context shows 5-day delay policy...</think>
    tool: rag_search({"collection": "communication_guidelines", "query": "delivery apology tone"})
  ── turn 3 (4.5s) ──
    content: {"intent":"delivery_issue","urgency":"high",...}
  ✓ parsed OK (3 turns, 9.8s)
    intent:          delivery_issue
    ✓ intent             gold='delivery_issue'  pred='delivery_issue'
```

**What bad output looks like (the #1 failure mode):**

```
  ── turn 1 (3.2s) ──
    content: <think>...</think>\n<tool_call>{"name":"rag_search",...}</tool_call>
    tool_calls: []       ← ❌ empty — Ollama didn't parse the tool_call block
```

If you see `tool_calls: []` with `<tool_call>` text in `content`, the Qwen3 chat template didn't survive GGUF conversion. **STOP and fix before proceeding** — see the "Template fix" section below. Shipping a model in this state will produce 0% parse rate in prod.

Gate criteria for proceeding to Step 7:

| Metric | Minimum |
|---|---:|
| parse rate | 4/5 (80%) |
| intent accuracy | 3/5 (60%) |
| avg turns | 1–3 |
| Each turn emits structured `tool_calls` | yes |

---

## Step 7 — Upload to S3

Once smoke test passes, upload to the S3 bucket DevOps gave you:

```bash
# Use versioned key — DO NOT overwrite the latest
KIKI_VERSION=v1   # bump this for every new model
aws s3 cp \
    ~/kiki-models/kiki-sft-v1/kiki-sft-v1-Q4_K_M.gguf \
    s3://<devops-bucket>/kiki-sft-${KIKI_VERSION}-Q4_K_M.gguf \
    --metadata "version=${KIKI_VERSION},trained=$(date -u +%Y-%m-%d),base=unsloth/Qwen3-4B-Thinking-2507"
```

Verify:

```bash
aws s3 ls s3://<devops-bucket>/ | grep kiki-sft
aws s3api head-object \
    --bucket <devops-bucket> \
    --key kiki-sft-${KIKI_VERSION}-Q4_K_M.gguf
```

---

## Step 8 — Hand off to DevOps

Send DevOps a single message with:

```
Model: kiki-sft-v1
S3 URI: s3://<bucket>/kiki-sft-v1-Q4_K_M.gguf
Quantization: Q4_K_M
Base model: unsloth/Qwen3-4B-Thinking-2507
File size: 2.6 GB
Context length: 4096
Smoke test: PASS (5/5 parse, 4/5 intent on local Ollama)
Eval on gold_100: intent=51%, is_valid=88%, json_parse=100%, tool_f1=87%
```

That's everything DevOps needs. They deploy, ping you when the ECS sidecar is up, you run the smoke test one more time against the ECS endpoint they give you (same script, different `--url`).

---

## Template fix (only if Step 6 showed the failure mode)

If GGUF conversion stripped the Qwen3 template's tool_call handling, override it manually in the Modelfile:

1. Get the correct template from the Unsloth base model:
   ```bash
   uv run python -c "
   from transformers import AutoTokenizer
   t = AutoTokenizer.from_pretrained('unsloth/Qwen3-4B-Thinking-2507')
   print(t.chat_template)
   " > qwen3_template.jinja
   ```

2. Add a `TEMPLATE` block to `Modelfile`:
   ```
   FROM ./kiki-sft-v1-Q4_K_M.gguf

   PARAMETER temperature 0.1
   PARAMETER num_ctx 4096
   PARAMETER num_predict 1024
   PARAMETER stop "<|im_end|>"

   TEMPLATE """<paste the contents of qwen3_template.jinja here, escaped>"""
   ```

3. Re-create the Ollama model:
   ```bash
   ollama rm kiki-sft-v1
   ollama create kiki-sft-v1 -f Modelfile
   ```

4. Re-run Step 6 smoke test. If it still fails, escalate — this is a deeper Unsloth/GGUF issue that needs investigation.

---

## Versioning

Every time you retrain:
1. Bump `KIKI_VERSION` (`v1` → `v2` → …)
2. Upload the new GGUF under the new version key
3. Tell DevOps the new S3 URI
4. DevOps redeploys with the new `KIKI_GGUF_S3_URI` — the agent code doesn't change
5. Old versions stay in S3 for rollback (don't delete for at least 90 days)

---

## What you are NOT responsible for

- ECR images, Docker builds, ECS task definitions → DevOps
- The Loopper agent's `USE_KIKI_SLM` flag → Agent engineer
- Production monitoring and on-call → DevOps + Agent engineer
- Networking, IAM, security groups → DevOps

Your contract with the rest of the org is one line: "The GGUF at `s3://.../kiki-sft-vN.gguf` is a fine-tuned Kiki SLM that passes the smoke test."
