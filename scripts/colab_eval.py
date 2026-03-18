#!/usr/bin/env python3
"""Kiki SLM evaluation — base Qwen3 vs fine-tuned on gold test data.

Loads models sequentially (base first, then fine-tuned) to avoid OOM on T4.
Strips Qwen3 <think> tokens before JSON parsing.
Batched inference for faster evaluation on high-VRAM GPUs.

Self-contained — does NOT import from the kiki package.

Usage:
    python -u scripts/colab_eval.py \
        --adapter-path /content/drive/MyDrive/kiki-slm/adapters/kiki-sft-v1 \
        --gold-file /content/drive/MyDrive/kiki-slm/data/gold_100.jsonl \
        --output-file /content/drive/MyDrive/kiki-slm/adapters/eval_results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import time
import warnings
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

# Suppress the transformers FutureWarning logging bug
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)

import torch


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiki SLM Base vs Fine-tuned Evaluation")
    parser.add_argument("--adapter-path", required=True, help="Path to fine-tuned LoRA adapter")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507", help="Base model ID")
    parser.add_argument("--gold-file", required=True, help="Path to gold JSONL")
    parser.add_argument("--output-file", default=None, help="Where to save results JSON")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size for inference (0=auto-detect)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVAL_SYSTEM_PROMPT = (
    "You are Kiki, an AI customer service agent. Analyze the customer message "
    "and respond with valid JSON containing: intent, urgency, workflow_steps, "
    "tools_required, reasoning, response."
)

GOLD_TEMPLATE = [
    {"ticket_id": "GOLD-001", "customer_message": "I placed order #12345 three days ago and it still shows processing.", "gold_intent": "order_status", "gold_urgency": "medium"},
    {"ticket_id": "GOLD-002", "customer_message": "I was charged twice for my subscription this month. I need a refund.", "gold_intent": "billing_inquiry", "gold_urgency": "high"},
    {"ticket_id": "GOLD-003", "customer_message": "My account has been locked and I cannot log in.", "gold_intent": "account_management", "gold_urgency": "critical"},
    {"ticket_id": "GOLD-004", "customer_message": "I want to return the headphones I bought last week.", "gold_intent": "return_request", "gold_urgency": "low"},
    {"ticket_id": "GOLD-005", "customer_message": "Someone made unauthorized purchases on my account totaling $500!", "gold_intent": "fraud_report", "gold_urgency": "critical"},
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gold_data(path: str) -> list[dict]:
    if not os.path.exists(path):
        print(f"  WARNING: Gold file not found: {path}")
        print(f"  Using built-in 5-example template")
        return list(GOLD_TEMPLATE)

    tickets = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tickets.append(json.loads(line))
    print(f"  Loaded {len(tickets)} gold tickets from {path}")
    return tickets


# ---------------------------------------------------------------------------
# JSON parsing (strips <think> tokens)
# ---------------------------------------------------------------------------

def parse_model_json(raw_text: str) -> dict | None:
    text = raw_text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def auto_detect_batch_size() -> int:
    """Auto-detect inference batch size based on GPU VRAM."""
    if not torch.cuda.is_available():
        return 1
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if vram_gb >= 70:   # H100 80GB
        return 16
    if vram_gb >= 35:   # A100 40/80GB
        return 8
    if vram_gb >= 20:   # L4
        return 4
    return 1             # T4 or smaller — sequential


# ---------------------------------------------------------------------------
# Batched inference
# ---------------------------------------------------------------------------

def run_batched_inference(
    model, tokenizer, messages: list[str], batch_size: int,
) -> list[tuple[dict | None, str, float]]:
    """Run inference in batches for faster evaluation on high-VRAM GPUs."""
    results = []

    for batch_start in range(0, len(messages), batch_size):
        batch_msgs = messages[batch_start:batch_start + batch_size]

        # Build prompts
        prompts = []
        for msg in batch_msgs:
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": EVAL_SYSTEM_PROMPT},
                 {"role": "user", "content": msg}],
                tokenize=False, add_generation_prompt=True,
            )
            prompts.append(prompt)

        if batch_size == 1 or len(prompts) == 1:
            # Sequential (T4 or single item) — no padding needed
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                input_len = inputs["input_ids"].shape[1]

                start = time.perf_counter()
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True)
                latency = time.perf_counter() - start

                raw = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
                parsed = parse_model_json(raw)
                results.append((parsed, raw, latency))
        else:
            # Batched — pad and generate together
            tokenizer.padding_side = "left"
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                              max_length=2048).to(model.device)
            input_lens = [inputs["attention_mask"][i].sum().item() for i in range(len(prompts))]

            start = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True)
            total_latency = time.perf_counter() - start
            per_latency = total_latency / len(prompts)

            for i in range(len(prompts)):
                raw = tokenizer.decode(outputs[i][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                parsed = parse_model_json(raw)
                results.append((parsed, raw, per_latency))

            tokenizer.padding_side = "right"

    return results


def evaluate_model(model, tokenizer, tickets: list[dict], label: str, batch_size: int) -> list[dict]:
    """Run a model on all gold tickets with batched inference."""
    messages = [t["customer_message"] for t in tickets]
    total = len(tickets)

    print(f"    Running {label} inference (batch_size={batch_size})...")
    start = time.perf_counter()
    infer_results = run_batched_inference(model, tokenizer, messages, batch_size)
    total_time = time.perf_counter() - start
    print(f"    Completed {total} tickets in {total_time:.1f}s ({total_time/total:.2f}s/ticket avg)")

    results = []
    for i, (parsed, raw, latency) in enumerate(infer_results):
        tid = tickets[i].get("ticket_id", f"ticket_{i+1}")
        results.append({"ticket_id": tid, "parsed": parsed, "raw": raw, "latency": latency})
    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict], tickets: list[dict]) -> dict:
    n = len(results)
    intent_correct = urgency_correct = json_valid = 0
    latencies = []

    for res, ticket in zip(results, tickets):
        p = res["parsed"]
        latencies.append(res["latency"])
        if p is not None:
            json_valid += 1
            pred_intent = p.get("intent", "").lower()
            gold_intent = ticket.get("gold_intent", "").lower()
            gold_secondary = ticket.get("gold_intent_secondary", "").lower()
            if pred_intent == gold_intent or (gold_secondary and pred_intent == gold_secondary):
                intent_correct += 1
            if p.get("urgency", "").lower() == ticket.get("gold_urgency", "").lower():
                urgency_correct += 1

    return {
        "intent_accuracy": intent_correct / n if n else 0,
        "urgency_accuracy": urgency_correct / n if n else 0,
        "json_parse_rate": json_valid / n if n else 0,
        "avg_latency_s": sum(latencies) / len(latencies) if latencies else 0,
        "total": n,
    }


# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------

def free_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_comparison(base_metrics: dict, ft_metrics: dict, base_results: list, ft_results: list, tickets: list):
    n = base_metrics["total"]
    print(f"\n{'='*65}")
    print(f"  BASE vs FINE-TUNED COMPARISON ({n} gold tickets)")
    print(f"{'='*65}")
    print(f"  {'Metric':<28s} {'Base Qwen3':>15s} {'Fine-tuned':>15s}")
    print(f"  {'-'*58}")

    for key, label, fmt in [
        ("intent_accuracy", "Intent Accuracy", "{:.1%}"),
        ("urgency_accuracy", "Urgency Accuracy", "{:.1%}"),
        ("json_parse_rate", "JSON Parse Rate", "{:.1%}"),
        ("avg_latency_s", "Avg Latency (s)", "{:.2f}"),
    ]:
        bv = fmt.format(base_metrics[key])
        fv = fmt.format(ft_metrics[key])
        print(f"  {label:<28s} {bv:>15s} {fv:>15s}")

    print(f"{'='*65}")

    for key, label in [("intent_accuracy", "Intent"), ("urgency_accuracy", "Urgency"), ("json_parse_rate", "JSON Parse")]:
        diff = ft_metrics[key] - base_metrics[key]
        arrow = "+" if diff >= 0 else ""
        print(f"  {label} improvement: {arrow}{diff:.1%}")

    print(f"\n{'='*65}")
    print(f"  PER-TICKET DETAIL (first 20)")
    print(f"{'='*65}")
    for i, ticket in enumerate(tickets[:20]):
        tid = ticket.get("ticket_id", f"ticket_{i + 1}")
        b = base_results[i]
        f = ft_results[i]
        b_intent = b["parsed"].get("intent", "PARSE_FAIL") if b["parsed"] else "PARSE_FAIL"
        f_intent = f["parsed"].get("intent", "PARSE_FAIL") if f["parsed"] else "PARSE_FAIL"
        gold = ticket.get("gold_intent", "?")
        b_ok = "Y" if b_intent.lower() == gold.lower() else "N"
        f_ok = "Y" if f_intent.lower() == gold.lower() else "N"
        print(f"\n  {tid} | Gold: {gold}")
        print(f"    Base:       {b_intent} [{b_ok}] ({b['latency']:.2f}s)")
        print(f"    Fine-tuned: {f_intent} [{f_ok}] ({f['latency']:.2f}s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  KIKI SLM — BASE vs FINE-TUNED EVALUATION")
    print(f"{'='*60}")

    tickets = load_gold_data(args.gold_file)

    batch_size = args.batch_size if args.batch_size > 0 else auto_detect_batch_size()
    print(f"  Inference batch size: {batch_size}")

    # --- Base model (load, evaluate, free) ---
    print(f"\n  Loading BASE model: {args.base_model}")
    from unsloth import FastLanguageModel

    base_model, base_tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(base_model)

    base_results = evaluate_model(base_model, base_tokenizer, tickets, "Base", batch_size)

    print(f"  Freeing base model VRAM...")
    free_model(base_model)
    del base_tokenizer
    gc.collect()

    # --- Fine-tuned model (load, evaluate, free) ---
    print(f"\n  Loading FINE-TUNED model: {args.adapter_path}")
    ft_model, ft_tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(ft_model)

    ft_results = evaluate_model(ft_model, ft_tokenizer, tickets, "Fine-tuned", batch_size)

    free_model(ft_model)
    del ft_tokenizer

    # --- Metrics & display ---
    base_metrics = compute_metrics(base_results, tickets)
    ft_metrics = compute_metrics(ft_results, tickets)

    print_comparison(base_metrics, ft_metrics, base_results, ft_results, tickets)

    # --- Save ---
    output_file = args.output_file
    if output_file is None:
        output_file = str(Path(args.adapter_path) / "eval_results.json")
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    output = {
        "base_metrics": base_metrics,
        "ft_metrics": ft_metrics,
        "base_results": [{k: v for k, v in r.items() if k != "raw"} for r in base_results],
        "ft_results": [{k: v for k, v in r.items() if k != "raw"} for r in ft_results],
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {output_file}")

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
