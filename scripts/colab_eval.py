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

def load_config(path: str) -> dict:
    if os.path.exists(path):
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiki SLM Base vs Fine-tuned Evaluation")
    parser.add_argument("--config", default="configs/colab_config.yaml", help="Config YAML path")
    # CLI overrides
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--gold-file", default=None)
    parser.add_argument("--train-file", default=None, help="Training JSONL to extract system prompt + tools from")
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--max-turns", type=int, default=DEFAULT_MAX_TURNS,
        help="Max generation turns per example (tool_call loop). Default: 4.",
    )
    parser.add_argument(
        "--save-trajectory", action="store_true",
        help="Include full multi-turn trajectory in the saved JSON (large).",
    )

    args = parser.parse_args()

    # Merge with config file
    cfg = load_config(args.config)
    args.adapter_path = args.adapter_path or cfg.get("output", {}).get("adapter_dir", "")
    args.base_model = args.base_model or cfg.get("model", {}).get("name", "Qwen/Qwen3-4B-Instruct-2507")
    args.gold_file = args.gold_file or cfg.get("data", {}).get("gold_file", "")
    args.max_seq_length = args.max_seq_length or cfg.get("model", {}).get("max_seq_length", 2048)
    args.batch_size = args.batch_size or cfg.get("eval", {}).get("batch_size", 0)

    # Require adapter-path and gold-file
    if not args.adapter_path:
        parser.error("--adapter-path is required (or set output.adapter_dir in config)")
    if not args.gold_file:
        parser.error("--gold-file is required (or set data.gold_file in config)")

    return args


# ---------------------------------------------------------------------------
# Constants — loaded from training data if available
# ---------------------------------------------------------------------------

# Will be populated from training data in main()
EVAL_SYSTEM_PROMPT = ""
EVAL_TOOLS = None  # Tool definitions from training data


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
# Output parsing
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _extract_balanced_json(text: str) -> dict | None:
    """Find the first `{...}` with balanced braces and parse it."""
    first = text.find("{")
    if first == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(first, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[first:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def parse_model_json(raw_text: str) -> dict | None:
    """Strip <think>/<tool_call> blocks and extract the final JSON response."""
    text = raw_text.strip()
    text = _THINK_RE.sub("", text).strip()
    text = _TOOL_CALL_RE.sub("", text).strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _extract_balanced_json(text)


def parse_assistant_output(raw_text: str) -> dict:
    """Parse raw model output into an assistant message dict.

    Returns a dict with keys: role, content, reasoning_content, tool_calls.
    This matches the training-data format so we can feed it back to the
    chat template as conversation history for multi-turn generation.
    """
    text = raw_text.strip()

    # Extract <think>...</think> as reasoning_content
    think_match = _THINK_RE.search(text)
    reasoning = think_match.group(1).strip() if think_match else ""
    text_no_think = _THINK_RE.sub("", text).strip()

    # Extract <tool_call>...</tool_call> blocks
    tool_calls: list[dict] = []
    for idx, tc_body in enumerate(_TOOL_CALL_RE.findall(text_no_think)):
        try:
            tc = json.loads(tc_body.strip())
        except json.JSONDecodeError:
            continue
        name = tc.get("name") or tc.get("function", {}).get("name", "")
        args = tc.get("arguments", tc.get("function", {}).get("arguments", {}))
        if isinstance(args, dict):
            args_str = json.dumps(args, ensure_ascii=False)
        else:
            args_str = str(args)
        tool_calls.append({
            "type": "function",
            "id": f"call_eval_{idx}",
            "function": {"name": name, "arguments": args_str},
        })

    # Content = whatever remains after stripping think + tool_call blocks
    content = _TOOL_CALL_RE.sub("", text_no_think).strip()

    msg: dict = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning_content"] = reasoning
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


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
# Multi-turn inference (simulates rag_search tool loop)
# ---------------------------------------------------------------------------

# Max generations per example before giving up. Most training traces have
# 1-3 tool calls before the final JSON, so 4 turns is plenty.
DEFAULT_MAX_TURNS = 4
DEFAULT_MAX_NEW_TOKENS = 1024


def _apply_template(tokenizer, chat_msgs: list[dict]) -> str:
    """Apply chat template with tools, falling back gracefully if tools param rejected."""
    try:
        return tokenizer.apply_chat_template(
            chat_msgs,
            tools=EVAL_TOOLS,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return tokenizer.apply_chat_template(
            chat_msgs, tokenize=False, add_generation_prompt=True,
        )


def _generate_batch(
    model, tokenizer, prompts: list[str], max_new_tokens: int,
) -> tuple[list[str], list[float]]:
    """Generate continuations for a list of prompts. Returns (raw_outputs, per_example_latencies)."""
    if len(prompts) == 1:
        inputs = tokenizer(prompts[0], return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - start
        raw = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        return [raw], [latency]

    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    total_latency = time.perf_counter() - start
    per_latency = total_latency / len(prompts)
    raws = [
        tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True)
        for i in range(len(prompts))
    ]
    tokenizer.padding_side = "right"
    return raws, [per_latency] * len(prompts)


def run_multiturn_inference(
    model,
    tokenizer,
    messages: list[str],
    batch_size: int,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> list[dict]:
    """Run multi-turn generation: feeds empty tool results back when the model emits <tool_call>.

    For each example the loop runs until the model emits a turn with NO tool_calls
    (final JSON) or until max_turns is reached. Within each turn active examples
    are batched together for throughput.

    Returns a list of dicts, one per input message:
        {"parsed": dict|None, "final_raw": str, "total_raw": str,
         "latency": float, "turns": int, "tool_call_names": list[str]}
    """
    # Per-example state
    states: list[dict] = []
    for msg in messages:
        states.append({
            "chat_msgs": [
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": msg},
            ],
            "done": False,
            "final_raw": "",
            "total_raw_parts": [],
            "latency": 0.0,
            "turns": 0,
            "tool_call_names": [],
        })

    for turn in range(max_turns):
        active = [i for i, s in enumerate(states) if not s["done"]]
        if not active:
            break

        # Batch active examples
        for bstart in range(0, len(active), batch_size):
            batch_ids = active[bstart:bstart + batch_size]
            prompts = [_apply_template(tokenizer, states[i]["chat_msgs"]) for i in batch_ids]
            raws, lats = _generate_batch(model, tokenizer, prompts, max_new_tokens)

            for j, i in enumerate(batch_ids):
                raw = raws[j]
                s = states[i]
                s["final_raw"] = raw
                s["total_raw_parts"].append(f"--- turn {turn + 1} ---\n{raw}")
                s["latency"] += lats[j]
                s["turns"] += 1

                assistant_msg = parse_assistant_output(raw)
                tcs = assistant_msg.get("tool_calls") or []

                if tcs:
                    # Record tool call names for diagnostics
                    for tc in tcs:
                        s["tool_call_names"].append(tc["function"]["name"])
                    # Append assistant turn + synthetic empty tool results
                    s["chat_msgs"].append(assistant_msg)
                    for tc in tcs:
                        s["chat_msgs"].append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": tc["function"]["name"],
                            "content": json.dumps({"results": []}),
                        })
                else:
                    # No tool calls → this is the final response
                    s["done"] = True

    results: list[dict] = []
    for s in states:
        parsed = parse_model_json(s["final_raw"])
        results.append({
            "parsed": parsed,
            "final_raw": s["final_raw"],
            "total_raw": "\n".join(s["total_raw_parts"]),
            "latency": s["latency"],
            "turns": s["turns"],
            "tool_call_names": s["tool_call_names"],
        })
    return results


def evaluate_model(
    model, tokenizer, tickets: list[dict], label: str, batch_size: int,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> list[dict]:
    """Run a model on all gold tickets with multi-turn tool-call simulation."""
    messages = [t["customer_message"] for t in tickets]
    total = len(tickets)

    print(f"    Running {label} inference (batch_size={batch_size}, max_turns={max_turns})...")
    start = time.perf_counter()
    infer_results = run_multiturn_inference(
        model, tokenizer, messages, batch_size, max_turns=max_turns,
    )
    total_time = time.perf_counter() - start

    avg_turns = sum(r["turns"] for r in infer_results) / max(1, len(infer_results))
    print(
        f"    Completed {total} tickets in {total_time:.1f}s "
        f"({total_time / total:.2f}s/ticket avg, {avg_turns:.2f} turns avg)"
    )

    results = []
    for i, r in enumerate(infer_results):
        tid = tickets[i].get("ticket_id", f"ticket_{i + 1}")
        results.append({
            "ticket_id": tid,
            "parsed": r["parsed"],
            "raw": r["final_raw"],
            "trajectory": r["total_raw"],
            "latency": r["latency"],
            "turns": r["turns"],
            "tool_call_names": r["tool_call_names"],
        })
    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _norm(v) -> str:
    """Normalize a value for comparison (handles None → empty string)."""
    if v is None:
        return ""
    return str(v).strip().lower()


def _set_f1(pred: list[str], gold: list[str]) -> float:
    """F1 between two sets of strings (lowercased)."""
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    p = {s.strip().lower() for s in pred if s}
    g = {s.strip().lower() for s in gold if s}
    tp = len(p & g)
    if tp == 0:
        return 0.0
    precision = tp / len(p)
    recall = tp / len(g)
    return 2 * precision * recall / (precision + recall)


def _extract_pred_tool_names(res: dict) -> list[str]:
    """Collect tool call names the model actually invoked during the turn loop."""
    return [n for n in (res.get("tool_call_names") or []) if n]


def _extract_gold_tool_names(ticket: dict) -> list[str]:
    """Pull tool names from gold_tool_calls (list of {name, collection, query})."""
    out = []
    for tc in ticket.get("gold_tool_calls", []) or []:
        name = tc.get("name") if isinstance(tc, dict) else None
        if name:
            out.append(name)
    return out


def compute_metrics(results: list[dict], tickets: list[dict]) -> dict:
    n = len(results)
    intent_correct = 0
    urgency_correct = 0
    is_valid_correct = 0
    rejection_correct = 0
    rejection_evaluated = 0
    resolution_correct = 0
    team_correct = 0
    json_valid = 0
    tool_f1_scores: list[float] = []
    latencies: list[float] = []
    turns_list: list[int] = []

    for res, ticket in zip(results, tickets):
        latencies.append(res.get("latency", 0.0))
        turns_list.append(res.get("turns", 0))

        # Tool-name F1 is computed from the model's actual tool calls — works
        # even when final JSON fails to parse.
        gold_tools = _extract_gold_tool_names(ticket)
        pred_tools = _extract_pred_tool_names(res)
        if gold_tools or pred_tools:
            tool_f1_scores.append(_set_f1(pred_tools, gold_tools))

        p = res.get("parsed")
        if p is None:
            continue
        json_valid += 1

        if _norm(p.get("intent")) == _norm(ticket.get("gold_intent")):
            intent_correct += 1
        if _norm(p.get("urgency")) == _norm(ticket.get("gold_urgency")):
            urgency_correct += 1
        if p.get("is_valid") == ticket.get("gold_is_valid"):
            is_valid_correct += 1
        if _norm(p.get("resolution_type")) == _norm(ticket.get("gold_resolution_type")):
            resolution_correct += 1
        if _norm(p.get("team")) == _norm(ticket.get("gold_team")):
            team_correct += 1

        # rejection_type only matters for is_valid=false rows
        if ticket.get("gold_is_valid") is False:
            rejection_evaluated += 1
            if _norm(p.get("rejection_type")) == _norm(ticket.get("gold_rejection_type")):
                rejection_correct += 1

    return {
        "intent_accuracy": intent_correct / n if n else 0.0,
        "urgency_accuracy": urgency_correct / n if n else 0.0,
        "is_valid_accuracy": is_valid_correct / n if n else 0.0,
        "resolution_accuracy": resolution_correct / n if n else 0.0,
        "team_accuracy": team_correct / n if n else 0.0,
        "rejection_accuracy": (
            rejection_correct / rejection_evaluated if rejection_evaluated else 0.0
        ),
        "tool_name_f1": (
            sum(tool_f1_scores) / len(tool_f1_scores) if tool_f1_scores else 0.0
        ),
        "json_parse_rate": json_valid / n if n else 0.0,
        "avg_latency_s": sum(latencies) / len(latencies) if latencies else 0.0,
        "avg_turns": sum(turns_list) / len(turns_list) if turns_list else 0.0,
        "total": n,
        "rejection_evaluated": rejection_evaluated,
        "tools_evaluated": len(tool_f1_scores),
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

    metric_rows = [
        ("intent_accuracy", "Intent Accuracy", "{:.1%}"),
        ("urgency_accuracy", "Urgency Accuracy", "{:.1%}"),
        ("is_valid_accuracy", "is_valid Accuracy", "{:.1%}"),
        ("rejection_accuracy", "Rejection-type Acc", "{:.1%}"),
        ("resolution_accuracy", "Resolution-type Acc", "{:.1%}"),
        ("team_accuracy", "Team Accuracy", "{:.1%}"),
        ("tool_name_f1", "Tool-name F1", "{:.1%}"),
        ("json_parse_rate", "JSON Parse Rate", "{:.1%}"),
        ("avg_latency_s", "Avg Latency (s)", "{:.2f}"),
        ("avg_turns", "Avg Turns", "{:.2f}"),
    ]
    for key, label, fmt in metric_rows:
        bv = fmt.format(base_metrics.get(key, 0))
        fv = fmt.format(ft_metrics.get(key, 0))
        print(f"  {label:<28s} {bv:>15s} {fv:>15s}")

    print(f"{'='*65}")
    for key, label in [
        ("intent_accuracy", "Intent"),
        ("urgency_accuracy", "Urgency"),
        ("is_valid_accuracy", "is_valid"),
        ("resolution_accuracy", "Resolution"),
        ("team_accuracy", "Team"),
        ("tool_name_f1", "Tool F1"),
        ("json_parse_rate", "JSON Parse"),
    ]:
        diff = ft_metrics.get(key, 0) - base_metrics.get(key, 0)
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
        b_ok = "Y" if _norm(b_intent) == _norm(gold) else "N"
        f_ok = "Y" if _norm(f_intent) == _norm(gold) else "N"
        print(f"\n  {tid} | Gold: {gold}")
        print(
            f"    Base:       {b_intent} [{b_ok}] "
            f"({b['latency']:.2f}s, {b.get('turns', 0)} turns, tools={b.get('tool_call_names', [])})"
        )
        print(
            f"    Fine-tuned: {f_intent} [{f_ok}] "
            f"({f['latency']:.2f}s, {f.get('turns', 0)} turns, tools={f.get('tool_call_names', [])})"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_training_context(train_file: str | None) -> None:
    """Load system prompt and tools from the training data so eval matches training format."""
    global EVAL_SYSTEM_PROMPT, EVAL_TOOLS

    if not train_file or not os.path.exists(train_file):
        print("  WARNING: train file not found — using generic system prompt (results may be poor)")
        EVAL_SYSTEM_PROMPT = (
            "You are a customer service agent. Analyze the customer message "
            "and respond with valid JSON containing: intent, urgency, confidence, "
            "is_valid, rejection_type, resolution_type, team, actions, summary, reasoning, response."
        )
        EVAL_TOOLS = None
        return

    # Read first training example to extract system prompt + tools
    with open(train_file) as f:
        first_line = f.readline().strip()
        if not first_line:
            return
        ex = json.loads(first_line)

    # Extract system prompt
    for m in ex.get("messages", []):
        if m.get("role") == "system":
            EVAL_SYSTEM_PROMPT = m.get("content", "")
            break

    # Extract tools
    EVAL_TOOLS = ex.get("tools")

    prompt_preview = EVAL_SYSTEM_PROMPT[:80] + "..." if len(EVAL_SYSTEM_PROMPT) > 80 else EVAL_SYSTEM_PROMPT
    print(f"  System prompt: {prompt_preview}")
    print(f"  Tools: {[t['function']['name'] for t in EVAL_TOOLS] if EVAL_TOOLS else 'none'}")


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  KIKI SLM — BASE vs FINE-TUNED EVALUATION")
    print(f"{'='*60}")

    tickets = load_gold_data(args.gold_file)

    # Load system prompt + tools from training data
    train_file = args.train_file
    if not train_file or not os.path.exists(train_file or ""):
        cfg = load_config(args.config) if os.path.exists(args.config) else {}
        train_file = cfg.get("data", {}).get("train_file", "")
    if not os.path.exists(train_file or ""):
        for candidate in [
            "/content/drive/MyDrive/kiki-slm/data/sft-data/train_trimmed.jsonl",
            "/content/drive/MyDrive/kiki-slm/data/sft-data/train.jsonl",
        ]:
            if os.path.exists(candidate):
                train_file = candidate
                break
    load_training_context(train_file)

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

    base_results = evaluate_model(
        base_model, base_tokenizer, tickets, "Base", batch_size, max_turns=args.max_turns,
    )

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

    ft_results = evaluate_model(
        ft_model, ft_tokenizer, tickets, "Fine-tuned", batch_size, max_turns=args.max_turns,
    )

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

    def _serialize(results: list[dict]) -> list[dict]:
        skip = {"trajectory"} if not args.save_trajectory else set()
        return [{k: v for k, v in r.items() if k not in skip} for r in results]

    output = {
        "config": {
            "adapter_path": args.adapter_path,
            "base_model": args.base_model,
            "gold_file": args.gold_file,
            "batch_size": batch_size,
            "max_turns": args.max_turns,
        },
        "base_metrics": base_metrics,
        "ft_metrics": ft_metrics,
        "base_results": _serialize(base_results),
        "ft_results": _serialize(ft_results),
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {output_file}")

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
