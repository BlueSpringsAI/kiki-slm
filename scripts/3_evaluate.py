#!/usr/bin/env python3
"""Script 3: Benchmark base Qwen3, fine-tuned SLM, GPT-4o, and Claude Sonnet.

Runs gold-labeled tickets through all four systems using the same Kiki system
prompt, then computes accuracy, F1, latency, cost, and LLM-judged response quality.

Systems:
  1. Base Qwen3-4B (before fine-tuning)
  2. Fine-tuned SLM (after fine-tuning, LoRA adapter)
  3. GPT-4o (via OpenAI API)
  4. Claude Sonnet (via Anthropic API)

Usage:
    python scripts/3_evaluate.py --model-path outputs/adapters/kiki-sft-v1
    python scripts/3_evaluate.py --skip-apis         # Local models only
    python scripts/3_evaluate.py --skip-base          # Skip base model
    python scripts/3_evaluate.py --skip-slm           # Skip fine-tuned model
"""

import argparse
import asyncio
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


# ── Kiki system prompt (same for ALL systems) ────────────────────────────────

KIKI_SYSTEM_PROMPT = """\
You are Kiki, an AI customer service agent. When given a customer message, analyze it and respond with:
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

BASE_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"


# ── Config & gold data ──────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


GOLD_TEMPLATE = [
    {
        "ticket_id": "GOLD-001",
        "customer_message": "I placed order #12345 three days ago and it still shows processing. Can you check the status?",
        "gold_intent": "order_status",
        "gold_urgency": "medium",
        "gold_workflow_steps": ["Look up order #12345", "Check current fulfillment status", "Provide estimated delivery date"],
        "gold_tools_required": ["order_lookup(order_id='12345')", "shipment_tracker(order_id='12345')"],
    },
    {
        "ticket_id": "GOLD-002",
        "customer_message": "I was charged twice for my subscription this month. I need a refund for the duplicate charge immediately.",
        "gold_intent": "billing_inquiry",
        "gold_urgency": "high",
        "gold_workflow_steps": ["Verify duplicate charge in billing system", "Initiate refund for duplicate amount", "Send confirmation email"],
        "gold_tools_required": ["billing_lookup(customer_id='CID')", "refund_processor(charge_id='CHARGE_ID', reason='duplicate')"],
    },
    {
        "ticket_id": "GOLD-003",
        "customer_message": "My account has been locked and I cannot log in. I need access urgently for a work deadline.",
        "gold_intent": "account_management",
        "gold_urgency": "critical",
        "gold_workflow_steps": ["Verify customer identity", "Check account lock reason", "Unlock account", "Reset credentials if needed"],
        "gold_tools_required": ["identity_verifier(customer_id='CID')", "account_manager(action='unlock', customer_id='CID')"],
    },
    {
        "ticket_id": "GOLD-004",
        "customer_message": "I want to return the headphones I bought last week. They are not comfortable.",
        "gold_intent": "return_request",
        "gold_urgency": "low",
        "gold_workflow_steps": ["Check return eligibility window", "Generate return label", "Provide return instructions"],
        "gold_tools_required": ["order_lookup(order_id='ORDER_ID')", "return_processor(order_id='ORDER_ID', reason='comfort')"],
    },
    {
        "ticket_id": "GOLD-005",
        "customer_message": "Someone made unauthorized purchases on my account totaling $500. This is fraud!",
        "gold_intent": "fraud_report",
        "gold_urgency": "critical",
        "gold_workflow_steps": ["Flag account for fraud review", "Freeze account to prevent further charges", "Initiate chargeback process", "Escalate to fraud investigation team"],
        "gold_tools_required": ["fraud_detector(customer_id='CID')", "account_manager(action='freeze', customer_id='CID')", "chargeback_processor(customer_id='CID', amount='500')"],
    },
]


def load_gold_data(gold_file: str) -> list[dict]:
    gold_path = Path(gold_file)
    if not gold_path.exists():
        print(f"\n  Gold file not found: {gold_path}")
        print("  Creating template with 5 example entries...\n")
        gold_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gold_path, "w") as f:
            for entry in GOLD_TEMPLATE:
                f.write(json.dumps(entry) + "\n")
        print(f"  Template written to {gold_path}")
        print("  Fill in entries with: ticket_id, customer_message, gold_intent, gold_urgency, gold_workflow_steps, gold_tools_required\n")
        return GOLD_TEMPLATE

    tickets = []
    with open(gold_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                tickets.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: Skipping malformed line {line_num}: {e}")
    print(f"  Loaded {len(tickets)} gold-labeled tickets from {gold_path}")
    return tickets


# ── JSON parsing helper ─────────────────────────────────────────────────────

def parse_model_json(raw_text: str) -> dict | None:
    text = raw_text.strip()
    # Strip Qwen3 <think>...</think> reasoning tokens
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ── Local model loading ─────────────────────────────────────────────────────

def load_local_model(model_path: str, label: str, load_in_4bit: bool = True):
    """Load a local model (base or fine-tuned). Returns (model, tokenizer) or None."""
    # For base model, model_path is an HF ID like "Qwen/Qwen3-4B-Instruct-2507"
    # For fine-tuned, model_path is a local adapter directory
    is_hf_id = "/" in model_path and not Path(model_path).exists()

    if not is_hf_id:
        model_dir = Path(model_path)
        if not model_dir.exists():
            print(f"  WARNING: {label} model not found at {model_dir}, skipping")
            return None

    try:
        from unsloth import FastLanguageModel

        print(f"  Loading {label} from {model_path} (Unsloth, 4bit={load_in_4bit})...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        print(f"  {label} loaded successfully via Unsloth")
        return model, tokenizer
    except ImportError:
        pass

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print(f"  Loading {label} from {model_path} (transformers)...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        print(f"  {label} loaded successfully via transformers")
        return model, tokenizer
    except Exception as e:
        print(f"  WARNING: Failed to load {label}: {e}")
        return None


def run_local_inference(model_tokenizer: tuple, message: str) -> tuple[str, float]:
    """Run inference through a local model. Returns (response_text, latency_seconds)."""
    import torch

    model, tokenizer = model_tokenizer
    messages = [
        {"role": "system", "content": KIKI_SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
        )
    latency = time.perf_counter() - start

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response_text, latency


# ── API inference ───────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30), retry=retry_if_exception_type((Exception,)), reraise=True)
async def run_gpt4o_inference(client, message: str) -> tuple[str, float, dict]:
    start = time.perf_counter()
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": KIKI_SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    latency = time.perf_counter() - start
    text = response.choices[0].message.content or ""
    usage = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
    }
    return text, latency, usage


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30), retry=retry_if_exception_type((Exception,)), reraise=True)
async def run_claude_inference(client, message: str) -> tuple[str, float, dict]:
    start = time.perf_counter()
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=KIKI_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": message}],
        temperature=0.1,
    )
    latency = time.perf_counter() - start
    text = response.content[0].text if response.content else ""
    usage = {
        "input_tokens": response.usage.input_tokens if response.usage else 0,
        "output_tokens": response.usage.output_tokens if response.usage else 0,
    }
    return text, latency, usage


# ── Metrics ─────────────────────────────────────────────────────────────────

def intent_accuracy(predictions: list[str], gold_labels: list[str]) -> float:
    if not predictions:
        return 0.0
    return sum(1 for p, g in zip(predictions, gold_labels) if p == g) / len(predictions)


def intent_f1_micro(predictions: list[str], gold_labels: list[str]) -> float:
    try:
        from sklearn.metrics import f1_score
        return f1_score(gold_labels, predictions, average="micro", zero_division=0)
    except ImportError:
        return intent_accuracy(predictions, gold_labels)


def normalized_edit_distance(predicted: list[str], gold: list[str]) -> float:
    m, n = len(predicted), len(gold)
    if m == 0 and n == 0:
        return 1.0
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if predicted[i - 1].strip().lower() == gold[j - 1].strip().lower() else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return 1.0 - (dp[m][n] / max(m, n))


def workflow_accuracy(predicted_lists: list[list[str]], gold_lists: list[list[str]]) -> float:
    if not predicted_lists:
        return 0.0
    return statistics.mean(normalized_edit_distance(p, g) for p, g in zip(predicted_lists, gold_lists))


def tool_selection_f1(predicted_sets: list[set[str]], gold_sets: list[set[str]]) -> dict:
    precisions, recalls, f1s = [], [], []
    for pred, gold in zip(predicted_sets, gold_sets):
        pred_norm = {t.strip().lower() for t in pred}
        gold_norm = {t.strip().lower() for t in gold}
        if not pred_norm and not gold_norm:
            precisions.append(1.0); recalls.append(1.0); f1s.append(1.0)
            continue
        tp = len(pred_norm & gold_norm)
        p = tp / len(pred_norm) if pred_norm else 0.0
        r = tp / len(gold_norm) if gold_norm else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        precisions.append(p); recalls.append(r); f1s.append(f1)
    return {
        "precision": statistics.mean(precisions) if precisions else 0.0,
        "recall": statistics.mean(recalls) if recalls else 0.0,
        "f1": statistics.mean(f1s) if f1s else 0.0,
    }


def extract_tool_names(tools_list: list[str]) -> set[str]:
    return {t.split("(")[0].strip().lower() for t in tools_list if t.split("(")[0].strip()}


# ── LLM-as-Judge ────────────────────────────────────────────────────────────

JUDGE_PROMPT = """\
You are an expert evaluator of customer service responses.

Given a customer message and an AI agent's response, rate the response on four dimensions (1-5 each):
1. **Helpfulness**: Does it address the customer's needs?
2. **Correctness**: Is the information accurate?
3. **Professionalism**: Is the tone business-appropriate?
4. **Empathy**: Does it acknowledge the customer's feelings?

Respond ONLY with valid JSON:
{"helpfulness": <1-5>, "correctness": <1-5>, "professionalism": <1-5>, "empathy": <1-5>}"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30), retry=retry_if_exception_type((Exception,)), reraise=True)
async def judge_response(client, customer_message: str, agent_response: str) -> dict:
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"Customer message:\n{customer_message}\n\nAgent response:\n{agent_response}"},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    parsed = parse_model_json(response.choices[0].message.content or "{}")
    return parsed or {"helpfulness": 0, "correctness": 0, "professionalism": 0, "empathy": 0}


# ── Cost estimation ─────────────────────────────────────────────────────────

PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
}


def estimate_cost(system: str, usage: dict) -> float:
    pricing = PRICING.get(system, {"input": 0.0, "output": 0.0})
    input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


# ── Run inference for one system on one ticket ──────────────────────────────

def run_local_ticket(model_tokenizer, sys_key: str, msg: str, acc: dict, ticket_result: dict) -> None:
    """Run a local model (base or fine-tuned) on one ticket."""
    try:
        raw_text, latency = run_local_inference(model_tokenizer, msg)
        parsed = parse_model_json(raw_text)
        acc["latencies"].append(latency)
        acc["costs"].append(0.0)

        if parsed:
            acc["intent_preds"].append(parsed.get("intent", ""))
            acc["urgency_preds"].append(parsed.get("urgency", ""))
            acc["workflow_preds"].append(parsed.get("workflow_steps", []))
            acc["tool_preds"].append(parsed.get("tools_required", []))
        else:
            acc["intent_preds"].append("")
            acc["urgency_preds"].append("")
            acc["workflow_preds"].append([])
            acc["tool_preds"].append([])

        ticket_result[f"{sys_key}_raw"] = raw_text
        ticket_result[f"{sys_key}_parsed"] = parsed
        ticket_result[f"{sys_key}_latency"] = latency
    except Exception as e:
        print(f"    {sys_key} error: {e}")
        acc["intent_preds"].append("")
        acc["urgency_preds"].append("")
        acc["workflow_preds"].append([])
        acc["tool_preds"].append([])
        acc["latencies"].append(0.0)
        acc["costs"].append(0.0)
        ticket_result[f"{sys_key}_error"] = str(e)


async def run_api_ticket(
    run_fn, client, sys_key: str, pricing_key: str,
    msg: str, acc: dict, ticket_result: dict,
) -> None:
    """Run an API model (GPT-4o or Claude) on one ticket."""
    try:
        raw_text, latency, usage = await run_fn(client, msg)
        parsed = parse_model_json(raw_text)
        cost = estimate_cost(pricing_key, usage)
        acc["latencies"].append(latency)
        acc["costs"].append(cost)

        if parsed:
            acc["intent_preds"].append(parsed.get("intent", ""))
            acc["urgency_preds"].append(parsed.get("urgency", ""))
            acc["workflow_preds"].append(parsed.get("workflow_steps", []))
            acc["tool_preds"].append(parsed.get("tools_required", []))
        else:
            acc["intent_preds"].append("")
            acc["urgency_preds"].append("")
            acc["workflow_preds"].append([])
            acc["tool_preds"].append([])

        ticket_result[f"{sys_key}_raw"] = raw_text
        ticket_result[f"{sys_key}_parsed"] = parsed
        ticket_result[f"{sys_key}_latency"] = latency
        ticket_result[f"{sys_key}_cost"] = cost
    except Exception as e:
        print(f"    {sys_key} error: {e}")
        acc["intent_preds"].append("")
        acc["urgency_preds"].append("")
        acc["workflow_preds"].append([])
        acc["tool_preds"].append([])
        acc["latencies"].append(0.0)
        acc["costs"].append(0.0)
        ticket_result[f"{sys_key}_error"] = str(e)


# ── Main evaluation ─────────────────────────────────────────────────────────

async def evaluate_all(
    tickets: list[dict],
    config: dict,
    model_path: str,
    base_model: str,
    skip_base: bool,
    skip_slm: bool,
    skip_apis: bool,
) -> dict:

    # -- Initialize all systems --
    systems: dict[str, dict[str, Any]] = {}
    loaded_models: dict[str, tuple] = {}

    # 1. Base Qwen3 (before fine-tuning)
    if not skip_base:
        base_mt = load_local_model(base_model, "Base Qwen3")
        if base_mt:
            systems["base"] = {"name": f"Base Qwen3", "model": base_model}
            loaded_models["base"] = base_mt

    # 2. Fine-tuned SLM (after fine-tuning)
    if not skip_slm:
        slm_mt = load_local_model(model_path, "Fine-tuned SLM")
        if slm_mt:
            systems["slm"] = {"name": "Fine-tuned SLM", "model_path": model_path}
            loaded_models["slm"] = slm_mt

    # 3. GPT-4o
    openai_client = None
    if not skip_apis and os.environ.get("OPENAI_API_KEY"):
        from openai import AsyncOpenAI
        openai_client = AsyncOpenAI()
        systems["gpt4o"] = {"name": "GPT-4o", "model": "gpt-4o"}
    elif not skip_apis:
        print("  WARNING: OPENAI_API_KEY not set, skipping GPT-4o")

    # 4. Claude Sonnet
    anthropic_client = None
    if not skip_apis and os.environ.get("ANTHROPIC_API_KEY"):
        from anthropic import AsyncAnthropic
        anthropic_client = AsyncAnthropic()
        systems["claude"] = {"name": "Claude Sonnet", "model": "claude-sonnet-4-20250514"}
    elif not skip_apis:
        print("  WARNING: ANTHROPIC_API_KEY not set, skipping Claude")

    if not systems:
        print("\n  ERROR: No systems available. Check API keys / model paths.")
        return {}

    print(f"\n  Evaluating {len(tickets)} tickets across {len(systems)} systems:")
    for k, v in systems.items():
        print(f"    - {v['name']} ({k})")

    # -- Accumulators --
    results_by_system: dict[str, dict] = {
        k: {"intent_preds": [], "urgency_preds": [], "workflow_preds": [],
            "tool_preds": [], "latencies": [], "costs": [], "judge_scores": []}
        for k in systems
    }

    gold_intents = [t["gold_intent"] for t in tickets]
    gold_urgencies = [t["gold_urgency"] for t in tickets]
    gold_workflows = [t["gold_workflow_steps"] for t in tickets]
    gold_tools = [t["gold_tools_required"] for t in tickets]
    per_ticket_results: list[dict] = []

    # -- Evaluate each ticket --
    total = len(tickets)
    for idx, ticket in enumerate(tickets, 1):
        msg = ticket["customer_message"]
        ticket_id = ticket.get("ticket_id", f"ticket_{idx}")
        ticket_result = {"ticket_id": ticket_id, "customer_message": msg}

        if idx % 5 == 0 or idx == 1 or idx == total:
            print(f"  [{idx}/{total}] {ticket_id}...")

        # Base model
        if "base" in systems:
            run_local_ticket(loaded_models["base"], "base", msg, results_by_system["base"], ticket_result)

        # Fine-tuned SLM
        if "slm" in systems:
            run_local_ticket(loaded_models["slm"], "slm", msg, results_by_system["slm"], ticket_result)

        # GPT-4o
        if "gpt4o" in systems:
            await run_api_ticket(run_gpt4o_inference, openai_client, "gpt4o", "gpt-4o", msg, results_by_system["gpt4o"], ticket_result)

        # Claude
        if "claude" in systems:
            await run_api_ticket(run_claude_inference, anthropic_client, "claude", "claude-sonnet-4-20250514", msg, results_by_system["claude"], ticket_result)

        per_ticket_results.append(ticket_result)

    # -- Free local model VRAM --
    loaded_models.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    # -- LLM-as-Judge --
    if openai_client:
        print("\n  Running LLM-as-Judge scoring...")
        for sys_key in systems:
            sys_name = systems[sys_key]["name"]
            print(f"    Judging {sys_name}...")
            for i, tr in enumerate(per_ticket_results):
                parsed = tr.get(f"{sys_key}_parsed")
                if parsed and isinstance(parsed, dict) and parsed.get("response"):
                    try:
                        score = await judge_response(openai_client, tr["customer_message"], parsed["response"])
                    except Exception as e:
                        print(f"      Judge error: {e}")
                        score = {"helpfulness": 0, "correctness": 0, "professionalism": 0, "empathy": 0}
                    results_by_system[sys_key]["judge_scores"].append(score)
                    per_ticket_results[i][f"{sys_key}_judge"] = score
                else:
                    results_by_system[sys_key]["judge_scores"].append(None)
    else:
        print("\n  Skipping judge scoring (no OpenAI client)")

    # -- Compute aggregate metrics --
    print("\n  Computing aggregate metrics...")
    metrics_by_system = {}

    for sys_key, acc in results_by_system.items():
        i_acc = intent_accuracy(acc["intent_preds"], gold_intents)
        i_f1 = intent_f1_micro(acc["intent_preds"], gold_intents)
        u_acc = intent_accuracy(acc["urgency_preds"], gold_urgencies)
        w_acc = workflow_accuracy(acc["workflow_preds"], gold_workflows)
        pred_tool_sets = [extract_tool_names(t) for t in acc["tool_preds"]]
        gold_tool_sets = [extract_tool_names(t) for t in gold_tools]
        t_metrics = tool_selection_f1(pred_tool_sets, gold_tool_sets)

        valid_latencies = [l for l in acc["latencies"] if l > 0]
        lat_p50 = statistics.median(valid_latencies) if valid_latencies else 0.0
        lat_p95 = sorted(valid_latencies)[int(len(valid_latencies) * 0.95)] if len(valid_latencies) > 1 else (valid_latencies[0] if valid_latencies else 0.0)

        total_cost = sum(acc["costs"])
        valid_scores = [s for s in acc["judge_scores"] if s is not None]

        metrics_by_system[sys_key] = {
            "system_name": systems[sys_key]["name"],
            "intent_accuracy": round(i_acc, 4),
            "intent_f1_micro": round(i_f1, 4),
            "urgency_accuracy": round(u_acc, 4),
            "workflow_accuracy": round(w_acc, 4),
            "tool_precision": round(t_metrics["precision"], 4),
            "tool_recall": round(t_metrics["recall"], 4),
            "tool_f1": round(t_metrics["f1"], 4),
            "latency_p50_s": round(lat_p50, 3),
            "latency_p95_s": round(lat_p95, 3),
            "total_cost_usd": round(total_cost, 6),
            "cost_per_ticket_usd": round(total_cost / len(tickets), 6) if tickets else 0.0,
            "judge_helpfulness": round(statistics.mean([s.get("helpfulness", 0) for s in valid_scores]), 2) if valid_scores else 0.0,
            "judge_correctness": round(statistics.mean([s.get("correctness", 0) for s in valid_scores]), 2) if valid_scores else 0.0,
            "judge_professionalism": round(statistics.mean([s.get("professionalism", 0) for s in valid_scores]), 2) if valid_scores else 0.0,
            "judge_empathy": round(statistics.mean([s.get("empathy", 0) for s in valid_scores]), 2) if valid_scores else 0.0,
            "num_tickets": len(tickets),
            "num_parse_failures": sum(1 for p in acc["intent_preds"] if p == ""),
        }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "gold_file": str(config.get("_gold_file", "")),
            "model_path": str(config.get("_model_path", "")),
            "base_model": base_model,
            "num_tickets": len(tickets),
        },
        "metrics_by_system": metrics_by_system,
        "per_ticket_results": per_ticket_results,
    }


# ── Output ──────────────────────────────────────────────────────────────────

def print_comparison_table(results: dict):
    metrics = results.get("metrics_by_system", {})
    if not metrics:
        print("\nNo metrics to display.")
        return

    # Order: base, slm, gpt4o, claude
    order = ["base", "slm", "gpt4o", "claude"]
    sys_keys = [k for k in order if k in metrics]
    sys_names = [metrics[k]["system_name"] for k in sys_keys]

    label_w = 26
    col_w = 18
    header_line = "=" * (label_w + col_w * len(sys_keys) + 4)

    print(f"\n{header_line}")
    print("  KIKI SLM EVALUATION — 4-WAY COMPARISON")
    print(f"  {results.get('timestamp', '')}")
    print(header_line)

    header = f"  {'Metric':<{label_w}}"
    for name in sys_names:
        header += f"{name:>{col_w}}"
    print(header)
    print("  " + "-" * (label_w + col_w * len(sys_keys)))

    rows = [
        ("Intent Accuracy", "intent_accuracy", "{:.1%}"),
        ("Intent F1 (micro)", "intent_f1_micro", "{:.1%}"),
        ("Urgency Accuracy", "urgency_accuracy", "{:.1%}"),
        ("Workflow Accuracy", "workflow_accuracy", "{:.1%}"),
        ("Tool Precision", "tool_precision", "{:.1%}"),
        ("Tool Recall", "tool_recall", "{:.1%}"),
        ("Tool F1", "tool_f1", "{:.1%}"),
        ("", None, None),
        ("Latency p50 (s)", "latency_p50_s", "{:.3f}"),
        ("Latency p95 (s)", "latency_p95_s", "{:.3f}"),
        ("Total Cost (USD)", "total_cost_usd", "${:.4f}"),
        ("Cost/Ticket (USD)", "cost_per_ticket_usd", "${:.6f}"),
        ("", None, None),
        ("Judge: Helpfulness", "judge_helpfulness", "{:.2f}/5"),
        ("Judge: Correctness", "judge_correctness", "{:.2f}/5"),
        ("Judge: Professionalism", "judge_professionalism", "{:.2f}/5"),
        ("Judge: Empathy", "judge_empathy", "{:.2f}/5"),
        ("", None, None),
        ("Parse Failures", "num_parse_failures", "{}"),
        ("Tickets Evaluated", "num_tickets", "{}"),
    ]

    for label, key, fmt in rows:
        if key is None:
            print(f"  {'':─<{label_w}}" + "".join(f"{'':─>{col_w}}" for _ in sys_keys))
            continue
        row = f"  {label:<{label_w}}"
        for sk in sys_keys:
            val = metrics[sk].get(key, 0)
            row += f"{fmt.format(val):>{col_w}}"
        print(row)

    print(header_line)

    # Print improvement summary (base vs fine-tuned)
    if "base" in metrics and "slm" in metrics:
        print("\n  Fine-tuning improvement (SLM vs Base):")
        for key, label in [
            ("intent_accuracy", "Intent Accuracy"),
            ("urgency_accuracy", "Urgency Accuracy"),
            ("workflow_accuracy", "Workflow Accuracy"),
            ("tool_f1", "Tool F1"),
        ]:
            base_val = metrics["base"].get(key, 0)
            slm_val = metrics["slm"].get(key, 0)
            diff = slm_val - base_val
            arrow = "+" if diff >= 0 else ""
            print(f"    {label:<24s} {base_val:.1%} -> {slm_val:.1%} ({arrow}{diff:.1%})")
    print()


def save_results(results: dict, output_path: str = "outputs/results/evaluation_results.json"):
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {out_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark Base Qwen3, Fine-tuned SLM, GPT-4o, and Claude Sonnet")
    parser.add_argument("--config", default="configs/poc_config.yaml", help="Config file")
    parser.add_argument("--gold-file", default="data/gold/gold_100.jsonl", help="Gold-labeled test set")
    parser.add_argument("--model-path", default="outputs/adapters/kiki-sft-v1", help="Fine-tuned adapter path")
    parser.add_argument("--base-model", default=BASE_MODEL_NAME, help=f"Base model (default: {BASE_MODEL_NAME})")
    parser.add_argument("--output", default="outputs/results/evaluation_results.json", help="Output JSON path")
    parser.add_argument("--skip-base", action="store_true", help="Skip base model evaluation")
    parser.add_argument("--skip-slm", action="store_true", help="Skip fine-tuned SLM evaluation")
    parser.add_argument("--skip-apis", action="store_true", help="Skip API calls (GPT-4o, Claude)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  KIKI SLM — 4-WAY EVALUATION BENCHMARK")
    print("  Base Qwen3 | Fine-tuned SLM | GPT-4o | Claude Sonnet")
    print("=" * 60)

    print(f"\n[1/5] Loading config from {args.config}")
    config = load_config(args.config)
    config["_gold_file"] = args.gold_file
    config["_model_path"] = args.model_path

    print(f"\n[2/5] Loading gold data from {args.gold_file}")
    tickets = load_gold_data(args.gold_file)
    if not tickets:
        print("  ERROR: No gold tickets loaded. Exiting.")
        return

    print(f"\n[3/5] Running evaluation")
    results = asyncio.run(evaluate_all(
        tickets=tickets,
        config=config,
        model_path=args.model_path,
        base_model=args.base_model,
        skip_base=args.skip_base,
        skip_slm=args.skip_slm,
        skip_apis=args.skip_apis,
    ))

    if not results:
        return

    print(f"\n[4/5] Saving results")
    save_results(results, args.output)

    print(f"\n[5/5] Evaluation complete")
    print_comparison_table(results)


if __name__ == "__main__":
    main()
