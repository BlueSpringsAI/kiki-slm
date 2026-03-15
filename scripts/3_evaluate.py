#!/usr/bin/env python3
"""Script 3: Benchmark fine-tuned SLM against GPT-4o and Claude Sonnet.

Runs each gold-labeled ticket through three systems (local SLM, GPT-4o,
Claude Sonnet) using the same Kiki system prompt, then computes accuracy,
F1, latency, cost, and LLM-judged response quality.

Usage:
    python scripts/3_evaluate.py                          # Full evaluation
    python scripts/3_evaluate.py --skip-slm               # Skip local model
    python scripts/3_evaluate.py --skip-apis               # Skip API calls
    python scripts/3_evaluate.py --gold-file data/gold/gold_100.jsonl
"""

import argparse
import asyncio
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


# ── Kiki system prompt (same as training) ────────────────────────────────────

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


# ── Config loading ───────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Gold data loading / template creation ────────────────────────────────────

GOLD_TEMPLATE = [
    {
        "ticket_id": "GOLD-001",
        "customer_message": "I placed order #12345 three days ago and it still shows processing. Can you check the status?",
        "gold_intent": "order_status",
        "gold_urgency": "medium",
        "gold_workflow_steps": [
            "Look up order #12345",
            "Check current fulfillment status",
            "Provide estimated delivery date",
        ],
        "gold_tools_required": [
            "order_lookup(order_id='12345')",
            "shipment_tracker(order_id='12345')",
        ],
    },
    {
        "ticket_id": "GOLD-002",
        "customer_message": "I was charged twice for my subscription this month. I need a refund for the duplicate charge immediately.",
        "gold_intent": "billing_inquiry",
        "gold_urgency": "high",
        "gold_workflow_steps": [
            "Verify duplicate charge in billing system",
            "Initiate refund for duplicate amount",
            "Send confirmation email",
        ],
        "gold_tools_required": [
            "billing_lookup(customer_id='CID')",
            "refund_processor(charge_id='CHARGE_ID', reason='duplicate')",
        ],
    },
    {
        "ticket_id": "GOLD-003",
        "customer_message": "My account has been locked and I cannot log in. I need access urgently for a work deadline.",
        "gold_intent": "account_management",
        "gold_urgency": "critical",
        "gold_workflow_steps": [
            "Verify customer identity",
            "Check account lock reason",
            "Unlock account",
            "Reset credentials if needed",
        ],
        "gold_tools_required": [
            "identity_verifier(customer_id='CID')",
            "account_manager(action='unlock', customer_id='CID')",
        ],
    },
    {
        "ticket_id": "GOLD-004",
        "customer_message": "I want to return the headphones I bought last week. They are not comfortable.",
        "gold_intent": "return_request",
        "gold_urgency": "low",
        "gold_workflow_steps": [
            "Check return eligibility window",
            "Generate return label",
            "Provide return instructions",
        ],
        "gold_tools_required": [
            "order_lookup(order_id='ORDER_ID')",
            "return_processor(order_id='ORDER_ID', reason='comfort')",
        ],
    },
    {
        "ticket_id": "GOLD-005",
        "customer_message": "Someone made unauthorized purchases on my account totaling $500. This is fraud!",
        "gold_intent": "fraud_report",
        "gold_urgency": "critical",
        "gold_workflow_steps": [
            "Flag account for fraud review",
            "Freeze account to prevent further charges",
            "Initiate chargeback process",
            "Escalate to fraud investigation team",
        ],
        "gold_tools_required": [
            "fraud_detector(customer_id='CID')",
            "account_manager(action='freeze', customer_id='CID')",
            "chargeback_processor(customer_id='CID', amount='500')",
        ],
    },
]


def load_gold_data(gold_file: str) -> list[dict]:
    """Load gold-labeled tickets from JSONL. Create template if missing."""
    gold_path = Path(gold_file)

    if not gold_path.exists():
        print(f"\n  Gold file not found: {gold_path}")
        print("  Creating template with 5 example entries...\n")
        gold_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gold_path, "w") as f:
            for entry in GOLD_TEMPLATE:
                f.write(json.dumps(entry) + "\n")
        print(f"  Template written to {gold_path}")
        print("  Please fill in all 100 entries following the same schema:")
        print("    - ticket_id: unique identifier")
        print("    - customer_message: the raw customer text")
        print("    - gold_intent: ground-truth intent label")
        print("    - gold_urgency: ground-truth urgency (critical/high/medium/low)")
        print("    - gold_workflow_steps: list of expected resolution steps")
        print("    - gold_tools_required: list of expected tool calls")
        print()
        # Return the template entries so the script can still run as a demo
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


# ── Model inference functions ────────────────────────────────────────────────

def parse_model_json(raw_text: str) -> dict | None:
    """Attempt to parse JSON from a model response, handling markdown fences."""
    text = raw_text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# -- Fine-tuned SLM inference -------------------------------------------------

def load_slm_model(model_path: str):
    """Load the fine-tuned SLM via transformers. Returns (model, tokenizer) or None."""
    model_dir = Path(model_path)
    if not model_dir.exists():
        print(f"  WARNING: SLM model not found at {model_dir}, skipping SLM evaluation")
        return None

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print(f"  Loading SLM from {model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        print("  SLM loaded successfully")
        return model, tokenizer
    except ImportError:
        print("  WARNING: transformers not installed, trying unsloth...")
        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(model_dir),
                max_seq_length=2048,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
            print("  SLM loaded via Unsloth")
            return model, tokenizer
        except ImportError:
            print("  WARNING: Neither transformers nor unsloth available. Skipping SLM.")
            return None
        except Exception as e:
            print(f"  WARNING: Failed to load SLM via Unsloth: {e}")
            return None
    except Exception as e:
        print(f"  WARNING: Failed to load SLM: {e}")
        return None


def run_slm_inference(model_tokenizer: tuple, message: str) -> tuple[str, float]:
    """Run inference through the local SLM. Returns (response_text, latency_seconds)."""
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

    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response_text, latency


# -- GPT-4o inference ----------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((Exception,)),
    reraise=True,
)
async def run_gpt4o_inference(
    client: AsyncOpenAI, message: str
) -> tuple[str, float, dict]:
    """Run inference through GPT-4o. Returns (response_text, latency, usage_dict)."""
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
        "total_tokens": response.usage.total_tokens if response.usage else 0,
    }
    return text, latency, usage


# -- Claude Sonnet inference ---------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((Exception,)),
    reraise=True,
)
async def run_claude_inference(
    client: AsyncAnthropic, message: str
) -> tuple[str, float, dict]:
    """Run inference through Claude Sonnet. Returns (response_text, latency, usage_dict)."""
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


# ── Metrics computation ──────────────────────────────────────────────────────

def intent_accuracy(predictions: list[str], gold_labels: list[str]) -> float:
    """Exact-match accuracy for intent classification."""
    if not predictions:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, gold_labels) if p == g)
    return correct / len(predictions)


def intent_f1_micro(predictions: list[str], gold_labels: list[str]) -> float:
    """Micro-averaged F1 for intent classification using sklearn."""
    try:
        from sklearn.metrics import f1_score

        return f1_score(gold_labels, predictions, average="micro", zero_division=0)
    except ImportError:
        # Fallback: micro-F1 equals accuracy when computed over all classes
        print("  WARNING: sklearn not available, using accuracy as F1 proxy")
        return intent_accuracy(predictions, gold_labels)


def urgency_accuracy(predictions: list[str], gold_labels: list[str]) -> float:
    """Exact-match accuracy for urgency classification."""
    if not predictions:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, gold_labels) if p == g)
    return correct / len(predictions)


def normalized_edit_distance(predicted: list[str], gold: list[str]) -> float:
    """Compute normalized Levenshtein edit distance between two step lists.

    Returns a similarity score in [0, 1] where 1.0 = perfect match.
    """
    m, n = len(predicted), len(gold)
    if m == 0 and n == 0:
        return 1.0
    if m == 0 or n == 0:
        return 0.0

    # Standard Levenshtein DP
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Normalize step strings for comparison
            cost = 0 if predicted[i - 1].strip().lower() == gold[j - 1].strip().lower() else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost, # substitution
            )

    edit_dist = dp[m][n]
    max_len = max(m, n)
    return 1.0 - (edit_dist / max_len)


def workflow_accuracy(predicted_lists: list[list[str]], gold_lists: list[list[str]]) -> float:
    """Average normalized edit distance similarity across all tickets."""
    if not predicted_lists:
        return 0.0
    scores = [
        normalized_edit_distance(p, g) for p, g in zip(predicted_lists, gold_lists)
    ]
    return statistics.mean(scores)


def tool_selection_f1(predicted_sets: list[set[str]], gold_sets: list[set[str]]) -> dict:
    """Compute set-based precision, recall, F1 for tool selection, averaged over tickets."""
    precisions = []
    recalls = []
    f1s = []

    for pred, gold in zip(predicted_sets, gold_sets):
        # Normalize tool names: lowercase, strip whitespace
        pred_norm = {t.strip().lower() for t in pred}
        gold_norm = {t.strip().lower() for t in gold}

        if not pred_norm and not gold_norm:
            precisions.append(1.0)
            recalls.append(1.0)
            f1s.append(1.0)
            continue

        tp = len(pred_norm & gold_norm)
        precision = tp / len(pred_norm) if pred_norm else 0.0
        recall = tp / len(gold_norm) if gold_norm else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "precision": statistics.mean(precisions) if precisions else 0.0,
        "recall": statistics.mean(recalls) if recalls else 0.0,
        "f1": statistics.mean(f1s) if f1s else 0.0,
    }


def extract_tool_names(tools_list: list[str]) -> set[str]:
    """Extract just the tool/function names from tool strings like 'order_lookup(order_id=...)'."""
    names = set()
    for tool_str in tools_list:
        # Take everything before the first '(' as the tool name
        name = tool_str.split("(")[0].strip().lower()
        if name:
            names.add(name)
    return names


# ── LLM-as-Judge (GPT-4o) ───────────────────────────────────────────────────

JUDGE_PROMPT = """\
You are an expert evaluator of customer service responses.

Given a customer message and an AI agent's response, rate the response on these four dimensions (1-5 scale each):

1. **Helpfulness**: Does the response address the customer's needs and move toward resolution?
2. **Correctness**: Is the information accurate and the proposed action appropriate?
3. **Professionalism**: Is the tone appropriate for a business context?
4. **Empathy**: Does the response acknowledge the customer's feelings and situation?

Respond ONLY with valid JSON:
{
  "helpfulness": <1-5>,
  "correctness": <1-5>,
  "professionalism": <1-5>,
  "empathy": <1-5>,
  "brief_justification": "<one sentence>"
}"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((Exception,)),
    reraise=True,
)
async def judge_response(
    client: AsyncOpenAI,
    customer_message: str,
    agent_response: str,
) -> dict:
    """Use GPT-4o to judge the quality of an agent response."""
    user_msg = (
        f"Customer message:\n{customer_message}\n\n"
        f"Agent response:\n{agent_response}"
    )
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    text = response.choices[0].message.content or "{}"
    parsed = parse_model_json(text)
    if parsed is None:
        return {
            "helpfulness": 0,
            "correctness": 0,
            "professionalism": 0,
            "empathy": 0,
            "brief_justification": "Failed to parse judge response",
        }
    return parsed


# ── Cost estimation ──────────────────────────────────────────────────────────

# Approximate pricing per 1M tokens (as of 2025)
PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "slm_local": {"input": 0.0, "output": 0.0},  # Self-hosted
}


def estimate_cost(system: str, usage: dict) -> float:
    """Estimate cost in USD from token usage."""
    pricing = PRICING.get(system, {"input": 0.0, "output": 0.0})
    input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
    cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
    return cost


# ── Main evaluation loop ────────────────────────────────────────────────────

async def evaluate_all(
    tickets: list[dict],
    config: dict,
    model_path: str,
    skip_slm: bool,
    skip_apis: bool,
) -> dict:
    """Run evaluation across all systems and compute metrics."""

    # -- Initialize clients & models --
    openai_client = None
    anthropic_client = None
    slm = None

    if not skip_apis:
        if os.environ.get("OPENAI_API_KEY"):
            openai_client = AsyncOpenAI()
        else:
            print("  WARNING: OPENAI_API_KEY not set, skipping GPT-4o evaluation")

        if os.environ.get("ANTHROPIC_API_KEY"):
            anthropic_client = AsyncAnthropic()
        else:
            print("  WARNING: ANTHROPIC_API_KEY not set, skipping Claude evaluation")

    if not skip_slm:
        slm = load_slm_model(model_path)

    systems = {}
    if slm is not None:
        systems["slm"] = {"name": "Fine-tuned SLM", "model_path": model_path}
    if openai_client is not None:
        systems["gpt4o"] = {"name": "GPT-4o", "model": "gpt-4o"}
    if anthropic_client is not None:
        systems["claude"] = {"name": "Claude Sonnet", "model": "claude-sonnet-4-20250514"}

    if not systems:
        print("\n  ERROR: No systems available for evaluation. Check API keys / model path.")
        return {}

    print(f"\n  Evaluating {len(tickets)} tickets across {len(systems)} system(s): "
          f"{', '.join(s['name'] for s in systems.values())}")

    # -- Per-system accumulators --
    results_by_system: dict[str, dict] = {}
    per_ticket_results: list[dict] = []

    for sys_key in systems:
        results_by_system[sys_key] = {
            "intent_preds": [],
            "urgency_preds": [],
            "workflow_preds": [],
            "tool_preds": [],
            "latencies": [],
            "costs": [],
            "judge_scores": [],
        }

    gold_intents = [t["gold_intent"] for t in tickets]
    gold_urgencies = [t["gold_urgency"] for t in tickets]
    gold_workflows = [t["gold_workflow_steps"] for t in tickets]
    gold_tools = [t["gold_tools_required"] for t in tickets]

    # -- Iterate over tickets --
    total = len(tickets)
    for idx, ticket in enumerate(tickets, 1):
        msg = ticket["customer_message"]
        ticket_id = ticket.get("ticket_id", f"ticket_{idx}")
        ticket_result = {"ticket_id": ticket_id, "customer_message": msg}

        if idx % 10 == 0 or idx == 1 or idx == total:
            print(f"  [{idx}/{total}] Processing {ticket_id}...")

        # -- SLM --
        if "slm" in systems:
            try:
                raw_text, latency = run_slm_inference(slm, msg)
                parsed = parse_model_json(raw_text)
                acc = results_by_system["slm"]
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

                ticket_result["slm_raw"] = raw_text
                ticket_result["slm_parsed"] = parsed
                ticket_result["slm_latency"] = latency
            except Exception as e:
                print(f"    SLM error on {ticket_id}: {e}")
                acc = results_by_system["slm"]
                acc["intent_preds"].append("")
                acc["urgency_preds"].append("")
                acc["workflow_preds"].append([])
                acc["tool_preds"].append([])
                acc["latencies"].append(0.0)
                acc["costs"].append(0.0)
                ticket_result["slm_error"] = str(e)

        # -- GPT-4o --
        if "gpt4o" in systems:
            try:
                raw_text, latency, usage = await run_gpt4o_inference(openai_client, msg)
                parsed = parse_model_json(raw_text)
                cost = estimate_cost("gpt-4o", usage)
                acc = results_by_system["gpt4o"]
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

                ticket_result["gpt4o_raw"] = raw_text
                ticket_result["gpt4o_parsed"] = parsed
                ticket_result["gpt4o_latency"] = latency
                ticket_result["gpt4o_cost"] = cost
            except Exception as e:
                print(f"    GPT-4o error on {ticket_id}: {e}")
                acc = results_by_system["gpt4o"]
                acc["intent_preds"].append("")
                acc["urgency_preds"].append("")
                acc["workflow_preds"].append([])
                acc["tool_preds"].append([])
                acc["latencies"].append(0.0)
                acc["costs"].append(0.0)
                ticket_result["gpt4o_error"] = str(e)

        # -- Claude Sonnet --
        if "claude" in systems:
            try:
                raw_text, latency, usage = await run_claude_inference(anthropic_client, msg)
                parsed = parse_model_json(raw_text)
                cost = estimate_cost("claude-sonnet-4-20250514", usage)
                acc = results_by_system["claude"]
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

                ticket_result["claude_raw"] = raw_text
                ticket_result["claude_parsed"] = parsed
                ticket_result["claude_latency"] = latency
                ticket_result["claude_cost"] = cost
            except Exception as e:
                print(f"    Claude error on {ticket_id}: {e}")
                acc = results_by_system["claude"]
                acc["intent_preds"].append("")
                acc["urgency_preds"].append("")
                acc["workflow_preds"].append([])
                acc["tool_preds"].append([])
                acc["latencies"].append(0.0)
                acc["costs"].append(0.0)
                ticket_result["claude_error"] = str(e)

        per_ticket_results.append(ticket_result)

    # -- LLM-as-Judge for response quality --
    print("\n  Running LLM-as-Judge scoring...")
    if openai_client is not None:
        judge_semaphore = asyncio.Semaphore(10)

        async def judge_with_semaphore(customer_msg: str, agent_resp: str) -> dict:
            async with judge_semaphore:
                return await judge_response(openai_client, customer_msg, agent_resp)

        for sys_key in systems:
            sys_name = systems[sys_key]["name"]
            print(f"    Judging {sys_name} responses...")
            judge_tasks = []
            for tr in per_ticket_results:
                parsed = tr.get(f"{sys_key}_parsed")
                if parsed and isinstance(parsed, dict) and parsed.get("response"):
                    judge_tasks.append(
                        judge_with_semaphore(tr["customer_message"], parsed["response"])
                    )
                else:
                    judge_tasks.append(None)

            # Gather results
            for i, task in enumerate(judge_tasks):
                if task is not None:
                    try:
                        score = await task
                    except Exception as e:
                        print(f"      Judge error: {e}")
                        score = {
                            "helpfulness": 0,
                            "correctness": 0,
                            "professionalism": 0,
                            "empathy": 0,
                        }
                    results_by_system[sys_key]["judge_scores"].append(score)
                    per_ticket_results[i][f"{sys_key}_judge"] = score
                else:
                    results_by_system[sys_key]["judge_scores"].append(None)
    else:
        print("    Skipping judge scoring (no OpenAI client)")

    # -- Compute aggregate metrics --
    print("\n  Computing aggregate metrics...")
    metrics_by_system = {}

    for sys_key, acc in results_by_system.items():
        sys_name = systems[sys_key]["name"]

        # Intent metrics
        i_acc = intent_accuracy(acc["intent_preds"], gold_intents)
        i_f1 = intent_f1_micro(acc["intent_preds"], gold_intents)

        # Urgency metrics
        u_acc = urgency_accuracy(acc["urgency_preds"], gold_urgencies)

        # Workflow accuracy (edit distance)
        w_acc = workflow_accuracy(acc["workflow_preds"], gold_workflows)

        # Tool selection F1
        pred_tool_sets = [extract_tool_names(t) for t in acc["tool_preds"]]
        gold_tool_sets = [extract_tool_names(t) for t in gold_tools]
        t_metrics = tool_selection_f1(pred_tool_sets, gold_tool_sets)

        # Latency stats
        valid_latencies = [l for l in acc["latencies"] if l > 0]
        if valid_latencies:
            lat_p50 = statistics.median(valid_latencies)
            lat_p95 = (
                sorted(valid_latencies)[int(len(valid_latencies) * 0.95)]
                if len(valid_latencies) > 1
                else valid_latencies[0]
            )
        else:
            lat_p50 = 0.0
            lat_p95 = 0.0

        # Cost
        total_cost = sum(acc["costs"])

        # Judge scores
        valid_scores = [s for s in acc["judge_scores"] if s is not None]
        if valid_scores:
            avg_helpfulness = statistics.mean([s.get("helpfulness", 0) for s in valid_scores])
            avg_correctness = statistics.mean([s.get("correctness", 0) for s in valid_scores])
            avg_professionalism = statistics.mean([s.get("professionalism", 0) for s in valid_scores])
            avg_empathy = statistics.mean([s.get("empathy", 0) for s in valid_scores])
        else:
            avg_helpfulness = 0.0
            avg_correctness = 0.0
            avg_professionalism = 0.0
            avg_empathy = 0.0

        metrics_by_system[sys_key] = {
            "system_name": sys_name,
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
            "judge_helpfulness": round(avg_helpfulness, 2),
            "judge_correctness": round(avg_correctness, 2),
            "judge_professionalism": round(avg_professionalism, 2),
            "judge_empathy": round(avg_empathy, 2),
            "num_tickets": len(tickets),
            "num_parse_failures": sum(
                1 for p in acc["intent_preds"] if p == ""
            ),
        }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "gold_file": str(config.get("_gold_file", "")),
            "model_path": str(config.get("_model_path", "")),
            "num_tickets": len(tickets),
        },
        "metrics_by_system": metrics_by_system,
        "per_ticket_results": per_ticket_results,
    }


# ── Output formatting ───────────────────────────────────────────────────────

def print_comparison_table(results: dict):
    """Print a formatted comparison table to stdout."""
    metrics = results.get("metrics_by_system", {})
    if not metrics:
        print("\nNo metrics to display.")
        return

    sys_keys = list(metrics.keys())
    sys_names = [metrics[k]["system_name"] for k in sys_keys]

    # Column widths
    label_w = 26
    col_w = 18

    # Header
    header_line = "=" * (label_w + col_w * len(sys_keys) + 4)
    print(f"\n{header_line}")
    print("  KIKI EVALUATION RESULTS")
    print(f"  {results.get('timestamp', '')}")
    print(header_line)

    # Column headers
    header = f"  {'Metric':<{label_w}}"
    for name in sys_names:
        header += f"{name:>{col_w}}"
    print(header)
    print("  " + "-" * (label_w + col_w * len(sys_keys)))

    # Metric rows
    rows = [
        ("Intent Accuracy", "intent_accuracy", "{:.1%}"),
        ("Intent F1 (micro)", "intent_f1_micro", "{:.1%}"),
        ("Urgency Accuracy", "urgency_accuracy", "{:.1%}"),
        ("Workflow Accuracy", "workflow_accuracy", "{:.1%}"),
        ("Tool Precision", "tool_precision", "{:.1%}"),
        ("Tool Recall", "tool_recall", "{:.1%}"),
        ("Tool F1", "tool_f1", "{:.1%}"),
        ("", None, None),  # separator
        ("Latency p50 (s)", "latency_p50_s", "{:.3f}"),
        ("Latency p95 (s)", "latency_p95_s", "{:.3f}"),
        ("Total Cost (USD)", "total_cost_usd", "${:.4f}"),
        ("Cost/Ticket (USD)", "cost_per_ticket_usd", "${:.6f}"),
        ("", None, None),  # separator
        ("Judge: Helpfulness", "judge_helpfulness", "{:.2f}/5"),
        ("Judge: Correctness", "judge_correctness", "{:.2f}/5"),
        ("Judge: Professionalism", "judge_professionalism", "{:.2f}/5"),
        ("Judge: Empathy", "judge_empathy", "{:.2f}/5"),
        ("", None, None),  # separator
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
            formatted = fmt.format(val)
            row += f"{formatted:>{col_w}}"
        print(row)

    print(header_line)
    print()


def save_results(results: dict, output_dir: str = "outputs/results"):
    """Save evaluation results to JSON."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "evaluation_results.json"

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {out_file}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark fine-tuned SLM against GPT-4o and Claude Sonnet"
    )
    parser.add_argument(
        "--config",
        default="configs/poc_config.yaml",
        help="Path to config file (default: configs/poc_config.yaml)",
    )
    parser.add_argument(
        "--gold-file",
        default="data/gold/gold_100.jsonl",
        help="Path to gold-labeled test set (default: data/gold/gold_100.jsonl)",
    )
    parser.add_argument(
        "--model-path",
        default="outputs/models/kiki-poc-v1",
        help="Path to fine-tuned SLM (default: outputs/models/kiki-poc-v1)",
    )
    parser.add_argument(
        "--skip-slm",
        action="store_true",
        help="Skip local SLM evaluation",
    )
    parser.add_argument(
        "--skip-apis",
        action="store_true",
        help="Skip API calls (GPT-4o, Claude). Useful for testing.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  KIKI SLM EVALUATION BENCHMARK")
    print("=" * 60)

    # Load config
    print(f"\n[1/5] Loading config from {args.config}")
    config = load_config(args.config)
    config["_gold_file"] = args.gold_file
    config["_model_path"] = args.model_path

    # Load gold data
    print(f"\n[2/5] Loading gold data from {args.gold_file}")
    tickets = load_gold_data(args.gold_file)
    if not tickets:
        print("  ERROR: No gold tickets loaded. Exiting.")
        return

    # Run evaluation
    print(f"\n[3/5] Running evaluation")
    if args.skip_slm:
        print("  (SLM evaluation skipped via --skip-slm)")
    if args.skip_apis:
        print("  (API evaluation skipped via --skip-apis)")

    results = asyncio.run(
        evaluate_all(
            tickets=tickets,
            config=config,
            model_path=args.model_path,
            skip_slm=args.skip_slm,
            skip_apis=args.skip_apis,
        )
    )

    if not results:
        print("\n  No results generated. Exiting.")
        return

    # Save results
    print(f"\n[4/5] Saving results")
    save_results(results)

    # Print comparison table
    print(f"\n[5/5] Evaluation complete")
    print_comparison_table(results)


if __name__ == "__main__":
    main()
