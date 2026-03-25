#!/usr/bin/env python3
"""Step 4: Transform agent traces into ChatML training data with <think> blocks.

Reads trace JSONs from data/traces/, synthesizes reasoning chains from the agent's
structured outputs, and produces ChatML JSONL ready for Qwen 3.5 4B QLoRA fine-tuning.

IMPORTANT: This script produces tool calls using Qwen 3.5's native format.
The tool_calls and tool responses include id/tool_call_id fields required by
Qwen's chat template. Tool definitions are included in the messages as a
separate tools field for apply_chat_template().

Generates:
  - data/chatml/train.jsonl  (90%)
  - data/chatml/eval.jsonl   (10%)

Usage:
    python scripts/loopper/build_chatml.py
    python scripts/loopper/build_chatml.py --correction-ratio 0.05
    python scripts/loopper/build_chatml.py --empty-rag-ratio 0.08
    python scripts/loopper/build_chatml.py --eval-split 0.10
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
import uuid
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Default paths ─────────────────────────────────────────────
TRACES_DIR = "/Users/vishnu/Dev/bluesprings/Loopper/Loopper-AI/finetune-dataset/data/traces"
OUTPUT_DIR = "/Users/vishnu/Dev/bluesprings/Loopper/Loopper-AI/finetune-dataset/data/chatml"

# ── Formal tool definition (for Qwen 3.5 tool calling) ────────
RAG_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "rag_search",
        "description": "Search the Loopper knowledge base for policies, processes, and tone guidance.",
        "parameters": {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "enum": ["faq", "operations", "communication_guidelines", "supplier_data"],
                    "description": "Which knowledge base collection to search.",
                },
                "query": {
                    "type": "string",
                    "description": "Search query — use domain-specific noun phrases, not full questions.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return.",
                    "default": 5,
                },
            },
            "required": ["collection", "query"],
        },
    },
}

# ── System prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a Loopper support agent for B2B promotional products (mugs, pens, bags, \
t-shirts, caps, water bottles, notebooks, lanyards, USB sticks, power banks, etc.). \
Headquartered in Amsterdam, serving Europe for 24+ years. 6,000+ customizable products.

You MUST think step by step in <think> blocks before every action.
You MUST call rag_search at least once before generating your final response \
(exception: simple acknowledgments like "OK thank you" that need no context).
You MUST ground your response in retrieved context — never invent policies, \
timelines, prices, or delivery dates.
When retrieved context is empty or irrelevant, acknowledge the gap and escalate \
to the human reviewer rather than guessing.

Preserve all PII tokens exactly as they appear (e.g. [NAME], [EMAIL], [ORG], [ADDRESS]).
Sign all responses as Marc Logier, Account Manager — Loopper.

Output your final response as a single JSON object with these fields:
intent, urgency, confidence, is_valid, rejection_type, resolution_type, team, \
actions, summary, reasoning, response"""

# ── Think block variation templates ───────────────────────────

THINK_OPENERS = [
    "Let me analyze this ticket.",
    "Looking at this ticket carefully.",
    "Let me break down what the customer is asking.",
    "Analyzing the customer's message.",
    "Let me understand what's happening here.",
]

SEARCH_PLAN_INTROS = [
    "I need to search for:",
    "To handle this properly, I should look up:",
    "I'll need context on:",
    "Before responding, I need to find:",
    "Let me search for the relevant information:",
]

GUARDRAIL_VARIATIONS = [
    [
        "Before responding, let me check my guardrails:",
        "- I can only use information from the retrieved context",
        "- I must not promise timelines unless the policy specifies them",
        "- I cannot confirm actions — only the human reviewer can execute",
    ],
    [
        "Checking what I can and cannot say:",
        "- Only facts from the KB, nothing invented",
        "- No specific dates or prices unless from the policy",
        "- I draft the response, I don't execute actions",
    ],
    [
        "Important constraints for this response:",
        "- Ground everything in retrieved context",
        "- Don't over-promise on timelines",
        "- Flag anything I'm uncertain about for human review",
    ],
]

# ── Category search plans ─────────────────────────────────────
CATEGORY_SEARCH_PLANS = {
    "new_order_inquiry": [
        "Ordering process, pricing, customization options, MOQ (faq)",
        "Payment methods, checkout process (faq)",
        "Welcoming, sales-oriented response tone (communication_guidelines)",
    ],
    "quality_complaint": [
        "Quality complaint handling, replacement/refund conditions (faq)",
        "Escalation steps, photo evidence requirements (operations)",
        "Empathetic, solution-focused response tone (communication_guidelines)",
    ],
    "delivery_issue": [
        "Delivery timelines, tracking, delay handling (faq)",
        "Reassuring, proactive response tone (communication_guidelines)",
    ],
    "refund_request": [
        "Refund conditions, timelines, eligibility (faq)",
        "Refund workflow, finance team handoff (operations)",
        "Empathetic response tone (communication_guidelines)",
    ],
    "order_cancellation": [
        "Cancellation conditions, retention strategies (faq)",
        "Cancellation workflow, alternative offering (operations)",
        "Understanding, retention-focused response tone (communication_guidelines)",
    ],
    "design_update": [
        "Design revision workflow, vector file requirements (operations)",
        "Collaborative response tone (communication_guidelines)",
    ],
    "payment_confirmation": [
        "Payment methods, invoice process (faq)",
        "Professional response tone (communication_guidelines)",
    ],
    "sample_request": [
        "Sample pricing, blank vs printed, transport costs (faq)",
        "Digital mockup alternative (faq)",
        "Helpful sales response tone (communication_guidelines)",
    ],
    "price_negotiation": [
        "Price matching, competitor handling (faq)",
        "Negotiation response tone (communication_guidelines)",
    ],
    "customer_feedback": [
        "Feedback handling, Trustpilot invitation (faq)",
        "Warm, appreciative response tone (communication_guidelines)",
    ],
}

DEFAULT_SEARCH_PLAN = [
    "Relevant policies for this ticket type (faq)",
    "Appropriate response tone (communication_guidelines)",
]

# ── Degraded responses for correction examples ────────────────
DEGRADED_RESPONSES = [
    "I'll process your request right away. Your refund will be issued within 3-5 business days.",
    "Thank you for your message. I've updated your order and the changes will be reflected shortly.",
    "I understand your concern. Our standard delivery time is 5-7 business days and your order is on track.",
    "I apologize for the inconvenience. We'll send a replacement immediately at no extra cost.",
    "Your payment has been confirmed and your order is now in production.",
]

CORRECTION_MESSAGES = [
    "You must search the knowledge base before responding. Search for relevant policies first.",
    "Do not respond without checking the knowledge base. Look up the relevant policy.",
    "You skipped the knowledge base search. You need to retrieve context before generating a response.",
    "Search the knowledge base first — your response must be grounded in retrieved policies.",
]


def _tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"


def synthesize_think_block_1(triage: dict) -> str:
    """Synthesize the first <think> block from triage output."""
    cr = triage.get("category_reasoning") or {}
    vr = triage.get("validation_reasoning") or {}
    category = triage.get("category", "other")
    confidence = triage.get("category_confidence", 0.0)

    lines = ["<think>"]
    lines.append(random.choice(THINK_OPENERS))
    lines.append("")

    # Intent analysis with deliberation
    primary_intent = cr.get("primary_intent", "")
    reasoning_summary = cr.get("reasoning_summary", "")
    key_indicators = cr.get("key_indicators", [])

    if primary_intent:
        lines.append(f"The customer's primary intent is: {primary_intent}")
    if key_indicators:
        lines.append(f"Key signals: {', '.join(str(k) for k in key_indicators)}")
    if reasoning_summary:
        lines.append(f"\n{reasoning_summary}")

    # Add deliberation for ambiguous cases
    if confidence < 0.8:
        lines.append(f"\nConfidence is {confidence:.0%} — this could also be interpreted differently, "
                      f"but {category.replace('_', ' ')} fits best based on the primary request.")

    # Validation reasoning
    val_reasoning = vr.get("reasoning", "")
    if val_reasoning:
        lines.append(f"\nTriage assessment: {val_reasoning}")

    # Search plan with rationale
    search_plan = CATEGORY_SEARCH_PLANS.get(category, DEFAULT_SEARCH_PLAN)
    lines.append(f"\n{random.choice(SEARCH_PLAN_INTROS)}")
    for i, plan in enumerate(search_plan, 1):
        lines.append(f"{i}. {plan}")

    lines.append("</think>")
    return "\n".join(lines)


def synthesize_think_between(tool_call: dict, prev_results: list) -> str:
    """Synthesize a <think> block between tool calls that analyzes previous results."""
    query = tool_call.get("query", "")
    collection = tool_call.get("collection", "")

    lines = ["<think>"]

    if prev_results:
        high_rel = [r for r in prev_results if r.get("rerank_score", 0) > 0.5]
        if high_rel:
            lines.append(f"The previous search returned {len(high_rel)} relevant result(s).")
            # Summarize what was found
            for r in high_rel[:2]:
                source = r.get("source", "unknown")
                text = r.get("text", "")[:100]
                lines.append(f"- [{source}]: {text}...")
            lines.append(f"\nThis gives me the policy context. Now I also need tone guidance "
                          f"for this type of response.")
        else:
            lines.append("The previous search returned results but none were highly relevant. "
                          "I'll proceed with what I have and supplement with additional context.")
    else:
        lines.append("No results from previous search were available.")

    lines.append(f"\nSearching {collection} for: {query}")
    lines.append("</think>")
    return "\n".join(lines)


def synthesize_think_block_2(response: dict, tool_results: list, category: str) -> str:
    """Synthesize the final <think> block from response output."""
    ar = response.get("action_reasoning") or {}
    rr = response.get("resolution_reasoning") or {}

    lines = ["<think>"]

    # Summarize what was retrieved
    high_relevance = [r for r in tool_results if r.get("rerank_score", 0) > 0.5]
    if high_relevance:
        lines.append("From the search results I found:")
        for r in high_relevance[:4]:
            source = r.get("source", "unknown")
            text = r.get("text", "")[:120]
            lines.append(f"- [{source}]: {text}...")
    elif tool_results:
        lines.append("Search results were available but none scored as highly relevant. "
                      "I should be cautious and escalate rather than guess at policies.")
    else:
        lines.append("No relevant results were found. I should acknowledge the gap "
                      "and let the human reviewer handle this with full context.")

    # Resolution reasoning
    why_resolution = rr.get("why_resolution_type", "")
    if why_resolution:
        lines.append(f"\nResolution decision: {why_resolution}")

    escalation_risk = rr.get("escalation_risk", "")
    if escalation_risk:
        lines.append(f"Escalation risk: {escalation_risk}")

    # Action reasoning
    why_actions = ar.get("why_these_actions", "")
    if why_actions:
        lines.append(f"\nActions for reviewer: {why_actions}")

    policy_basis = ar.get("policy_basis", "")
    if policy_basis:
        lines.append(f"Policy basis: {policy_basis}")

    urgency = ar.get("urgency_level", "")
    if urgency:
        lines.append(f"Urgency: {urgency}")

    # Context-dependent guardrails (varied per example)
    guardrail = random.choice(GUARDRAIL_VARIATIONS)
    lines.append("")
    for line in guardrail:
        lines.append(line)

    lines.append("</think>")
    return "\n".join(lines)


def build_output_json(trace: dict) -> dict:
    """Build the final output JSON from the trace."""
    triage = trace.get("triage", {})
    response = trace.get("response", {})
    ar = response.get("action_reasoning") or {}
    rr = response.get("resolution_reasoning") or {}
    vr = triage.get("validation_reasoning") or {}

    return {
        "intent": triage.get("category", "other"),
        "urgency": ar.get("urgency_level", "medium"),
        "confidence": triage.get("category_confidence", 0.0),
        "is_valid": triage.get("is_valid", True),
        "rejection_type": vr.get("rejection_type"),
        "resolution_type": response.get("resolution_type", "requires_human_action"),
        "team": response.get("human_team_required", "account_manager"),
        "actions": response.get("action_list") or [],
        "summary": response.get("summary") or [],
        "reasoning": {
            "intent_basis": (triage.get("category_reasoning") or {}).get("reasoning_summary", ""),
            "urgency_basis": ar.get("why_these_actions", ""),
            "resolution_basis": rr.get("why_resolution_type", ""),
            "policy_used": ar.get("policy_basis", ""),
        },
        "response": response.get("response_english") or "",
    }


def format_ticket_text(input_state: dict) -> str:
    """Format the ticket input into the user message."""
    ticket = input_state.get("ticket", {})
    ticket_id = ticket.get("ticket_id", "Unknown")
    messages = ticket.get("messages", [])

    parts = [f"Ticket ID: {ticket_id}"]
    incoming = sum(1 for m in messages if m.get("direction") == "incoming")
    outgoing = sum(1 for m in messages if m.get("direction") == "outgoing")
    parts.append(f"Message Count: {len(messages)} (Incoming: {incoming}, Outgoing: {outgoing})")
    parts.append("")

    for m in messages:
        direction = m.get("direction", "incoming")
        role = "Customer" if direction == "incoming" else "Loopper Agent"
        idx = m.get("message_index", 0)
        body = m.get("clean_body", "")
        parts.append(f"[{role}] (Message {idx}):\n{body}")
        parts.append("")

    return "\n".join(parts)


def trace_to_chatml(trace: dict) -> dict | None:
    """Convert a single agent trace into a ChatML training example.

    Uses Qwen 3.5's native tool calling format with tool_call_id linkage.
    """
    triage = trace.get("triage", {})
    retrieval = trace.get("retrieval", {})
    response = trace.get("response", {})
    category = triage.get("category", "other")

    if not triage.get("category") or not response.get("response_english"):
        return None

    messages = []

    # 1. System prompt
    messages.append({"role": "system", "content": SYSTEM_PROMPT})

    # 2. User message
    messages.append({"role": "user", "content": format_ticket_text(trace["ticket_input"])})

    # 3. Tool calls with <think> blocks
    tool_calls_data = retrieval.get("tool_calls", [])
    tool_results_data = retrieval.get("tool_results", [])

    if tool_calls_data:
        # First tool call with reasoning
        think_1 = synthesize_think_block_1(triage)
        tc_id = _tool_call_id()

        messages.append({
            "role": "assistant",
            "content": think_1,
            "tool_calls": [{
                "type": "function",
                "id": tc_id,
                "function": {
                    "name": "rag_search",
                    "arguments": json.dumps({
                        "collection": tool_calls_data[0].get("collection", "faq"),
                        "query": tool_calls_data[0].get("query", ""),
                        "top_k": tool_calls_data[0].get("top_k", 5),
                    }),
                },
            }],
        })

        # First tool result
        first_collection = tool_calls_data[0].get("collection", "")
        first_results = [r for r in tool_results_data if r.get("collection") == first_collection]
        if not first_results:
            first_results = tool_results_data[:3] if tool_results_data else []

        messages.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "name": "rag_search",
            "content": json.dumps({"results": first_results}, ensure_ascii=False),
        })

        # Additional tool calls
        for i in range(1, len(tool_calls_data)):
            tc = tool_calls_data[i]
            tc_id_n = _tool_call_id()
            between_think = synthesize_think_between(tc, first_results if i == 1 else [])

            messages.append({
                "role": "assistant",
                "content": between_think,
                "tool_calls": [{
                    "type": "function",
                    "id": tc_id_n,
                    "function": {
                        "name": "rag_search",
                        "arguments": json.dumps({
                            "collection": tc.get("collection", "communication_guidelines"),
                            "query": tc.get("query", ""),
                            "top_k": tc.get("top_k", 3),
                        }),
                    },
                }],
            })

            tc_results = [r for r in tool_results_data if r.get("collection") == tc.get("collection")]
            if not tc_results:
                remaining_results = [r for r in tool_results_data if r not in first_results]
                tc_results = remaining_results[:3] if remaining_results else []

            messages.append({
                "role": "tool",
                "tool_call_id": tc_id_n,
                "name": "rag_search",
                "content": json.dumps({"results": tc_results}, ensure_ascii=False),
            })

        # Final <think> + output JSON
        think_2 = synthesize_think_block_2(response, tool_results_data, category)
        final_output = build_output_json(trace)
        messages.append({
            "role": "assistant",
            "content": f"{think_2}\n{json.dumps(final_output, ensure_ascii=False)}",
        })

    else:
        # No tool calls — simple acknowledgment
        think_no_search = (
            "<think>\n"
            "This is a simple acknowledgment or follow-up that doesn't require "
            "searching the knowledge base. The customer is confirming receipt or "
            "saying thank you — I can respond directly with a brief, warm reply.\n"
            "</think>"
        )
        final_output = build_output_json(trace)
        messages.append({
            "role": "assistant",
            "content": f"{think_no_search}\n{json.dumps(final_output, ensure_ascii=False)}",
        })

    # Include tool definitions for Qwen's chat template
    return {
        "tools": [RAG_SEARCH_TOOL],
        "messages": messages,
    }


def create_correction_example(trace: dict) -> dict | None:
    """Create a training example where the model skips RAG and gets corrected.

    The 'wrong' attempt uses a DEGRADED response (hallucinated/generic), not
    the correct one, so the model learns that skipping RAG produces bad output.
    """
    correct_chatml = trace_to_chatml(trace)
    if not correct_chatml or len(correct_chatml["messages"]) < 4:
        return None

    # Build a degraded output (hallucinated, not grounded)
    output = build_output_json(trace)
    degraded_output = copy.deepcopy(output)
    degraded_output["response"] = random.choice(DEGRADED_RESPONSES)
    degraded_output["reasoning"]["policy_used"] = "No policy consulted"

    corrected_messages = [
        correct_chatml["messages"][0],  # system
        correct_chatml["messages"][1],  # user
        {"role": "assistant", "content": json.dumps(degraded_output, ensure_ascii=False)},
        {"role": "user", "content": random.choice(CORRECTION_MESSAGES)},
    ]
    # Then the correct chain (skip system and user, start from first assistant+tool)
    corrected_messages.extend(correct_chatml["messages"][2:])

    return {
        "tools": [RAG_SEARCH_TOOL],
        "messages": corrected_messages,
    }


def create_empty_rag_example(trace: dict, category: str) -> dict | None:
    """Create an example with empty RAG results teaching graceful degradation.

    The escalation response varies by category to teach context-appropriate fallbacks.
    """
    modified = copy.deepcopy(trace)
    modified["retrieval"]["tool_results"] = []

    # Category-specific escalation responses
    escalation_responses = {
        "quality_complaint": (
            "Hello,\n\nThank you for bringing this to our attention. I'm looking into this "
            "with our quality team and will get back to you shortly with more information.\n\n"
            "Best regards,\nMarc Logier, Account Manager — Loopper"
        ),
        "delivery_issue": (
            "Hello,\n\nThank you for reaching out. I'm checking on the status of your delivery "
            "with our logistics team and will update you as soon as I have more details.\n\n"
            "Best regards,\nMarc Logier, Account Manager — Loopper"
        ),
        "refund_request": (
            "Hello,\n\nThank you for your message. I've forwarded your request to our finance team "
            "and they will review it and get back to you shortly.\n\n"
            "Best regards,\nMarc Logier, Account Manager — Loopper"
        ),
    }
    default_escalation = (
        "Hello,\n\nThank you for reaching out. I'm looking into this with our team "
        "and will get back to you shortly with more information.\n\n"
        "Best regards,\nMarc Logier, Account Manager — Loopper"
    )

    modified["response"]["resolution_type"] = "needs_escalation"
    modified["response"]["response_english"] = escalation_responses.get(category, default_escalation)
    if modified["response"].get("action_list"):
        modified["response"]["action_list"] = [
            "Review ticket manually — automated context retrieval returned no results"
        ]

    return trace_to_chatml(modified)


def load_traces(traces_dir: str) -> list[dict]:
    """Load all trace JSONs, filtering to only 'full' completion status."""
    traces_path = Path(traces_dir)
    files = sorted(traces_path.glob("*.json"))
    files = [f for f in files if not f.name.startswith("_")]

    traces = []
    skipped = {"triage_rejected": 0, "schema_invalid": 0, "partial": 0}
    errors = 0

    for f in files:
        try:
            with open(f) as fh:
                trace = json.load(fh)
            status = trace.get("completion_status", "full")
            if status != "full":
                skipped[status] = skipped.get(status, 0) + 1
                continue
            traces.append(trace)
        except json.JSONDecodeError:
            errors += 1

    logger.info("Loaded %d full traces (%d errors, skipped: %s) from %s",
                len(traces), errors, dict(skipped), traces_dir)
    return traces


def main():
    parser = argparse.ArgumentParser(description="Build ChatML training data from agent traces")
    parser.add_argument("--input-dir", default=TRACES_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--correction-ratio", type=float, default=0.05)
    parser.add_argument("--empty-rag-ratio", type=float, default=0.08)
    parser.add_argument("--eval-split", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    traces = load_traces(args.input_dir)
    if not traces:
        logger.error("No traces found. Run generate_traces.py first.")
        sys.exit(1)

    # CRITICAL: Partition traces into train/eval pools BEFORE generating
    # any example variants. This prevents data leakage.
    random.shuffle(traces)
    n_eval_traces = max(1, int(len(traces) * args.eval_split))
    eval_traces = traces[:n_eval_traces]
    train_traces = traces[n_eval_traces:]

    logger.info("Trace pool: %d train, %d eval (no overlap)", len(train_traces), len(eval_traces))

    # Convert train traces to ChatML
    train_examples = []
    skipped = 0
    for trace in train_traces:
        chatml = trace_to_chatml(trace)
        if chatml:
            train_examples.append(chatml)
        else:
            skipped += 1

    # Convert eval traces
    eval_examples = []
    for trace in eval_traces:
        chatml = trace_to_chatml(trace)
        if chatml:
            eval_examples.append(chatml)
        else:
            skipped += 1

    logger.info("Converted: %d train, %d eval (%d skipped)", len(train_examples), len(eval_examples), skipped)

    # Add correction examples (TRAIN ONLY — from train traces)
    n_corrections = int(len(train_traces) * args.correction_ratio)
    correction_pool = random.sample(train_traces, min(n_corrections * 2, len(train_traces)))
    corrections = 0
    for trace in correction_pool:
        if corrections >= n_corrections:
            break
        ex = create_correction_example(trace)
        if ex:
            train_examples.append(ex)
            corrections += 1
    logger.info("Added %d correction examples to train", corrections)

    # Add empty-RAG examples (TRAIN ONLY — from train traces)
    n_empty = int(len(train_traces) * args.empty_rag_ratio)
    empty_pool = random.sample(train_traces, min(n_empty * 2, len(train_traces)))
    empties = 0
    for trace in empty_pool:
        if empties >= n_empty:
            break
        category = trace.get("triage", {}).get("category", "other")
        ex = create_empty_rag_example(trace, category)
        if ex:
            train_examples.append(ex)
            empties += 1
    logger.info("Added %d empty-RAG examples to train", empties)

    # Shuffle train
    random.shuffle(train_examples)

    # Write JSONL
    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"

    for path, data in [(train_path, train_examples), (eval_path, eval_examples)]:
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Stats
    cat_counts = Counter()
    tool_call_counts = Counter()
    for ex in train_examples + eval_examples:
        for msg in reversed(ex["messages"]):
            if msg["role"] == "assistant" and msg.get("content"):
                content = msg["content"]
                think_end = content.rfind("</think>")
                search_from = think_end + len("</think>") if think_end >= 0 else 0
                json_start = content.find("{", search_from)
                if json_start >= 0:
                    try:
                        output = json.loads(content[json_start:])
                        cat_counts[output.get("intent", "unknown")] += 1
                    except json.JSONDecodeError:
                        pass
                break
        n_tools = sum(1 for m in ex["messages"] if m["role"] == "tool")
        tool_call_counts[n_tools] += 1

    logger.info("")
    logger.info("=" * 60)
    logger.info("  CHATML DATASET GENERATED")
    logger.info("=" * 60)
    logger.info("  Train: %s (%d examples)", train_path, len(train_examples))
    logger.info("  Eval:  %s (%d examples)", eval_path, len(eval_examples))
    logger.info("")
    logger.info("  Category distribution:")
    for cat, count in cat_counts.most_common():
        logger.info("    %-25s %5d", cat, count)
    logger.info("")
    logger.info("  Tool call distribution:")
    for n_tools, count in sorted(tool_call_counts.items()):
        logger.info("    %d tool calls: %5d examples", n_tools, count)


if __name__ == "__main__":
    main()
