#!/usr/bin/env python3
"""Step 4: Transform agent traces into ChatML training data for Qwen 3.5 9B.

Uses Qwen's official format:
  - reasoning goes in `reasoning_content` field (separate from `content`)
  - tool calls use `tool_calls` field with id linkage
  - tool definitions in top-level `tools` field
  - final response in `content` as JSON

This avoids the confirmed <think> + tool_call leakage bug on Qwen 3.5 small models
by keeping reasoning in its own field rather than mixing it into content.

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

# ── Default paths (from configs/loopper_pipeline.yaml or project-relative) ──
from scripts.loopper.config import get_default_paths as _get_paths
_PATHS = _get_paths()
TRACES_DIR = _PATHS["traces"]
OUTPUT_DIR = _PATHS["chatml_output"]

# ── Formal tool definition ────────────────────────────────────
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

You MUST reason before every action and before your final response.
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

# ── Reasoning variation templates ─────────────────────────────

REASONING_OPENERS = [
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
    "I can only use information from retrieved context. No invented timelines or prices. I draft, the human reviewer executes.",
    "Only facts from the KB. No specific dates or prices unless from policy. I draft the response, I don't execute actions.",
    "Ground everything in retrieved context. Don't over-promise on timelines. Flag anything uncertain for human review.",
]

# ── Category search plans ─────────────────────────────────────
CATEGORY_SEARCH_PLANS = {
    "new_order_inquiry": [
        "Ordering process, pricing, customization options, MOQ (faq)",
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
    ],
    "order_cancellation": [
        "Cancellation conditions, retention strategies (faq)",
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


def synthesize_reasoning_1(triage: dict) -> str:
    """Synthesize reasoning for the first assistant turn (before first tool call).

    Goes in the `reasoning_content` field, NOT in `content`.
    Keep it concise: 50-150 tokens. The model should reason, not ramble.
    """
    cr = triage.get("category_reasoning") or {}
    vr = triage.get("validation_reasoning") or {}
    category = triage.get("category", "other")
    confidence = triage.get("category_confidence", 0.0)

    lines = [random.choice(REASONING_OPENERS)]

    primary_intent = cr.get("primary_intent", "")
    reasoning_summary = cr.get("reasoning_summary", "")
    key_indicators = cr.get("key_indicators", [])

    if primary_intent:
        lines.append(f"Primary intent: {primary_intent}")
    if key_indicators:
        lines.append(f"Key signals: {', '.join(str(k) for k in key_indicators)}")
    if reasoning_summary:
        lines.append(reasoning_summary)

    if confidence < 0.8:
        lines.append(f"Confidence {confidence:.0%} — ambiguous, but {category.replace('_', ' ')} fits best.")

    val_reasoning = vr.get("reasoning", "")
    if val_reasoning:
        lines.append(f"Triage: {val_reasoning}")

    search_plan = CATEGORY_SEARCH_PLANS.get(category, DEFAULT_SEARCH_PLAN)
    lines.append(random.choice(SEARCH_PLAN_INTROS))
    for plan in search_plan:
        lines.append(f"- {plan}")

    return "\n".join(lines)


def synthesize_reasoning_between(tool_call: dict, prev_results: list) -> str:
    """Synthesize reasoning between tool calls. Concise: 30-80 tokens."""
    query = tool_call.get("query", "")
    collection = tool_call.get("collection", "")

    lines = []
    if prev_results:
        high_rel = [r for r in prev_results if r.get("rerank_score", 0) > 0.5]
        if high_rel:
            lines.append(f"Found {len(high_rel)} relevant result(s). Got the policy context.")
            lines.append(f"Now need tone guidance from {collection}.")
        else:
            lines.append("Previous search had weak results. Supplementing with additional context.")
    else:
        lines.append("No prior results available.")

    lines.append(f"Searching {collection}: {query}")
    return "\n".join(lines)


def synthesize_reasoning_final(response: dict, tool_results: list) -> str:
    """Synthesize reasoning for the final assistant turn (before JSON output).

    Concise: 50-120 tokens. Summarize findings, state resolution, check guardrails.
    """
    ar = response.get("action_reasoning") or {}
    rr = response.get("resolution_reasoning") or {}

    lines = []

    high_relevance = [r for r in tool_results if r.get("rerank_score", 0) > 0.5]
    if high_relevance:
        lines.append("Retrieved context:")
        for r in high_relevance[:3]:
            source = r.get("source", "unknown")
            text = r.get("text", "")[:80]
            lines.append(f"- [{source}]: {text}...")
    elif tool_results:
        lines.append("Search results were weak. Being cautious — escalating rather than guessing.")
    else:
        lines.append("No results found. Acknowledging the gap, escalating to human reviewer.")

    why_resolution = rr.get("why_resolution_type", "")
    if why_resolution:
        lines.append(f"Resolution: {why_resolution}")

    why_actions = ar.get("why_these_actions", "")
    if why_actions:
        lines.append(f"Actions: {why_actions}")

    policy_basis = ar.get("policy_basis", "")
    if policy_basis:
        lines.append(f"Policy: {policy_basis}")

    lines.append(random.choice(GUARDRAIL_VARIATIONS))

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

    Uses Qwen 3.5's official format:
    - reasoning_content field for reasoning (separate from content)
    - tool_calls field with id linkage
    - content field for final JSON output only
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

    # 3. Tool calls with reasoning_content
    tool_calls_data = retrieval.get("tool_calls", [])
    tool_results_data = retrieval.get("tool_results", [])

    if tool_calls_data:
        # First tool call: reasoning in reasoning_content, tool call in tool_calls
        reasoning_1 = synthesize_reasoning_1(triage)
        tc_id = _tool_call_id()

        messages.append({
            "role": "assistant",
            "content": "",
            "reasoning_content": reasoning_1,
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
            reasoning_between = synthesize_reasoning_between(tc, first_results if i == 1 else [])

            messages.append({
                "role": "assistant",
                "content": "",
                "reasoning_content": reasoning_between,
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

        # Final response: reasoning in reasoning_content, JSON in content
        reasoning_final = synthesize_reasoning_final(response, tool_results_data)
        final_output = build_output_json(trace)

        messages.append({
            "role": "assistant",
            "reasoning_content": reasoning_final,
            "content": json.dumps(final_output, ensure_ascii=False),
        })

    else:
        # No tool calls — simple acknowledgment
        reasoning_no_search = (
            "Simple acknowledgment or follow-up. Customer is confirming receipt or "
            "saying thank you. No knowledge base search needed — responding directly."
        )
        final_output = build_output_json(trace)

        messages.append({
            "role": "assistant",
            "reasoning_content": reasoning_no_search,
            "content": json.dumps(final_output, ensure_ascii=False),
        })

    return {
        "tools": [RAG_SEARCH_TOOL],
        "messages": messages,
    }


def create_correction_example(trace: dict) -> dict | None:
    """Create a training example where the model skips RAG and gets corrected."""
    correct_chatml = trace_to_chatml(trace)
    if not correct_chatml or len(correct_chatml["messages"]) < 4:
        return None

    output = build_output_json(trace)
    degraded_output = copy.deepcopy(output)
    degraded_output["response"] = random.choice(DEGRADED_RESPONSES)
    degraded_output["reasoning"]["policy_used"] = "No policy consulted"

    corrected_messages = [
        correct_chatml["messages"][0],  # system
        correct_chatml["messages"][1],  # user
        # Model tries to respond without searching (no reasoning_content — it didn't think)
        {"role": "assistant", "content": json.dumps(degraded_output, ensure_ascii=False)},
        {"role": "user", "content": random.choice(CORRECTION_MESSAGES)},
    ]
    # Then the correct chain
    corrected_messages.extend(correct_chatml["messages"][2:])

    return {
        "tools": [RAG_SEARCH_TOOL],
        "messages": corrected_messages,
    }


def create_empty_rag_example(trace: dict, category: str) -> dict | None:
    """Create an example with empty RAG results teaching graceful degradation."""
    modified = copy.deepcopy(trace)
    modified["retrieval"]["tool_results"] = []

    escalation_responses = {
        "quality_complaint": (
            "Hello,\n\nThank you for bringing this to our attention. I'm looking into this "
            "with our quality team and will get back to you shortly.\n\n"
            "Best regards,\nMarc Logier, Account Manager — Loopper"
        ),
        "delivery_issue": (
            "Hello,\n\nThank you for reaching out. I'm checking on the status of your delivery "
            "with our logistics team and will update you as soon as I have details.\n\n"
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
        "and will get back to you shortly.\n\n"
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

    # Partition traces into train/eval BEFORE generating variants (prevents leakage)
    random.shuffle(traces)
    n_eval_traces = max(1, int(len(traces) * args.eval_split))
    eval_traces = traces[:n_eval_traces]
    train_traces = traces[n_eval_traces:]

    logger.info("Trace pool: %d train, %d eval (no overlap)", len(train_traces), len(eval_traces))

    # Convert to ChatML
    train_examples = []
    eval_examples = []
    skipped = 0

    for trace in train_traces:
        chatml = trace_to_chatml(trace)
        if chatml:
            train_examples.append(chatml)
        else:
            skipped += 1

    for trace in eval_traces:
        chatml = trace_to_chatml(trace)
        if chatml:
            eval_examples.append(chatml)
        else:
            skipped += 1

    logger.info("Converted: %d train, %d eval (%d skipped)", len(train_examples), len(eval_examples), skipped)

    # Correction examples (TRAIN ONLY)
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

    # Empty-RAG examples (TRAIN ONLY)
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
        # Extract category from final assistant content
        for msg in reversed(ex["messages"]):
            if msg["role"] == "assistant" and msg.get("content"):
                try:
                    output = json.loads(msg["content"])
                    cat_counts[output.get("intent", "unknown")] += 1
                except json.JSONDecodeError:
                    pass
                break
        n_tools = sum(1 for m in ex["messages"] if m["role"] == "tool")
        tool_call_counts[n_tools] += 1

    logger.info("")
    logger.info("=" * 60)
    logger.info("  CHATML DATASET GENERATED (Qwen 3.5 reasoning_content format)")
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
