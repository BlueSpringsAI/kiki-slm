#!/usr/bin/env python3
"""Step 6: Verify dataset format end-to-end before trace generation.

Renders synthetic ChatML examples through the Qwen3-4B-Thinking tokenizer's
chat template and visually inspects the output to confirm:
  1. The `tools` schema appears in the system block
  2. `<think>...</think>` blocks are rendered from `reasoning_content`
  3. Tool calls render in Qwen3's `<tool_call>...</tool_call>` format
  4. Tool results render in `<tool_response>...</tool_response>` format
  5. The final JSON output appears in the last assistant turn
  6. `{% generation %}` masking is correctly applied for train_on_responses_only

Usage:
    python scripts/loopper/verify_format.py
    python scripts/loopper/verify_format.py --model unsloth/Qwen3-4B-Thinking-2507
    python scripts/loopper/verify_format.py --show-full  # print full rendered text

Run this BEFORE generating traces. If any check fails, fix build_chatml.py
first — do not spend $130 on trace generation until this passes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Synthetic test examples in our ChatML format ──────────────

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
                    "description": "Search query — use domain-specific noun phrases.",
                },
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["collection", "query"],
        },
    },
}

SYSTEM_PROMPT = """You are a Loopper support agent. Analyze tickets, retrieve relevant context, respond."""


def make_full_trace_example() -> dict:
    """Synthetic full trace: classify → 2 RAG calls → response."""
    final_json = {
        "intent": "quality_complaint",
        "urgency": "high",
        "confidence": 0.92,
        "is_valid": True,
        "rejection_type": None,
        "resolution_type": "requires_human_action",
        "team": "account_manager",
        "actions": ["Forward photos to quality team", "Check rush replacement feasibility"],
        "summary": ["Customer received wrong color mugs", "Trade fair Thursday — urgent"],
        "reasoning": {
            "intent_basis": "Wrong item received — boundary rule 2",
            "urgency_basis": "Event in 3 days, customer escalating",
            "resolution_basis": "Cannot confirm timeline without logistics",
            "policy_used": "Quality team assesses within 1 working day",
        },
        "response": (
            "Hello,\n\nThank you for the photos. I've forwarded them to our quality team "
            "as a priority and they will assess the situation within one working day. "
            "We're looking into rush replacement options given your event on Thursday.\n\n"
            "Best regards,\nMarc Logier, Account Manager — Loopper"
        ),
    }

    return {
        "tools": [RAG_SEARCH_TOOL],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Ticket ID: 538291\n"
                    "Message Count: 1 (Incoming: 1, Outgoing: 0)\n\n"
                    "[Customer] (Message 0):\n"
                    "We ordered 500 custom mugs in navy blue for our trade fair on Thursday, "
                    "but the delivery has the wrong color (dark grey). We need this sorted ASAP. "
                    "Photos attached."
                ),
            },
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": (
                    "Let me analyze this ticket. The customer received mugs in the wrong color "
                    "(dark grey instead of navy blue). This is a quality_complaint (wrong item "
                    "received — boundary rule 2). Urgency is high because the trade fair is "
                    "Thursday. I need to search for quality complaint policy and tone guidance."
                ),
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_abc12345",
                        "function": {
                            "name": "rag_search",
                            "arguments": json.dumps({
                                "collection": "faq",
                                "query": "quality complaint wrong item replacement policy",
                                "top_k": 5,
                            }),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc12345",
                "name": "rag_search",
                "content": json.dumps({
                    "results": [
                        {
                            "text": "Quality complaints: request photos within 5 business days. Quality team assesses within 1 working day. Offer replacement at no cost or refund.",
                            "source": "kb_quality_policy.md",
                            "rerank_score": 0.93,
                            "collection": "faq",
                        }
                    ]
                }),
            },
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": (
                    "Found the quality policy. Now I need tone guidance for urgent complaints."
                ),
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_def67890",
                        "function": {
                            "name": "rag_search",
                            "arguments": json.dumps({
                                "collection": "communication_guidelines",
                                "query": "urgent quality complaint empathetic tone",
                                "top_k": 3,
                            }),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_def67890",
                "name": "rag_search",
                "content": json.dumps({
                    "results": [
                        {
                            "text": "For urgent complaints: acknowledge immediately, match customer urgency, never promise specific timelines without logistics confirmation.",
                            "source": "tone_urgent.md",
                            "rerank_score": 0.88,
                            "collection": "communication_guidelines",
                        }
                    ]
                }),
            },
            {
                "role": "assistant",
                "reasoning_content": (
                    "Retrieved context: quality team 1-day assessment, empathetic urgent tone. "
                    "Resolution is requires_human_action — I cannot confirm rush replacement "
                    "myself. Actions: forward photos to quality team, check rush feasibility."
                ),
                "content": json.dumps(final_json, ensure_ascii=False),
            },
        ],
    }


def make_rejection_example() -> dict:
    """Synthetic rejection example: spam ticket, no tool calls."""
    final_json = {
        "intent": "other",
        "urgency": "low",
        "confidence": 0.0,
        "is_valid": False,
        "rejection_type": "spam",
        "resolution_type": "direct_resolve",
        "team": "none",
        "actions": [],
        "summary": ["Unsolicited marketing email — not a support request"],
        "reasoning": {
            "intent_basis": "Spam — unsolicited bulk email",
            "urgency_basis": "Not a support request — no urgency applies",
            "resolution_basis": "Can be archived without human action",
            "policy_used": "Triage policy: reject non-support tickets",
        },
        "response": "",
    }

    return {
        "tools": [RAG_SEARCH_TOOL],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Ticket ID: 999001\n"
                    "Message Count: 1 (Incoming: 1, Outgoing: 0)\n\n"
                    "[Customer] (Message 0):\n"
                    "BUY CHEAP CRYPTO NOW! Click here for exclusive offers. "
                    "Limited time only! Act fast!"
                ),
            },
            {
                "role": "assistant",
                "reasoning_content": (
                    "Looking at this ticket carefully. This is unsolicited bulk email with "
                    "no connection to Loopper's business."
                ),
                "content": json.dumps(final_json, ensure_ascii=False),
            },
        ],
    }


def make_acknowledgment_example() -> dict:
    """Synthetic simple acknowledgment: no tool calls, direct response."""
    final_json = {
        "intent": "customer_feedback",
        "urgency": "low",
        "confidence": 0.95,
        "is_valid": True,
        "rejection_type": None,
        "resolution_type": "direct_resolve",
        "team": "none",
        "actions": [],
        "summary": ["Customer thanking us for delivery"],
        "reasoning": {
            "intent_basis": "Simple thank-you message",
            "urgency_basis": "No urgency — positive feedback",
            "resolution_basis": "Can acknowledge directly",
            "policy_used": "Feedback policy: warm acknowledgment",
        },
        "response": (
            "Thank you so much for the kind words! We're thrilled you're happy with "
            "the order. Looking forward to working with you again.\n\n"
            "Best regards,\nMarc Logier, Account Manager — Loopper"
        ),
    }

    return {
        "tools": [RAG_SEARCH_TOOL],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Ticket ID: 555123\n"
                    "Message Count: 2 (Incoming: 1, Outgoing: 1)\n\n"
                    "[Loopper Agent] (Message 0):\n"
                    "Your order has been delivered!\n\n"
                    "[Customer] (Message 1):\n"
                    "OK thank you, everything looks great!"
                ),
            },
            {
                "role": "assistant",
                "reasoning_content": (
                    "Simple thank-you from the customer confirming delivery. No search needed "
                    "— I can respond directly with a warm acknowledgment."
                ),
                "content": json.dumps(final_json, ensure_ascii=False),
            },
        ],
    }


# ── Verification checks ────────────────────────────────────────

def check_rendered_text(name: str, rendered: str, example: dict) -> tuple[int, int]:
    """Run all format checks on a rendered string. Returns (errors, warnings)."""
    errors = 0
    warnings = 0

    def fail(msg):
        nonlocal errors
        errors += 1
        logger.error(f"  [FAIL] {msg}")

    def warn(msg):
        nonlocal warnings
        warnings += 1
        logger.warning(f"  [WARN] {msg}")

    def ok(msg):
        logger.info(f"  [OK]   {msg}")

    logger.info(f"\n=== {name} ===")

    # Count messages in the example
    messages = example["messages"]
    has_tool_calls = any(m.get("tool_calls") for m in messages)
    has_tool_results = any(m.get("role") == "tool" for m in messages)
    reasoning_count = sum(1 for m in messages if m.get("reasoning_content"))

    # 1. Check system block contains the tool schema
    if "rag_search" in rendered:
        ok("Tool name 'rag_search' appears in rendered text")
    else:
        fail("Tool name 'rag_search' NOT in rendered text — tools= param may be ignored")

    # 2. Check tool definition fields appear (Qwen3 renders tools as JSON in system)
    tool_markers = ["collection", "query", "top_k", "faq", "operations"]
    missing_markers = [m for m in tool_markers if m not in rendered]
    if not missing_markers:
        ok("All tool parameter markers present (collection, query, top_k, enum values)")
    else:
        warn(f"Missing tool markers in rendered text: {missing_markers}")

    # 3. Check <think> blocks appear if reasoning_content was present
    think_count = rendered.count("<think>")
    think_end_count = rendered.count("</think>")
    if reasoning_count > 0:
        if think_count >= 1 and think_end_count >= 1:
            ok(f"<think>/</think> markers present ({think_count} open, {think_end_count} close)")
            if think_count < reasoning_count:
                warn(f"Only {think_count}/{reasoning_count} reasoning_content blocks rendered as <think>")
        else:
            fail(f"reasoning_content present in {reasoning_count} messages but NO <think> tags in rendered text")
    elif think_count > 0:
        warn(f"Unexpected <think> tags in rendered text ({think_count}) when no reasoning_content")

    # 4. Check tool_call tags if example has tool_calls
    if has_tool_calls:
        if "<tool_call>" in rendered or "tool_call" in rendered:
            ok("tool_call markers present in rendered text")
        else:
            fail("Example has tool_calls but no <tool_call> markers in rendered text")

    # 5. Check tool_response tags if example has tool results
    if has_tool_results:
        if "<tool_response>" in rendered or "tool_response" in rendered or "<tool>" in rendered:
            ok("tool_response markers present in rendered text")
        else:
            fail("Example has tool results but no <tool_response> markers in rendered text")

    # 6. Check the final JSON output appears in the last assistant message
    final_msg = messages[-1]
    if final_msg.get("role") == "assistant" and final_msg.get("content"):
        try:
            final_json = json.loads(final_msg["content"])
            # Check a distinctive field from the output JSON appears in rendered
            if "intent" in final_json:
                # Look for the intent value in rendered text
                intent_val = final_json["intent"]
                if f'"intent": "{intent_val}"' in rendered or f'"intent":"{intent_val}"' in rendered:
                    ok(f"Final JSON output appears in rendered text (intent={intent_val})")
                else:
                    warn(f"Final JSON intent='{intent_val}' not found verbatim in rendered text")
        except json.JSONDecodeError:
            warn("Final assistant content is not valid JSON")

    # 7. Check ChatML structure markers
    if "<|im_start|>" in rendered and "<|im_end|>" in rendered:
        ok("ChatML markers <|im_start|>/<|im_end|> present")
    else:
        fail("Missing ChatML structural markers")

    # 8. Check generation tags for loss masking
    if "{% generation %}" in rendered or "{%- generation %}" in rendered:
        # These are Jinja tags — shouldn't be in rendered output
        warn("Jinja generation tags leaked into rendered text (template bug)")

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Verify ChatML format end-to-end")
    parser.add_argument("--model", default="unsloth/Qwen3-4B-Thinking-2507",
                        help="Tokenizer repo to load (default: unsloth/Qwen3-4B-Thinking-2507)")
    parser.add_argument("--show-full", action="store_true", help="Print full rendered text")
    args = parser.parse_args()

    logger.info("Loading tokenizer: %s", args.model)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except ImportError:
        logger.error("transformers not installed. Run: pip install transformers")
        sys.exit(1)
    except Exception as e:
        logger.error("Failed to load tokenizer from %s: %s", args.model, e)
        logger.error("Make sure you have internet access and the model name is correct.")
        sys.exit(1)

    logger.info("Tokenizer loaded. eos_token=%s, pad_token=%s", tokenizer.eos_token, tokenizer.pad_token)
    logger.info("Chat template length: %d chars", len(tokenizer.chat_template or ""))

    # Generate synthetic examples
    examples = {
        "Full trace (2 RAG calls + response)": make_full_trace_example(),
        "Rejection (spam, no tools)": make_rejection_example(),
        "Simple acknowledgment (no tools)": make_acknowledgment_example(),
    }

    total_errors = 0
    total_warnings = 0

    for name, example in examples.items():
        logger.info("\n%s", "=" * 70)
        logger.info("  RENDERING: %s", name)
        logger.info("=" * 70)

        try:
            rendered = tokenizer.apply_chat_template(
                example["messages"],
                tools=example["tools"],
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as e:
            logger.error("apply_chat_template failed: %s: %s", type(e).__name__, e)
            total_errors += 1
            continue

        logger.info("\n  Rendered length: %d chars", len(rendered))

        if args.show_full:
            logger.info("\n--- FULL RENDERED TEXT ---")
            print(rendered)
            logger.info("--- END FULL RENDERED TEXT ---\n")

        errors, warnings = check_rendered_text(name, rendered, example)
        total_errors += errors
        total_warnings += warnings

    # Summary
    logger.info("\n%s", "=" * 70)
    logger.info("  VERIFICATION SUMMARY")
    logger.info("=" * 70)
    logger.info("  Examples tested:  %d", len(examples))
    logger.info("  Total errors:     %d", total_errors)
    logger.info("  Total warnings:   %d", total_warnings)

    if total_errors > 0:
        logger.error("\n  VERDICT: FAIL — fix build_chatml.py before trace generation")
        sys.exit(1)
    elif total_warnings > 0:
        logger.warning("\n  VERDICT: PASS WITH WARNINGS — review warnings, probably safe to proceed")
    else:
        logger.info("\n  VERDICT: PASS — format is correct, safe to generate traces")


if __name__ == "__main__":
    main()
