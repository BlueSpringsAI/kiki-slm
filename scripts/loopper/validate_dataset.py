#!/usr/bin/env python3
"""Step 5: Validate the generated ChatML dataset before training.

Runs quality checks on every training example to catch data issues before
they corrupt training. Produces a validation report with pass/fail counts.

Checks:
  1. JSON validity of all assistant outputs
  2. Schema compliance (all 11 required fields)
  3. reasoning_content presence, count, and quality (Qwen3-Thinking format)
  4. Tool call validity (collection names, query content, id linkage)
  5. Role ordering (correct assistant→tool pairing)
  6. Category distribution balance
  7. Response quality (signature, banned phrases, length)
  8. Token length estimation (warns on examples that may exceed max_seq_length)
  9. Cross-file train/eval leakage detection
  10. is_valid / rejection_type consistency (intent="other" for rejections)

Usage:
    python scripts/loopper/validate_dataset.py
    python scripts/loopper/validate_dataset.py --input data/chatml/train.jsonl
    python scripts/loopper/validate_dataset.py --max-seq-length 4096
    python scripts/loopper/validate_dataset.py --strict
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Default path (from configs/loopper_pipeline.yaml or project-relative) ──
from scripts.loopper.config import get_default_paths as _get_paths
_PATHS = _get_paths()
CHATML_DIR = _PATHS["chatml_output"]

# ── Valid values ──────────────────────────────────────────────
# NOTE: "intent" values are the business categories only. Rejection examples
# use intent="other" with rejection_type as the discriminator (matches the
# teacher agent's native schema). The agent's ValidationReasoning Pydantic
# Literal has exactly 5 rejection types: spam, misdirected, newsletter,
# auto_reply, unrelated. system_notification is NOT in the agent's schema.
VALID_INTENTS = {
    "new_order_inquiry", "design_update", "payment_confirmation",
    "delivery_issue", "refund_request", "order_cancellation",
    "quality_complaint", "sample_request", "price_negotiation",
    "customer_feedback", "other",
}

VALID_URGENCIES = {"low", "medium", "high", "critical"}
VALID_RESOLUTION_TYPES = {"direct_resolve", "requires_human_action", "needs_escalation", "needs_more_info"}
VALID_TEAMS = {"design", "logistics", "finance", "account_manager", "none"}
VALID_COLLECTIONS = {
    "faq", "operations", "communication_guidelines", "supplier_data",
    # Actual names the teacher agent uses (RAG service collection aliases)
    "customer_policy_faq", "sales_operations_playbook", "supplier_intelligence",
}
# Rejection types come from the agent's ValidationReasoning Literal (5 values).
VALID_REJECTION_TYPES = {"spam", "misdirected", "newsletter", "auto_reply", "unrelated"}

REQUIRED_OUTPUT_FIELDS = {
    "intent", "urgency", "confidence", "is_valid", "rejection_type",
    "resolution_type", "team", "actions", "summary", "reasoning", "response",
}

BANNED_RESPONSE_PHRASES = [
    "n'hésitez pas", "à votre disposition", "don't hesitate to contact",
    "at your service", "for your review", "zur Verfügung",
    "as an ai", "as a language model", "i'm an ai assistant",
    "i don't have access to", "rest assured", "please be advised",
    "i apologize for any inconvenience",
]

DEFAULT_MAX_SEQ_LENGTH = 8192


def extract_final_json(content: str) -> dict | None:
    """Extract the JSON object from the final assistant message.

    Searches for the first { AFTER the last </think> block to avoid
    matching braces inside response text.
    """
    think_end = content.rfind("</think>")
    search_from = think_end + len("</think>") if think_end >= 0 else 0

    json_start = content.find("{", search_from)
    if json_start < 0:
        return None

    # Try progressively larger substrings from json_start
    for end in range(json_start + 2, len(content) + 1):
        candidate = content[json_start:end]
        if candidate.count("{") == candidate.count("}"):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return None


def estimate_token_count(messages: list[dict]) -> int:
    """Rough token estimate (~1 token per 3.5 chars for English content)."""
    total_chars = 0
    for m in messages:
        total_chars += len(m.get("content", "") or "")
        for tc in m.get("tool_calls", []):
            total_chars += len(json.dumps(tc))
    return int(total_chars / 3.5)


def extract_ticket_id(messages: list[dict]) -> str | None:
    """Extract ticket ID from the user message."""
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content", "")
            match = re.search(r"Ticket ID:\s*(\S+)", content)
            if match:
                return match.group(1)
    return None


def validate_example(example: dict, idx: int, max_seq_length: int) -> list[dict]:
    """Validate a single ChatML example. Returns list of issues."""
    issues = []
    messages = example.get("messages", [])

    def issue(severity: str, check: str, detail: str):
        issues.append({"idx": idx, "severity": severity, "check": check, "detail": detail})

    # ── 1. Basic structure ──
    if not messages:
        issue("error", "structure", "Empty messages array")
        return issues

    roles = [m.get("role") for m in messages]

    if roles[0] != "system":
        issue("error", "structure", f"First message must be 'system', got '{roles[0]}'")
    if len(roles) < 3:
        issue("error", "structure", f"Need at least 3 messages, got {len(roles)}")
        return issues
    if roles[1] != "user":
        issue("error", "structure", f"Second message must be 'user', got '{roles[1]}'")
    if roles[-1] != "assistant":
        issue("error", "structure", f"Last message must be 'assistant', got '{roles[-1]}'")

    # System only at position 0
    for i in range(1, len(roles)):
        if roles[i] == "system":
            issue("error", "structure", f"System message at position {i} (only allowed at 0)")

    # ── 2. Role ordering (tool-call/tool-result pairing) ──
    for i in range(len(messages)):
        if messages[i].get("role") == "tool":
            if i == 0:
                issue("error", "role_order", "Tool message at position 0")
            elif messages[i - 1].get("role") != "assistant" or not messages[i - 1].get("tool_calls"):
                issue("error", "role_order", f"Tool message at position {i} not preceded by assistant with tool_calls")

        if (messages[i].get("role") == "assistant" and i > 0
                and messages[i - 1].get("role") == "assistant"
                and not messages[i - 1].get("tool_calls")):
            # Two consecutive assistants without tool_calls on the first = structural error
            # (unless it's the correction pattern: assistant wrong → user correction → assistant right)
            if i >= 2 and messages[i - 2].get("role") == "user":
                pass  # correction pattern
            else:
                issue("warning", "role_order", f"Consecutive assistant messages at {i-1} and {i}")

    # ── 3. Tool call validity ──
    has_tool_call = False
    for msg in messages:
        for tc in msg.get("tool_calls", []):
            has_tool_call = True
            func = tc.get("function", {})
            name = func.get("name", "")
            tc_id = tc.get("id")

            if name != "rag_search":
                issue("error", "tool_name", f"Unknown tool: '{name}'")
            if not tc_id:
                issue("warning", "tool_call_id", "Tool call missing 'id' field")
            if tc.get("type") != "function":
                issue("warning", "tool_type", f"Tool call type should be 'function', got '{tc.get('type')}'")

            try:
                args_str = func.get("arguments", "{}")
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
                collection = args.get("collection", "")
                query = args.get("query", "")

                if collection not in VALID_COLLECTIONS:
                    issue("error", "tool_collection", f"Invalid collection: '{collection}'")
                if not query or len(query.strip()) < 3:
                    issue("error", "tool_query", f"Empty or too short query: '{query}'")
            except json.JSONDecodeError:
                issue("error", "tool_args", f"Invalid tool arguments JSON")

    # Tool response id linkage
    for msg in messages:
        if msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id")
            if not tc_id:
                issue("warning", "tool_response_id", "Tool response missing 'tool_call_id'")

    # ── 4. Reasoning content presence and quality ──
    # For Qwen3-Thinking models, reasoning lives in the `reasoning_content`
    # field on assistant messages (NOT in <think> tags inside content).
    # The Unsloth-patched chat template renders reasoning_content → <think>
    # blocks in the final tokenized string, but in our training data they
    # are a separate field.
    #
    # We also support the legacy <think>...</think> format inside content
    # for backward compatibility, but new data should use reasoning_content.
    reasoning_blocks = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        # Primary: reasoning_content field (Qwen3-Thinking format)
        rc = msg.get("reasoning_content")
        if rc:
            reasoning_blocks.append(str(rc).strip())
            continue
        # Legacy fallback: <think>...</think> tags inside content
        content = msg.get("content", "") or ""
        for match in re.finditer(r"<think>(.*?)</think>", content, re.DOTALL):
            reasoning_blocks.append(match.group(1).strip())

    if not reasoning_blocks:
        issue("error", "reasoning_missing",
              "No reasoning_content field or <think> block found in any assistant message")
    else:
        # Quality checks on first reasoning block
        if len(reasoning_blocks[0]) < 30:
            issue("warning", "reasoning_quality",
                  f"First reasoning block too short ({len(reasoning_blocks[0])} chars)")

        # First reasoning block should contain intent analysis or search plan
        first_reasoning_lower = reasoning_blocks[0].lower()
        has_reasoning = any(kw in first_reasoning_lower for kw in [
            "intent", "customer", "search", "need to", "looking at",
            "analyzing", "ticket", "request", "loopper", "primary",
        ])
        if not has_reasoning:
            issue("warning", "reasoning_quality",
                  "First reasoning block lacks reasoning keywords")

        # Check for output schema leakage in reasoning
        # (reasoning should explain WHY, not mirror the JSON output fields)
        for rb in reasoning_blocks:
            if '"intent":' in rb or '"response":' in rb or '"rejection_type":' in rb:
                issue("warning", "reasoning_leakage",
                      "Reasoning block contains JSON output field keys")

        # For examples with tool calls, should have at least 2 reasoning blocks
        # (one before first tool call, one before final output)
        if has_tool_call and len(reasoning_blocks) < 2:
            issue("warning", "reasoning_count",
                  f"Only {len(reasoning_blocks)} reasoning block(s) with tool calls (expected >= 2)")

    # ── 5. Final output JSON ──
    final_output = None
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            final_output = extract_final_json(msg["content"])
            break

    if final_output is None:
        issue("error", "output_missing", "No valid JSON output in final assistant message")
        return issues

    # Required fields
    for field in REQUIRED_OUTPUT_FIELDS:
        if field not in final_output:
            issue("error", "output_field", f"Missing required field: '{field}'")

    # Field value validation
    intent = final_output.get("intent", "")
    if intent and intent not in VALID_INTENTS:
        issue("error", "output_intent", f"Invalid intent: '{intent}'")

    urgency = final_output.get("urgency", "")
    if not urgency or urgency not in VALID_URGENCIES:
        issue("error", "output_urgency", f"Invalid or empty urgency: '{urgency}'")

    resolution = final_output.get("resolution_type", "")
    if not resolution or resolution not in VALID_RESOLUTION_TYPES:
        issue("error", "output_resolution", f"Invalid or empty resolution_type: '{resolution}'")

    team = final_output.get("team", "")
    if team and team not in VALID_TEAMS:
        issue("warning", "output_team", f"Invalid team: '{team}'")

    confidence = final_output.get("confidence", -1)
    if not isinstance(confidence, (int, float)):
        issue("error", "output_confidence", f"Confidence is not a number: {type(confidence)}")
    elif not (0.0 <= confidence <= 1.0):
        issue("warning", "output_confidence", f"Confidence out of range: {confidence}")

    # is_valid / rejection_type consistency
    is_valid = final_output.get("is_valid")
    rejection_type = final_output.get("rejection_type")
    if is_valid is False and not rejection_type:
        issue("warning", "validity_consistency", "is_valid=False but rejection_type is null")
    if is_valid is True and rejection_type:
        issue("warning", "validity_consistency", f"is_valid=True but rejection_type='{rejection_type}'")

    # actions and summary should be lists
    if "actions" in final_output and not isinstance(final_output["actions"], list):
        issue("warning", "output_type", f"'actions' should be a list, got {type(final_output['actions'])}")
    if "summary" in final_output and not isinstance(final_output["summary"], list):
        issue("warning", "output_type", f"'summary' should be a list, got {type(final_output['summary'])}")

    # reasoning should be a dict with subfields
    reasoning = final_output.get("reasoning")
    if reasoning and isinstance(reasoning, dict):
        for key in ["intent_basis", "urgency_basis", "resolution_basis", "policy_used"]:
            if key not in reasoning:
                issue("warning", "output_reasoning", f"Missing reasoning subfield: '{key}'")
    elif "reasoning" in final_output:
        issue("warning", "output_reasoning", f"'reasoning' should be a dict, got {type(reasoning)}")

    # ── 6. Response quality ──
    # Coerce None → "" defensively. Rejection examples set response="" intentionally.
    response_text = final_output.get("response") or ""
    # Rejection examples are keyed by is_valid=False (matches teacher schema).
    # Rejection type is checked separately via rejection_type field.
    is_rejection = final_output.get("is_valid") is False

    if not is_rejection:
        if not response_text or len(response_text.strip()) < 20:
            issue("warning", "response_empty", f"Response too short ({len(response_text)} chars)")

        if response_text and "Marc Logier" not in response_text:
            issue("warning", "response_signature", "Response not signed as Marc Logier")

    for phrase in BANNED_RESPONSE_PHRASES:
        if phrase.lower() in response_text.lower():
            issue("warning", "response_banned", f"Contains banned phrase: '{phrase}'")

    # Validate rejection_type matches the allowed set when is_valid=False
    if is_rejection:
        rt = final_output.get("rejection_type")
        if not rt:
            issue("error", "rejection_type_missing", "is_valid=False but rejection_type is empty")
        elif rt not in VALID_REJECTION_TYPES:
            issue("error", "rejection_type_invalid", f"Invalid rejection_type: '{rt}'")

    # ── 7. RAG enforcement ──
    is_acknowledgment = intent == "customer_feedback" and urgency == "low"
    is_correction = any(
        m.get("role") == "user" and "must search" in (m.get("content", "") or "").lower()
        for m in messages
    )
    if not has_tool_call and not is_acknowledgment and not is_correction and not is_rejection:
        issue("warning", "no_rag", "No rag_search call and not an acknowledgment/correction/rejection example")

    # ── 8. Token length ──
    token_estimate = estimate_token_count(messages)
    if token_estimate > max_seq_length:
        issue("error", "token_length", f"Estimated {token_estimate} tokens exceeds max_seq_length {max_seq_length}")
    elif token_estimate > max_seq_length * 0.9:
        issue("warning", "token_length", f"Estimated {token_estimate} tokens — close to max_seq_length {max_seq_length}")

    return issues


def validate_dataset(input_path: str, max_seq_length: int) -> dict:
    """Validate an entire JSONL dataset file."""
    path = Path(input_path)
    if not path.exists():
        logger.error("File not found: %s", input_path)
        sys.exit(1)

    examples = []
    parse_errors = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                parse_errors += 1

    logger.info("Loaded %d examples from %s (%d parse errors)", len(examples), input_path, parse_errors)

    all_issues = []
    category_counts = Counter()
    ticket_ids = []

    for i, example in enumerate(examples):
        issues = validate_example(example, i, max_seq_length)
        all_issues.extend(issues)

        msgs = example.get("messages", [])
        tid = extract_ticket_id(msgs)
        if tid:
            ticket_ids.append(tid)

        for msg in reversed(msgs):
            if msg.get("role") == "assistant" and msg.get("content"):
                output = extract_final_json(msg["content"])
                if output:
                    category_counts[output.get("intent", "unknown")] += 1
                break

    errors = [i for i in all_issues if i["severity"] == "error"]
    warnings = [i for i in all_issues if i["severity"] == "warning"]
    check_counts = Counter(i["check"] for i in all_issues)

    # Category distribution checks
    distribution_issues = []
    # Rejection examples have intent="other", so "other" should always exist.
    # The 11 business intents should all be present; flag any missing.
    missing_cats = VALID_INTENTS - set(category_counts.keys())
    if missing_cats:
        distribution_issues.append(f"Missing categories: {missing_cats}")
    for cat, count in category_counts.items():
        # "other" can legitimately be lower (it's the rejection bucket and catch-all)
        if cat in VALID_INTENTS and count < 5 and cat != "other":
            distribution_issues.append(f"Category '{cat}' has only {count} examples (< 5)")
    total_examples = len(examples)
    if total_examples > 0:
        for cat, count in category_counts.items():
            # Allow "other" to exceed 40% since it holds all rejection examples
            if count / total_examples > 0.40 and cat != "other":
                distribution_issues.append(f"Category '{cat}' represents {count/total_examples:.0%} of dataset (> 40%)")

    # Duplicate ticket detection
    tid_counts = Counter(ticket_ids)
    duplicates = {tid: cnt for tid, cnt in tid_counts.items() if cnt > 2}

    error_count_by_example = Counter(i["idx"] for i in errors)
    passed = total_examples - len(error_count_by_example)

    return {
        "file": str(path),
        "total_examples": total_examples,
        "parse_errors": parse_errors,
        "validation_errors": len(errors),
        "validation_warnings": len(warnings),
        "passed": passed,
        "issues_by_check": dict(check_counts.most_common()),
        "category_distribution": dict(category_counts.most_common()),
        "distribution_issues": distribution_issues,
        "duplicate_tickets": duplicates,
        "ticket_ids": ticket_ids,
        "error_details": errors[:100],
        "warning_details": warnings[:100],
    }


def check_train_eval_leakage(train_result: dict, eval_result: dict) -> list[str]:
    """Check for ticket ID overlap between train and eval sets."""
    train_ids = set(train_result.get("ticket_ids", []))
    eval_ids = set(eval_result.get("ticket_ids", []))
    overlap = train_ids & eval_ids
    if overlap:
        return [f"LEAKAGE: {len(overlap)} ticket IDs appear in both train and eval: {list(overlap)[:10]}..."]
    return []


def print_report(results: dict):
    """Print the validation report."""
    total = results["total_examples"]

    logger.info("")
    logger.info("=" * 60)
    logger.info("  DATASET VALIDATION REPORT")
    logger.info("=" * 60)
    logger.info("  File:              %s", results["file"])
    logger.info("  Total examples:    %d", total)
    logger.info("  Parse errors:      %d", results["parse_errors"])
    logger.info("  Validation errors: %d", results["validation_errors"])
    logger.info("  Warnings:          %d", results["validation_warnings"])
    logger.info("")
    logger.info("  PASSED: %d / %d (%.1f%%)", results["passed"], total,
                results["passed"] / total * 100 if total else 0)
    logger.info("")

    if results["issues_by_check"]:
        logger.info("  Issues by check:")
        for check, count in sorted(results["issues_by_check"].items(), key=lambda x: -x[1]):
            logger.info("    %-30s %5d", check, count)
        logger.info("")

    if results["category_distribution"]:
        logger.info("  Category distribution:")
        for cat, count in sorted(results["category_distribution"].items(), key=lambda x: -x[1]):
            pct = count / total * 100 if total else 0
            logger.info("    %-25s %5d (%5.1f%%)", cat, count, pct)
        logger.info("")

    if results["distribution_issues"]:
        logger.info("  Distribution warnings:")
        for d in results["distribution_issues"]:
            logger.info("    - %s", d)
        logger.info("")

    if results["duplicate_tickets"]:
        logger.info("  Duplicate tickets (>2 occurrences): %d", len(results["duplicate_tickets"]))
        logger.info("")

    if results["error_details"]:
        logger.info("  First 10 errors:")
        for err in results["error_details"][:10]:
            logger.info("    Example %d [%s]: %s", err["idx"], err["check"], err["detail"])
        logger.info("")

    if results["validation_errors"] == 0:
        logger.info("  VERDICT: PASS")
    else:
        logger.info("  VERDICT: FAIL — %d errors must be fixed", results["validation_errors"])


def main():
    parser = argparse.ArgumentParser(description="Validate ChatML training dataset")
    parser.add_argument("--input-dir", default=CHATML_DIR)
    parser.add_argument("--input", default=None, help="Validate a specific JSONL file")
    parser.add_argument("--output-report", default=None)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    args = parser.parse_args()

    if args.input:
        files = [Path(args.input)]
    else:
        input_dir = Path(args.input_dir)
        files = sorted(input_dir.glob("*.jsonl"))

    if not files:
        logger.error("No JSONL files found")
        sys.exit(1)

    all_results = []
    total_errors = 0

    for filepath in files:
        logger.info("Validating %s...", filepath)
        results = validate_dataset(str(filepath), args.max_seq_length)
        print_report(results)
        all_results.append(results)
        total_errors += results["validation_errors"]
        if args.strict:
            total_errors += results["validation_warnings"]

    # Cross-file leakage check
    if len(all_results) >= 2:
        train_result = next((r for r in all_results if "train" in r["file"]), None)
        eval_result = next((r for r in all_results if "eval" in r["file"]), None)
        if train_result and eval_result:
            leakage = check_train_eval_leakage(train_result, eval_result)
            if leakage:
                for msg in leakage:
                    logger.error("  %s", msg)
                total_errors += len(leakage)

    # Save report
    report_dir = Path(args.input_dir) if not args.input else Path(args.input).parent
    report_path = Path(args.output_report) if args.output_report else report_dir / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip ticket_ids from saved report (too large)
    save_results = []
    for r in all_results:
        r_copy = {k: v for k, v in r.items() if k != "ticket_ids"}
        save_results.append(r_copy)

    with open(report_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    logger.info("Report saved to %s", report_path)

    if total_errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
