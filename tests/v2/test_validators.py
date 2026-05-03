"""Validator round-trip tests. For every error_kind:
  - a positive case (passes)
  - a negative case (fails, returns the right kind)

Plus aggregator priority and the v1 trailing-comma round-trip — the bug that
went undetected in v1 must now fail loudly."""
from __future__ import annotations

import copy
import json

import pytest

from scripts.loopper.v2.error_kinds import ErrorKind
from scripts.loopper.v2.validators import (
    ValidationFailure,
    check_chatml_banned_response_phrase,
    check_chatml_bad_role_order,
    check_chatml_missing_required_field,
    check_chatml_reasoning_leakage,
    check_chatml_token_overflow_estimate,
    check_chatml_tool_call_id_mismatch,
    check_distribution_below_min,
    check_eval_underrepresented,
    check_non_target_language,
    check_raw_duplicate,
    check_raw_no_content,
    check_raw_too_short,
    check_rendered_tool_call_json_valid,
    check_rendered_trailing_comma,
    check_train_eval_leakage,
    check_trace_empty_response_on_valid,
    check_trace_intent_mismatch,
    check_trace_invalid_collection_name,
    check_trace_malformed_final_json,
    check_trace_missing_followup_pattern,
    check_trace_no_tool_calls_on_valid,
    check_trace_template_response,
    validate_chatml,
    validate_raw_ticket,
    validate_trace,
)


# ──────────────────────────────────────────────────────────────────────
# Stage 0: source
# ──────────────────────────────────────────────────────────────────────

def test_check_raw_too_short_passes(good_raw_ticket):
    assert check_raw_too_short(good_raw_ticket) is None


def test_check_raw_too_short_fails():
    bad = {"messages": [{"clean_body": "hi"}]}
    f = check_raw_too_short(bad)
    assert f and f.kind == ErrorKind.RAW_TOO_SHORT


def test_check_raw_no_content_passes(good_raw_ticket):
    assert check_raw_no_content(good_raw_ticket) is None


def test_check_raw_no_content_fails_on_empty():
    f = check_raw_no_content({"messages": []})
    assert f and f.kind == ErrorKind.RAW_NO_CONTENT


def test_check_raw_no_content_fails_on_whitespace():
    f = check_raw_no_content({"messages": [{"clean_body": "   "}, {"clean_body": ""}]})
    assert f and f.kind == ErrorKind.RAW_NO_CONTENT


def test_check_non_target_language_passes_english(good_raw_ticket):
    assert check_non_target_language(good_raw_ticket) is None


def test_check_non_target_language_fails_french():
    f = check_non_target_language({"languages": ["fr"], "messages": []})
    assert f and f.kind == ErrorKind.NON_TARGET_LANGUAGE


def test_check_non_target_language_passes_unknown():
    """Unknown / empty language should pass — the heavy detector runs separately."""
    assert check_non_target_language({"languages": [None], "messages": []}) is None
    assert check_non_target_language({"languages": ["unknown"], "messages": []}) is None


def test_check_raw_duplicate(good_raw_ticket):
    seen: set[str] = set()
    assert check_raw_duplicate(good_raw_ticket, seen_hashes=seen) is None
    f = check_raw_duplicate(good_raw_ticket, seen_hashes=seen)  # second call: same hash
    assert f and f.kind == ErrorKind.RAW_DUPLICATE


def test_validate_raw_ticket_first_failure_wins():
    """The aggregator should return RAW_NO_CONTENT, not RAW_TOO_SHORT (cheaper / more specific)."""
    bad = {"messages": []}
    f = validate_raw_ticket(bad)
    assert f and f.kind == ErrorKind.RAW_NO_CONTENT


def test_validate_raw_ticket_passes(good_raw_ticket):
    assert validate_raw_ticket(good_raw_ticket, seen_hashes=set()) is None


# ──────────────────────────────────────────────────────────────────────
# Stage 1: trace
# ──────────────────────────────────────────────────────────────────────

def test_check_trace_no_tool_calls_on_valid_passes(good_trace):
    assert check_trace_no_tool_calls_on_valid(good_trace) is None


def test_check_trace_no_tool_calls_on_valid_fails(good_trace):
    bad = copy.deepcopy(good_trace)
    bad["retrieval"]["tool_calls"] = []
    f = check_trace_no_tool_calls_on_valid(bad)
    assert f and f.kind == ErrorKind.TRACE_NO_TOOL_CALLS_ON_VALID


def test_check_trace_no_tool_calls_skipped_for_rejection(good_trace):
    """Rejection paths (is_valid=false) legitimately have no tool calls."""
    bad = copy.deepcopy(good_trace)
    bad["triage"]["is_valid"] = False
    bad["retrieval"]["tool_calls"] = []
    assert check_trace_no_tool_calls_on_valid(bad) is None


def test_check_trace_invalid_collection_name_passes(good_trace):
    assert check_trace_invalid_collection_name(good_trace) is None


def test_check_trace_invalid_collection_name_fails(good_trace):
    bad = copy.deepcopy(good_trace)
    bad["retrieval"]["tool_calls"][0]["args"]["collection"] = "made_up_collection"
    f = check_trace_invalid_collection_name(bad)
    assert f and f.kind == ErrorKind.TRACE_INVALID_COLLECTION_NAME
    assert "made_up_collection" in f.detail["unknown_collections"]


def test_check_trace_invalid_collection_name_accepts_alias(good_trace):
    """customer_policy_faq is a known alias, not invalid."""
    bad = copy.deepcopy(good_trace)
    bad["retrieval"]["tool_calls"][0]["args"]["collection"] = "customer_policy_faq"
    assert check_trace_invalid_collection_name(bad) is None


def test_check_trace_malformed_final_json_passes(good_trace):
    assert check_trace_malformed_final_json(good_trace) is None


def test_check_trace_malformed_final_json_fails_when_missing_required(good_trace):
    bad = copy.deepcopy(good_trace)
    del bad["response"]["resolution_type"]
    f = check_trace_malformed_final_json(bad)
    assert f and f.kind == ErrorKind.TRACE_MALFORMED_FINAL_JSON


def test_check_trace_malformed_final_json_fails_when_response_not_dict(good_trace):
    bad = copy.deepcopy(good_trace)
    bad["response"] = "oops a string"
    f = check_trace_malformed_final_json(bad)
    assert f and f.kind == ErrorKind.TRACE_MALFORMED_FINAL_JSON


def test_check_trace_template_response_fails(good_trace):
    bad = copy.deepcopy(good_trace)
    bad["response"]["response_english"] = "Thanks, I'll check shortly and get back to you."
    f = check_trace_template_response(bad)
    assert f and f.kind == ErrorKind.TRACE_TEMPLATE_RESPONSE


def test_check_trace_template_response_passes_when_specific(good_trace):
    """If the response has KB-specific markers, even with template phrases, it passes."""
    bad = copy.deepcopy(good_trace)
    bad["response"]["response_english"] = (
        "I'll check shortly. In the meantime, please send a photo and your order number."
    )
    assert check_trace_template_response(bad) is None


def test_check_trace_empty_response_on_valid_fails(good_trace):
    bad = copy.deepcopy(good_trace)
    bad["response"]["response_english"] = "ok"
    f = check_trace_empty_response_on_valid(bad)
    assert f and f.kind == ErrorKind.TRACE_EMPTY_RESPONSE_ON_VALID


def test_check_trace_intent_mismatch_passes(good_trace):
    assert check_trace_intent_mismatch(good_trace, target_intent="quality_complaint") is None


def test_check_trace_intent_mismatch_fails(good_trace):
    f = check_trace_intent_mismatch(good_trace, target_intent="refund_request")
    assert f and f.kind == ErrorKind.TRACE_INTENT_MISMATCH
    assert f.detail["expected"] == "refund_request"
    assert f.detail["detected"] == "quality_complaint"


def test_check_trace_missing_followup_pattern_fails(good_trace):
    bad = copy.deepcopy(good_trace)
    # quality_complaint expects "photo|image|picture"; remove all of them.
    bad["response"]["response_english"] = "Sorry to hear that. Let me look into this."
    f = check_trace_missing_followup_pattern(bad, target_intent="quality_complaint")
    assert f and f.kind == ErrorKind.TRACE_MISSING_FOLLOWUP_PATTERN


def test_validate_trace_passes(good_trace):
    assert validate_trace(good_trace, target_intent="quality_complaint") is None


def test_validate_trace_priority_malformed_first(good_trace):
    """Malformed JSON should be reported before content checks like template."""
    bad = copy.deepcopy(good_trace)
    bad["response"] = None  # both malformed AND would also flag empty response, etc.
    f = validate_trace(bad, target_intent="quality_complaint")
    assert f and f.kind == ErrorKind.TRACE_MALFORMED_FINAL_JSON


# ──────────────────────────────────────────────────────────────────────
# Stage 3: chatml structural
# ──────────────────────────────────────────────────────────────────────

def test_check_chatml_bad_role_order_passes(good_chatml_example):
    assert check_chatml_bad_role_order(good_chatml_example) is None


def test_check_chatml_bad_role_order_fails_first_not_system(good_chatml_example):
    bad = copy.deepcopy(good_chatml_example)
    bad["messages"][0]["role"] = "user"
    f = check_chatml_bad_role_order(bad)
    assert f and f.kind == ErrorKind.CHATML_BAD_ROLE_ORDER


def test_check_chatml_bad_role_order_fails_unmatched_tool(good_chatml_example):
    bad = copy.deepcopy(good_chatml_example)
    # Insert a tool message without preceding assistant tool_calls.
    bad["messages"].insert(2, {"role": "tool", "tool_call_id": "x", "content": "stale"})
    f = check_chatml_bad_role_order(bad)
    assert f and f.kind == ErrorKind.CHATML_BAD_ROLE_ORDER


def test_check_chatml_tool_call_id_mismatch_passes(good_chatml_example):
    assert check_chatml_tool_call_id_mismatch(good_chatml_example) is None


def test_check_chatml_tool_call_id_mismatch_fails(good_chatml_example):
    bad = copy.deepcopy(good_chatml_example)
    # Tool message references nonexistent id.
    for m in bad["messages"]:
        if m.get("role") == "tool":
            m["tool_call_id"] = "wrong_id"
    f = check_chatml_tool_call_id_mismatch(bad)
    assert f and f.kind == ErrorKind.CHATML_TOOL_CALL_ID_MISMATCH


def test_check_chatml_missing_required_field_passes(good_chatml_example):
    assert check_chatml_missing_required_field(good_chatml_example) is None


def test_check_chatml_missing_required_field_fails(good_chatml_example):
    bad = copy.deepcopy(good_chatml_example)
    final = bad["messages"][-1]
    obj = json.loads(final["content"])
    del obj["response"]
    final["content"] = json.dumps(obj)
    f = check_chatml_missing_required_field(bad)
    assert f and f.kind == ErrorKind.CHATML_MISSING_REQUIRED_FIELD
    assert "response" in f.detail["missing"]


def test_check_chatml_missing_required_field_fails_invalid_intent(good_chatml_example):
    bad = copy.deepcopy(good_chatml_example)
    final = bad["messages"][-1]
    obj = json.loads(final["content"])
    obj["intent"] = "made_up_intent"
    final["content"] = json.dumps(obj)
    f = check_chatml_missing_required_field(bad)
    assert f and f.kind == ErrorKind.CHATML_MISSING_REQUIRED_FIELD


def test_check_chatml_banned_response_phrase_passes(good_chatml_example):
    assert check_chatml_banned_response_phrase(good_chatml_example) is None


def test_check_chatml_banned_response_phrase_fails(good_chatml_example):
    bad = copy.deepcopy(good_chatml_example)
    final = bad["messages"][-1]
    obj = json.loads(final["content"])
    obj["response"] = "As an AI assistant, I cannot help with this."
    final["content"] = json.dumps(obj)
    f = check_chatml_banned_response_phrase(bad)
    assert f and f.kind == ErrorKind.CHATML_BANNED_RESPONSE_PHRASE
    assert "as an ai" in f.detail["hits"]


def test_check_chatml_reasoning_leakage_fails(good_chatml_example):
    bad = copy.deepcopy(good_chatml_example)
    bad["messages"][2]["reasoning_content"] = (
        'Customer wants a refund. So my output will be {"intent": "refund_request"}.'
    )
    f = check_chatml_reasoning_leakage(bad)
    assert f and f.kind == ErrorKind.CHATML_REASONING_LEAKAGE


def test_check_chatml_token_overflow_passes(good_chatml_example):
    assert check_chatml_token_overflow_estimate(good_chatml_example, max_tokens=10000) is None


def test_check_chatml_token_overflow_fails(good_chatml_example):
    f = check_chatml_token_overflow_estimate(good_chatml_example, max_tokens=10)
    assert f and f.kind == ErrorKind.CHATML_TOKEN_OVERFLOW


# ──────────────────────────────────────────────────────────────────────
# The v1 killer: trailing-comma in rendered <tool_call>
# ──────────────────────────────────────────────────────────────────────

def test_rendered_trailing_comma_passes_clean(good_rendered_chatml):
    assert check_rendered_trailing_comma(good_rendered_chatml) is None


def test_rendered_trailing_comma_fails_on_v1_bug():
    """This is the literal v1 production failure (see docs/known-issues.md §1)."""
    rendered = (
        "<|im_start|>assistant\n"
        '<tool_call>\n'
        '{"name": "rag_search", "arguments": {"collection": "faq", "query": "delivery delay"}},\n'
        '</tool_call>'
        "<|im_end|>"
    )
    f = check_rendered_trailing_comma(rendered)
    assert f and f.kind == ErrorKind.CHATML_TRAILING_COMMA_TOOL_CALL
    assert f.detail["count"] == 1


def test_rendered_tool_call_json_valid_passes(good_rendered_chatml):
    assert check_rendered_tool_call_json_valid(good_rendered_chatml) is None


def test_rendered_tool_call_json_valid_fails_on_garbage():
    rendered = "<tool_call>\n{not json at all}\n</tool_call>"
    f = check_rendered_tool_call_json_valid(rendered)
    assert f and f.kind == ErrorKind.CHATML_RENDER_FAILURE


def test_validate_chatml_with_rendered(good_chatml_example, good_rendered_chatml):
    f = validate_chatml(good_chatml_example, rendered=good_rendered_chatml, n_tokens=500)
    assert f is None


def test_validate_chatml_catches_trailing_comma_after_structure_passes(good_chatml_example):
    """Even with a structurally-clean source example, a trailing-comma render must be caught."""
    rendered_with_bug = (
        '<tool_call>\n{"name":"rag_search","arguments":{"collection":"faq","query":"x"}},\n</tool_call>'
    )
    f = validate_chatml(good_chatml_example, rendered=rendered_with_bug)
    assert f and f.kind == ErrorKind.CHATML_TRAILING_COMMA_TOOL_CALL


# ──────────────────────────────────────────────────────────────────────
# Aggregate / split
# ──────────────────────────────────────────────────────────────────────

def test_check_distribution_below_min():
    fails = check_distribution_below_min(
        {"refund_request": 30, "quality_complaint": 250},
        min_per_intent={"refund_request": 300, "quality_complaint": 250},
    )
    assert len(fails) == 1
    f = fails[0]
    assert f.kind == ErrorKind.DISTRIBUTION_BELOW_MIN
    assert f.detail["intent"] == "refund_request"
    assert f.detail["shortage"] == 270


def test_check_train_eval_leakage_clean():
    assert check_train_eval_leakage({"T1", "T2"}, {"T3", "T4"}) is None


def test_check_train_eval_leakage_dirty():
    f = check_train_eval_leakage({"T1", "T2"}, {"T2", "T3"})
    assert f and f.kind == ErrorKind.TRAIN_EVAL_LEAKAGE
    assert f.detail["overlap_count"] == 1


def test_check_eval_underrepresented():
    fails = check_eval_underrepresented({"refund_request": 5, "quality_complaint": 20}, min_per_intent=10)
    kinds = [f.detail["intent"] for f in fails]
    assert "refund_request" in kinds
    assert "quality_complaint" not in kinds
