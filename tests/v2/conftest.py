"""Shared fixtures for v2 pipeline tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.loopper.v2.ledger import Ledger


@pytest.fixture
def tmp_ledger(tmp_path: Path) -> Ledger:
    """Fresh on-disk SQLite ledger; cleaned up automatically by tmp_path."""
    ledger = Ledger(tmp_path / "ledger.db")
    yield ledger
    ledger.close()


# ── Sample fixtures matching real v1 data shapes ──

@pytest.fixture
def good_raw_ticket() -> dict:
    return {
        "ticket_id": "T1",
        "subject": "Damaged mugs",
        "languages": ["en"],
        "messages": [
            {
                "message_index": 0,
                "clean_body": "We received 500 custom mugs in dark grey but ordered navy blue. "
                              "Trade fair on Thursday — we need this resolved urgently.",
                "direction": "incoming",
                "language": "en",
            }
        ],
    }


@pytest.fixture
def good_trace() -> dict:
    return {
        "ticket_id": "T1",
        "completion_status": "completed",
        "triage": {
            "is_valid": True,
            "category": "quality_complaint",
            "category_confidence": 0.92,
        },
        "retrieval": {
            "tool_calls": [
                {
                    "name": "rag_search",
                    "args": {"collection": "faq", "query": "quality complaint photo evidence"},
                    "id": "call_1",
                }
            ],
            "tool_results": [{"id": "call_1", "results": [{"text": "Quality complaints: please share photos within 5 business days."}]}],
        },
        "response": {
            "summary": ["Customer received wrong color mugs"],
            "resolution_type": "requires_human_action",
            "human_team_required": "account_manager",
            "action_list": ["Forward photos to quality team"],
            "response_english": (
                "Hello,\n\nThank you for letting us know. Could you share photos of the mugs so our "
                "quality team can review? Our standard turnaround is 1 working day. "
                "Best regards,\nMarc Logier, Account Manager — Loopper"
            ),
        },
    }


@pytest.fixture
def good_chatml_example() -> dict:
    final_json = {
        "intent": "quality_complaint",
        "urgency": "high",
        "confidence": 0.92,
        "is_valid": True,
        "rejection_type": None,
        "resolution_type": "requires_human_action",
        "team": "account_manager",
        "actions": ["Forward photos to quality team"],
        "summary": ["Wrong color mugs received"],
        "reasoning": {
            "intent_basis": "Wrong item received",
            "urgency_basis": "Trade fair Thursday",
            "resolution_basis": "Cannot confirm timeline alone",
            "policy_used": "Quality policy: photo + 1-day assessment",
        },
        "response": (
            "Hello,\n\nCould you share photos so our quality team can review? "
            "Our standard turnaround is 1 working day.\n\nMarc Logier"
        ),
    }
    return {
        "tools": [{"type": "function", "function": {"name": "rag_search", "parameters": {}}}],
        "messages": [
            {"role": "system", "content": "You are a Loopper agent."},
            {"role": "user", "content": "Ticket ID: T1\nWe got the wrong color mugs."},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "Customer received wrong items. Need to search quality policy.",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "rag_search",
                        "arguments": json.dumps({"collection": "faq", "query": "quality wrong item"}),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "rag_search",
                "content": json.dumps({"results": [{"text": "Photos within 5 business days."}]}),
            },
            {
                "role": "assistant",
                "reasoning_content": "Found the policy. Drafting response now.",
                "content": json.dumps(final_json, ensure_ascii=False),
            },
        ],
    }


@pytest.fixture
def good_rendered_chatml() -> str:
    """A clean rendered example (no trailing comma) for round-trip checks."""
    return (
        "<|im_start|>system\nYou are a Loopper agent.<|im_end|>\n"
        "<|im_start|>user\nTicket ID: T1\nWe got the wrong color mugs.<|im_end|>\n"
        "<|im_start|>assistant\n<think>Need to search policy.</think>\n"
        '<tool_call>\n{"name":"rag_search","arguments":{"collection":"faq","query":"quality"}}\n</tool_call>'
        "<|im_end|>\n"
        "<|im_start|>tool\n"
        '<tool_response>\n{"results":[{"text":"Photos within 5 days."}]}\n</tool_response>'
        "<|im_end|>\n"
        '<|im_start|>assistant\n<think>Drafting response.</think>\n{"intent":"quality_complaint"}<|im_end|>\n'
    )
