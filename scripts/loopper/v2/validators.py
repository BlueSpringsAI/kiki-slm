"""Pure validators — one per ErrorKind.

Each validator returns either `None` (the input passed) or a `ValidationFailure`
with the error_kind, a human-readable message, and a structured `detail` dict
that the fixer reads to decide what to do.

Validators are pure: they don't touch the ledger, don't read files, don't make
network calls. Stage scripts wire them up: they read the input, call
`validate_*`, and then call `ledger.mark_quarantined(...)` with the failure.

Aggregators
-----------
Per-stage functions like `validate_raw_ticket()` run the relevant individual
checks and return the FIRST failure (or None if all passed). Most stages
quarantine on the first failure — a record can have only one error_kind at a
time. If you want all failures, use `collect_failures_*` variants.

Tokenizer round-trip
--------------------
The two ChatML errors that v1 silently ignored (trailing-comma tool_call,
malformed final JSON at inference) can ONLY be detected by rendering through
the actual tokenizer's chat template. `validate_chatml_rendered()` takes a
pre-rendered string (so tests can run without loading the 4B model) plus the
source example. Stage 3 wires up the real tokenizer.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from scripts.loopper.v2.constants import (
    BANNED_RESPONSE_PHRASES,
    DEFAULTS,
    FOLLOWUP_PATTERNS_BY_INTENT,
    KB_SPECIFIC_MARKERS,
    KNOWN_COLLECTIONS,
    REQUIRED_OUTPUT_FIELDS,
    REQUIRED_REASONING_SUBFIELDS,
    TEMPLATE_RESPONSE_MARKERS,
    VALID_INTENTS,
    VALID_REJECTION_TYPES,
    VALID_RESOLUTION_TYPES,
    VALID_TEAMS,
    VALID_URGENCIES,
)
from scripts.loopper.v2.error_kinds import ErrorKind


@dataclass
class ValidationFailure:
    """A failed validation. Stage scripts pass this to ledger.mark_quarantined()."""

    kind: ErrorKind
    msg: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Serializable form for quarantine files."""
        return {"error_kind": self.kind.value, "error_msg": self.msg, "error_detail": self.detail}


# ──────────────────────────────────────────────────────────────────────
# Stage 0: source / raw ticket validators
# ──────────────────────────────────────────────────────────────────────

def check_raw_too_short(
    ticket: dict[str, Any], *, min_chars: int = DEFAULTS["min_raw_chars"]
) -> ValidationFailure | None:
    total = sum(len((m.get("clean_body") or "")) for m in ticket.get("messages") or [])
    if total < min_chars:
        return ValidationFailure(
            ErrorKind.RAW_TOO_SHORT,
            f"ticket text is {total} chars, below min={min_chars}",
            {"total_chars": total, "min_chars": min_chars},
        )
    return None


def check_raw_no_content(ticket: dict[str, Any]) -> ValidationFailure | None:
    msgs = ticket.get("messages") or []
    if not msgs:
        return ValidationFailure(
            ErrorKind.RAW_NO_CONTENT, "ticket has no messages array", {}
        )
    if all(not (m.get("clean_body") or "").strip() for m in msgs):
        return ValidationFailure(
            ErrorKind.RAW_NO_CONTENT, "all message bodies are empty after strip", {"n_messages": len(msgs)}
        )
    return None


def check_non_target_language(
    ticket: dict[str, Any], *, target_languages: tuple[str, ...] = DEFAULTS["target_languages"]
) -> ValidationFailure | None:
    """Approximate language gate. Stage script can pass detected lang explicitly."""
    detected = (ticket.get("detected_language")
                or (ticket.get("languages") or [None])[0]
                or "")
    detected = detected.lower() if isinstance(detected, str) else ""
    # Empty / None / "unknown" => let it pass; a stricter detector is run elsewhere.
    if not detected or detected in {"unknown", "none"}:
        return None
    if detected not in target_languages:
        return ValidationFailure(
            ErrorKind.NON_TARGET_LANGUAGE,
            f"detected language '{detected}' not in target {list(target_languages)}",
            {"detected": detected, "target": list(target_languages)},
        )
    return None


def check_raw_duplicate(
    ticket: dict[str, Any], *, seen_hashes: set[str]
) -> ValidationFailure | None:
    """Caller maintains the seen_hashes set; we just check membership.
    Returns failure if the ticket's body hash is already in seen_hashes;
    SIDE EFFECT: adds the new hash. (This is the one impure validator —
    duplicate detection inherently needs cross-record state.)"""
    import hashlib
    body = "\n".join((m.get("clean_body") or "").strip() for m in ticket.get("messages") or [])
    h = hashlib.sha256(body.encode("utf-8")).hexdigest()
    if h in seen_hashes:
        return ValidationFailure(
            ErrorKind.RAW_DUPLICATE,
            f"ticket body matches a prior ticket (hash={h[:12]})",
            {"hash": h, "ticket_id": ticket.get("ticket_id")},
        )
    seen_hashes.add(h)
    return None


def validate_raw_ticket(
    ticket: dict[str, Any],
    *,
    target_languages: tuple[str, ...] = DEFAULTS["target_languages"],
    min_chars: int = DEFAULTS["min_raw_chars"],
    seen_hashes: set[str] | None = None,
) -> ValidationFailure | None:
    """Run all raw-ticket checks; return first failure or None.

    Order: cheapest checks first; semantic before duplicate (so duplicates of
    bad tickets are reported as bad-tickets, not duplicates)."""
    for fn in (
        lambda t: check_raw_no_content(t),
        lambda t: check_raw_too_short(t, min_chars=min_chars),
        lambda t: check_non_target_language(t, target_languages=target_languages),
    ):
        if (f := fn(ticket)) is not None:
            return f
    if seen_hashes is not None:
        if (f := check_raw_duplicate(ticket, seen_hashes=seen_hashes)) is not None:
            return f
    return None


# ──────────────────────────────────────────────────────────────────────
# Stage 1: trace validators
# ──────────────────────────────────────────────────────────────────────

def _trace_tool_calls(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull all tool calls from a teacher trace, regardless of which key the agent used."""
    retrieval = trace.get("retrieval") or {}
    return retrieval.get("tool_calls") or []


def _trace_response_text(trace: dict[str, Any]) -> str:
    return ((trace.get("response") or {}).get("response_english") or "").strip()


def _trace_is_valid(trace: dict[str, Any]) -> bool:
    return bool((trace.get("triage") or {}).get("is_valid", False))


def _trace_intent(trace: dict[str, Any]) -> str:
    return ((trace.get("triage") or {}).get("category") or "").strip()


def check_trace_no_tool_calls_on_valid(trace: dict[str, Any]) -> ValidationFailure | None:
    if not _trace_is_valid(trace):
        return None
    if not _trace_tool_calls(trace):
        return ValidationFailure(
            ErrorKind.TRACE_NO_TOOL_CALLS_ON_VALID,
            "is_valid=true but trace has 0 tool calls — model would learn skip pattern",
            {"is_valid": True, "n_tool_calls": 0},
        )
    return None


def check_trace_invalid_collection_name(trace: dict[str, Any]) -> ValidationFailure | None:
    bad: list[str] = []
    for tc in _trace_tool_calls(trace):
        # tool_calls in v1 traces look like {name, args:{collection, query}, ...}
        # or {function:{name, arguments}}. Handle both.
        coll = ""
        if isinstance(tc.get("args"), dict):
            coll = tc["args"].get("collection") or ""
        elif isinstance(tc.get("function"), dict):
            try:
                coll = (json.loads(tc["function"].get("arguments") or "{}") or {}).get("collection", "")
            except json.JSONDecodeError:
                pass
        if coll and coll not in KNOWN_COLLECTIONS:
            bad.append(coll)
    if bad:
        return ValidationFailure(
            ErrorKind.TRACE_INVALID_COLLECTION_NAME,
            f"trace uses unknown collection name(s): {bad}",
            {"unknown_collections": bad},
        )
    return None


def check_trace_malformed_final_json(trace: dict[str, Any]) -> ValidationFailure | None:
    """The trace's `response` block must contain enough fields to assemble the SLM output.
    We don't require all 11 here (that's a chatml-stage gate); we require the fields the
    teacher must produce: resolution_type, human_team_required, response_english (when valid)."""
    response = trace.get("response") or {}
    if not isinstance(response, dict):
        return ValidationFailure(
            ErrorKind.TRACE_MALFORMED_FINAL_JSON,
            f"trace.response is {type(response).__name__}, expected dict",
            {"observed_type": type(response).__name__},
        )
    needed = {"resolution_type", "human_team_required"}
    if _trace_is_valid(trace):
        needed.add("response_english")
    missing = [k for k in needed if not response.get(k)]
    if missing:
        return ValidationFailure(
            ErrorKind.TRACE_MALFORMED_FINAL_JSON,
            f"trace.response missing required fields: {missing}",
            {"missing": missing},
        )
    return None


def check_trace_template_response(trace: dict[str, Any]) -> ValidationFailure | None:
    """Response is a generic 'I'll check shortly' template with no KB-specific content."""
    if not _trace_is_valid(trace):
        return None
    text = _trace_response_text(trace).lower()
    if not text:
        return None  # caught by check_trace_empty_response_on_valid
    has_template = any(m in text for m in TEMPLATE_RESPONSE_MARKERS)
    has_specific = any(m in text for m in KB_SPECIFIC_MARKERS)
    if has_template and not has_specific:
        # which markers fired (helps the fixer pick a prompt variant)
        fired = [m for m in TEMPLATE_RESPONSE_MARKERS if m in text]
        return ValidationFailure(
            ErrorKind.TRACE_TEMPLATE_RESPONSE,
            f"response is a generic template — contains {fired} but no KB-specific content",
            {"markers": fired, "response_preview": text[:200]},
        )
    return None


def check_trace_missing_followup_pattern(
    trace: dict[str, Any], *, target_intent: str | None = None
) -> ValidationFailure | None:
    if not _trace_is_valid(trace):
        return None
    intent = target_intent or _trace_intent(trace)
    patterns = FOLLOWUP_PATTERNS_BY_INTENT.get(intent)
    if not patterns:
        return None
    text = _trace_response_text(trace).lower()
    if not text:
        return None
    if not any(p in text for p in patterns):
        return ValidationFailure(
            ErrorKind.TRACE_MISSING_FOLLOWUP_PATTERN,
            f"response for intent={intent} missing follow-up patterns {list(patterns)}",
            {"intent": intent, "expected_any_of": list(patterns)},
        )
    return None


def check_trace_empty_response_on_valid(
    trace: dict[str, Any], *, min_chars: int = DEFAULTS["min_response_chars"]
) -> ValidationFailure | None:
    if not _trace_is_valid(trace):
        return None
    text = _trace_response_text(trace)
    if len(text) < min_chars:
        return ValidationFailure(
            ErrorKind.TRACE_EMPTY_RESPONSE_ON_VALID,
            f"is_valid=true but response is {len(text)} chars (< {min_chars})",
            {"length": len(text)},
        )
    return None


def check_trace_intent_mismatch(
    trace: dict[str, Any], *, target_intent: str
) -> ValidationFailure | None:
    """Records claimed for shard X must produce intent X. Otherwise either re-shard or re-classify."""
    detected = _trace_intent(trace)
    if not detected or detected == target_intent:
        return None
    if detected not in VALID_INTENTS:
        return ValidationFailure(
            ErrorKind.TRACE_INTENT_MISMATCH,
            f"trace classified as '{detected}' (not in valid set)",
            {"detected": detected, "expected": target_intent},
        )
    return ValidationFailure(
        ErrorKind.TRACE_INTENT_MISMATCH,
        f"trace classified as '{detected}', expected '{target_intent}' for this shard",
        {"detected": detected, "expected": target_intent},
    )


def validate_trace(
    trace: dict[str, Any],
    *,
    target_intent: str | None = None,
) -> ValidationFailure | None:
    """All trace checks in execution order; first failure wins."""
    checks = [
        check_trace_malformed_final_json,
        check_trace_no_tool_calls_on_valid,
        check_trace_empty_response_on_valid,
        check_trace_invalid_collection_name,
        check_trace_template_response,
    ]
    for fn in checks:
        if (f := fn(trace)) is not None:
            return f
    if target_intent:
        if (f := check_trace_intent_mismatch(trace, target_intent=target_intent)) is not None:
            return f
        if (f := check_trace_missing_followup_pattern(trace, target_intent=target_intent)) is not None:
            return f
    return None


# ──────────────────────────────────────────────────────────────────────
# Stage 3: chatml validators
# ──────────────────────────────────────────────────────────────────────

def _chatml_messages(example: dict[str, Any]) -> list[dict[str, Any]]:
    return example.get("messages") or []


def _chatml_final_assistant(example: dict[str, Any]) -> dict[str, Any] | None:
    for m in reversed(_chatml_messages(example)):
        if m.get("role") == "assistant" and m.get("content"):
            return m
    return None


def _extract_final_json(content: str) -> dict | None:
    """Extract the final-output JSON object from an assistant content string.
    Mirrors v1 logic — searches after </think> if present, otherwise from the
    first '{', and tries progressively larger substrings."""
    think_end = content.rfind("</think>")
    start = think_end + len("</think>") if think_end >= 0 else 0
    json_start = content.find("{", start)
    if json_start < 0:
        return None
    for end in range(json_start + 2, len(content) + 1):
        candidate = content[json_start:end]
        if candidate.count("{") == candidate.count("}"):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return None


def check_chatml_bad_role_order(example: dict[str, Any]) -> ValidationFailure | None:
    msgs = _chatml_messages(example)
    if not msgs:
        return ValidationFailure(ErrorKind.CHATML_BAD_ROLE_ORDER, "empty messages", {})
    roles = [m.get("role") for m in msgs]
    if roles[0] != "system":
        return ValidationFailure(
            ErrorKind.CHATML_BAD_ROLE_ORDER,
            f"first message must be 'system', got '{roles[0]}'",
            {"roles": roles},
        )
    if roles[-1] != "assistant":
        return ValidationFailure(
            ErrorKind.CHATML_BAD_ROLE_ORDER,
            f"last message must be 'assistant', got '{roles[-1]}'",
            {"roles": roles},
        )
    for i in range(1, len(msgs)):
        if roles[i] == "system":
            return ValidationFailure(
                ErrorKind.CHATML_BAD_ROLE_ORDER,
                f"system role at position {i} (only allowed at 0)",
                {"roles": roles, "bad_index": i},
            )
        if roles[i] == "tool":
            prev = msgs[i - 1] if i > 0 else None
            if not prev or prev.get("role") != "assistant" or not prev.get("tool_calls"):
                return ValidationFailure(
                    ErrorKind.CHATML_BAD_ROLE_ORDER,
                    f"tool message at {i} not preceded by assistant with tool_calls",
                    {"roles": roles, "bad_index": i},
                )
    return None


def check_chatml_tool_call_id_mismatch(example: dict[str, Any]) -> ValidationFailure | None:
    """Every tool message's tool_call_id must reference an id from the preceding
    assistant.tool_calls[].id."""
    msgs = _chatml_messages(example)
    pending_ids: set[str] = set()
    for m in msgs:
        role = m.get("role")
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                tcid = tc.get("id")
                if tcid:
                    pending_ids.add(tcid)
        elif role == "tool":
            tcid = m.get("tool_call_id")
            if not tcid:
                return ValidationFailure(
                    ErrorKind.CHATML_TOOL_CALL_ID_MISMATCH,
                    "tool message missing tool_call_id",
                    {"message": m},
                )
            if tcid not in pending_ids:
                return ValidationFailure(
                    ErrorKind.CHATML_TOOL_CALL_ID_MISMATCH,
                    f"tool_call_id '{tcid}' has no matching assistant tool_call",
                    {"tool_call_id": tcid, "known_ids": sorted(pending_ids)},
                )
    return None


def check_chatml_missing_required_field(example: dict[str, Any]) -> ValidationFailure | None:
    final = _chatml_final_assistant(example)
    if final is None:
        return ValidationFailure(
            ErrorKind.CHATML_MISSING_REQUIRED_FIELD,
            "no final assistant message with content",
            {},
        )
    obj = _extract_final_json(final.get("content") or "")
    if obj is None:
        return ValidationFailure(
            ErrorKind.CHATML_MISSING_REQUIRED_FIELD,
            "final assistant content is not valid JSON",
            {"content_preview": (final.get("content") or "")[:200]},
        )
    missing = sorted(REQUIRED_OUTPUT_FIELDS - set(obj.keys()))
    if missing:
        return ValidationFailure(
            ErrorKind.CHATML_MISSING_REQUIRED_FIELD,
            f"final JSON missing fields: {missing}",
            {"missing": missing, "present": sorted(obj.keys())},
        )
    # Field-value validation
    if obj.get("intent") not in VALID_INTENTS:
        return ValidationFailure(
            ErrorKind.CHATML_MISSING_REQUIRED_FIELD,
            f"intent '{obj.get('intent')}' not in valid set",
            {"intent": obj.get("intent")},
        )
    if obj.get("urgency") not in VALID_URGENCIES:
        return ValidationFailure(
            ErrorKind.CHATML_MISSING_REQUIRED_FIELD,
            f"urgency '{obj.get('urgency')}' not in valid set",
            {"urgency": obj.get("urgency")},
        )
    if obj.get("resolution_type") not in VALID_RESOLUTION_TYPES:
        return ValidationFailure(
            ErrorKind.CHATML_MISSING_REQUIRED_FIELD,
            f"resolution_type '{obj.get('resolution_type')}' not in valid set",
            {"resolution_type": obj.get("resolution_type")},
        )
    team = obj.get("team")
    if team and team not in VALID_TEAMS:
        return ValidationFailure(
            ErrorKind.CHATML_MISSING_REQUIRED_FIELD,
            f"team '{team}' not in valid set",
            {"team": team},
        )
    if obj.get("is_valid") is False:
        rt = obj.get("rejection_type")
        if rt not in VALID_REJECTION_TYPES:
            return ValidationFailure(
                ErrorKind.CHATML_MISSING_REQUIRED_FIELD,
                f"is_valid=false but rejection_type '{rt}' invalid",
                {"rejection_type": rt},
            )
    reasoning = obj.get("reasoning")
    if not isinstance(reasoning, dict):
        return ValidationFailure(
            ErrorKind.CHATML_MISSING_REQUIRED_FIELD,
            f"reasoning must be dict, got {type(reasoning).__name__}",
            {"observed_type": type(reasoning).__name__},
        )
    sub_missing = sorted(REQUIRED_REASONING_SUBFIELDS - set(reasoning.keys()))
    if sub_missing:
        return ValidationFailure(
            ErrorKind.CHATML_MISSING_REQUIRED_FIELD,
            f"reasoning missing subfields: {sub_missing}",
            {"missing_subfields": sub_missing},
        )
    return None


def check_chatml_banned_response_phrase(example: dict[str, Any]) -> ValidationFailure | None:
    final = _chatml_final_assistant(example)
    if final is None:
        return None
    obj = _extract_final_json(final.get("content") or "") or {}
    response = (obj.get("response") or "").lower()
    if not response:
        return None
    hits = [p for p in BANNED_RESPONSE_PHRASES if p in response]
    if hits:
        return ValidationFailure(
            ErrorKind.CHATML_BANNED_RESPONSE_PHRASE,
            f"response contains banned phrase(s): {hits}",
            {"hits": hits, "response_preview": response[:200]},
        )
    return None


def check_chatml_reasoning_leakage(example: dict[str, Any]) -> ValidationFailure | None:
    """reasoning_content should reason ABOUT the answer, not parrot the JSON keys."""
    suspicious_keys = ('"intent":', '"response":', '"rejection_type":', '"resolution_type":')
    for m in _chatml_messages(example):
        if m.get("role") != "assistant":
            continue
        rc = m.get("reasoning_content") or ""
        hits = [k for k in suspicious_keys if k in rc]
        if hits:
            return ValidationFailure(
                ErrorKind.CHATML_REASONING_LEAKAGE,
                f"reasoning_content contains JSON output keys: {hits}",
                {"hits": hits, "preview": rc[:200]},
            )
    return None


def check_chatml_token_overflow_estimate(
    example: dict[str, Any],
    *,
    max_tokens: int = DEFAULTS["max_seq_length"],
    chars_per_token: float = DEFAULTS["token_chars_ratio"],
) -> ValidationFailure | None:
    """Cheap pre-tokenizer estimate. The accurate check is in
    validate_chatml_rendered (which uses the actual tokenizer)."""
    total_chars = 0
    for m in _chatml_messages(example):
        total_chars += len(m.get("content") or "") + len(m.get("reasoning_content") or "")
        for tc in m.get("tool_calls") or []:
            total_chars += len(json.dumps(tc, ensure_ascii=False))
    est = int(total_chars / chars_per_token)
    if est > max_tokens:
        return ValidationFailure(
            ErrorKind.CHATML_TOKEN_OVERFLOW,
            f"estimated {est} tokens exceeds max_seq_length={max_tokens}",
            {"estimated_tokens": est, "max": max_tokens, "char_count": total_chars},
        )
    return None


# ── tokenizer round-trip checks (the v1 trailing-comma killer) ──

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def check_rendered_trailing_comma(rendered: str) -> ValidationFailure | None:
    """Scan rendered ChatML for the v1 trailing-comma-after-}-tool_call bug.

    Symptom: '<tool_call>\n{"name":"rag_search",...},\n</tool_call>' — that ',' breaks
    vLLM's hermes parser at inference. We catch it BEFORE training so the model never
    learns the artifact.
    """
    bad: list[str] = []
    for body_match in _TOOL_CALL_RE.finditer(rendered):
        body = body_match.group(1).strip()
        clean = body.rstrip(",").strip()
        if body != clean:
            bad.append(body[:120])
    if bad:
        return ValidationFailure(
            ErrorKind.CHATML_TRAILING_COMMA_TOOL_CALL,
            f"{len(bad)} tool_call(s) have trailing comma after closing brace",
            {"samples": bad[:3], "count": len(bad)},
        )
    return None


def check_rendered_tool_call_json_valid(rendered: str) -> ValidationFailure | None:
    """Stronger version: every <tool_call> body must parse as JSON after comma-strip."""
    bad: list[str] = []
    for body_match in _TOOL_CALL_RE.finditer(rendered):
        body = body_match.group(1).strip().rstrip(",").strip()
        try:
            json.loads(body)
        except json.JSONDecodeError as e:
            bad.append(f"{e}: {body[:120]}")
    if bad:
        return ValidationFailure(
            ErrorKind.CHATML_RENDER_FAILURE,
            f"{len(bad)} tool_call body/bodies failed JSON parse",
            {"errors": bad[:3], "count": len(bad)},
        )
    return None


def check_rendered_token_count(
    rendered: str,
    n_tokens: int,
    *,
    max_tokens: int = DEFAULTS["max_seq_length"],
) -> ValidationFailure | None:
    if n_tokens > max_tokens:
        return ValidationFailure(
            ErrorKind.CHATML_TOKEN_OVERFLOW,
            f"rendered example tokenized to {n_tokens} (> max {max_tokens})",
            {"tokens": n_tokens, "max": max_tokens, "rendered_chars": len(rendered)},
        )
    return None


def validate_chatml(
    example: dict[str, Any],
    *,
    rendered: str | None = None,
    n_tokens: int | None = None,
    max_tokens: int = DEFAULTS["max_seq_length"],
) -> ValidationFailure | None:
    """All ChatML checks; first failure wins. Pass `rendered` (and optionally n_tokens)
    to enable the tokenizer-level checks. If both are None, only structural checks run."""
    structural = (
        check_chatml_bad_role_order,
        check_chatml_tool_call_id_mismatch,
        check_chatml_missing_required_field,
        check_chatml_banned_response_phrase,
        check_chatml_reasoning_leakage,
    )
    for fn in structural:
        if (f := fn(example)) is not None:
            return f
    if (f := check_chatml_token_overflow_estimate(example, max_tokens=max_tokens)) is not None:
        return f
    if rendered is not None:
        if (f := check_rendered_trailing_comma(rendered)) is not None:
            return f
        if (f := check_rendered_tool_call_json_valid(rendered)) is not None:
            return f
        if n_tokens is not None:
            if (f := check_rendered_token_count(rendered, n_tokens, max_tokens=max_tokens)) is not None:
                return f
    return None


# ──────────────────────────────────────────────────────────────────────
# Stage 5: distribution validators (aggregate)
# ──────────────────────────────────────────────────────────────────────

def check_distribution_below_min(
    counts: dict[str, int],
    *,
    min_per_intent: dict[str, int],
) -> list[ValidationFailure]:
    """Returns one failure per intent below its minimum. Aggregate failures, list-shaped."""
    fails: list[ValidationFailure] = []
    for intent, target in min_per_intent.items():
        actual = counts.get(intent, 0)
        if actual < target:
            fails.append(ValidationFailure(
                ErrorKind.DISTRIBUTION_BELOW_MIN,
                f"intent '{intent}' has {actual} examples, below min {target}",
                {"intent": intent, "actual": actual, "min": target, "shortage": target - actual},
            ))
    return fails


def check_distribution_over_max(
    counts: dict[str, int],
    *,
    max_per_intent: dict[str, int],
) -> list[ValidationFailure]:
    fails: list[ValidationFailure] = []
    for intent, cap in max_per_intent.items():
        actual = counts.get(intent, 0)
        if actual > cap:
            fails.append(ValidationFailure(
                ErrorKind.DISTRIBUTION_OVER_MAX,
                f"intent '{intent}' has {actual} examples, above max {cap}",
                {"intent": intent, "actual": actual, "max": cap, "excess": actual - cap},
            ))
    return fails


# ──────────────────────────────────────────────────────────────────────
# Stage 6: split validators
# ──────────────────────────────────────────────────────────────────────

def check_train_eval_leakage(
    train_ticket_ids: set[str], eval_ticket_ids: set[str]
) -> ValidationFailure | None:
    overlap = train_ticket_ids & eval_ticket_ids
    if overlap:
        return ValidationFailure(
            ErrorKind.TRAIN_EVAL_LEAKAGE,
            f"{len(overlap)} ticket_id(s) appear in both train and eval",
            {"overlap_count": len(overlap), "samples": sorted(list(overlap))[:10]},
        )
    return None


def check_eval_underrepresented(
    eval_counts_by_intent: dict[str, int],
    *,
    min_per_intent: int = DEFAULTS["min_eval_per_intent"],
) -> list[ValidationFailure]:
    fails: list[ValidationFailure] = []
    for intent in VALID_INTENTS:
        actual = eval_counts_by_intent.get(intent, 0)
        if actual < min_per_intent:
            fails.append(ValidationFailure(
                ErrorKind.EVAL_UNDERREPRESENTED,
                f"eval has {actual} '{intent}' examples (< {min_per_intent})",
                {"intent": intent, "actual": actual, "min": min_per_intent},
            ))
    return fails
