"""Error-kind taxonomy — the spine of v2 fixability.

Every failure mode the pipeline can detect is a member of `ErrorKind`. Each
member has metadata describing which stage emits it, how severe it is, and
which fixer strategy resolves it. Validators return ErrorKind values;
fixers register against ErrorKind values; the dashboard groups by ErrorKind.

To add a new failure mode: add the member here, write the validator in
`validators.py`, register a fixer in `fixer.py`. That's the contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Stage(str, Enum):
    """Pipeline stages, ordered. Each stage's output is the next stage's input."""

    SOURCE = "source"          # raw ticket selection + language filter
    TRACE = "trace"            # teacher agent run
    FILTER = "filter"          # quality filter (template responses, missing follow-ups)
    CHATML = "chatml"          # ChatML construction + tokenizer round-trip
    BALANCE = "balance"        # per-intent oversampling / downsampling
    VALIDATE = "validate"      # final structural + distribution gates
    SPLIT = "split"            # stratified train/eval split

    @classmethod
    def ordered(cls) -> list["Stage"]:
        return [cls.SOURCE, cls.TRACE, cls.FILTER, cls.CHATML, cls.BALANCE, cls.VALIDATE, cls.SPLIT]


class Severity(str, Enum):
    """How a failure should be handled by default."""

    DROP = "drop"                  # unfixable; record permanently dropped
    QUARANTINE = "quarantine"      # fixable, fixer must regenerate via teacher
    FIXABLE = "fixable"            # fixable, fixer applies a deterministic transform
    AGGREGATE = "aggregate"        # cross-record (e.g. distribution); resolved by re-sampling


class ErrorKind(str, Enum):
    """Every failure mode the pipeline can emit. Order doesn't matter; values must be stable."""

    # ── Stage 0: source ──
    RAW_TOO_SHORT = "raw_too_short"
    RAW_NO_CONTENT = "raw_no_content"
    NON_TARGET_LANGUAGE = "non_target_language"
    RAW_DUPLICATE = "raw_duplicate"

    # ── Stage 1: trace generation ──
    TRACE_NO_TOOL_CALLS_ON_VALID = "trace_no_tool_calls_on_valid"
    TRACE_INVALID_COLLECTION_NAME = "trace_invalid_collection_name"
    TRACE_MALFORMED_FINAL_JSON = "trace_malformed_final_json"
    TRACE_TEMPLATE_RESPONSE = "trace_template_response"
    TRACE_MISSING_FOLLOWUP_PATTERN = "trace_missing_followup_pattern"
    TRACE_EMPTY_RESPONSE_ON_VALID = "trace_empty_response_on_valid"
    TRACE_INTENT_MISMATCH = "trace_intent_mismatch"
    TRACE_RAG_SERVER_ERROR = "trace_rag_server_error"

    # ── Stage 2: filter ──
    # (filter stage flags traces as having one of the trace_* errors above.
    # Stage 2 doesn't introduce new error kinds — it's a re-validation pass.)

    # ── Stage 3: chatml ──
    CHATML_TRAILING_COMMA_TOOL_CALL = "chatml_trailing_comma_tool_call"
    CHATML_RENDER_FAILURE = "chatml_render_failure"
    CHATML_TOKEN_OVERFLOW = "chatml_token_overflow"
    CHATML_MISSING_REQUIRED_FIELD = "chatml_missing_required_field"
    CHATML_BANNED_RESPONSE_PHRASE = "chatml_banned_response_phrase"
    CHATML_BAD_ROLE_ORDER = "chatml_bad_role_order"
    CHATML_TOOL_CALL_ID_MISMATCH = "chatml_tool_call_id_mismatch"
    CHATML_REASONING_LEAKAGE = "chatml_reasoning_leakage"  # JSON keys appear in reasoning_content

    # ── Stage 4-5: balance + validate (aggregate) ──
    DISTRIBUTION_BELOW_MIN = "distribution_below_min"
    DISTRIBUTION_OVER_MAX = "distribution_over_max"

    # ── Stage 6: split ──
    EVAL_UNDERREPRESENTED = "eval_underrepresented"
    TRAIN_EVAL_LEAKAGE = "train_eval_leakage"


@dataclass(frozen=True)
class ErrorKindMeta:
    """Metadata for an error kind. Drives fixer dispatch and dashboard rendering."""

    stage: Stage
    severity: Severity
    description: str
    suggested_fix: str

    @property
    def is_droppable(self) -> bool:
        return self.severity == Severity.DROP

    @property
    def needs_fixer(self) -> bool:
        return self.severity in (Severity.QUARANTINE, Severity.FIXABLE)


# ── Metadata table ──
# This dict IS the contract. New ErrorKind members MUST have an entry here;
# tests assert no orphans.
META: dict[ErrorKind, ErrorKindMeta] = {
    # source
    ErrorKind.RAW_TOO_SHORT: ErrorKindMeta(
        Stage.SOURCE, Severity.DROP,
        "Ticket text is shorter than 50 chars (likely truncation bug or empty body).",
        "Drop the ticket; it cannot be a useful training example.",
    ),
    ErrorKind.RAW_NO_CONTENT: ErrorKindMeta(
        Stage.SOURCE, Severity.DROP,
        "Ticket has no message bodies after stripping whitespace.",
        "Drop the ticket.",
    ),
    ErrorKind.NON_TARGET_LANGUAGE: ErrorKindMeta(
        Stage.SOURCE, Severity.DROP,
        "Detected language is not in the target set (English-only for v1).",
        "Drop the ticket. To include the language, add it to configs/v2_pipeline.yaml.",
    ),
    ErrorKind.RAW_DUPLICATE: ErrorKindMeta(
        Stage.SOURCE, Severity.DROP,
        "Ticket body duplicates another ticket already in the source pool.",
        "Drop the duplicate; keep the earliest occurrence.",
    ),

    # trace
    ErrorKind.TRACE_NO_TOOL_CALLS_ON_VALID: ErrorKindMeta(
        Stage.TRACE, Severity.QUARANTINE,
        "is_valid=true but the trace contains no rag_search calls. The model would learn to skip grounding.",
        "Re-run the teacher agent with prompt variant 'strict_tool_use' that forbids skipping rag_search.",
    ),
    ErrorKind.TRACE_INVALID_COLLECTION_NAME: ErrorKindMeta(
        Stage.TRACE, Severity.FIXABLE,
        "rag_search call uses a collection name not in the canonical set.",
        "Apply collection-name normalization map (configs/v2_pipeline.yaml::collection_aliases).",
    ),
    ErrorKind.TRACE_MALFORMED_FINAL_JSON: ErrorKindMeta(
        Stage.TRACE, Severity.QUARANTINE,
        "The teacher's final JSON output does not parse, or is missing required fields.",
        "Re-run the teacher agent. If it persists across 3 attempts, drop the ticket.",
    ),
    ErrorKind.TRACE_TEMPLATE_RESPONSE: ErrorKindMeta(
        Stage.TRACE, Severity.QUARANTINE,
        "Response is a generic 'I'll check shortly' template with no KB-specific content.",
        "Re-run with prompt variant 'policy_citation' that demands at least one specific fact.",
    ),
    ErrorKind.TRACE_MISSING_FOLLOWUP_PATTERN: ErrorKindMeta(
        Stage.TRACE, Severity.QUARANTINE,
        "Category-specific follow-up pattern missing (e.g. quality_complaint without 'photo').",
        "Re-run with category-specific prompt variant.",
    ),
    ErrorKind.TRACE_EMPTY_RESPONSE_ON_VALID: ErrorKindMeta(
        Stage.TRACE, Severity.QUARANTINE,
        "is_valid=true but response field is empty or under min_response_chars.",
        "Re-run the teacher agent.",
    ),
    ErrorKind.TRACE_INTENT_MISMATCH: ErrorKindMeta(
        Stage.TRACE, Severity.QUARANTINE,
        "Teacher classified the ticket into an intent different from the requested target intent shard.",
        "Either re-run with category-priming prompt, or move record to the correct intent shard.",
    ),
    ErrorKind.TRACE_RAG_SERVER_ERROR: ErrorKindMeta(
        Stage.TRACE, Severity.QUARANTINE,
        "RAG MCP server returned an error or timeout during trace generation.",
        "Verify the RAG server is up; re-run the teacher agent.",
    ),

    # chatml
    ErrorKind.CHATML_TRAILING_COMMA_TOOL_CALL: ErrorKindMeta(
        Stage.CHATML, Severity.FIXABLE,
        "Rendered <tool_call> body has a trailing comma after the closing brace; breaks vLLM hermes parser.",
        "Strip trailing comma in the source tool_calls before re-rendering.",
    ),
    ErrorKind.CHATML_RENDER_FAILURE: ErrorKindMeta(
        Stage.CHATML, Severity.QUARANTINE,
        "tokenizer.apply_chat_template raised an exception or produced a malformed output.",
        "Inspect the source messages; usually a missing role or malformed tool_call.",
    ),
    ErrorKind.CHATML_TOKEN_OVERFLOW: ErrorKindMeta(
        Stage.CHATML, Severity.FIXABLE,
        "Rendered example exceeds max_seq_length tokens.",
        "Truncate tool_results to top-k or shorten the user ticket body.",
    ),
    ErrorKind.CHATML_MISSING_REQUIRED_FIELD: ErrorKindMeta(
        Stage.CHATML, Severity.QUARANTINE,
        "Final JSON is missing one or more of the 11 required output fields.",
        "Re-run upstream trace; the teacher dropped a field.",
    ),
    ErrorKind.CHATML_BANNED_RESPONSE_PHRASE: ErrorKindMeta(
        Stage.CHATML, Severity.QUARANTINE,
        "Customer-facing response contains a banned phrase (AI self-references, French stock politeness, etc.).",
        "Re-run with the phrase added to the negative-examples prompt section.",
    ),
    ErrorKind.CHATML_BAD_ROLE_ORDER: ErrorKindMeta(
        Stage.CHATML, Severity.QUARANTINE,
        "Message role sequence violates ChatML invariants (tool without preceding tool_calls, etc.).",
        "Re-run upstream stage; this is a build_chatml.py logic bug.",
    ),
    ErrorKind.CHATML_TOOL_CALL_ID_MISMATCH: ErrorKindMeta(
        Stage.CHATML, Severity.FIXABLE,
        "tool_call_id on a tool result doesn't match any preceding assistant tool_call id.",
        "Regenerate ids deterministically from (message_idx, call_idx).",
    ),
    ErrorKind.CHATML_REASONING_LEAKAGE: ErrorKindMeta(
        Stage.CHATML, Severity.FIXABLE,
        "reasoning_content contains literal JSON-output field keys (e.g. \"intent\":); model may parrot.",
        "Strip JSON keys from reasoning_content before re-rendering.",
    ),

    # aggregate
    ErrorKind.DISTRIBUTION_BELOW_MIN: ErrorKindMeta(
        Stage.VALIDATE, Severity.AGGREGATE,
        "Per-intent count is below configured minimum.",
        "Run stage0/stage1 with --add N to expand the source pool for this intent.",
    ),
    ErrorKind.DISTRIBUTION_OVER_MAX: ErrorKindMeta(
        Stage.VALIDATE, Severity.AGGREGATE,
        "Per-intent count exceeds configured maximum (skews training).",
        "Downsample randomly to max, preserving rare sub-categories.",
    ),

    # split
    ErrorKind.EVAL_UNDERREPRESENTED: ErrorKindMeta(
        Stage.SPLIT, Severity.FIXABLE,
        "Eval split has fewer than min_eval_per_intent examples for some intent.",
        "Re-shuffle the train/eval split with stratification on intent.",
    ),
    ErrorKind.TRAIN_EVAL_LEAKAGE: ErrorKindMeta(
        Stage.SPLIT, Severity.FIXABLE,
        "A ticket_id appears in both train and eval splits.",
        "Regenerate the split using ticket_id-based stratified k-fold instead of row sampling.",
    ),
}


def meta(kind: ErrorKind) -> ErrorKindMeta:
    """Return the metadata for an error kind. Raises if missing (taxonomy invariant)."""
    if kind not in META:
        raise KeyError(f"ErrorKind {kind} has no metadata entry in error_kinds.META")
    return META[kind]


def kinds_for_stage(stage: Stage) -> list[ErrorKind]:
    """Return all error kinds emitted at this stage."""
    return [k for k, m in META.items() if m.stage == stage]


def fixable_kinds() -> list[ErrorKind]:
    """Error kinds a fixer can resolve (excludes DROP and AGGREGATE)."""
    return [k for k, m in META.items() if m.needs_fixer]
