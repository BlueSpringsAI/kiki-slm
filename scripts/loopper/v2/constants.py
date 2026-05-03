"""Pipeline-wide constants. Single source of truth for vocabulary that
appears in validators, stage scripts, and fixer prompts. Values mirror
configs/loopper_pipeline.yaml — keep in sync if the YAML changes."""
from __future__ import annotations

# ── Output schema vocabulary (the SLM's 11 fields) ──

VALID_INTENTS: frozenset[str] = frozenset({
    "new_order_inquiry", "design_update", "payment_confirmation",
    "delivery_issue", "refund_request", "order_cancellation",
    "quality_complaint", "sample_request", "price_negotiation",
    "customer_feedback", "other",
})

# Business intents (excludes "other", which is the rejection bucket).
BUSINESS_INTENTS: frozenset[str] = VALID_INTENTS - {"other"}

VALID_URGENCIES: frozenset[str] = frozenset({"low", "medium", "high", "critical"})
VALID_RESOLUTION_TYPES: frozenset[str] = frozenset({
    "direct_resolve", "requires_human_action", "needs_escalation", "needs_more_info",
})
VALID_TEAMS: frozenset[str] = frozenset({"design", "logistics", "finance", "account_manager", "none"})
VALID_REJECTION_TYPES: frozenset[str] = frozenset({
    "spam", "misdirected", "newsletter", "auto_reply", "unrelated",
})

REQUIRED_OUTPUT_FIELDS: frozenset[str] = frozenset({
    "intent", "urgency", "confidence", "is_valid", "rejection_type",
    "resolution_type", "team", "actions", "summary", "reasoning", "response",
})

REQUIRED_REASONING_SUBFIELDS: frozenset[str] = frozenset({
    "intent_basis", "urgency_basis", "resolution_basis", "policy_used",
})

# ── Tool / RAG vocabulary ──

CANONICAL_COLLECTIONS: frozenset[str] = frozenset({
    "faq", "operations", "communication_guidelines", "supplier_data",
})

# Map verbose teacher names → canonical short names.
# Used by trace_invalid_collection_name fixer.
COLLECTION_ALIASES: dict[str, str] = {
    "customer_policy_faq": "faq",
    "sales_operations_playbook": "operations",
    "supplier_intelligence": "supplier_data",
    # already-canonical names map to themselves
    "faq": "faq",
    "operations": "operations",
    "communication_guidelines": "communication_guidelines",
    "supplier_data": "supplier_data",
}

# Names accepted at training time (pre-normalization). Anything outside this set
# is unfixable and triggers TRACE_INVALID_COLLECTION_NAME quarantine.
KNOWN_COLLECTIONS: frozenset[str] = frozenset(COLLECTION_ALIASES.keys())


# ── Response-quality vocabulary ──

# Short phrases that, in isolation, indicate a non-substantive "I'll check shortly" response.
# Rule: a response is template iff it contains a TEMPLATE_MARKER and does NOT contain
# any KB_SPECIFIC_MARKER (numbers, named policy nouns, etc.). See validators.py.
TEMPLATE_RESPONSE_MARKERS: tuple[str, ...] = (
    "i'll check shortly",
    "i'll get back to you shortly",
    "checking with our team",
    "i will check with",
    "let me check with",
    "i'll need to check",
    "i'll look into it",
    "looking into this",
)

# If the response contains ANY of these, it is considered substantive enough.
# Numbers (days, weeks, prices), policy keywords, follow-up requests.
KB_SPECIFIC_MARKERS: tuple[str, ...] = (
    "business day", "working day", "business days", "working days",
    "moq", "minimum order", "lead time", "production time",
    "refund processing", "return policy", "warranty",
    "please send", "could you share", "could you provide",
    "photo", "image", "picture",
    "vector", ".ai", ".eps", ".pdf",
    "sku", "product code", "order number",
)

BANNED_RESPONSE_PHRASES: tuple[str, ...] = (
    # AI self-references
    "as an ai", "as a language model", "i'm an ai",
    "i don't have access to", "i cannot",
    # French stock politeness (was leaking into English responses)
    "n'hésitez pas", "à votre disposition", "en tant qu", "je suis", "je ne peux",
    # German leakage
    "als ki", "ich bin ein", "zur verfügung",
    # English filler
    "rest assured", "please be advised", "for your review", "at your service",
    "i apologize for any inconvenience",
)

# Per-intent expected follow-up patterns. Response should contain at least one
# of the listed substrings. Empty tuple = no follow-up requirement.
FOLLOWUP_PATTERNS_BY_INTENT: dict[str, tuple[str, ...]] = {
    "quality_complaint":   ("photo", "image", "picture"),
    "design_update":       ("vector", ".ai", ".eps", "file format", "high-res"),
    "new_order_inquiry":   ("product", "sku", "quantity", "moq"),
    "sample_request":      ("address", "shipping", "delivery"),
    "delivery_issue":      ("order number", "tracking", "delivery date"),
    "refund_request":      ("order number", "invoice"),
    "order_cancellation":  ("order number", "invoice"),
    "payment_confirmation": ("invoice", "payment reference", "transaction id"),
    "price_negotiation":   ("quantity", "moq", "annual volume"),
    # customer_feedback and "other" are intentionally empty — no follow-up expected.
    "customer_feedback":   (),
    "other":               (),
}


# ── Numeric thresholds ──

DEFAULTS = {
    # Stage 0 source
    "min_raw_chars": 50,
    "target_languages": ("en",),
    # Stage 1 trace
    "min_response_chars": 20,
    "min_reasoning_chars": 30,
    # Stage 3 chatml
    "max_seq_length": 8192,
    "token_chars_ratio": 3.5,  # rough estimator for pre-tokenizer gate
    # Stage 4-5 distribution (defaults; per-intent overrides via configs/v2_pipeline.yaml)
    "min_per_intent": 200,
    "max_per_intent": 1500,
    "min_eval_per_intent": 10,
    "rejection_pct_target": 0.10,
}

# Per-intent minimum counts (from docs/known-issues.md §5a fix table).
MIN_PER_INTENT_OVERRIDES: dict[str, int] = {
    "refund_request":       300,
    "customer_feedback":    200,
    "price_negotiation":    200,
    "sample_request":       200,
    "payment_confirmation": 250,
    "order_cancellation":   250,
    "quality_complaint":    250,
}


def min_count_for(intent: str) -> int:
    return MIN_PER_INTENT_OVERRIDES.get(intent, DEFAULTS["min_per_intent"])
