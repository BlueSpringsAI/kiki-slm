"""Enterprise policy compliance reward function for GRPO.

Task 3.17: Rule-based reward checking refund limits, PII exposure,
escalation triggers, fabrication detection, and scope boundaries.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PII detection patterns
# ---------------------------------------------------------------------------

_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_PHONE_PATTERN = re.compile(r"\b(?:\+1[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b")

_PII_PATTERNS = {
    "ssn": _SSN_PATTERN,
    "credit_card": _CREDIT_CARD_PATTERN,
    "email": _EMAIL_PATTERN,
    "phone": _PHONE_PATTERN,
}

# ---------------------------------------------------------------------------
# Scope boundary keywords
# ---------------------------------------------------------------------------

_LEGAL_ADVICE_KEYWORDS = (
    "legal advice", "lawyer", "attorney", "sue", "lawsuit",
    "liable", "liability", "legally obligated", "court",
)
_MEDICAL_ADVICE_KEYWORDS = (
    "medical advice", "diagnosis", "prescribe", "medication",
    "doctor", "treatment plan", "medical opinion",
)
_FINANCIAL_ADVICE_KEYWORDS = (
    "investment advice", "stock", "portfolio", "financial planning",
    "tax advice", "securities", "trading recommendation",
)

# ---------------------------------------------------------------------------
# Refund thresholds
# ---------------------------------------------------------------------------

AUTO_APPROVE_LIMIT = 20.0
FLAG_THRESHOLD = 500.0


class PolicyComplianceReward:
    """GRPO reward function for enterprise policy compliance.

    Scoring:
        +0.5  — fully compliant (no violations)
        -0.1  — minor violation (e.g. slightly verbose, borderline scope)
        -0.3  — moderate violation (e.g. refund over flag threshold)
        -1.0  — critical violation (PII leak, fraud not escalated)
    """

    def __call__(self, completions: list[str], **kwargs) -> list[float]:
        """Score a batch of completions."""
        return [self._score_single(c, **kwargs) for c in completions]

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    def _score_single(self, completion: str, **kwargs) -> float:
        score = 0.5  # Start fully compliant

        # 1. PII exposure check (critical)
        pii_penalty = self._check_pii(completion)
        score += pii_penalty

        # 2. Refund limit check
        refund_penalty = self._check_refund_limits(completion)
        score += refund_penalty

        # 3. Escalation rules
        escalation_penalty = self._check_escalation(completion)
        score += escalation_penalty

        # 4. Scope boundaries
        scope_penalty = self._check_scope(completion)
        score += scope_penalty

        # 5. Fabrication check
        fabrication_penalty = self._check_fabrication(completion)
        score += fabrication_penalty

        return max(-1.0, min(0.5, score))

    def _check_pii(self, text: str) -> float:
        """Detect PII patterns in the response. Returns negative penalty."""
        penalty = 0.0
        for pii_type, pattern in _PII_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                logger.debug("PII detected (%s): %d occurrences", pii_type, len(matches))
                penalty -= 0.5  # Critical violation per PII type
        return penalty

    def _check_refund_limits(self, text: str) -> float:
        """Check if refund amounts exceed authorized limits."""
        # Look for dollar amounts in the text
        amounts = re.findall(r"\$\s*([\d,]+(?:\.\d{2})?)", text)
        penalty = 0.0
        for amount_str in amounts:
            try:
                amount = float(amount_str.replace(",", ""))
                if amount > FLAG_THRESHOLD:
                    penalty -= 0.3  # Moderate: over flag threshold
                    logger.debug("Refund amount $%.2f exceeds flag threshold", amount)
            except ValueError:
                continue
        return penalty

    def _check_escalation(self, text: str) -> float:
        """Check that escalation is triggered when required."""
        text_lower = text.lower()
        needs_escalation = any(
            kw in text_lower
            for kw in ("fraud", "unauthorized", "stolen", "identity theft", "suspicious activity")
        )

        has_escalation = any(
            kw in text_lower
            for kw in ("escalat", "supervisor", "specialist", "security team", "fraud team")
        )

        if needs_escalation and not has_escalation:
            logger.debug("Escalation required but not triggered")
            return -0.5  # Critical: fraud not escalated
        return 0.0

    def _check_scope(self, text: str) -> float:
        """Ensure response doesn't provide out-of-scope advice."""
        text_lower = text.lower()
        penalty = 0.0

        for keywords, label in [
            (_LEGAL_ADVICE_KEYWORDS, "legal"),
            (_MEDICAL_ADVICE_KEYWORDS, "medical"),
            (_FINANCIAL_ADVICE_KEYWORDS, "financial"),
        ]:
            # Only penalize if the model is *providing* advice, not just mentioning it
            providing_markers = ("i recommend", "you should", "i suggest", "i advise")
            if any(kw in text_lower for kw in keywords):
                if any(m in text_lower for m in providing_markers):
                    logger.debug("Scope violation: providing %s advice", label)
                    penalty -= 0.3
        return penalty

    def _check_fabrication(self, text: str) -> float:
        """Check for fabricated information (claims without tool call references)."""
        text_lower = text.lower()
        # If the response makes specific claims about order status, tracking, etc.
        # but doesn't reference a tool call result, penalize
        specific_claim_patterns = [
            r"your order (?:is|has been|was) (?:shipped|delivered|processing)",
            r"tracking number (?:is|:)\s*\w+",
            r"refund of \$[\d.,]+ has been (?:processed|issued|approved)",
        ]

        try:
            parsed = json.loads(text)
            has_tool_refs = bool(parsed.get("tools_required")) or bool(parsed.get("workflow_steps"))
        except (json.JSONDecodeError, AttributeError):
            has_tool_refs = "tool" in text_lower or "api" in text_lower

        penalty = 0.0
        for pattern in specific_claim_patterns:
            if re.search(pattern, text_lower) and not has_tool_refs:
                penalty -= 0.2
                break  # One penalty for fabrication
        return penalty
