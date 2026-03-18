"""Safety filter, response formatting, and escalation logic.

Task 3.31: Scan for PII, check policy violations, score confidence,
and apply escalation rules.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from kiki.data.validators import SLMOutput, parse_slm_output
from kiki.utils.logging import log_with_data

logger = logging.getLogger(__name__)

_PII_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+1[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b"),
}


class ResponsePostprocessor:
    """Post-process SLM output for safety and quality."""

    def __init__(self, confidence_threshold: float = 0.7) -> None:
        self.confidence_threshold = confidence_threshold

    def process(self, response: str, slm_output: SLMOutput | None = None) -> dict[str, Any]:
        """Full post-processing pipeline."""
        if slm_output is None:
            slm_output = parse_slm_output(response)

        flags: list[str] = []

        # 1. PII scan
        pii_found = self.scan_pii(response)
        if pii_found:
            flags.extend(f"pii:{p['type']}" for p in pii_found)

        # 2. Policy check
        if slm_output:
            violations = self.check_policy(slm_output)
            flags.extend(violations)

        # 3. Confidence scoring
        confidence = self.score_confidence(slm_output) if slm_output else 0.3

        # 4. Escalation check
        intent = slm_output.intent if slm_output else "unknown"
        escalate = self.should_escalate(confidence, flags, intent)

        # 5. Format response
        clean_response = self.format_response(
            slm_output.response if slm_output else response
        )

        result = {
            "response": clean_response,
            "safe": len(flags) == 0,
            "confidence": round(confidence, 3),
            "escalate": escalate,
            "flags": flags,
        }

        if flags:
            log_with_data(logger, logging.WARNING, "Safety flags triggered", {"flags": flags, "confidence": confidence, "escalate": escalate})
        if escalate:
            log_with_data(logger, logging.WARNING, "Request escalated to human", {"intent": intent, "confidence": confidence, "flags": flags})

        return result

    def scan_pii(self, text: str) -> list[dict]:
        """Detect PII patterns in response text."""
        found = []
        for pii_type, pattern in _PII_PATTERNS.items():
            for match in pattern.finditer(text):
                found.append({"type": pii_type, "start": match.start(), "end": match.end()})
        return found

    def check_policy(self, slm_output: SLMOutput) -> list[str]:
        """Check for policy violations."""
        violations = []
        text = slm_output.response.lower()

        # Check scope boundaries
        if any(kw in text for kw in ("legal advice", "i advise you to sue", "legally obligated")):
            violations.append("scope:legal_advice")
        if any(kw in text for kw in ("medical advice", "prescribe", "diagnosis")):
            violations.append("scope:medical_advice")

        # Check for unescalated fraud
        if slm_output.intent == "fraud_report":
            if not any(kw in text for kw in ("escalat", "security", "specialist", "supervisor")):
                violations.append("policy:fraud_not_escalated")

        return violations

    def score_confidence(self, slm_output: SLMOutput) -> float:
        """Heuristic confidence scoring based on output quality signals."""
        score = 0.5

        # Bonus for complete output
        if slm_output.intent and slm_output.urgency:
            score += 0.15
        if slm_output.workflow_steps:
            score += 0.10
        if slm_output.tools_required:
            score += 0.10
        if len(slm_output.response) > 50:
            score += 0.10
        if slm_output.reasoning:
            score += 0.05

        return min(1.0, score)

    def should_escalate(self, confidence: float, violations: list[str], intent: str) -> bool:
        """Determine if request should be escalated to a human."""
        if confidence < self.confidence_threshold:
            return True
        if any(v.startswith("policy:") for v in violations):
            return True
        if any(v.startswith("pii:") for v in violations):
            return True
        if intent in ("fraud_report",):
            return True
        return False

    def format_response(self, response: str) -> str:
        """Clean up response formatting."""
        text = response.strip()
        # Remove leading/trailing quotes
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        # Unescape
        text = text.replace("\\n", "\n").replace('\\"', '"')
        return text
