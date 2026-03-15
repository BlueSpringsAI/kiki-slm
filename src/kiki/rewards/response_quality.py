"""Response quality reward function for GRPO.

Task 3.19: Evaluates customer-facing response quality on
concreteness, tone, length, and relevance.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quality signals
# ---------------------------------------------------------------------------

_ACTION_WORDS = (
    "will", "going to", "have", "has been", "processed",
    "submitted", "initiated", "scheduled", "arranged",
    "confirmed", "updated", "resolved", "forwarded",
    "refund", "replacement", "credit", "track",
)

_INFORMAL_PATTERNS = re.compile(
    r"\b(?:gonna|wanna|gotta|ya|nah|lol|omg|btw|idk|tbh|imho|bruh|dude|bro)\b",
    re.IGNORECASE,
)

_EXCESSIVE_PUNCTUATION = re.compile(r"[!?]{3,}")
_ALL_CAPS_WORDS = re.compile(r"\b[A-Z]{4,}\b")

# Known abbreviations / acronyms to exclude from ALL_CAPS check
_ALLOWED_CAPS = {"API", "SLM", "JSON", "CSV", "HTML", "URL", "FAQ", "ETA", "ID", "ASAP"}


class ResponseQualityReward:
    """GRPO reward for customer-facing response quality.

    Scoring dimensions (each 0.0 to ~0.25, summed to ~1.0):
        - Concreteness: contains specific next steps / action words
        - Length: appropriate length (not too short or verbose)
        - Tone: professional, no slang, no ALL CAPS
        - Relevance: addresses the customer directly
    """

    def __call__(self, completions: list[str], **kwargs) -> list[float]:
        """Score a batch of completions."""
        return [self._score_single(c) for c in completions]

    def _score_single(self, completion: str) -> float:
        # Extract the customer-facing response from JSON if possible
        response_text = self._extract_response(completion)
        if not response_text:
            return 0.0

        scores = [
            self._score_concreteness(response_text),
            self._score_length(response_text),
            self._score_tone(response_text),
            self._score_relevance(response_text),
        ]
        return sum(scores)

    # ------------------------------------------------------------------
    # Response extraction
    # ------------------------------------------------------------------

    def _extract_response(self, completion: str) -> str:
        """Extract the customer-facing response text from completion."""
        # Try JSON parsing first
        try:
            parsed = json.loads(completion)
            if isinstance(parsed, dict):
                resp = parsed.get("response", "")
                if resp:
                    return str(resp)
        except json.JSONDecodeError:
            pass
        # Fall back to raw text
        return completion.strip()

    # ------------------------------------------------------------------
    # Scoring dimensions
    # ------------------------------------------------------------------

    def _score_concreteness(self, text: str) -> float:
        """Check for concrete next steps and action words (0.0 to 0.25)."""
        text_lower = text.lower()
        action_count = sum(1 for word in _ACTION_WORDS if word in text_lower)
        if action_count >= 3:
            return 0.25
        if action_count >= 1:
            return 0.15
        return 0.05  # Vague response

    def _score_length(self, text: str) -> float:
        """Check appropriate response length (0.0 to 0.25)."""
        char_count = len(text)
        if char_count < 20:
            return 0.0  # Too short
        if char_count < 50:
            return 0.10
        if char_count > 2000:
            return 0.05  # Too verbose
        if char_count > 1000:
            return 0.15
        return 0.25  # Sweet spot: 50-1000 chars

    def _score_tone(self, text: str) -> float:
        """Check for professional tone (0.0 to 0.25)."""
        score = 0.25

        # Penalty for informal language
        if _INFORMAL_PATTERNS.search(text):
            score -= 0.15

        # Penalty for excessive punctuation
        if _EXCESSIVE_PUNCTUATION.search(text):
            score -= 0.10

        # Penalty for ALL CAPS words (excluding known abbreviations)
        caps_words = _ALL_CAPS_WORDS.findall(text)
        non_allowed_caps = [w for w in caps_words if w not in _ALLOWED_CAPS]
        if len(non_allowed_caps) > 2:
            score -= 0.10

        return max(0.0, score)

    def _score_relevance(self, text: str) -> float:
        """Check if response addresses the customer (0.0 to 0.25)."""
        text_lower = text.lower()
        score = 0.0

        # Addresses the customer
        if any(word in text_lower for word in ("you", "your", "you're")):
            score += 0.10

        # Shows empathy
        empathy_words = ("sorry", "understand", "appreciate", "apologize", "inconvenience", "frustrat")
        if any(word in text_lower for word in empathy_words):
            score += 0.10

        # Provides context or explanation
        context_words = ("because", "reason", "due to", "this means", "next step")
        if any(word in text_lower for word in context_words):
            score += 0.05

        return min(0.25, score)
