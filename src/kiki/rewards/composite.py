"""Composite reward function and format validity reward for GRPO.

Task 3.20: Combines policy compliance, tool accuracy, response quality,
and format validity rewards with configurable weights.
"""

from __future__ import annotations

import json
import logging

from kiki.data.validators import parse_slm_output
from kiki.rewards.policy_compliance import PolicyComplianceReward
from kiki.rewards.response_quality import ResponseQualityReward
from kiki.rewards.tool_accuracy import ToolAccuracyReward

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Format validity reward
# ---------------------------------------------------------------------------


class FormatValidityReward:
    """Reward for producing well-structured JSON matching the SLMOutput schema.

    Scoring:
        1.0  — valid JSON that passes SLMOutput validation
        0.5  — valid JSON but doesn't match SLMOutput schema
        0.0  — invalid JSON or empty
    """

    def __call__(self, completions: list[str], **kwargs) -> list[float]:
        return [self._score_single(c) for c in completions]

    def _score_single(self, completion: str) -> float:
        text = completion.strip()
        if not text:
            return 0.0

        # Check valid JSON
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return 0.0

        if not isinstance(parsed, dict):
            return 0.0

        # Check SLMOutput schema
        slm_output = parse_slm_output(text)
        if slm_output is not None:
            return 1.0

        # Valid JSON but wrong schema — give partial credit
        return 0.5


# ---------------------------------------------------------------------------
# Composite reward
# ---------------------------------------------------------------------------


class CompositeReward:
    """Weighted combination of all reward functions for GRPO training.

    Default weights:
        policy_compliance  0.35
        tool_accuracy      0.25
        response_quality   0.25
        format_validity    0.15
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "policy_compliance": 0.35,
        "tool_accuracy": 0.25,
        "response_quality": 0.25,
        "format_validity": 0.15,
    }

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)
        self.rewards: dict[str, object] = {
            "policy_compliance": PolicyComplianceReward(),
            "tool_accuracy": ToolAccuracyReward(),
            "response_quality": ResponseQualityReward(),
            "format_validity": FormatValidityReward(),
        }
        # Validate weights sum roughly to 1
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning("Reward weights sum to %.3f (expected ~1.0)", total)

    def __call__(self, completions: list[str], **kwargs) -> list[float]:
        """Compute weighted sum of all reward components."""
        batch_size = len(completions)
        combined = [0.0] * batch_size

        component_scores: dict[str, list[float]] = {}
        for name, reward_fn in self.rewards.items():
            weight = self.weights.get(name, 0.0)
            if weight == 0.0:
                continue
            scores = reward_fn(completions, **kwargs)
            component_scores[name] = scores
            for i in range(batch_size):
                combined[i] += weight * scores[i]

        # Log average scores per component for monitoring
        if logger.isEnabledFor(logging.DEBUG):
            for name, scores in component_scores.items():
                avg = sum(scores) / len(scores) if scores else 0.0
                logger.debug("Reward component '%s': avg=%.3f", name, avg)

        return combined

    def score_detailed(self, completions: list[str], **kwargs) -> list[dict]:
        """Return per-component scores for analysis."""
        batch_size = len(completions)
        details = [{"total": 0.0} for _ in range(batch_size)]

        for name, reward_fn in self.rewards.items():
            weight = self.weights.get(name, 0.0)
            scores = reward_fn(completions, **kwargs)
            for i in range(batch_size):
                details[i][name] = scores[i]
                details[i]["total"] += weight * scores[i]

        return details
