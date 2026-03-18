"""Tracked wrapper around CompositeReward for GRPO per-component logging.

Calls score_detailed() on every reward invocation and pushes per-component
averages to ExperimentTracker.
"""

from __future__ import annotations

import logging
from typing import Any

from kiki.rewards.composite import CompositeReward
from kiki.utils.experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)


class TrackedReward:
    """Wraps CompositeReward to log per-component scores every call."""

    def __init__(self, composite: CompositeReward, tracker: ExperimentTracker) -> None:
        self.composite = composite
        self.tracker = tracker
        self._call_count = 0

    def __call__(self, completions: list[str], **kwargs: Any) -> list[float]:
        self._call_count += 1

        # Use score_detailed to get per-component breakdown
        details = self.composite.score_detailed(completions, **kwargs)

        # Compute per-component averages
        batch_size = len(details)
        if batch_size == 0:
            return []

        component_names = [k for k in details[0] if k != "total"]
        averages: dict[str, float] = {}
        for name in component_names:
            avg = sum(d.get(name, 0.0) for d in details) / batch_size
            averages[name] = avg

        total_avg = sum(d["total"] for d in details) / batch_size

        # Push to ExperimentTracker
        metrics = {f"reward/{name}": avg for name, avg in averages.items()}
        metrics["reward/total"] = total_avg
        self.tracker.log_metrics(metrics)

        # Human-readable summary
        parts = [f"{name}={avg:.2f}" for name, avg in averages.items()]
        logger.info(
            "Rewards: %s | total=%.2f",
            " | ".join(parts),
            total_avg,
        )

        # Return the combined scores (same as CompositeReward.__call__)
        return [d["total"] for d in details]
