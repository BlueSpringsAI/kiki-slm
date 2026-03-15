"""Deterministic A/B testing with statistical significance testing.

Task 3.32: Experiment assignment via user_id hashing,
chi-squared for rates, t-test for continuous metrics.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


class ABTestManager:
    """Deterministic A/B testing with statistical significance testing."""

    def __init__(self) -> None:
        self.experiments: dict[str, dict] = {}
        self.results: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

    def create_experiment(
        self,
        experiment_id: str,
        variants: list[str],
        traffic_split: dict[str, float],
    ) -> None:
        """Create a new A/B experiment."""
        total = sum(traffic_split.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Traffic split must sum to 1.0, got {total}")

        for v in variants:
            if v not in traffic_split:
                raise ValueError(f"Variant '{v}' missing from traffic_split")

        self.experiments[experiment_id] = {
            "variants": variants,
            "traffic_split": traffic_split,
        }
        logger.info("Created experiment '%s' with variants: %s", experiment_id, variants)

    def assign_variant(self, experiment_id: str, user_id: str) -> str:
        """Deterministic assignment via user_id hash."""
        if experiment_id not in self.experiments:
            raise KeyError(f"Experiment '{experiment_id}' not found")

        experiment = self.experiments[experiment_id]
        seed = f"{experiment_id}:{user_id}"
        bucket = int(hashlib.md5(seed.encode()).hexdigest(), 16) % 100

        cumulative = 0.0
        for variant, fraction in experiment["traffic_split"].items():
            cumulative += fraction * 100
            if bucket < cumulative:
                return variant

        return experiment["variants"][-1]

    def record_metric(
        self,
        experiment_id: str,
        variant: str,
        metric_name: str,
        value: float,
    ) -> None:
        """Record a metric observation."""
        self.results[experiment_id][variant][metric_name].append(value)

    def get_results(self, experiment_id: str) -> dict[str, Any]:
        """Get current results per variant."""
        if experiment_id not in self.results:
            return {}

        summary: dict[str, Any] = {}
        for variant, metrics in self.results[experiment_id].items():
            summary[variant] = {}
            for metric_name, values in metrics.items():
                n = len(values)
                mean = sum(values) / n if n else 0.0
                summary[variant][metric_name] = {
                    "count": n,
                    "mean": round(mean, 4),
                    "min": round(min(values), 4) if values else 0.0,
                    "max": round(max(values), 4) if values else 0.0,
                }
        return summary

    def check_significance(self, experiment_id: str, metric_name: str) -> dict[str, Any]:
        """Statistical significance testing between variants.

        Uses t-test for continuous metrics.
        """
        if experiment_id not in self.results:
            return {"significant": False, "reason": "No data"}

        variant_data = self.results[experiment_id]
        variants = list(variant_data.keys())

        if len(variants) < 2:
            return {"significant": False, "reason": "Need at least 2 variants"}

        # Get data for first two variants
        a_data = variant_data[variants[0]].get(metric_name, [])
        b_data = variant_data[variants[1]].get(metric_name, [])

        if len(a_data) < 5 or len(b_data) < 5:
            return {"significant": False, "reason": "Insufficient data (need >= 5 per variant)"}

        try:
            from scipy import stats

            t_stat, p_value = stats.ttest_ind(a_data, b_data)

            a_mean = sum(a_data) / len(a_data)
            b_mean = sum(b_data) / len(b_data)
            winner = variants[0] if a_mean > b_mean else variants[1]

            return {
                "significant": p_value < 0.05,
                "p_value": round(p_value, 6),
                "t_statistic": round(t_stat, 4),
                "winner": winner if p_value < 0.05 else None,
                "means": {variants[0]: round(a_mean, 4), variants[1]: round(b_mean, 4)},
            }
        except ImportError:
            # Fallback without scipy
            a_mean = sum(a_data) / len(a_data)
            b_mean = sum(b_data) / len(b_data)
            return {
                "significant": None,
                "reason": "scipy not installed for significance testing",
                "means": {variants[0]: round(a_mean, 4), variants[1]: round(b_mean, 4)},
            }
