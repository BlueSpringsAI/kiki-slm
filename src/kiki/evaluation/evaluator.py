"""Evaluation orchestrator that runs all test suites and aggregates results.

Task 3.21: Run intent, tool calling, workflow, and safety test suites,
produce comprehensive evaluation reports.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kiki.evaluation.judges import LLMJudge
from kiki.evaluation.metrics import EvaluationMetrics

logger = logging.getLogger(__name__)


class Evaluator:
    """Orchestrator for running evaluation suites and generating reports."""

    AVAILABLE_SUITES = ("intent", "tool_calling", "workflow", "safety")

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self.metrics = EvaluationMetrics()
        self._judge = None

    @property
    def judge(self) -> LLMJudge:
        if self._judge is None:
            if self.config and hasattr(self.config, "judge"):
                self._judge = LLMJudge(
                    provider=getattr(self.config.judge, "provider", "openai"),
                    model=getattr(self.config.judge, "model", "gpt-4o"),
                )
            else:
                self._judge = LLMJudge()
        return self._judge

    def run_full_evaluation(
        self,
        predict_fn,
        test_data: list[dict],
        suites: list[str] | None = None,
    ) -> dict:
        """Run all (or selected) test suites and aggregate results."""
        suites = suites or list(self.AVAILABLE_SUITES)
        results: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_test_examples": len(test_data),
            "suites": {},
        }

        for suite_name in suites:
            logger.info("Running test suite: %s", suite_name)
            try:
                suite_results = self.run_suite(suite_name, predict_fn, test_data)
                results["suites"][suite_name] = suite_results
                logger.info("Suite '%s' complete: %s", suite_name, {k: v for k, v in suite_results.items() if isinstance(v, (int, float))})
            except Exception as exc:
                logger.error("Suite '%s' failed: %s", suite_name, exc)
                results["suites"][suite_name] = {"error": str(exc)}

        return results

    def run_suite(self, suite_name: str, predict_fn, test_data: list[dict]) -> dict:
        """Run a single test suite."""
        from kiki.evaluation.test_suites import (
            IntentTestSuite,
            SafetyTestSuite,
            ToolCallingTestSuite,
            WorkflowTestSuite,
        )

        suite_map = {
            "intent": IntentTestSuite,
            "tool_calling": ToolCallingTestSuite,
            "workflow": WorkflowTestSuite,
            "safety": SafetyTestSuite,
        }

        suite_cls = suite_map.get(suite_name)
        if suite_cls is None:
            raise ValueError(f"Unknown suite '{suite_name}'. Available: {list(suite_map.keys())}")

        suite = suite_cls(test_data)
        return suite.run(predict_fn)

    def generate_report(self, results: dict, output_path: str | None = None) -> dict:
        """Format results as JSON report, optionally save to file."""
        report = {
            "kiki_slm_evaluation_report": True,
            **results,
        }

        # Check against targets if configured
        if self.config and hasattr(self.config, "targets"):
            targets = dict(self.config.targets)
            report["target_checks"] = {}
            for metric, target in targets.items():
                actual = self._find_metric(results, metric)
                if actual is not None:
                    passed = actual >= float(target)
                    report["target_checks"][metric] = {
                        "target": float(target),
                        "actual": actual,
                        "passed": passed,
                    }

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info("Report saved to '%s'", path)

        return report

    def compare_models(self, results_a: dict, results_b: dict) -> dict:
        """Compare two evaluation runs, highlight improvements/regressions."""
        comparison: dict[str, Any] = {"improvements": [], "regressions": [], "unchanged": []}

        for suite_name in set(results_a.get("suites", {})) | set(results_b.get("suites", {})):
            suite_a = results_a.get("suites", {}).get(suite_name, {})
            suite_b = results_b.get("suites", {}).get(suite_name, {})

            for key in set(suite_a) | set(suite_b):
                val_a = suite_a.get(key)
                val_b = suite_b.get(key)
                if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                    diff = val_b - val_a
                    entry = {"suite": suite_name, "metric": key, "before": val_a, "after": val_b, "diff": round(diff, 4)}
                    if abs(diff) < 0.001:
                        comparison["unchanged"].append(entry)
                    elif diff > 0:
                        comparison["improvements"].append(entry)
                    else:
                        comparison["regressions"].append(entry)

        return comparison

    @staticmethod
    def _find_metric(results: dict, metric_name: str) -> float | None:
        """Search for a metric across all suites."""
        for suite_results in results.get("suites", {}).values():
            if isinstance(suite_results, dict) and metric_name in suite_results:
                val = suite_results[metric_name]
                if isinstance(val, (int, float)):
                    return float(val)
        return None
