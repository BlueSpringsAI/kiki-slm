"""Tool calling test suite.

Task 3.25: Tests tool selection and parameter accuracy.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from kiki.data.validators import parse_slm_output
from kiki.evaluation.metrics import EvaluationMetrics

logger = logging.getLogger(__name__)


class ToolCallingTestSuite:
    """Test tool selection and parameter accuracy against gold labels."""

    def __init__(self, test_data: list[dict]) -> None:
        self.test_data = [d for d in test_data if "tools_required" in d and "customer_message" in d]
        if not self.test_data:
            logger.warning("No valid tool calling test data found")

    def run(self, predict_fn: Callable[[str], Any]) -> dict:
        """Run predictions and compute tool selection metrics."""
        gold_tools: list[list[str]] = []
        pred_tools: list[list[str]] = []
        errors: list[dict] = []

        for i, example in enumerate(self.test_data):
            message = example["customer_message"]
            output = predict_fn(message)

            if isinstance(output, str):
                parsed = parse_slm_output(output)
            elif isinstance(output, dict):
                parsed = type("Obj", (), output)()
            else:
                parsed = output

            gold = example["tools_required"]
            if isinstance(gold, str):
                gold = [gold]
            gold_tools.append(gold)

            pred = getattr(parsed, "tools_required", []) if parsed else []
            if isinstance(pred, str):
                pred = [pred]
            pred_tools.append(pred)

            # Track errors
            gold_set = set(gold)
            pred_set = set(pred)
            if gold_set != pred_set:
                errors.append({
                    "index": i,
                    "missing": list(gold_set - pred_set),
                    "extra": list(pred_set - gold_set),
                })

        tool_f1 = EvaluationMetrics.tool_set_f1(pred_tools, gold_tools)

        return {
            "tool_f1": tool_f1["f1"],
            "tool_precision": tool_f1["precision"],
            "tool_recall": tool_f1["recall"],
            "num_examples": len(self.test_data),
            "num_errors": len(errors),
            "errors": errors[:10],  # First 10 errors for review
        }
