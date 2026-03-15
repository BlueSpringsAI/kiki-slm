"""Workflow planning test suite.

Task 3.26: Tests workflow step planning accuracy via edit distance.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from kiki.data.validators import parse_slm_output
from kiki.evaluation.metrics import EvaluationMetrics

logger = logging.getLogger(__name__)


class WorkflowTestSuite:
    """Test workflow step planning accuracy against gold labels."""

    def __init__(self, test_data: list[dict]) -> None:
        self.test_data = [d for d in test_data if "workflow_steps" in d and "customer_message" in d]
        if not self.test_data:
            logger.warning("No valid workflow test data found")

    def run(self, predict_fn: Callable[[str], Any]) -> dict:
        """Run predictions and compute workflow accuracy."""
        gold_steps: list[list[str]] = []
        pred_steps: list[list[str]] = []

        for example in self.test_data:
            message = example["customer_message"]
            output = predict_fn(message)

            if isinstance(output, str):
                parsed = parse_slm_output(output)
            elif isinstance(output, dict):
                parsed = type("Obj", (), output)()
            else:
                parsed = output

            gold = example["workflow_steps"]
            if isinstance(gold, str):
                gold = [gold]
            gold_steps.append(gold)

            pred = getattr(parsed, "workflow_steps", []) if parsed else []
            if isinstance(pred, str):
                pred = [pred]
            pred_steps.append(pred)

        result = EvaluationMetrics.workflow_edit_distance(pred_steps, gold_steps)

        return {
            "workflow_accuracy": result["mean_accuracy"],
            "num_examples": len(self.test_data),
            "per_example": result["per_example"][:20],  # First 20 for review
        }
