"""Intent classification test suite.

Task 3.24: Tests intent and urgency classification accuracy.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from kiki.data.validators import parse_slm_output
from kiki.evaluation.metrics import EvaluationMetrics

logger = logging.getLogger(__name__)


class IntentTestSuite:
    """Test intent classification accuracy against gold labels."""

    def __init__(self, test_data: list[dict]) -> None:
        self.test_data = [d for d in test_data if "intent" in d and "customer_message" in d]
        if not self.test_data:
            logger.warning("No valid intent test data found")

    def run(self, predict_fn: Callable[[str], Any]) -> dict:
        """Run predictions and compute intent/urgency metrics."""
        gold_intents = []
        pred_intents = []
        gold_urgency = []
        pred_urgency = []

        for example in self.test_data:
            message = example["customer_message"]
            output = predict_fn(message)

            # Parse prediction
            if isinstance(output, str):
                parsed = parse_slm_output(output)
            elif isinstance(output, dict):
                parsed = type("Obj", (), output)()
            else:
                parsed = output

            gold_intents.append(example["intent"])
            pred_intents.append(getattr(parsed, "intent", "unknown") if parsed else "unknown")

            if "urgency" in example:
                gold_urgency.append(example["urgency"])
                pred_urgency.append(getattr(parsed, "urgency", "medium") if parsed else "medium")

        results = {
            "intent_f1": EvaluationMetrics.intent_f1(pred_intents, gold_intents),
            "intent_accuracy": EvaluationMetrics.decision_accuracy(pred_intents, gold_intents),
            "num_examples": len(self.test_data),
        }

        if gold_urgency:
            results["urgency_accuracy"] = EvaluationMetrics.decision_accuracy(pred_urgency, gold_urgency)

        return results
