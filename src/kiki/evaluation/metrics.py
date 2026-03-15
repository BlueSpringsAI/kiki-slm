"""Custom evaluation metrics for the Kiki SLM pipeline.

Task 3.22: Intent F1, workflow edit distance, tool set F1,
decision accuracy, and schema validity rate.
"""

from __future__ import annotations

import logging
from typing import Any

from kiki.data.validators import parse_slm_output

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Static methods for computing Kiki SLM evaluation metrics."""

    @staticmethod
    def intent_f1(predictions: list[str], labels: list[str]) -> dict[str, Any]:
        """Micro/macro F1 with per-class breakdown.

        Returns:
            {"micro_f1": float, "macro_f1": float, "per_class": dict}
        """
        from sklearn.metrics import classification_report, f1_score

        micro = f1_score(labels, predictions, average="micro", zero_division=0)
        macro = f1_score(labels, predictions, average="macro", zero_division=0)

        report = classification_report(labels, predictions, output_dict=True, zero_division=0)
        per_class = {
            k: {"precision": v["precision"], "recall": v["recall"], "f1": v["f1-score"]}
            for k, v in report.items()
            if k not in ("accuracy", "macro avg", "weighted avg", "micro avg")
        }

        return {"micro_f1": round(micro, 4), "macro_f1": round(macro, 4), "per_class": per_class}

    @staticmethod
    def workflow_edit_distance(
        predicted_steps: list[list[str]],
        gold_steps: list[list[str]],
    ) -> dict[str, Any]:
        """Normalized Levenshtein distance on step lists.

        Returns:
            {"mean_accuracy": float, "per_example": list[float]}
        """
        try:
            import Levenshtein
        except ImportError:
            logger.warning("python-Levenshtein not installed; using basic comparison")
            return EvaluationMetrics._basic_workflow_accuracy(predicted_steps, gold_steps)

        per_example = []
        for pred, gold in zip(predicted_steps, gold_steps):
            pred_str = " → ".join(pred)
            gold_str = " → ".join(gold)
            if not gold_str:
                per_example.append(1.0 if not pred_str else 0.0)
                continue
            dist = Levenshtein.distance(pred_str, gold_str)
            max_len = max(len(pred_str), len(gold_str))
            accuracy = 1.0 - (dist / max_len) if max_len > 0 else 1.0
            per_example.append(round(accuracy, 4))

        mean_acc = sum(per_example) / len(per_example) if per_example else 0.0
        return {"mean_accuracy": round(mean_acc, 4), "per_example": per_example}

    @staticmethod
    def _basic_workflow_accuracy(
        predicted_steps: list[list[str]], gold_steps: list[list[str]]
    ) -> dict[str, Any]:
        """Fallback workflow accuracy using set overlap."""
        per_example = []
        for pred, gold in zip(predicted_steps, gold_steps):
            if not gold:
                per_example.append(1.0 if not pred else 0.0)
                continue
            overlap = len(set(pred) & set(gold))
            accuracy = overlap / len(gold)
            per_example.append(round(accuracy, 4))
        mean_acc = sum(per_example) / len(per_example) if per_example else 0.0
        return {"mean_accuracy": round(mean_acc, 4), "per_example": per_example}

    @staticmethod
    def tool_set_f1(
        predicted_tools: list[list[str]],
        gold_tools: list[list[str]],
    ) -> dict[str, float]:
        """Set-based precision/recall/F1 for tool selection.

        Returns:
            {"precision": float, "recall": float, "f1": float}
        """
        total_tp = 0
        total_pred = 0
        total_gold = 0

        for pred, gold in zip(predicted_tools, gold_tools):
            pred_set = set(pred)
            gold_set = set(gold)
            total_tp += len(pred_set & gold_set)
            total_pred += len(pred_set)
            total_gold += len(gold_set)

        precision = total_tp / total_pred if total_pred > 0 else 0.0
        recall = total_tp / total_gold if total_gold > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    @staticmethod
    def decision_accuracy(predicted: list[str], gold: list[str]) -> float:
        """Multi-class accuracy for operational decisions."""
        if not gold:
            return 0.0
        correct = sum(1 for p, g in zip(predicted, gold) if p == g)
        return round(correct / len(gold), 4)

    @staticmethod
    def schema_validity_rate(outputs: list[str]) -> dict[str, Any]:
        """Percentage of outputs that parse as valid JSON matching SLMOutput schema.

        Returns:
            {"valid_json": float, "valid_schema": float, "total": int}
        """
        import json

        total = len(outputs)
        valid_json = 0
        valid_schema = 0

        for text in outputs:
            try:
                json.loads(text)
                valid_json += 1
            except (json.JSONDecodeError, TypeError):
                continue

            if parse_slm_output(text) is not None:
                valid_schema += 1

        return {
            "valid_json": round(valid_json / total, 4) if total else 0.0,
            "valid_schema": round(valid_schema / total, 4) if total else 0.0,
            "total": total,
        }
