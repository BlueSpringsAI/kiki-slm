"""Preference pair construction for the Kiki SLM pipeline.

Task 3.5: Build DPO/SimPO training pairs from scored responses,
HelpSteer, UltraFeedback, human corrections, and CSAT scores.
"""

from __future__ import annotations

import logging
from typing import Any

from kiki.data.validators import PreferencePair

logger = logging.getLogger(__name__)


class PreferencePairBuilder:
    """Construct preference pairs for DPO/SimPO training."""

    @staticmethod
    def from_scored_responses(
        prompt: list[dict],
        responses: list[str],
        scores: list[float],
        min_margin: float = 0.5,
    ) -> dict | None:
        """Pair highest vs lowest scored response. Returns None if margin too small."""
        if len(responses) < 2 or len(responses) != len(scores):
            return None

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        worst_idx = min(range(len(scores)), key=lambda i: scores[i])

        if best_idx == worst_idx:
            return None
        if scores[best_idx] - scores[worst_idx] < min_margin:
            return None

        return {
            "prompt": prompt,
            "chosen": [{"role": "assistant", "content": responses[best_idx]}],
            "rejected": [{"role": "assistant", "content": responses[worst_idx]}],
        }

    @staticmethod
    def from_helpsteer(example: dict) -> dict | None:
        """Convert HelpSteer2/3 attribute scores to binary preferences.

        Uses average of helpfulness + correctness as quality signal.
        """
        # If already has chosen/rejected, pass through
        if "chosen" in example and "rejected" in example:
            return {
                "prompt": example.get("prompt", []),
                "chosen": example["chosen"] if isinstance(example["chosen"], list) else [{"role": "assistant", "content": example["chosen"]}],
                "rejected": example["rejected"] if isinstance(example["rejected"], list) else [{"role": "assistant", "content": example["rejected"]}],
            }

        # HelpSteer format: prompt, response, helpfulness, correctness, coherence, complexity, verbosity
        prompt_text = example.get("prompt", "")
        response_text = example.get("response", "")
        helpfulness = example.get("helpfulness", 3.0)
        correctness = example.get("correctness", 3.0)

        quality = (float(helpfulness) + float(correctness)) / 2.0

        if quality >= 4.0:
            # Good response — can be used as chosen
            return {
                "prompt": [{"role": "user", "content": prompt_text}],
                "chosen": [{"role": "assistant", "content": response_text}],
                "rejected": [],  # Needs pairing later
                "_quality": quality,
                "_needs_pairing": True,
            }
        elif quality <= 2.0:
            return {
                "prompt": [{"role": "user", "content": prompt_text}],
                "chosen": [],  # Needs pairing later
                "rejected": [{"role": "assistant", "content": response_text}],
                "_quality": quality,
                "_needs_pairing": True,
            }

        return None  # Ambiguous quality, skip

    @staticmethod
    def from_ultrafeedback(example: dict) -> dict | None:
        """Use pre-computed chosen/rejected from Argilla cleaned version."""
        chosen = example.get("chosen", [])
        rejected = example.get("rejected", [])

        if not chosen or not rejected:
            return None

        # UltraFeedback already has message lists
        prompt = []
        if isinstance(chosen, list) and len(chosen) > 0:
            # Extract prompt messages (non-assistant turns)
            for msg in chosen:
                if isinstance(msg, dict) and msg.get("role") != "assistant":
                    prompt.append(msg)

            chosen_msgs = [msg for msg in chosen if isinstance(msg, dict) and msg.get("role") == "assistant"]
            rejected_msgs = [msg for msg in rejected if isinstance(msg, dict) and msg.get("role") == "assistant"] if isinstance(rejected, list) else [{"role": "assistant", "content": str(rejected)}]
        else:
            chosen_msgs = [{"role": "assistant", "content": str(chosen)}]
            rejected_msgs = [{"role": "assistant", "content": str(rejected)}]

        if not prompt:
            prompt = [{"role": "user", "content": example.get("prompt", example.get("instruction", ""))}]

        return {"prompt": prompt, "chosen": chosen_msgs, "rejected": rejected_msgs}

    @staticmethod
    def from_on_policy_generation(
        prompt: list[dict],
        responses: list[str],
        scores: list[float],
    ) -> dict | None:
        """Given responses and their scores, pair best vs worst."""
        if len(responses) < 2 or len(responses) != len(scores):
            return None

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        worst_idx = min(range(len(scores)), key=lambda i: scores[i])

        if best_idx == worst_idx:
            return None

        return {
            "prompt": prompt,
            "chosen": [{"role": "assistant", "content": responses[best_idx]}],
            "rejected": [{"role": "assistant", "content": responses[worst_idx]}],
        }

    @staticmethod
    def from_human_corrections(
        original_response: str,
        corrected_response: str,
        prompt_messages: list[dict],
    ) -> dict | None:
        """Agent edits as preference signal. Corrected = chosen, original = rejected."""
        if original_response == corrected_response:
            return None

        return {
            "prompt": prompt_messages,
            "chosen": [{"role": "assistant", "content": corrected_response}],
            "rejected": [{"role": "assistant", "content": original_response}],
        }

    @staticmethod
    def from_csat_scores(
        conversation: str,
        csat_score: int | float,
        prompt_messages: list[dict],
    ) -> dict | None:
        """CSAT 4-5 as chosen, 1-2 as rejected. Returns None for score 3 (ambiguous)."""
        score = int(csat_score)
        if score >= 4:
            return {
                "prompt": prompt_messages,
                "chosen": [{"role": "assistant", "content": conversation}],
                "rejected": [],
                "_csat": score,
                "_needs_pairing": True,
            }
        elif score <= 2:
            return {
                "prompt": prompt_messages,
                "chosen": [],
                "rejected": [{"role": "assistant", "content": conversation}],
                "_csat": score,
                "_needs_pairing": True,
            }
        return None  # Score 3 is ambiguous

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    @classmethod
    def build_pairs(cls, dataset: Any, strategy: str) -> list[dict]:
        """Apply the named strategy to each example in the dataset."""
        strategy_map = {
            "helpsteer": cls.from_helpsteer,
            "ultrafeedback": cls.from_ultrafeedback,
        }

        converter = strategy_map.get(strategy)
        if converter is None:
            raise ValueError(f"Unknown strategy '{strategy}'. Available: {list(strategy_map.keys())}")

        pairs = []
        skipped = 0
        for item in dataset:
            row = dict(item) if not isinstance(item, dict) else item
            result = converter(row)
            if result is not None:
                pairs.append(result)
            else:
                skipped += 1

        logger.info("Built %d preference pairs (skipped %d) using strategy '%s'", len(pairs), skipped, strategy)
        return pairs

    @staticmethod
    def validate_pairs(pairs: list[dict]) -> tuple[list[dict], list[str]]:
        """Validate that chosen/rejected are different and prompt is non-empty.

        Returns (valid_pairs, error_messages).
        """
        valid = []
        errors = []

        for i, pair in enumerate(pairs):
            # Skip incomplete pairs that need pairing
            if pair.get("_needs_pairing"):
                continue

            try:
                PreferencePair.model_validate(pair)
                valid.append(pair)
            except Exception as exc:
                errors.append(f"Pair {i}: {exc}")

        logger.info("Validated %d/%d pairs (%d errors)", len(valid), len(pairs), len(errors))
        return valid, errors
