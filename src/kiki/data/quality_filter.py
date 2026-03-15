"""Data quality filtering for the Kiki SLM pipeline.

Task 3.6: Clean and filter training data — deduplication, length filtering,
language detection, confidence thresholds, intent balancing, and edge case checks.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from typing import Any

import datasets

logger = logging.getLogger(__name__)


class QualityFilter:
    """Filter and clean training datasets."""

    def dedup_exact(self, dataset: datasets.Dataset, column: str = "customer_message") -> datasets.Dataset:
        """Remove exact duplicates based on the given column."""
        if column not in dataset.column_names:
            logger.warning("Column '%s' not found, skipping exact dedup", column)
            return dataset

        seen: set[str] = set()
        keep_indices: list[int] = []
        for i, text in enumerate(dataset[column]):
            h = hashlib.md5(str(text).encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                keep_indices.append(i)

        removed = len(dataset) - len(keep_indices)
        logger.info("Exact dedup: removed %d duplicates (%d → %d)", removed, len(dataset), len(keep_indices))
        return dataset.select(keep_indices)

    def dedup_semantic(
        self,
        dataset: datasets.Dataset,
        threshold: float = 0.95,
        column: str = "customer_message",
        batch_size: int = 512,
    ) -> datasets.Dataset:
        """Remove near-duplicates using sentence-transformer embeddings."""
        if column not in dataset.column_names:
            logger.warning("Column '%s' not found, skipping semantic dedup", column)
            return dataset

        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError:
            logger.warning("sentence-transformers not installed, skipping semantic dedup")
            return dataset

        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = dataset[column]
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)

        keep_indices: list[int] = []
        removed = 0
        for i in range(len(embeddings)):
            is_dup = False
            for j in keep_indices[-100:]:  # Compare against last 100 kept items for efficiency
                sim = float(np.dot(embeddings[i], embeddings[j]))
                if sim >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                keep_indices.append(i)
            else:
                removed += 1

        logger.info("Semantic dedup: removed %d near-duplicates (%d → %d)", removed, len(dataset), len(keep_indices))
        return dataset.select(keep_indices)

    def filter_length(
        self,
        dataset: datasets.Dataset,
        min_tokens: int = 10,
        max_tokens: int = 2000,
        column: str = "customer_message",
    ) -> datasets.Dataset:
        """Remove examples that are too short or too long (whitespace token approximation)."""
        if column not in dataset.column_names:
            logger.warning("Column '%s' not found, skipping length filter", column)
            return dataset

        original = len(dataset)
        dataset = dataset.filter(
            lambda x: min_tokens <= len(str(x[column]).split()) <= max_tokens,
            desc="Filtering by length",
        )
        removed = original - len(dataset)
        logger.info("Length filter: removed %d examples (%d → %d)", removed, original, len(dataset))
        return dataset

    def filter_language(
        self,
        dataset: datasets.Dataset,
        allowed_languages: list[str] | None = None,
        column: str = "customer_message",
        max_non_ascii_ratio: float = 0.3,
    ) -> datasets.Dataset:
        """Filter by language using ASCII ratio heuristic."""
        if allowed_languages is None:
            allowed_languages = ["en"]
        if column not in dataset.column_names:
            logger.warning("Column '%s' not found, skipping language filter", column)
            return dataset

        # Simple heuristic: English text should be mostly ASCII
        if "en" in allowed_languages:
            original = len(dataset)

            def _is_english(example: dict) -> bool:
                text = str(example[column])
                if not text:
                    return False
                non_ascii = sum(1 for c in text if ord(c) > 127)
                return (non_ascii / max(len(text), 1)) < max_non_ascii_ratio

            dataset = dataset.filter(_is_english, desc="Filtering by language")
            removed = original - len(dataset)
            logger.info("Language filter: removed %d non-English examples (%d → %d)", removed, original, len(dataset))

        return dataset

    def filter_confidence(
        self,
        dataset: datasets.Dataset,
        min_confidence: float = 0.8,
        column: str = "confidence",
    ) -> datasets.Dataset:
        """Remove low-confidence annotations."""
        if column not in dataset.column_names:
            logger.warning("Column '%s' not found, skipping confidence filter", column)
            return dataset

        original = len(dataset)
        dataset = dataset.filter(lambda x: float(x[column]) >= min_confidence, desc="Filtering by confidence")
        removed = original - len(dataset)
        logger.info(
            "Confidence filter (>= %.2f): removed %d examples (%d → %d)",
            min_confidence,
            removed,
            original,
            len(dataset),
        )
        return dataset

    def filter_quality_score(
        self,
        dataset: datasets.Dataset,
        min_score: float = 3.0,
        column: str = "quality_score",
    ) -> datasets.Dataset:
        """Remove low-quality responses."""
        if column not in dataset.column_names:
            logger.warning("Column '%s' not found, skipping quality score filter", column)
            return dataset

        original = len(dataset)
        dataset = dataset.filter(lambda x: float(x[column]) >= min_score, desc="Filtering by quality score")
        removed = original - len(dataset)
        logger.info(
            "Quality filter (>= %.1f): removed %d examples (%d → %d)", min_score, removed, original, len(dataset)
        )
        return dataset

    def balance_intents(
        self,
        dataset: datasets.Dataset,
        max_ratio: float = 0.3,
        column: str = "intent",
    ) -> datasets.Dataset:
        """Downsample any intent exceeding max_ratio of the dataset."""
        if column not in dataset.column_names:
            logger.warning("Column '%s' not found, skipping intent balancing", column)
            return dataset

        counts = Counter(dataset[column])
        max_per_intent = int(len(dataset) * max_ratio)

        # Find intents that exceed the threshold
        over_represented = {intent: count for intent, count in counts.items() if count > max_per_intent}
        if not over_represented:
            logger.info("Intent balance: all intents within %.0f%% threshold", max_ratio * 100)
            return dataset

        logger.info("Over-represented intents: %s (max per intent: %d)", over_represented, max_per_intent)

        # Build keep indices
        intent_indices: dict[str, list[int]] = {}
        for i, intent in enumerate(dataset[column]):
            intent_indices.setdefault(intent, []).append(i)

        import random

        rng = random.Random(42)
        keep_indices: list[int] = []
        for intent, indices in intent_indices.items():
            if intent in over_represented:
                rng.shuffle(indices)
                keep_indices.extend(indices[:max_per_intent])
            else:
                keep_indices.extend(indices)

        keep_indices.sort()
        original = len(dataset)
        dataset = dataset.select(keep_indices)
        logger.info("Intent balance: downsampled %d → %d", original, len(dataset))
        return dataset

    def ensure_edge_cases(
        self,
        dataset: datasets.Dataset,
        min_edge_ratio: float = 0.1,
        edge_intents: list[str] | None = None,
    ) -> datasets.Dataset:
        """Verify at least min_edge_ratio of examples are edge cases. Logs warning if not met."""
        if edge_intents is None:
            edge_intents = ["fraud_report", "complaint", "cancellation"]

        if "intent" not in dataset.column_names:
            logger.warning("No 'intent' column, skipping edge case check")
            return dataset

        counts = Counter(dataset["intent"])
        edge_count = sum(counts.get(intent, 0) for intent in edge_intents)
        edge_ratio = edge_count / max(len(dataset), 1)

        if edge_ratio < min_edge_ratio:
            logger.warning(
                "Edge case ratio %.1f%% is below minimum %.1f%% (edge intents: %s, count: %d/%d). "
                "Consider adding more edge case examples.",
                edge_ratio * 100,
                min_edge_ratio * 100,
                edge_intents,
                edge_count,
                len(dataset),
            )
        else:
            logger.info("Edge case ratio: %.1f%% (meets %.1f%% threshold)", edge_ratio * 100, min_edge_ratio * 100)

        return dataset

    def apply_all(self, dataset: datasets.Dataset, config: dict[str, Any] | None = None) -> tuple[datasets.Dataset, dict]:
        """Run all configured filters in sequence.

        Config keys map to filter methods with their kwargs:
        {"dedup_exact": {}, "filter_length": {"min_tokens": 15}, ...}
        """
        if config is None:
            config = {
                "dedup_exact": {},
                "filter_length": {},
                "filter_language": {},
                "filter_confidence": {},
                "balance_intents": {},
                "ensure_edge_cases": {},
            }

        report: dict[str, Any] = {"initial_count": len(dataset), "stages": {}}

        for filter_name, kwargs in config.items():
            method = getattr(self, filter_name, None)
            if method is None:
                logger.warning("Unknown filter '%s', skipping", filter_name)
                continue

            before = len(dataset)
            dataset = method(dataset, **(kwargs or {}))
            after = len(dataset)
            report["stages"][filter_name] = {"before": before, "after": after, "removed": before - after}

        report["final_count"] = len(dataset)
        report["total_removed"] = report["initial_count"] - report["final_count"]
        logger.info(
            "Quality filtering complete: %d → %d (removed %d)",
            report["initial_count"],
            report["final_count"],
            report["total_removed"],
        )
        return dataset, report
