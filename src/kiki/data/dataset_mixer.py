"""Dataset mixing for the Kiki SLM pipeline.

Task 3.7: Combine multiple datasets with configurable weights
into a single training dataset using proportional sampling.
"""

from __future__ import annotations

import logging
from typing import Any

import datasets

from kiki.data.loaders import CSVLoader, HuggingFaceLoader, JSONLLoader

logger = logging.getLogger(__name__)

_LOADER_MAP = {
    "huggingface": HuggingFaceLoader,
    "csv": CSVLoader,
    "jsonl": JSONLLoader,
}


class DatasetMixer:
    """Combine multiple datasets with configurable proportional sampling."""

    def __init__(
        self,
        datasets_config: dict[str, dict[str, Any]],
        total_examples: int = 100_000,
        seed: int = 42,
    ) -> None:
        self.datasets_config = datasets_config
        self.total_examples = total_examples
        self.seed = seed
        self._loaded: dict[str, datasets.Dataset] = {}
        self._composition: dict[str, int] = {}

    def _load_source(self, name: str, config: dict[str, Any]) -> datasets.Dataset:
        """Load a single source dataset."""
        loader_type = config.get("loader", "huggingface")
        loader_cls = _LOADER_MAP.get(loader_type)
        if loader_cls is None:
            raise ValueError(f"Unknown loader type '{loader_type}' for dataset '{name}'")

        # Build loader kwargs from config (exclude meta fields)
        meta_keys = {"loader", "weight", "converter"}
        loader_kwargs = {k: v for k, v in config.items() if k not in meta_keys}

        # Remap common config keys to loader constructor params
        if loader_type == "huggingface":
            if "id" in loader_kwargs:
                loader_kwargs["dataset_id"] = loader_kwargs.pop("id")
        elif loader_type in ("csv", "jsonl"):
            pass  # path is already the right kwarg

        loader = loader_cls(**loader_kwargs)
        return loader.load()

    def mix(self) -> datasets.Dataset:
        """Load all sources and combine with proportional sampling."""
        sampled_parts: list[datasets.Dataset] = []

        for name, config in self.datasets_config.items():
            weight = config.get("weight", 1.0 / len(self.datasets_config))
            target_count = int(self.total_examples * weight)

            if target_count <= 0:
                logger.info("Skipping '%s' (weight=%.3f, target=0)", name, weight)
                continue

            logger.info("Loading '%s' (weight=%.3f, target=%d)", name, weight, target_count)

            if name in self._loaded:
                ds = self._loaded[name]
            else:
                try:
                    ds = self._load_source(name, config)
                    self._loaded[name] = ds
                except Exception as exc:
                    logger.error("Failed to load '%s': %s — skipping", name, exc)
                    continue

            # Sample with replacement if target exceeds available
            available = len(ds)
            if target_count <= available:
                sampled = ds.shuffle(seed=self.seed).select(range(target_count))
            else:
                # Sample with replacement by repeating + truncating
                repeats = (target_count // available) + 1
                indices = list(range(available)) * repeats
                indices = indices[:target_count]
                sampled = ds.select(indices).shuffle(seed=self.seed)
                logger.info("'%s': upsampled %d → %d (with replacement)", name, available, target_count)

            self._composition[name] = len(sampled)
            sampled_parts.append(sampled)

        if not sampled_parts:
            raise ValueError("No datasets were successfully loaded and sampled")

        # Ensure all parts have the same columns
        all_columns = set()
        for part in sampled_parts:
            all_columns.update(part.column_names)

        # Align columns across all parts
        aligned_parts = []
        for part in sampled_parts:
            missing = all_columns - set(part.column_names)
            if missing:
                for col in missing:
                    part = part.add_column(col, [None] * len(part))
            aligned_parts.append(part)

        combined = datasets.concatenate_datasets(aligned_parts)
        combined = combined.shuffle(seed=self.seed)

        logger.info("Mixed dataset: %d total examples from %d sources", len(combined), len(sampled_parts))
        return combined

    def get_composition_report(self) -> dict[str, Any]:
        """Show actual counts per source after mixing."""
        total = sum(self._composition.values())
        return {
            "total_examples": total,
            "sources": {
                name: {
                    "count": count,
                    "actual_ratio": round(count / max(total, 1), 4),
                    "target_ratio": round(self.datasets_config[name].get("weight", 0), 4),
                }
                for name, count in self._composition.items()
            },
        }

    def validate_format_consistency(self) -> list[str]:
        """Check that all loaded datasets have a 'messages' column (i.e., they've been through ChatML conversion).

        Returns a list of error messages (empty if all valid).
        """
        errors = []
        for name, ds in self._loaded.items():
            if "messages" not in ds.column_names:
                errors.append(
                    f"Dataset '{name}' missing 'messages' column (has: {ds.column_names}). "
                    "Run ChatMLConverter.process_dataset() first."
                )
        if errors:
            for e in errors:
                logger.warning(e)
        else:
            logger.info("All %d datasets have consistent 'messages' format", len(self._loaded))
        return errors
