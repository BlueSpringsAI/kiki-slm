"""Adapter routing with A/B testing support.

Task 3.29: Routes requests to the correct LoRA adapter,
supports dynamic loading/unloading and A/B experiments.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


class AdapterRouter:
    """Routes inference requests to the correct LoRA adapter."""

    DEFAULT_ADAPTERS = {
        "intent": "intent-classifier",
        "workflow": "workflow-reasoner",
        "tools": "tool-caller",
        "response": "response-gen",
    }

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "kiki-internal",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.adapters = dict(self.DEFAULT_ADAPTERS)
        self._experiments: dict[str, dict] = {}

    def route(self, stage: str, user_id: str | None = None) -> str:
        """Return adapter name for the given stage.

        If an A/B test is active for this stage, use user_id hash to select variant.
        """
        if stage in self._experiments and user_id:
            return self._ab_route(stage, user_id)
        return self.adapters.get(stage, self.adapters.get("response", "response-gen"))

    def load_adapter(self, name: str, path: str) -> bool:
        """Load a new LoRA adapter via vLLM's endpoint."""
        try:
            resp = requests.post(
                f"{self.base_url}/v1/load_lora_adapter",
                json={"lora_name": name, "lora_path": path},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30,
            )
            resp.raise_for_status()
            self.adapters[name] = name
            logger.info("Loaded adapter '%s' from '%s'", name, path)
            return True
        except Exception as exc:
            logger.error("Failed to load adapter '%s': %s", name, exc)
            return False

    def unload_adapter(self, name: str) -> bool:
        """Unload an adapter via vLLM endpoint."""
        try:
            resp = requests.post(
                f"{self.base_url}/v1/unload_lora_adapter",
                json={"lora_name": name},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30,
            )
            resp.raise_for_status()
            self.adapters.pop(name, None)
            logger.info("Unloaded adapter '%s'", name)
            return True
        except Exception as exc:
            logger.error("Failed to unload adapter '%s': %s", name, exc)
            return False

    def list_adapters(self) -> list[str]:
        """List currently registered adapters."""
        return list(self.adapters.values())

    def register_experiment(
        self,
        stage: str,
        variants: dict[str, str],
        traffic_split: dict[str, float],
    ) -> None:
        """Register an A/B test for a pipeline stage.

        Args:
            stage: Pipeline stage (e.g. "response").
            variants: Mapping of variant_id -> adapter_name.
            traffic_split: Mapping of variant_id -> fraction (must sum to 1.0).
        """
        total = sum(traffic_split.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Traffic split must sum to 1.0, got {total}")

        self._experiments[stage] = {"variants": variants, "split": traffic_split}
        logger.info("Registered A/B experiment for stage '%s': %s", stage, list(variants.keys()))

    def _ab_route(self, stage: str, user_id: str) -> str:
        """Deterministic A/B assignment via user_id hash."""
        experiment = self._experiments[stage]
        bucket = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100

        cumulative = 0.0
        for variant_id, fraction in experiment["split"].items():
            cumulative += fraction * 100
            if bucket < cumulative:
                adapter = experiment["variants"].get(variant_id, self.adapters.get(stage, "response-gen"))
                return adapter

        # Fallback to first variant
        first_variant = next(iter(experiment["variants"].values()))
        return first_variant
