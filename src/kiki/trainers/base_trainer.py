"""Abstract base trainer with shared infrastructure.

Task 3.12: Config loading, experiment tracking, GPU monitoring,
checkpoint management, and post-training evaluation trigger.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from kiki.utils.callbacks import KikiMetricsCallback
from kiki.utils.config import config_to_dict, load_config
from kiki.utils.experiment_tracker import ExperimentTracker
from kiki.utils.gpu_utils import clear_gpu_cache, get_gpu_memory

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract base for all Kiki SLM trainers."""

    def __init__(self, config_path: str, overrides: dict[str, Any] | None = None) -> None:
        self.config = load_config(config_path, overrides)
        self.tracker = ExperimentTracker(self.config)
        self.model = None
        self.tokenizer = None
        self.trainer = None  # TRL trainer instance
        self._start_time: float | None = None

    @abstractmethod
    def setup_model(self) -> None:
        """Load model and apply PEFT config."""

    @abstractmethod
    def setup_trainer(self, train_dataset: Any, eval_dataset: Any = None) -> None:
        """Initialize the TRL trainer with datasets."""

    @abstractmethod
    def train(self) -> dict:
        """Run the training loop. Returns training metrics."""

    def run(self, train_dataset: Any, eval_dataset: Any = None) -> dict:
        """Full training pipeline: setup -> train -> save -> cleanup."""
        self._log_gpu_stats("before_setup")

        task_name = getattr(self.config, "task", "training")
        self.tracker.init_run(name=task_name, config=config_to_dict(self.config))

        self.setup_model()
        self._log_gpu_stats("after_model_load")

        self.setup_trainer(train_dataset, eval_dataset)
        self._inject_callbacks()

        self._start_time = time.time()
        result = self.train()
        self._log_training_summary(result)

        output_dir = getattr(self.config, "output_dir", "runs/default")
        self.save(output_dir)

        self.cleanup()
        return result

    def evaluate(self) -> dict | None:
        """Run evaluation if trainer supports it."""
        if self.trainer is None:
            logger.warning("No trainer initialized for evaluation")
            return None
        try:
            metrics = self.trainer.evaluate()
            self.tracker.log_metrics(metrics)
            return metrics
        except Exception as exc:
            logger.error("Evaluation failed: %s", exc)
            return None

    def save(self, output_dir: str | None = None) -> None:
        """Save adapter and tokenizer."""
        output_dir = output_dir or getattr(self.config, "output_dir", "runs/default")
        if self.model is not None:
            logger.info("Saving model to '%s'", output_dir)
            self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        self.tracker.log_artifact(output_dir, name="model", artifact_type="model")

    def cleanup(self) -> None:
        """Clear GPU cache and finish tracking."""
        clear_gpu_cache()
        self.tracker.finish()
        logger.info("Training cleanup complete")

    def _inject_callbacks(self) -> None:
        """Add KikiMetricsCallback to the TRL trainer."""
        if self.trainer is not None:
            callback = KikiMetricsCallback(tracker=self.tracker)
            self.trainer.add_callback(callback)
            logger.info("Injected KikiMetricsCallback into trainer")

    def _log_gpu_stats(self, stage: str = "") -> None:
        """Log current GPU memory usage."""
        mem = get_gpu_memory()
        if mem:
            logger.info(
                "GPU [%s]: %.1fGB used / %.1fGB total (%.1fGB free)",
                stage, mem["allocated_gb"], mem["total_gb"], mem["free_gb"],
            )
            self.tracker.log_metrics({f"gpu/{stage}_allocated_gb": mem["allocated_gb"]})

    def _log_training_summary(self, result: dict) -> None:
        """Log training metrics summary."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        logger.info("Training completed in %.1f minutes", elapsed / 60)

        summary = {"training/duration_minutes": elapsed / 60}
        if isinstance(result, dict):
            for key, val in result.items():
                if isinstance(val, (int, float)):
                    summary[f"training/{key}"] = val

        self.tracker.log_metrics(summary)
        self._log_gpu_stats("after_training")
