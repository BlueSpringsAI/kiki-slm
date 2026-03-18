"""TRL TrainerCallback that bridges per-step metrics to ExperimentTracker.

Works for all 4 trainer types (SFT, DPO, GRPO, KTO). Captures loss, lr,
grad_norm, DPO margins, GRPO reward/kl, and samples GPU memory periodically.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from kiki.utils.experiment_tracker import ExperimentTracker
from kiki.utils.gpu_utils import get_gpu_memory

logger = logging.getLogger(__name__)


class KikiMetricsCallback(TrainerCallback):
    """Forward TRL per-step metrics to ExperimentTracker with alerts."""

    def __init__(
        self,
        tracker: ExperimentTracker,
        gpu_sample_interval: int = 50,
    ) -> None:
        self.tracker = tracker
        self.gpu_sample_interval = gpu_sample_interval
        self._total_steps: int | None = None
        self._train_start: float | None = None

    # ------------------------------------------------------------------
    # on_train_begin
    # ------------------------------------------------------------------

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._total_steps = state.max_steps
        self._train_start = time.time()
        logger.info(
            "Training started: %d steps, %s epochs, lr=%s",
            state.max_steps,
            args.num_train_epochs,
            args.learning_rate,
        )

    # ------------------------------------------------------------------
    # on_log — main metrics capture
    # ------------------------------------------------------------------

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if logs is None:
            return

        step = state.global_step
        metrics: dict[str, float] = {}

        # Core metrics
        for key in ("loss", "learning_rate", "grad_norm", "epoch"):
            if key in logs and isinstance(logs[key], (int, float)):
                metrics[key] = float(logs[key])

        # DPO-specific
        for key in ("rewards/chosen", "rewards/rejected", "rewards/margins"):
            if key in logs and isinstance(logs[key], (int, float)):
                metrics[key] = float(logs[key])

        # GRPO-specific
        for key in ("reward", "kl", "clip_ratio"):
            if key in logs and isinstance(logs[key], (int, float)):
                metrics[key] = float(logs[key])

        # Forward to ExperimentTracker
        if metrics:
            self.tracker.log_metrics(metrics, step=step)

        # Human-readable step summary
        total = self._total_steps or "?"
        loss_str = f"loss={metrics['loss']:.4f}" if "loss" in metrics else ""
        lr_str = f"lr={metrics['learning_rate']:.2e}" if "learning_rate" in metrics else ""
        parts = [p for p in (loss_str, lr_str) if p]
        logger.info("Step %d/%s  %s", step, total, "  ".join(parts))

        # --- Alerts ---

        # DPO margin collapse
        margins = metrics.get("rewards/margins")
        if margins is not None and margins < 0.1:
            logger.warning(
                "DPO rewards/margins=%.3f is dangerously low — consider increasing beta or decreasing lr",
                margins,
            )

        # Loss spike
        loss = metrics.get("loss")
        if loss is not None and loss > 10.0:
            logger.warning("Loss spike detected: %.4f at step %d", loss, step)

        # Periodic GPU sampling
        if step > 0 and step % self.gpu_sample_interval == 0:
            mem = get_gpu_memory()
            if mem:
                gpu_metrics = {
                    "gpu/allocated_gb": mem["allocated_gb"],
                    "gpu/free_gb": mem["free_gb"],
                }
                self.tracker.log_metrics(gpu_metrics, step=step)

    # ------------------------------------------------------------------
    # on_evaluate
    # ------------------------------------------------------------------

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if metrics:
            eval_metrics = {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            }
            if eval_metrics:
                self.tracker.log_metrics(eval_metrics, step=state.global_step)
                logger.info("Eval at step %d: %s", state.global_step, eval_metrics)

    # ------------------------------------------------------------------
    # on_train_end
    # ------------------------------------------------------------------

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        elapsed = time.time() - self._train_start if self._train_start else 0
        logger.info("Training finished: %d steps in %.1f minutes", state.global_step, elapsed / 60)
