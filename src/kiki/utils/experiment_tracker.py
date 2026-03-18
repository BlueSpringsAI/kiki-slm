"""Thin wrapper around W&B and MLflow for experiment tracking.

Task 3.34: Auto-log configs, metrics, and artifacts. Supports
``report_to: ["wandb"]`` or ``report_to: ["mlflow"]`` in configs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Unified experiment tracking across W&B and MLflow."""

    def __init__(self, config: Any = None) -> None:
        self._wandb_run = None
        self._mlflow_active = False

        if config is not None:
            # Prefer experiment_tracking key; fall back to report_to for compat
            backends = getattr(config, "experiment_tracking", None)
            if backends is None:
                backends = getattr(config, "report_to", None)
            if backends is None and hasattr(config, "defaults"):
                backends = getattr(config.defaults, "report_to", None)
            self._backends: list[str] = list(backends) if backends else []
        else:
            self._backends = []

    def init_run(
        self,
        name: str,
        config: dict | None = None,
        tags: list[str] | None = None,
        project: str = "kiki-slm",
    ) -> None:
        """Initialize tracking run(s)."""
        if "wandb" in self._backends:
            self._init_wandb(name, config, tags, project)
        if "mlflow" in self._backends:
            self._init_mlflow(name, config, tags)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to all active backends."""
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)
        if self._mlflow_active:
            try:
                import mlflow

                mlflow.log_metrics(metrics, step=step)
            except Exception as exc:
                logger.debug("MLflow log_metrics failed: %s", exc)

    def log_config(self, config: dict) -> None:
        """Log full config dict."""
        if self._wandb_run is not None:
            self._wandb_run.config.update(config)
        if self._mlflow_active:
            try:
                import mlflow

                mlflow.log_params({k: str(v)[:250] for k, v in _flatten_dict(config).items()})
            except Exception as exc:
                logger.debug("MLflow log_params failed: %s", exc)

    def log_artifact(self, path: str | Path, name: str | None = None, artifact_type: str = "model") -> None:
        """Log a file or directory as an artifact."""
        path = str(path)
        if self._wandb_run is not None:
            try:
                import wandb

                artifact = wandb.Artifact(name or Path(path).name, type=artifact_type)
                if Path(path).is_dir():
                    artifact.add_dir(path)
                else:
                    artifact.add_file(path)
                self._wandb_run.log_artifact(artifact)
            except Exception as exc:
                logger.debug("W&B artifact logging failed: %s", exc)

        if self._mlflow_active:
            try:
                import mlflow

                if Path(path).is_dir():
                    mlflow.log_artifacts(path, artifact_path=name)
                else:
                    mlflow.log_artifact(path)
            except Exception as exc:
                logger.debug("MLflow artifact logging failed: %s", exc)

    def finish(self) -> None:
        """Finish all active tracking runs."""
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None
        if self._mlflow_active:
            try:
                import mlflow

                mlflow.end_run()
            except Exception:
                pass
            self._mlflow_active = False

    # ------------------------------------------------------------------
    # Backend initialization
    # ------------------------------------------------------------------

    def _init_wandb(self, name: str, config: dict | None, tags: list[str] | None, project: str) -> None:
        try:
            import wandb

            self._wandb_run = wandb.init(project=project, name=name, config=config, tags=tags, reinit=True)
            logger.info("W&B run initialized: %s", name)
        except ImportError:
            logger.warning("wandb not installed — skipping W&B tracking")
        except Exception as exc:
            logger.warning("W&B init failed: %s", exc)

    def _init_mlflow(self, name: str, config: dict | None, tags: list[str] | None) -> None:
        try:
            import mlflow

            mlflow.set_experiment("kiki-slm")
            mlflow.start_run(run_name=name, tags={t: "true" for t in (tags or [])})
            if config:
                mlflow.log_params({k: str(v)[:250] for k, v in _flatten_dict(config).items()})
            self._mlflow_active = True
            logger.info("MLflow run initialized: %s", name)
        except ImportError:
            logger.warning("mlflow not installed — skipping MLflow tracking")
        except Exception as exc:
            logger.warning("MLflow init failed: %s", exc)


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dict for MLflow params."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
