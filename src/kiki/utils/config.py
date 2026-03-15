"""YAML config loading with OmegaConf.

Task 3.33: Merge base.yaml with task-specific configs,
environment variable interpolation, and validation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> DictConfig:
    """Load a YAML config, resolve inheritance, and apply overrides.

    If the config contains an ``inherits`` key, the referenced base config
    is loaded first and the current config is merged on top.

    Args:
        path: Path to the YAML config file.
        overrides: Optional dict of dotlist overrides (e.g. ``{"training.lr": 1e-5}``).

    Returns:
        Resolved ``DictConfig``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    cfg = OmegaConf.load(path)

    # Handle inheritance
    inherits = OmegaConf.select(cfg, "inherits", default=None)
    if inherits:
        base_path = path.parent / inherits
        if not base_path.exists():
            # Try configs/ directory
            base_path = path.parent.parent / inherits
        if base_path.exists():
            base_cfg = OmegaConf.load(base_path)
            cfg = OmegaConf.merge(base_cfg, cfg)
            logger.info("Merged base config from '%s'", base_path)
        else:
            logger.warning("Base config '%s' not found, skipping inheritance", inherits)

    # Apply overrides
    if overrides:
        override_cfg = OmegaConf.create(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # Resolve interpolations and env vars
    OmegaConf.resolve(cfg)

    logger.info("Loaded config from '%s'", path)
    return cfg


def validate_config(config: DictConfig, required_keys: list[str]) -> list[str]:
    """Validate that all required keys are present in the config.

    Returns:
        List of missing keys (empty if valid).
    """
    missing = []
    for key in required_keys:
        if OmegaConf.select(config, key, default=None) is None:
            missing.append(key)
    if missing:
        logger.warning("Config missing required keys: %s", missing)
    return missing


def save_config(config: DictConfig, path: str | Path) -> None:
    """Save a resolved config to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, path)
    logger.info("Saved config to '%s'", path)


def config_to_dict(config: DictConfig) -> dict:
    """Convert DictConfig to a plain Python dict."""
    return OmegaConf.to_container(config, resolve=True)
