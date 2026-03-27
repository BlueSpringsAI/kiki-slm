"""Config loader for the Loopper organic data pipeline.

All pipeline scripts call get_default_paths() to resolve directory paths.
Reads from configs/loopper_pipeline.yaml if it exists, otherwise uses
project-relative defaults.

Usage in scripts:
    from scripts.loopper.config import get_default_paths
    paths = get_default_paths()
    FILTERED_DIR = paths["filtered_tickets"]
    SAMPLED_DIR  = paths["sampled_tickets"]
"""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "loopper_pipeline.yaml"


def get_default_paths() -> dict[str, str]:
    """Load paths from config YAML, fall back to project-relative defaults."""
    defaults = {
        "filtered_tickets": str(PROJECT_ROOT / "data" / "filtered_tickets"),
        "sampled_tickets": str(PROJECT_ROOT / "data" / "sampled_tickets"),
        "traces": str(PROJECT_ROOT / "data" / "traces"),
        "chatml_output": str(PROJECT_ROOT / "data" / "chatml"),
        "agent_src": None,
    }

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
        paths = cfg.get("paths", {})
        for key in defaults:
            if key in paths and paths[key] is not None:
                defaults[key] = str(paths[key])

    return defaults
