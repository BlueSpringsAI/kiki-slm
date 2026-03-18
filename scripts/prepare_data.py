"""Orchestrate the full data pipeline: load → anonymize → convert → filter → mix.

Usage:
    python scripts/prepare_data.py --config configs/data_pipeline.yaml
    python scripts/prepare_data.py --config configs/data_pipeline.yaml --dry-run
    python scripts/prepare_data.py --config configs/data_pipeline.yaml --skip-pii --skip-filter
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from kiki.data.loaders import CSVLoader, HuggingFaceLoader, JSONLLoader
from kiki.data.pii_anonymizer import PIIAnonymizer
from kiki.data.processors import ChatMLConverter
from kiki.data.quality_filter import QualityFilter
from kiki.data.dataset_mixer import DatasetMixer
from kiki.data.validators import ChatMLExample, validate_dataset
from kiki.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

_LOADER_MAP = {
    "huggingface": HuggingFaceLoader,
    "csv": CSVLoader,
    "jsonl": JSONLLoader,
}


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(
    config: dict[str, Any],
    output_dir: Path,
    skip_pii: bool = False,
    skip_filter: bool = False,
    dry_run: bool = False,
    num_proc: int = 4,
) -> dict[str, Any]:
    """Execute the full data pipeline and return a report."""
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "stages": {},
    }
    t_start = time.monotonic()

    datasets_config = config.get("datasets", {})
    pii_config = config.get("pii", {})
    filter_config = config.get("filters", {})
    mix_config = config.get("mixing", {})

    if dry_run:
        logger.info("=== DRY RUN ===")
        logger.info("Datasets to load: %d", len(datasets_config))
        for name, cfg in datasets_config.items():
            logger.info("  - %s (loader=%s, converter=%s, weight=%.2f)",
                       name, cfg.get("loader"), cfg.get("converter"), cfg.get("weight", 0))
        logger.info("PII anonymization: %s", "enabled" if pii_config.get("enabled") and not skip_pii else "disabled")
        logger.info("Quality filtering: %s", "enabled" if not skip_filter else "disabled")
        logger.info("Total target examples: %d", mix_config.get("total_examples", 100000))
        logger.info("Output directory: %s", output_dir)
        return report

    # ------------------------------------------------------------------
    # Stage 1: Load datasets
    # ------------------------------------------------------------------
    logger.info("=== Stage 1: Loading datasets ===")
    t0 = time.monotonic()
    loaded = {}
    for name, cfg in datasets_config.items():
        loader_type = cfg.get("loader", "huggingface")
        loader_cls = _LOADER_MAP.get(loader_type)
        if loader_cls is None:
            logger.error("Unknown loader '%s' for dataset '%s', skipping", loader_type, name)
            continue

        try:
            loader_kwargs = {k: v for k, v in cfg.items() if k not in ("loader", "weight", "converter")}
            if loader_type == "huggingface" and "id" in loader_kwargs:
                loader_kwargs["dataset_id"] = loader_kwargs.pop("id")
            loader = loader_cls(**loader_kwargs)
            loaded[name] = loader.load()
            logger.info("Loaded '%s': %d examples", name, len(loaded[name]))
        except Exception as exc:
            logger.error("Failed to load '%s': %s", name, exc)

    report["stages"]["load"] = {
        "duration_s": round(time.monotonic() - t0, 2),
        "datasets_loaded": len(loaded),
        "counts": {name: len(ds) for name, ds in loaded.items()},
    }

    # ------------------------------------------------------------------
    # Stage 2: PII anonymization (optional)
    # ------------------------------------------------------------------
    if pii_config.get("enabled") and not skip_pii:
        logger.info("=== Stage 2: PII anonymization ===")
        t0 = time.monotonic()
        anonymizer = PIIAnonymizer()
        text_columns = pii_config.get("columns", ["customer_message", "agent_response"])

        for name, ds in loaded.items():
            cols_to_process = [c for c in text_columns if c in ds.column_names]
            if cols_to_process:
                loaded[name] = anonymizer.process_dataset(ds, cols_to_process, num_proc=num_proc)
                logger.info("Anonymized '%s' columns: %s", name, cols_to_process)

        report["stages"]["pii"] = {"duration_s": round(time.monotonic() - t0, 2)}
    else:
        logger.info("=== Stage 2: PII anonymization (skipped) ===")

    # ------------------------------------------------------------------
    # Stage 3: ChatML conversion
    # ------------------------------------------------------------------
    logger.info("=== Stage 3: ChatML conversion ===")
    t0 = time.monotonic()
    converted = {}
    for name, ds in loaded.items():
        converter_name = datasets_config[name].get("converter")
        if not converter_name:
            logger.warning("No converter specified for '%s', skipping conversion", name)
            converted[name] = ds
            continue

        try:
            converted[name] = ChatMLConverter.process_dataset(ds, converter_name)
            logger.info("Converted '%s' with '%s': %d examples", name, converter_name, len(converted[name]))
        except Exception as exc:
            logger.error("Conversion failed for '%s': %s", name, exc)
            converted[name] = ds

    report["stages"]["convert"] = {
        "duration_s": round(time.monotonic() - t0, 2),
        "counts": {name: len(ds) for name, ds in converted.items()},
    }

    # ------------------------------------------------------------------
    # Stage 4: Quality filtering (optional)
    # ------------------------------------------------------------------
    if not skip_filter and filter_config:
        logger.info("=== Stage 4: Quality filtering ===")
        t0 = time.monotonic()
        qf = QualityFilter()

        for name, ds in converted.items():
            converted[name], stage_report = qf.apply_all(ds, filter_config)
            logger.info("Filtered '%s': %d → %d", name, stage_report["initial_count"], stage_report["final_count"])

        report["stages"]["filter"] = {
            "duration_s": round(time.monotonic() - t0, 2),
            "counts": {name: len(ds) for name, ds in converted.items()},
        }
    else:
        logger.info("=== Stage 4: Quality filtering (skipped) ===")

    # ------------------------------------------------------------------
    # Stage 5: Mix datasets
    # ------------------------------------------------------------------
    logger.info("=== Stage 5: Mixing datasets ===")
    t0 = time.monotonic()

    # Build mixer config from loaded datasets
    mixer_datasets = {}
    for name, ds in converted.items():
        weight = datasets_config[name].get("weight", 1.0 / max(len(converted), 1))
        mixer_datasets[name] = {"weight": weight}

    mixer = DatasetMixer(
        datasets_config=mixer_datasets,
        total_examples=mix_config.get("total_examples", 100_000),
        seed=mix_config.get("seed", 42),
    )
    # Pre-load the already-converted datasets into the mixer
    mixer._loaded = converted

    mixed = mixer.mix()
    composition = mixer.get_composition_report()

    report["stages"]["mix"] = {
        "duration_s": round(time.monotonic() - t0, 2),
        "total_examples": len(mixed),
        "composition": composition,
    }

    # ------------------------------------------------------------------
    # Stage 6: Validate final dataset
    # ------------------------------------------------------------------
    logger.info("=== Stage 6: Validation ===")
    t0 = time.monotonic()

    if "messages" in mixed.column_names:
        validation_report = validate_dataset(mixed, ChatMLExample, sample_size=1000)
        report["stages"]["validate"] = {
            "duration_s": round(time.monotonic() - t0, 2),
            "valid_ratio": validation_report.valid_ratio,
            "total_checked": validation_report.total,
            "valid": validation_report.valid,
            "invalid": validation_report.invalid,
        }
    else:
        logger.warning("No 'messages' column — skipping ChatML validation")
        report["stages"]["validate"] = {"skipped": True}

    # ------------------------------------------------------------------
    # Stage 7: Save output
    # ------------------------------------------------------------------
    logger.info("=== Stage 7: Saving output ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    mixed.save_to_disk(str(output_dir / "train"))
    logger.info("Saved dataset to %s/train (%d examples)", output_dir, len(mixed))

    report["total_duration_s"] = round(time.monotonic() - t_start, 2)

    report_path = output_dir / "pipeline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Pipeline report saved to %s", report_path)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare training data for Kiki SLM")
    parser.add_argument("--config", type=str, required=True, help="Path to pipeline YAML config")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--skip-pii", action="store_true", help="Skip PII anonymization")
    parser.add_argument("--skip-filter", action="store_true", help="Skip quality filtering")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without processing")
    parser.add_argument("--num-proc", type=int, default=4, help="Number of parallel processes")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)

    report = run_pipeline(
        config=config,
        output_dir=output_dir,
        skip_pii=args.skip_pii,
        skip_filter=args.skip_filter,
        dry_run=args.dry_run,
        num_proc=args.num_proc,
    )

    print(f"\nPipeline complete: {report.get('total_duration_s', 0):.1f}s")


if __name__ == "__main__":
    main()
