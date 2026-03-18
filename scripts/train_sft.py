#!/usr/bin/env python3
"""Entry point for SFT fine-tuning.

Usage:
    python scripts/train_sft.py --config configs/sft/intent_classifier.yaml
    python scripts/train_sft.py --config configs/sft/tool_caller.yaml --override training.num_train_epochs=5
"""

from __future__ import annotations

import argparse
import logging
import sys

from kiki.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiki SLM SFT Training")
    parser.add_argument("--config", required=True, help="Path to SFT config YAML")
    parser.add_argument("--override", nargs="*", help="Config overrides in key=value format")
    parser.add_argument("--data-path", help="Override training data path")
    parser.add_argument("--eval-path", help="Evaluation data path")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without training")
    args = parser.parse_args()

    # Parse overrides
    overrides = {}
    if args.override:
        for ov in args.override:
            key, _, val = ov.partition("=")
            overrides[key] = val

    from kiki.utils.config import load_config

    config = load_config(args.config, overrides or None)

    if args.dry_run:
        from omegaconf import OmegaConf

        print(OmegaConf.to_yaml(config))
        logger.info("Dry run — config validated successfully")
        return

    # Load datasets
    from datasets import load_dataset

    logger.info("Loading training data...")
    if args.data_path:
        if args.data_path.endswith(".jsonl"):
            train_dataset = load_dataset("json", data_files=args.data_path, split="train")
        else:
            train_dataset = load_dataset(args.data_path, split="train")
    elif hasattr(config, "datasets"):
        # Load and mix datasets according to config weights
        from kiki.data.dataset_mixer import DatasetMixer

        datasets_cfg = {d.id: {"loader": "huggingface", "id": d.id, "weight": d.weight} for d in config.datasets}
        mixer = DatasetMixer(datasets_config=datasets_cfg, total_examples=50000, seed=42)
        train_dataset = mixer.mix()
    else:
        logger.error("No data source specified. Use --data-path or configure datasets in YAML.")
        sys.exit(1)

    eval_dataset = None
    if args.eval_path:
        eval_dataset = load_dataset("json", data_files=args.eval_path, split="train")

    # Create trainer and run
    from kiki.trainers.sft_trainer import KikiSFTTrainer

    trainer = KikiSFTTrainer(args.config, overrides or None)
    trainer.run(train_dataset, eval_dataset)

    logger.info("SFT training complete! Model saved to: %s", getattr(config, "output_dir", "runs/"))


if __name__ == "__main__":
    main()
