#!/usr/bin/env python3
"""Entry point for alignment training (DPO/SimPO/GRPO/KTO).

Usage:
    python scripts/train_alignment.py --config configs/alignment/dpo.yaml
    python scripts/train_alignment.py --config configs/alignment/grpo.yaml
    python scripts/train_alignment.py --config configs/alignment/kto.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiki SLM Alignment Training")
    parser.add_argument("--config", required=True, help="Path to alignment config YAML")
    parser.add_argument("--method", choices=["dpo", "simpo", "grpo", "kto"], help="Override alignment method")
    parser.add_argument("--data-path", help="Override training data path")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without training")
    args = parser.parse_args()

    from kiki.utils.config import load_config

    overrides = {}
    if args.method:
        overrides["alignment.method"] = args.method

    config = load_config(args.config, overrides or None)

    if args.dry_run:
        from omegaconf import OmegaConf

        print(OmegaConf.to_yaml(config))
        logger.info("Dry run — config validated successfully")
        return

    method = getattr(config.alignment, "method", "dpo") if hasattr(config, "alignment") else "dpo"
    logger.info("Alignment method: %s", method)

    # Load data
    from datasets import load_dataset

    if args.data_path:
        train_dataset = load_dataset("json", data_files=args.data_path, split="train")
    elif hasattr(config, "datasets"):
        data_cfg = config.datasets[0]
        path = getattr(data_cfg, "path", None) or getattr(data_cfg, "id", "")
        if path.endswith(".jsonl"):
            train_dataset = load_dataset("json", data_files=path, split="train")
        else:
            train_dataset = load_dataset(path, split="train")
    else:
        logger.error("No data source specified")
        sys.exit(1)

    # Select trainer
    if method in ("dpo", "simpo"):
        from kiki.trainers.dpo_trainer import KikiDPOTrainer

        trainer = KikiDPOTrainer(args.config, overrides or None)
        trainer.run(train_dataset)

    elif method == "grpo":
        from kiki.rewards import CompositeReward

        # Build reward functions
        reward_weights = None
        if hasattr(config, "rewards"):
            reward_weights = {k: v.weight for k, v in config.rewards.items()}

        composite = CompositeReward(weights=reward_weights)
        reward_fns = [composite]

        from kiki.trainers.grpo_trainer import KikiGRPOTrainer

        trainer = KikiGRPOTrainer(args.config, reward_functions=reward_fns, overrides=overrides or None)
        trainer.run(train_dataset)

    elif method == "kto":
        from kiki.trainers.kto_trainer import KikiKTOTrainer

        trainer = KikiKTOTrainer(args.config, overrides or None)
        trainer.run(train_dataset)

    else:
        logger.error("Unknown method: %s", method)
        sys.exit(1)

    logger.info("Alignment training (%s) complete!", method)


if __name__ == "__main__":
    main()
