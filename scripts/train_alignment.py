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

from kiki.utils.logging import setup_logging

setup_logging()
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

    # Load data — supports multiple datasets with proportional weighting
    import datasets as ds_lib
    from datasets import load_dataset

    if args.data_path:
        train_dataset = load_dataset("json", data_files=args.data_path, split="train")
    elif hasattr(config, "datasets"):
        parts: list[tuple] = []
        for data_cfg in config.datasets:
            path = getattr(data_cfg, "path", None) or getattr(data_cfg, "id", "")
            weight = float(getattr(data_cfg, "weight", 1.0))
            if path.endswith(".jsonl"):
                ds = load_dataset("json", data_files=path, split="train")
            else:
                ds = load_dataset(path, split="train")
            parts.append((ds, weight))
            logger.info("Loaded '%s': %d examples (weight=%.2f)", path, len(ds), weight)

        if len(parts) == 1:
            train_dataset = parts[0][0]
        else:
            # Sample each dataset proportionally by weight, then concatenate
            total_weight = sum(w for _, w in parts)
            total_available = sum(len(d) for d, _ in parts)
            sampled = []
            for ds, weight in parts:
                ratio = weight / total_weight
                n = min(len(ds), max(1, int(total_available * ratio)))
                sampled.append(ds.shuffle(seed=42).select(range(n)))
            train_dataset = ds_lib.concatenate_datasets(sampled).shuffle(seed=42)
            logger.info("Combined %d datasets: %d total examples", len(parts), len(train_dataset))
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
        from kiki.utils.reward_tracker import TrackedReward

        # Build reward functions
        reward_weights = None
        if hasattr(config, "rewards"):
            reward_weights = {k: v.weight for k, v in config.rewards.items()}

        composite = CompositeReward(weights=reward_weights)

        from kiki.trainers.grpo_trainer import KikiGRPOTrainer

        grpo_trainer = KikiGRPOTrainer(args.config, reward_functions=[composite], overrides=overrides or None)

        # Wrap composite in TrackedReward after trainer init so tracker is available
        tracked = TrackedReward(composite, tracker=grpo_trainer.tracker)
        grpo_trainer.reward_functions = [tracked]

        grpo_trainer.run(train_dataset)

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
