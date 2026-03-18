"""Supervised Fine-Tuning trainer wrapping TRL SFTTrainer.

Task 3.13: QLoRA SFT with packing, Unsloth gradient checkpointing,
and proper Qwen3 chat template handling.
"""

from __future__ import annotations

import logging
from typing import Any

from kiki.models.model_loader import ModelLoader
from kiki.models.peft_config import PEFTConfigFactory
from kiki.trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class KikiSFTTrainer(BaseTrainer):
    """Supervised fine-tuning for Kiki SLM adapters."""

    def setup_model(self) -> None:
        """Load model with Unsloth and apply task-specific PEFT config."""
        self.model, self.tokenizer = ModelLoader.load_for_training(self.config.model)

        # Apply PEFT via Unsloth if available, else standard PEFT
        lora_config = PEFTConfigFactory.from_config(self.config)

        try:
            from unsloth import FastLanguageModel

            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                target_modules=list(lora_config.target_modules) if lora_config.target_modules != "all-linear" else lora_config.target_modules,
                lora_dropout=lora_config.lora_dropout,
                bias=lora_config.bias,
                use_gradient_checkpointing="unsloth",
                random_state=getattr(self.config, "seed", getattr(self.config.defaults, "seed", 42)) if hasattr(self.config, "defaults") else 42,
            )
            logger.info("Applied Unsloth PEFT config (r=%d)", lora_config.r)
        except (ImportError, AttributeError):
            from peft import get_peft_model

            self.model = get_peft_model(self.model, lora_config)
            logger.info("Applied standard PEFT config (r=%d)", lora_config.r)

    def setup_trainer(self, train_dataset: Any, eval_dataset: Any = None) -> None:
        """Create TRL SFTTrainer with SFTConfig."""
        from trl import SFTConfig, SFTTrainer

        training_cfg = self.config.training if hasattr(self.config, "training") else self.config
        defaults = self.config.defaults if hasattr(self.config, "defaults") else {}

        def _get(key, default=None):
            val = getattr(training_cfg, key, None)
            if val is None and defaults:
                val = getattr(defaults, key, None) if hasattr(defaults, key) else defaults.get(key)
            return val if val is not None else default

        sft_config = SFTConfig(
            output_dir=getattr(self.config, "output_dir", "runs/sft-default"),
            per_device_train_batch_size=_get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=_get("gradient_accumulation_steps", 8),
            num_train_epochs=_get("num_train_epochs", 3),
            learning_rate=float(_get("learning_rate", 2e-4)),
            lr_scheduler_type=_get("lr_scheduler_type", "cosine"),
            warmup_ratio=float(_get("warmup_ratio", 0.03)),
            bf16=_get("bf16", True),
            optim=_get("optim", "adamw_8bit"),
            max_seq_length=int(_get("max_seq_length", 2048)),
            packing=_get("packing", True),
            gradient_checkpointing=_get("gradient_checkpointing", True),
            gradient_checkpointing_kwargs={"use_reentrant": False},
            save_strategy=_get("save_strategy", "steps"),
            save_steps=int(_get("save_steps", 500)),
            eval_strategy=_get("eval_strategy", "steps") if eval_dataset else "no",
            eval_steps=int(_get("eval_steps", 500)) if eval_dataset else None,
            logging_steps=int(_get("logging_steps", 10)),
            # Metrics flow through KikiMetricsCallback → ExperimentTracker, not TRL
            report_to=["none"],
            seed=int(_get("seed", 42)),
        )

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=sft_config,
        )
        logger.info("SFTTrainer initialized (packing=%s, lr=%s)", sft_config.packing, sft_config.learning_rate)

    def train(self) -> dict:
        """Run SFT training and return metrics."""
        result = self.trainer.train()
        metrics = {k: v for k, v in result.metrics.items() if isinstance(v, (int, float))}
        logger.info("SFT training complete: loss=%.4f", metrics.get("train_loss", 0))
        return metrics
