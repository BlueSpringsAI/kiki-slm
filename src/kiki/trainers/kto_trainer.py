"""KTO trainer wrapping TRL KTOTrainer.

Task 3.16: Binary label (good/bad) alignment with asymmetric loss.
Conservative lr (1e-6), single epoch, for final safety/compliance tuning.
"""

from __future__ import annotations

import logging
from typing import Any

from kiki.models.model_loader import ModelLoader
from kiki.models.peft_config import PEFTConfigFactory
from kiki.trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class KikiKTOTrainer(BaseTrainer):
    """KTO safety/compliance fine-tuning trainer."""

    def setup_model(self) -> None:
        """Load post-GRPO model with conservative PEFT config."""
        base = getattr(self.config, "base_model_or_adapter", None)
        if base:
            self.model, self.tokenizer = ModelLoader.load_for_inference(base, quantize=True)
        else:
            self.model, self.tokenizer = ModelLoader.load_for_training(self.config.model)

        lora_config = PEFTConfigFactory.from_config(self.config)

        try:
            from unsloth import FastLanguageModel

            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                target_modules=list(lora_config.target_modules) if hasattr(lora_config.target_modules, '__iter__') and not isinstance(lora_config.target_modules, str) else lora_config.target_modules,
                lora_dropout=lora_config.lora_dropout,
                bias=lora_config.bias,
                use_gradient_checkpointing="unsloth",
            )
        except ImportError:
            from peft import get_peft_model

            self.model = get_peft_model(self.model, lora_config)

        logger.info("KTO model loaded with PEFT (r=%d)", lora_config.r)

    def setup_trainer(self, train_dataset: Any, eval_dataset: Any = None) -> None:
        """Create KTOTrainer. Binary labels, asymmetric loss, conservative lr."""
        from trl import KTOConfig, KTOTrainer

        alignment = self.config.alignment if hasattr(self.config, "alignment") else self.config

        lr = float(getattr(alignment, "learning_rate", 1e-6))
        if lr > 1e-5:
            logger.warning("KTO learning rate %.1e is high — recommended: 1e-6", lr)

        kto_config = KTOConfig(
            output_dir=getattr(self.config, "output_dir", "runs/kto-default"),
            learning_rate=lr,
            per_device_train_batch_size=int(getattr(alignment, "per_device_train_batch_size", 2)),
            gradient_accumulation_steps=int(getattr(alignment, "gradient_accumulation_steps", 8)),
            num_train_epochs=int(getattr(alignment, "num_train_epochs", 1)),
            max_length=int(getattr(alignment, "max_length", 1536)),
            max_prompt_length=int(getattr(alignment, "max_prompt_length", 768)),
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=10,
            report_to=list(getattr(alignment, "report_to", ["none"])),
            seed=int(getattr(alignment, "seed", 42)),
        )

        self.trainer = KTOTrainer(
            model=self.model,
            args=kto_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )
        logger.info("KTOTrainer initialized (lr=%s, epochs=%d)", lr, kto_config.num_train_epochs)

    def train(self) -> dict:
        """Run KTO training for safety/compliance alignment."""
        result = self.trainer.train()
        metrics = {k: v for k, v in result.metrics.items() if isinstance(v, (int, float))}
        logger.info("KTO training complete")
        return metrics
