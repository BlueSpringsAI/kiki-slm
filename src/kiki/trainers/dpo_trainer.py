"""DPO and SimPO trainer wrapping TRL DPOTrainer / CPOTrainer.

Task 3.14: CRITICAL — learning rate must be 5e-6 (NOT 2e-4).
Monitor rewards/margins. Supports DPO (sigmoid) and SimPO (reference-free).
"""

from __future__ import annotations

import logging
from typing import Any

from kiki.models.model_loader import ModelLoader
from kiki.models.peft_config import PEFTConfigFactory
from kiki.trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class KikiDPOTrainer(BaseTrainer):
    """DPO/SimPO alignment trainer for Kiki SLM."""

    def setup_model(self) -> None:
        """Load SFT checkpoint and apply DPO-specific PEFT config."""
        # Load base model or SFT checkpoint
        base = getattr(self.config, "base_model_or_adapter", None)
        if base:
            self.model, self.tokenizer = ModelLoader.load_for_inference(base, quantize=True)
        else:
            self.model, self.tokenizer = ModelLoader.load_for_training(self.config.model)

        # Apply DPO-specific PEFT
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

        logger.info("DPO model loaded with PEFT (r=%d)", lora_config.r)

    def setup_trainer(self, train_dataset: Any, eval_dataset: Any = None) -> None:
        """Create DPOTrainer or CPOTrainer depending on loss_type."""
        alignment = self.config.alignment if hasattr(self.config, "alignment") else self.config
        loss_type = getattr(alignment, "loss_type", "sigmoid")
        method = getattr(alignment, "method", "dpo")

        # CRITICAL: DPO lr must be much smaller than SFT
        lr = float(getattr(alignment, "learning_rate", 5e-6))
        if lr > 1e-4:
            logger.warning("DPO learning rate %.1e is suspiciously high! Should be ~5e-6.", lr)

        if method == "simpo" or loss_type == "simpo":
            self._setup_cpo_trainer(train_dataset, eval_dataset, alignment, lr)
        else:
            self._setup_dpo_trainer(train_dataset, eval_dataset, alignment, lr, loss_type)

    def _setup_dpo_trainer(self, train_dataset, eval_dataset, alignment, lr, loss_type) -> None:
        from trl import DPOConfig, DPOTrainer

        defaults = self.config.defaults if hasattr(self.config, "defaults") else {}

        def _get(key, default=None):
            val = getattr(alignment, key, None)
            if val is None and defaults:
                val = getattr(defaults, key, None) if hasattr(defaults, key) else None
            return val if val is not None else default

        dpo_config = DPOConfig(
            output_dir=getattr(self.config, "output_dir", "runs/dpo-default"),
            beta=float(_get("beta", 0.1)),
            loss_type=loss_type,
            learning_rate=lr,
            per_device_train_batch_size=int(_get("per_device_train_batch_size", 1)),
            gradient_accumulation_steps=int(_get("gradient_accumulation_steps", 8)),
            num_train_epochs=int(_get("num_train_epochs", 3)),
            max_length=int(_get("max_length", 1536)),
            max_prompt_length=int(_get("max_prompt_length", 768)),
            bf16=_get("bf16", True),
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=int(_get("logging_steps", 10)),
            report_to=list(_get("report_to", ["none"])),
            seed=int(_get("seed", 42)),
        )

        self.trainer = DPOTrainer(
            model=self.model,
            args=dpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )
        logger.info("DPOTrainer initialized (beta=%.2f, lr=%s, loss=%s)", dpo_config.beta, lr, loss_type)

    def _setup_cpo_trainer(self, train_dataset, eval_dataset, alignment, lr) -> None:
        from trl import CPOConfig, CPOTrainer

        cpo_config = CPOConfig(
            output_dir=getattr(self.config, "output_dir", "runs/simpo-default"),
            learning_rate=lr,
            per_device_train_batch_size=int(getattr(alignment, "per_device_train_batch_size", 1)),
            gradient_accumulation_steps=int(getattr(alignment, "gradient_accumulation_steps", 8)),
            num_train_epochs=int(getattr(alignment, "num_train_epochs", 3)),
            max_length=int(getattr(alignment, "max_length", 1536)),
            max_prompt_length=int(getattr(alignment, "max_prompt_length", 768)),
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=10,
            loss_type="simpo",
        )

        self.trainer = CPOTrainer(
            model=self.model,
            args=cpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )
        logger.info("CPOTrainer (SimPO) initialized (lr=%s)", lr)

    def train(self) -> dict:
        """Train and monitor rewards/margins metric."""
        result = self.trainer.train()
        metrics = {k: v for k, v in result.metrics.items() if isinstance(v, (int, float))}

        # DPO diagnostic: check rewards/margins
        margins = metrics.get("rewards/margins", metrics.get("train_rewards/margins"))
        if margins is not None and margins < 0.1:
            logger.warning(
                "rewards/margins=%.3f is low. Consider increasing beta or decreasing lr.", margins
            )

        return metrics
