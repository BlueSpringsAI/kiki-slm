"""GRPO trainer wrapping TRL GRPOTrainer.

Task 3.15: Group Relative Policy Optimization with vLLM generation
and composite reward functions from kiki.rewards.
"""

from __future__ import annotations

import logging
from typing import Any

from kiki.models.model_loader import ModelLoader
from kiki.models.peft_config import PEFTConfigFactory
from kiki.trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class KikiGRPOTrainer(BaseTrainer):
    """GRPO alignment trainer with verifiable reward functions."""

    def __init__(
        self,
        config_path: str,
        reward_functions: list | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config_path, overrides)
        self.reward_functions = reward_functions or []

    def setup_model(self) -> None:
        """Load post-DPO model for GRPO training."""
        base = getattr(self.config, "base_model_or_adapter", None)
        if base:
            self.model, self.tokenizer = ModelLoader.load_for_inference(base, quantize=True)
        else:
            self.model, self.tokenizer = ModelLoader.load_for_training(self.config.model)

        # GRPO typically uses the same PEFT config as DPO
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

        logger.info("GRPO model loaded with PEFT (r=%d)", lora_config.r)

    def setup_trainer(self, train_dataset: Any, eval_dataset: Any = None) -> None:
        """Create GRPOTrainer with reward functions."""
        from trl import GRPOConfig, GRPOTrainer

        alignment = self.config.alignment if hasattr(self.config, "alignment") else self.config

        grpo_config = GRPOConfig(
            output_dir=getattr(self.config, "output_dir", "runs/grpo-default"),
            learning_rate=float(getattr(alignment, "learning_rate", 1e-6)),
            per_device_train_batch_size=int(getattr(alignment, "per_device_train_batch_size", 4)),
            gradient_accumulation_steps=int(getattr(alignment, "gradient_accumulation_steps", 1)),
            num_train_epochs=int(getattr(alignment, "num_train_epochs", 1)),
            num_generations=int(getattr(alignment, "num_generations", 8)),
            max_completion_length=int(getattr(alignment, "max_completion_length", 512)),
            max_prompt_length=int(getattr(alignment, "max_prompt_length", 1024)),
            beta=float(getattr(alignment, "kl_coef", 0.04)),
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=10,
            seed=int(getattr(alignment, "seed", 42)),
            # Metrics flow through KikiMetricsCallback → ExperimentTracker, not TRL
            report_to=["none"],
        )

        # vLLM integration for fast generation
        use_vllm = getattr(alignment, "use_vllm", False)
        if use_vllm:
            grpo_config.use_vllm = True
            grpo_config.vllm_mode = getattr(alignment, "vllm_mode", "colocate")
            logger.info("GRPO using vLLM for generation (mode=%s)", grpo_config.vllm_mode)

        if not self.reward_functions:
            logger.warning("No reward functions provided for GRPO training")

        self.trainer = GRPOTrainer(
            model=self.model,
            args=grpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            reward_funcs=self.reward_functions,
        )

        logger.info(
            "GRPOTrainer initialized (lr=%s, generations=%d, rewards=%d)",
            grpo_config.learning_rate,
            grpo_config.num_generations,
            len(self.reward_functions),
        )

    def train(self) -> dict:
        """Run GRPO training with reward functions."""
        result = self.trainer.train()
        metrics = {k: v for k, v in result.metrics.items() if isinstance(v, (int, float))}
        logger.info("GRPO training complete")
        return metrics
