"""Kiki SLM training modules: SFT, DPO, GRPO, and KTO."""

from kiki.trainers.base_trainer import BaseTrainer
from kiki.trainers.dpo_trainer import KikiDPOTrainer
from kiki.trainers.grpo_trainer import KikiGRPOTrainer
from kiki.trainers.kto_trainer import KikiKTOTrainer
from kiki.trainers.sft_trainer import KikiSFTTrainer

__all__ = [
    "BaseTrainer",
    "KikiDPOTrainer",
    "KikiGRPOTrainer",
    "KikiKTOTrainer",
    "KikiSFTTrainer",
]
