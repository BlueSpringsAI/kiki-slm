"""GPU memory monitoring and utilities.

Task 3.35: Memory stats, cache clearing, training time estimation,
and Flash Attention availability check.
"""

from __future__ import annotations

import gc
import logging

logger = logging.getLogger(__name__)


def get_gpu_memory() -> dict | None:
    """Return GPU memory stats in GB: total, used, free.

    Returns None if CUDA is not available.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_mem / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        free = total - reserved

        return {
            "device": torch.cuda.get_device_name(device),
            "total_gb": round(total, 2),
            "reserved_gb": round(reserved, 2),
            "allocated_gb": round(allocated, 2),
            "free_gb": round(free, 2),
        }
    except Exception as exc:
        logger.debug("Could not query GPU memory: %s", exc)
        return None


def clear_gpu_cache() -> None:
    """Clear GPU cache and run garbage collection."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU cache cleared")
    except ImportError:
        pass


def estimate_training_time(
    model_params_m: float,
    dataset_size: int,
    batch_size: int,
    gradient_accum: int = 1,
    num_epochs: int = 3,
    tokens_per_example: int = 512,
) -> dict:
    """Rough estimate of training time on a single A100.

    Args:
        model_params_m: Model parameters in millions (e.g. 4000 for 4B).
        dataset_size: Number of training examples.
        batch_size: Per-device batch size.
        gradient_accum: Gradient accumulation steps.
        num_epochs: Number of training epochs.
        tokens_per_example: Average tokens per example.

    Returns:
        Dict with estimated hours and steps.
    """
    effective_batch = batch_size * gradient_accum
    steps_per_epoch = dataset_size // effective_batch
    total_steps = steps_per_epoch * num_epochs
    total_tokens = dataset_size * tokens_per_example * num_epochs

    # Rough throughput: ~3000 tokens/sec for 4B QLoRA on A100
    # Scale inversely with model size
    base_throughput = 3000  # tokens/sec for 4B model
    throughput = base_throughput * (4000 / model_params_m)
    estimated_seconds = total_tokens / throughput
    estimated_hours = estimated_seconds / 3600

    return {
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "total_tokens": total_tokens,
        "estimated_throughput_tokens_sec": round(throughput),
        "estimated_hours": round(estimated_hours, 2),
    }


def check_flash_attention_available() -> bool:
    """Check if Flash Attention 2 is available."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        # Check for flash_attn package
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False
