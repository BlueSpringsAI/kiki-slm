#!/usr/bin/env python3
"""Launch vLLM server with multi-LoRA support or MLX for local serving.

Usage:
    python scripts/serve.py                                    # vLLM with default config
    python scripts/serve.py --config configs/serving/vllm_multi_lora.yaml
    python scripts/serve.py --mlx                              # Apple Silicon local serving
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiki SLM Serving")
    parser.add_argument("--config", default="configs/serving/vllm_multi_lora.yaml", help="Serving config")
    parser.add_argument("--mlx", action="store_true", help="Use MLX for Apple Silicon local serving")
    parser.add_argument("--port", type=int, help="Override port")
    args = parser.parse_args()

    from kiki.utils.config import load_config

    config = load_config(args.config if not args.mlx else "configs/serving/mlx_local.yaml")

    if args.mlx:
        _serve_mlx(config, args.port)
    else:
        _serve_vllm(config, args.port)


def _serve_vllm(config, port_override: int | None = None) -> None:
    """Launch vLLM server from config."""
    port = port_override or getattr(config, "port", 8000)
    model = getattr(config, "model", "Qwen/Qwen3-4B-Instruct-2507")

    cmd = ["vllm", "serve", model, "--port", str(port)]

    if getattr(config, "quantization", None):
        cmd += ["--quantization", config.quantization]
    if getattr(config, "enable_prefix_caching", False):
        cmd += ["--enable-prefix-caching"]
    if getattr(config, "enable_lora", False):
        cmd += ["--enable-lora"]
        cmd += ["--max-loras", str(getattr(config, "max_loras", 4))]
        cmd += ["--max-lora-rank", str(getattr(config, "max_lora_rank", 64))]
        cmd += ["--max-cpu-loras", str(getattr(config, "max_cpu_loras", 8))]

        # Add LoRA modules
        if hasattr(config, "lora_modules"):
            lora_args = []
            for name, path in config.lora_modules.items():
                lora_args.append(f"{name}={path}")
            if lora_args:
                cmd += ["--lora-modules"] + lora_args

    if getattr(config, "gpu_memory_utilization", None):
        cmd += ["--gpu-memory-utilization", str(config.gpu_memory_utilization)]
    if getattr(config, "max_model_len", None):
        cmd += ["--max-model-len", str(config.max_model_len)]
    if getattr(config, "enable_chunked_prefill", False):
        cmd += ["--enable-chunked-prefill"]
    if getattr(config, "api_key", None):
        cmd += ["--api-key", config.api_key]

    logger.info("Launching vLLM: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        logger.error("vLLM not found. Install with: uv pip install 'kiki-slm[serve]'")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("vLLM server stopped")


def _serve_mlx(config, port_override: int | None = None) -> None:
    """Launch MLX server for Apple Silicon."""
    port = port_override or getattr(config, "port", 8080)
    model = getattr(config, "model", "outputs/exports/kiki-mlx-4bit")

    cmd = [
        "python", "-m", "mlx_lm.server",
        "--model", model,
        "--port", str(port),
    ]
    if getattr(config, "chat_template", None):
        cmd += ["--chat-template", config.chat_template]

    logger.info("Launching MLX server: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        logger.error("mlx_lm not found. Install with: uv pip install 'kiki-slm[apple]'")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("MLX server stopped")


if __name__ == "__main__":
    main()
