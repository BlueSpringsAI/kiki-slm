#!/usr/bin/env python3
"""Merge LoRA adapters into base model and export to various formats.

Usage:
    python scripts/merge_and_export.py --adapter-path runs/sft-response-v1 --format gguf
    python scripts/merge_and_export.py --adapter-path runs/grpo-v1 --format mlx --output-dir outputs/exports/kiki-mlx
"""

from __future__ import annotations

import argparse
import logging

from kiki.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiki SLM Adapter Merge & Export")
    parser.add_argument("--adapter-path", required=True, help="Path to trained LoRA adapter")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base model name/path")
    parser.add_argument("--output-dir", default="outputs/exports", help="Output directory")
    parser.add_argument(
        "--format",
        required=True,
        choices=["gguf", "mlx", "safetensors", "awq"],
        help="Export format",
    )
    parser.add_argument("--quantization", default="q4_k_m", help="Quantization method (for GGUF)")
    parser.add_argument("--quantize-bits", type=int, default=4, help="Quantization bits (for MLX)")
    args = parser.parse_args()

    from kiki.models.merge_adapter import AdapterMerger

    if args.format == "gguf":
        logger.info("Merging and exporting to GGUF...")
        model, tokenizer = AdapterMerger.merge_adapter(args.base_model, args.adapter_path)
        AdapterMerger.export_gguf(model, tokenizer, args.output_dir, args.quantization)

    elif args.format == "mlx":
        logger.info("Merging and exporting to MLX...")
        model, tokenizer = AdapterMerger.merge_adapter(args.base_model, args.adapter_path)
        # Save merged model first, then convert to MLX
        merged_path = f"{args.output_dir}/merged-safetensors"
        AdapterMerger.export_safetensors(model, tokenizer, merged_path)
        AdapterMerger.export_mlx(merged_path, args.output_dir, args.quantize_bits)

    elif args.format == "safetensors":
        logger.info("Merging and exporting to SafeTensors...")
        model, tokenizer = AdapterMerger.merge_adapter(args.base_model, args.adapter_path)
        AdapterMerger.export_safetensors(model, tokenizer, args.output_dir)

    elif args.format == "awq":
        logger.info("Merging and exporting to AWQ...")
        model, tokenizer = AdapterMerger.merge_adapter(args.base_model, args.adapter_path)
        merged_path = f"{args.output_dir}/merged-safetensors"
        AdapterMerger.export_safetensors(model, tokenizer, merged_path)
        AdapterMerger.export_awq(merged_path, args.output_dir)

    logger.info("Export complete: %s -> %s", args.adapter_path, args.output_dir)


if __name__ == "__main__":
    main()
