#!/usr/bin/env python3
"""Push the merged Kiki SLM to HuggingFace Hub for Inference Endpoints.

Run on Colab AFTER export_gguf.py has created the merged fp16 model at
/content/drive/MyDrive/kiki-slm/merged/kiki-sft-v1/.

Usage in Colab notebook:
    !huggingface-cli login --token hf_YOUR_TOKEN

    !python -u scripts/push_to_hf_hub.py \\
        --model-dir /content/drive/MyDrive/kiki-slm/merged/kiki-sft-v1 \\
        --repo-id BlueSpringsAI/kiki-sft-v1 \\
        --private

Or push from your Mac after downloading the merged model:
    huggingface-cli login
    python scripts/push_to_hf_hub.py \\
        --model-dir ~/kiki-models/kiki-sft-v1-merged \\
        --repo-id BlueSpringsAI/kiki-sft-v1 \\
        --private
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _log(msg: str) -> None:
    print(f"── [HF PUSH] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", required=True,
        help="Local path to the merged fp16 model directory (contains config.json + safetensors).",
    )
    parser.add_argument(
        "--repo-id", required=True,
        help="HuggingFace Hub repo (e.g. BlueSpringsAI/kiki-sft-v1).",
    )
    parser.add_argument(
        "--private", action="store_true", default=True,
        help="Create as a private repo (default: True).",
    )
    parser.add_argument(
        "--public", action="store_true",
        help="Create as a public repo.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    private = not args.public

    # Verify the model dir looks right
    required = ["config.json", "tokenizer_config.json"]
    for f in required:
        if not (model_dir / f).exists():
            _log(f"ERROR: {f} not found in {model_dir}")
            _log("This doesn't look like a merged model directory.")
            _log("Expected: config.json, *.safetensors, tokenizer_config.json, tokenizer.json, ...")
            sys.exit(1)

    safetensors = list(model_dir.glob("*.safetensors"))
    if not safetensors:
        _log(f"ERROR: No .safetensors files in {model_dir}")
        sys.exit(1)

    total_size = sum(f.stat().st_size for f in model_dir.iterdir() if f.is_file())
    _log(f"model: {model_dir}")
    _log(f"  files: {len(list(model_dir.iterdir()))}")
    _log(f"  size: {total_size / 1024**3:.2f} GB")
    _log(f"  safetensors: {[f.name for f in safetensors]}")
    _log(f"repo: {args.repo_id} ({'private' if private else 'public'})")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        _log("Installing huggingface_hub...")
        os.system("pip install -q huggingface_hub")
        from huggingface_hub import HfApi

    api = HfApi()

    # Check auth
    try:
        user = api.whoami()
        _log(f"authenticated as: {user.get('name', user.get('fullname', '?'))}")
    except Exception as e:
        _log(f"ERROR: not authenticated with HuggingFace Hub: {e}")
        _log("Run: huggingface-cli login --token hf_YOUR_TOKEN")
        sys.exit(1)

    # Create repo if needed
    _log(f"creating/verifying repo {args.repo_id}...")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )

    # Upload the entire directory
    _log(f"uploading {total_size / 1024**3:.2f} GB to {args.repo_id}...")
    _log("(this takes 5-15 min depending on connection speed)")
    api.upload_folder(
        repo_id=args.repo_id,
        folder_path=str(model_dir),
        repo_type="model",
        commit_message="Upload merged Kiki SLM (fp16, Qwen3-4B-Thinking-2507 + LoRA)",
    )
    _log(f"DONE — model is at https://huggingface.co/{args.repo_id}")
    _log("")
    _log("Next steps:")
    _log(f"  1. Go to https://huggingface.co/{args.repo_id}/settings")
    _log(f"     → Deploy → Inference Endpoints → New endpoint")
    _log(f"  2. Pick region (us-east-1 for lowest latency to your AWS VPC)")
    _log(f"  3. Pick GPU: Nvidia T4 ($0.60/hr) or L4 ($1.00/hr)")
    _log(f"  4. Enable scale-to-zero if you want to save cost during idle")
    _log(f"  5. Copy the endpoint URL and set in your agent:")
    _log(f"     KIKI_SLM_URL=https://<endpoint-id>.us-east-1.aws.endpoints.huggingface.cloud")
    _log(f"     KIKI_SLM_API_KEY=hf_YOUR_TOKEN")
    _log(f"     KIKI_SLM_MODEL={args.repo_id.split('/')[-1]}")


if __name__ == "__main__":
    main()
