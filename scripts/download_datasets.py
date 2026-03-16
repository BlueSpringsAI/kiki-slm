"""Download all open-source datasets from HuggingFace for the Kiki SLM pipeline.

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --datasets bitext_cs,banking77
    python scripts/download_datasets.py --hf-token hf_xxx
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, dict] = {
    # SFT — Customer service
    "bitext_cs": {
        "id": "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        "description": "General CS intent + response (27K)",
    },
    "bitext_ecom": {
        "id": "bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset",
        "description": "E-commerce specific intents (50K+)",
    },
    "bitext_banking": {
        "id": "bitext/Bitext-retail-banking-llm-chatbot-training-dataset",
        "description": "Banking intents (37K+)",
    },
    "bitext_insurance": {
        "id": "bitext/Bitext-insurance-llm-chatbot-training-dataset",
        "description": "Insurance intents (38K+)",
    },
    "customer_support_tickets": {
        "id": "Tobi-Bueck/customer-support-tickets",
        "description": "Helpdesk tickets with priority/queue/tags (61.8K)",
    },
    "banking77": {
        "id": "legacy-datasets/banking77",
        "description": "Fine-grained banking intent classification (13K, 77 classes)",
    },
    "clinc_oos": {
        "id": "clinc/clinc_oos",
        "subset": "plus",
        "description": "Intent classification with OOS detection (23.7K, 150+1 classes)",
    },
    # SFT — Tool calling
    "arcee_agent": {
        "id": "arcee-ai/agent-data",
        "description": "Combined tool calling + chat (486K)",
    },
    "xlam_60k": {
        "id": "Salesforce/xlam-function-calling-60k",
        "description": "High-precision function calling (60K)",
    },
    "hermes_fc": {
        "id": "NousResearch/hermes-function-calling-v1",
        "description": "Multi-turn tool calling with XML tags (11.6K)",
    },
    "toolace": {
        "id": "Team-ACE/ToolACE",
        "description": "Additional tool calling diversity (11.3K)",
    },
    # SFT — Workflow reasoning
    "capitalone_t1": {
        "id": "capitalone/T1",
        "gated": True,
        "description": "Multi-domain multi-turn tool conversations (13.5K)",
    },
    # SFT — Safety
    "gretel_safety": {
        "id": "gretelai/gretel-safety-alignment-en-v1",
        "description": "Unsafe→safe response pairs for safety SFT",
    },
    # Preference datasets
    "ultrafeedback": {
        "id": "argilla/ultrafeedback-binarized-preferences-cleaned",
        "description": "Primary DPO dataset, cleaned (61K pairs)",
    },
    "helpsteer3": {
        "id": "nvidia/HelpSteer3",
        "gated": True,
        "description": "Rich preference with annotator reasoning (40.5K pairs)",
    },
    "hh_rlhf": {
        "id": "Anthropic/hh-rlhf",
        "description": "Helpfulness + harmlessness preferences (170K pairs)",
    },
    "nectar": {
        "id": "berkeley-nest/Nectar",
        "description": "7-way ranking for reward model (182K × 7)",
    },
}


def download_dataset(
    name: str,
    config: dict,
    output_dir: Path,
    token: str | None = None,
) -> tuple[bool, str]:
    """Download a single dataset. Returns (success, message)."""
    dataset_id = config["id"]
    save_path = output_dir / name

    if save_path.exists():
        logger.info("'%s' already exists at %s — skipping", name, save_path)
        return True, "already exists"

    try:
        kwargs: dict = {"path": dataset_id, "split": "train"}
        if config.get("subset"):
            kwargs["name"] = config["subset"]
        if token:
            kwargs["token"] = token

        t0 = time.monotonic()
        ds = load_dataset(**kwargs)
        elapsed = time.monotonic() - t0

        save_path.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(save_path))

        msg = f"downloaded {len(ds)} examples in {elapsed:.1f}s"
        logger.info("✓ %s: %s", name, msg)
        return True, msg

    except Exception as exc:
        msg = f"FAILED: {exc}"
        logger.error("✗ %s: %s", name, msg)
        return False, msg


def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets for Kiki SLM training")
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset names to download (default: all)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for gated datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/datasets",
        help="Output directory (default: data/datasets)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit",
    )
    args = parser.parse_args()

    if args.list:
        print(f"\n{'Name':<25} {'HF ID':<55} {'Description'}")
        print("-" * 130)
        for name, cfg in DATASET_REGISTRY.items():
            gated = " [GATED]" if cfg.get("gated") else ""
            print(f"{name:<25} {cfg['id']:<55} {cfg['description']}{gated}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # Select datasets to download
    if args.datasets:
        names = [n.strip() for n in args.datasets.split(",")]
        unknown = [n for n in names if n not in DATASET_REGISTRY]
        if unknown:
            logger.error("Unknown datasets: %s. Use --list to see available.", unknown)
            return
    else:
        names = list(DATASET_REGISTRY.keys())

    logger.info("Downloading %d datasets to %s", len(names), output_dir)

    # Download
    results: dict[str, tuple[bool, str]] = {}
    for name in names:
        config = DATASET_REGISTRY[name]
        if config.get("gated") and not token:
            logger.warning("'%s' is gated — skipping (provide --hf-token)", name)
            results[name] = (False, "gated dataset, no token provided")
            continue
        results[name] = download_dataset(name, config, output_dir, token)

    # Summary
    success = sum(1 for ok, _ in results.values() if ok)
    failed = len(results) - success
    print(f"\n{'='*60}")
    print(f"Download summary: {success} succeeded, {failed} failed")
    print(f"{'='*60}")
    for name, (ok, msg) in results.items():
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {name}: {msg}")


if __name__ == "__main__":
    main()
