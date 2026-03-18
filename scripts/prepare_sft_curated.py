#!/usr/bin/env python3
"""Prepare a curated, high-quality SFT dataset for Kiki SLM.

Replaces prepare_sft_chatml.py with research-backed data curation:
  1. Download & convert all 10 datasets (same as before)
  2. Per-source hard caps (CS-core 60%, tool-calling 20%, secondary 20%)
  3. Exact dedup (MD5 on user+assistant content)
  4. Semantic dedup (sentence-transformer, threshold 0.85)
  5. Length filtering (min 20, max 1500 whitespace tokens)
  6. Intent balancing (no single intent > 25%)
  7. Add empty <think></think> blocks for Qwen3 non-thinking mode
  8. Stratified train/eval split by source
  9. Save as JSONL

Output: data/curated/kiki_sft_curated_train.jsonl  (90%)
        data/curated/kiki_sft_curated_eval.jsonl   (10%)

Usage:
    python scripts/prepare_sft_curated.py
    python scripts/prepare_sft_curated.py --target 15000
    python scripts/prepare_sft_curated.py --target 10000 --semantic-dedup-threshold 0.85
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset config: hard caps replace proportional weights
# ---------------------------------------------------------------------------

# Tier 1: CS-core (60% of target)
# Tier 2: Tool/function calling (20% of target)
# Tier 3: Secondary domains (20% of target)

SFT_DATASETS = {
    # --- Tier 1: CS-core ---
    "bitext_cs": {
        "hf_id": "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        "converter": "bitext",
        "tier": 1,
        "tier_weight": 0.35,  # weight within tier
        "description": "General CS (27K)",
    },
    "bitext_ecom": {
        "hf_id": "bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset",
        "converter": "bitext",
        "tier": 1,
        "tier_weight": 0.35,
        "description": "E-commerce (50K+)",
    },
    "customer_support_tickets": {
        "hf_id": "Tobi-Bueck/customer-support-tickets",
        "converter": "ticket",
        "tier": 1,
        "tier_weight": 0.30,
        "filter_language": "en",
        "description": "Helpdesk tickets (61.8K, English subset)",
    },
    # --- Tier 2: Tool/function calling ---
    "xlam_60k": {
        "hf_id": "Salesforce/xlam-function-calling-60k",
        "converter": "xlam",
        "tier": 2,
        "tier_weight": 0.55,
        "description": "Function calling (60K)",
    },
    "hermes_fc": {
        "hf_id": "NousResearch/hermes-function-calling-v1",
        "converter": "hermes",
        "tier": 2,
        "tier_weight": 0.45,
        "description": "Tool calling (11.6K)",
    },
    # --- Tier 3: Secondary domains ---
    "bitext_banking": {
        "hf_id": "bitext/Bitext-retail-banking-llm-chatbot-training-dataset",
        "converter": "bitext",
        "tier": 3,
        "tier_weight": 0.30,
        "description": "Banking (37K+)",
    },
    "bitext_insurance": {
        "hf_id": "bitext/Bitext-insurance-llm-chatbot-training-dataset",
        "converter": "bitext",
        "tier": 3,
        "tier_weight": 0.30,
        "description": "Insurance (38K+)",
    },
    "banking77": {
        "hf_id": "legacy-datasets/banking77",
        "converter": "banking77",
        "tier": 3,
        "tier_weight": 0.40,
        "description": "Banking intent 77-class (13K)",
    },
}

# Dropped datasets:
# - arcee_agent: Generic agent data, low CS relevance, was 28.5% of old dataset
# - clinc_oos: General intent classification, low CS relevance

TIER_ALLOCATIONS = {
    1: 0.60,  # CS-core: 60%
    2: 0.20,  # Tool calling: 20%
    3: 0.20,  # Secondary: 20%
}


# ---------------------------------------------------------------------------
# Download & convert (reuses existing converters)
# ---------------------------------------------------------------------------

def download_and_convert(
    name: str,
    config: dict,
    num_samples: int | None,
    token: str | None = None,
) -> list[dict]:
    """Download a dataset, convert to ChatML, subsample to num_samples."""
    from kiki.data.processors import ChatMLConverter

    hf_id = config["hf_id"]
    converter_name = config["converter"]

    logger.info("Downloading '%s' (%s)...", name, hf_id)
    t0 = time.monotonic()

    kwargs = {"path": hf_id, "split": "train"}
    if config.get("subset"):
        kwargs["name"] = config["subset"]
    if token:
        kwargs["token"] = token

    try:
        ds = load_dataset(**kwargs)
    except Exception as exc:
        logger.error("Failed to download '%s': %s", name, exc)
        return []

    elapsed = time.monotonic() - t0
    logger.info("  Downloaded %d examples in %.1fs", len(ds), elapsed)

    # Language filter
    lang_filter = config.get("filter_language")
    if lang_filter and "language" in ds.column_names:
        before = len(ds)
        ds = ds.filter(lambda x: x.get("language") == lang_filter)
        logger.info("  Filtered to language='%s': %d -> %d", lang_filter, before, len(ds))

    # Convert all rows
    converter = ChatMLConverter.get_converter(converter_name)
    converted = []
    skipped = 0
    skip_reasons: Counter = Counter()

    for example in ds:
        try:
            result = converter(dict(example))
            messages = result.get("messages", [])

            if len(messages) < 2:
                skipped += 1
                skip_reasons["too_few_messages"] += 1
                continue

            has_assistant = any(
                m.get("role") == "assistant" and m.get("content", "").strip()
                for m in messages
            )
            if not has_assistant:
                skipped += 1
                skip_reasons["no_assistant_content"] += 1
                continue

            converted.append({"messages": messages, "source": name})
        except Exception:
            skipped += 1
            skip_reasons["conversion_error"] += 1

    logger.info(
        "  Converted %d examples (%d skipped: %s) from '%s'",
        len(converted), skipped, dict(skip_reasons), name,
    )

    # Subsample AFTER conversion
    if num_samples is not None and len(converted) > num_samples:
        converted = random.sample(converted, num_samples)
        logger.info("  Subsampled to %d examples", num_samples)
    elif num_samples is not None and len(converted) < num_samples:
        logger.warning(
            "  '%s' has only %d examples (target: %d) — using all",
            name, len(converted), num_samples,
        )

    return converted


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def dedup_exact(examples: list[dict]) -> list[dict]:
    """Remove exact duplicates by hashing user+assistant content."""
    seen: set[str] = set()
    unique = []
    for ex in examples:
        key_parts = [
            f"{m.get('role')}:{m.get('content', '')}"
            for m in ex.get("messages", [])
            if m.get("role") != "system"
        ]
        h = hashlib.md5("|".join(key_parts).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(ex)

    removed = len(examples) - len(unique)
    logger.info("Exact dedup: %d -> %d (removed %d)", len(examples), len(unique), removed)
    return unique


def dedup_semantic(
    examples: list[dict],
    threshold: float = 0.85,
    batch_size: int = 512,
) -> list[dict]:
    """Remove near-duplicates using sentence-transformer embeddings.

    Uses FAISS index for efficient similarity search instead of O(n^2) comparison.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        logger.warning("sentence-transformers not installed, skipping semantic dedup")
        return examples

    logger.info("Computing embeddings for %d examples...", len(examples))
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Extract user message text for embedding
    texts = []
    for ex in examples:
        user_msgs = [
            m.get("content", "")
            for m in ex.get("messages", [])
            if m.get("role") == "user"
        ]
        texts.append(" ".join(user_msgs) if user_msgs else "")

    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True,
    )

    # Use FAISS for efficient nearest-neighbor search if available
    try:
        import faiss

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product = cosine for normalized vecs

        keep_indices = []
        removed = 0

        for i in range(len(embeddings)):
            vec = embeddings[i:i+1].astype(np.float32)
            if index.ntotal > 0:
                scores, _ = index.search(vec, min(10, index.ntotal))
                if scores[0][0] >= threshold:
                    removed += 1
                    continue
            index.add(vec)
            keep_indices.append(i)

    except ImportError:
        logger.info("FAISS not available, using sliding window comparison")
        keep_indices = []
        removed = 0
        # Sliding window: compare against last N kept items
        window_size = 500  # larger window than original 100
        for i in range(len(embeddings)):
            is_dup = False
            for j in keep_indices[-window_size:]:
                sim = float(np.dot(embeddings[i], embeddings[j]))
                if sim >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                keep_indices.append(i)
            else:
                removed += 1

    logger.info("Semantic dedup: %d -> %d (removed %d)", len(examples), len(keep_indices), removed)
    return [examples[i] for i in keep_indices]


# ---------------------------------------------------------------------------
# Length filtering
# ---------------------------------------------------------------------------

def filter_length(
    examples: list[dict],
    min_tokens: int = 20,
    max_tokens: int = 1500,
) -> list[dict]:
    """Remove examples where total content is too short or too long."""
    filtered = []
    removed_short = 0
    removed_long = 0

    for ex in examples:
        total_text = " ".join(
            m.get("content", "") for m in ex.get("messages", [])
        )
        token_count = len(total_text.split())

        if token_count < min_tokens:
            removed_short += 1
        elif token_count > max_tokens:
            removed_long += 1
        else:
            filtered.append(ex)

    logger.info(
        "Length filter: %d -> %d (removed %d short, %d long)",
        len(examples), len(filtered), removed_short, removed_long,
    )
    return filtered


# ---------------------------------------------------------------------------
# Intent balancing
# ---------------------------------------------------------------------------

def _extract_intent(ex: dict) -> str | None:
    """Try to extract intent from assistant JSON response."""
    for m in ex.get("messages", []):
        if m.get("role") == "assistant":
            content = m.get("content", "")
            try:
                parsed = json.loads(content)
                return parsed.get("intent")
            except (json.JSONDecodeError, TypeError):
                # Try regex fallback
                match = re.search(r'"intent"\s*:\s*"([^"]+)"', content)
                if match:
                    return match.group(1)
    return None


def balance_intents(
    examples: list[dict],
    max_ratio: float = 0.25,
) -> list[dict]:
    """Downsample any intent exceeding max_ratio of the dataset."""
    # Separate examples with/without extractable intents
    with_intent: list[tuple[int, str]] = []
    without_intent: list[int] = []

    for i, ex in enumerate(examples):
        intent = _extract_intent(ex)
        if intent:
            with_intent.append((i, intent))
        else:
            without_intent.append(i)

    counts = Counter(intent for _, intent in with_intent)
    total = len(examples)
    max_per_intent = int(total * max_ratio)

    over_represented = {
        intent: count for intent, count in counts.items()
        if count > max_per_intent
    }

    if not over_represented:
        logger.info("Intent balance: all intents within %.0f%% threshold", max_ratio * 100)
        return examples

    logger.info("Over-represented intents: %s (max: %d)", over_represented, max_per_intent)

    # Group indices by intent
    intent_indices: dict[str, list[int]] = {}
    for idx, intent in with_intent:
        intent_indices.setdefault(intent, []).append(idx)

    rng = random.Random(42)
    keep_indices: set[int] = set(without_intent)

    for intent, indices in intent_indices.items():
        if intent in over_represented:
            rng.shuffle(indices)
            keep_indices.update(indices[:max_per_intent])
        else:
            keep_indices.update(indices)

    keep_sorted = sorted(keep_indices)
    logger.info("Intent balance: %d -> %d", len(examples), len(keep_sorted))
    return [examples[i] for i in keep_sorted]


# ---------------------------------------------------------------------------
# Qwen3 non-thinking mode: inject empty <think></think>
# ---------------------------------------------------------------------------

def inject_empty_think_blocks(examples: list[dict]) -> list[dict]:
    """Add empty <think></think> block to all assistant messages.

    Per Qwen3 team recommendation (GitHub Discussion #1429):
    Use empty think blocks to explicitly train non-thinking mode.
    """
    modified = 0
    for ex in examples:
        for m in ex.get("messages", []):
            if m.get("role") == "assistant":
                content = m.get("content", "")
                # Skip if already has think block
                if "<think>" in content:
                    continue
                m["content"] = "<think>\n\n</think>\n\n" + content
                modified += 1

    logger.info("Injected empty <think> blocks into %d assistant messages", modified)
    return examples


# ---------------------------------------------------------------------------
# Stratified split by source
# ---------------------------------------------------------------------------

def stratified_split(
    examples: list[dict],
    eval_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split train/eval stratified by source so eval represents all sources."""
    rng = random.Random(seed)

    # Group by source
    by_source: dict[str, list[dict]] = {}
    for ex in examples:
        source = ex.get("source", "unknown")
        by_source.setdefault(source, []).append(ex)

    train_data: list[dict] = []
    eval_data: list[dict] = []

    for source, source_examples in by_source.items():
        rng.shuffle(source_examples)
        eval_size = max(1, int(len(source_examples) * eval_ratio))
        eval_data.extend(source_examples[:eval_size])
        train_data.extend(source_examples[eval_size:])

    rng.shuffle(train_data)
    rng.shuffle(eval_data)

    logger.info("Stratified split: %d train + %d eval", len(train_data), len(eval_data))
    return train_data, eval_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare curated SFT dataset for Kiki SLM")
    parser.add_argument("--target", type=int, default=10000, help="Target total examples (default: 10000)")
    parser.add_argument("--eval-ratio", type=float, default=0.10, help="Eval split ratio (default: 0.10)")
    parser.add_argument("--output-dir", type=str, default="data/curated", help="Output directory")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace token")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--semantic-dedup-threshold", type=float, default=0.85,
        help="Cosine similarity threshold for semantic dedup (default: 0.85)",
    )
    parser.add_argument("--skip-semantic-dedup", action="store_true", help="Skip semantic dedup (faster)")
    parser.add_argument("--no-think-blocks", action="store_true", help="Skip injecting <think> blocks")
    args = parser.parse_args()

    random.seed(args.seed)
    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print(f"\n{'='*70}")
    print(f"KIKI SLM — Curated SFT Dataset Preparation")
    print(f"  Target: {args.target:,} examples")
    print(f"  Tiers: CS-core 60% | Tool-calling 20% | Secondary 20%")
    print(f"  Dropped: arcee_agent, clinc_oos (low CS relevance)")
    print(f"{'='*70}\n")

    # Calculate per-dataset caps based on tier allocation
    all_examples: list[dict] = []

    for tier, tier_ratio in TIER_ALLOCATIONS.items():
        tier_budget = int(args.target * tier_ratio)
        tier_datasets = {k: v for k, v in SFT_DATASETS.items() if v["tier"] == tier}
        total_tier_weight = sum(v["tier_weight"] for v in tier_datasets.values())

        tier_names = ", ".join(tier_datasets.keys())
        logger.info("Tier %d (%s): budget=%d examples", tier, tier_names, tier_budget)

        for name, config in tier_datasets.items():
            # Allocate within tier by weight, with 20% buffer for dedup losses
            raw_target = int(tier_budget * config["tier_weight"] / total_tier_weight)
            download_target = int(raw_target * 1.2)  # 20% buffer

            examples = download_and_convert(name, config, download_target, token)
            all_examples.extend(examples)

    print(f"\n  Total downloaded: {len(all_examples):,} examples")

    # --- Curation pipeline ---

    # Step 1: Exact dedup
    all_examples = dedup_exact(all_examples)

    # Step 2: Semantic dedup
    if not args.skip_semantic_dedup:
        all_examples = dedup_semantic(all_examples, threshold=args.semantic_dedup_threshold)
    else:
        logger.info("Skipping semantic dedup (--skip-semantic-dedup)")

    # Step 3: Length filter
    all_examples = filter_length(all_examples, min_tokens=20, max_tokens=1500)

    # Step 4: Intent balancing
    all_examples = balance_intents(all_examples, max_ratio=0.25)

    # Step 5: Final cap to target size
    if len(all_examples) > args.target:
        # Proportionally downsample by source to maintain distribution
        by_source: dict[str, list[dict]] = {}
        for ex in all_examples:
            by_source.setdefault(ex["source"], []).append(ex)

        ratio = args.target / len(all_examples)
        capped: list[dict] = []
        for source, source_examples in by_source.items():
            n = max(1, int(len(source_examples) * ratio))
            random.shuffle(source_examples)
            capped.extend(source_examples[:n])
        all_examples = capped
        logger.info("Capped to target: %d examples", len(all_examples))

    # Step 6: Inject empty <think></think> blocks
    if not args.no_think_blocks:
        all_examples = inject_empty_think_blocks(all_examples)
    else:
        logger.info("Skipping <think> block injection (--no-think-blocks)")

    # Step 7: Stratified split
    train_data, eval_data = stratified_split(all_examples, eval_ratio=args.eval_ratio, seed=args.seed)

    # --- Save ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "kiki_sft_curated_train.jsonl"
    eval_path = output_dir / "kiki_sft_curated_eval.jsonl"

    for path, data in [(train_path, train_data), (eval_path, eval_data)]:
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"DONE! Saved {len(train_data):,} train + {len(eval_data):,} eval examples")
    print(f"{'='*70}")
    print(f"  Train: {train_path} ({train_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  Eval:  {eval_path} ({eval_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Source distribution
    source_counts = Counter(ex["source"] for ex in all_examples)
    print(f"\n  Source distribution ({len(all_examples):,} total):")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        tier = SFT_DATASETS.get(source, {}).get("tier", "?")
        print(f"    [T{tier}] {source:<30s} {count:>5d} ({count/len(all_examples)*100:.1f}%)")

    # Intent distribution
    intent_counts: Counter = Counter()
    for ex in all_examples:
        intent = _extract_intent(ex)
        if intent:
            intent_counts[intent] += 1
    if intent_counts:
        print(f"\n  Intent distribution:")
        for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
            print(f"    {intent:<25s} {count:>5d} ({count/len(all_examples)*100:.1f}%)")

    print(f"\n  Upload to Google Drive:")
    print(f"    1. {train_path}")
    print(f"    2. {eval_path}")
    print()


if __name__ == "__main__":
    main()
