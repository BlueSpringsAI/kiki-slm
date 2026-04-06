#!/usr/bin/env python3
"""Step 2: Stratified sampling of filtered tickets for trace generation.

Selects a diverse subset from filtered Freshdesk tickets, stratified by
conversation depth, sentiment, and tag-based category proxy. English-only.

Usage:
    python scripts/loopper/sample_tickets.py --n 5000
    python scripts/loopper/sample_tickets.py --n 5000 --input-dir /path/to/filtered
    python scripts/loopper/sample_tickets.py --n 1000 --dry-run   # stats only
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    from langdetect import detect, DetectorFactory, LangDetectException
    DetectorFactory.seed = 42  # deterministic results
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Default paths (from configs/loopper_pipeline.yaml or project-relative) ──
from scripts.loopper.config import get_default_paths as _get_paths
_PATHS = _get_paths()
FILTERED_DIR = _PATHS["filtered_tickets"]
OUTPUT_DIR = _PATHS["sampled_tickets"]

# ── English-only filter ───────────────────────────────────────
# Include tickets detected as English OR with no language detected
# (undetected are mostly English based on analysis)
ENGLISH_LANGUAGES = {"en", None, "none", ""}

# ── Stratification targets (3 dimensions) ─────────────────────

DEPTH_TARGETS = {
    "single_turn": 0.35,       # no public conversations at all
    "customer_reply": 0.05,    # customer follow-up but no agent reply
    "with_agent": 0.30,        # has agent reply (1 public conv)
    "multi_turn": 0.20,        # 2-4 public conversations
    "deep_thread": 0.10,       # 5+ public conversations
}

SENTIMENT_TARGETS = {
    "negative": 0.20,     # 0-40 score — over-sampled (only ~9% in raw data)
    "neutral": 0.40,      # 41-60
    "positive": 0.40,     # 61-100
}

CATEGORY_PROXY_TARGETS = {
    "reply": 0.30,         # REPLY tag — ongoing thread, likely has context
    "eproof": 0.10,        # EPROOF — design proofs, design_update category
    "tracking": 0.10,      # Tracking — delivery related
    "attachment": 0.15,    # Attachment — often quality complaints with photos
    "promo": 0.05,         # PROMO — promotional, likely new_order
    "none": 0.30,          # No relevant tags — diverse mix
}


def classify_depth(ticket_data: dict) -> str:
    convs = ticket_data.get("conversations", [])
    public = [c for c in convs if not c.get("private", False)]
    n = len(public)
    if n == 0:
        return "single_turn"
    has_agent = any(not c.get("incoming", True) for c in public)
    if n >= 5:
        return "deep_thread"
    if n >= 2:
        return "multi_turn"
    if has_agent:
        return "with_agent"
    return "customer_reply"


def classify_sentiment(score: int | None) -> str:
    if score is None:
        return "neutral"
    if score <= 40:
        return "negative"
    if score <= 60:
        return "neutral"
    return "positive"


def is_english(description: str, metadata_lang: str | None = None) -> bool:
    """Detect if a ticket is English by running langdetect on its description text.

    Trusts metadata_lang ONLY when description is too short to detect reliably.
    Freshdesk's detected_language field is unreliable — some German tickets are
    marked as 'en', so we verify using actual content.
    """
    text = (description or "").strip()

    # Too short to detect reliably — trust metadata
    if len(text) < 20:
        if not metadata_lang:
            return True
        return metadata_lang.lower().split("-")[0] in {"en", "none", ""}

    if not LANGDETECT_AVAILABLE:
        # Fallback to metadata-only filter
        if not metadata_lang:
            return True
        return metadata_lang.lower().split("-")[0] in {"en", "none", ""}

    # Use first 500 chars — enough for reliable detection, fast
    try:
        return detect(text[:500]) == "en"
    except LangDetectException:
        # Detection failed (unusual characters, etc.) — err on the side of excluding
        return False


def classify_category_proxy(tags: list) -> str:
    """Assign a rough category proxy based on Freshdesk tags."""
    tags_upper = {t.upper() for t in (tags or [])}
    if "REPLY" in tags_upper:
        return "reply"
    if "EPROOF" in tags_upper:
        return "eproof"
    if "TRACKING" in tags_upper or "Tracking" in (tags or []):
        return "tracking"
    if "PROMO" in tags_upper:
        return "promo"
    if "ATTACHMENT" in tags_upper or "Attachment" in (tags or []):
        return "attachment"
    return "none"


def get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def load_ticket_features(filtered_dir: str) -> list[dict]:
    """Load filtered tickets, keep only English, extract stratification features."""
    filtered_path = Path(filtered_dir)
    files = sorted(filtered_path.glob("*.json"))

    if not files:
        logger.error("No JSON files found in %s", filtered_dir)
        sys.exit(1)

    logger.info("Scanning %d filtered tickets...", len(files))

    tickets = []
    skipped_language = 0
    load_errors = 0

    for i, filepath in enumerate(files):
        if i % 50_000 == 0 and i > 0:
            logger.info("  ... scanned %d / %d (kept %d English)", i, len(files), len(tickets))
        try:
            with open(filepath) as f:
                data = json.load(f)

            ticket = data.get("ticket", {})
            lang = ticket.get("detected_language")
            desc = (ticket.get("description_text", "") or "").strip()

            # English-only filter — uses langdetect on description text,
            # falls back to metadata only when text is too short
            if not is_english(desc, lang):
                skipped_language += 1
                continue

            tags = ticket.get("tags", []) or []

            tickets.append({
                "filepath": filepath,
                "filename": filepath.name,
                "ticket_id": ticket.get("id"),
                "depth": classify_depth(data),
                "sentiment": classify_sentiment(ticket.get("sentiment_score")),
                "category_proxy": classify_category_proxy(tags),
                "tags": tags,
                "desc_length": len(desc),
            })
        except json.JSONDecodeError:
            load_errors += 1
        except OSError as e:
            load_errors += 1
            if load_errors <= 5:
                logger.warning("  OS error reading %s: %s", filepath, e)

    logger.info("English tickets: %d (skipped %d non-English, %d errors)",
                len(tickets), skipped_language, load_errors)

    if load_errors > 0 and load_errors > len(files) * 0.01:
        logger.warning("  WARNING: >1%% of files failed to load — check data integrity")

    # Sanity check
    depth_counts = Counter(t["depth"] for t in tickets)
    non_single = sum(v for k, v in depth_counts.items() if k != "single_turn")
    if non_single == 0 and len(tickets) > 100:
        logger.warning(
            "  WARNING: ALL %d tickets classified as single_turn — "
            "conversations may not be loading correctly", len(tickets)
        )

    return tickets


def stratified_sample(
    tickets: list[dict],
    n: int,
    seed: int = 42,
) -> list[dict]:
    """Sample n tickets using 3-dimensional stratification.

    Dimensions: depth x sentiment x category_proxy = 5 x 3 x 6 = 90 cells.
    """
    random.seed(seed)

    # Group tickets by (depth, sentiment, category_proxy)
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for t in tickets:
        key = (t["depth"], t["sentiment"], t["category_proxy"])
        strata[key].append(t)

    # Calculate raw target counts per stratum
    raw_targets: dict[tuple, float] = {}
    for key in strata:
        depth, sent, cat_proxy = key
        depth_w = DEPTH_TARGETS.get(depth, 0.02)
        sent_w = SENTIMENT_TARGETS.get(sent, 0.33)
        cat_w = CATEGORY_PROXY_TARGETS.get(cat_proxy, 0.05)
        raw_targets[key] = depth_w * sent_w * cat_w

    # Normalize to sum to n
    total_weight = sum(raw_targets.values())
    if total_weight == 0:
        logger.error("All stratum weights are zero")
        return random.sample(tickets, min(n, len(tickets)))

    stratum_targets: dict[tuple, int] = {}
    for key, weight in raw_targets.items():
        stratum_targets[key] = max(1, round(n * weight / total_weight))

    # Greedy adjustment to hit exactly n
    current_sum = sum(stratum_targets.values())
    if current_sum != n:
        fractional = {}
        for key, weight in raw_targets.items():
            exact = n * weight / total_weight
            fractional[key] = exact - int(exact)

        if current_sum > n:
            for key in sorted(fractional, key=fractional.get):
                if current_sum <= n:
                    break
                if stratum_targets[key] > 1:
                    stratum_targets[key] -= 1
                    current_sum -= 1
        else:
            for key in sorted(fractional, key=fractional.get, reverse=True):
                if current_sum >= n:
                    break
                stratum_targets[key] += 1
                current_sum += 1

    # Sample from each stratum
    sampled = []
    deficit_by_stratum: dict[tuple, int] = {}

    for key, target in sorted(stratum_targets.items(), key=lambda x: -x[1]):
        available = strata.get(key, [])
        actual = min(target, len(available))
        if actual < target:
            deficit_by_stratum[key] = target - actual
        if available:
            sampled.extend(random.sample(available, actual))

    # Stratification-aware deficit fill
    total_deficit = sum(deficit_by_stratum.values())
    if total_deficit > 0 and len(sampled) < n:
        already_sampled = {id(t) for t in sampled}
        stratum_headroom = []
        for key, items in strata.items():
            remaining = [t for t in items if id(t) not in already_sampled]
            if remaining:
                stratum_headroom.append((key, len(remaining), remaining))

        stratum_headroom.sort(key=lambda x: -x[1])
        remaining_to_fill = n - len(sampled)
        for key, headroom, available in stratum_headroom:
            if remaining_to_fill <= 0:
                break
            take = min(headroom, max(1, remaining_to_fill // max(len(stratum_headroom), 1)))
            sampled.extend(random.sample(available, take))
            remaining_to_fill -= take

    random.shuffle(sampled)
    sampled = sampled[:n]

    if total_deficit > 0:
        logger.info("  Deficit fill: %d tickets redistributed from strata with headroom", total_deficit)

    return sampled


def print_distribution(sampled: list[dict], label: str = "Sampled"):
    total = len(sampled)
    if total == 0:
        logger.info("  (empty)")
        return

    logger.info("")
    logger.info("=" * 60)
    logger.info("  %s DISTRIBUTION (%d tickets)", label.upper(), total)
    logger.info("=" * 60)

    # Depth
    depth_counts = Counter(t["depth"] for t in sampled)
    logger.info("")
    logger.info("  Conversation depth:")
    for d in ["single_turn", "customer_reply", "with_agent", "multi_turn", "deep_thread"]:
        c = depth_counts.get(d, 0)
        target = DEPTH_TARGETS.get(d, 0) * 100
        logger.info("    %-18s %5d (%5.1f%%)  target: %.0f%%", d, c, c / total * 100, target)

    # Sentiment
    sent_counts = Counter(t["sentiment"] for t in sampled)
    logger.info("")
    logger.info("  Sentiment:")
    for s in ["negative", "neutral", "positive"]:
        c = sent_counts.get(s, 0)
        target = SENTIMENT_TARGETS.get(s, 0) * 100
        logger.info("    %-18s %5d (%5.1f%%)  target: %.0f%%", s, c, c / total * 100, target)

    # Category proxy
    cat_counts = Counter(t["category_proxy"] for t in sampled)
    logger.info("")
    logger.info("  Category proxy (from tags):")
    for cat in ["reply", "eproof", "tracking", "attachment", "promo", "none"]:
        c = cat_counts.get(cat, 0)
        target = CATEGORY_PROXY_TARGETS.get(cat, 0) * 100
        logger.info("    %-18s %5d (%5.1f%%)  target: %.0f%%", cat, c, c / total * 100, target)

    # Message length
    lengths = [t["desc_length"] for t in sampled]
    logger.info("")
    logger.info("  Message length:")
    logger.info("    < 50 chars      %5d", sum(1 for l in lengths if l < 50))
    logger.info("    50-500 chars    %5d", sum(1 for l in lengths if 50 <= l < 500))
    logger.info("    500-2000 chars  %5d", sum(1 for l in lengths if 500 <= l < 2000))
    logger.info("    > 2000 chars    %5d", sum(1 for l in lengths if l >= 2000))


def main():
    parser = argparse.ArgumentParser(description="Stratified ticket sampling (English-only)")
    parser.add_argument("--input-dir", default=FILTERED_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--n", type=int, default=5000, help="Number of tickets to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Print stats only, don't copy files")
    args = parser.parse_args()

    tickets = load_ticket_features(args.input_dir)
    print_distribution(tickets, "Raw (English filtered)")

    sampled = stratified_sample(tickets, args.n, args.seed)
    print_distribution(sampled, "Sampled")

    if args.dry_run:
        logger.info("\n  (dry run — no files copied)")
        return

    # Deduplicate filenames
    filenames = [t["filename"] for t in sampled]
    dupes = [fn for fn, cnt in Counter(filenames).items() if cnt > 1]
    if dupes:
        logger.warning("  WARNING: %d duplicate filenames, deduplicating", len(dupes))
        seen = set()
        sampled = [t for t in sampled if t["filename"] not in seen and not seen.add(t["filename"])]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for t in sampled:
        shutil.copy2(t["filepath"], output_dir / t["filename"])

    # Manifest
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "git_hash": get_git_hash(),
        "language_filter": "english_only",
        "total_filtered": len(tickets),
        "sampled": len(sampled),
        "seed": args.seed,
        "input_dir": args.input_dir,
        "targets": {
            "depth": DEPTH_TARGETS,
            "sentiment": SENTIMENT_TARGETS,
            "category_proxy": CATEGORY_PROXY_TARGETS,
        },
        "actual": {
            "depth": dict(Counter(t["depth"] for t in sampled)),
            "sentiment": dict(Counter(t["sentiment"] for t in sampled)),
            "category_proxy": dict(Counter(t["category_proxy"] for t in sampled)),
        },
        "joint_distribution": {
            f"{d}|{s}|{c}": count
            for (d, s, c), count in Counter(
                (t["depth"], t["sentiment"], t["category_proxy"])
                for t in sampled
            ).most_common(50)
        },
        "ticket_ids": [t["ticket_id"] for t in sampled],
    }
    manifest_path = output_dir / "_sampling_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info("\n  Sampled %d English tickets saved to %s", len(sampled), output_dir)
    logger.info("  Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
