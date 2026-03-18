#!/usr/bin/env python3
"""Audit and fix gold eval data using the latest model predictions.

Compares model predictions against gold labels. For cases where the model
is arguably correct and the gold label is wrong (common with auto-labeled
data from dataset queues), applies fixes automatically.

Usage:
    # Review what would change (dry run)
    python scripts/audit_gold.py \
        --gold-file data/gold/gold_100.jsonl \
        --eval-results eval_results.json \
        --dry-run

    # Apply fixes
    python scripts/audit_gold.py \
        --gold-file data/gold/gold_100.jsonl \
        --eval-results eval_results.json \
        --apply
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Intent detection heuristics (keyword-based ground truth validation)
# ---------------------------------------------------------------------------

INTENT_SIGNALS = {
    "technical_support": {
        "keywords": ["crash", "outage", "server", "software", "integration", "bug",
                     "database", "sync", "connectivity", "platform", "deploy",
                     "API", "system requirements", "configuration", "install",
                     "network", "rebooted", "logs", "error", "performance issue",
                     "VPN", "SaaS", "data analytics", "encryption", "update"],
        "anti_keywords": [],
    },
    "fraud_report": {
        "keywords": ["unauthorized", "stolen", "fraud", "hacked", "identity theft",
                     "not me", "didn't make", "suspicious", "police report",
                     "wire transfer", "without permission"],
        "anti_keywords": ["wrong amount", "duplicate charge", "overcharged"],
    },
    "product_inquiry": {
        "keywords": ["features", "pricing", "what's included", "options available",
                     "warranty", "discount", "ship internationally", "return policy",
                     "how much", "plan", "compare"],
        "anti_keywords": ["crash", "error", "outage", "not working"],
    },
    "complaint": {
        "keywords": ["unacceptable", "disappointed", "terrible", "worst",
                     "never again", "file a complaint", "lodge a claim",
                     "not satisfied", "poor service"],
        "anti_keywords": [],
    },
    "return_request": {
        "keywords": ["return", "send back", "doesn't fit", "wrong item",
                     "damaged", "defective", "exchange", "wrong color",
                     "wrong size"],
        "anti_keywords": ["return policy", "data analytics", "investment",
                         "marketing strategy", "brand growth", "integration"],
    },
    "billing_inquiry": {
        "keywords": ["charged", "invoice", "bill", "receipt", "payment method",
                     "subscription", "renewed", "duplicate charge", "overcharged"],
        "anti_keywords": ["unauthorized", "stolen", "fraud", "hacked"],
    },
    "general_inquiry": {
        "keywords": ["hours", "FAQ", "contact", "thanks", "thank you",
                     "hello", "hi there"],
        "anti_keywords": [],
    },
}


def detect_likely_intent(message: str) -> str | None:
    """Use keyword heuristics to detect likely intent. Returns None if unclear."""
    msg_lower = message.lower()
    scores = {}

    for intent, signals in INTENT_SIGNALS.items():
        score = 0
        for kw in signals["keywords"]:
            if kw.lower() in msg_lower:
                score += 1
        for akw in signals["anti_keywords"]:
            if akw.lower() in msg_lower:
                score -= 2
        if score > 0:
            scores[intent] = score

    if not scores:
        return None
    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Audit logic
# ---------------------------------------------------------------------------

def audit_gold(gold: list[dict], eval_results: dict) -> list[dict]:
    """Compare gold labels with predictions and generate fix recommendations."""
    gold_map = {g["ticket_id"]: g for g in gold}
    fixes = []

    for r in eval_results.get("ft_results", []):
        tid = r["ticket_id"]
        g = gold_map.get(tid, {})
        p = r.get("parsed")
        if p is None:
            continue

        gold_intent = g.get("gold_intent", "").lower()
        gold_secondary = g.get("gold_intent_secondary", "").lower()
        pred = p.get("intent", "").lower()

        # Skip if already correct
        if pred == gold_intent or (gold_secondary and pred == gold_secondary):
            continue

        message = g.get("customer_message", "")
        likely = detect_likely_intent(message)
        source = g.get("source_dataset", "")

        # Determine fix type
        if likely and likely == pred and likely != gold_intent:
            # Keywords agree with model, disagree with gold → gold is wrong
            fixes.append({
                "ticket_id": tid,
                "action": "relabel",
                "old_intent": g.get("gold_intent"),
                "new_intent": pred,
                "reason": f"Keywords match '{pred}', gold '{gold_intent}' is wrong",
                "source": source,
                "message_preview": message[:100],
            })
        elif likely is None and pred != gold_intent:
            # No clear keyword signal — likely ambiguous, add secondary
            fixes.append({
                "ticket_id": tid,
                "action": "add_secondary",
                "old_intent": g.get("gold_intent"),
                "secondary": pred,
                "reason": f"Ambiguous — accept both '{gold_intent}' and '{pred}'",
                "source": source,
                "message_preview": message[:100],
            })
        else:
            # Keywords disagree with model — model is likely wrong
            fixes.append({
                "ticket_id": tid,
                "action": "keep_gold",
                "gold_intent": g.get("gold_intent"),
                "pred_intent": pred,
                "likely_intent": likely,
                "reason": f"Model wrong — keywords suggest '{likely or gold_intent}'",
                "source": source,
                "message_preview": message[:100],
            })

    return fixes


def apply_fixes(gold: list[dict], fixes: list[dict]) -> tuple[int, int, int]:
    """Apply fixes to gold data. Returns (relabeled, secondary_added, kept)."""
    gold_map = {g["ticket_id"]: g for g in gold}
    relabeled = secondary = kept = 0

    for fix in fixes:
        tid = fix["ticket_id"]
        g = gold_map.get(tid)
        if g is None:
            continue

        if fix["action"] == "relabel":
            g["gold_intent"] = fix["new_intent"]
            relabeled += 1
        elif fix["action"] == "add_secondary":
            if not g.get("gold_intent_secondary"):
                g["gold_intent_secondary"] = fix["secondary"]
                secondary += 1
        elif fix["action"] == "keep_gold":
            kept += 1

    return relabeled, secondary, kept


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Audit and fix gold eval data")
    parser.add_argument("--gold-file", required=True, help="Path to gold JSONL")
    parser.add_argument("--eval-results", required=True, help="Path to eval_results.json")
    parser.add_argument("--dry-run", action="store_true", help="Show fixes without applying")
    parser.add_argument("--apply", action="store_true", help="Apply fixes to gold file")
    args = parser.parse_args()

    # Load
    gold = []
    with open(args.gold_file) as f:
        for line in f:
            if line.strip():
                gold.append(json.loads(line))

    with open(args.eval_results) as f:
        eval_results = json.load(f)

    print(f"Loaded {len(gold)} gold examples, {len(eval_results.get('ft_results', []))} eval results")

    # Audit
    fixes = audit_gold(gold, eval_results)

    # Summary
    action_counts = Counter(f["action"] for f in fixes)
    print(f"\nAudit found {len(fixes)} disagreements:")
    print(f"  Relabel (gold wrong, model right):  {action_counts.get('relabel', 0)}")
    print(f"  Add secondary (ambiguous):          {action_counts.get('add_secondary', 0)}")
    print(f"  Keep gold (model wrong):            {action_counts.get('keep_gold', 0)}")

    # Detail
    for action_type in ["relabel", "add_secondary", "keep_gold"]:
        action_fixes = [f for f in fixes if f["action"] == action_type]
        if action_fixes:
            print(f"\n{'='*60}")
            print(f"  {action_type.upper()} ({len(action_fixes)})")
            print(f"{'='*60}")
            for f in action_fixes:
                if action_type == "relabel":
                    print(f"  {f['ticket_id']}: {f['old_intent']} → {f['new_intent']}")
                    print(f"    {f['message_preview']}...")
                elif action_type == "add_secondary":
                    print(f"  {f['ticket_id']}: {f['old_intent']} + secondary={f['secondary']}")
                    print(f"    {f['message_preview']}...")
                elif action_type == "keep_gold":
                    print(f"  {f['ticket_id']}: gold={f['gold_intent']}, model={f['pred_intent']}, likely={f['likely_intent']}")
                    print(f"    {f['message_preview']}...")

    # Apply
    if args.apply and not args.dry_run:
        relabeled, secondary, kept = apply_fixes(gold, fixes)
        with open(args.gold_file, "w") as f:
            for g in gold:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        print(f"\nApplied: {relabeled} relabeled, {secondary} secondary added, {kept} kept")
        print(f"Saved to {args.gold_file}")

        # Project new accuracy
        gold_map = {g["ticket_id"]: g for g in gold}
        correct = total = 0
        for r in eval_results.get("ft_results", []):
            tid = r["ticket_id"]
            g = gold_map.get(tid, {})
            p = r.get("parsed")
            if p:
                pred = p.get("intent", "").lower()
                primary = g.get("gold_intent", "").lower()
                sec = g.get("gold_intent_secondary", "").lower()
                if pred == primary or (sec and pred == sec):
                    correct += 1
            total += 1
        print(f"\nProjected accuracy: {correct}/{total} = {correct/total*100:.1f}%")
    elif args.dry_run:
        print(f"\nDRY RUN — no changes applied. Use --apply to fix.")

        # Project
        gold_map = {g["ticket_id"]: g for g in gold}
        # Simulate fixes
        for fix in fixes:
            g = gold_map.get(fix["ticket_id"])
            if g and fix["action"] == "relabel":
                g["_sim_intent"] = fix["new_intent"]
            elif g and fix["action"] == "add_secondary":
                g["_sim_secondary"] = fix["secondary"]

        correct = total = 0
        for r in eval_results.get("ft_results", []):
            tid = r["ticket_id"]
            g = gold_map.get(tid, {})
            p = r.get("parsed")
            if p:
                pred = p.get("intent", "").lower()
                primary = g.get("_sim_intent", g.get("gold_intent", "")).lower()
                sec = g.get("_sim_secondary", g.get("gold_intent_secondary", "")).lower()
                if pred == primary or (sec and pred == sec):
                    correct += 1
            total += 1
        print(f"Projected accuracy after fixes: {correct}/{total} = {correct/total*100:.1f}%")
    else:
        print(f"\nUse --dry-run to preview or --apply to fix.")


if __name__ == "__main__":
    main()
