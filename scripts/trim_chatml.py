#!/usr/bin/env python3
"""Trim bloated fields from ChatML training data to reduce token count.

Removes unnecessary retrieval metadata from tool results (score, rerank_score,
chunk_index, doc_type, source) and optionally truncates long tool result texts.

Usage:
    python scripts/trim_chatml.py --input data/sft-data/chatml/train.jsonl --output data/sft-data/chatml/train_trimmed.jsonl
    python scripts/trim_chatml.py --input data/sft-data/chatml/eval.jsonl --output data/sft-data/chatml/eval_trimmed.jsonl
    python scripts/trim_chatml.py --input data/sft-data/chatml/train.jsonl --output data/sft-data/chatml/train_trimmed.jsonl --max-result-chars 800
    python scripts/trim_chatml.py --input data/sft-data/chatml/train.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# Fields to KEEP in tool results (model actually reads these)
KEEP_FIELDS = {"text", "collection"}

# Fields to REMOVE (retrieval metadata, useless for training)
# score, rerank_score, chunk_index, doc_type, source


def trim_tool_result(content: str, max_result_chars: int, max_results: int) -> str:
    """Trim a tool result message content, removing bloat fields."""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return content

    results = parsed.get("results", [])

    trimmed_results = []
    for r in results[:max_results]:
        trimmed = {}
        for field in KEEP_FIELDS:
            if field in r:
                val = r[field]
                if field == "text" and isinstance(val, str) and len(val) > max_result_chars:
                    val = val[:max_result_chars] + "..."
                trimmed[field] = val
        trimmed_results.append(trimmed)

    parsed["results"] = trimmed_results
    return json.dumps(parsed, ensure_ascii=False)


def trim_example(example: dict, max_result_chars: int, max_results: int) -> dict:
    """Trim a single training example."""
    trimmed = dict(example)
    new_messages = []

    for msg in trimmed.get("messages", []):
        if msg.get("role") == "tool" and msg.get("content"):
            new_msg = dict(msg)
            new_msg["content"] = trim_tool_result(msg["content"], max_result_chars, max_results)
            new_messages.append(new_msg)
        else:
            new_messages.append(msg)

    trimmed["messages"] = new_messages
    return trimmed


def estimate_tokens(example: dict) -> int:
    """Rough token estimate: 1 token per 3.5 chars."""
    total_chars = 0
    for m in example.get("messages", []):
        total_chars += len(m.get("content") or "")
        total_chars += len(m.get("reasoning_content") or "")
    if example.get("tools"):
        total_chars += len(json.dumps(example["tools"]))
    return int(total_chars / 3.5)


def main():
    parser = argparse.ArgumentParser(description="Trim ChatML training data to reduce token count")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", default=None, help="Output JSONL file (default: overwrite input)")
    parser.add_argument("--max-result-chars", type=int, default=800, help="Max chars per tool result text (default: 800)")
    parser.add_argument("--max-results", type=int, default=5, help="Max results per tool call (default: 5)")
    parser.add_argument("--dry-run", action="store_true", help="Show stats without writing")
    args = parser.parse_args()

    # Load
    examples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"Loaded {len(examples):,} examples from {args.input}")

    # Before stats
    before_tokens = [estimate_tokens(ex) for ex in examples]
    before_size = os.path.getsize(args.input)

    print(f"\nBEFORE TRIMMING:")
    print(f"  File size: {before_size / 1024 / 1024:.1f} MB")
    print(f"  Avg tokens: {sum(before_tokens) / len(before_tokens):,.0f}")
    print(f"  Median tokens: {sorted(before_tokens)[len(before_tokens) // 2]:,.0f}")
    print(f"  Fits in 4096: {sum(1 for t in before_tokens if t <= 4096):,} ({sum(1 for t in before_tokens if t <= 4096) / len(examples) * 100:.1f}%)")
    print(f"  Fits in 8192: {sum(1 for t in before_tokens if t <= 8192):,} ({sum(1 for t in before_tokens if t <= 8192) / len(examples) * 100:.1f}%)")

    # Trim
    trimmed = [trim_example(ex, args.max_result_chars, args.max_results) for ex in examples]

    # After stats
    after_tokens = [estimate_tokens(ex) for ex in trimmed]

    print(f"\nAFTER TRIMMING (max_result_chars={args.max_result_chars}, max_results={args.max_results}):")
    print(f"  Avg tokens: {sum(after_tokens) / len(after_tokens):,.0f}")
    print(f"  Median tokens: {sorted(after_tokens)[len(after_tokens) // 2]:,.0f}")
    print(f"  Fits in 4096: {sum(1 for t in after_tokens if t <= 4096):,} ({sum(1 for t in after_tokens if t <= 4096) / len(trimmed) * 100:.1f}%)")
    print(f"  Fits in 8192: {sum(1 for t in after_tokens if t <= 8192):,} ({sum(1 for t in after_tokens if t <= 8192) / len(trimmed) * 100:.1f}%)")

    saved_tokens = sum(before_tokens) - sum(after_tokens)
    print(f"\n  Tokens saved: {saved_tokens:,} ({saved_tokens / sum(before_tokens) * 100:.1f}%)")

    if args.dry_run:
        print("\nDRY RUN — no files written")
        return

    # Save
    output_path = args.output or args.input
    with open(output_path, "w") as f:
        for ex in trimmed:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    after_size = os.path.getsize(output_path)
    print(f"\n  Saved to: {output_path}")
    print(f"  File size: {before_size / 1024 / 1024:.1f} MB → {after_size / 1024 / 1024:.1f} MB ({(before_size - after_size) / 1024 / 1024:.1f} MB saved)")


if __name__ == "__main__":
    main()
