#!/usr/bin/env python3
"""Script 1: GPT-4o-mini batch annotation of customer support tickets.

Samples tickets from raw data, sends to GPT-4o-mini for structured annotation,
and outputs labeled JSONL ready for fine-tuning.

Usage:
    python scripts/1_annotate.py                          # Use config defaults
    python scripts/1_annotate.py --sample-size 3           # Override sample size
    python scripts/1_annotate.py --raw-dir data/raw        # Override raw directory
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import pandas as pd
import yaml
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Optional
from tqdm import tqdm


# ── Pydantic schema for annotation output ──────────────────────────────────

class KeyEntities(BaseModel):
    order_id: Optional[str] = None
    amount: Optional[str] = None
    product: Optional[str] = None
    date: Optional[str] = None
    customer_name: Optional[str] = None
    tracking_number: Optional[str] = None
    account_id: Optional[str] = None
    other: Optional[str] = None


class TicketAnnotation(BaseModel):
    intent: str = Field(
        description="Primary customer intent",
        json_schema_extra={
            "enum": [
                "order_status", "refund_request", "billing_inquiry",
                "technical_support", "complaint", "shipping_issue",
                "cancellation", "return_request", "account_management",
                "product_inquiry", "payment_issue", "fraud_report",
                "general_inquiry",
            ]
        },
    )
    urgency: str = Field(
        description="Urgency level",
        json_schema_extra={"enum": ["critical", "high", "medium", "low"]},
    )
    workflow_steps: list[str] = Field(description="Ordered resolution steps")
    tools_required: list[str] = Field(description="Enterprise APIs needed")
    key_entities: KeyEntities = Field(description="Extracted entities")
    confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence")


# ── Config loading ──────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/poc_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Data loading ────────────────────────────────────────────────────────────

# Column name variants → canonical names
COLUMN_MAPPING = {
    "customer_message": ["customer_message", "body", "text", "message", "input", "instruction", "query", "content"],
    "agent_response": ["agent_response", "response", "answer", "reply", "output", "resolution"],
}


def detect_column(df: pd.DataFrame, canonical: str) -> str | None:
    """Auto-detect column name from known variants."""
    for variant in COLUMN_MAPPING[canonical]:
        if variant in df.columns:
            return variant
    return None


def load_raw_tickets(raw_dir: str) -> pd.DataFrame:
    """Load all raw tickets from CSV/JSONL files in the raw directory."""
    raw_path = Path(raw_dir)
    frames = []

    for file_path in sorted(raw_path.iterdir()):
        if file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
        elif file_path.suffix in (".jsonl", ".json"):
            if file_path.suffix == ".jsonl":
                df = pd.read_json(file_path, lines=True)
            else:
                df = pd.read_json(file_path)
        else:
            continue

        # Auto-detect and rename columns
        for canonical, _variants in COLUMN_MAPPING.items():
            detected = detect_column(df, canonical)
            if detected and detected != canonical:
                df = df.rename(columns={detected: canonical})

        if "customer_message" not in df.columns:
            print(f"  WARNING: Skipping {file_path.name} — no customer_message column found")
            continue

        df["source_file"] = file_path.name
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No valid ticket files found in {raw_dir}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(combined)} tickets from {len(frames)} file(s)")
    return combined


# ── Stratified sampling ─────────────────────────────────────────────────────

def stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Sample n tickets. Stratify by existing category/type labels if available."""
    if n >= len(df):
        print(f"  Sample size ({n}) >= dataset size ({len(df)}), using all tickets")
        return df.copy()

    # Look for stratification columns
    strat_col = None
    for col in ["category", "type", "intent", "label", "department"]:
        if col in df.columns:
            strat_col = col
            break

    if strat_col:
        print(f"  Stratifying by '{strat_col}' ({df[strat_col].nunique()} groups)")
        sampled = df.groupby(strat_col, group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), max(1, int(n * len(x) / len(df)))),
                               random_state=seed),
            include_groups=False,
        )
        # Adjust to exact count
        if len(sampled) < n:
            remaining = df.drop(sampled.index)
            extra = remaining.sample(n=n - len(sampled), random_state=seed)
            sampled = pd.concat([sampled, extra])
        elif len(sampled) > n:
            sampled = sampled.sample(n=n, random_state=seed)
    else:
        print("  No stratification column found, using random sample")
        sampled = df.sample(n=n, random_state=seed)

    return sampled.reset_index(drop=True)


# ── Annotation logic ────────────────────────────────────────────────────────

def build_user_prompt(record: dict) -> str:
    """Build the user message for annotation."""
    parts = [f"Customer message: {record['customer_message']}"]
    if pd.notna(record.get("agent_response")):
        parts.append(f"Agent response: {record['agent_response']}")
    return "\n".join(parts)


class AnnotationPipeline:
    def __init__(self, config: dict, system_prompt: str):
        self.client = AsyncOpenAI()
        self.config = config["annotation"]
        self.system_prompt = system_prompt
        self.semaphore = asyncio.Semaphore(self.config["max_concurrent"])
        self.stats = {"success": 0, "failed": 0, "total_tokens": 0, "start_time": None}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    async def annotate_single(self, record: dict) -> dict | None:
        """Annotate a single ticket with GPT-4o-mini structured output."""
        async with self.semaphore:
            user_prompt = build_user_prompt(record)
            response = await self.client.responses.parse(
                model=self.config["model"],
                temperature=self.config["temperature"],
                input=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=TicketAnnotation,
            )

            parsed = response.output_parsed
            self.stats["total_tokens"] += response.usage.total_tokens if response.usage else 0

            # Merge original record with annotation
            annotated = {**record}
            annotated.update(parsed.model_dump())
            return annotated

    async def annotate_batch(self, records: list[dict]) -> list[dict]:
        """Annotate a batch of tickets concurrently."""
        self.stats["start_time"] = time.time()
        results = []
        batch_size = self.config["batch_size"]

        for batch_start in range(0, len(records), batch_size):
            batch = records[batch_start : batch_start + batch_size]
            tasks = [self.annotate_single(rec) for rec in batch]

            batch_results = []
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    if result:
                        batch_results.append(result)
                        self.stats["success"] += 1
                except Exception as e:
                    self.stats["failed"] += 1
                    print(f"  ERROR: Annotation failed: {e}")

            results.extend(batch_results)

            processed = batch_start + len(batch)
            if processed % 100 == 0 or processed == len(records):
                elapsed = time.time() - self.stats["start_time"]
                rate = self.stats["success"] / elapsed if elapsed > 0 else 0
                print(f"  Progress: {processed}/{len(records)} | "
                      f"Success: {self.stats['success']} | "
                      f"Failed: {self.stats['failed']} | "
                      f"Rate: {rate:.1f} tickets/s")

        return results


# ── Summary statistics ──────────────────────────────────────────────────────

def print_summary(results: list[dict], stats: dict):
    """Print annotation summary statistics."""
    if not results:
        print("\nNo results to summarize.")
        return

    df = pd.DataFrame(results)
    elapsed = time.time() - stats["start_time"]

    # Cost estimate: gpt-4o-mini pricing ~$0.15/1M input, $0.60/1M output
    est_cost = stats["total_tokens"] * 0.3 / 1_000_000  # rough average

    print("\n" + "=" * 60)
    print("ANNOTATION SUMMARY")
    print("=" * 60)
    print(f"Total processed:  {stats['success']} success, {stats['failed']} failed")
    print(f"Processing time:  {elapsed:.1f}s ({stats['success'] / elapsed:.1f} tickets/s)")
    print(f"Total tokens:     {stats['total_tokens']:,}")
    print(f"Estimated cost:   ${est_cost:.4f}")

    print(f"\nIntent distribution:")
    for intent, count in df["intent"].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {intent:25s} {count:4d} ({pct:.1f}%)")

    print(f"\nUrgency distribution:")
    for urgency, count in df["urgency"].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {urgency:25s} {count:4d} ({pct:.1f}%)")

    print(f"\nAverage confidence: {df['confidence'].mean():.3f}")
    print(f"Min confidence:     {df['confidence'].min():.3f}")
    print("=" * 60)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Annotate customer support tickets with GPT-4o-mini")
    parser.add_argument("--config", default="configs/poc_config.yaml", help="Config file path")
    parser.add_argument("--raw-dir", default=None, help="Override raw data directory")
    parser.add_argument("--sample-size", type=int, default=None, help="Override sample size")
    parser.add_argument("--output", default="data/annotated/annotated_5k.jsonl", help="Output file path")
    args = parser.parse_args()

    config = load_config(args.config)

    raw_dir = args.raw_dir or config["data"]["raw_dir"]
    sample_size = args.sample_size or config["data"]["sample_size"]

    # Step 1: Load raw tickets
    print(f"\n[1/4] Loading raw tickets from {raw_dir}")
    df = load_raw_tickets(raw_dir)

    # Step 2: Sample
    print(f"\n[2/4] Sampling {sample_size} tickets")
    sampled = stratified_sample(df, sample_size)

    # Save sample
    sample_out = Path("data/sampled/sample_5k.jsonl")
    sample_out.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_json(sample_out, orient="records", lines=True)
    print(f"  Saved sample to {sample_out}")

    # Step 3: Annotate
    print(f"\n[3/4] Annotating {len(sampled)} tickets with GPT-4o-mini")
    system_prompt = Path("prompts/annotator_system.txt").read_text()
    records = sampled.to_dict(orient="records")

    pipeline = AnnotationPipeline(config, system_prompt)
    results = asyncio.run(pipeline.annotate_batch(records))

    # Step 4: Save results
    print(f"\n[4/4] Saving annotated data")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in results:
            f.write(json.dumps(record, default=str) + "\n")
    print(f"  Saved {len(results)} annotated tickets to {output_path}")

    # Summary
    print_summary(results, pipeline.stats)


if __name__ == "__main__":
    main()
