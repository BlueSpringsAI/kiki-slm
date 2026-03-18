#!/usr/bin/env python3
"""Enrich training data with GPT-4o-generated message-specific workflow steps.

Reads ChatML JSONL, sends each example to GPT-4o to generate contextual
workflow steps and tool selections, writes enriched JSONL back.

Uses async batched requests with semaphore rate limiting for parallel execution.

Usage:
    python scripts/enrich_workflows.py \
        --input data/formatted/kiki_sft_chatml_train.jsonl \
        --output data/formatted/kiki_sft_chatml_train_enriched.jsonl \
        --max-concurrent 30 \
        --sample 50000

    # Dry run — process 100 examples only
    python scripts/enrich_workflows.py \
        --input data/formatted/kiki_sft_chatml_train.jsonl \
        --output /dev/null \
        --sample 100 --dry-run

    # Cost estimate only
    python scripts/enrich_workflows.py \
        --input data/formatted/kiki_sft_chatml_train.jsonl \
        --estimate-cost
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Available tools (for GPT-4o context)
# ---------------------------------------------------------------------------

AVAILABLE_TOOLS = [
    "order_lookup_api", "shipment_tracking_api", "customer_profile_api",
    "refund_processing_api", "payment_gateway_api", "invoice_verification_api",
    "warranty_check_api", "ticket_update_api", "notification_service",
    "policy_engine", "vision_api", "document_verification",
]

VALID_INTENTS = [
    "order_status", "refund_request", "billing_inquiry", "technical_support",
    "complaint", "shipping_issue", "cancellation", "return_request",
    "account_management", "product_inquiry", "payment_issue", "fraud_report",
    "general_inquiry",
]

# ---------------------------------------------------------------------------
# GPT-4o enrichment prompt
# ---------------------------------------------------------------------------

ENRICHMENT_SYSTEM_PROMPT = f"""You are an expert customer service workflow designer.

Given a customer message and its classified intent, generate SPECIFIC workflow steps
and tool selections for THIS exact message. Do NOT use generic steps.

Available tools: {', '.join(AVAILABLE_TOOLS)}

Valid intents: {', '.join(VALID_INTENTS)}

Rules:
- Workflow steps must be SPECIFIC to the customer's actual problem, not generic templates
- Each step should be a concrete action (e.g., "look up order #ORD-48293" not "check order")
- Tool selections must come from the available tools list above
- Include 2-6 workflow steps depending on complexity
- Include 1-3 tools depending on what's needed
- Also provide the correct urgency: critical/high/medium/low based on message content

Respond with ONLY valid JSON:
{{"workflow_steps": ["step1", "step2", ...], "tools_required": ["tool1", "tool2", ...], "urgency": "medium"}}"""


# ---------------------------------------------------------------------------
# Async GPT-4o client with rate limiting
# ---------------------------------------------------------------------------

class GPT4oEnricher:
    """Async GPT-4o client with semaphore rate limiting and retry logic."""

    def __init__(self, max_concurrent: int = 30, max_retries: int = 3):
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.client = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.failed_requests = 0

    async def init_client(self):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI()

    async def enrich_single(self, customer_message: str, current_intent: str) -> dict | None:
        """Send one message to GPT-4o for workflow enrichment."""
        async with self.semaphore:
            user_prompt = (
                f"Customer message: {customer_message}\n"
                f"Classified intent: {current_intent}\n\n"
                f"Generate specific workflow steps and tools for this exact message."
            )

            for attempt in range(self.max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": ENRICHMENT_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.2,
                        max_tokens=300,
                    )

                    text = response.choices[0].message.content or ""
                    self.total_requests += 1
                    if response.usage:
                        self.total_input_tokens += response.usage.prompt_tokens
                        self.total_output_tokens += response.usage.completion_tokens

                    # Parse response
                    text = text.strip()
                    if text.startswith("```"):
                        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
                        text = "\n".join(lines).strip()
                    result = json.loads(text)

                    # Validate
                    steps = result.get("workflow_steps", [])
                    tools = result.get("tools_required", [])
                    urgency = result.get("urgency", "medium")

                    if not steps or not isinstance(steps, list):
                        continue

                    # Filter tools to valid ones
                    tools = [t for t in tools if t in AVAILABLE_TOOLS]

                    return {
                        "workflow_steps": steps[:6],
                        "tools_required": tools[:4],
                        "urgency": urgency if urgency in ("critical", "high", "medium", "low") else "medium",
                    }

                except json.JSONDecodeError:
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                except Exception as e:
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        wait = 2 ** (attempt + 1)
                        logger.warning("Rate limited, waiting %ds...", wait)
                        await asyncio.sleep(wait)
                        continue
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    logger.debug("Failed after %d attempts: %s", self.max_retries, e)

            self.failed_requests += 1
            return None

    async def enrich_batch(self, examples: list[dict], progress_interval: int = 500) -> list[dict | None]:
        """Enrich a batch of examples in parallel."""
        tasks = []
        for ex in examples:
            # Extract customer message and intent from the ChatML messages
            messages = ex.get("messages", [])
            customer_msg = ""
            current_intent = "general_inquiry"

            for msg in messages:
                if msg.get("role") == "user":
                    customer_msg = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    try:
                        parsed = json.loads(msg.get("content", ""))
                        current_intent = parsed.get("intent", "general_inquiry")
                    except (json.JSONDecodeError, AttributeError):
                        pass

            if customer_msg:
                tasks.append(self.enrich_single(customer_msg, current_intent))
            else:
                tasks.append(asyncio.coroutine(lambda: None)())

        # Process with progress reporting
        results = []
        total = len(tasks)
        completed = 0

        # Process in chunks to report progress
        chunk_size = min(self.max_concurrent * 2, 200)
        for chunk_start in range(0, total, chunk_size):
            chunk = tasks[chunk_start:chunk_start + chunk_size]
            chunk_results = await asyncio.gather(*chunk, return_exceptions=True)

            for r in chunk_results:
                if isinstance(r, Exception):
                    results.append(None)
                    self.failed_requests += 1
                else:
                    results.append(r)

            completed += len(chunk)
            if completed % progress_interval < chunk_size or completed == total:
                cost = self.estimate_cost()
                logger.info(
                    "  [%d/%d] (%.0f%%) | requests=%d | failed=%d | cost=$%.2f",
                    completed, total, completed / total * 100,
                    self.total_requests, self.failed_requests, cost,
                )

        return results

    def estimate_cost(self) -> float:
        """Estimate cost in USD (GPT-4o-mini pricing)."""
        # GPT-4o-mini: $0.15/1M input, $0.60/1M output
        input_cost = self.total_input_tokens * 0.15 / 1_000_000
        output_cost = self.total_output_tokens * 0.60 / 1_000_000
        return input_cost + output_cost


# ---------------------------------------------------------------------------
# Data processing
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def apply_enrichment(example: dict, enrichment: dict | None) -> dict:
    """Apply GPT-4o enrichment to a training example's assistant JSON."""
    if enrichment is None:
        return example

    messages = example.get("messages", [])
    new_messages = []

    for msg in messages:
        if msg.get("role") == "assistant":
            try:
                parsed = json.loads(msg["content"])
                # Replace static steps/tools/urgency with GPT-4o generated ones
                parsed["workflow_steps"] = enrichment["workflow_steps"]
                parsed["tools_required"] = enrichment["tools_required"]
                parsed["urgency"] = enrichment["urgency"]
                new_messages.append({"role": "assistant", "content": json.dumps(parsed, indent=2)})
            except (json.JSONDecodeError, KeyError):
                new_messages.append(msg)
        else:
            new_messages.append(msg)

    result = dict(example)
    result["messages"] = new_messages
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main(args):
    logger.info("Loading data from %s", args.input)
    examples = load_jsonl(args.input)
    logger.info("Loaded %d examples", len(examples))

    # Sample if requested
    if args.sample and args.sample < len(examples):
        random.seed(args.seed)
        examples = random.sample(examples, args.sample)
        logger.info("Sampled %d examples", len(examples))

    # Cost estimate only
    if args.estimate_cost:
        # Average ~400 input tokens + ~100 output tokens per request
        avg_input = 400
        avg_output = 100
        total = len(examples)
        input_cost = total * avg_input * 0.15 / 1_000_000
        output_cost = total * avg_output * 0.60 / 1_000_000
        total_cost = input_cost + output_cost
        logger.info("Cost estimate for %d examples:", total)
        logger.info("  Input:  %d tokens × $0.15/1M = $%.2f", total * avg_input, input_cost)
        logger.info("  Output: %d tokens × $0.60/1M = $%.2f", total * avg_output, output_cost)
        logger.info("  TOTAL:  $%.2f", total_cost)
        logger.info("  Time:   ~%.0f minutes at %d concurrent", total / args.max_concurrent / 3, args.max_concurrent)
        return

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set")
        sys.exit(1)

    # Enrich
    enricher = GPT4oEnricher(max_concurrent=args.max_concurrent, max_retries=3)
    await enricher.init_client()

    logger.info("Starting enrichment with %d concurrent requests...", args.max_concurrent)
    start = time.time()

    enrichments = await enricher.enrich_batch(examples, progress_interval=500)

    elapsed = time.time() - start
    logger.info("Enrichment complete in %.1f minutes", elapsed / 60)
    logger.info("  Requests: %d successful, %d failed", enricher.total_requests, enricher.failed_requests)
    logger.info("  Cost: $%.2f", enricher.estimate_cost())

    # Apply enrichments
    enriched_count = 0
    enriched_examples = []
    for ex, enr in zip(examples, enrichments):
        if enr is not None:
            enriched_examples.append(apply_enrichment(ex, enr))
            enriched_count += 1
        else:
            enriched_examples.append(ex)  # Keep original if enrichment failed

    logger.info("  Enriched: %d/%d examples (%.0f%%)", enriched_count, len(examples),
                enriched_count / len(examples) * 100 if examples else 0)

    # Save
    if not args.dry_run:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for ex in enriched_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        logger.info("Saved to %s (%.1f MB)", output_path, output_path.stat().st_size / 1024 / 1024)
    else:
        logger.info("DRY RUN — not saving output")
        # Show a sample
        for ex in enriched_examples[:3]:
            msgs = ex.get("messages", [])
            for m in msgs:
                if m.get("role") == "assistant":
                    try:
                        parsed = json.loads(m["content"])
                        logger.info("  Sample steps: %s", parsed.get("workflow_steps"))
                        logger.info("  Sample tools: %s", parsed.get("tools_required"))
                    except json.JSONDecodeError:
                        pass
                    break


def main():
    parser = argparse.ArgumentParser(description="Enrich training data with GPT-4o workflow steps")
    parser.add_argument("--input", required=True, help="Input ChatML JSONL")
    parser.add_argument("--output", required=True, help="Output enriched JSONL")
    parser.add_argument("--max-concurrent", type=int, default=30, help="Max concurrent API requests")
    parser.add_argument("--sample", type=int, default=None, help="Process only N random examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--estimate-cost", action="store_true", help="Print cost estimate and exit")
    parser.add_argument("--dry-run", action="store_true", help="Process but don't save output")
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
