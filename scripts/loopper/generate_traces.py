#!/usr/bin/env python3
"""Step 3: Run sampled tickets through the Loopper LangGraph agent and capture traces.

For each sampled ticket:
  1. Formats it as the agent's InputState
  2. Runs it through the full LangGraph pipeline (triage → retrieve → compose)
  3. Captures outputs at each node: classification, tool calls, RAG results, response
  4. Saves the complete trace as JSON for downstream ChatML conversion

Requires:
  - The Loopper agent installed (pip install -e Looper-Support-Agent-Server/)
  - OPENAI_API_KEY environment variable
  - RAG_MCP_URL for context retrieval (optional — traces without RAG are still useful)

Usage:
    python scripts/loopper/generate_traces.py
    python scripts/loopper/generate_traces.py --sample 100        # test run
    python scripts/loopper/generate_traces.py --concurrency 3     # parallel tickets
    python scripts/loopper/generate_traces.py --resume             # skip already-traced
    python scripts/loopper/generate_traces.py --estimate-cost      # cost estimate only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Default paths (from configs/loopper_pipeline.yaml or project-relative) ──
from scripts.loopper.config import get_default_paths as _get_paths
_PATHS = _get_paths()
SAMPLED_DIR = _PATHS["sampled_tickets"]
TRACES_DIR = _PATHS["traces"]
AGENT_SRC = _PATHS["agent_src"] or ""


def setup_agent_path():
    """Add agent source to Python path and configure environment for batch mode."""
    # Load the agent's .env file (OPENAI_API_KEY, RAG_MCP_URL, etc.).
    # The agent itself doesn't call load_dotenv — LangGraph dev does it
    # when running as a server. We must do it ourselves for direct ainvoke().
    if AGENT_SRC:
        agent_root = os.path.dirname(AGENT_SRC)  # AGENT_SRC = .../src, root = ...
        env_file = os.path.join(agent_root, ".env")
        if os.path.exists(env_file):
            try:
                from dotenv import load_dotenv
                load_dotenv(env_file, override=False)
            except ImportError:
                # Fall back to manual parsing if python-dotenv isn't available
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = value

    src_path = AGENT_SRC
    if src_path and src_path not in sys.path:
        sys.path.insert(0, src_path)

    # CRITICAL: Disable human review interrupt — otherwise ainvoke will hang
    # waiting for a human response that never comes in batch mode.
    # Must be set BEFORE importing the agent (settings are read at import time).
    os.environ["ENABLE_HUMAN_REVIEW_INTERRUPT"] = "false"

    # Suppress verbose logging from the agent during batch processing
    os.environ.setdefault("LOG_LEVEL", "WARNING")
    os.environ.setdefault("AUDIT_LOG_LEVEL", "WARNING")


def format_ticket_for_agent(ticket_data: dict) -> dict:
    """Convert raw Freshdesk ticket JSON into the agent's InputState format."""
    ticket = ticket_data.get("ticket", {})
    convs = ticket_data.get("conversations", [])

    # Ensure we have a valid timestamp (required datetime field on TicketMessage)
    default_timestamp = ticket.get("created_at") or "1970-01-01T00:00:00Z"

    messages = []

    # Initial description as first incoming message
    desc = (ticket.get("description_text", "") or "").strip()
    if desc:
        messages.append({
            "message_index": 0,
            "timestamp": default_timestamp,
            "sender_email": "",
            "recipient": "",
            "clean_body": desc[:4000],
            "direction": "incoming",
            "language": ticket.get("detected_language") or "en",
        })

    # Add public conversations
    public_convs = [c for c in convs if not c.get("private", False)]
    for i, conv in enumerate(public_convs[:8]):
        body = (conv.get("body_text", "") or "").strip()
        if not body:
            continue
        direction = "incoming" if conv.get("incoming", False) else "outgoing"
        conv_timestamp = conv.get("created_at") or default_timestamp

        messages.append({
            "message_index": i + 1,
            "timestamp": conv_timestamp,
            "sender_email": conv.get("from_email") or "",
            "recipient": "",
            "clean_body": body[:3000],
            "direction": direction,
            "language": conv.get("detected_language") or ticket.get("detected_language") or "en",
        })

    lang = ticket.get("detected_language") or "en"
    if lang.lower() == "none":
        lang = "en"

    return {
        "ticket": {
            "ticket_id": str(ticket.get("id", "unknown")),
            "messages": messages,
            "languages": [lang],
            "subject": ticket.get("subject") or "",
        }
    }


def extract_trace_from_state(final_state: dict, input_state: dict) -> dict:
    """Extract structured trace from the agent's final state.

    Returns a trace dict with triage, retrieval, response, and metadata sections.
    All enum values are already strings (agent uses use_enum_values=True).
    """
    # Determine completion status
    schema_valid = final_state.get("schema_valid", True)
    is_valid = final_state.get("is_valid")
    has_response = bool(final_state.get("response_english"))

    if not schema_valid:
        completion_status = "schema_invalid"
    elif is_valid is False:
        completion_status = "triage_rejected"
    elif has_response:
        completion_status = "full"
    else:
        completion_status = "partial"

    # Category reasoning (already a dict due to use_enum_values)
    cat_reasoning = final_state.get("category_reasoning")
    if cat_reasoning and hasattr(cat_reasoning, "model_dump"):
        cat_reasoning = cat_reasoning.model_dump()

    val_reasoning = final_state.get("validation_reasoning")
    if val_reasoning and hasattr(val_reasoning, "model_dump"):
        val_reasoning = val_reasoning.model_dump()

    action_reasoning = final_state.get("action_reasoning")
    if action_reasoning and hasattr(action_reasoning, "model_dump"):
        action_reasoning = action_reasoning.model_dump()

    resolution_reasoning = final_state.get("resolution_reasoning")
    if resolution_reasoning and hasattr(resolution_reasoning, "model_dump"):
        resolution_reasoning = resolution_reasoning.model_dump()

    # RAG context
    rag_context = final_state.get("rag_context")
    rag_data = None
    if rag_context:
        if hasattr(rag_context, "model_dump"):
            rag_data = rag_context.model_dump()
        elif isinstance(rag_context, dict):
            rag_data = rag_context

    # Extract tool calls from RAG context
    # Best source: rag_context.tool_calls (if agent was extended with Option A)
    # Fallback: infer from results' collection field + split query string
    tool_calls = []
    tool_results = []

    if rag_data:
        # Option A: structured tool_calls field (preferred)
        if rag_data.get("tool_calls"):
            for tc in rag_data["tool_calls"]:
                tool_calls.append({
                    "collection": tc.get("collection", "faq"),
                    "query": tc.get("query", ""),
                    "top_k": tc.get("top_k", 5),
                })
        else:
            # Fallback: infer from results + query string
            queries = (rag_data.get("query", "") or "").split(" | ")

            # Build collection list from results
            collections_seen = []
            for r in rag_data.get("results", []):
                c = r.get("collection", "")
                if c and c not in collections_seen:
                    collections_seen.append(c)

            for i, q in enumerate(queries):
                q = q.strip()
                if q:
                    tool_calls.append({
                        "collection": collections_seen[i] if i < len(collections_seen) else "faq",
                        "query": q,
                        "top_k": 5,
                    })

        # Collect all results
        for result in rag_data.get("results", []):
            tool_results.append({
                "text": result.get("text", ""),
                "source": result.get("source", ""),
                "score": result.get("score", 0.0),
                "rerank_score": result.get("rerank_score", 0.0),
                "collection": result.get("collection", ""),
                "chunk_index": result.get("chunk_index"),
                "doc_type": result.get("doc_type"),
            })

    return {
        "ticket_id": input_state["ticket"]["ticket_id"],
        "ticket_input": input_state,
        "completion_status": completion_status,

        "triage": {
            "is_valid": final_state.get("is_valid", True),
            "category": final_state.get("category", "other"),
            "category_confidence": final_state.get("category_confidence", 0.0),
            "category_reasoning": cat_reasoning,
            "validation_reasoning": val_reasoning,
        },

        "retrieval": {
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "query": rag_data.get("query") if rag_data else None,
            "num_results": rag_data.get("num_results", 0) if rag_data else 0,
            "cache_hit": rag_data.get("cache_hit", False) if rag_data else False,
        },

        "response": {
            "summary": final_state.get("summary"),
            "resolution_type": final_state.get("resolution_type", "requires_human_action"),
            "human_team_required": final_state.get("human_team_required", "account_manager"),
            "action_list": final_state.get("action_list"),
            "action_reasoning": action_reasoning,
            "resolution_reasoning": resolution_reasoning,
            "response_english": final_state.get("response_english"),
        },

        "translated_messages": final_state.get("translated_messages", []),

        "metadata": {
            "detected_language": final_state.get("detected_language"),
            "ticket_status": final_state.get("ticket_status"),
            "schema_valid": schema_valid,
        },
    }


async def generate_single_trace(
    filepath: Path,
    graph,
    semaphore: asyncio.Semaphore,
    output_dir: Path,
) -> dict:
    """Run one ticket through the agent and save the trace."""
    ticket_id = filepath.stem

    async with semaphore:
        try:
            with open(filepath) as f:
                ticket_data = json.load(f)

            input_state = format_ticket_for_agent(ticket_data)

            # Skip tickets with no messages
            if not input_state["ticket"]["messages"]:
                return {"ticket_id": ticket_id, "status": "skipped", "reason": "no_messages"}

            t0 = time.monotonic()
            # 5-min timeout per ticket — pilot max was 103s, so 300s is generous.
            # Prevents one stuck ticket from blocking the entire batch.
            result = await asyncio.wait_for(graph.ainvoke(input_state), timeout=300)
            elapsed_ms = (time.monotonic() - t0) * 1000

            trace = extract_trace_from_state(result, input_state)
            trace["metadata"]["elapsed_ms"] = elapsed_ms

            output_path = output_dir / f"{ticket_id}.json"
            with open(output_path, "w") as f:
                json.dump(trace, f, indent=2, ensure_ascii=False, default=str)

            return {
                "ticket_id": ticket_id,
                "status": "success",
                "completion_status": trace["completion_status"],
                "elapsed_ms": elapsed_ms,
            }

        except asyncio.TimeoutError:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.error("Ticket %s timed out after %.0fs", ticket_id, elapsed_ms / 1000)
            return {"ticket_id": ticket_id, "status": "error", "error": "timeout_300s"}
        except Exception as e:
            logger.error("Failed on ticket %s: %s", ticket_id, e)
            return {"ticket_id": ticket_id, "status": "error", "error": str(e)}


async def run_batch(args: argparse.Namespace):
    """Run all sampled tickets through the agent."""
    sampled_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(sampled_dir.glob("*.json"))
    files = [f for f in files if not f.name.startswith("_")]

    if not files:
        logger.error("No ticket files found in %s", sampled_dir)
        sys.exit(1)

    logger.info("Found %d sampled tickets", len(files))

    if args.sample:
        import random
        random.seed(42)
        files = random.sample(files, min(args.sample, len(files)))
        logger.info("Processing %d tickets (sample mode)", len(files))

    if args.estimate_cost:
        cost = len(files) * 0.03
        logger.info("Estimated cost: $%.2f for %d tickets", cost, len(files))
        logger.info("Estimated time: %.0f minutes at concurrency=%d",
                     len(files) * 10 / args.concurrency / 60, args.concurrency)
        return

    if args.resume:
        already_done = {f.stem for f in output_dir.glob("*.json") if not f.name.startswith("_")}
        before = len(files)
        files = [f for f in files if f.stem not in already_done]
        logger.info("Resume: %d already done, %d remaining", before - len(files), len(files))

    if not files:
        logger.info("Nothing to process!")
        return

    # Import the agent (env vars already set by setup_agent_path)
    setup_agent_path()
    try:
        from react_agent.graph import graph
        logger.info("Loopper agent imported successfully")
    except ImportError as e:
        logger.error(
            "Cannot import Loopper agent. Install with:\n"
            "  cd %s && pip install -e .\n"
            "Error: %s", AGENT_SRC, e
        )
        sys.exit(1)

    semaphore = asyncio.Semaphore(args.concurrency)

    batch_size = 50
    total_batches = (len(files) + batch_size - 1) // batch_size
    results = []
    start_time = time.time()
    completion_counts = {"full": 0, "triage_rejected": 0, "schema_invalid": 0, "partial": 0}

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(files))
        batch_files = files[batch_start:batch_end]

        batch_results = await asyncio.gather(*[
            generate_single_trace(f, graph, semaphore, output_dir)
            for f in batch_files
        ])
        results.extend(batch_results)

        for r in batch_results:
            cs = r.get("completion_status")
            if cs in completion_counts:
                completion_counts[cs] += 1

        success = sum(1 for r in results if r["status"] == "success")
        errors = sum(1 for r in results if r["status"] == "error")
        elapsed = time.time() - start_time
        rate = success / elapsed if elapsed > 0 else 0
        eta = (len(files) - len(results)) / rate / 60 if rate > 0 else 0

        logger.info(
            "Batch %d/%d: %d success, %d errors (%.1f/sec, ETA %.0fm)",
            batch_idx + 1, total_batches, success, errors, rate, eta,
        )

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    errors = sum(1 for r in results if r["status"] == "error")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    elapsed = time.time() - start_time
    avg_ms = sum(r.get("elapsed_ms", 0) for r in results if r["status"] == "success") / max(success, 1)

    logger.info("")
    logger.info("=" * 60)
    logger.info("  TRACE GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info("  Success:        %d", success)
    logger.info("  Errors:         %d", errors)
    logger.info("  Skipped:        %d", skipped)
    logger.info("  Total time:     %.1f minutes", elapsed / 60)
    logger.info("  Avg per ticket: %.0f ms", avg_ms)
    logger.info("")
    logger.info("  Completion status:")
    for status, count in completion_counts.items():
        logger.info("    %-20s %5d", status, count)
    logger.info("")
    logger.info("  Output: %s", output_dir)

    # Save results log
    log_path = output_dir / "_trace_log.jsonl"
    with open(log_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    logger.info("  Log: %s", log_path)

    if errors:
        error_log = output_dir / "_errors.jsonl"
        with open(error_log, "w") as f:
            for r in results:
                if r["status"] == "error":
                    f.write(json.dumps(r) + "\n")
        logger.info("  Error log: %s", error_log)


def main():
    parser = argparse.ArgumentParser(description="Generate agent traces from sampled tickets")
    parser.add_argument("--input-dir", default=SAMPLED_DIR)
    parser.add_argument("--output-dir", default=TRACES_DIR)
    parser.add_argument("--concurrency", type=int, default=3, help="Parallel tickets")
    parser.add_argument("--sample", type=int, default=None, help="Process only N tickets")
    parser.add_argument("--resume", action="store_true", help="Skip already-traced tickets")
    parser.add_argument("--estimate-cost", action="store_true", help="Print cost estimate only")
    args = parser.parse_args()

    asyncio.run(run_batch(args))


if __name__ == "__main__":
    main()
