#!/usr/bin/env python3
"""Local smoke test for the Kiki SLM — multi-turn tool loop against Ollama.

Loads real tickets from gold_100.jsonl, runs the same multi-turn tool loop the
agent runs in production, prints every turn, and compares the final JSON to
the gold labels. No dependency on the Loopper agent repo — this is the
fastest way to validate a freshly-loaded Ollama model end-to-end.

The ``rag_search`` tool is mocked with empty results by default — the goal is
to prove the SLM template + tool loop works, not the RAG quality. Use
``--rag-url`` to point at a real MCP endpoint.

Usage:
    # Quick check — 3 tickets, fake (empty) RAG results
    uv run python scripts/test_kiki_local.py

    # More tickets + verbose trajectory
    uv run python scripts/test_kiki_local.py --limit 10 --verbose

    # Point at a non-default Ollama
    uv run python scripts/test_kiki_local.py --url http://localhost:11434 --model kiki-sft-v1

    # Use real RAG results (if you have the MCP server running)
    uv run python scripts/test_kiki_local.py --rag-url http://127.0.0.1:8010/mcp
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx


# ---------------------------------------------------------------------------
# System prompt + tool schema — MUST match training data verbatim
# (same constants as src/react_agent/client/kiki_client.py in the agent repo)
# ---------------------------------------------------------------------------

KIKI_SYSTEM_PROMPT = (
    "You are a Loopper support agent for B2B promotional products "
    "(mugs, pens, bags, t-shirts, caps, water bottles, notebooks, lanyards, "
    "USB sticks, power banks, etc.). Headquartered in Amsterdam, serving Europe "
    "for 24+ years. 6,000+ customizable products.\n\n"
    "You MUST reason before every action and before your final response.\n"
    "You MUST call rag_search at least once before generating your final response "
    "(exception: simple acknowledgments like \"OK thank you\" that need no context).\n"
    "You MUST ground your response in retrieved context — never invent policies, "
    "timelines, prices, or delivery dates.\n"
    "When retrieved context is empty or irrelevant, acknowledge the gap and "
    "escalate to the human reviewer rather than guessing.\n\n"
    "Preserve all PII tokens exactly as they appear (e.g. [NAME], [EMAIL], [ORG], [ADDRESS]).\n"
    "Sign all responses as Marc Logier, Account Manager — Loopper.\n\n"
    "Output your final response as a single JSON object with these fields:\n"
    "intent, urgency, confidence, is_valid, rejection_type, resolution_type, "
    "team, actions, summary, reasoning, response"
)

KIKI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": (
                "Search the Loopper knowledge base for policies, processes, "
                "and tone guidance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "enum": [
                            "faq", "operations",
                            "communication_guidelines", "supplier_data",
                        ],
                    },
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["collection", "query"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _extract_balanced_json(text: str) -> dict | None:
    first = text.find("{")
    if first == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(first, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[first:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def parse_final_json(raw: str) -> dict | None:
    text = _THINK_RE.sub("", raw or "").strip()
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.split("\n") if not line.strip().startswith("```")
        ).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _extract_balanced_json(text)


# ---------------------------------------------------------------------------
# Fake RAG — returns empty results
# ---------------------------------------------------------------------------

def fake_rag_search(args: dict) -> dict:
    """Return empty results. The SLM was trained to handle this gracefully."""
    return {"results": [], "note": "smoke-test mode — real RAG not wired up"}


# ---------------------------------------------------------------------------
# Multi-turn invoke
# ---------------------------------------------------------------------------

def invoke_kiki(
    client: httpx.Client,
    endpoint: str,
    model: str,
    ticket_text: str,
    max_turns: int = 4,
    verbose: bool = False,
) -> tuple[dict | None, list[dict]]:
    """Run the multi-turn tool loop against Ollama. Returns (parsed_json, trajectory).

    trajectory is a list of per-turn dicts for debugging / printing.
    """
    messages = [
        {"role": "system", "content": KIKI_SYSTEM_PROMPT},
        {"role": "user", "content": ticket_text},
    ]
    trajectory: list[dict] = []

    for turn in range(max_turns):
        payload = {
            "model": model,
            "messages": messages,
            "tools": KIKI_TOOLS,
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 4096, "num_predict": 1024},
        }
        t0 = time.perf_counter()
        try:
            resp = client.post(endpoint, json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return None, trajectory + [{"turn": turn + 1, "error": str(e)}]
        latency = time.perf_counter() - t0

        data = resp.json()
        msg = data.get("message") or {}
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []

        trajectory.append({
            "turn": turn + 1,
            "latency_s": round(latency, 2),
            "content": content,
            "tool_calls": tool_calls,
        })

        if verbose:
            print(f"  ── turn {turn + 1} ({latency:.1f}s) ──")
            if content:
                preview = content if len(content) < 400 else content[:400] + "…"
                print(f"    content: {preview}")
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    print(f"    tool: {fn.get('name')}({fn.get('arguments')})")

        if not tool_calls:
            parsed = parse_final_json(content)
            return parsed, trajectory

        # Append assistant + fake tool results, loop
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        })
        for tc in tool_calls:
            fn = tc.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            result = fake_rag_search(args)
            messages.append({
                "role": "tool",
                "content": json.dumps(result, ensure_ascii=False),
            })

    return None, trajectory


# ---------------------------------------------------------------------------
# Ticket formatter — matches training data shape
# ---------------------------------------------------------------------------

def format_ticket(gold_entry: dict) -> str:
    """gold_100.jsonl entries already have customer_message in the right shape."""
    return gold_entry.get("customer_message", "")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

GOLD_FIELDS = [
    ("gold_intent", "intent"),
    ("gold_urgency", "urgency"),
    ("gold_is_valid", "is_valid"),
    ("gold_rejection_type", "rejection_type"),
    ("gold_resolution_type", "resolution_type"),
    ("gold_team", "team"),
]


def compare_to_gold(parsed: dict, gold: dict) -> list[tuple[str, Any, Any, bool]]:
    rows = []
    for gk, pk in GOLD_FIELDS:
        gv = gold.get(gk)
        pv = parsed.get(pk)
        # Normalize for comparison
        def _n(x):
            if x is None:
                return None
            if isinstance(x, bool):
                return x
            return str(x).strip().lower()
        ok = _n(gv) == _n(pv)
        rows.append((pk, gv, pv, ok))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:11434",
                        help="Ollama base URL")
    parser.add_argument("--model", default="kiki-sft-v1",
                        help="Ollama model name (ollama list)")
    parser.add_argument("--gold-file",
                        default="data/sft-data/gold/gold_100.jsonl",
                        help="Path to gold_100.jsonl")
    parser.add_argument("--limit", type=int, default=3,
                        help="Number of gold tickets to run")
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--verbose", action="store_true",
                        help="Print every turn's output")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    gold_path = Path(args.gold_file)
    if not gold_path.exists():
        print(f"ERROR: gold file not found: {gold_path}", file=sys.stderr)
        sys.exit(1)

    # Load N gold tickets
    tickets: list[dict] = []
    with open(gold_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tickets.append(json.loads(line))
            if len(tickets) >= args.limit:
                break

    endpoint = f"{args.url.rstrip('/')}/api/chat"

    print(f"{'=' * 70}")
    print(f"  KIKI SMOKE TEST")
    print(f"{'=' * 70}")
    print(f"  URL:    {endpoint}")
    print(f"  Model:  {args.model}")
    print(f"  Gold:   {args.gold_file} ({len(tickets)} tickets)")
    print()

    # Check Ollama is reachable + model exists
    with httpx.Client(timeout=10.0) as client:
        try:
            r = client.get(f"{args.url.rstrip('/')}/api/tags")
            r.raise_for_status()
        except httpx.HTTPError as e:
            print(f"ERROR: can't reach Ollama at {args.url}: {e}")
            sys.exit(2)
        tags = r.json().get("models", [])
        tag_names = [t.get("name", "") for t in tags]
        if not any(args.model in name for name in tag_names):
            print(f"ERROR: model '{args.model}' not found in Ollama")
            print(f"  Available: {tag_names}")
            print(f"  Run: ollama create {args.model} -f Modelfile")
            sys.exit(3)
        print(f"  ✓ Ollama reachable, model '{args.model}' loaded")
        print()

    # Run
    totals = {"tickets": 0, "parse_ok": 0, "intent_ok": 0, "is_valid_ok": 0, "total_turns": 0, "total_latency": 0.0}

    with httpx.Client(timeout=args.timeout) as client:
        for i, ticket in enumerate(tickets):
            tid = ticket.get("ticket_id", f"ticket-{i + 1}")
            gold_intent = ticket.get("gold_intent", "?")
            ticket_text = format_ticket(ticket)

            print(f"{'─' * 70}")
            print(f"  [{i + 1}/{len(tickets)}] {tid} | gold: {gold_intent}")
            print(f"{'─' * 70}")
            print(f"  ticket preview: {ticket_text[:200].replace(chr(10), ' ')}…")
            print()

            t0 = time.perf_counter()
            parsed, trajectory = invoke_kiki(
                client, endpoint, args.model, ticket_text,
                max_turns=args.max_turns, verbose=args.verbose,
            )
            total_latency = time.perf_counter() - t0

            totals["tickets"] += 1
            totals["total_turns"] += len(trajectory)
            totals["total_latency"] += total_latency

            print()
            if parsed is None:
                print(f"  ✗ PARSE FAIL after {len(trajectory)} turns ({total_latency:.1f}s)")
                last = trajectory[-1] if trajectory else {}
                if "error" in last:
                    print(f"    error: {last['error']}")
                elif last.get("tool_calls"):
                    print(f"    stopped waiting for tool result (loop exceeded max_turns)")
                elif last.get("content"):
                    print(f"    last content: {last['content'][:300]}")
                continue

            totals["parse_ok"] += 1
            print(f"  ✓ parsed OK ({len(trajectory)} turns, {total_latency:.1f}s)")
            print(f"    intent:          {parsed.get('intent')}")
            print(f"    urgency:         {parsed.get('urgency')}")
            print(f"    is_valid:        {parsed.get('is_valid')}")
            print(f"    resolution_type: {parsed.get('resolution_type')}")
            print(f"    team:            {parsed.get('team')}")
            print(f"    actions:         {len(parsed.get('actions', []))} items")
            response_text = str(parsed.get("response") or "")
            print(f"    response:        {response_text[:120]}…")
            print()

            print(f"    Compare to gold:")
            rows = compare_to_gold(parsed, ticket)
            for field, gv, pv, ok in rows:
                mark = "✓" if ok else "✗"
                print(f"      {mark} {field:18s}  gold={gv!r:30s} pred={pv!r}")
                if field == "intent" and ok:
                    totals["intent_ok"] += 1
                if field == "is_valid" and ok:
                    totals["is_valid_ok"] += 1
            print()

    # Summary
    n = totals["tickets"]
    print(f"{'=' * 70}")
    print(f"  SUMMARY ({n} tickets)")
    print(f"{'=' * 70}")
    print(f"  parse rate:       {totals['parse_ok']}/{n} ({100 * totals['parse_ok'] / max(1, n):.0f}%)")
    print(f"  intent accuracy:  {totals['intent_ok']}/{n} ({100 * totals['intent_ok'] / max(1, n):.0f}%)")
    print(f"  is_valid correct: {totals['is_valid_ok']}/{n} ({100 * totals['is_valid_ok'] / max(1, n):.0f}%)")
    print(f"  avg turns/ticket: {totals['total_turns'] / max(1, n):.2f}")
    print(f"  avg latency:      {totals['total_latency'] / max(1, n):.1f}s")

    if totals["parse_ok"] == 0:
        print()
        print("  ⚠ NO TICKETS PARSED. Likely causes:")
        print("    - GGUF chat template didn't survive conversion (Ollama emits raw <tool_call>)")
        print("    - Wrong system prompt / tool schema (must match training)")
        print("    - Model loaded from the wrong checkpoint")
        sys.exit(4)


if __name__ == "__main__":
    main()
