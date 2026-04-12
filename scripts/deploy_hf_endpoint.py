#!/usr/bin/env python3
"""Deploy / manage Kiki SLM on HuggingFace Inference Endpoints.

Creates a dedicated GPU endpoint running TGI, waits for it to go live,
tests it with a tool-calling request, and prints the env vars you need
to paste into your Loopper agent ECS task definition.

Usage:
    # First time — create endpoint
    python scripts/deploy_hf_endpoint.py create \
        --repo-id bluespringsAI/qiki-v0 \
        --name qiki-v0 \
        --token hf_YOUR_TOKEN

    # Check status
    python scripts/deploy_hf_endpoint.py status --name qiki-v0 --token hf_...

    # Pause (stop billing)
    python scripts/deploy_hf_endpoint.py pause --name qiki-v0 --token hf_...

    # Resume (from pause)
    python scripts/deploy_hf_endpoint.py resume --name qiki-v0 --token hf_...

    # Scale to zero (auto-resumes on first request, cold start ~30-60s)
    python scripts/deploy_hf_endpoint.py scale-to-zero --name qiki-v0 --token hf_...

    # Delete (irreversible)
    python scripts/deploy_hf_endpoint.py delete --name qiki-v0 --token hf_...

    # Test the endpoint with a tool-calling request
    python scripts/deploy_hf_endpoint.py test --name qiki-v0 --token hf_...
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import httpx


def _log(msg: str) -> None:
    print(f"── [HF DEPLOY] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Endpoint creation
# ---------------------------------------------------------------------------

def cmd_create(args: argparse.Namespace) -> None:
    from huggingface_hub import create_inference_endpoint, get_inference_endpoint

    # Check if endpoint already exists
    try:
        existing = get_inference_endpoint(args.name, namespace=args.namespace, token=args.token)
        _log(f"endpoint '{args.name}' already exists (status: {existing.status})")
        if existing.status in ("paused", "scaledToZero"):
            _log("resuming existing endpoint...")
            existing.resume()
            _log("waiting for endpoint to become running...")
            existing.wait(timeout=600)
            _print_endpoint_info(existing, args.token)
            return
        elif existing.status == "running":
            _print_endpoint_info(existing, args.token)
            return
        else:
            _log(f"endpoint is in state '{existing.status}', waiting...")
            existing.wait(timeout=600)
            _print_endpoint_info(existing, args.token)
            return
    except Exception:
        pass  # Doesn't exist yet — create it

    _log(f"creating endpoint '{args.name}'...")
    _log(f"  repo:     {args.repo_id}")
    _log(f"  instance: {args.instance_type} ({args.instance_size})")
    _log(f"  region:   {args.vendor}/{args.region}")
    _log(f"  type:     {args.endpoint_type}")

    # Build custom_image config for TGI with tool-calling support
    custom_image = {
        "health_route": "/health",
        "env": {
            "MAX_INPUT_LENGTH": "3072",
            "MAX_TOTAL_TOKENS": "4096",
            "MAX_BATCH_PREFILL_TOKENS": "4096",
            "MODEL_ID": "/repository",
        },
        "url": "ghcr.io/huggingface/text-generation-inference:latest",
    }

    endpoint = create_inference_endpoint(
        name=args.name,
        repository=args.repo_id,
        namespace=args.namespace,
        framework="pytorch",
        task="text-generation",
        accelerator="gpu",
        vendor=args.vendor,
        region=args.region,
        type=args.endpoint_type,
        instance_size=args.instance_size,
        instance_type=args.instance_type,
        custom_image=custom_image,
        token=args.token,
    )

    _log(f"endpoint created: {endpoint.name} (status: {endpoint.status})")
    _log("waiting for endpoint to become running (this takes 3-10 min)...")

    try:
        endpoint.wait(timeout=600)
    except Exception as e:
        _log(f"timeout waiting for endpoint: {e}")
        _log(f"check status at: https://ui.endpoints.huggingface.co")
        sys.exit(1)

    _print_endpoint_info(endpoint, args.token)


# ---------------------------------------------------------------------------
# Status / lifecycle
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> None:
    from huggingface_hub import get_inference_endpoint
    ep = get_inference_endpoint(args.name, namespace=args.namespace, token=args.token)
    ep.fetch()
    _print_endpoint_info(ep, args.token)


def cmd_pause(args: argparse.Namespace) -> None:
    from huggingface_hub import get_inference_endpoint
    ep = get_inference_endpoint(args.name, namespace=args.namespace, token=args.token)
    ep.pause()
    _log(f"endpoint '{args.name}' paused. Billing stopped.")
    _log("to resume: python scripts/deploy_hf_endpoint.py resume ...")


def cmd_resume(args: argparse.Namespace) -> None:
    from huggingface_hub import get_inference_endpoint
    ep = get_inference_endpoint(args.name, namespace=args.namespace, token=args.token)
    _log(f"resuming '{args.name}'...")
    ep.resume()
    ep.wait(timeout=600)
    _print_endpoint_info(ep, args.token)


def cmd_scale_to_zero(args: argparse.Namespace) -> None:
    from huggingface_hub import get_inference_endpoint
    ep = get_inference_endpoint(args.name, namespace=args.namespace, token=args.token)
    ep.scale_to_zero()
    _log(f"endpoint '{args.name}' scaled to zero. Will auto-resume on first request (~30-60s cold start).")
    if ep.url:
        _log(f"URL still valid: {ep.url}")


def cmd_delete(args: argparse.Namespace) -> None:
    from huggingface_hub import get_inference_endpoint
    ep = get_inference_endpoint(args.name, namespace=args.namespace, token=args.token)
    _log(f"DELETING endpoint '{args.name}' — this is irreversible!")
    confirm = input("Type 'yes' to confirm: ")
    if confirm.strip().lower() != "yes":
        _log("aborted.")
        return
    ep.delete()
    _log(f"endpoint '{args.name}' deleted.")


# ---------------------------------------------------------------------------
# Test with a tool-calling request
# ---------------------------------------------------------------------------

def cmd_test(args: argparse.Namespace) -> None:
    from huggingface_hub import get_inference_endpoint
    ep = get_inference_endpoint(args.name, namespace=args.namespace, token=args.token)
    ep.fetch()

    if ep.status != "running":
        _log(f"endpoint is '{ep.status}', not 'running'. Cannot test.")
        if ep.status in ("paused", "scaledToZero"):
            _log("run: python scripts/deploy_hf_endpoint.py resume ...")
        sys.exit(1)

    url = ep.url.rstrip("/")
    _log(f"testing endpoint at {url}")

    # Discover the model name
    _log("checking /v1/models...")
    with httpx.Client(timeout=30.0) as client:
        r = client.get(
            f"{url}/v1/models",
            headers={"Authorization": f"Bearer {args.token}"},
        )
        r.raise_for_status()
        models_data = r.json()
        model_ids = [m.get("id", "") for m in models_data.get("data", [])]
        model_name = model_ids[0] if model_ids else args.repo_id
        _log(f"model name: {model_name}")
        _log(f"available models: {model_ids}")

    # Send a tool-calling request
    _log("sending tool-calling test request...")
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a Loopper support agent. "
                    "You MUST call rag_search before responding."
                ),
            },
            {
                "role": "user",
                "content": "My order has been delayed by 3 days, can you help?",
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "rag_search",
                    "description": "Search the Loopper knowledge base",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "collection": {
                                "type": "string",
                                "enum": ["faq", "operations", "communication_guidelines", "supplier_data"],
                            },
                            "query": {"type": "string"},
                        },
                        "required": ["collection", "query"],
                    },
                },
            },
        ],
        "temperature": 0.1,
        "max_tokens": 512,
        "stream": False,
    }

    t0 = time.perf_counter()
    with httpx.Client(timeout=120.0) as client:
        r = client.post(
            f"{url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {args.token}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    latency = time.perf_counter() - t0

    if r.status_code != 200:
        _log(f"ERROR: {r.status_code} — {r.text[:500]}")
        sys.exit(1)

    data = r.json()
    msg = data.get("choices", [{}])[0].get("message", {})
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []
    usage = data.get("usage", {})

    _log(f"response received in {latency:.1f}s")
    _log(f"  content length:  {len(content)} chars")
    _log(f"  tool_calls:      {len(tool_calls)}")
    _log(f"  usage:           {usage}")

    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function", {})
            _log(f"  tool: {fn.get('name')}({fn.get('arguments')})")
        _log("")
        _log("  ✓ Tool calling works! The model emitted structured tool_calls.")
    elif "<tool_call>" in content:
        _log("")
        _log("  ⚠ Model emitted <tool_call> as raw text, but TGI didn't parse it.")
        _log("  The tool_call_parser may not be configured for Qwen3.")
        _log("  Try updating the custom_image env with TOOL_CALL_PARSER=hermes")
    else:
        _log(f"  content preview: {content[:300]}")
        _log("")
        _log("  ⚠ No tool_calls in response. Model may have skipped tool use.")
        _log("  This could be normal for a simple test prompt.")

    # Print the env vars
    _log("")
    _log("=" * 60)
    _log("AGENT ENV VARS (paste into ECS task definition):")
    _log("=" * 60)
    print(f"USE_KIKI_SLM=true")
    print(f"KIKI_SLM_URL={url}")
    print(f"KIKI_SLM_MODEL={model_name}")
    print(f"KIKI_SLM_API_KEY={args.token}")
    print(f"KIKI_SLM_MAX_TURNS=4")
    print(f"KIKI_SLM_TIMEOUT_S=300")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _print_endpoint_info(ep, token: str) -> None:
    _log("")
    _log(f"  name:     {ep.name}")
    _log(f"  repo:     {ep.repository}")
    _log(f"  status:   {ep.status}")
    _log(f"  url:      {ep.url or '(not yet available)'}")
    _log(f"  created:  {getattr(ep, 'created_at', '?')}")

    if ep.status == "running" and ep.url:
        _log("")
        _log("=" * 60)
        _log("AGENT ENV VARS (paste into ECS task definition):")
        _log("=" * 60)
        print(f"USE_KIKI_SLM=true")
        print(f"KIKI_SLM_URL={ep.url}")
        print(f"KIKI_SLM_MODEL={ep.repository}")
        print(f"KIKI_SLM_API_KEY={token}")
        print(f"KIKI_SLM_MAX_TURNS=4")
        print(f"KIKI_SLM_TIMEOUT_S=300")
        _log("")
        _log(f"test it: python scripts/deploy_hf_endpoint.py test --name {ep.name} --token {token[:10]}...")
    elif ep.status == "running":
        _log("  (running but URL not yet available — fetch again in a moment)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy Kiki SLM on HF Inference Endpoints")
    parser.add_argument("--token", required=True, help="HuggingFace token with write access")
    parser.add_argument("--namespace", default=None, help="HF org namespace (default: your user)")
    sub = parser.add_subparsers(dest="command", required=True)

    # create
    p_create = sub.add_parser("create", help="Create (or resume) an inference endpoint")
    p_create.add_argument("--repo-id", required=True, help="HF Hub model repo (e.g. bluespringsAI/qiki-v0)")
    p_create.add_argument("--name", required=True, help="Endpoint name (e.g. qiki-v0)")
    p_create.add_argument("--vendor", default="aws", choices=["aws", "gcp", "azure"])
    p_create.add_argument("--region", default="us-east-1", help="Cloud region")
    p_create.add_argument("--instance-type", default="nvidia-t4", help="GPU type (nvidia-t4, nvidia-l4, nvidia-a10g)")
    p_create.add_argument("--instance-size", default="x1", help="Instance size (x1, x2, x4)")
    p_create.add_argument("--endpoint-type", default="protected", choices=["protected", "private", "public"])

    # status / lifecycle
    for cmd_name in ("status", "pause", "resume", "scale-to-zero", "delete", "test"):
        p = sub.add_parser(cmd_name)
        p.add_argument("--name", required=True, help="Endpoint name")

    args = parser.parse_args()

    cmd_map = {
        "create": cmd_create,
        "status": cmd_status,
        "pause": cmd_pause,
        "resume": cmd_resume,
        "scale-to-zero": cmd_scale_to_zero,
        "delete": cmd_delete,
        "test": cmd_test,
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
