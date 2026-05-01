"""Deploy Kiki SLM on Modal.com with vLLM — OpenAI-compatible /v1/chat/completions.

This is the ENTIRE deployment. No Docker, no ECS, no IAM, no image builds.
Modal handles the container, GPU scheduling, and autoscaling. Scales to zero
when idle — you only pay for actual inference time.

Setup (one-time):
    pip install modal
    modal setup                        # creates account + auth
    modal secret create huggingface-secret HF_TOKEN=hf_YOUR_TOKEN

Deploy:
    modal deploy scripts/modal_deploy.py

Test:
    curl https://bluespringsai--qiki-v0-serve.modal.run/v1/chat/completions \\
      -H "Authorization: Bearer YOUR_MODAL_TOKEN" \\
      -H "Content-Type: application/json" \\
      -d '{"model":"kiki-sft-v1","messages":[{"role":"user","content":"hello"}],"max_tokens":50}'

Stop (scales to zero automatically — or tear down completely):
    modal app stop qiki-v0
"""

import os
import subprocess

import modal

# ---------------------------------------------------------------------------
# Modal app + persistent volume for model weights cache
# ---------------------------------------------------------------------------

app = modal.App("qiki-v0")

# Cache the 7.5 GB model so it's not re-downloaded on every cold start
model_volume = modal.Volume.from_name("qiki-model-cache", create_if_missing=True)

# Container image with vLLM + deps
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .run_commands(
        "pip install uv",
        "uv pip install --system vllm>=0.8.0 huggingface_hub hf_transfer",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "CUDA_HOME": "/usr/local/cuda",
    })
)

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

MODEL_REPO = "bluespringsAI/qiki-v0"       # private HF Hub repo
SERVED_NAME = "kiki-sft-v1"                 # model name in /v1/chat/completions
MAX_MODEL_LEN = 4096                        # matches training
GPU_TYPE = "A10G"                           # A10G 24GB — T4 (16GB) OOMs with fp16 + vLLM CUDA graphs


# ---------------------------------------------------------------------------
# Pre-download model weights into the cached volume
# ---------------------------------------------------------------------------

@app.function(
    image=vllm_image,
    volumes={"/model_cache": model_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
)
def download_model():
    """Download model weights to the persistent volume (run once)."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir="/model_cache/model",
        token=os.environ["HF_TOKEN"],
    )
    model_volume.commit()
    print(f"Model downloaded to /model_cache/model")


# ---------------------------------------------------------------------------
# vLLM server — OpenAI-compatible /v1/chat/completions with tool calling
# ---------------------------------------------------------------------------

@app.function(
    image=vllm_image,
    gpu=GPU_TYPE,
    volumes={"/model_cache": model_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    scaledown_window=600,
)
@modal.concurrent(max_inputs=10)
@modal.web_server(port=8000, startup_timeout=300)
def serve():
    """Launch vLLM OpenAI-compatible server with tool calling."""
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_REPO,
        "--served-model-name", SERVED_NAME,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--dtype", "float16",
        "--gpu-memory-utilization", "0.92",
        # Qwen3-Thinking emits <think>...</think> BEFORE <tool_call>.
        # The reasoning_parser strips the <think> block into a separate
        # `reasoning_content` field so the tool_call_parser sees clean output.
        "--reasoning-parser", "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
    ]

    env = os.environ.copy()
    # vLLM downloads from HF Hub; volume caches it for subsequent cold starts
    env["HF_HOME"] = "/model_cache"
    env["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    subprocess.Popen(cmd, env=env)


# ---------------------------------------------------------------------------
# One-shot test from your terminal
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Run: modal run scripts/modal_deploy.py"""
    import json

    import httpx

    # Step 1: ensure model is cached
    print("Ensuring model is downloaded...")
    download_model.remote()

    # Step 2: get the serve URL
    url = serve.web_url
    print(f"\nEndpoint: {url}")
    print(f"  /v1/models:           {url}/v1/models")
    print(f"  /v1/chat/completions: {url}/v1/chat/completions")

    # Step 3: test it
    print("\nTesting /v1/chat/completions with tool calling...")
    payload = {
        "model": SERVED_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are a Loopper support agent. Call rag_search before responding.",
            },
            {"role": "user", "content": "My order is delayed by 3 days"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "rag_search",
                    "description": "Search Loopper KB",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "collection": {
                                "type": "string",
                                "enum": [
                                    "faq",
                                    "operations",
                                    "communication_guidelines",
                                    "supplier_data",
                                ],
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
    }

    with httpx.Client(timeout=120.0) as client:
        r = client.post(f"{url}/v1/chat/completions", json=payload)
        data = r.json()

    msg = data.get("choices", [{}])[0].get("message", {})
    tool_calls = msg.get("tool_calls") or []
    content = msg.get("content") or ""

    print(f"\nResponse:")
    print(f"  tool_calls: {len(tool_calls)}")
    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function", {})
            print(f"    {fn.get('name')}({fn.get('arguments')})")
        print("\n  ✓ Tool calling works!")
    else:
        print(f"  content: {content[:300]}")

    print(f"\n--- AGENT ENV VARS ---")
    print(f"USE_KIKI_SLM=true")
    print(f"KIKI_SLM_URL={url}")
    print(f"KIKI_SLM_MODEL={SERVED_NAME}")
    print(f"KIKI_SLM_API_KEY=")
    print(f"KIKI_SLM_MAX_TURNS=4")
    print(f"KIKI_SLM_TIMEOUT_S=300")
