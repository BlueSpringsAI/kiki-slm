#!/usr/bin/env python3
"""
Kiki SLM — Gradio Web Demo
===========================
Launch a Gradio web interface showing the Kiki SLM processing customer
service tickets in real-time with full pipeline visualization.

Usage:
    python scripts/4_demo.py
    python scripts/4_demo.py --port 7861
    python scripts/4_demo.py --share
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# System prompt — same as training (2_train.py)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are Kiki, an AI customer service agent. When given a customer message, analyze it and respond with:
1. Your classification (intent, urgency)
2. The workflow steps needed to resolve this
3. Which tools to invoke with what parameters
4. A professional, empathetic response to the customer

Always respond in valid JSON with these fields:
- intent: string
- urgency: string (critical/high/medium/low)
- workflow_steps: list of strings
- tools_required: list of strings with parameters
- reasoning: brief explanation of your analysis
- response: the customer-facing reply"""

# ---------------------------------------------------------------------------
# Example tickets
# ---------------------------------------------------------------------------
EXAMPLE_TICKETS: list[str] = [
    "My order #ORD-48293 hasn't arrived yet. I placed it 5 days ago.",
    "I received a damaged laptop. The screen is cracked. I want a full refund.",
    "Can you help me change my delivery address for order #ORD-77102?",
    "I've been charged twice for my subscription. Please fix this.",
    "I want to cancel my account and get a refund for the remaining months.",
    "What's your return policy for electronics?",
    "Someone made unauthorized purchases on my account!",
    "My warranty claim for product SN-847291 was denied. I want to appeal.",
    "I need to update my payment method to a new credit card.",
    "The product I received doesn't match the description on your website.",
]

# ---------------------------------------------------------------------------
# Model loading — three-tier fallback
# ---------------------------------------------------------------------------

# Sentinels for which backend is active
_backend: Optional[str] = None
_model: Any = None
_tokenizer: Any = None
_openai_client: Any = None


def _try_load_mlx() -> bool:
    """Attempt to load model via MLX (Apple Silicon)."""
    global _backend, _model, _tokenizer
    try:
        from mlx_lm import load  # type: ignore

        model_path = "outputs/exports/kiki-poc-q4"
        if not Path(model_path).exists():
            print(f"[MLX] Model path '{model_path}' not found, skipping MLX.")
            return False

        print("[MLX] Loading model …")
        _model, _tokenizer = load(model_path)
        _backend = "mlx"
        print("[MLX] Model loaded successfully.")
        return True
    except ImportError:
        print("[MLX] mlx_lm not installed, skipping.")
        return False
    except Exception as exc:
        print(f"[MLX] Failed to load: {exc}")
        return False


def _try_load_transformers() -> bool:
    """Attempt to load model via Unsloth / transformers with QLoRA."""
    global _backend, _model, _tokenizer
    try:
        from unsloth import FastLanguageModel  # type: ignore

        model_path = "outputs/models/kiki-poc-v1"
        if not Path(model_path).exists():
            print(f"[Transformers] Model path '{model_path}' not found, skipping.")
            return False

        print("[Transformers] Loading model …")
        _model, _tokenizer = FastLanguageModel.from_pretrained(
            model_path,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(_model)
        _backend = "transformers"
        print("[Transformers] Model loaded successfully.")
        return True
    except ImportError:
        print("[Transformers] unsloth not installed, skipping.")
        return False
    except Exception as exc:
        print(f"[Transformers] Failed to load: {exc}")
        return False


def _try_load_openai() -> bool:
    """Fall back to OpenAI API (gpt-4o-mini as stand-in)."""
    global _backend, _openai_client
    try:
        from openai import OpenAI  # type: ignore

        _openai_client = OpenAI()
        _backend = "openai"
        print("[OpenAI] Using gpt-4o-mini as API fallback.")
        return True
    except ImportError:
        print("[OpenAI] openai package not installed.")
        return False
    except Exception as exc:
        print(f"[OpenAI] Failed to initialise client: {exc}")
        return False


def load_model() -> str:
    """Load model with three-tier fallback.  Returns the backend name."""
    if _try_load_mlx():
        return "mlx"
    if _try_load_transformers():
        return "transformers"
    if _try_load_openai():
        return "openai"
    raise RuntimeError(
        "Could not load any model backend. Install mlx_lm, unsloth, or openai."
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _generate_mlx(user_message: str, channel: str) -> tuple[str, dict]:
    """Generate with MLX backend. Returns (raw_text, stats)."""
    from mlx_lm import generate  # type: ignore

    prompt = _tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"[Channel: {channel}] {user_message}"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    t0 = time.perf_counter()
    raw = generate(
        _model,
        _tokenizer,
        prompt=prompt,
        max_tokens=1024,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    token_count = len(_tokenizer.encode(raw))
    stats = {
        "tokens": token_count,
        "latency_s": round(elapsed, 3),
        "tokens_per_sec": round(token_count / max(elapsed, 1e-6), 1),
    }
    return raw, stats


def _generate_transformers(user_message: str, channel: str) -> tuple[str, dict]:
    """Generate with transformers / Unsloth backend."""
    import torch  # type: ignore

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"[Channel: {channel}] {user_message}"},
    ]
    input_ids = _tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(_model.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = _model.generate(
            input_ids=input_ids,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    elapsed = time.perf_counter() - t0

    new_tokens = output_ids[0][input_ids.shape[1]:]
    raw = _tokenizer.decode(new_tokens, skip_special_tokens=True)
    token_count = len(new_tokens)
    stats = {
        "tokens": token_count,
        "latency_s": round(elapsed, 3),
        "tokens_per_sec": round(token_count / max(elapsed, 1e-6), 1),
    }
    return raw, stats


def _generate_openai(user_message: str, channel: str) -> tuple[str, dict]:
    """Generate with OpenAI API fallback."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"[Channel: {channel}] {user_message}"},
    ]

    t0 = time.perf_counter()
    response = _openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )
    elapsed = time.perf_counter() - t0

    raw = response.choices[0].message.content or ""
    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
    completion_tokens = response.usage.completion_tokens if response.usage else 0
    total_tokens = prompt_tokens + completion_tokens
    stats = {
        "tokens": completion_tokens,
        "latency_s": round(elapsed, 3),
        "tokens_per_sec": round(completion_tokens / max(elapsed, 1e-6), 1),
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
    }
    return raw, stats


def run_inference(user_message: str, channel: str) -> tuple[str, dict]:
    """Run inference on whichever backend is loaded."""
    if _backend == "mlx":
        return _generate_mlx(user_message, channel)
    elif _backend == "transformers":
        return _generate_transformers(user_message, channel)
    elif _backend == "openai":
        return _generate_openai(user_message, channel)
    else:
        raise RuntimeError("No model backend loaded.")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Best-effort extraction of the first JSON object from model output."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Look for JSON between ```json ... ``` or { ... }
    import re

    # Try fenced code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try first { ... } block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def parse_response(raw: str) -> dict:
    """Parse the raw model output into structured sections."""
    parsed = _extract_json(raw)

    result = {
        "intent": parsed.get("intent", "unknown"),
        "urgency": parsed.get("urgency", "medium"),
        "reasoning": parsed.get("reasoning", ""),
        "workflow_steps": parsed.get("workflow_steps", []),
        "tools_required": parsed.get("tools_required", []),
        "response": parsed.get("response", raw),
    }
    return result


# ---------------------------------------------------------------------------
# Cost comparison helper
# ---------------------------------------------------------------------------

def _cost_comparison(stats: dict) -> dict:
    """Rough cost comparison between Kiki local and GPT-4o API."""
    tokens = stats.get("tokens", 0)
    prompt_tokens = stats.get("prompt_tokens", tokens)

    # GPT-4o pricing (as of 2025): $2.50/1M input, $10/1M output
    gpt4o_input_cost = (prompt_tokens / 1_000_000) * 2.50
    gpt4o_output_cost = (tokens / 1_000_000) * 10.00
    gpt4o_total = gpt4o_input_cost + gpt4o_output_cost

    if _backend == "openai":
        # gpt-4o-mini pricing: $0.15/1M input, $0.60/1M output
        mini_input_cost = (prompt_tokens / 1_000_000) * 0.15
        mini_output_cost = (tokens / 1_000_000) * 0.60
        kiki_cost = mini_input_cost + mini_output_cost
        kiki_label = "gpt-4o-mini (API fallback)"
    else:
        kiki_cost = 0.0
        kiki_label = f"Kiki local ({_backend})"

    return {
        "kiki_backend": kiki_label,
        "kiki_cost_usd": f"${kiki_cost:.6f}",
        "gpt4o_cost_usd": f"${gpt4o_total:.6f}",
        "savings": "100% (local)" if kiki_cost == 0 else f"{max(0, (1 - kiki_cost / gpt4o_total)) * 100:.1f}%",
    }


# ---------------------------------------------------------------------------
# Urgency badge
# ---------------------------------------------------------------------------

_URGENCY_COLORS = {
    "critical": "#dc2626",
    "high": "#ea580c",
    "medium": "#ca8a04",
    "low": "#16a34a",
}


def _urgency_badge(urgency: str) -> str:
    color = _URGENCY_COLORS.get(urgency.lower(), "#6b7280")
    return (
        f'<span style="background:{color};color:white;padding:2px 10px;'
        f'border-radius:12px;font-weight:600;font-size:0.85em;">'
        f"{urgency.upper()}</span>"
    )


# ---------------------------------------------------------------------------
# Main pipeline (wired to Gradio)
# ---------------------------------------------------------------------------

def process_ticket(
    customer_message: str, channel: str
) -> tuple[str, str, str, Any, Any, str, str, str]:
    """Process a single customer ticket through the full pipeline.

    Returns eight values matching the eight output components:
        stage1_md, stage2_md, stage3_json, stage4_md,
        raw_output, model_info_md, stats_md, cost_md
    """
    if not customer_message or not customer_message.strip():
        empty = "Please enter a customer message."
        return empty, "", "", None, None, "", "", ""

    # ---- Inference -----------------------------------------------------------
    t_total_start = time.perf_counter()
    raw, stats = run_inference(customer_message.strip(), channel)
    t_total = time.perf_counter() - t_total_start

    # ---- Parse ---------------------------------------------------------------
    t_parse_start = time.perf_counter()
    parsed = parse_response(raw)
    t_parse = time.perf_counter() - t_parse_start

    # ---- Stage 1: Intent Classification -------------------------------------
    intent = parsed["intent"]
    urgency = parsed["urgency"]
    reasoning = parsed["reasoning"]
    confidence_note = "based on model output"

    stage1_md = (
        f"### Intent Classification\n\n"
        f"**Intent:** `{intent}`  \n"
        f"**Urgency:** {_urgency_badge(urgency)}  \n"
        f"**Reasoning:** {reasoning}  \n"
        f"*Confidence: {confidence_note}*  \n"
        f"*Stage time: {t_parse * 1000:.1f} ms (parse)*"
    )

    # ---- Stage 2: Workflow Planning ------------------------------------------
    steps = parsed["workflow_steps"]
    if steps:
        step_lines = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
    else:
        step_lines = "_No workflow steps returned._"

    stage2_md = f"### Workflow Plan\n\n{step_lines}"

    # ---- Stage 3: Tool Invocation Plan ---------------------------------------
    tools = parsed["tools_required"]
    stage3_json = tools if tools else []

    # ---- Stage 4: Response Generation ----------------------------------------
    customer_response = parsed["response"]
    stage4_md = f"### Customer Response\n\n{customer_response}"

    # ---- Raw output ----------------------------------------------------------
    raw_output = raw

    # ---- Sidebar: Model info -------------------------------------------------
    if _backend == "mlx":
        model_name = "Kiki PoC (MLX Q4)"
        quant = "4-bit (MLX)"
        params_info = "~4B (Qwen3-4B base)"
    elif _backend == "transformers":
        model_name = "Kiki PoC (QLoRA)"
        quant = "4-bit (QLoRA)"
        params_info = "~4B (Qwen3-4B base)"
    else:
        model_name = "gpt-4o-mini (API fallback)"
        quant = "N/A (cloud)"
        params_info = "N/A"

    model_info_md = (
        f"### Model Info\n\n"
        f"| Property | Value |\n"
        f"|----------|-------|\n"
        f"| **Model** | {model_name} |\n"
        f"| **Parameters** | {params_info} |\n"
        f"| **Quantization** | {quant} |\n"
        f"| **Backend** | `{_backend}` |"
    )

    # ---- Sidebar: Inference stats --------------------------------------------
    stats_md = (
        f"### Inference Stats\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| **Tokens generated** | {stats.get('tokens', '?')} |\n"
        f"| **Latency** | {stats.get('latency_s', '?')} s |\n"
        f"| **Tokens / sec** | {stats.get('tokens_per_sec', '?')} |\n"
        f"| **Total wall time** | {t_total:.3f} s |"
    )

    # ---- Sidebar: Cost comparison --------------------------------------------
    cost = _cost_comparison(stats)
    cost_md = (
        f"### Cost Comparison\n\n"
        f"| | Cost |\n"
        f"|---|------|\n"
        f"| **{cost['kiki_backend']}** | {cost['kiki_cost_usd']} |\n"
        f"| **GPT-4o (reference)** | {cost['gpt4o_cost_usd']} |\n"
        f"| **Savings** | {cost['savings']} |"
    )

    return (
        stage1_md,
        stage2_md,
        stage3_json,
        stage4_md,
        raw_output,
        model_info_md,
        stats_md,
        cost_md,
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> "gr.Blocks":
    import gradio as gr  # type: ignore

    with gr.Blocks(
        title="Kiki SLM — Customer Service Demo",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# Kiki SLM — Customer Service Agent Demo\n"
            "Process customer tickets through the full Kiki pipeline: "
            "**Intent Classification → Workflow Planning → Tool Invocation → Response Generation**"
        )

        with gr.Row():
            # ---- Left column: inputs + processing stages --------------------
            with gr.Column(scale=3):
                gr.Markdown("## Input")
                with gr.Row():
                    txt_message = gr.Textbox(
                        label="Customer Message",
                        placeholder="Type a customer message or select an example below …",
                        lines=3,
                        scale=3,
                    )
                    dd_channel = gr.Dropdown(
                        choices=["email", "chat", "phone"],
                        value="email",
                        label="Channel",
                        scale=1,
                    )

                btn_submit = gr.Button("Process Ticket", variant="primary")

                gr.Markdown("## Pipeline Stages")

                with gr.Accordion("Stage 1 — Intent Classification", open=True):
                    out_stage1 = gr.Markdown()

                with gr.Accordion("Stage 2 — Workflow Planning", open=True):
                    out_stage2 = gr.Markdown()

                with gr.Accordion("Stage 3 — Tool Invocation Plan", open=True):
                    out_stage3 = gr.JSON(label="Tool Calls")

                with gr.Accordion("Stage 4 — Response Generation", open=True):
                    out_stage4 = gr.Markdown()

                with gr.Accordion("Raw Model Output", open=False):
                    out_raw = gr.Textbox(label="Raw output", lines=12, interactive=False)

            # ---- Right column: sidebar info ---------------------------------
            with gr.Column(scale=1):
                out_model_info = gr.Markdown()
                out_stats = gr.Markdown()
                out_cost = gr.Markdown()

        # ---- Examples -------------------------------------------------------
        gr.Markdown("## Example Tickets")
        gr.Examples(
            examples=[[t, "email"] for t in EXAMPLE_TICKETS],
            inputs=[txt_message, dd_channel],
            label="Click an example to populate the input",
        )

        # ---- Wiring ---------------------------------------------------------
        btn_submit.click(
            fn=process_ticket,
            inputs=[txt_message, dd_channel],
            outputs=[
                out_stage1,
                out_stage2,
                out_stage3,
                out_stage4,
                out_raw,
                out_model_info,
                out_stats,
                out_cost,
            ],
        )

    return demo


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kiki SLM — Gradio web demo for customer service agent"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to serve the Gradio app on (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a publicly shareable Gradio link",
    )
    args = parser.parse_args()

    # Load model (three-tier fallback)
    backend = load_model()
    print(f"\nBackend ready: {backend}\n")

    # Import gradio here so model loading errors surface before UI code
    import gradio as gr  # noqa: F811

    demo = build_ui()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
