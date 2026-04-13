# Known Issues — Fix Before Retraining

## 1. Trailing comma in `<tool_call>` JSON output

**Severity:** Medium — breaks vLLM server-side tool parsing, requires client-side workaround.

**Symptom:** The model generates:
```
<tool_call>
{"name": "rag_search", "arguments": {"collection": "faq", "query": "delivery delay"}},
</tool_call>
```

That trailing `,` after `}` causes `json.loads()` to fail. vLLM's hermes parser logs:
```
JSONDecodeError: Extra data: line 2 column 132 (char 132)
```

Tool calls end up as raw text in `content` instead of structured `tool_calls` field.

**Root cause:** The Jinja2 chat template renders tool calls cleanly (verified — no comma). The comma is a generation artifact the model learned during training, likely from:
- Multi-tool-call turns where commas separated items in the original JSON array
- The teacher agent's raw traces in `generate_traces.py` may have included comma-separated formatting that leaked into the ChatML

**Fix for v2 retrain:**

In `scripts/loopper/build_chatml.py`, add a post-processing validation step:

```python
def validate_tool_call_rendering(example: dict, tokenizer) -> list[str]:
    """Check that rendered tool_calls parse as clean JSON."""
    warnings = []
    rendered = tokenizer.apply_chat_template(
        example["messages"],
        tools=example.get("tools"),
        tokenize=False,
    )
    import re
    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", rendered, re.DOTALL):
        body = match.group(1).strip()
        # Strip trailing comma if present
        clean = body.rstrip(",").strip()
        try:
            json.loads(clean)
        except json.JSONDecodeError as e:
            warnings.append(f"Bad tool_call JSON: {e} — body: {body[:100]}")
        if body != clean:
            warnings.append(f"Trailing comma in tool_call: {body[:100]}")
    return warnings
```

Run this on every example in the dataset before training. If any warnings fire, fix the source data.

Also check `generate_traces.py` — specifically how `tool_calls` from the teacher agent (LangGraph ReACT) are serialized into the ChatML messages. The `arguments` field should be a JSON string, not a Python repr with trailing commas.

**Current workaround (v1 model):**
- `kiki_client.py` has `_parse_tool_calls_from_content()` which does `.rstrip(",")` before `json.loads()`
- vLLM deployed WITHOUT `--enable-auto-tool-choice` / `--tool-call-parser hermes` to avoid the error logs
- Works correctly, just not as clean as server-side parsing would be

**Discovered:** 2026-04-12, Modal.com vLLM 0.19 + A10G + Qwen3-4B-Thinking-2507.

---

## 2. Model hallucinates invalid RAG collection names

**Severity:** Low — RAG server returns empty results, model proceeds anyway.

**Symptom:** Model calls `rag_search` with collections like `customer_policy_faq` or `sales_operations_playbook` which don't exist in the training schema. Valid collections: `faq`, `operations`, `communication_guidelines`, `supplier_data`.

**Root cause:** The tool definition's `enum` constraint is passed to the model but the model doesn't always respect it. The teacher agent may have used different collection names internally.

**Fix for v2:** Verify the tool schema enum in `configs/loopper_pipeline.yaml` matches exactly what the teacher agent uses. Add a validation step in `build_chatml.py` that checks every `rag_search` call in the training data uses a valid collection name.

**Discovered:** 2026-04-12, local Ollama smoke test on gold_100.jsonl.
