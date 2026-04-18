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

---

## 3. Model skips rag_search on some tickets despite "MUST" in system prompt

**Severity:** High — model invents policy from pretraining instead of grounding in KB.

**Symptom:** On some tickets (observed on refund_request, simple queries), the model goes straight to final JSON with `"policy_used": "No policy consulted"` instead of calling `rag_search` first. Its own `<think>` block says it needs to search, then it doesn't.

**Why it's dangerous:** The response is ungrounded — the model makes up policy from Qwen3's pretraining. For B2B support at Loopper, this could mean the model tells a customer the wrong refund policy, wrong lead times, wrong pricing.

**Root cause:** Training data likely contains examples where the teacher agent skipped tool calls (for rejection paths or simple acknowledgments). The model generalized this to "it's OK to skip searching if I think I know the answer."

**Fix for v2:**
- Filter training data: every `is_valid=true` example MUST have at least one `rag_search` call before the final assistant turn. Reject training examples that don't follow this pattern.
- Consider a reward/penalty signal in a DPO stage: prefer outputs that call rag_search over outputs that don't.
- At inference time (stopgap): if the client receives a final JSON with no tool calls in the trajectory AND `is_valid=true`, reject it and retry with a stronger system prompt.

**Discovered:** 2026-04-16, live Modal endpoint test on ticket 5496 (refund policy query).

---

## 4. Malformed final JSON — `response` field outside the outer object

**Severity:** High — `response_english` ends up None, every affected ticket requires manual response.

**Symptom:** The model closes the outer JSON object too early (after `reasoning`), then emits `response` as a dangling sibling:

```
{"intent":"refund_request", ..., "reasoning":{...}}, "response":"Hello..."}
                                                 ↑ outer closes here, too early
```

`json.loads()` fails with `Extra data: line 1 column N`. Client's balanced-brace fallback extracts the inner object, but without the `response` field — so `response_english=None` reaches the human review gate.

**Root cause:** Training data generation bug. The `reasoning` field in training examples was likely rendered with an extra `}` somewhere, or the JSON serialization of the 11-field output had a bug where `response` was added after the dict was already closed. The model learned this malformed pattern.

**Fix for v2:** In `build_chatml.py`, validate that every final assistant turn's `content` parses as clean JSON with all 11 fields present. Reject examples where the JSON is malformed. Add a unit test:

```python
def test_final_assistant_content_is_valid_json():
    for ex in load_chatml("train.jsonl"):
        final = ex["messages"][-1]
        assert final["role"] == "assistant"
        content = final.get("content") or "{}"
        parsed = json.loads(content)  # must succeed
        assert set(parsed.keys()) >= {"intent","urgency","confidence","is_valid",
            "rejection_type","resolution_type","team","actions","summary",
            "reasoning","response"}
```

**Discovered:** 2026-04-16, live Modal endpoint test on ticket 5496. Also likely causing some of the 0% response generation on specific tickets during eval.

**UPDATE (2026-04-16):** Training data verified clean — all 4,099 final JSON objects parse with all 11 fields. The malformed output is an inference-time generation artifact that correlates with the model skipping tool calls (issue #3). When the model follows the trained path (search → JSON), the JSON is clean. When it goes off-path (skip search → JSON directly), the `<think>` distribution shifts and JSON structure degrades. Fix tool-calling consistency and this likely resolves too.

---

## 5. v1 training data distribution analysis — issues to fix for v2

**Analysis date:** 2026-04-18, from `data/sft-data/chatml/train_trimmed.jsonl` (4,099 examples).

### 5a. Severe intent class imbalance

```
  design_update         1,030  (25.1%)  ████████████████████████████████
  other                   879  (21.4%)  ███████████████████████████
  delivery_issue          873  (21.3%)  ███████████████████████████
  new_order_inquiry       677  (16.5%)  █████████████████████
  quality_complaint       168  ( 4.1%)  █████
  order_cancellation      147  ( 3.6%)  ████
  payment_confirmation    140  ( 3.4%)  ████
  sample_request           78  ( 1.9%)  ██
  price_negotiation        44  ( 1.1%)  █
  customer_feedback        33  ( 0.8%)  █
  refund_request           30  ( 0.7%)  █
```

**Impact:** The model saw only **30 refund_request examples** in all of training. When the live eval sent refund tickets, the model either misclassified them or generated generic responses. Same for price_negotiation (44), customer_feedback (33), and sample_request (78).

The top 4 categories cover **84.3%** of all data. The bottom 4 cover **4.5%**. The model has no chance of learning rare categories reliably.

**Fix for v2:**
- **Oversample** rare categories to at least 200 examples each. Use the teacher agent to generate more traces from the Freshdesk ticket pool, filtering by category.
- **Target distribution** (minimum per category):

| Category | v1 count | v2 minimum |
|---|---:|---:|
| refund_request | 30 | 300 |
| customer_feedback | 33 | 200 |
| price_negotiation | 44 | 200 |
| sample_request | 78 | 200 |
| payment_confirmation | 140 | 250 |
| order_cancellation | 147 | 250 |
| quality_complaint | 168 | 250 |

- **Reduce `other` intent** from 21.4% to <10%. Many "other" tickets are probably classifiable — re-run the teacher agent with a better triage prompt to reclassify them.

### 5b. Missing `critical` urgency — zero examples

```
  medium   2,020  (49.3%)
  low      1,666  (40.6%)
  high       413  (10.1%)
  critical     0  ( 0.0%)  ← model cannot learn this
```

**Impact:** The model will never assign `critical` urgency because it never saw one in training. Any ticket that should be critical gets downgraded to `high`.

**Fix for v2:**
- Add at least 50 `critical` urgency examples. These should be: production-blocking delivery failures, data breaches, legal/compliance issues, health/safety complaints.
- If real critical tickets are rare, synthesize them from existing high-urgency tickets with amplified severity.

### 5c. Finance team severely underrepresented

```
  none              1,847  (45.1%)
  account_manager     864  (21.1%)
  design              657  (16.0%)
  logistics           652  (15.9%)
  finance              79  ( 1.9%)  ← 25x fewer than logistics
```

**Impact:** The model routes refund/payment tickets to `account_manager` or `none` instead of `finance`. Only 34 out of 140 `payment_confirmation` tickets route to `finance` — the rest go to `none`.

**Fix for v2:** Increase finance-team examples to at least 200. Ensure `refund_request` and `payment_confirmation` tickets consistently route to `finance`.

### 5d. Collection names don't match the original tool schema

Training data tool calls use these collection names:

```
  communication_guidelines     3,352  (matches schema)
  customer_policy_faq          2,438  (schema said "faq")
  sales_operations_playbook    1,357  (schema said "operations")
  faq                            785  (matches schema)
  supplier_intelligence           20  (schema said "supplier_data")
```

The teacher agent used verbose internal names (`customer_policy_faq`) while the tool schema's `enum` said `faq`. The model learned the verbose names.

**Current workaround (v1):** Tool enum in the agent was updated to match the model's actual output.

**Fix for v2:** Decide on ONE canonical set of names. Either:
- (a) Update the tool schema enum to match the teacher's names → keep `customer_policy_faq`, etc.
- (b) Normalize the teacher traces during ChatML building → map `customer_policy_faq` → `faq` before writing tool_calls

Option (b) is cleaner — short, consistent names reduce token count and model confusion. But requires updating the RAG MCP server to accept the short names too.

### 5e. Response quality — 27% contain "shortly", too many generic templates

```
  Responses containing "shortly":    1,125  (27.4%)
  Responses containing "checking with":  198  (4.8%)
  Empty responses (rejections):        737  (18.0%)
  Response median length:              244 chars
```

**Impact:** The model learned from the TEACHER to say "I'll check shortly" on over a quarter of tickets. When the model reproduces this at inference, the customer gets a non-answer. The Loopper Agent (OpenAI path) gives the same kind of responses in training — but at inference it's grounded by real KB context, making the "shortly" feel appropriate. The SLM, skipping the KB, produces the same template without the grounding, making it feel hollow.

**Fix for v2:**
- In `generate_traces.py` or `build_chatml.py`, flag responses that are pure "I'll check and get back to you" without any actionable content. Either:
  - (a) Re-run the teacher on those tickets with a prompt that demands specific policy citations, follow-up questions, or concrete next steps.
  - (b) Post-process: append KB-specific details to the response template (e.g., "Our standard refund processing time is X days" from the KB).
- Target: <10% of responses should be pure "I'll check shortly" templates. The rest should include at least one specific fact from KB context.

### 5f. Responses don't ask for follow-up information

Live eval showed the Loopper Agent asks for photos (quality complaints), vector files (design updates), and order specifics (new orders). The Qiki LM doesn't.

**Root cause:** The training responses were generated by the teacher agent AFTER seeing RAG results. The KB told the teacher "quality policy: request photos" so it asked. The SLM was trained on these responses but doesn't reliably reproduce the follow-up requests because:
1. Some training responses include the ask, some don't (inconsistency).
2. Without KB context at inference, the model doesn't know WHAT to ask for.

**Fix for v2:**
- In `build_chatml.py`, validate that training responses for specific categories contain expected follow-up patterns:
  - `quality_complaint` → response should contain "photo" or "image" or "picture"
  - `design_update` → response should contain "vector" or ".AI" or ".EPS" or "file"
  - `new_order_inquiry` → response should contain "product" or "SKU" or "quantity"
- Flag non-conforming examples and re-generate their responses with a more specific teacher prompt.

### 5g. 18% rejection rate creates a large skip-path in the model's training

```
  is_valid=True:   3,362 (82.0%)  → ALL have tool calls ✓
  is_valid=False:    737 (18.0%)  → ALL have 0 tool calls ✓
```

**Good news:** The training data is clean — every valid ticket HAS tool calls, every rejection doesn't. The model's inference-time tool-skipping on valid tickets is NOT from training data corruption.

**Bad news:** 18% of examples teach "sometimes you skip tool calls and go straight to JSON." The model over-generalizes this skip path to valid tickets that look "simple" or have formal/passive phrasing.

**Fix for v2:**
- Reduce rejection ratio to ~10%. The current 18% is higher than necessary and gives the skip pattern too much weight.
- Make the rejection path STRUCTURALLY distinct: rejections should have a different format or shorter reasoning, so the model can clearly distinguish "rejection mode" from "valid ticket mode."
- Add a diverse set of "simple-looking but valid" tickets that DO call tools. For example: "Thanks, got it" → classify as `customer_feedback`, is_valid=true, still call `rag_search(communication_guidelines, "acknowledgment response tone")`. This teaches the model that even simple tickets need grounding.

---

## Summary — v2 dataset checklist

| # | What to fix | How | Priority |
|---|---|---|---|
| 1 | Oversample rare intents (refund, feedback, price, sample) | More teacher traces from filtered tickets | **Critical** |
| 2 | Add critical urgency examples | Synthesize from high-urgency tickets | High |
| 3 | Increase finance team representation | More refund/payment traces routed to finance | High |
| 4 | Normalize collection names | Map verbose → short in build_chatml.py | Medium |
| 5 | Reduce "I'll check shortly" responses | Re-run teacher with policy-citation prompt | High |
| 6 | Add follow-up question patterns | Validate per-category response requirements | Medium |
| 7 | Reduce rejection ratio to ~10% | Remove excess rejection examples | Medium |
| 8 | Reduce "other" intent from 21% to <10% | Re-classify with better triage prompt | Medium |
| 9 | Validate tool_call JSON (no trailing commas) | Post-process validation in build_chatml.py | Medium |
| 10 | Validate final JSON (all 11 fields present) | Assertion in build_chatml.py | Medium |
| 11 | Target total: 8-10K examples (vs 4K in v1) | More teacher traces across all categories | High |
