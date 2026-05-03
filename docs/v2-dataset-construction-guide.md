# v2 Dataset Construction & Validation Pipeline

> A concrete, code-driven guide to building the v2 training dataset with validation gates at every stage. Designed to fail LOUDLY when something is wrong, not silently produce bad data.

---

## The core principle (read this first)

**Validation must happen at every stage, not just at the end.**

v1 failed because validation was either absent or run only at training time. By the time you saw the problems, the data was already corrupted. The fix is **continuous validation** with checkpoints you can inspect manually.

```
Bad pipeline (v1):
  source → trace → chatml → train → "why is the model bad?"
                                     ↑ Problems detected too late

Good pipeline (v2):
  source ✓ → trace ✓ → chatml ✓ → balance ✓ → validate ✓ → train
            ↑          ↑         ↑          ↑
            Inspect    Inspect   Inspect    Inspect
            and FAIL   stats     stats      gate
            on bad
            traces
```

Every stage produces:
1. **Output artifacts** (the data)
2. **Stats** (distributions, counts, validation results)
3. **A pass/fail gate** that prevents bad data from flowing downstream

If a stage fails, you fix it BEFORE moving on. No fast-forwarding through failures.

---

## Pipeline overview

```
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 0: Source pool acquisition                                │
│   - Real Freshdesk tickets (existing 4,544 + new)               │
│   - Synthetic tickets (for rare categories)                     │
│   - Manual seed examples (for rarest patterns)                  │
│   Validates: ticket text quality, language detection, dedup     │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ STAGE 1: Trace generation (run teacher agent)                   │
│   - Run LangGraph agent on every source ticket                  │
│   - Capture full multi-turn trace (reasoning, tools, response)  │
│   Validates: trace structure, tool call presence,               │
│              JSON output cleanliness, response quality          │
│   Gate: traces with bad JSON or skip patterns are flagged       │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ STAGE 2: Quality filtering & re-generation                      │
│   - Flag template responses, missing tool calls, wrong format   │
│   - Re-run teacher with stricter prompts on flagged traces      │
│   - Drop unfixable traces                                       │
│   Validates: tool-call invariant, response substance,           │
│              warmth markers, empathy on complaints              │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ STAGE 3: ChatML construction                                    │
│   - Convert traces to Qwen3 chat format                         │
│   - Add reasoning_content, tool_calls, tool results             │
│   - Apply transformations (collection name normalization)       │
│   Validates: chat template renders cleanly,                     │
│              token count under max_seq_length                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ STAGE 4: Distribution balancing                                 │
│   - Count per-category, per-language, per-team, per-urgency     │
│   - Oversample rare classes (with deduplication awareness)      │
│   - Downsample dominant classes                                 │
│   Validates: distribution gates pass (min counts per category)  │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ STAGE 5: Final validation gates (CRITICAL)                      │
│   - All 11+ structural checks                                   │
│   - Distribution gates                                          │
│   - Token count gates                                           │
│   - Sample-based human review (10 random examples)              │
│   Gate: ALL gates must pass. No "warnings" — only PASS or FAIL  │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ STAGE 6: Train/eval split (stratified)                          │
│   - 90/10 split                                                 │
│   - Stratified by category (rare classes get eval representation)│
│   Validates: every category has ≥10 eval examples                │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ STAGE 7: Pre-training smoke test                                │
│   - Tokenize 5 random examples through Qwen3 tokenizer          │
│   - Verify rendered output is identical to expected             │
│   - Verify token counts match what trainer will see             │
│   Gate: tokenizer round-trip must succeed                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
                  Ready to train ✓
```

---

## Stage 0: Source pool acquisition

### 0.1 Existing Freshdesk tickets

You already have ~4,544 raw tickets used for v1. Reuse them, but with stricter filtering for v2.

```python
# scripts/loopper/sample_tickets_v2.py
from collections import Counter
import json

SOURCE_PATH = "data/raw_tickets/freshdesk_dump.jsonl"

def load_existing_tickets():
    tickets = []
    with open(SOURCE_PATH) as f:
        for line in f:
            t = json.loads(line)
            if validate_raw_ticket(t):
                tickets.append(t)
    return tickets

def validate_raw_ticket(t: dict) -> bool:
    """Filter v1 had bugs — apply stricter checks for v2."""
    if not t.get("messages"):
        return False
    if not t.get("ticket_id"):
        return False
    # Reject tickets where ALL messages are empty
    if all(not m.get("clean_body", "").strip() for m in t["messages"]):
        return False
    # Reject if total text is implausibly short (likely truncation bug)
    total_chars = sum(len(m.get("clean_body", "")) for m in t["messages"])
    if total_chars < 50:
        return False
    return True
```

### 0.2 Filter by target category for rare classes

For categories that need oversampling, filter the raw pool by keywords:

```python
CATEGORY_KEYWORDS_MULTILINGUAL = {
    "refund_request": [
        # English
        "refund", "reimburse", "money back", "credit back",
        # French
        "remboursement", "rembourser",
        # German
        "Erstattung", "rückzahlung", "rückerstattung",
        # Spanish
        "reembolso", "devolver dinero",
        # Italian
        "rimborso",
        # Dutch
        "terugbetaling",
    ],
    "price_negotiation": [
        "discount", "rebate", "bulk price", "volume discount", "negotiate",
        "remise", "rabais", "tarif",
        "Rabatt", "Mengenrabatt", "Verhandeln",
        "descuento", "mejor precio",
        "sconto", "prezzo migliore",
        "korting",
    ],
    "sample_request": [
        "sample", "test sample", "trial",
        "échantillon", "spécimen",
        "Muster", "Probemuster",
        "muestra", "prueba",
        "campione",
        "monster",
    ],
    "customer_feedback": [
        "feedback", "review", "experience",
        "retour d'expérience", "avis",
        "Feedback", "Bewertung",
        "comentario", "opinión",
        "feedback",
        "feedback",
    ],
    # ... add for all rare categories
}

def filter_tickets_for_category(tickets: list, category: str) -> list:
    """Return tickets likely to be of the target category based on keywords."""
    keywords = CATEGORY_KEYWORDS_MULTILINGUAL.get(category, [])
    if not keywords:
        return []

    matching = []
    for t in tickets:
        ticket_text = " ".join(m.get("clean_body", "") for m in t["messages"]).lower()
        if any(kw.lower() in ticket_text for kw in keywords):
            matching.append(t)
    return matching
```

**Stage 0 validation gate:**

```python
def stage_0_validate(tickets_by_category: dict) -> bool:
    """Check we have enough source tickets per category."""
    MIN_PER_CATEGORY = {
        "refund_request": 350,
        "customer_feedback": 250,
        "price_negotiation": 300,
        "sample_request": 300,
        "payment_confirmation": 350,
        "order_cancellation": 350,
        "quality_complaint": 350,
        "new_order_inquiry": 700,
        "delivery_issue": 700,
        "design_update": 700,
        # New v2 intents
        "documentation_request": 200,
        "lead_time_inquiry": 200,
        "tracking_inquiry": 250,
        "product_support": 150,
    }

    failures = []
    for cat, min_count in MIN_PER_CATEGORY.items():
        actual = len(tickets_by_category.get(cat, []))
        if actual < min_count:
            failures.append(f"{cat}: have {actual}, need {min_count}")

    if failures:
        print("STAGE 0 FAILED — insufficient source tickets:")
        for f in failures:
            print(f"  ❌ {f}")
        print("\nFix: synthesize more tickets for failing categories before proceeding.")
        return False

    print("✓ Stage 0: source pool sufficient for all categories")
    return True
```

### 0.3 Synthetic ticket generation for shortfalls

When Freshdesk doesn't have enough tickets for a rare category, generate synthetic ones. This is OK as long as you validate them.

```python
SYNTHETIC_TICKET_PROMPT = """Generate {n} realistic B2B customer support tickets for:

Category: {category}
Description: {description}

Context:
- Customer is a European business buying promotional products
- Languages: rotate between English, French, German, Dutch, Spanish, Italian
- Realistic order numbers (format: ORD-{{5-8 digits}} or {{4-6 digits}}-{{2 letters}})
- Real-sounding business scenarios (events, marketing campaigns, employee gifts)

Each ticket should:
1. Be 3-8 sentences long
2. Have a clear primary intent matching the category
3. Include at least one specific question, request, or complaint
4. Sign off with a realistic business signature
5. Vary in tone (some formal, some casual, some urgent)

Output as JSONL, one ticket per line:
{{"ticket_id": "SYNTH-{{n}}", "language": "...", "messages": [{{"clean_body": "...", "direction": "incoming"}}]}}
"""

CATEGORY_DESCRIPTIONS = {
    "refund_request": "Customer wants money back for a problem with their order. Specific reasons: product defect, wrong item, late delivery missed event, duplicate charge.",
    "documentation_request": "Customer needs paperwork: invoices, quotes, certifications (GOTS, ISO), product specs, compliance documents.",
    "lead_time_inquiry": "Customer asking about production schedules, capacity, or ability to meet a deadline. Pre-order question, not yet ready to buy.",
    # ... etc
}

def generate_synthetic_tickets(category: str, count: int, model: str = "gpt-5") -> list:
    prompt = SYNTHETIC_TICKET_PROMPT.format(
        n=count,
        category=category,
        description=CATEGORY_DESCRIPTIONS[category],
    )
    # Call LLM, parse JSONL response, return tickets
    # ... implementation
    return tickets
```

**Synthetic ticket validation:**

Synthetic tickets must be flagged as such and validated:

```python
def validate_synthetic_ticket(t: dict) -> list[str]:
    """Strict checks for synthetic tickets — they're more error-prone."""
    issues = []
    text = t["messages"][0].get("clean_body", "")

    # 1. Must be in valid range
    if len(text) < 100:
        issues.append("too short — probably truncated")
    if len(text) > 3000:
        issues.append("too long — probably hallucinated")

    # 2. Must look real (no obvious LLM artifacts)
    artifacts = ["[Customer Name]", "[Your Company]", "[Insert", "Lorem ipsum"]
    for art in artifacts:
        if art in text:
            issues.append(f"contains placeholder: {art}")

    # 3. Must have a signature
    if not any(line.strip() for line in text.split("\n")[-3:]):
        issues.append("missing signature/sign-off")

    # 4. Language detection must succeed
    try:
        from langdetect import detect
        lang = detect(text)
        if lang not in {"en", "fr", "de", "nl", "es", "it"}:
            issues.append(f"unsupported language: {lang}")
    except Exception:
        issues.append("language detection failed")

    return issues
```

### 0.4 Deduplication

Before proceeding, dedupe based on text similarity (not exact match — synthetic tickets and real ones can have minor variations).

```python
from hashlib import sha256

def normalize_for_dedup(text: str) -> str:
    """Normalize text for fuzzy dedup."""
    import re
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<pii_\w+:[a-z0-9]+>", "<PII>", text)
    text = re.sub(r"\d", "0", text)  # collapse all numbers
    return text.strip()

def dedup_tickets(tickets: list) -> list:
    seen_hashes = set()
    deduped = []
    for t in tickets:
        text = " ".join(m.get("clean_body", "") for m in t["messages"])
        h = sha256(normalize_for_dedup(text).encode()).hexdigest()[:16]
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(t)
    return deduped
```

### Stage 0 deliverables

After Stage 0 you should have:
- `data/v2/source_pool.jsonl` — all source tickets (real + synthetic), deduplicated
- `data/v2/source_pool_stats.json` — counts per category, per language, real vs synthetic
- A passing Stage 0 validation gate

**Manual checkpoint:** spot-check 20 random tickets. Read them. Do they look like real customer support tickets? If 5+ look weird, fix the source before proceeding.

---

## Stage 1: Trace generation with quality gates

Now run the teacher agent (the existing LangGraph pipeline) on each source ticket. The teacher produces a multi-turn trace: reasoning, tool calls, tool results, final JSON.

### 1.1 Run teacher agent

```bash
# Existing script with v2 enhancements
uv run --project ../Looper-Support-Agent-Server python -m scripts.loopper.generate_traces_v2 \
    --source data/v2/source_pool.jsonl \
    --output data/v2/raw_traces.jsonl \
    --concurrency 5 \
    --resume \
    --validate-each-trace
```

The `--validate-each-trace` flag is new. It runs validation as each trace is generated, not just at the end.

### 1.2 Per-trace validation

```python
def validate_raw_trace(ticket: dict, trace: dict) -> tuple[bool, list[str]]:
    """Check a freshly-generated teacher trace.
    Returns (passed, list of issues).
    """
    issues = []

    # 1. Trace must have all expected stages
    required_keys = ["triage_result", "rag_calls", "final_response"]
    for k in required_keys:
        if k not in trace:
            issues.append(f"missing trace stage: {k}")

    # 2. If valid ticket, must have at least one rag call
    triage = trace.get("triage_result", {})
    if triage.get("is_valid") and len(trace.get("rag_calls", [])) == 0:
        issues.append("VALID ticket but ZERO rag_search calls (tool-skip violation)")

    # 3. Final response must parse as expected schema
    final = trace.get("final_response", {})
    required_fields = {"intent", "urgency", "confidence", "is_valid",
                       "rejection_type", "resolution_type", "team",
                       "actions", "summary", "reasoning", "response"}
    missing = required_fields - set(final.keys())
    if missing:
        issues.append(f"final_response missing fields: {missing}")

    # 4. Response text must not be empty for valid tickets
    if triage.get("is_valid") and not final.get("response", "").strip():
        issues.append("VALID ticket but empty response text")

    # 5. Tool calls must use valid collection names
    valid_collections = {"customer_policy_faq", "sales_operations_playbook",
                         "communication_guidelines", "supplier_intelligence"}
    for call in trace.get("rag_calls", []):
        coll = call.get("arguments", {}).get("collection")
        if coll not in valid_collections:
            issues.append(f"invalid collection in rag_call: {coll}")

    # 6. Tool calls must have well-formed arguments
    for i, call in enumerate(trace.get("rag_calls", [])):
        args = call.get("arguments", {})
        if not args.get("query") or len(args.get("query", "")) < 3:
            issues.append(f"rag_call {i} has empty/short query")

    return (len(issues) == 0, issues)
```

### 1.3 Decision tree on validation result

```python
def handle_trace_validation(ticket, trace, issues):
    if not issues:
        # Save to validated_traces.jsonl
        save_validated(ticket, trace)
        return "passed"

    # Categorize the failure
    if any("tool-skip violation" in i for i in issues):
        # Re-run with stricter prompt that emphasizes tool calling
        new_trace = regenerate_with_stricter_tool_call(ticket)
        passed, new_issues = validate_raw_trace(ticket, new_trace)
        if passed:
            save_validated(ticket, new_trace)
            log_recovery(ticket["ticket_id"], "tool_skip → fixed")
            return "recovered"

    if any("empty response text" in i for i in issues):
        new_trace = regenerate_with_response_emphasis(ticket)
        # ... etc

    # Couldn't fix — log and drop
    log_dropped(ticket["ticket_id"], issues)
    return "dropped"
```

### Stage 1 deliverables

- `data/v2/validated_traces.jsonl` — passing traces only
- `data/v2/dropped_traces.jsonl` — traces that failed validation, with reasons
- `data/v2/trace_stats.json` — pass rate, drop reasons, per-category counts

**Stage 1 gate:**

```python
def stage_1_validate(stats: dict) -> bool:
    pass_rate = stats["validated"] / stats["total"]
    if pass_rate < 0.85:
        print(f"❌ Stage 1: pass rate {pass_rate:.0%} < 85% threshold")
        print(f"   Top drop reasons: {stats['top_drop_reasons']}")
        print("Fix: investigate why teacher is producing bad traces. Don't proceed.")
        return False
    return True
```

**Manual checkpoint:** read 10 random validated traces and 5 dropped traces. Do you understand why the dropped ones failed? If not, refine the validation rules.

---

## Stage 2: Quality filtering & re-generation

Now apply quality filters that go beyond structural correctness. This is where you fix the "I'll check shortly" template problem and the missing warmth/personalization issues.

### 2.1 Detect template responses

```python
TEMPLATE_INDICATORS = [
    "i'll be in touch shortly",
    "i will update you shortly",
    "checking with our",
    "i'm checking with",
    "will get back to you",
    "looking forward to your feedback",  # generic close
]

SUBSTANCE_INDICATORS = [
    # Policy/specifics
    "policy", "within", "business days", "per our", "according to",
    # Specific actions
    "could you please", "can you send", "would you mind", "kindly send",
    # Specific items
    "photo", "image", "picture", "vector file", "tracking number",
    "order number", "invoice", "certificate",
    # Numbers (likely policy timelines or pricing)
    "5 days", "10 days", "14 days", "1-2 days", "3 business",
]

def is_template_response(response: str) -> bool:
    """Return True if response is generic without substance."""
    rl = response.lower()
    has_template = any(t in rl for t in TEMPLATE_INDICATORS)
    has_substance = any(s in rl for s in SUBSTANCE_INDICATORS)
    return has_template and not has_substance
```

### 2.2 Detect missing warmth

```python
LANGUAGE_GREETINGS = {
    "english": ["Hello", "Hi", "Dear"],
    "french": ["Bonjour", "Cher", "Chère"],
    "german": ["Guten Tag", "Hallo", "Sehr geehrte", "Liebe"],
    "spanish": ["Hola", "Estimado", "Estimada", "Buenos días"],
    "italian": ["Buongiorno", "Salve", "Gentile"],
    "dutch": ["Hallo", "Beste", "Goedendag"],
}

LANGUAGE_CLOSINGS = {
    "english": ["Looking forward", "Best regards", "Have a great",
                "Wishing you", "Talk soon", "Kind regards"],
    "french": ["Cordialement", "Avec plaisir", "Au plaisir",
               "Belle journée", "Bonne journée"],
    "german": ["Mit freundlichen Grüßen", "Liebe Grüße", "Beste Grüße"],
    "spanish": ["Saludos cordiales", "Cordialmente",
                "Quedo a la espera", "Un saludo"],
    "italian": ["Cordiali saluti", "Distinti saluti",
                "In attesa", "Buon proseguimento"],
    "dutch": ["Met vriendelijke groet", "Hartelijk dank"],
}

def check_warmth(response: str, language: str) -> list[str]:
    issues = []
    greetings = LANGUAGE_GREETINGS.get(language, [])
    closings = LANGUAGE_CLOSINGS.get(language, [])

    if greetings and not any(g in response for g in greetings):
        issues.append(f"missing {language} greeting")
    if closings and not any(c in response for c in closings):
        issues.append(f"missing {language} warm closing")
    return issues
```

### 2.3 Detect missing empathy on complaints

```python
EMPATHY_MARKERS = {
    "english": ["sorry to hear", "we regret", "we apologize", "I understand",
                "we're sorry", "unfortunate", "I appreciate"],
    "french": ["nous sommes navrés", "je suis navré", "je comprends",
               "désolé", "regret", "nous nous excusons"],
    "german": ["es tut uns leid", "wir bedauern", "wir verstehen",
               "Entschuldigung", "Wir entschuldigen uns"],
    "spanish": ["lamentamos", "sentimos", "entendemos", "comprendemos",
                "disculpe", "Lo siento"],
    "italian": ["ci dispiace", "ci scusiamo", "comprendiamo",
                "siamo spiacenti", "ci scusiamo"],
    "dutch": ["het spijt ons", "onze excuses", "we begrijpen"],
}

COMPLAINT_INTENTS = {"refund_request", "quality_complaint",
                     "delivery_issue", "service_complaint"}

def check_empathy(response: str, language: str, intent: str) -> list[str]:
    if intent not in COMPLAINT_INTENTS:
        return []  # not a complaint, empathy not required

    markers = EMPATHY_MARKERS.get(language, [])
    rl = response.lower()
    if not any(m.lower() in rl for m in markers):
        return [f"missing empathy marker for {intent} in {language}"]
    return []
```

### 2.4 Per-category content requirements

```python
CATEGORY_RESPONSE_REQUIREMENTS = {
    "quality_complaint": {
        "must_contain_one_of": ["photo", "image", "picture",
                                "foto", "imagen"],
        "reason": "Quality policy requires photographic evidence",
    },
    "design_update": {
        "must_contain_one_of": ["vector", ".AI", ".EPS", ".PDF",
                                "file", "fichier", "Datei"],
        "reason": "Need vector source files",
    },
    "refund_request": {
        "must_contain_one_of": ["day", "business", "process", "review",
                                "eligible", "jour", "Tag", "día"],
        "reason": "Customer needs timeline and eligibility info",
    },
    # ... etc per category
}

def check_category_content(response: str, intent: str) -> list[str]:
    req = CATEGORY_RESPONSE_REQUIREMENTS.get(intent)
    if not req:
        return []
    rl = response.lower()
    if not any(kw.lower() in rl for kw in req["must_contain_one_of"]):
        return [f"{intent}: response missing required content ({req['reason']})"]
    return []
```

### 2.5 Combined quality filter

```python
def assess_response_quality(trace: dict) -> tuple[str, list[str]]:
    """
    Returns (verdict, issues) where verdict is:
      'good' — keep as-is
      'fixable' — re-generate with stricter prompt
      'unfixable' — drop
    """
    final = trace["final_response"]
    response = final["response"]
    intent = final["intent"]
    language = final.get("language", "english")

    all_issues = []
    all_issues.extend(check_warmth(response, language))
    all_issues.extend(check_empathy(response, language, intent))
    all_issues.extend(check_category_content(response, intent))

    if is_template_response(response):
        all_issues.append("template response without substance")

    if not all_issues:
        return ("good", [])

    # Decide if fixable
    if any("template" in i or "missing.*content" in i for i in all_issues):
        return ("fixable", all_issues)

    # Warmth/empathy issues are usually fixable too
    return ("fixable", all_issues)
```

### 2.6 Re-generation with stricter prompt

```python
WARM_RESPONSE_REGEN_PROMPT = """The previous response was inadequate:

{issues}

Rewrite the response to fix these specific issues. Required elements:

1. **Greeting**: Use customer's name ({customer_name}) if available, in {language}
2. **Acknowledgment**: Reference the specific concern (order #{order_number}, {category})
3. **Empathy**: For {category} tickets in {language}, use a culturally appropriate empathy phrase
4. **Substance**: Cite the actual policy/timeline from the retrieved KB context, not vague "I'll check"
5. **Forward-looking close**: End with warmth in {language}

KB context retrieved:
{rag_results}

Customer ticket:
{ticket_text}

Original (inadequate) response:
{original_response}

Rewrite the response. Output only the response text, NO JSON wrapping.
"""

def regenerate_response(trace, issues):
    # Call teacher agent with the regen prompt
    # Validate the new response
    # Return new trace if valid, None if still bad after 2 retries
    pass
```

### Stage 2 deliverables

- `data/v2/quality_filtered_traces.jsonl` — traces with quality verdict
- `data/v2/regen_log.jsonl` — record of all re-generations attempted
- `data/v2/quality_stats.json` — % good, fixable, unfixable, regen success rate

**Stage 2 gate:**

```python
def stage_2_validate(stats: dict) -> bool:
    if stats["good"] / stats["total"] < 0.70:
        print(f"❌ Stage 2: only {stats['good']/stats['total']:.0%} traces are 'good' quality")
        return False
    if stats["unfixable"] / stats["total"] > 0.10:
        print(f"❌ Stage 2: too many unfixable ({stats['unfixable']/stats['total']:.0%})")
        return False
    return True
```

**Manual checkpoint:** read 10 "good" traces and 10 re-generated ones. Are the re-generated responses actually better? If they look the same as the originals, refine the regen prompt.

---

## Stage 3: ChatML construction

Convert validated traces into the Qwen3 ChatML training format.

### 3.1 Build messages list

```python
def build_chatml(trace: dict, ticket: dict) -> dict:
    """Convert a validated trace to ChatML training example."""
    messages = []

    # System prompt — from configs/loopper_pipeline.yaml
    messages.append({
        "role": "system",
        "content": load_system_prompt(),
    })

    # User message — the formatted ticket
    messages.append({
        "role": "user",
        "content": format_ticket_text(ticket),
    })

    # For each rag call, add assistant turn + tool result
    for i, call in enumerate(trace["rag_calls"]):
        # Reasoning before this call
        reasoning = synthesize_reasoning_before_call(trace, i)

        # Assistant message with tool call
        messages.append({
            "role": "assistant",
            "content": "",  # tool-call turns have empty content
            "reasoning_content": reasoning,
            "tool_calls": [{
                "type": "function",
                "id": f"call_{i:03d}_{trace['ticket_id']}",
                "function": {
                    "name": "rag_search",
                    "arguments": json.dumps(call["arguments"]),
                },
            }],
        })

        # Tool result message
        messages.append({
            "role": "tool",
            "tool_call_id": f"call_{i:03d}_{trace['ticket_id']}",
            "name": "rag_search",
            "content": json.dumps(trim_rag_results(call["results"])),
        })

    # Final assistant turn — the JSON output
    final = trace["final_response"]
    final_reasoning = synthesize_final_reasoning(trace)
    messages.append({
        "role": "assistant",
        "reasoning_content": final_reasoning,
        "content": json.dumps(final, ensure_ascii=False),
    })

    return {
        "messages": messages,
        "tools": load_tool_schema(),
        "metadata": {
            "ticket_id": ticket["ticket_id"],
            "intent": final["intent"],
            "language": detect_language(messages[1]["content"]),
            "is_synthetic": ticket.get("is_synthetic", False),
            "num_tool_calls": len(trace["rag_calls"]),
            "is_valid_ticket": final["is_valid"],
        },
    }
```

### 3.2 Per-example structural validation

```python
def validate_chatml_example(ex: dict) -> list[str]:
    """Strict structural checks on a single ChatML example."""
    issues = []

    # 1. Must have messages and tools
    if "messages" not in ex or "tools" not in ex:
        issues.append("missing messages or tools")
        return issues

    msgs = ex["messages"]

    # 2. Must start with system, then user
    if msgs[0]["role"] != "system":
        issues.append("first message not system")
    if msgs[1]["role"] != "user":
        issues.append("second message not user")

    # 3. Tool-call assistant turns must have empty content + tool_calls
    for i, m in enumerate(msgs):
        if m["role"] == "assistant" and m.get("tool_calls"):
            if m.get("content") not in ("", None):
                issues.append(f"msg {i}: tool-call turn has non-empty content")

    # 4. Every tool message must follow an assistant tool_calls message
    for i, m in enumerate(msgs):
        if m["role"] == "tool":
            if i == 0 or msgs[i-1]["role"] != "assistant":
                issues.append(f"msg {i}: tool message not preceded by assistant")
            if not m.get("tool_call_id"):
                issues.append(f"msg {i}: tool message missing tool_call_id")

    # 5. Final assistant message must have valid JSON content
    final = msgs[-1]
    if final["role"] != "assistant":
        issues.append("last message not assistant")
    elif not final.get("content"):
        issues.append("final assistant message has empty content")
    else:
        try:
            parsed = json.loads(final["content"])
            required = {"intent", "urgency", "confidence", "is_valid",
                       "rejection_type", "resolution_type", "team",
                       "actions", "summary", "reasoning", "response"}
            missing = required - set(parsed.keys())
            if missing:
                issues.append(f"final JSON missing fields: {missing}")
        except json.JSONDecodeError as e:
            issues.append(f"final JSON parse failed: {e}")

    # 6. Tool call arguments must parse as JSON
    for i, m in enumerate(msgs):
        if m["role"] == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                args_str = tc["function"]["arguments"]
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError as e:
                    issues.append(f"msg {i} tool_call args malformed: {e}")
                    continue
                if "collection" not in args or "query" not in args:
                    issues.append(f"msg {i} tool_call args missing required fields")

    # 7. Tool-call invariant: valid tickets MUST have at least one tool call
    final_parsed = json.loads(final["content"]) if final.get("content") else {}
    if final_parsed.get("is_valid"):
        has_tool_call = any(
            m["role"] == "assistant" and m.get("tool_calls")
            for m in msgs
        )
        if not has_tool_call:
            issues.append("INVARIANT VIOLATION: valid ticket with zero tool calls")

    return issues
```

### 3.3 Token count check

```python
def check_token_count(ex: dict, max_seq_len: int = 4096) -> int:
    """Render through tokenizer and count tokens. Returns token count."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("unsloth/Qwen3-4B-Thinking-2507")

    rendered = tok.apply_chat_template(
        ex["messages"],
        tools=ex.get("tools"),
        tokenize=True,
    )
    return len(rendered)
```

If an example exceeds `max_seq_len`, you have two choices:
1. Drop it (loses data)
2. Trim the RAG results in tool messages (already done by `trim_chatml.py` in v1)

### Stage 3 deliverables

- `data/v2/chatml_raw.jsonl` — every passing example
- `data/v2/chatml_failed.jsonl` — examples that failed structural validation
- `data/v2/chatml_token_stats.json` — distribution of token counts

---

## Stage 4: Distribution balancing

Now look at the current distribution and balance it.

```python
from collections import Counter

def analyze_distribution(examples: list) -> dict:
    """Compute all the distribution stats we care about."""
    intents = Counter(ex["metadata"]["intent"] for ex in examples)
    languages = Counter(ex["metadata"]["language"] for ex in examples)
    teams = Counter(
        json.loads(ex["messages"][-1]["content"])["team"]
        for ex in examples
    )
    urgencies = Counter(
        json.loads(ex["messages"][-1]["content"])["urgency"]
        for ex in examples
    )
    is_valid_counts = Counter(
        json.loads(ex["messages"][-1]["content"])["is_valid"]
        for ex in examples
    )
    is_synthetic = Counter(ex["metadata"]["is_synthetic"] for ex in examples)

    return {
        "total": len(examples),
        "by_intent": dict(intents),
        "by_language": dict(languages),
        "by_team": dict(teams),
        "by_urgency": dict(urgencies),
        "by_validity": dict(is_valid_counts),
        "by_source": dict(is_synthetic),
    }

def balance_distribution(examples: list, targets: dict) -> list:
    """Oversample underrepresented, downsample overrepresented."""
    by_intent = {}
    for ex in examples:
        intent = ex["metadata"]["intent"]
        by_intent.setdefault(intent, []).append(ex)

    balanced = []
    for intent, examples_for_intent in by_intent.items():
        target = targets.get(intent, len(examples_for_intent))
        actual = len(examples_for_intent)

        if actual >= target:
            # Downsample
            import random
            random.seed(42)
            balanced.extend(random.sample(examples_for_intent, target))
        else:
            # Oversample (with replacement, but mark duplicates)
            balanced.extend(examples_for_intent)  # all originals first
            shortfall = target - actual
            for i in range(shortfall):
                ex = examples_for_intent[i % actual].copy()
                ex["metadata"]["is_oversample"] = True
                ex["metadata"]["oversample_idx"] = i
                balanced.append(ex)

    return balanced

V2_TARGETS = {
    "design_update": 500,
    "delivery_issue": 500,
    "new_order_inquiry": 500,
    "tracking_inquiry": 350,         # NEW
    "quality_complaint": 350,
    "order_cancellation": 350,
    "payment_confirmation": 350,
    "refund_request": 350,
    "documentation_request": 300,    # NEW
    "sample_request": 300,
    "price_negotiation": 300,
    "lead_time_inquiry": 250,        # NEW
    "customer_feedback": 250,
    "product_support": 200,          # NEW
    "other": 250,                    # capped
    # Rejections (is_valid=false), distributed across rejection_types
    "rejections_total": 500,
}
```

### Stage 4 gate

```python
def stage_4_validate(stats: dict, targets: dict) -> bool:
    failures = []
    for intent, target in targets.items():
        actual = stats["by_intent"].get(intent, 0)
        # Allow ±10% slack
        if actual < target * 0.9 or actual > target * 1.1:
            failures.append(f"{intent}: have {actual}, target {target}")

    if failures:
        print("❌ Stage 4: distribution targets not met:")
        for f in failures:
            print(f"   {f}")
        return False

    # Check critical urgency
    if stats["by_urgency"].get("critical", 0) < 50:
        print(f"❌ Stage 4: only {stats['by_urgency'].get('critical', 0)} critical urgency examples (need ≥50)")
        return False

    # Check finance team
    if stats["by_team"].get("finance", 0) < 200:
        print(f"❌ Stage 4: only {stats['by_team'].get('finance', 0)} finance team (need ≥200)")
        return False

    return True
```

### Stage 4 deliverables

- `data/v2/chatml_balanced.jsonl` — balanced dataset
- `data/v2/balance_report.md` — before/after distribution comparison

---

## Stage 5: Final validation gates (CRITICAL)

This is the last chance to catch problems before training. Run ALL checks. Failure of ANY check blocks training.

```python
GATES = [
    ("structural_validity",      gate_all_examples_pass_structural),
    ("tool_call_invariant",      gate_valid_tickets_have_tool_calls),
    ("json_strict_parse",        gate_final_json_parses_with_all_fields),
    ("collection_names",         gate_all_collections_in_enum),
    ("token_count_under_limit",  gate_token_counts_under_max),
    ("warmth_per_language",      gate_responses_have_greetings_and_closings),
    ("empathy_on_complaints",    gate_complaints_have_empathy),
    ("category_content",         gate_responses_meet_category_requirements),
    ("distribution_targets",     gate_distribution_meets_targets),
    ("urgency_diversity",        gate_critical_urgency_present),
    ("language_coverage",        gate_all_six_languages_represented),
    ("team_coverage",            gate_all_teams_represented),
    ("rejection_distinct",       gate_rejections_structurally_short),
    ("no_pii_leakage",           gate_no_real_pii_in_examples),
    ("no_template_responses",    gate_template_response_rate_under_10pct),
]

def run_final_validation(examples_path: str) -> bool:
    print(f"\n{'='*70}\nFINAL VALIDATION — {examples_path}\n{'='*70}\n")

    examples = list(load_jsonl(examples_path))
    print(f"Loaded {len(examples)} examples\n")

    results = {}
    for gate_name, gate_fn in GATES:
        print(f"Running gate: {gate_name}...", end=" ", flush=True)
        try:
            passed, details = gate_fn(examples)
            results[gate_name] = {"passed": passed, "details": details}
            print("✓ PASS" if passed else "✗ FAIL")
            if not passed:
                for line in details[:10]:
                    print(f"     {line}")
                if len(details) > 10:
                    print(f"     ... and {len(details)-10} more issues")
        except Exception as e:
            print(f"✗ ERROR: {e}")
            results[gate_name] = {"passed": False, "error": str(e)}

    failed = [name for name, r in results.items() if not r["passed"]]
    if failed:
        print(f"\n{'='*70}\n❌ {len(failed)} GATES FAILED — DO NOT TRAIN\n{'='*70}")
        print(f"Failed gates: {failed}")
        save_validation_report(results, "data/v2/validation_report.md")
        return False

    print(f"\n{'='*70}\n✓ ALL {len(GATES)} GATES PASSED — READY TO TRAIN\n{'='*70}")
    return True
```

**Manual checkpoint:** read 10 random examples in detail, end-to-end. Pretend you're the model. Would these examples teach you what you need to know? If any look weird, debug before training.

---

## Stage 6: Train/eval split (stratified)

```python
import random
from collections import defaultdict

def stratified_split(examples: list, eval_ratio: float = 0.10, seed: int = 42) -> tuple[list, list]:
    """Split per-intent so every category has eval representation."""
    random.seed(seed)
    by_intent = defaultdict(list)
    for ex in examples:
        by_intent[ex["metadata"]["intent"]].append(ex)

    train, eval_set = [], []
    for intent, examples_for_intent in by_intent.items():
        random.shuffle(examples_for_intent)
        n_eval = max(10, int(len(examples_for_intent) * eval_ratio))
        eval_set.extend(examples_for_intent[:n_eval])
        train.extend(examples_for_intent[n_eval:])

    random.shuffle(train)
    random.shuffle(eval_set)
    return train, eval_set
```

**Stage 6 gate:**

```python
def stage_6_validate(train: list, eval_set: list) -> bool:
    train_intents = Counter(ex["metadata"]["intent"] for ex in train)
    eval_intents = Counter(ex["metadata"]["intent"] for ex in eval_set)

    for intent in train_intents:
        if eval_intents.get(intent, 0) < 10:
            print(f"❌ Stage 6: {intent} has only {eval_intents.get(intent, 0)} eval examples (need ≥10)")
            return False
    return True
```

---

## Stage 7: Pre-training smoke test

Final paranoia check: render 5 random examples through the actual Qwen3 tokenizer and verify the output looks sane.

```python
def stage_7_smoke_test(train_path: str) -> bool:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("unsloth/Qwen3-4B-Thinking-2507")

    examples = list(load_jsonl(train_path))
    sample = random.sample(examples, 5)

    for i, ex in enumerate(sample):
        print(f"\n--- SMOKE TEST EXAMPLE {i+1}/5 ---")
        rendered = tok.apply_chat_template(
            ex["messages"],
            tools=ex["tools"],
            tokenize=False,
        )

        # Check for known patterns
        checks = [
            ("contains <think>",          "<think>" in rendered),
            ("contains </think>",         "</think>" in rendered),
            ("contains <tool_call>",      "<tool_call>" in rendered),
            ("contains </tool_call>",     "</tool_call>" in rendered),
            ("contains <tool_response>",  "<tool_response>" in rendered or "tool_response" in rendered),
            ("ends with <|im_end|>",      "<|im_end|>" in rendered[-100:]),
            ("no Python literals",        "True" not in rendered.replace('"True"', '') and "None" not in rendered.replace('null', '')),
            ("no trailing comma in JSON", "}}, " not in rendered.replace(']}}, ', ']},,, ')),  # heuristic
        ]

        all_passed = True
        for check_name, passed in checks:
            symbol = "✓" if passed else "✗"
            print(f"  {symbol} {check_name}")
            if not passed:
                all_passed = False

        if not all_passed:
            print("\n  Sample rendering (first 1000 chars):")
            print(rendered[:1000])
            return False

    print("\n✓ All 5 smoke test examples passed")
    return True
```

---

## Tooling for debugging failures

When a gate fails, you need to debug WHY. Provide tooling:

```python
# scripts/loopper/inspect_example.py
"""Inspect a specific training example — render it, validate it, show details."""

def inspect_example(jsonl_path: str, ticket_id: str = None, idx: int = None):
    examples = list(load_jsonl(jsonl_path))

    if ticket_id:
        ex = next(e for e in examples if e["metadata"]["ticket_id"] == ticket_id)
    elif idx is not None:
        ex = examples[idx]
    else:
        import random
        ex = random.choice(examples)

    print(f"\n{'='*70}\nINSPECTING: {ex['metadata']['ticket_id']}\n{'='*70}")

    # Show metadata
    print("\n[METADATA]")
    print(json.dumps(ex["metadata"], indent=2))

    # Show user message
    user = next(m for m in ex["messages"] if m["role"] == "user")
    print(f"\n[USER MESSAGE — {len(user['content'])} chars]")
    print(user["content"][:500] + ("..." if len(user["content"]) > 500 else ""))

    # Show all assistant turns
    for i, m in enumerate(ex["messages"]):
        if m["role"] == "assistant":
            print(f"\n[ASSISTANT TURN {i}]")
            if m.get("reasoning_content"):
                print(f"  reasoning: {m['reasoning_content'][:200]}")
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    print(f"  tool_call: {tc['function']['name']}({tc['function']['arguments']})")
            if m.get("content"):
                print(f"  content: {m['content'][:300]}")

    # Run all validations
    print("\n[VALIDATION RESULTS]")
    issues = validate_chatml_example(ex)
    if issues:
        for i in issues:
            print(f"  ❌ {i}")
    else:
        print("  ✓ All validations passed")

# Usage:
#   python scripts/loopper/inspect_example.py --idx 42
#   python scripts/loopper/inspect_example.py --ticket-id 5517
#   python scripts/loopper/inspect_example.py --random
```

---

## The pre-training checklist

Before kicking off training, every box must be checked:

- [ ] Stage 0: source pool has ≥targets per category
- [ ] Stage 0: synthetic tickets validated (no LLM artifacts, plausible language)
- [ ] Stage 0: deduplication done (no exact text duplicates)
- [ ] Stage 1: trace pass rate ≥85%
- [ ] Stage 1: every valid ticket has ≥1 tool call (invariant check)
- [ ] Stage 2: ≥70% of traces are "good" quality
- [ ] Stage 2: re-generated responses spot-checked manually (10 examples)
- [ ] Stage 3: every example passes structural validation
- [ ] Stage 3: every example fits in 4096 tokens (or trimmed appropriately)
- [ ] Stage 4: distribution targets met (±10%)
- [ ] Stage 4: ≥50 critical urgency examples
- [ ] Stage 4: ≥200 finance team examples
- [ ] Stage 5: ALL 15 final gates pass (no warnings, only PASS)
- [ ] Stage 6: every category has ≥10 eval examples
- [ ] Stage 7: smoke test renders all 5 samples without artifacts
- [ ] Manual: 10 random end-to-end examples reviewed by you personally
- [ ] Manual: distribution report reviewed and approved

If any item is unchecked, do NOT train. Fix the issue. Re-run validation.

---

## Common failure modes and how to fix them

### "Trace pass rate is only 40%"
- Read 20 dropped traces. Find the most common drop reason.
- If it's "missing tool calls": the teacher agent's prompt isn't strict enough → fix the prompt
- If it's "malformed JSON": the teacher's response generation has a bug → fix the prompt or post-process

### "Distribution targets aren't met for refund_request"
- Source pool didn't have enough refund tickets
- Either: (a) generate more synthetic refund tickets and re-run from Stage 0
- Or: (b) lower the target if it's truly unattainable

### "Critical urgency count is 0"
- The teacher agent isn't producing critical urgency
- Manually create 50 critical urgency examples by promoting high urgency tickets
- Add them to the source pool as a special category

### "Token count check fails on 30% of examples"
- RAG results are too verbose
- Increase trimming aggressiveness in `trim_chatml.py`
- Reduce max_results from 5 → 3
- Reduce text per result from 800 → 500 chars

### "Empathy gate fails for German complaints"
- Teacher agent's German responses lack empathy markers
- Check the response generation prompt — add explicit instructions for German empathy phrases
- Re-run Stage 2 on German complaint tickets

### "Token count varies wildly (some 200 tokens, some 4000)"
- Examples are inconsistent
- This is OK if both extremes are reasonable
- But if very-short examples lack reasoning_content, they're missing the chain-of-thought training signal — fix that

---

## A summary you can put on your wall

```
Build dataset →           ← Validate every step
Synthesize what's missing → ← Drop what can't be fixed
Balance distributions →    ← Don't over-rely on synthetic data
Validate structurally →    ← Cleanup at every stage
Validate semantically →    ← Don't trust the teacher blindly
Spot-check manually →      ← Trust your eyes more than metrics
Smoke test rendering →     ← Tokenizer is the source of truth
ALL gates pass → train     ← One failing gate = NO training
```

The single most important thing: **fail loudly and early.** A failing pipeline is a successful pipeline — it caught the problem before it cost you a 12-hour training run on bad data.

If you build this pipeline and ALL gates pass, you'll have a v2 dataset that won't have the v1 problems. The model can still fall short for other reasons, but it won't be data quality.
