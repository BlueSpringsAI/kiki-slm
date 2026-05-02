# Kiki SLM v2 — Training Strategy & Dataset Preparation Guide

## Executive Summary

v1 shipped a working SLM that handles tool calling, multilingual responses, and structured JSON output. But production testing on ~50 real tickets revealed four quality gaps:

1. **Model skips `rag_search`** on ~20-30% of valid tickets → ungrounded responses that hallucinate policy
2. **Responses are generic** ("I'll check shortly") instead of citing actual KB policy or asking for specific follow-ups
3. **Rare categories fail** — refund_request (30 examples), price_negotiation (44), customer_feedback (33) are barely learned
4. **Tone is transactional**, not warm — bare "Hello," greetings, no customer name personalization, formulaic closings instead of language-appropriate warmth (French "Cordialement", German "Mit freundlichen Grüßen", etc.)

All three trace to the same root cause: **training data distribution and quality**, not model architecture or hyperparameters.

v2 fixes the data. The architecture stays the same (single LoRA adapter, Qwen3-4B-Thinking-2507). The expected outcome: intent accuracy from 51% → 75%+, tool-call rate on valid tickets from ~70% → 95%+, and customer-facing response quality matching the OpenAI teacher agent.

---

## Part 1 — What Went Wrong in v1 (Root Cause Analysis)

### 1.1 The tool-skipping failure

**Observed:** On tickets like "Clarification on Refund Policy for Delayed Orders", the model generates `<think>I need to search faq and operations</think>` then immediately outputs the final JSON without calling `rag_search`. The response contains `"policy_used": "No policy consulted"` and invents policy from pretraining.

**Root cause from data analysis:**

The training data has two structural patterns:

```
Pattern A (82% of training — valid tickets):
  <think>analysis</think> → <tool_call>rag_search</tool_call> → tool result → <think>more</think> → JSON

Pattern B (18% of training — rejections):
  <think>this is spam</think> → JSON with is_valid=false
```

Both patterns share the same prefix: `<think>text</think>`. The model's decision to call a tool vs emit JSON happens at the token right after `</think>`. With 18% of examples teaching "after `</think>`, emit `{`", the model sometimes takes that path on valid tickets.

The overgeneralization is worse on:
- Formal, third-person phrasing ("I wanted to understand your policy...")
- Informational requests (no explicit "help me" urgency)
- Tickets in the model's pretraining domain (the model "knows" enough to guess)

**The system prompt's exception clause** — `"exception: simple acknowledgments like 'OK thank you'"` — gave the model explicit permission to skip, which it over-applied.

**Key finding:** All 3,362 valid training examples DO have tool calls. The training data is clean. This is a generalization problem, not a data corruption problem.

### 1.2 The generic response problem

**Observed:** 27.4% of v1 training responses contain "shortly". The model faithfully reproduces the teacher's template: "Hello, thank you for your message. I'm checking with our [team]. I'll be in touch shortly."

**Root cause:** The teacher agent (LangGraph with o4-mini) generates these templates because its compose_response prompt emphasizes speed ("respond promptly") over specificity. The KB context IS used by the teacher, but the resulting response text is still templated. The student learned the template, not the grounding.

Compare what the teacher actually produces vs what the SLM reproduces:

```
Teacher (with KB context):
  "Per our delivery guarantee, orders delayed beyond 5 business days qualify
   for a 10% credit. I'm checking your order status now and will confirm
   the specific delay timeline shortly."

SLM (without KB context, reproducing the template):
  "Hello, thank you for your message. I'm checking with our logistics team
   and will update you shortly."
```

Both contain "shortly". But the teacher's version has a policy citation ("5 business days", "10% credit"). The SLM's version strips the substance and keeps the structure.

### 1.3 The transactional tone problem

**Observed:** v1 responses lack warmth and personalization compared to the teacher:

| Element | Teacher | v1 SLM |
|---|---|---|
| Greeting | "Hello Camille" (uses signed name) | "Hello," (generic) |
| Acknowledgment | "about your order ORD-7745, I understand the urgency" | "Thank you for your message" |
| Empathy on complaints | "We regret the issue with your order" | (often absent) |
| Closing | "Looking forward to resolving this for you soon" | "I'll be in touch shortly" |
| Language closings | "Cordialement", "Mit freundlichen Grüßen", "Cordiali saluti" | English-style "Best regards" leaks into all languages |

**Root cause:**
1. **Customer name not surfaced in reasoning** — the training reasoning_content doesn't explicitly extract and track the customer's name from the ticket signature, so the model doesn't learn to use it
2. **Response length compression** — median response is 244 chars (very short). When the model compresses, the warmth elements get cut first because they're "optional"
3. **No per-language tone validation** — the training data has French tickets that use English-style closings ("Best regards" instead of "Cordialement") because the teacher agent's response generation isn't language-aware enough
4. **No empathy-marker training** — complaints, refunds, and quality issues need empathy phrases ("I'm sorry to hear", "We regret"), but these aren't consistently present in v1 training responses

### 1.4 The class imbalance catastrophe

```
Category             v1 Count    % of Data    Model's Learned Priority
─────────────────────────────────────────────────────────────────────
design_update          1,030       25.1%       ████████████████ Primary
other                    879       21.4%       █████████████ (waste)
delivery_issue           873       21.3%       █████████████ Primary
new_order_inquiry        677       16.5%       ██████████ Secondary
quality_complaint        168        4.1%       ███ Weak
order_cancellation       147        3.6%       ██ Weak
payment_confirmation     140        3.4%       ██ Weak
sample_request            78        1.9%       █ Very weak
price_negotiation         44        1.1%       ▌ Nearly absent
customer_feedback         33        0.8%       ▌ Nearly absent
refund_request            30        0.7%       ▌ Nearly absent
```

The model saw `refund_request` 30 times across 3 epochs = 90 total exposures. In contrast, `design_update` got 3,090 exposures. The model literally doesn't know what a refund looks like.

Additional gaps:
- **Zero `critical` urgency examples** — model can never assign critical
- **Finance team: 79 examples** (1.9%) — refund/payment tickets misroute
- **`other` intent at 21.4%** — this is a waste bucket, many are classifiable

---

## Part 2 — v2 Architecture Decision: Single Adapter

### Why single adapter is correct for v2

Three adapter strategies were evaluated:

| Strategy | Description | Pros | Cons |
|---|---|---|---|
| **Single adapter** | One LoRA for all tasks | Simple serving, single inference pass, proven | Must learn 3 skills at once |
| **Per-task adapters** | Triage adapter + response adapter | Each specializes | 2x latency, tool calling becomes hardcoded |
| **Per-category adapters** | 11 LoRA experts | Deep specialization | 11x training cost, complex routing |

**Decision: single adapter, rank increased from 32 → 64.**

Rationale:
1. The v1 failures are **data problems**, not capacity problems. Rank 32 is enough for 11-way classification + tool calling + response generation. Doubling to 64 gives headroom without architectural changes.
2. Per-task adapters would **hardcode tool calling** between the adapters, removing the model's agency to decide when to search. This conflicts with the design goal of the SLM replacing the full pipeline in one call.
3. Per-category adapters are overkill at 11 categories. They make sense at 100+ categories.
4. vLLM does support dynamic LoRA swapping, but adding this complexity before fixing the data is premature optimization.

### v2 hyperparameter changes

```
Parameter            v1 Value    v2 Value    Rationale
────────────────────────────────────────────────────────────────
LoRA rank            32          64          More capacity for rare categories
Alpha                64          128         Maintain alpha/rank ratio at 2.0
Dropout              0           0.05        Mild regularization against overfitting on oversampled data
Learning rate        2e-4        1.5e-4      Slightly lower with more data to prevent overfitting
Epochs               3           3           Same (3 epochs on 8-10K = enough)
Batch size           auto        auto        Keep GPU-profile auto-detection
Max seq length       4096        4096        No change (86% fit in v1, should be similar)
Packing              True        True        Keep for throughput
Loss masking         True        True        train_on_responses_only
Optimizer            adamw_8bit  adamw_8bit  No change
Warmup ratio         0.03        0.05        Slightly longer warmup for larger dataset
```

---

## Part 3 — Dataset Preparation Pipeline

### 3.1 Target distribution

**Total target: 8,000–10,000 examples** (vs 4,099 in v1).

**Per-category targets:**

```
Category                v1 Count → v2 Target    Source
───────────────────────────────────────────────────────────────────────
refund_request              30 →   350          Filtered Freshdesk + synthetic
customer_feedback           33 →   250          Filtered Freshdesk + synthetic
price_negotiation           44 →   300          Filtered Freshdesk + synthetic
sample_request              78 →   300          Filtered Freshdesk
payment_confirmation       140 →   350          Filtered Freshdesk
order_cancellation         147 →   350          Filtered Freshdesk
quality_complaint          168 →   350          Filtered Freshdesk
new_order_inquiry          677 →   500          Downsample
delivery_issue             873 →   500          Downsample
design_update            1,030 →   500          Downsample
other                      879 →   250          Reclassify most, keep only true "other"
───────────────────────────────────────────────────────────────────────
Rejections (is_valid=false) 737 →   500          Downsample from 18% to ~6%
───────────────────────────────────────────────────────────────────────
TOTAL                    4,099 →  4,500 base
+ DPO pairs                  0 →    200          Preference pairs for tool-calling
───────────────────────────────────────────────────────────────────────
```

### 3.2 Acquiring rare-category examples

**Step 1 — Filter Freshdesk tickets by category keyword**

```python
# In scripts/loopper/sample_tickets_v2.py
CATEGORY_KEYWORDS = {
    "refund_request": ["refund", "reimburse", "money back", "credit", "compensation",
                       "remboursement", "Erstattung", "rimborso", "reembolso"],
    "price_negotiation": ["discount", "price", "bulk", "quote", "negotiate",
                          "prix", "Preis", "precio", "prezzo"],
    "sample_request": ["sample", "test", "try", "échantillon", "Muster",
                       "muestra", "campione"],
    "customer_feedback": ["feedback", "review", "experience", "suggest",
                          "retour", "Feedback", "comentario"],
}

# For each category, search Freshdesk for tickets containing these keywords
# This gives us REAL tickets to run through the teacher agent
```

**Step 2 — Generate synthetic tickets for categories where Freshdesk doesn't have enough**

```python
# Use Claude/GPT to generate realistic ticket text
# Then run the teacher agent on them as if they were real
SYNTHETIC_PROMPT = """
Generate a realistic B2B customer support ticket for the category: {category}
The customer is a European business that buys promotional products (mugs, pens, etc.)
from Loopper.

Requirements:
- Write in one of: English, French, German, Dutch, Spanish, Italian
- Include realistic order numbers, product names, dates
- Use the tone of a real business email (not overly casual or formal)
- Length: 3-8 sentences
- Include at least one specific question or request

Generate 10 unique tickets, each different in tone, language, and scenario.
"""
```

**Step 3 — Run teacher agent on all acquired tickets**

```bash
# Same pipeline as v1, but with the expanded ticket pool
uv run --project ../Looper-Support-Agent-Server python -m scripts.loopper.generate_traces \
    --sample 8000 --concurrency 5 --resume
```

### 3.3 Reclassifying "other" intent

879 tickets are classified as "other" (21.4%). Many are real support requests the teacher agent failed to categorize. Re-run classification with an improved triage prompt:

```python
# In generate_traces_v2.py
RECLASSIFY_PROMPT = """
This ticket was previously classified as "other". Re-evaluate using this
stricter definition:

"other" means ONLY: the ticket has no actionable support content AND is valid
(not spam/newsletter/auto-reply). Examples: generic greetings ("Hello"),
test messages, internal forwards with no customer question.

If the ticket contains ANY identifiable intent (order question, delivery check,
design request, pricing inquiry, complaint, feedback), classify it as that
intent instead of "other".
"""
```

Target: reduce "other" from 879 to ~250 (reclassify ~600 into proper categories).

### 3.4 Adding critical urgency

v1 had zero critical examples. Create 50-80 by:

1. **Promote from high urgency** — take existing high-urgency tickets and amplify:
   - Add deadline pressure ("event is TOMORROW")
   - Add financial stakes ("€50,000 order at risk")
   - Add legal/compliance context ("GDPR audit requires this documentation by Friday")

2. **Synthesize new critical scenarios:**
   - Health/safety complaint about a product
   - Data breach or privacy concern
   - Legal demand or regulatory deadline
   - Complete production failure affecting a customer's event

### 3.5 Boosting finance team routing

Only 79 examples route to `finance` (1.9%). Ensure:
- All `refund_request` → `finance` team (currently split between finance and account_manager)
- All `payment_confirmation` with issues → `finance` team
- Price negotiations involving credit terms → `finance` team

Target: 200+ examples routing to finance.

---

## Part 4 — Fixing Tool Calling in Training Data

### 4.1 The invariant: every valid ticket must search

**Rule:** If `is_valid == true` in the final JSON, the conversation history MUST contain at least one `rag_search` tool call.

**Validation in `build_chatml_v2.py`:**

```python
def validate_tool_call_invariant(example: dict) -> bool:
    """Every valid ticket must have at least one tool call."""
    messages = example.get("messages", [])
    
    # Find final JSON
    final_content = ""
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content"):
            final_content = m["content"]
            break
    
    try:
        parsed = json.loads(final_content)
    except json.JSONDecodeError:
        return False  # malformed, reject
    
    if not parsed.get("is_valid"):
        return True  # rejections are exempt
    
    # Count tool calls
    tool_calls = sum(
        len(m.get("tool_calls", []))
        for m in messages
        if m.get("role") == "assistant"
    )
    
    if tool_calls == 0:
        logger.warning(f"REJECTED: valid ticket with 0 tool calls")
        return False
    
    return True
```

### 4.2 Making rejections structurally distinct

v1 rejections and valid tickets both start with detailed `<think>` blocks. Make rejections shorter and more distinctive:

```python
# In build_chatml_v2.py — when constructing rejection reasoning_content
def build_rejection_reasoning(ticket_text: str, rejection_type: str) -> str:
    """Short, distinctive reasoning for rejection tickets.
    
    v1 problem: rejection reasoning was as detailed as valid ticket reasoning,
    making the two patterns hard to distinguish. v2 uses a short, declarative
    format.
    """
    return (
        f"Rejection: {rejection_type}. "
        f"This message is not a Loopper support request. No action needed."
    )

# vs valid ticket reasoning (stays detailed):
def build_valid_reasoning(intent: str, signals: list[str], search_plan: list[str]) -> str:
    return (
        f"Primary intent: {intent}\n"
        f"Key signals: {', '.join(signals)}\n"
        f"Search plan:\n" +
        "\n".join(f"- {s}" for s in search_plan)
    )
```

This creates a clear structural distinction:
- **Rejection:** "Rejection: spam. This message is not a Loopper support request. No action needed." → JSON
- **Valid:** "Primary intent: delivery_issue\nKey signals: delayed, 3 days\nSearch plan:\n- faq: delivery policy\n- guidelines: tone" → `<tool_call>`

### 4.3 Adding "simple but valid" examples WITH tool calls

The model skips on simple-looking tickets. Add 50-100 examples where trivial messages still call `rag_search`:

```python
SIMPLE_VALID_EXAMPLES = [
    {
        "ticket": "Thanks for the update!",
        "intent": "customer_feedback",
        "reasoning": "Simple acknowledgment, but checking tone guidelines for best follow-up",
        "tool_call": {"collection": "communication_guidelines", "query": "acknowledgment follow-up tone"},
        "response": "You're welcome! If anything else comes up, don't hesitate to reach out.",
    },
    {
        "ticket": "Got it, merci.",
        "intent": "customer_feedback",
        "reasoning": "Brief confirmation in French. Checking communication guidelines for appropriate close",
        "tool_call": {"collection": "communication_guidelines", "query": "brief confirmation response closing"},
        "response": "Avec plaisir ! N'hésitez pas si vous avez d'autres questions.",
    },
    {
        "ticket": "Can you send the invoice please?",
        "intent": "payment_confirmation",
        "reasoning": "Invoice request. Need to check invoicing procedure.",
        "tool_call": {"collection": "sales_operations_playbook", "query": "invoice request fulfillment process"},
        "response": "Of course! I'm pulling up your invoice now and will email it to you within the hour.",
    },
]
```

These examples teach: **even when the ticket looks simple, ground your response in the KB.**

---

## Part 5 — Fixing Response Quality

### 5.1 Eliminating "I'll check shortly" templates

**Target:** Reduce from 27.4% to <10% of responses containing "shortly" without specific content.

**Approach:** Post-process the teacher's traces before building ChatML.

```python
def is_template_response(response: str) -> bool:
    """Detect generic 'I'll check' responses without substance."""
    indicators = [
        "i'll be in touch shortly",
        "i will update you shortly",
        "checking with our",
        "i'm checking",
        "will get back to you",
    ]
    has_template = any(ind in response.lower() for ind in indicators)
    
    # Check for substance: policy citations, specific next steps, follow-up questions
    has_substance = any(sub in response.lower() for sub in [
        "policy", "within", "business days", "per our",
        "could you please", "can you send", "would you mind",
        "photo", "image", "file", "vector",
        "tracking", "order number", "status",
    ])
    
    return has_template and not has_substance

# Usage in build_chatml_v2.py:
# If is_template_response(response), re-run the teacher with a stronger prompt
# that demands specifics
```

**Re-generation prompt for template responses:**

```python
SPECIFIC_RESPONSE_PROMPT = """
The previous response was too generic ("I'll check shortly"). Rewrite it to include:

1. At least ONE specific fact from the retrieved KB context (policy, timeline, process)
2. At least ONE specific next step or follow-up question for the customer
3. Keep the professional Loopper tone

Context from KB:
{rag_results}

Category: {category}
Customer message: {ticket_text}

Write the improved response:
"""
```

### 5.2 Salutations, personalization, and warmth in tone

**Observed in production:** Qiki LM responses are more transactional than the teacher's. They open with bare "Hello," and close with templated phrases. The teacher's responses use customer names, acknowledge the specific situation, and close with warmth that matches the language/culture of the ticket.

**Side-by-side from eval (ticket 5517 — French refund):**

```
Teacher (warm, personalized):
  "Hello Camille,
   Thank you for reaching out about your order ORD-7745.
   We have forwarded your refund request to our finance team for
   review and will share the procedure and expected processing time
   with you shortly.
   Please let us know if you have any further questions in the meantime.
   Looking forward to resolving this for you soon."

SLM (transactional, generic):
  "Hello,
   Thank you for your refund request for order ORD-7745.
   I will coordinate with our finance team to process your refund.
   I will follow up with you shortly once the refund is processed.
   Thank you for your understanding."
```

What the SLM is missing:
1. **Customer name** ("Hello Camille") — the customer signed the email "Camille", but the SLM doesn't extract it
2. **Specific situation acknowledgment** ("about your order ORD-7745") — too brief
3. **Forward-looking warmth** ("Looking forward to resolving this for you soon") — replaced with bland "Thank you for your understanding"

**Root cause:** Training responses contain warmth and personalization (the teacher learned this from Loopper's brand voice docs in the KB). But:
1. The reasoning chain doesn't explicitly track customer name extraction
2. Response length is short on average (median 244 chars), which encourages compression of warmth out
3. No validation that responses include personalization or warm closings

**Fix for v2 — three concrete changes:**

#### (a) Extract and use customer name in the reasoning chain

In `build_chatml_v2.py`, when building the reasoning_content, explicitly include name extraction:

```python
def build_response_reasoning(ticket_text: str, intent: str, rag_context: str) -> str:
    """Build reasoning that explicitly notes customer name extraction."""
    customer_name = extract_customer_name(ticket_text)  # heuristic from signature/header
    
    reasoning_parts = []
    
    if customer_name:
        reasoning_parts.append(
            f"Customer name detected: {customer_name}. Will personalize greeting."
        )
    
    reasoning_parts.append(f"Retrieved context: {summarize_rag(rag_context)}")
    reasoning_parts.append(f"Tone guidance from communication_guidelines: {extract_tone(rag_context)}")
    reasoning_parts.append("Closing: warm, language-appropriate, forward-looking.")
    
    return "\n".join(reasoning_parts)
```

This teaches the model: when there's a customer name in the ticket, surface it in the reasoning, then USE it in the response.

#### (b) Per-language salutation/closing validation

Each language has appropriate openings and closings. Validate that the response uses them:

```python
LANGUAGE_TONE_REQUIREMENTS = {
    "english": {
        "must_have_greeting": ["Hello", "Hi", "Dear"],
        "should_have_closing": ["Looking forward", "Best regards", "Thank you",
                                 "Have a great", "Wishing you", "Talk soon"],
    },
    "french": {
        "must_have_greeting": ["Bonjour", "Cher", "Chère"],
        "should_have_closing": ["Cordialement", "Avec plaisir", "Dans l'attente",
                                 "Merci de votre", "Au plaisir", "Belle journée",
                                 "Bonne journée"],
    },
    "german": {
        "must_have_greeting": ["Guten Tag", "Hallo", "Sehr geehrte", "Liebe"],
        "should_have_closing": ["Freundliche Grüße", "Mit freundlichen Grüßen",
                                 "Liebe Grüße", "Beste Grüße", "Schönen Tag"],
    },
    "spanish": {
        "must_have_greeting": ["Hola", "Estimado", "Estimada", "Buenos días"],
        "should_have_closing": ["Saludos cordiales", "Cordialmente", "Atentamente",
                                 "Quedo a la espera", "Un saludo"],
    },
    "italian": {
        "must_have_greeting": ["Buongiorno", "Salve", "Gentile"],
        "should_have_closing": ["Cordiali saluti", "Distinti saluti",
                                 "In attesa", "La ringrazio", "Buon proseguimento"],
    },
    "dutch": {
        "must_have_greeting": ["Hallo", "Beste", "Goedendag"],
        "should_have_closing": ["Met vriendelijke groet", "Vriendelijke groet",
                                 "Hartelijk dank", "Fijne dag"],
    },
}

def validate_warmth(response: str, language: str) -> list[str]:
    """Check for language-appropriate greeting and warm closing."""
    warnings = []
    req = LANGUAGE_TONE_REQUIREMENTS.get(language)
    if not req:
        return warnings  # unsupported language, skip
    
    has_greeting = any(g in response for g in req["must_have_greeting"])
    if not has_greeting:
        warnings.append(f"Response missing language-appropriate greeting in {language}")
    
    has_closing = any(c in response for c in req["should_have_closing"])
    if not has_closing:
        warnings.append(f"Response missing warm closing in {language}")
    
    return warnings
```

Examples that fail this validation get flagged for re-generation.

#### (c) Explicitly require personalization in the response prompt

When re-generating template responses (Part 5.1), include personalization in the prompt:

```python
WARM_RESPONSE_PROMPT = """
Rewrite this response with warmth, personalization, and language-appropriate tone.

Required elements:
1. **Greeting**: Use the customer's name if it appears in the ticket signature
   (e.g., "Hello Camille" not "Hello,"). If no name, use a warm but generic
   greeting in the customer's language.

2. **Acknowledge the specific situation**: Reference the order number, the
   product, or the specific concern they raised — not just generic
   "thank you for your message."

3. **Show empathy when appropriate**: For complaints, refunds, delays, or
   quality issues, include a brief empathy phrase:
   - English: "We're sorry to hear", "I understand the urgency"
   - French: "Nous sommes navrés", "Je comprends votre préoccupation"
   - German: "Es tut uns leid", "Wir verstehen Ihre Sorge"
   - Spanish: "Lamentamos", "Entendemos su preocupación"
   - Italian: "Ci dispiace", "Comprendiamo la sua preoccupazione"

4. **Forward-looking close**: End with something warm and forward-looking,
   not just "I'll be in touch shortly":
   - English: "Looking forward to resolving this for you soon"
   - French: "Au plaisir de vous tenir informé"
   - German: "Wir freuen uns, Ihnen helfen zu können"
   - Spanish: "Quedo a su disposición"
   - Italian: "Resto a sua disposizione"

5. **Match Loopper's brand voice**: Warm, professional, never dismissive.

Customer ticket: {ticket_text}
Customer name detected: {customer_name}
Detected language: {language}
Category: {category}
KB context: {rag_results}
Sign-off: Marc Logier, Account Manager — Loopper

Original response (too transactional):
{original_response}

Rewrite the response. Output only the response text, no JSON wrapping.
"""
```

#### Customer name extraction (lightweight heuristic)

```python
import re

NAME_PATTERNS = [
    # Pattern 1: signature line "Best regards, NAME"
    r"(?:best regards|cordialement|mit freundlichen grüßen|saludos|cordiali saluti),?\s*([A-Z][a-zà-ÿ]+(?:\s+[A-Z][a-zà-ÿ]+)?)",
    # Pattern 2: signature line "Thanks, NAME"
    r"(?:thanks|merci|danke|gracias|grazie),?\s*([A-Z][a-zà-ÿ]+(?:\s+[A-Z][a-zà-ÿ]+)?)\s*$",
    # Pattern 3: signed at the end "NAME\nemail" or "NAME\n+phone"
    r"\n([A-Z][a-zà-ÿ]+(?:\s+[A-Z][a-zà-ÿ]+)?)\n[\w.]+@",
    # Pattern 4: PII placeholder
    r"<PII_NAME[:_][a-z0-9]+>",
]

def extract_customer_name(ticket_text: str) -> str | None:
    """Best-effort extraction of customer name from ticket text."""
    for pattern in NAME_PATTERNS:
        match = re.search(pattern, ticket_text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1) if match.lastindex else "[NAME]"
    return None
```

For tickets where the name is a PII placeholder (`<PII_NAME:abc123>`), the response should use `[NAME]` as the placeholder so the agent's PII restoration step puts the real name back at delivery time.

### 5.3 Per-category response requirements

Each category has specific information the response SHOULD contain. Validate in `build_chatml_v2.py`:

```python
CATEGORY_RESPONSE_REQUIREMENTS = {
    "quality_complaint": {
        "must_contain_one_of": ["photo", "image", "picture", "foto", "Foto", "imagen"],
        "reason": "Quality policy requires photographic evidence before processing",
    },
    "design_update": {
        "must_contain_one_of": ["vector", ".AI", ".EPS", ".PDF", "file", "fichier", "Datei"],
        "reason": "Design team needs vector source files",
    },
    "refund_request": {
        "must_contain_one_of": ["day", "business", "process", "review", "eligible",
                                "jour", "Tag", "día", "giorno"],
        "reason": "Customer needs to know the refund timeline and eligibility criteria",
    },
    "delivery_issue": {
        "must_contain_one_of": ["tracking", "courier", "logistics", "status",
                                "suivi", "Sendungsverfolgung", "seguimiento"],
        "reason": "Customer needs tracking info or logistics team follow-up",
    },
    "new_order_inquiry": {
        "must_contain_one_of": ["product", "quantity", "SKU", "produit", "Produkt",
                                "producto", "prodotto"],
        "reason": "Need specific product/quantity details to prepare a quote",
    },
    "sample_request": {
        "must_contain_one_of": ["sample", "cost", "transport", "blank", "printed",
                                "échantillon", "Muster", "muestra"],
        "reason": "Customer needs sample pricing and availability info",
    },
}

def validate_response_content(category: str, response: str) -> list[str]:
    """Check that the response includes category-specific content."""
    warnings = []
    req = CATEGORY_RESPONSE_REQUIREMENTS.get(category)
    if req and not any(kw.lower() in response.lower() for kw in req["must_contain_one_of"]):
        warnings.append(
            f"Response for {category} missing required content: {req['reason']}"
        )
    return warnings
```

Responses that fail validation get flagged for re-generation with the specific response prompt.

---

## Part 6 — Collection Name Normalization

### The problem

Training data uses teacher agent's internal names:

```
customer_policy_faq        2,438 calls  (should be: faq)
sales_operations_playbook  1,357 calls  (should be: operations)
communication_guidelines   3,352 calls  (matches)
supplier_intelligence         20 calls  (should be: supplier_data)
faq                          785 calls  (matches but inconsistent with above)
```

### The fix for v2

**Option chosen: normalize to short names in training data.**

In `build_chatml_v2.py`:

```python
COLLECTION_NAME_MAP = {
    "customer_policy_faq": "faq",
    "sales_operations_playbook": "operations",
    "supplier_intelligence": "supplier_data",
    # Keep these as-is:
    "faq": "faq",
    "operations": "operations",
    "communication_guidelines": "communication_guidelines",
    "supplier_data": "supplier_data",
}

def normalize_tool_call(tool_call: dict) -> dict:
    """Normalize collection names in tool_call arguments."""
    args = tool_call.get("function", {}).get("arguments", "{}")
    if isinstance(args, str):
        args = json.loads(args)
    
    collection = args.get("collection", "")
    if collection in COLLECTION_NAME_MAP:
        args["collection"] = COLLECTION_NAME_MAP[collection]
    
    tool_call["function"]["arguments"] = json.dumps(args)
    return tool_call
```

Also update the tool schema enum in `configs/loopper_pipeline.yaml` and the agent's `KIKI_TOOLS` to use the short names. Ensure the RAG MCP server accepts the short names too.

---

## Part 7 — Optional DPO Stage

### What DPO adds

Supervised fine-tuning teaches "do what the teacher did." DPO (Direct Preference Optimization) teaches "this behavior is BETTER than that behavior." This is powerful for fixing the tool-skipping issue because we can explicitly penalize ungrounded responses.

### Creating preference pairs

Generate 150-200 preference pairs from real tickets:

```python
# For each ticket, generate TWO responses:
# 
# PREFERRED (chosen): Full pipeline — <think> → rag_search → grounded JSON
# REJECTED:           Skipped pipeline — <think> → ungrounded JSON
#
# The DPO loss trains the model to prefer the grounded path.

preference_pair = {
    "prompt": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ticket_text},
    ],
    "chosen": [
        {"role": "assistant", "reasoning_content": "...", "tool_calls": [rag_search(...)]},
        {"role": "tool", "content": rag_results},
        {"role": "assistant", "reasoning_content": "Retrieved context: ...", 
         "content": '{"intent":"refund_request", ..., "policy_used":"Refund policy from KB"}'},
    ],
    "rejected": [
        {"role": "assistant", "reasoning_content": "I need to search but I know the answer",
         "content": '{"intent":"refund_request", ..., "policy_used":"No policy consulted"}'},
    ],
}
```

### DPO training setup

```python
# After SFT, load the SFT checkpoint and run DPO
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    beta=0.1,                    # KL penalty strength
    learning_rate=5e-5,          # lower than SFT
    num_train_epochs=1,          # single pass
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_length=4096,
    max_prompt_length=2048,
)

trainer = DPOTrainer(
    model=sft_model,
    ref_model=None,              # use implicit reference (beta=0.1)
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
    args=dpo_config,
)
trainer.train()
```

### When to skip DPO

Skip DPO for v2 if:
- SFT alone achieves >90% tool-calling rate on valid tickets (data fix was enough)
- You have <50 preference pairs (not enough signal)
- Training compute is constrained (DPO adds another training stage)

**Recommendation:** Prepare the DPO data during v2 data collection (it's low effort — just save both grounded and ungrounded teacher outputs per ticket). Decide whether to use it after evaluating SFT-only results.

---

## Part 8 — Validation Pipeline

### Pre-training checks (run on the final dataset before training)

```python
# In scripts/validate_dataset_v2.py

def validate_v2_dataset(dataset_path: str) -> dict:
    """Comprehensive pre-training validation."""
    checks = {
        "total_examples": 0,
        "json_valid": 0,
        "all_11_fields": 0,
        "tool_call_invariant": 0,  # valid tickets with ≥1 tool call
        "tool_call_json_clean": 0, # no trailing commas in tool_call JSON
        "collection_names_valid": 0,
        "response_not_template": 0,
        "category_specific_content": 0,
        "warmth_greeting_present": 0,    # language-appropriate greeting
        "warmth_closing_present": 0,     # language-appropriate warm closing
        "personalization_when_name_available": 0,  # uses customer name when present
        "rejection_ratio": 0,
    }
    
    category_counts = Counter()
    urgency_counts = Counter()
    team_counts = Counter()
    
    # ... run all checks, collect stats ...
    
    # Final gates
    assert checks["json_valid"] / checks["total_examples"] > 0.99
    assert checks["tool_call_invariant"] / checks["total_examples"] > 0.99
    assert checks["tool_call_json_clean"] / checks["total_examples"] > 0.99
    
    # Distribution gates
    for cat, count in category_counts.items():
        if cat != "other":
            assert count >= 200, f"{cat} has only {count} examples (need ≥200)"
    assert urgency_counts.get("critical", 0) >= 30
    assert team_counts.get("finance", 0) >= 150
    assert checks["rejection_ratio"] < 0.12  # under 12%
    
    return checks
```

### Post-training evaluation (run on gold_100 + new gold examples)

```
Metric                       v1 Result    v2 Target    Gate
──────────────────────────────────────────────────────────────────
intent_accuracy                51%          75%         ≥70%
urgency_accuracy               59%          75%         ≥65%
is_valid_accuracy              88%          95%         ≥90%
tool_call_rate (valid)         ~70%*        95%         ≥90%
json_strict_parse              ~60%*        95%         ≥90%
response_has_substance         ~40%*        80%         ≥70%
language_appropriate_greeting   ~85%*        95%         ≥90%
language_appropriate_closing    ~50%*        90%         ≥80%
uses_customer_name_when_present ~10%*        70%         ≥60%
empathy_on_complaint_tickets    ~30%*        80%         ≥70%
avg_turns                      3.05         2.5         1.5-3.5

* estimated from production testing, not formal eval
```

If any gate fails, investigate before deploying. Do NOT ship a model that regresses on any metric vs v1.

---

## Part 9 — v2 Execution Checklist

### Phase 1 — Data Collection (1-2 weeks)

- [ ] Filter Freshdesk tickets for rare categories (refund, feedback, price, sample)
- [ ] Generate synthetic tickets for categories where Freshdesk is insufficient
- [ ] Re-run teacher agent on expanded ticket pool (target: 8,000 traces)
- [ ] Reclassify "other" tickets with improved triage prompt
- [ ] Create 50+ critical urgency examples
- [ ] Create 50-100 "simple but valid" examples with tool calls
- [ ] Generate DPO preference pairs (150-200, for optional DPO stage)

### Phase 2 — Data Processing (2-3 days)

- [ ] Run `build_chatml_v2.py` with all validations enabled
- [ ] Normalize collection names
- [ ] Make rejection reasoning structurally distinct (short + declarative)
- [ ] Flag template responses ("I'll check shortly") for re-generation
- [ ] Re-generate flagged responses with specific-response prompt
- [ ] Extract customer names from ticket signatures and surface in reasoning_content
- [ ] Validate per-language greetings and warm closings (English/French/German/Spanish/Italian/Dutch)
- [ ] Validate empathy markers on complaint/refund/quality tickets
- [ ] Re-generate responses that fail warmth/personalization validation
- [ ] Validate per-category response requirements
- [ ] Run `trim_chatml.py` (same as v1)
- [ ] Run `validate_dataset_v2.py` — all gates must pass

### Phase 3 — Training (1 day)

- [ ] Upload train_v2.jsonl + eval_v2.jsonl to Drive
- [ ] Update `configs/colab_config.yaml` with v2 hyperparameters (rank=64, alpha=128, lr=1.5e-4)
- [ ] Train on Colab Pro (L4 minimum, A100 preferred)
- [ ] Monitor: loss should drop below v1's final loss
- [ ] Run eval on gold_100 — check all gates

### Phase 4 — Optional DPO (1 day)

- [ ] Only if SFT-only tool-call rate < 90% on valid tickets
- [ ] Load SFT checkpoint, run DPO with 150-200 preference pairs
- [ ] Re-evaluate on gold_100 — tool-call rate should improve
- [ ] Verify no regression on other metrics

### Phase 5 — Deploy & Validate (2-3 days)

- [ ] Export GGUF (for local Ollama dev testing)
- [ ] Push merged fp16 to HuggingFace Hub
- [ ] Deploy on Modal / vLLM
- [ ] Run smoke test against live endpoint with 20 gold tickets
- [ ] Run side-by-side comparison against v1 on 50 production tickets
- [ ] Shadow mode in dev for 2-3 days
- [ ] Promote to production

---

## Part 10 — What NOT to Change in v2

1. **Base model**: Keep `unsloth/Qwen3-4B-Thinking-2507`. Switching models resets all template/inference learnings.
2. **Output schema**: Keep the 11-field JSON. Adding/removing fields requires agent code changes.
3. **System prompt structure**: Keep the same overall shape. Small wording changes are OK (see Part 4.2 for rejection reasoning changes), but don't rewrite the whole prompt — the model was trained on it.
4. **Chat template**: Keep Unsloth's Qwen3 template. The inference pipeline (vLLM reasoning parser + hermes tool parser) depends on it.
5. **Tool schema shape**: Keep `rag_search` with `collection`, `query`, `top_k`. Change collection names (normalize) but not the parameter structure.
6. **Training framework**: Keep Unsloth + SFTTrainer. It works, it's fast, no reason to switch.

---

## Appendix A — v1 vs v2 Comparison Table

```
                        v1 (current)              v2 (target)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total examples:         4,099                      8,000-10,000
Min per category:       30 (refund_request)        300
Max per category:       1,030 (design_update)      500
Rejection ratio:        18%                        6-8%
"other" ratio:          21.4%                      <5%
Critical urgency:       0                          50+
Finance team:           79                         200+

LoRA rank:              32                         64
Alpha:                  64                         128
Learning rate:          2e-4                       1.5e-4
Dropout:                0                          0.05

Response quality:       27% "shortly" templates    <10%
Tone:                   Transactional, generic     Warm, personalized, language-appropriate
Customer name usage:    Rare (~10%)                Frequent (~70% when name available)
Per-language closings:  Mixed (English leaks)      Validated per-language
Empathy on complaints:  Inconsistent               Required by validation
Tool-call enforcement:  Data-level only            Data + structural separation + DPO
Collection names:       Mixed (verbose + short)    Normalized (short only)
Rejection format:       Detailed (same as valid)   Short + distinctive

Expected intent acc:    51%                        75%+
Expected tool-call %:   ~70%                       95%+
Expected parse rate:    100% (lenient)             95%+ (strict, all 11 fields)
```
