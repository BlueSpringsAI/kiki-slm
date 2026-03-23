# Building a Training Dataset From Organic Company Tickets

> Complete guide for creating an SFT dataset from real company tickets that teaches intent classification, urgency assessment, workflow planning, RAG tool calling, tool response handling, and resolution generation.

---

## Table of Contents

1. [Can You Skip Public Datasets?](#1-can-you-skip-public-datasets)
2. [Universal CS Intent Taxonomy](#2-universal-cs-intent-taxonomy)
3. [The Complete Training Example Structure](#3-the-complete-training-example-structure)
4. [RAG Tool Calling Patterns](#4-rag-tool-calling-patterns)
5. [Step-by-Step Dataset Creation Process](#5-step-by-step-dataset-creation-process)
6. [Data Augmentation Strategy](#6-data-augmentation-strategy)
7. [Example Training Conversations](#7-example-training-conversations)

---

## 1. Can You Skip Public Datasets?

**Yes, if you have enough volume AND diversity.**

| Approach | Pros | Cons |
|:---------|:-----|:-----|
| **Public only** (current) | Broad intent coverage, diverse language | Generic workflows, doesn't know your business |
| **Organic only** | YOUR workflows, YOUR language, YOUR tools | May miss rare intents if your data is skewed |
| **Hybrid** (recommended) | Best of both | Slightly more complex pipeline |
| **Organic + Synthetic** (best for you) | YOUR data + GPT-4o fills gaps | No dependency on public datasets |

**With 400K tickets, you can absolutely go organic-primary.** The recommended split:

```
Organic tickets (annotated):     80%    — YOUR real data
Synthetic (GPT-4o generated):    15%    — fills intent/urgency gaps
Public datasets:                  5%    — only if specific gaps remain
```

**When to keep public datasets:**
- If your 400K tickets are >90% one language (e.g., German) and you need English too
- If certain intents are completely missing from your organic data (e.g., you have 0 fraud tickets)
- For tool calling patterns — xlam/hermes datasets teach function calling format

**When to drop them entirely:**
- If your organic data covers all 13+ intents with 500+ examples each
- If you have multi-turn conversations with tool calls in your organic data
- If your company's response style is very different from public dataset responses

---

## 2. Universal CS Intent Taxonomy

### How many intents exist in CS?

There is no single standard, but analyzing across Zendesk, Freshdesk, Intercom, Salesforce, and academic datasets, customer service intents organize into **3 tiers**:

### Tier 1: Core Intents (15-20 universal categories)

These exist in EVERY customer service operation regardless of industry:

```
INFORMATION SEEKING
  1.  order_status          "Where is my order?"
  2.  product_inquiry       "What features does X have?"
  3.  pricing_inquiry       "How much does X cost?"
  4.  policy_inquiry        "What's your return policy?"
  5.  general_inquiry       "What are your hours?"

FINANCIAL
  6.  billing_inquiry       "Why was I charged $X?"
  7.  refund_request        "I want my money back"
  8.  payment_issue         "My payment failed"
  9.  subscription_management "Change/cancel my plan"

PROBLEM RESOLUTION
  10. technical_support     "X isn't working"
  11. complaint             "Your service is terrible"
  12. shipping_issue        "Package is late/damaged"
  13. return_request        "I want to return X"

ACCOUNT
  14. account_management    "Update my info/password"
  15. cancellation          "Cancel my account/order"
  16. fraud_report          "Unauthorized activity"

OPERATIONAL
  17. feedback              "Suggestion/praise"
  18. escalation_request    "Talk to a manager"
  19. follow_up             "Following up on ticket #X"
  20. system_event          Automated notifications (not human)
```

### Tier 2: Sub-Intents (~100-150 per industry)

Each Tier 1 intent breaks down into specific sub-intents. Examples:

```
order_status (Tier 1)
  ├── order_tracking            "Where is my package right now?"
  ├── order_confirmation        "Did my order go through?"
  ├── order_modification        "Can I change my order?"
  ├── order_eta                 "When will it arrive?"
  ├── order_missing             "Order shows delivered but I don't have it"
  └── order_wrong_item          "I received the wrong product"

technical_support (Tier 1)
  ├── login_issue               "Can't log in"
  ├── app_crash                 "App keeps crashing"
  ├── feature_not_working       "X feature doesn't work"
  ├── integration_issue         "API/integration broken"
  ├── performance_issue         "Site is slow"
  ├── configuration_help        "How do I set up X?"
  └── data_issue                "My data is wrong/missing"

billing_inquiry (Tier 1)
  ├── unexpected_charge         "I see a charge I don't recognize"
  ├── duplicate_charge          "Charged twice"
  ├── invoice_request           "Send me an invoice"
  ├── payment_method_update     "Change my credit card"
  ├── promo_code_issue          "My discount code didn't work"
  └── tax_question              "Why was I charged tax?"
```

### Tier 3: Company-Specific Intents (~500+)

These are unique to YOUR business:

```
Your logistics company might have:
  ├── shipment_pickup_scheduled
  ├── shipment_customs_hold
  ├── shipment_address_correction
  ├── carrier_complaint
  ├── warehouse_stock_inquiry
  ├── bulk_shipping_quote
  ├── return_label_request
  ├── packaging_damage_claim
  └── ... (dozens more specific to your products/services)
```

### Recommendation: Start with Tier 1, discover Tier 2 from your data

1. Use the **20 Tier 1 intents** as your initial taxonomy
2. Run the analysis script on 400K tickets
3. Let GPT-4o annotate a sample of 5K tickets
4. Look at the annotation distribution — Tier 2 sub-intents will emerge naturally
5. Decide which sub-intents are frequent enough to become their own training category

---

## 3. The Complete Training Example Structure

### What the model should learn from each ticket

A single organic ticket should teach the model the **COMPLETE resolution chain**:

```
Customer sends message
    ↓
1. CLASSIFY: What is the intent? How urgent?
    ↓
2. PLAN: What workflow steps are needed?
    ↓
3. SEARCH: What information do we need? (RAG tool call)
    ↓
4. RECEIVE: Process the search results
    ↓
5. DECIDE: Based on results, what action to take?
    ↓
6. RESPOND: Generate customer-facing response
    ↓
7. RESOLVE: Close the ticket or continue the conversation
```

### The training example format

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are Kiki, an AI customer service agent for [Company]. You help customers by classifying their intent, planning resolution workflows, searching the knowledge base when needed, and generating professional responses.\n\nAvailable tools:\n- knowledge_base_search(query): Search company knowledge base and policies\n- order_lookup(order_id): Look up order details\n- customer_profile(customer_id): Get customer account info\n- create_ticket_action(action, details): Execute a resolution action\n\nAlways respond in structured JSON."
    },
    {
      "role": "user",
      "content": "ich wollt gerade die Bestellung zu Angebot 5355779 auslösen. Allerdings sind die Zahlungsmodi nicht wie vereinbart."
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "type": "function",
          "function": {
            "name": "knowledge_base_search",
            "arguments": "{\"query\": \"payment method configuration order quote\"}"
          }
        },
        {
          "type": "function",
          "function": {
            "name": "order_lookup",
            "arguments": "{\"order_id\": \"5355779\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "name": "knowledge_base_search",
      "content": "{\"results\": [{\"title\": \"Payment Method Configuration\", \"content\": \"To change payment method on a quote: 1. Open quote in system 2. Navigate to payment settings 3. Select new method 4. Resubmit order. Supported methods: Invoice, Credit Card, Direct Debit.\"}]}"
    },
    {
      "role": "tool",
      "name": "order_lookup",
      "content": "{\"quote_id\": \"5355779\", \"status\": \"pending\", \"payment_method\": \"credit_card\", \"requested_method\": \"invoice\", \"amount\": 1250.00, \"customer\": \"[ORG]\"}"
    },
    {
      "role": "assistant",
      "content": "{\"intent\": \"payment_issue\", \"sub_intent\": \"payment_method_change\", \"urgency\": \"high\", \"workflow_steps\": [\"lookup_quote_5355779\", \"verify_payment_method_mismatch\", \"update_payment_to_invoice\", \"resubmit_order\", \"send_confirmation\"], \"tools_used\": [\"knowledge_base_search\", \"order_lookup\"], \"resolution_action\": {\"action\": \"update_payment_method\", \"from\": \"credit_card\", \"to\": \"invoice\", \"quote_id\": \"5355779\"}, \"reasoning\": \"Customer wants to order from quote 5355779 but payment method shows credit card instead of agreed invoice. Knowledge base confirms we can change payment method and resubmit.\", \"response\": \"Natürlich. Ich habe die Zahlungsmethode für Angebot 5355779 auf Rechnung umgestellt und die Bestellung aufgegeben. Sie erhalten in Kürze eine Bestätigung per E-Mail.\"}"
    }
  ]
}
```

### What each part teaches the model

| Message | What it teaches |
|:--------|:----------------|
| `system` | Available tools, company identity, output format |
| `user` (customer message) | Real customer language, how problems are described |
| `assistant` with `tool_calls` | WHEN to search, WHAT query to use, WHICH tools to call |
| `tool` (search results) | What real knowledge base results look like |
| `tool` (order lookup) | What real API responses look like |
| `assistant` (final JSON) | Complete classification + reasoning + response using tool results |

---

## 4. RAG Tool Calling Patterns

### Pattern 1: Knowledge Base Search (most common)

The model decides it needs information, generates a search query, receives results, and uses them to respond.

```
CUSTOMER: "What's your return policy for electronics?"

MODEL THINKS: I need to look up the return policy → search knowledge base

TOOL CALL: knowledge_base_search(query="return policy electronics")

TOOL RESULT: {"results": [{"title": "Return Policy", "content": "Electronics
can be returned within 30 days of purchase in original packaging..."}]}

MODEL RESPONDS: {
  "intent": "policy_inquiry",
  "tools_used": ["knowledge_base_search"],
  "response": "Our return policy for electronics allows returns within 30 days
               of purchase in original packaging..."
}
```

**Training signal:** The model learns to generate good search queries from customer questions.

### Pattern 2: Order/Account Lookup + Knowledge Base

Multiple tool calls in sequence — look up customer data THEN search for relevant policy.

```
CUSTOMER: "I ordered a laptop 3 weeks ago and it hasn't arrived"

TOOL CALLS:
  1. order_lookup(order_id="extracted from context or customer profile")
  2. knowledge_base_search(query="shipping delay policy compensation")

TOOL RESULTS:
  1. {"status": "in_transit", "shipped": "2025-12-01", "carrier": "DHL", "eta": "overdue"}
  2. {"results": [{"content": "If shipment is >14 days overdue, offer reshipping or refund..."}]}

MODEL RESPONDS: {
  "intent": "shipping_issue",
  "sub_intent": "shipment_delayed",
  "urgency": "high",
  "workflow_steps": ["lookup_order", "check_carrier_status", "apply_delay_policy", "offer_resolution"],
  "resolution_action": {"action": "offer_choice", "options": ["reship", "refund"]},
  "response": "Ihr Paket ist seit über 2 Wochen unterwegs und überfällig.
               Ich kann Ihnen eine Neulieferung veranlassen oder eine Rückerstattung
               anbieten. Was wäre Ihnen lieber?"
}
```

### Pattern 3: No Tool Call Needed

Simple queries that don't require external data.

```
CUSTOMER: "What are your business hours?"

MODEL RESPONDS (no tool call): {
  "intent": "general_inquiry",
  "urgency": "low",
  "workflow_steps": ["provide_information"],
  "tools_used": [],
  "response": "Unsere Geschäftszeiten sind Montag bis Freitag, 8:00 bis 18:00 Uhr."
}
```

**Training signal:** The model learns WHEN NOT to call tools — simple FAQs don't need RAG.

### Pattern 4: Multi-Turn with Progressive Tool Calls

```
TURN 1:
  Customer: "I need to change my delivery address"
  Model: tool_call → customer_profile("customer_123")
  Result: {"name": "...", "address": "...", "active_orders": ["ORD-456"]}
  Model: "I see you have order ORD-456. What's the new address?"

TURN 2:
  Customer: "Hauptstraße 15, 80331 München"
  Model: tool_call → order_lookup("ORD-456")
  Result: {"status": "processing", "shipped": false}
  Model: tool_call → knowledge_base_search("change delivery address before shipment")
  Result: {"content": "Address can be changed if order hasn't shipped yet..."}
  Model: {
    "intent": "shipping_issue",
    "resolution_action": {"action": "update_address", "order": "ORD-456", "new_address": "Hauptstraße 15, 80331 München"},
    "response": "Die Lieferadresse für Bestellung ORD-456 wurde aktualisiert.
                 Das Paket wird an Hauptstraße 15, 80331 München geliefert."
  }
```

### Pattern 5: Tool Call → Decision → Different Tool Call

The model reasons about tool results and decides what to do next.

```
CUSTOMER: "I want a refund for order ORD-789"

STEP 1: tool_call → order_lookup("ORD-789")
RESULT: {"status": "delivered", "amount": 89.99, "delivered_date": "2025-12-01"}

STEP 2: tool_call → knowledge_base_search("refund policy delivered orders timeframe")
RESULT: {"content": "Refunds within 30 days of delivery. After 30 days, store credit only."}

STEP 3: MODEL REASONS: Delivered Dec 1, today is Dec 15, within 30 days → full refund eligible

STEP 4: tool_call → create_ticket_action("process_refund", {"order": "ORD-789", "amount": 89.99})
RESULT: {"refund_id": "REF-001", "status": "processing", "eta": "3-5 business days"}

FINAL RESPONSE: {
  "intent": "refund_request",
  "urgency": "medium",
  "workflow_steps": ["lookup_order", "check_refund_eligibility", "verify_timeframe", "process_refund"],
  "tools_used": ["order_lookup", "knowledge_base_search", "create_ticket_action"],
  "resolution_action": {"action": "refund_processed", "refund_id": "REF-001", "amount": 89.99},
  "response": "Ihre Rückerstattung von 89,99€ für Bestellung ORD-789 wurde eingeleitet.
               Sie erhalten den Betrag innerhalb von 3-5 Werktagen zurück."
}
```

---

## 5. Step-by-Step Dataset Creation Process

### Overview

```
STEP 1: Export & Analyze (1-2 days)
    ↓
STEP 2: Filter & Clean (1 day)
    ↓
STEP 3: Annotate with GPT-4o (1-2 days, ~$50-100)
    ↓
STEP 4: Extract Multi-Turn Conversations (1 day)
    ↓
STEP 5: Synthesize Tool Call Patterns (1-2 days, ~$20-30)
    ↓
STEP 6: Augment Missing Intents (1 day, ~$10-20)
    ↓
STEP 7: Quality Filter & Validate (1 day)
    ↓
STEP 8: Format as ChatML JSONL (automated)
    ↓
TRAINING DATASET READY
```

### Step 1: Export & Analyze

```bash
# Export all tickets from Freshdesk (your team does this)
# Place in raw_tickets/ directory

# Run analysis
python scripts/analyze_raw_tickets.py --input-dir raw_tickets/
```

**Key numbers to get from the analysis:**
- How many tickets have agent responses? (= usable for full SFT)
- How many have multi-turn conversations? (= usable for conversation training)
- What's the language distribution?
- What tags/categories exist? (= existing intent signals)
- How many are automated? (= filter out)

### Step 2: Filter & Clean

Remove tickets that won't help training:

```
REMOVE:
  - Automated notifications (shipping alerts, system emails)
  - Spam tickets
  - Unresolved/open tickets (no resolution to learn from)
  - Empty or very short descriptions (<20 chars)
  - Duplicate tickets (same description)

KEEP:
  - Status = resolved (4) or closed (5)
  - Has customer message ≥ 20 chars
  - Has at least 1 agent response
  - Not spam
```

**Expected yield:** From 400K total → approximately 50K-150K usable tickets (depends on how many are automated).

### Step 3: Annotate with GPT-4o

For each usable ticket, send to GPT-4o-mini for structured annotation:

**Annotation prompt:**

```
You are annotating a customer service ticket for training an AI model.

Given the customer message and (optionally) the agent's response, provide:

1. intent: One of [order_status, product_inquiry, pricing_inquiry, policy_inquiry,
   general_inquiry, billing_inquiry, refund_request, payment_issue,
   subscription_management, technical_support, complaint, shipping_issue,
   return_request, account_management, cancellation, fraud_report,
   feedback, escalation_request, follow_up, system_event]

2. sub_intent: A specific sub-category (e.g., "duplicate_charge" under billing_inquiry)

3. urgency: One of [critical, high, medium, low]

4. workflow_steps: The specific steps needed to resolve THIS ticket (not generic)

5. tools_needed: Which tools would be needed:
   - knowledge_base_search: When agent needs to look up policies/procedures
   - order_lookup: When ticket references an order
   - customer_profile: When need to check account details
   - create_ticket_action: When need to execute an action (refund, update, etc.)

6. search_queries: If knowledge_base_search is needed, what would you search for?

7. resolution_summary: How was this resolved (from agent response)?

Output as JSON.
```

**Cost estimate:** 50K tickets × ~500 input tokens × $0.15/1M = ~$4 for annotation.

**Use the existing script:** `scripts/1_annotate.py` with modifications for Freshdesk format, or build a new `scripts/annotate_organic.py`.

### Step 4: Extract Multi-Turn Conversations

For tickets with `conversations[]`, extract the full thread:

```python
def extract_multi_turn(ticket_data, annotation):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Initial customer message
    messages.append({
        "role": "user",
        "content": ticket_data["ticket"]["description_text"]
    })

    # First assistant response (with classification)
    messages.append({
        "role": "assistant",
        "content": json.dumps({
            "intent": annotation["intent"],
            "urgency": annotation["urgency"],
            "workflow_steps": annotation["workflow_steps"],
            "tools_required": annotation["tools_needed"],
            "response": "..."  # from first agent response
        })
    })

    # Subsequent turns
    for conv in ticket_data["conversations"]:
        if conv["incoming"]:  # Customer follow-up
            messages.append({"role": "user", "content": conv["body_text"]})
        else:  # Agent response
            messages.append({
                "role": "assistant",
                "content": json.dumps({
                    "intent": "...",  # may shift from original
                    "response": conv["body_text"]
                })
            })

    return {"messages": messages}
```

### Step 5: Synthesize Tool Call Patterns

Your organic tickets don't have explicit tool calls — agents used internal systems directly. We need to SYNTHESIZE what tool calls would look like:

**Approach: Use GPT-4o to generate tool call patterns from resolved tickets**

```
For each resolved ticket:
  INPUT:
    - Customer message
    - Agent's response
    - Resolution outcome

  GPT-4o GENERATES:
    - What tools the agent WOULD have called
    - What search queries they WOULD have used
    - What the tool results WOULD look like
    - The complete tool_call → tool_result → response chain
```

**This is the most important step.** It transforms flat "customer message → agent response" pairs into rich "customer message → tool calls → tool results → response" training examples.

**Example synthesis:**

```
ORIGINAL TICKET:
  Customer: "Die Zahlungsmethode für Angebot 5355779 stimmt nicht"
  Agent: "Ich habe die Zahlungsmethode umgestellt und die Bestellung aufgegeben"

GPT-4o SYNTHESIZES:
  Tool calls that WOULD have happened:
    1. order_lookup("5355779") → {"payment_method": "credit_card", "requested": "invoice"}
    2. knowledge_base_search("change payment method quote") → {"content": "...procedure..."}
    3. create_ticket_action("update_payment", {"quote": "5355779", "to": "invoice"}) → {"status": "done"}
```

### Step 6: Augment Missing Intents

After annotating your organic data, check coverage:

```python
# Count intents in annotated organic data
intent_counts = Counter(t["intent"] for t in annotated_tickets)

# Check against full taxonomy
all_intents = ["order_status", "product_inquiry", ..., "system_event"]
for intent in all_intents:
    count = intent_counts.get(intent, 0)
    if count < 200:
        print(f"  GAP: {intent} has only {count} examples (need 200+)")
        # Generate synthetic examples with GPT-4o
```

For each underrepresented intent, generate 200-500 synthetic examples using GPT-4o:

```
Generate a realistic customer service message where:
- Company: [Your company description]
- Intent: fraud_report
- Language: German
- Include: customer name, order reference, specific details
- Urgency: [vary across critical/high/medium/low]

Also generate the complete resolution chain:
1. What tools would be called
2. What the tool results would look like
3. The agent's response
```

### Step 7: Quality Filter & Validate

```python
# Validation checks for each training example:
def validate_example(example):
    messages = example["messages"]

    # Has system + user + assistant
    roles = [m["role"] for m in messages]
    assert "system" in roles
    assert "user" in roles
    assert "assistant" in roles

    # Assistant response is valid JSON
    for m in messages:
        if m["role"] == "assistant" and m.get("content"):
            parsed = json.loads(m["content"])
            assert "intent" in parsed
            assert "response" in parsed

    # Tool calls have matching tool results
    for i, m in enumerate(messages):
        if m.get("tool_calls"):
            assert messages[i+1]["role"] == "tool"

    # No PII in responses (already redacted in Freshdesk export)
    # No empty messages
    # Response length > 20 chars
```

### Step 8: Format as ChatML JSONL

Convert everything to the standard format and save:

```bash
# Output: data/formatted/kiki_organic_train.jsonl
# Each line is one training example in ChatML format
```

---

## 6. Data Augmentation Strategy

### Augmentation 1: Urgency Variation

For each organic ticket, create 3 more versions at different urgency levels:

```
ORIGINAL: "Können Sie den Bestellstatus prüfen?" (medium)

AUGMENTED:
  LOW:      "Wenn Sie Zeit haben, könnten Sie mal schauen wo meine Bestellung ist?"
  HIGH:     "Ich warte seit 2 Wochen auf meine Bestellung! Das ist inakzeptabel!"
  CRITICAL: "Meine Bestellung für die Messe morgen ist nicht da! Ich verliere Geschäft!"
```

### Augmentation 2: Language Paraphrasing

For each ticket, generate 2-3 paraphrased versions:

```
ORIGINAL: "Die Zahlungsmethode stimmt nicht"
PARA 1:   "Ich kann nicht mit Rechnung bezahlen, obwohl das vereinbart war"
PARA 2:   "Bei der Bestellung wird mir nur Kreditkarte angeboten, nicht Rechnung"
PARA 3:   "Zahlungsart falsch hinterlegt"
```

### Augmentation 3: Tool Call Variation

Same intent, different tool call patterns:

```
INTENT: order_status

PATTERN A (simple):
  → order_lookup(id) → response with status

PATTERN B (with KB search):
  → order_lookup(id) → status is "delayed"
  → knowledge_base_search("shipping delay compensation")
  → response with status + compensation offer

PATTERN C (with escalation):
  → order_lookup(id) → status is "lost"
  → knowledge_base_search("lost shipment procedure")
  → create_ticket_action("escalate", {"reason": "lost_shipment"})
  → response with escalation confirmation
```

### Augmentation 4: Custom Override Examples

Train the model to follow runtime system prompt overrides:

```json
{"messages": [
    {"role": "system", "content": "You are Kiki for [Client A]. TOOLS: [erp_lookup, sap_query, email_service]. POLICY: Auto-approve refunds under 100€. Escalate fraud to security@client-a.com. LANGUAGE: Respond in German."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "{\"tools_required\": [\"erp_lookup\"], ...}"}
]}

{"messages": [
    {"role": "system", "content": "You are Kiki for [Client B]. TOOLS: [shopify_api, zendesk_api, slack_notify]. POLICY: No refunds after 14 days. Free shipping on replacements. LANGUAGE: Respond in English."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "{\"tools_required\": [\"shopify_api\"], ...}"}
]}
```

The model learns: "read the system prompt → use THOSE tools → follow THOSE policies." This is how you make one model work for multiple clients.

---

## 7. Example Training Conversations

### Example 1: Simple Single-Turn (No Tool Call)

```json
{
  "messages": [
    {"role": "system", "content": "You are Kiki, an AI customer service agent. Available tools: knowledge_base_search, order_lookup, customer_profile, create_ticket_action. Respond in structured JSON."},
    {"role": "user", "content": "Wie sind Ihre Öffnungszeiten?"},
    {"role": "assistant", "content": "{\"intent\": \"general_inquiry\", \"sub_intent\": \"business_hours\", \"urgency\": \"low\", \"workflow_steps\": [\"provide_standard_information\"], \"tools_used\": [], \"reasoning\": \"Simple FAQ question about business hours. No tool call needed.\", \"response\": \"Unsere Geschäftszeiten sind Montag bis Freitag von 8:00 bis 18:00 Uhr. Am Wochenende sind wir nicht erreichbar.\"}"}
  ]
}
```

### Example 2: Single-Turn with RAG Search

```json
{
  "messages": [
    {"role": "system", "content": "You are Kiki..."},
    {"role": "user", "content": "Wie ist Ihre Rückgabepolitik für Elektronikprodukte?"},
    {"role": "assistant", "content": null, "tool_calls": [{"type": "function", "function": {"name": "knowledge_base_search", "arguments": "{\"query\": \"Rückgabepolitik Elektronik Rückgabefrist\"}"}}]},
    {"role": "tool", "name": "knowledge_base_search", "content": "{\"results\": [{\"title\": \"Rückgaberichtlinie\", \"content\": \"Elektronikprodukte können innerhalb von 30 Tagen nach Kauf in Originalverpackung zurückgegeben werden. Nach 30 Tagen nur Garantiereparatur. Rücksendung kostenfrei über DHL-Retourenschein.\"}]}"},
    {"role": "assistant", "content": "{\"intent\": \"policy_inquiry\", \"sub_intent\": \"return_policy_electronics\", \"urgency\": \"low\", \"workflow_steps\": [\"search_knowledge_base\", \"provide_policy_details\"], \"tools_used\": [\"knowledge_base_search\"], \"reasoning\": \"Customer asks about return policy for electronics. Found relevant policy in knowledge base.\", \"response\": \"Elektronikprodukte können innerhalb von 30 Tagen nach Kauf in der Originalverpackung zurückgegeben werden. Die Rücksendung ist für Sie kostenfrei über einen DHL-Retourenschein. Nach 30 Tagen besteht die Möglichkeit einer Garantiereparatur. Kann ich Ihnen einen Retourenschein erstellen?\"}"}
  ]
}
```

### Example 3: Multi-Turn with Order Lookup + RAG + Action

```json
{
  "messages": [
    {"role": "system", "content": "You are Kiki..."},

    {"role": "user", "content": "Ich möchte eine Rückerstattung für Bestellung ORD-12345. Das Produkt war beschädigt."},

    {"role": "assistant", "content": null, "tool_calls": [
      {"type": "function", "function": {"name": "order_lookup", "arguments": "{\"order_id\": \"ORD-12345\"}"}},
      {"type": "function", "function": {"name": "knowledge_base_search", "arguments": "{\"query\": \"Rückerstattung beschädigtes Produkt Verfahren\"}"}}
    ]},

    {"role": "tool", "name": "order_lookup", "content": "{\"order_id\": \"ORD-12345\", \"status\": \"delivered\", \"amount\": 149.99, \"delivered_date\": \"2025-12-10\", \"product\": \"Bluetooth Lautsprecher\", \"customer_id\": \"CUST-678\"}"},

    {"role": "tool", "name": "knowledge_base_search", "content": "{\"results\": [{\"title\": \"Schadensreklamation\", \"content\": \"Bei beschädigter Ware: 1. Fotos anfordern 2. Schadensbericht erstellen 3. Rückerstattung oder Ersatz anbieten. Automatische Genehmigung bis 200€.\"}]}"},

    {"role": "assistant", "content": "{\"intent\": \"refund_request\", \"sub_intent\": \"damaged_product_refund\", \"urgency\": \"high\", \"workflow_steps\": [\"lookup_order\", \"search_damage_policy\", \"request_damage_photos\", \"process_refund_or_replacement\"], \"tools_used\": [\"order_lookup\", \"knowledge_base_search\"], \"reasoning\": \"Order ORD-12345 delivered Dec 10, Bluetooth speaker, 149.99€. Customer reports damage. Policy allows auto-approval under 200€. Need photos first.\", \"response\": \"Es tut mir leid, dass Ihr Bluetooth Lautsprecher beschädigt angekommen ist. Könnten Sie mir bitte Fotos des Schadens senden? Sobald ich diese erhalten habe, kann ich Ihnen umgehend eine Rückerstattung von 149,99€ oder einen Ersatz anbieten.\"}"},

    {"role": "user", "content": "Hier sind die Fotos. Ich möchte bitte eine Rückerstattung."},

    {"role": "assistant", "content": null, "tool_calls": [
      {"type": "function", "function": {"name": "create_ticket_action", "arguments": "{\"action\": \"process_refund\", \"order_id\": \"ORD-12345\", \"amount\": 149.99, \"reason\": \"damaged_product\", \"evidence\": \"customer_photos_received\"}"}}
    ]},

    {"role": "tool", "name": "create_ticket_action", "content": "{\"action_id\": \"ACT-001\", \"status\": \"refund_initiated\", \"refund_amount\": 149.99, \"estimated_days\": 5}"},

    {"role": "assistant", "content": "{\"intent\": \"refund_request\", \"sub_intent\": \"damaged_product_refund\", \"urgency\": \"high\", \"workflow_steps\": [\"verify_damage_photos\", \"auto_approve_refund\", \"initiate_refund\", \"send_confirmation\"], \"tools_used\": [\"create_ticket_action\"], \"resolution_action\": {\"type\": \"refund\", \"amount\": 149.99, \"refund_id\": \"ACT-001\"}, \"reasoning\": \"Photos received confirming damage. Amount 149.99€ is under 200€ auto-approval threshold. Refund initiated.\", \"response\": \"Vielen Dank für die Fotos. Ihre Rückerstattung von 149,99€ wurde soeben eingeleitet. Der Betrag wird innerhalb von 5 Werktagen auf Ihr Konto zurückgebucht. Möchten Sie das beschädigte Produkt behalten oder zurücksenden?\"}"}
  ]
}
```

### Example 4: System Event (Automated Notification)

```json
{
  "messages": [
    {"role": "system", "content": "You are Kiki. Analyze if this ticket needs human attention or can be auto-processed."},
    {"role": "user", "content": "Automatische Benachrichtigung: Transport von 7 Packstücken für [ORG], Sendungsreferenz SA-2025-1234, wurde vom Versender übernommen."},
    {"role": "assistant", "content": "{\"intent\": \"system_event\", \"sub_intent\": \"shipping_notification\", \"urgency\": \"low\", \"workflow_steps\": [\"log_shipment_notification\", \"update_tracking_system\", \"auto_close_ticket\"], \"tools_used\": [], \"requires_human\": false, \"auto_close\": true, \"reasoning\": \"Automated shipping notification from carrier. No customer action needed. Log and auto-close.\", \"response\": null}"}
  ]
}
```

### Example 5: Custom Override (Client-Specific)

```json
{
  "messages": [
    {"role": "system", "content": "You are Kiki for TechCorp GmbH.\n\nCUSTOM TOOLS:\n- jira_lookup(ticket_id): Search Jira for related issues\n- confluence_search(query): Search internal documentation\n- slack_notify(channel, message): Notify team on Slack\n\nCUSTOM POLICY:\n- Critical bugs: notify #engineering Slack channel immediately\n- All responses in German\n- SLA: respond within 2 hours for high priority"},

    {"role": "user", "content": "Unser Dashboard zeigt seit heute Morgen keine Daten mehr an. Mehrere Kunden sind betroffen."},

    {"role": "assistant", "content": null, "tool_calls": [
      {"type": "function", "function": {"name": "jira_lookup", "arguments": "{\"query\": \"dashboard data display issue\"}"}},
      {"type": "function", "function": {"name": "confluence_search", "arguments": "{\"query\": \"dashboard troubleshooting data not loading\"}"}},
      {"type": "function", "function": {"name": "slack_notify", "arguments": "{\"channel\": \"#engineering\", \"message\": \"CRITICAL: Dashboard data outage reported by customer. Multiple customers affected.\"}"}}
    ]},

    {"role": "tool", "name": "jira_lookup", "content": "{\"issues\": [{\"key\": \"BUG-1234\", \"summary\": \"Dashboard API timeout\", \"status\": \"investigating\"}]}"},
    {"role": "tool", "name": "confluence_search", "content": "{\"results\": [{\"title\": \"Dashboard Troubleshooting\", \"content\": \"If data not loading: 1. Check API health 2. Verify data pipeline 3. Clear cache\"}]}"},
    {"role": "tool", "name": "slack_notify", "content": "{\"status\": \"sent\", \"channel\": \"#engineering\"}"},

    {"role": "assistant", "content": "{\"intent\": \"technical_support\", \"sub_intent\": \"service_outage\", \"urgency\": \"critical\", \"workflow_steps\": [\"check_known_issues\", \"search_documentation\", \"notify_engineering\", \"provide_status_update\"], \"tools_used\": [\"jira_lookup\", \"confluence_search\", \"slack_notify\"], \"reasoning\": \"Multiple customers affected by dashboard data outage. Found related Jira issue BUG-1234 under investigation. Engineering team notified via Slack.\", \"response\": \"Vielen Dank für die Meldung. Wir haben das Problem identifiziert — unser Engineering-Team untersucht bereits einen API-Timeout im Dashboard (Referenz: BUG-1234). Ich habe das Team soeben über die Auswirkungen auf mehrere Kunden informiert. Wir arbeiten mit Hochdruck an einer Lösung und melden uns innerhalb der nächsten 2 Stunden mit einem Update.\"}"}
  ]
}
```

---

## Summary: The Complete Dataset Recipe

```
DATASET = (
    Organic tickets (annotated + multi-turn extracted)     50K-100K examples
    + Synthesized tool call patterns (from organic)        30K-50K examples
    + Urgency-augmented variations                         10K-20K examples
    + Custom override examples                             2K-5K examples
    + Missing intent gap-fill (synthetic)                  2K-5K examples
    + System event / automated ticket examples             2K-5K examples
)

TOTAL: ~100K-200K high-quality, diverse training examples
ALL derived from or inspired by YOUR organic company data
```

This dataset teaches the model to be YOUR company's customer service expert — not a generic chatbot, but a specialized reasoning engine that knows your products, your policies, your tools, and your customers' language.
