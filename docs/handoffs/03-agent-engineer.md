# Agent Engineer Handoff — Enable Kiki SLM in the Loopper Support Agent

**Owner:** `Looper-Support-Agent-Server` repo owner.
**Goal:** merge the integration PR, flip the env flag on a dev ECS task, shadow-test against the OpenAI path, roll out to prod gradually.
**Input:** One message from DevOps: `http://kiki-ollama:11434` is live, model `kiki-sft-v1` responds.
**Output:** Production tickets flowing through the SLM instead of the 3-node OpenAI path.

You do NOT need to know how the model was trained or how Ollama is hosted. You review the PR, flip a flag, and watch the logs.

---

## The PR you're reviewing

**Repo:** BlueSpringsAI/Looper-Support-Agent-Server
**Branch:** `feat/kiki-slm-integration` → base `dev`
**PR:** BlueSpringsAI/Looper-Support-Agent-Server#117
**Commit:** `15a2ccd` (1 commit, 7 files changed, 698 insertions)

### What it does

Adds an alternative inference path behind a new env var `USE_KIKI_SLM`. When `true`, the graph replaces three OpenAI nodes (`triage_and_classify` + `retrieve_context` + `compose_response`) with a single `kiki_slm_inference` node that calls a self-hosted Ollama endpoint running a fine-tuned Qwen3-4B-Thinking model.

**Defaults:** `USE_KIKI_SLM=false`. If you merge and deploy without setting anything, the agent behaves identically to before.

### File-by-file review checklist

| File | What to check |
|---|---|
| `src/react_agent/config/settings.py` | 5 new env vars with sensible defaults. Does the naming fit your convention? |
| `src/react_agent/client/kiki_client.py` | New file. Reads like `invoke_structured` — one function, async, returns a dict. System prompt + tool schema are hardcoded (intentional — must match training data). |
| `src/react_agent/nodes/kiki_slm.py` | New file. Mirrors the fail-open pattern in `triage_and_classify.py:137`. Check the field mapping table in the module docstring — this is the "how does 11-field SLM output land in your State" answer. |
| `src/react_agent/graph.py` | Compile-time `if settings.USE_KIKI_SLM:` branch. Both paths listed cleanly. Shared nodes factored out at the top. |
| `src/react_agent/nodes/__init__.py` | Added `kiki_slm_inference` export. Alphabetical. |
| `src/react_agent/analytics/agent_metrics.py` | 1-line addition: `"kiki-sft-v1": (0.0, 0.0, 0.0)` so cost tracker doesn't KeyError. |
| `.env.example` | 5 new vars documented. |

### State mapping — the one thing you should scrutinize

The SLM emits an 11-field JSON. None of the State schema changes. Instead, the new node "stuffs" SLM fields into existing State fields:

| SLM JSON field | State field | Notes |
|---|---|---|
| `intent` | `category` (`TicketCategory` enum) | Invalid values coerced to `OTHER` with warning |
| `urgency` | `action_reasoning.urgency_level` | Already a `Literal["low","medium","high"]` — clean fit |
| `confidence` | `category_confidence` + `resolution_reasoning.resolution_confidence` | Duplicated, clamped to 0–1 |
| `is_valid` | `is_valid` (+ `ticket_status="spam"` if false) | Direct |
| `rejection_type` | `validation_reasoning.rejection_type` | Already a Literal, field pre-existed |
| `resolution_type` | `resolution_type` | Invalid → `NEEDS_ESCALATION` |
| `team` | `human_team_required` | Invalid → `NONE` |
| `actions` | `action_list` | Direct list |
| `summary` | `summary` | Direct list |
| `reasoning` (dict) | split across `category_reasoning.reasoning_summary`, `action_reasoning.why_these_actions`, `resolution_reasoning.why_resolution_type` | Lossy — all three reasoning fields get the SLM's freeform text |
| `response` | `response_english` | Direct |

**Things the SLM does NOT emit** that the OpenAI path does:
- `CategoryReasoning.key_indicators` → set to empty list
- `ActionReasoning.policy_basis` → hardcoded to "Grounded in RAG context retrieved by Kiki SLM"
- `ResolutionReasoning.escalation_risk` → hardcoded to `"medium"` (this is **not** used for routing, just analytics)

If any of these are load-bearing for downstream nodes or reporting, flag it. Otherwise they're fine as defaults.

---

## Step 1 — Merge the PR to dev

Standard review flow. The critical things to verify during review:

- [ ] Both graph paths compile (the PR description has the startup log lines proving this)
- [ ] No changes to `localize_response`, `analytics_metrics`, `human_review_gate`, `apply_revision`, or the State schema
- [ ] Fail-open behavior matches existing nodes — SLM errors → ticket passes through to human review, not dropped
- [ ] No new dependencies added (httpx is already transitively available)
- [ ] No secrets or URLs committed to the repo

Merge to `dev`. Don't merge to `main` yet.

---

## Step 2 — Verify DevOps handoff is real

Before touching any task definition, confirm the Ollama service actually works. SSH or ECS Exec into any running task in the same Service Connect namespace and run:

```bash
# From inside a task in the loopper.local namespace
curl -sf http://kiki-ollama:11434/api/tags
# Expected: {"models":[{"name":"kiki-sft-v1:latest",...,"size":2600000000,...}]}

curl -s http://kiki-ollama:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kiki-sft-v1",
    "messages": [{"role":"user","content":"Test ticket: my order is delayed"}],
    "stream": false,
    "tools": [{
      "type":"function",
      "function":{
        "name":"rag_search",
        "parameters":{"type":"object","properties":{"collection":{"type":"string"},"query":{"type":"string"}},"required":["collection","query"]}
      }
    }]
  }' | jq '.message.tool_calls, .message.content'
```

**What you want to see:**
- `tool_calls` is a non-empty array (the model is calling the tool)
- `content` may have `<think>...</think>` text or be empty — both are OK

**Red flag:**
- `tool_calls` is `null` or missing AND `content` contains literal `<tool_call>...</tool_call>` text — this means the chat template didn't survive GGUF conversion and the ML engineer needs to fix it before you deploy. **Do not proceed to Step 3 if you see this.**

---

## Step 3 — Deploy to dev ECS with the flag ON

1. Open your dev Loopper agent task definition.
2. Add these environment variables:
   ```
   USE_KIKI_SLM=true
   KIKI_SLM_URL=http://kiki-ollama:11434
   KIKI_SLM_MODEL=kiki-sft-v1
   KIKI_SLM_MAX_TURNS=4
   KIKI_SLM_TIMEOUT_S=300
   ```
3. Register a new task definition revision.
4. Update the dev ECS service to the new revision. Force new deployment.
5. Watch CloudWatch logs for the startup line:
   ```
   ── [GRAPH] USE_KIKI_SLM=true → using Kiki SLM single-node inference path
   ```
   If you see `USE_KIKI_SLM=false` instead, the env var didn't propagate — check the task definition.

---

## Step 4 — Send test tickets through dev

Pick 5–10 real tickets from your usual dev test set and send them through the agent end-to-end.

**What to watch in the logs:**

```
── [KIKI] starting SLM inference: ticket=TICKET-123 rag_tools=2 text_chars=1847
── [kiki_slm] invoking SLM model=kiki-sft-v1 url=http://kiki-ollama:11434/api/chat max_turns=4
── [kiki_slm] turn 1/4: tool_calls=1 content_len=342
── [kiki_slm] turn 2/4: tool_calls=1 content_len=298
── [kiki_slm] turn 3/4: tool_calls=0 content_len=1523
── [KIKI] SLM returned: turns=3 tool_calls=['rag_search', 'rag_search'] intent=delivery_issue resolution=requires_human_action
── [KIKI] resolution=requires_human_action, can_resolve=False, team=logistics
```

Gate criteria to proceed to Step 5:

| Metric | Target | Red flag |
|---|---:|---:|
| JSON parse failures | 0/10 | ≥2/10 |
| Avg turns per ticket | 1.5–3.5 | >3.5 (slow) or =1 (no tool use) |
| Avg SLM latency (p50) | <15s on CPU | >30s |
| `KikiInferenceError` in logs | 0 | any |
| `validation_reasoning.rejection_type` correctly set on rejections | yes | no |

If anything fails, **do not** roll out to prod. Open an issue, tag the ML engineer for parse/quality issues or DevOps for infra/latency issues.

---

## Step 5 — Shadow-compare against the OpenAI path

You have two options:

### Option A — Manual diff (simplest, slowest)
Run 20 tickets through the OpenAI path (dev env), then flip the flag and run the **same** 20 tickets through the SLM path. Diff the final State (intent, resolution_type, team, response_english). Look for:
- Where do they disagree on intent? Is the SLM right or wrong?
- Are the responses factually consistent, even if phrased differently?
- Does the team assignment match?

### Option B — Parallel shadow mode (requires a small code change)
Add a shadow node that runs the SLM path alongside the OpenAI path and logs both results without affecting routing. If you want this, ping me — it's ~50 lines of code, 1–2 hours work.

For the first rollout, Option A is enough.

---

## Step 6 — Staging / canary

If dev looks good, promote to staging (same env var changes on the staging task def). Run for 2–3 days. Watch:

1. **JSON parse rate** (should be 100% — any failures get flagged open to human review but cost you trust)
2. **Human review approval rate** — has it dropped vs OpenAI baseline?
3. **Customer-facing response quality** — spot-check a dozen responses for tone/accuracy
4. **Cost dashboard** — OpenAI spend should visibly drop, CloudWatch compute for kiki-ollama should appear

If any metric degrades significantly, flip the flag back (Step 8 below).

---

## Step 7 — Production rollout

Two flavors:

### 7a. Full cutover (aggressive)
- Update prod task def: `USE_KIKI_SLM=true`
- Force new deployment
- Watch for 1 hour
- If anything breaks, roll back (Step 8)

### 7b. Gradual by intent category (safer — requires a small code change)
Only route certain intents through the SLM. This isn't in the PR — you'd need to add a pre-filter that checks `category` (from a light OpenAI triage) and routes only `customer_feedback` + `other` to the SLM. Takes ~2 hours of code. I'd recommend skipping this and just doing 7a with a ready rollback.

---

## Step 8 — Rollback (keep this bookmarked)

**You always have two rollback levers:**

### Instant rollback — flip the flag (no DevOps involvement)
1. Update the Loopper agent task definition: `USE_KIKI_SLM=false`
2. Register new revision
3. Force new deployment on the agent service
4. Traffic is back on OpenAI within ~60 seconds
5. Leave Kiki Ollama service running (it's idle, $0 marginal cost)

### Full rollback — revert the PR
1. `git revert 15a2ccd` on `dev`
2. Merge the revert
3. Normal deploy pipeline
4. Also flip the flag off, just for belt + suspenders

**Do NOT** try to roll back by scaling kiki-ollama to 0 and keeping `USE_KIKI_SLM=true`. That leaves the agent hitting a dead endpoint and failing open on every ticket — all tickets go to human review, which is worse than the OpenAI path being slow.

---

## Monitoring dashboard you should set up

In your existing agent dashboard, add these panels:

1. **SLM-path count** — count of log events containing `── [KIKI] SLM returned`
2. **Avg SLM turns** — histogram of `turns=N` values from the KIKI log line
3. **SLM failure rate** — count of `KikiInferenceError` / total SLM invocations
4. **State override count** — count of `overriding resolution_type to requires_human_action` events (triggered by `validate_response` post-check)
5. **Cost delta** — OpenAI spend (from `agent_metrics.py`) vs previous 7-day average

If (1) drops or (3) spikes, you have a problem. If (4) spikes, the SLM is producing responses that fail the deterministic validator — might be a fine-tuning issue.

---

## FAQ for you

**Q: What if the ML engineer ships a new model version?**
A: DevOps updates `KIKI_GGUF_S3_URI` in the kiki-ollama task def and redeploys. Your agent code doesn't change. You don't need to do anything unless you want to re-shadow-test the new version before it affects traffic.

**Q: Can I run some tickets through SLM and others through OpenAI at the same time?**
A: Not with this PR — the flag is compile-time. You'd need to either run two ECS services with different flags, or add a runtime branch node. Ping me if you want the runtime branch version.

**Q: The SLM is slower than OpenAI. Is that expected?**
A: On a 2 vCPU Fargate task, yes. 5–15s per ticket vs ~3–5s for OpenAI. Cost savings are the reason — you trade latency for money. If latency matters more than cost, DevOps can move to a GPU task (~$200/mo).

**Q: Why is the `response_english` sometimes empty?**
A: Happens when the SLM fails open (caught `KikiInferenceError`). The ticket still flows through to `localize_response` and `human_review_gate` with the existing State fields. The human reviewer sees an empty response and knows to write one manually. Not ideal, but safer than blocking the ticket.

**Q: What happens if kiki-ollama goes down mid-request?**
A: The `invoke_kiki` client raises `KikiInferenceError`. LangGraph's `_llm_retry` retries the node up to 3 times with exponential backoff (2s, 4s, 8s). If all 3 fail, the node's fail-open path runs: ticket proceeds with `is_valid=True`, `confidence=0.0`, no response, `rejection_type=None`. Downstream human_review_gate flags it for manual handling.

---

## What you are NOT responsible for

- Model quality / retraining → ML engineer
- Ollama uptime, scaling, networking → DevOps
- Writing new features on top of the SLM path → separate PRs after rollout

Your contract is: "When `USE_KIKI_SLM=true`, real Loopper tickets flow through the SLM path successfully in production, and I can roll back in <5 minutes if something goes wrong."
