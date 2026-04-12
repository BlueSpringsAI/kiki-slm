# Kiki SLM Rollout — Handoff Docs

Three separate docs, one per role, in the order the work flows. Each is
self-contained — read only the one for your role.

```
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ ML Engineer      │  →   │ DevOps Engineer  │  →   │ Agent Engineer   │
│ (kiki-train)     │      │ (infra)          │      │ (loopper-agent)  │
├──────────────────┤      ├──────────────────┤      ├──────────────────┤
│ Train model      │      │ Build image      │      │ Review PR        │
│ Export GGUF      │      │ Push to ECR      │      │ Merge to dev     │
│ Smoke test       │      │ Deploy ECS task  │      │ Flip env flag    │
│ Upload to S3     │      │ Service Connect  │      │ Shadow test      │
│                  │      │ Security groups  │      │ Prod rollout     │
│                  │      │                  │      │                  │
│ Hands off:       │      │ Hands off:       │      │ Hands off:       │
│ S3 URI           │      │ kiki-ollama:     │      │ (end state —     │
│ + version        │      │ 11434 DNS        │      │  tickets flow    │
│                  │      │                  │      │  through SLM)    │
└──────────────────┘      └──────────────────┘      └──────────────────┘
```

| Doc | Audience | Owns |
|---|---|---|
| [`01-ml-engineer.md`](./01-ml-engineer.md) | kiki-train repo owner | Training, eval, GGUF export, S3 upload |
| [`02-devops-engineer.md`](./02-devops-engineer.md) | Infra / DevOps engineer | ECR, ECS task, Service Connect, SGs |
| [`03-agent-engineer.md`](./03-agent-engineer.md) | Looper-Support-Agent-Server owner | PR review, env flag, shadow test, rollout |

## How to read them

- **If you only read one doc**, read the one for your role.
- **Don't read the other docs** unless a handoff is unclear — they have enough detail to be noise for other roles.
- **Each doc has a "What you are NOT responsible for" section** at the bottom. If something feels like it should be yours but isn't listed in your doc, ping the doc owner.

## The contract between roles

Each handoff is one short message:

**ML → DevOps:**
```
Model: kiki-sft-v1
S3 URI: s3://loopper-models-eu-central-1/kiki-sft-v1-Q4_K_M.gguf
Eval: intent=51%, parse=100%, tool_f1=87%
Smoke test on local Ollama: PASS
```

**DevOps → Agent engineer:**
```
Endpoint: http://kiki-ollama:11434
Model name: kiki-sft-v1
CloudWatch: /ecs/kiki-ollama
Smoke test from inside VPC: PASS
Ready to flip USE_KIKI_SLM=true
```

**Agent engineer → the rest of the org:**
```
Kiki SLM is live in production.
Before: $0.028/ticket (OpenAI o4-mini + gpt-5-mini)
After:  $0.005/ticket (OpenAI for translation only + self-hosted SLM)
Savings: ~82% at current volume
Rollback: flip USE_KIKI_SLM=false + redeploy (60 seconds)
```

## Not covered

- Streaming responses (Ollama supports it, not wired through)
- Multi-language SLM (current model is English-only, still uses OpenAI for translation)
- GPU deployment (CPU Fargate is enough for Loopper's current volume)
- A/B traffic splitting (the flag is compile-time; two ECS services if you need real A/B)

Open a separate issue for any of these after the initial rollout is stable.
