# Kiki Ollama sidecar — ECS Fargate deploy

Runs the fine-tuned Kiki SLM behind a private HTTP endpoint that the Loopper
support agent calls when `USE_KIKI_SLM=true`. One file per concern:

| File | Purpose |
|---|---|
| `Dockerfile` | Image: Ollama base + AWS CLI, no GGUF baked in |
| `Modelfile` | Ollama parameters (temp, num_ctx, stop tokens) |
| `entrypoint.sh` | Boot sequence: S3 download → serve → create model → warmup |
| `task-definition.json` | ECS Fargate task def template (2 vCPU / 6 GB) |

## Prerequisites

- A GGUF file for the trained model (from `scripts/export_gguf.py`)
- An ECR repository for the image
- An S3 bucket for the GGUF
- Two IAM roles:
  - **Execution role** — standard `ecsTaskExecutionRole` (ECR pull, CloudWatch)
  - **Task role** — `kiki-ollama-task-role` with `s3:GetObject` on the GGUF key
- VPC + private subnets where the Loopper agent can reach port 11434

## Step 1 — Upload the GGUF to S3

```bash
aws s3 cp kiki-sft-v1-Q4_K_M.gguf \
    s3://YOUR-BUCKET/kiki-sft-v1-Q4_K_M.gguf
```

Version the key name (`kiki-sft-v1`, `kiki-sft-v2`, …) so you can roll back
by changing `KIKI_GGUF_S3_URI` in the task definition.

## Step 2 — Build + push the image

```bash
# From this directory
docker build -t kiki-ollama:latest .

# Tag for ECR
aws ecr get-login-password --region REGION | \
    docker login --username AWS --password-stdin \
    ACCOUNT.dkr.ecr.REGION.amazonaws.com

docker tag kiki-ollama:latest \
    ACCOUNT.dkr.ecr.REGION.amazonaws.com/kiki-ollama:latest

docker push ACCOUNT.dkr.ecr.REGION.amazonaws.com/kiki-ollama:latest
```

The image is ~500 MB — no GGUF inside.

## Step 3 — IAM task role

The task role needs exactly one permission: pulling the GGUF object.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": ["s3:GetObject"],
            "Resource": "arn:aws:s3:::YOUR-BUCKET/kiki-sft-v1-*.gguf"
        }
    ]
}
```

Trust policy (standard ECS):

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }
    ]
}
```

## Step 4 — Register the task definition

Edit `task-definition.json` and replace every `ACCOUNT`, `REGION`, and
`YOUR-BUCKET` placeholder. Then:

```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

## Step 5 — Create the ECS service

Use Service Connect so the Loopper agent can resolve `kiki-ollama` by DNS.

```bash
aws ecs create-service \
    --cluster loopper-cluster \
    --service-name kiki-ollama \
    --task-definition kiki-ollama \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={
        subnets=[subnet-xxx,subnet-yyy],
        securityGroups=[sg-kiki-ollama],
        assignPublicIp=DISABLED
    }" \
    --service-connect-configuration '{
        "enabled": true,
        "namespace": "loopper.local",
        "services": [{
            "portName": "ollama",
            "discoveryName": "kiki-ollama",
            "clientAliases": [{
                "port": 11434,
                "dnsName": "kiki-ollama"
            }]
        }]
    }'
```

After this, the Loopper agent task (in the same `loopper.local` namespace)
reaches Ollama at `http://kiki-ollama:11434` — that's what you set as
`KIKI_SLM_URL` in the agent task definition.

## Step 6 — Security group

**Kiki Ollama SG**: allow inbound TCP 11434 **only** from the Loopper agent
task's security group. **Do not** open 11434 to 0.0.0.0/0 — Ollama has no
auth. Outbound: allow HTTPS 443 so the container can fetch from S3 and ECR.

## Step 7 — Flip the agent

Update the Loopper agent task definition:

```json
{
  "environment": [
    {"name": "USE_KIKI_SLM", "value": "true"},
    {"name": "KIKI_SLM_URL", "value": "http://kiki-ollama:11434"},
    {"name": "KIKI_SLM_MODEL", "value": "kiki-sft-v1"},
    {"name": "KIKI_SLM_MAX_TURNS", "value": "4"},
    {"name": "KIKI_SLM_TIMEOUT_S", "value": "300"}
  ]
}
```

Force a new deployment on the agent service. The graph rebuilds in SLM mode
on container start. Verify with a log line:

```
── [GRAPH] USE_KIKI_SLM=true → using Kiki SLM single-node inference path
```

## Operational notes

- **First boot** takes ~60–90 s: S3 download (~2.6 GB at ~80 MB/s) + model
  load into RAM (~10 s) + warmup call. `startPeriod: 180` in the health check
  covers this.
- **Every boot** re-downloads from S3 because Fargate has no persistent disk.
  If this becomes a cost issue, switch to an EFS mount and keep the GGUF there.
- **Memory**: 4B model in Q4_K_M uses ~3.5 GB RAM. Task allocates 6 GB to
  leave headroom for Ollama + OS + 2 concurrent inferences. Do not drop below
  6 GB — OOMs are silent.
- **CPU throughput**: 2 vCPU ≈ 15–20 tokens/sec. Each ticket runs ~3 turns of
  ~700 tokens each = ~2 minutes. If you need <30 s latency, scale up to
  4 vCPU / 12 GB (~$140/mo) or move to GPU (see `GPU.md` — TODO).
- **Scaling**: ECS auto-scaling on CPU works fine. Start with `desired=1`,
  add a target-tracking policy at 70% CPU once you have real traffic.

## Rollback

Change `KIKI_GGUF_S3_URI` back to the previous GGUF key and force a new
deployment. Task boots, downloads the old GGUF, creates the model with the
same `kiki-sft-v1` name. The agent doesn't need any change — it calls the
same `kiki-sft-v1` name regardless.

Or: flip `USE_KIKI_SLM=false` on the agent and redeploy the agent task.
Traffic goes back to the OpenAI 3-node path immediately.
