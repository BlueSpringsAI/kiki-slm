# DevOps Engineer Handoff — Kiki Ollama Sidecar on ECS Fargate

**Owner:** Loopper infra / DevOps engineer.
**Goal:** run Ollama serving the fine-tuned Kiki SLM as a new ECS Fargate service, reachable from the Loopper agent via VPC-internal service discovery.
**Input:** One message from the ML engineer: `s3://<bucket>/kiki-sft-vN-Q4_K_M.gguf` + version string.
**Output:** `http://kiki-ollama:11434` (internal DNS) working from inside the agent's task.

You do NOT need to know anything about training, LoRA, or Qwen3. You're shipping a Docker container that pulls a blob from S3 and serves HTTP on port 11434.

---

## What you're deploying

A single-container ECS Fargate task. Image: Ollama base + AWS CLI. The GGUF model file (~2.6 GB) is downloaded from S3 on container boot — not baked into the image. This means you can swap models by changing an env var and redeploying; no image rebuild needed.

```
┌─────────────────────────────────────────────────────┐
│  ECS Fargate task: kiki-ollama                      │
│  ┌──────────────────────────────────────────────┐   │
│  │ Container: ollama                            │   │
│  │  1. entrypoint.sh pulls GGUF from S3         │   │
│  │  2. ollama serve on 0.0.0.0:11434            │   │
│  │  3. ollama create kiki-sft-v1 -f Modelfile   │   │
│  │  4. warmup inference call                    │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  2 vCPU / 6 GB RAM / X86_64                         │
│  Service Connect: kiki-ollama.loopper.local:11434   │
└─────────────────────────────────────────────────────┘
                    ▲
                    │ HTTP (internal SG only)
                    │
┌───────────────────┴─────────────────────────────────┐
│  ECS Fargate task: loopper-agent                    │
│  env: USE_KIKI_SLM=true                             │
│       KIKI_SLM_URL=http://kiki-ollama:11434         │
└─────────────────────────────────────────────────────┘
```

---

## Source files

All of these are in the `kiki-train` repo at `infra/ollama-kiki/`:

| File | Use |
|---|---|
| `Dockerfile` | Build the image (no GGUF baked in) |
| `entrypoint.sh` | Boot flow: S3 download → ollama serve → create model → warmup |
| `Modelfile` | Ollama model parameters (temp, num_ctx, stop tokens) |
| `task-definition.json` | ECS Fargate task def template (placeholders for your account/region) |
| `DEPLOY.md` | Long-form reference (this doc is the condensed version) |

Clone and work from `kiki-train/infra/ollama-kiki/`.

---

## Prerequisites

- AWS account with the Loopper ECS cluster (the one running the support agent)
- ECR repository creation permissions
- IAM role creation permissions
- S3 bucket that the ML engineer can `PutObject` to (create if missing)
- Existing VPC + private subnets (same as the Loopper agent)
- Existing ECS Service Connect namespace (if not, create `loopper.local`)
- Local Docker + AWS CLI v2

---

## Step 1 — Create the S3 bucket for model artifacts

```bash
REGION=eu-central-1                  # match your existing stack
BUCKET=loopper-models-${REGION}      # or whatever name works

aws s3api create-bucket \
    --bucket "$BUCKET" \
    --region "$REGION" \
    --create-bucket-configuration LocationConstraint="$REGION"

# Block public access
aws s3api put-public-access-block \
    --bucket "$BUCKET" \
    --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

# Enable versioning so rollback is easy
aws s3api put-bucket-versioning \
    --bucket "$BUCKET" \
    --versioning-configuration Status=Enabled
```

Give the ML engineer an IAM user/role with:

```json
{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
        "Resource": [
            "arn:aws:s3:::loopper-models-eu-central-1",
            "arn:aws:s3:::loopper-models-eu-central-1/*"
        ]
    }]
}
```

Hand `s3://loopper-models-eu-central-1/` to the ML engineer and wait for them to upload the GGUF.

---

## Step 2 — Create the ECR repository

```bash
aws ecr create-repository \
    --repository-name kiki-ollama \
    --region "$REGION" \
    --image-scanning-configuration scanOnPush=true

# Get the full repo URI
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REPO_URI=${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/kiki-ollama
echo "$REPO_URI"
```

Note `$REPO_URI` — you'll need it in Step 5.

---

## Step 3 — Create the task IAM roles

You need **two** roles:

### 3a. Task Execution Role (standard, reuse existing if you have one)

Standard `ecsTaskExecutionRole` with `AmazonECSTaskExecutionRolePolicy` attached. This is what ECS uses to pull the image from ECR and push logs to CloudWatch. You probably already have this from your existing Loopper agent deploys.

### 3b. Task Role (new, Kiki-specific)

This is what the **container** uses at runtime to pull the GGUF from S3.

```bash
# Trust policy
cat > trust-policy.json <<'EOF'
{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "ecs-tasks.amazonaws.com"},
        "Action": "sts:AssumeRole"
    }]
}
EOF

aws iam create-role \
    --role-name kiki-ollama-task-role \
    --assume-role-policy-document file://trust-policy.json

# Permissions: only read the kiki GGUF objects
cat > kiki-s3-read.json <<EOF
{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Action": ["s3:GetObject"],
        "Resource": "arn:aws:s3:::${BUCKET}/kiki-sft-*.gguf"
    }]
}
EOF

aws iam put-role-policy \
    --role-name kiki-ollama-task-role \
    --policy-name kiki-s3-read \
    --policy-document file://kiki-s3-read.json
```

**Scope this tightly.** The task role should only be able to read kiki GGUF files, not the entire bucket.

---

## Step 4 — Create the CloudWatch log group

```bash
aws logs create-log-group \
    --log-group-name /ecs/kiki-ollama \
    --region "$REGION"

aws logs put-retention-policy \
    --log-group-name /ecs/kiki-ollama \
    --retention-in-days 30
```

---

## Step 5 — Build and push the Docker image

```bash
cd kiki-train/infra/ollama-kiki

# Build
docker build -t kiki-ollama:latest .

# Size check — should be ~500 MB (no GGUF inside)
docker images kiki-ollama:latest

# Login to ECR
aws ecr get-login-password --region "$REGION" | \
    docker login --username AWS --password-stdin "$REPO_URI"

# Tag + push
docker tag kiki-ollama:latest "$REPO_URI:latest"
docker tag kiki-ollama:latest "$REPO_URI:v1"     # also pin a version tag
docker push "$REPO_URI:latest"
docker push "$REPO_URI:v1"
```

---

## Step 6 — Register the ECS task definition

Edit `kiki-train/infra/ollama-kiki/task-definition.json`:

- `ACCOUNT` → your AWS account ID
- `REGION` → your region
- `YOUR-BUCKET` → the bucket from Step 1
- Update the two IAM role ARNs to match what you created in Step 3
- If your GGUF filename isn't `kiki-sft-v1-Q4_K_M.gguf`, update `KIKI_GGUF_S3_URI` to match what the ML engineer actually uploaded

Then:

```bash
aws ecs register-task-definition \
    --cli-input-json file://task-definition.json \
    --region "$REGION"
```

---

## Step 7 — Create the security group

**Kiki Ollama SG** — inbound TCP 11434 **only** from the Loopper agent task SG. No public access.

```bash
# Create the Kiki SG
KIKI_SG=$(aws ec2 create-security-group \
    --group-name kiki-ollama-sg \
    --description "Kiki Ollama sidecar — inbound only from loopper-agent-sg" \
    --vpc-id vpc-xxxxxxxx \
    --query GroupId --output text)

# Find the existing Loopper agent SG
AGENT_SG=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=loopper-agent-sg" \
    --query 'SecurityGroups[0].GroupId' --output text)

# Ingress: only from agent SG on 11434
aws ec2 authorize-security-group-ingress \
    --group-id "$KIKI_SG" \
    --protocol tcp \
    --port 11434 \
    --source-group "$AGENT_SG"
```

Egress is default (all outbound) — the container needs 443 for ECR + S3, and the default egress rule allows it. Lock down further if your security policy requires.

⚠️ **Do NOT** open 11434 to `0.0.0.0/0`. Ollama has **no authentication** — anyone on the internet could use your model. Keep the SG tight.

---

## Step 8 — Create the ECS service with Service Connect

Service Connect gives the Loopper agent a stable DNS name (`kiki-ollama`) inside the namespace. If you already have a namespace like `loopper.local`, reuse it.

```bash
# Create namespace if you don't have one (one-time)
aws servicediscovery create-private-dns-namespace \
    --name loopper.local \
    --vpc vpc-xxxxxxxx \
    --region "$REGION"
# ⏳ wait ~1 minute for it to be CREATED

# Create the service
aws ecs create-service \
    --cluster loopper-cluster \
    --service-name kiki-ollama \
    --task-definition kiki-ollama \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={
        subnets=[subnet-xxx,subnet-yyy],
        securityGroups=[${KIKI_SG}],
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
    }' \
    --region "$REGION"
```

After this, any task in the `loopper.local` Service Connect namespace can reach the SLM at `http://kiki-ollama:11434`.

---

## Step 9 — Watch the first boot

```bash
# Find the task ID
TASK=$(aws ecs list-tasks --cluster loopper-cluster --service-name kiki-ollama \
    --query 'taskArns[0]' --output text)

# Watch the logs
aws logs tail /ecs/kiki-ollama --follow --region "$REGION"
```

Expected boot sequence (~60–90 seconds):

```
── [KIKI-OLLAMA] downloading s3://.../kiki-sft-v1-Q4_K_M.gguf → /models/kiki.gguf
── [KIKI-OLLAMA] download complete: 2.6G
── [KIKI-OLLAMA] starting ollama serve on 0.0.0.0:11434
── [KIKI-OLLAMA] ollama ready after 3s
── [KIKI-OLLAMA] creating model 'kiki-sft-v1' from /Modelfile...
── [KIKI-OLLAMA] model created
── [KIKI-OLLAMA] warming up model...
── [KIKI-OLLAMA] ready — serving on 0.0.0.0:11434
```

If boot fails:
- **S3 403** → task role permissions (Step 3b)
- **S3 404** → wrong bucket/key in task def (Step 6)
- **"model already loaded"** on a fresh boot → entrypoint.sh grep regex issue, safe to ignore
- **Ollama never becomes ready** → image/arch issue; check the Dockerfile built for `X86_64`, not ARM

---

## Step 10 — Verify from inside the VPC

Easiest: spin up a temporary debug task in the same SG and hit the endpoint.

```bash
# Using ECS Exec into the Loopper agent task (if it's running)
AGENT_TASK=$(aws ecs list-tasks --cluster loopper-cluster --service-name loopper-agent \
    --query 'taskArns[0]' --output text)

aws ecs execute-command \
    --cluster loopper-cluster \
    --task "$AGENT_TASK" \
    --container loopper-agent \
    --interactive \
    --command "/bin/sh"

# Inside the container:
curl -sf http://kiki-ollama:11434/api/tags
# Expected: {"models":[{"name":"kiki-sft-v1:latest",...}]}

curl -s http://kiki-ollama:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"model":"kiki-sft-v1","messages":[{"role":"user","content":"test"}],"stream":false}'
# Expected: {"message":{"role":"assistant","content":"..."}, ...}
```

If both curls work, you're done with deployment.

---

## Step 11 — Hand off to the Agent engineer

Send the Agent engineer a single message with:

```
Kiki SLM endpoint is live.

URL:            http://kiki-ollama:11434
Model name:     kiki-sft-v1
Service:        kiki-ollama (ECS Fargate)
Task def:       kiki-ollama (revision 1)
Log group:      /ecs/kiki-ollama
CloudWatch:     https://<region>.console.aws.amazon.com/cloudwatch/home?region=<region>#logsV2:log-groups/log-group/$252Fecs$252Fkiki-ollama

Health check:   passing
Smoke test:     `curl http://kiki-ollama:11434/api/tags` returns the model

Ready to flip USE_KIKI_SLM=true on the agent.
```

They then update the Loopper agent task definition and redeploy. You're done unless they ping you about connectivity issues.

---

## Monitoring (set up on day 1, not day 30)

1. **CloudWatch alarms** on the `/ecs/kiki-ollama` log group:
   - Error pattern: `ollama failed to become ready` → alarm
   - Error pattern: `download.*failed` → alarm
2. **Container Insights** on the ECS service:
   - CPU > 90% for 5 min → alarm (time to scale up)
   - Memory > 80% → alarm (OOM risk)
3. **Service-level alarm**: `HealthyTaskCount < 1` for 2 min → page
4. **Log-based metric**: count of `── [KIKI-OLLAMA] ready` events → should match deploy events. If a restart doesn't produce this line within 3 min, something's wrong.

---

## Scaling & cost

**Baseline**: 1 task, 2 vCPU / 6 GB, $60–80/mo (eu-central-1).

**Per-ticket inference**: ~5–15 seconds of CPU time (3 turns × 700 tokens × ~15 tok/s). With `OLLAMA_NUM_PARALLEL=2`, a single task handles ~20 concurrent tickets without queueing.

**When to scale up**:
- Avg ticket latency > 30s → add more tasks (horizontal) or move to GPU (vertical)
- `OLLAMA_KEEP_ALIVE` expires between bursts → increase it to 2h–24h
- Memory > 5 GB consistently → bump to 8 GB RAM task size

**GPU option** (if CPU is too slow): rewrite the task def to use EC2 launch type with a `g4dn.xlarge` instance (T4 16 GB VRAM) or similar. Fargate doesn't support GPUs. This is a bigger migration — start with CPU, move to GPU only if latency is a real problem.

---

## Rollback procedures

### Rollback the model to an older version
1. Tell ML engineer which version to use
2. Update `KIKI_GGUF_S3_URI` in the task definition
3. Register a new revision
4. Force new deployment — container pulls the old GGUF on boot

### Rollback the image
1. Pin the task def to `kiki-ollama:v0` (the previous ECR tag)
2. Register new revision
3. Force deployment

### Rollback the entire SLM path (fastest — done by the Agent engineer)
1. Agent engineer flips `USE_KIKI_SLM=false` on the Loopper agent task
2. Redeploys agent service
3. Traffic returns to the OpenAI 3-node path within 1 minute
4. Kiki Ollama service can stay running (idle costs continue) or be scaled to 0

---

## What you are NOT responsible for

- Training the model, choosing quantization, chat templates → ML engineer
- Reviewing the Loopper agent PR or flipping the `USE_KIKI_SLM` flag → Agent engineer
- Answering "why is the model wrong about X" → ML engineer

Your contract is: "When the Loopper agent task hits `http://kiki-ollama:11434/api/chat`, it gets a response within `KIKI_SLM_TIMEOUT_S`."
