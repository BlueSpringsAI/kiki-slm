# Kiki SLM Ollama sidecar — runs on ECS Fargate alongside the Loopper agent.
#
# The GGUF is NOT baked into the image. It's downloaded from S3 on container
# boot so you can swap models without rebuilding. Image stays small (~500 MB).
#
# Build:
#   docker build -t kiki-ollama:latest .
#
# Run locally (for testing):
#   docker run --rm -p 11434:11434 \
#     -e KIKI_GGUF_S3_URI=s3://your-bucket/kiki-sft-v1-Q4_K_M.gguf \
#     -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... -e AWS_REGION=... \
#     kiki-ollama:latest
#
# On ECS: the task role grants S3:GetObject, no credentials needed in env.

FROM ollama/ollama:latest

# AWS CLI v2 — needed to pull the GGUF from S3 at startup
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl unzip ca-certificates && \
    curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip && \
    unzip -q /tmp/awscliv2.zip -d /tmp && \
    /tmp/aws/install && \
    rm -rf /tmp/aws /tmp/awscliv2.zip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY Modelfile /Modelfile
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# --- Ollama runtime knobs ---
# Bind on all interfaces so other containers in the task / VPC can reach it.
ENV OLLAMA_HOST=0.0.0.0:11434
# Keep the model loaded in RAM for an hour after the last request.
# Bursty support traffic will otherwise pay cold-start every few minutes.
ENV OLLAMA_KEEP_ALIVE=1h
# Number of concurrent inference requests a single model instance handles.
ENV OLLAMA_NUM_PARALLEL=2
# Where the GGUF lands after S3 download (inside the container).
ENV KIKI_GGUF_PATH=/models/kiki.gguf
ENV KIKI_MODEL_NAME=kiki-sft-v1
# REQUIRED at runtime: s3://bucket/key.gguf
ENV KIKI_GGUF_S3_URI=""

EXPOSE 11434

# Override ollama/ollama base-image entrypoint with ours (downloads + creates model first)
ENTRYPOINT ["/entrypoint.sh"]
