# Kiki SLM via vLLM — ECS Fargate / EC2 GPU deployment.
#
# vLLM serves an OpenAI-compatible API at /v1/chat/completions with native
# multi-turn tool calling support. The merged fp16 model is downloaded from
# S3 on container boot — not baked into the image.
#
# Build:
#   docker build -t kiki-vllm:latest .
#
# Run locally (CPU, for testing only — slow):
#   docker run --rm -p 8000:8000 \
#     -e KIKI_MODEL_S3_URI=s3://bucket/kiki-sft-v1-merged/ \
#     -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... -e AWS_REGION=... \
#     kiki-vllm:latest
#
# Run with GPU (EC2 g4dn.xlarge):
#   docker run --gpus all --rm -p 8000:8000 \
#     -e KIKI_MODEL_S3_URI=s3://bucket/kiki-sft-v1-merged/ \
#     kiki-vllm:latest

FROM vllm/vllm-openai:latest

# AWS CLI v2 — to pull model from S3 on startup
RUN pip install --no-cache-dir awscli

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# --- Config ---
# S3 prefix containing the merged fp16 model (config.json, *.safetensors, tokenizer.*)
ENV KIKI_MODEL_S3_URI=""
# Model name exposed at /v1/chat/completions
ENV KIKI_MODEL_NAME=kiki-sft-v1
# Local path where model is downloaded
ENV KIKI_MODEL_PATH=/models/kiki-sft-v1
# vLLM tuning — override via ECS task env
ENV VLLM_MAX_MODEL_LEN=4096
# Tool call parser for Qwen3 (hermes handles <tool_call> XML format)
ENV VLLM_TOOL_PARSER=hermes
# Number of workers (1 for single-GPU, increase for multi-GPU)
ENV VLLM_TENSOR_PARALLEL=1

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
