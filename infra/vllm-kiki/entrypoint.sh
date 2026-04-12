#!/usr/bin/env bash
# Kiki vLLM entrypoint — downloads model from S3, starts vLLM OpenAI server.
#
# Flow:
#   1. aws s3 sync the merged fp16 model directory from S3
#   2. Start vLLM with OpenAI-compat API + tool calling enabled
#
# The model stays loaded in GPU/CPU memory for the lifetime of the container.
# vLLM handles concurrent requests via continuous batching internally.

set -euo pipefail

log() { echo "── [KIKI-VLLM] $*" >&2; }

: "${KIKI_MODEL_S3_URI:?KIKI_MODEL_S3_URI must be set (e.g. s3://bucket/kiki-sft-v1-merged/)}"
: "${KIKI_MODEL_PATH:=/models/kiki-sft-v1}"
: "${KIKI_MODEL_NAME:=kiki-sft-v1}"
: "${VLLM_MAX_MODEL_LEN:=4096}"
: "${VLLM_TOOL_PARSER:=hermes}"
: "${VLLM_TENSOR_PARALLEL:=1}"

mkdir -p "$KIKI_MODEL_PATH"

# --- Step 1: Download model from S3 -----------------------------------------
if [[ -f "$KIKI_MODEL_PATH/config.json" ]]; then
    log "model already at $KIKI_MODEL_PATH — skipping download"
else
    log "syncing model from $KIKI_MODEL_S3_URI → $KIKI_MODEL_PATH"
    aws s3 sync "$KIKI_MODEL_S3_URI" "$KIKI_MODEL_PATH" --quiet
    log "sync complete: $(du -sh "$KIKI_MODEL_PATH" | cut -f1)"
fi

# Verify required files exist
for f in config.json tokenizer_config.json; do
    if [[ ! -f "$KIKI_MODEL_PATH/$f" ]]; then
        log "ERROR: $f not found in $KIKI_MODEL_PATH — did S3 sync complete?"
        ls -la "$KIKI_MODEL_PATH/" >&2
        exit 1
    fi
done

# --- Step 2: Detect GPU availability ----------------------------------------
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    log "GPU detected: ${GPU_COUNT}x ${GPU_NAME}"
    DEVICE_FLAG=""
else
    log "No GPU detected — running on CPU (expect 15-30 tok/s)"
    DEVICE_FLAG="--device cpu"
fi

# --- Step 3: Start vLLM -----------------------------------------------------
log "starting vLLM: model=$KIKI_MODEL_NAME max_len=$VLLM_MAX_MODEL_LEN tool_parser=$VLLM_TOOL_PARSER"

exec python -m vllm.entrypoints.openai.api_server \
    --model "$KIKI_MODEL_PATH" \
    --served-model-name "$KIKI_MODEL_NAME" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --enable-auto-tool-choice \
    --tool-call-parser "$VLLM_TOOL_PARSER" \
    --tensor-parallel-size "$VLLM_TENSOR_PARALLEL" \
    --port 8000 \
    --host 0.0.0.0 \
    $DEVICE_FLAG
