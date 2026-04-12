#!/usr/bin/env bash
# Kiki Ollama sidecar entrypoint.
#
# Flow:
#   1. Download the GGUF from S3 (only if not already present on disk)
#   2. Start `ollama serve` in the background
#   3. Wait for the server to be ready
#   4. `ollama create kiki-sft-v1` from the Modelfile (only if not already loaded)
#   5. Foreground the ollama serve process so ECS sees a running container

set -euo pipefail

log() { echo "── [KIKI-OLLAMA] $*" >&2; }

: "${KIKI_GGUF_S3_URI:?KIKI_GGUF_S3_URI must be set (e.g. s3://bucket/kiki-sft-v1.gguf)}"
: "${KIKI_GGUF_PATH:?KIKI_GGUF_PATH must be set}"
: "${KIKI_MODEL_NAME:?KIKI_MODEL_NAME must be set}"

mkdir -p "$(dirname "$KIKI_GGUF_PATH")"

# --- Step 1: Download GGUF from S3 ------------------------------------------
if [[ -f "$KIKI_GGUF_PATH" ]]; then
    log "GGUF already at $KIKI_GGUF_PATH — skipping download"
else
    log "downloading $KIKI_GGUF_S3_URI → $KIKI_GGUF_PATH"
    aws s3 cp "$KIKI_GGUF_S3_URI" "$KIKI_GGUF_PATH"
    log "download complete: $(du -h "$KIKI_GGUF_PATH" | cut -f1)"
fi

# --- Step 2: Start ollama serve in background -------------------------------
log "starting ollama serve on $OLLAMA_HOST"
ollama serve &
OLLAMA_PID=$!

# --- Step 3: Wait for server readiness (up to 60s) --------------------------
READY=false
for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
        READY=true
        log "ollama ready after ${i}s"
        break
    fi
    sleep 1
done

if [[ "$READY" != "true" ]]; then
    log "ollama failed to become ready within 60s — exiting"
    kill $OLLAMA_PID 2>/dev/null || true
    exit 1
fi

# --- Step 4: Create the model from the Modelfile if not already loaded ------
if ollama list 2>/dev/null | grep -q "^${KIKI_MODEL_NAME}[[:space:]]"; then
    log "model '$KIKI_MODEL_NAME' already loaded"
else
    log "creating model '$KIKI_MODEL_NAME' from /Modelfile..."
    # Modelfile's FROM line points at a placeholder — we rewrite it to the real path
    sed "s|__KIKI_GGUF_PATH__|$KIKI_GGUF_PATH|" /Modelfile > /tmp/Modelfile.rendered
    ollama create "$KIKI_MODEL_NAME" -f /tmp/Modelfile.rendered
    log "model created"
fi

# --- Step 5: Warm up the model so the first real request isn't a cold start -
log "warming up model..."
curl -sf http://127.0.0.1:11434/api/chat \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$KIKI_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"stream\":false,\"options\":{\"num_predict\":1}}" \
    >/dev/null 2>&1 || log "warmup call failed (non-fatal)"

log "ready — serving on $OLLAMA_HOST"

# --- Step 6: Foreground the ollama process so ECS knows the container is up -
wait $OLLAMA_PID
