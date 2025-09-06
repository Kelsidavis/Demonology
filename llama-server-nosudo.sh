#!/usr/bin/env bash
set -euo pipefail

LLAMA_BIN="/home/k/llama.cpp/build/bin/llama-server"
MODEL_PATH="/media/k/vbox1/models/Qwen3/Qwen3-14B-UD-Q5_K_XL.gguf"

CONTEXT=32768
NGL=48
THREADS=20
BATCH_SIZE=192
ALIAS="qwen3-14b"                  # 5 minute keep-alive

echo "=========================================="
echo "Model: $(basename "$MODEL_PATH")"
echo "Context: $CONTEXT | NGL: $NGL"
echo "GPU Split: layer mode across GPUs (auto)"
echo "Threads: $THREADS | Batch: $BATCH_SIZE"
echo "Parallel Slots: $PARALLEL | Keep-Alive: ${KEEP_ALIVE}s"
echo "Cache: ${CACHE_TYPE_K}/${CACHE_TYPE_V} (quantized) | Flash Attention: Enabled"


# Function to handle cleanup on exit
cleanup() {
    echo
    echo "🛑 Shutting down server gracefully..."
    if [ -n "${SERVER_PID:-}" ]; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    echo "👋 Server shutdown complete"
    exit 0
}

# Register cleanup function
trap cleanup SIGINT SIGTERM

exec "$LLAMA_BIN" \
  -m "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8080 \
  --cache-type-k bf16 \
  --cache-type-v bf16 \
  -c "$CONTEXT" \
  --batch-size "$BATCH_SIZE" \
  --ubatch-size 80 \
  --keep 3072 \
  -ngl "$NGL" \
  --main-gpu 1 \
  --tensor-split 1 \
  --alias "$ALIAS" \
  --jinja \
  --temp 0.6 \
  --top-p 0.9 \
  --min-p 0.05 \
  --repeat-penalty 1.1 \
  --repeat-last-n 256 \
  --no-warmup \
  -t "$THREADS" \
  --verbose


SERVER_PID=$!

echo "🟢 Server started with PID: $SERVER_PID"

# Wait for the server process
wait "$SERVER_PID"
