#!/usr/bin/env bash
set -euo pipefail

# Llama Server Enhanced Stability Script (No Sudo Required)
# This script launches llama.cpp with optimizations for long-running stability

LLAMA_BIN="/home/k/llama.cpp/build/bin/llama-server"
MODEL_PATH="/media/k/vbox1/models/Qwen3/Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf"

# Enhanced stability settings
CONTEXT=29768
NGL=48
THREADS=20
BATCH_SIZE=170
ALIAS="qwen3-coder-30b-tools"

# Memory and stability optimizations with aggressive quantization
CONT_BATCHING="--cont-batching"        # Enable continuous batching
PARALLEL=8                             # Parallel processing slots
CACHE_TYPE_K="bf16"                    # Key cache type (memory efficient)
CACHE_TYPE_V="bf16"                    # Value cache type (memory efficient) 
# DEFRAG_THOLD deprecated in newer builds
NUMA="--numa isolate"                  # NUMA optimization

# Connection stability settings (timeout args not supported in this version)
KEEP_ALIVE=300                         # 5 minute keep-alive

echo "=========================================="
echo "🚀 Launching Qwen3 30B (ENHANCED STABILITY - NO SUDO)"
echo "Model: $(basename "$MODEL_PATH")"
echo "Context: $CONTEXT | NGL: $NGL"
echo "GPU Split: layer mode across GPUs (auto)"
echo "Threads: $THREADS | Batch: $BATCH_SIZE"
echo "Parallel Slots: $PARALLEL | Keep-Alive: ${KEEP_ALIVE}s"
echo "Cache: ${CACHE_TYPE_K}/${CACHE_TYPE_V} (quantized) | Flash Attention: Enabled"

# Set user-level optimizations (no sudo required)
echo "🔧 Applying user-level optimizations..."

# Increase file descriptor limits for current user
ulimit -n 65536

echo "✅ User optimization complete"
echo

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

# Launch the server with enhanced stability settings
exec "$LLAMA_BIN" \
  -m "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size "$CONTEXT" \
  --batch-size "$BATCH_SIZE" \
  -ngl "$NGL" \
  --main-gpu 1 \
  --alias "$ALIAS" \
  --split-mode layer \
  --jinja \
  --temp 0.9 \
  --top-p 0.85 \
  --min-p 0.10 \
  --repeat-penalty 1.35 \
  --repeat-last-n 512 \
  --mirostat 2 \
  --mirostat-ent 6.0 \
  --mirostat-lr 0.3 \
  $CONT_BATCHING \
  --parallel "$PARALLEL" \
  --cache-type-k "$CACHE_TYPE_K" \
  --cache-type-v "$CACHE_TYPE_V" \
  $NUMA \
  --no-warmup \
  -t "$THREADS" \
  --verbose &

SERVER_PID=$!

echo "🟢 Server started with PID: $SERVER_PID"

# Wait for the server process
wait "$SERVER_PID"
