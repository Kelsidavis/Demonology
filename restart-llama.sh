#!/usr/bin/env bash
# Simple server restart script for Demonology auto-restart

echo "🔄 Restarting llama server..."

# Kill existing processes
pkill -f llama-server
sleep 3

# Clear VRAM if possible
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "🧹 Clearing GPU memory..."
    nvidia-smi --gpu-reset >/dev/null 2>&1 || true
    sleep 2
fi

# Start server in background (using q8_0 extended context version)
echo "🚀 Starting server..."
cd /home/k/Desktop/Demonology
nohup ./llama-server-q8-extended.sh > /tmp/llama-server.log 2>&1 &

echo "✅ Server restart initiated"
echo "📄 Log file: /tmp/llama-server.log"