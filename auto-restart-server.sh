#!/usr/bin/env bash
set -euo pipefail

# Auto-Restart Llama Server Wrapper
# This script monitors and automatically restarts your llama server if it crashes
# It also launches Demonology and handles reconnections

LLAMA_SCRIPT="/home/k/Desktop/Demonology/llama-server-stable.sh"
DEMONOLOGY_CMD="cd /home/k/Desktop/Demonology && python -m demonology"
SERVER_URL="http://127.0.0.1:8080/v1/models"
CHECK_INTERVAL=30  # Check every 30 seconds
RESTART_DELAY=5    # Wait 5 seconds before restarting
MAX_RESTARTS=10    # Maximum restarts per hour
RESTART_WINDOW=3600  # 1 hour window

# State tracking
RESTART_COUNT=0
WINDOW_START=$(date +%s)
SERVER_PID=""

echo "🚀 Demonology Auto-Restart Manager"
echo "=================================="
echo "• Server Script: $LLAMA_SCRIPT"
echo "• Check Interval: ${CHECK_INTERVAL}s"
echo "• Max Restarts: $MAX_RESTARTS per hour"
echo "• Server URL: $SERVER_URL"
echo "=================================="

# Function to check if server is responding
check_server() {
    curl -s --connect-timeout 5 --max-time 10 "$SERVER_URL" >/dev/null 2>&1
    return $?
}

# Function to start the llama server
start_server() {
    echo "🟢 Starting llama server..."
    
    # Make sure no old processes are running
    pkill -f llama-server >/dev/null 2>&1 || true
    sleep 2
    
    # Start the server in background
    bash "$LLAMA_SCRIPT" &
    SERVER_PID=$!
    
    echo "📊 Server started with PID: $SERVER_PID"
    
    # Wait for server to become ready
    echo "⏳ Waiting for server to become ready..."
    for i in {1..30}; do  # Wait up to 60 seconds
        if check_server; then
            echo "✅ Server is ready and responding!"
            return 0
        fi
        sleep 2
        echo "   • Attempt $i/30..."
    done
    
    echo "❌ Server failed to become ready within 60 seconds"
    return 1
}

# Function to restart the server with rate limiting
restart_server() {
    local current_time=$(date +%s)
    
    # Reset counter if we're in a new hour window
    if [ $((current_time - WINDOW_START)) -gt $RESTART_WINDOW ]; then
        RESTART_COUNT=0
        WINDOW_START=$current_time
        echo "🔄 Reset restart counter (new hour window)"
    fi
    
    # Check if we've hit the restart limit
    if [ $RESTART_COUNT -ge $MAX_RESTARTS ]; then
        echo "⚠️  Maximum restart limit ($MAX_RESTARTS) reached in this hour"
        echo "   Waiting until next hour window to prevent restart loops..."
        local wait_time=$((RESTART_WINDOW - (current_time - WINDOW_START)))
        sleep $wait_time
        RESTART_COUNT=0
        WINDOW_START=$(date +%s)
    fi
    
    RESTART_COUNT=$((RESTART_COUNT + 1))
    echo "🔄 Server restart #$RESTART_COUNT (this hour)"
    
    # Kill existing server
    if [ -n "$SERVER_PID" ]; then
        echo "🛑 Stopping server (PID: $SERVER_PID)..."
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        sleep 2
        kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
    
    # Additional cleanup
    pkill -f llama-server >/dev/null 2>&1 || true
    
    # Wait before restart
    echo "⏸️  Waiting ${RESTART_DELAY}s before restart..."
    sleep $RESTART_DELAY
    
    # Try to restart
    if start_server; then
        echo "✅ Server successfully restarted"
        return 0
    else
        echo "❌ Server restart failed"
        return 1
    fi
}

# Function to handle cleanup on exit
cleanup() {
    echo
    echo "🛑 Shutting down auto-restart manager..."
    if [ -n "$SERVER_PID" ]; then
        echo "   • Stopping server (PID: $SERVER_PID)..."
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        sleep 2
        kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
    pkill -f llama-server >/dev/null 2>&1 || true
    echo "👋 Cleanup complete"
    exit 0
}

# Register cleanup function
trap cleanup SIGINT SIGTERM EXIT

# Initial server start
echo "🚀 Initial server startup..."
if ! start_server; then
    echo "❌ Failed to start server initially. Exiting."
    exit 1
fi

# Launch Demonology in background
echo "🎭 Starting Demonology client..."
bash -c "$DEMONOLOGY_CMD" &
DEMONOLOGY_PID=$!

echo "🔍 Starting health monitoring loop..."
echo "   • Press Ctrl+C to stop the auto-restart manager"
echo "   • Server health check every ${CHECK_INTERVAL}s"
echo

# Main monitoring loop
while true; do
    sleep $CHECK_INTERVAL
    
    # Check if server is still responding
    if ! check_server; then
        echo "⚠️  $(date): Server health check failed"
        
        # Try a few more times before restarting
        failed_checks=1
        for attempt in {2..3}; do
            echo "   • Retry check $attempt/3..."
            sleep 5
            if check_server; then
                echo "✅ Server recovered on retry $attempt"
                break
            else
                failed_checks=$attempt
            fi
        done
        
        if [ $failed_checks -eq 3 ]; then
            echo "💥 Server confirmed down after 3 checks, initiating restart..."
            if ! restart_server; then
                echo "❌ Server restart failed. Waiting longer before next attempt..."
                sleep 30
            fi
        fi
    else
        # Server is healthy
        echo "✅ $(date): Server health check passed"
    fi
done