\
#!/usr/bin/env bash
# auto-restart-server_patched.sh
# Robust supervisor for llama server + Demonology client
# - Single instance via flock
# - Health checks with retries
# - Rate-limited restarts + exponential backoff
# - Optional Demonology client monitoring
# - Env-driven config; sane defaults
# - Aligns with client env (LLAMA_RESTART_SCRIPT)

set -euo pipefail

########################################
# Config (env overrides supported)
########################################
LLAMA_SCRIPT="${LLAMA_SCRIPT:-/home/k/Desktop/Demonology/llama-server-stable.sh}"
DEMONOLOGY_CMD="${DEMONOLOGY_CMD:-cd /home/k/Desktop/Demonology && python -m demonology}"
LAUNCH_DEMONOLOGY="${LAUNCH_DEMONOLOGY:-1}"          # 1=launch client, 0=skip
MONITOR_CLIENT="${MONITOR_CLIENT:-1}"               # 1=restart client if it exits

SERVER_HEALTH_URL="${SERVER_HEALTH_URL:-http://127.0.0.1:8080/health}"
SERVER_MODELS_URL="${SERVER_MODELS_URL:-http://127.0.0.1:8080/v1/models}"
CHECK_INTERVAL="${CHECK_INTERVAL:-30}"              # seconds between health checks
RESTART_DELAY="${RESTART_DELAY:-5}"                 # delay before restart (seconds)
MAX_RESTARTS="${MAX_RESTARTS:-10}"                  # max restarts per hour
RESTART_WINDOW="${RESTART_WINDOW:-3600}"            # seconds (1h)

HEALTH_RETRIES="${HEALTH_RETRIES:-3}"               # consecutive failures before restart
HEALTH_RETRY_DELAY="${HEALTH_RETRY_DELAY:-5}"       # seconds between quick retries
BACKOFF_BASE="${BACKOFF_BASE:-2}"                   # exponential backoff base delay (seconds)
BACKOFF_CAP="${BACKOFF_CAP:-60}"                    # cap backoff to 60s

LOG_DIR="${LOG_DIR:-$HOME/.cache/demonology/auto-restart}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/auto-restart.log}"

########################################
# Single-instance lock
########################################
mkdir -p "$LOG_DIR"
exec 9>"/tmp/demonology-auto-restart.lock"
if ! flock -n 9; then
  echo "Another auto-restart instance is running. Exiting."
  exit 1
fi

########################################
# Logging helpers
########################################
log() {
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$ts] $*" | tee -a "$LOG_FILE"
}

banner() {
  echo "🚀 Demonology Auto-Restart Manager (patched)" | tee -a "$LOG_FILE"
  echo "============================================" | tee -a "$LOG_FILE"
  echo "• LLAMA_SCRIPT:       $LLAMA_SCRIPT" | tee -a "$LOG_FILE"
  echo "• DEMONOLOGY_CMD:     $DEMONOLOGY_CMD" | tee -a "$LOG_FILE"
  echo "• Check Interval:     ${CHECK_INTERVAL}s" | tee -a "$LOG_FILE"
  echo "• Max Restarts:       $MAX_RESTARTS per $RESTART_WINDOW s" | tee -a "$LOG_FILE"
  echo "• Health URL:         $SERVER_HEALTH_URL" | tee -a "$LOG_FILE"
  echo "• Models URL:         $SERVER_MODELS_URL" | tee -a "$LOG_FILE"
  echo "• Launch Client:      $LAUNCH_DEMONOLOGY (monitor=$MONITOR_CLIENT)" | tee -a "$LOG_FILE"
  echo "============================================" | tee -a "$LOG_FILE"
}

########################################
# Health check
########################################
check_server() {
  # Prefer /health when available; fall back to /v1/models
  if curl -fsS --connect-timeout 5 --max-time 10 "$SERVER_HEALTH_URL" >/dev/null 2>&1; then
    return 0
  fi
  curl -fsS --connect-timeout 5 --max-time 10 "$SERVER_MODELS_URL" >/dev/null 2>&1
}

########################################
# Start server
########################################
start_server() {
  log "🟢 Starting llama server..."
  # Best-effort kill of old processes
  pkill -f llama-server >/dev/null 2>&1 || true
  pkill -f llama\.cpp >/dev/null 2>&1 || true
  sleep 2

  if [[ ! -x "$LLAMA_SCRIPT" ]]; then
    log "❌ LLAMA_SCRIPT not found or not executable: $LLAMA_SCRIPT"
    return 1
  fi

  # Export env so the Python client can reuse the same script on restart
  export LLAMA_RESTART_SCRIPT="$LLAMA_SCRIPT"

  # Start server in the background; note: if the script double-forks, $! may not be the final server PID.
  bash "$LLAMA_SCRIPT" >>"$LOG_FILE" 2>&1 &
  SERVER_WRAPPER_PID=$!
  log "📊 Server wrapper started with PID: $SERVER_WRAPPER_PID"

  # Wait for readiness
  log "⏳ Waiting for server to become ready..."
  for i in $(seq 1 30); do
    if check_server; then
      log "✅ Server is ready and responding!"
      return 0
    fi
    sleep 2
    log "   • Attempt $i/30..."
  done
  log "❌ Server failed to become ready within ~60 seconds"
  return 1
}

########################################
# Stop server (best-effort)
########################################
stop_server() {
  if [[ -n "${SERVER_WRAPPER_PID:-}" ]]; then
    log "🛑 Stopping server wrapper (PID: $SERVER_WRAPPER_PID)..."
    kill -TERM "$SERVER_WRAPPER_PID" 2>/dev/null || true
    sleep 2
    kill -KILL "$SERVER_WRAPPER_PID" 2>/dev/null || true
  fi
  # Also kill underlying processes just in case
  pkill -f llama-server >/dev/null 2>&1 || true
  pkill -f llama\.cpp >/dev/null 2>&1 || true
}

########################################
# Restart with rate limit + backoff
########################################
RESTART_COUNT=0
WINDOW_START=$(date +%s)

restart_server() {
  local now; now=$(date +%s)
  # Reset the window if needed
  if (( now - WINDOW_START > RESTART_WINDOW )); then
    RESTART_COUNT=0
    WINDOW_START=$now
    log "🔄 Reset restart counter (new hour window)"
  fi

  # Rate limit
  if (( RESTART_COUNT >= MAX_RESTARTS )); then
    local wait_time=$(( RESTART_WINDOW - (now - WINDOW_START) ))
    log "⚠️  Maximum restart limit ($MAX_RESTARTS) reached; waiting ${wait_time}s..."
    sleep "$wait_time"
    RESTART_COUNT=0
    WINDOW_START=$(date +%s)
  fi

  RESTART_COUNT=$(( RESTART_COUNT + 1 ))
  local backoff=$(( BACKOFF_BASE * 2 ** (RESTART_COUNT - 1) ))
  if (( backoff > BACKOFF_CAP )); then backoff="$BACKOFF_CAP"; fi
  log "🔄 Server restart #$RESTART_COUNT (this window); backoff=${backoff}s"

  stop_server
  log "⏸️  Waiting ${RESTART_DELAY}s before restart..."
  sleep "$RESTART_DELAY"

  if start_server; then
    log "✅ Server successfully restarted"
    # small cool-down to avoid flapping
    sleep "$backoff"
    return 0
  else
    log "❌ Server restart failed; will retry after backoff ${backoff}s"
    sleep "$backoff"
    return 1
  fi
}

########################################
# Demonology client
########################################
start_client() {
  if [[ "$LAUNCH_DEMONOLOGY" != "1" ]]; then
    return 0
  fi
  log "🎭 Starting Demonology client..."
  # Ensure client sees LLAMA_RESTART_SCRIPT
  export LLAMA_RESTART_SCRIPT="$LLAMA_SCRIPT"
  bash -lc "$DEMONOLOGY_CMD" >>"$LOG_FILE" 2>&1 &
  DEMONOLOGY_PID=$!
  log "🎭 Demonology started with PID: $DEMONOLOGY_PID"
}

check_client() {
  if [[ "$LAUNCH_DEMONOLOGY" != "1" || "$MONITOR_CLIENT" != "1" ]]; then
    return 0
  fi
  if ! kill -0 "${DEMONOLOGY_PID:-0}" 2>/dev/null; then
    log "⚠️  Demonology client is not running; restarting..."
    start_client
  fi
}

########################################
# Cleanup
########################################
cleanup() {
  echo
  log "🛑 Shutting down auto-restart manager..."
  if [[ -n "${DEMONOLOGY_PID:-}" ]]; then
    log "   • Stopping Demonology (PID: $DEMONOLOGY_PID)..."
    kill -TERM "$DEMONOLOGY_PID" 2>/dev/null || true
    sleep 2
    kill -KILL "$DEMONOLOGY_PID" 2>/dev/null || true
  fi
  stop_server
  log "👋 Cleanup complete"
  exit 0
}
trap cleanup SIGINT SIGTERM EXIT

########################################
# Main
########################################
banner

# Sanity checks
command -v curl >/dev/null 2>&1 || { echo "curl is required"; exit 1; }
[[ -x "$LLAMA_SCRIPT" ]] || { echo "LLAMA_SCRIPT not found or not executable: $LLAMA_SCRIPT"; exit 1; }

log "🚀 Initial server startup..."
if ! start_server; then
  log "❌ Failed to start server initially. Exiting."
  exit 1
fi

start_client

log "🔍 Starting health monitoring loop..."
log "   • Ctrl+C to stop"
echo

# Monitoring loop
while true; do
  sleep "$CHECK_INTERVAL"

  # Quick retry loop before declaring failure
  if ! check_server; then
    log "⚠️  $(date): Server health check failed"
    failed=1
    for i in $(seq 2 "$HEALTH_RETRIES"); do
      log "   • Retry check $i/$HEALTH_RETRIES..."
      sleep "$HEALTH_RETRY_DELAY"
      if check_server; then
        log "✅ Server recovered on retry $i"
        failed=0
        break
      fi
    done
    if (( failed == 1 )); then
      log "💥 Server confirmed down after ${HEALTH_RETRIES} checks; initiating restart..."
      restart_server || true
    fi
  else
    log "✅ $(date): Server health check passed"
  fi

  check_client
done
