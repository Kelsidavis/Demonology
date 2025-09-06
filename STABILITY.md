# 🛡️ Demonology Ultra‑Stability System (Updated)

This document reflects the current **client**, **UI**, and **supervisor** behavior you have in this repo.

## 🔧 Client‑Side (Demonology)

### Connection & Streaming
- **SSE heartbeat** (default **120s**) with configurable timeout (`api.sse_heartbeat_timeout`).
- **429 handling** with `Retry-After` honor + exponential backoff.
- **Explicit Accept headers** for streaming (`text/event-stream`) and JSON.
- **Configurable retries/timeouts** (see *Tunables* below).

### Context Management
- **Context length**: 26,768 tokens with a **2048 buffer**.
- **Smart auto‑trimming** kicks in at **85%** of available context, preserving recent/important messages.
- `/context`, `/optimize`, `/trim smart` available in the CLI.

### Optional Server Restart (Opt‑in)
- Client exposes a guarded restart hook controlled by `api.allow_server_restart` (default **false**).
- If enabled *and* `LLAMA_RESTART_SCRIPT` is set, the client triggers the same restart path as the supervisor.
- Heavy actions like GPU resets are **not** performed by the client—keep them in your server script/supervisor.

### CLI & TUI Stability QoL
- **CLI history** (Unix ↑/↓ via readline) and **`/history [N]`** popover everywhere.
- **Scrollback limit** in TUI (env `DEMONOLOGY_SCROLLBACK_LIMIT`, default **400**) to keep rendering smooth.
- Throttled layout updates and Windows‑safe imports in the TUI.

## 🚀 Server‑Side (llama.cpp + scripts)

### Main Server Script: `llama-server-nosudo.sh` ✅
- Continuous batching, defrag threshold, parallel slots, NUMA isolation, f16 caches, dual‑GPU offload.
- Runs without sudo and handles graceful shutdown.

### Supervisor: `auto-restart-server_patched.sh` ✅ (Recommended)
- **Single‑instance** via `flock`.
- Health checks against **/health** with fallback **/v1/models**.
- **Rate‑limited restarts** with **exponential backoff** (avoid flapping).
- Optional launch/monitoring of the **Demonology** client.
- Logs to `~/.cache/demonology/auto-restart/auto-restart.log`.
- Exports `LLAMA_RESTART_SCRIPT` for the client’s optional restart hook.

> You can still use the simpler `restart-llama.sh`, but the patched supervisor is the production choice.

## ⚙️ Tunables (config.yaml)

```yaml
api:
  timeout: 60.0
  max_retries: 3
  retry_delay: 2.0
  sse_heartbeat_timeout: 120.0
  allow_server_restart: false
ui:
  auto_save_conversations: true
tools:
  enabled: true
# See AGENT_CAPABILITIES.md / TOOLS.md for tool toggles
```

**Environment (optional)**
```
DEMONOLOGY_HISTORY_MAX=1000
DEMONOLOGY_SCROLLBACK_LIMIT=400
DEMONOLOGY_HTTP_TIMEOUT=15
DEMONOLOGY_HTTP_RETRIES=2
DEMONOLOGY_HTTP_BACKOFF=0.6
```

## 🧪 Expected Behavior
- Long‑running sessions with fewer stalls; automatic recovery from transient network errors.
- Clear retries/backoff in logs; CLI remains responsive thanks to trimmed scrollback.
- If the server becomes unhealthy, the supervisor restarts it with backoff and rate limiting.

## 🚨 Troubleshooting
1. **Server not ready** → check supervisor log at `~/.cache/demonology/auto-restart/auto-restart.log`.
2. **Context pressure** → run `/context` then `/optimize` or `/trim smart`.
3. **Frequent restarts** → raise `CHECK_INTERVAL` and/or reduce model size; verify GPU VRAM.
4. **Client restart permissions** → ensure `allow_server_restart: true` **and** `LLAMA_RESTART_SCRIPT` is exported if you want the CLI command to act.

_Last updated: 2025-09-06_
