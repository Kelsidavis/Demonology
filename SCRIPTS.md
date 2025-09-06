# 📜 Demonology Scripts Reference (Updated)

## 🚀 Primary Scripts

### `llama-server-nosudo.sh` ✅ **MAIN SERVER**
Optimized llama.cpp server launch (no sudo). Uses continuous batching, defrag threshold, NUMA isolation, f16 caches, dual‑GPU layering, and graceful shutdown.

**Run**
```bash
./llama-server-nosudo.sh
```

### `auto-restart-server_patched.sh` ✅ **SUPERVISOR (Recommended)**
Single‑instance watchdog with health checks, rate‑limited restarts + exponential backoff, optional Demonology client launch/monitoring, and consistent logging.

**Run**
```bash
chmod +x auto-restart-server_patched.sh
./auto-restart-server_patched.sh
```

**Notes**
- Exports `LLAMA_RESTART_SCRIPT` so the CLI’s optional restart hook targets the **same** path.
- Health checks hit `/health` first; they fall back to `/v1/models`.

### `restart-llama.sh` ⚙️ **HELPER**
A simple restart helper (pkill + relaunch). Kept for compatibility and manual use.

```bash
./restart-llama.sh
```

## 📁 Paths & Logs

```
/home/k/Desktop/Demonology/
├── llama-server-nosudo.sh
├── auto-restart-server_patched.sh
└── restart-llama.sh

Logs:
└── ~/.cache/demonology/auto-restart/auto-restart.log
```

## 🧪 Quick Checks
- Port 8080 busy? `sudo netstat -tulpn | grep 8080`
- Model path ok? `ls -la /media/k/vbox1/models/Qwen3/`
- Binary present? `ls -la /home/k/llama.cpp/build/bin/llama-server`
- Live logs: `tail -f ~/.cache/demonology/auto-restart/auto-restart.log`
