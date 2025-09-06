# 🚀 Demonology Quick Start (Updated)

Get running in **2 minutes** with ultra‑stable sessions.

## ⚡ Launch
Terminal 1 (server):
```bash
./llama-server-nosudo.sh
```

Terminal 2 (client):
```bash
demonology
```

**Or automate everything:**
```bash
./auto-restart-server_patched.sh
```

## 📋 Commands you’ll use
| Command | What it does |
|---|---|
| `/help` | All commands |
| `/context` | Context stats & warnings |
| `/optimize` | Smart trimming |
| `/trim smart` | Keep important messages |
| `/history [N]` | Show your last N inputs (default 20) |
| `/restart` | Manual server restart (if enabled) |
| `/quit` | Exit |

## 🧪 Verify
- Connection OK: you see a banner and can chat
- If not: check server logs, verify `base_url` in config, or use the supervisor

## 🧿 Pro tips
- Long sessions: let trimming handle context automatically
- Slow terminals: lower scrollback via `DEMONOLOGY_SCROLLBACK_LIMIT`
- Need images, projects, RE, or sound design? Enable tools in config

Happy trails! 🔮
