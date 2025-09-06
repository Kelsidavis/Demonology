# 📦 Demonology Installation Guide (Updated)

This guide walks you through installing **Demonology CLI**, connecting to a local **llama.cpp**-compatible server,
and enabling the newest stability + tooling features (auto‑restart supervisor, free web search, RE suite, audio SFX, etc.).

## 🔧 Requirements

- **Python 3.9+** (3.10+ recommended)
- **pip** and a terminal with Unicode/UTF‑8
- A running **llama.cpp** (or compatible) server exposing an **OpenAI‑style API**
- Optional: **Node.js** (JS execution), **objdump/radare2/gdb/ghidra** (RE tools), **yara-python** (pattern scans)

## 🚀 Install

### Option A — Dev install (recommended while the repo is local)
```bash
pip install -r requirements.txt
pip install -e .
# (dev extras)
pip install -r requirements-dev.txt
```

### Option B — PyPI (when published)
```bash
pip install demonology-cli
```

## 🐳 Docker (optional)
```bash
docker build -t demonology-cli .
docker run -it --rm -v ~/.config/demonology:/root/.config/demonology demonology-cli
```

## ⚙️ Configure & First Run

```bash
# First launch creates config at:
# Linux/macOS: ~/.config/demonology/config.yaml
# Windows:     %APPDATA%\demonology\config.yaml
demonology
```

Key config fields (aligns with the updated client):
```yaml
api:
  base_url: "http://127.0.0.1:8080/v1"
  model: "Qwen3-30B"
  timeout: 60.0
  sse_heartbeat_timeout: 120.0      # NEW
  allow_server_restart: false        # NEW (opt-in)

ui:
  theme: "amethyst"
  auto_save_conversations: true

tools:
  enabled: true
  allowed_tools:
    - "file_operations"
    - "code_execution"
    - "web_search"
    - "reddit_search"
    - "wikipedia_search"           # NEW
    - "hackernews_search"          # NEW
    - "stackoverflow_search"       # NEW
    - "open_web_search"            # NEW
    - "image_generation"
    - "project_planning"
    - "disassembler"               # NEW
    - "hex_editor"                 # NEW
    - "pattern_search"             # NEW
    - "debugger"                   # NEW
    - "ghidra_analysis"            # NEW
    - "waveform_generator"         # NEW
    - "synthesizer"                # NEW
    - "audio_analysis"             # NEW
    - "described_sfx"              # NEW
```

Env quality‑of‑life:
```
DEMONOLOGY_HISTORY_MAX=1000
DEMONOLOGY_SCROLLBACK_LIMIT=400
DEMONOLOGY_HTTP_TIMEOUT=15
DEMONOLOGY_HTTP_RETRIES=2
DEMONOLOGY_HTTP_BACKOFF=0.6
```

## 🖧 Pointing at your server

- **llama.cpp**: build the server and run with `--host 0.0.0.0 --port 8080 --api-key <key>`
- **Alternatives**: Ollama or other backends that expose OpenAI‑compatible routes

Quick test:
```bash
curl -sS http://127.0.0.1:8080/v1/models | jq
```

## 🔄 Supervisor (optional but recommended)

Use the patched supervisor to auto‑restart + monitor everything:

```bash
chmod +x auto-restart-server_patched.sh
./auto-restart-server_patched.sh
```

It will: start your llama server, wait for health, optionally launch **Demonology**, and watch both with
rate‑limited, backoff restarts. A systemd unit is included for unattended runs.

## 🧪 Verify

```bash
demonology --version
demonology --help
demonology --theme infernal
```

If connection fails, check your server logs and `base_url` in config.

---

Happy conjuring! 🔮
