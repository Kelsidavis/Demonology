# ⛧ Demonology CLI ⛧ 

Conjure local LLMs through a mystical terminal UI with streaming chat, resilient connections,
tool integrations, and now: CLI history, scrollback control, free web search, RE tools, and audio SFX.

## ✨ Highlights
- **Ultra‑Stability**: auto‑restart supervisor, retries/backoff, heartbeat monitoring
- **Context Mastery**: smart trimming, `/context`, `/optimize`, `/trim smart`
- **CLI QoL**: ↑/↓ history (Unix), `/history` popover, clean shutdown
- **TUI QoL**: scrollback limit (env), smoother updates, safer theme switching
- **Tool Arsenal**: file ops, code exec, project planning, image gen, **free web search**, **RE suite**, **audio SFX**

## 🚀 Quick Start
```bash
# Terminal 1
./llama-server-nosudo.sh

# Terminal 2
demonology
```
Or run everything under the **auto‑restart supervisor**:
```bash
./auto-restart-server_patched.sh
```

## 📦 Install
See **INSTALL.md** for detailed options. Minimal dev setup:
```bash
pip install -r requirements.txt
pip install -e .
```

## ⚙️ Config
`~/.config/demonology/config.yaml` (Windows: `%APPDATA%\demonology\config.yaml`)
```yaml
api:
  base_url: "http://127.0.0.1:8080/v1"
  model: "Qwen3-30B"
  sse_heartbeat_timeout: 120.0
  allow_server_restart: false
ui:
  theme: "amethyst"
tools:
  enabled: true
  allowed_tools: ["file_operations","code_execution","web_search","wikipedia_search","hackernews_search","stackoverflow_search","open_web_search","project_planning","image_generation","disassembler","hex_editor","pattern_search","debugger","ghidra_analysis","waveform_generator","synthesizer","audio_analysis","described_sfx"]
```

## 🛠️ Tools (excerpt)
- **Search**: DuckDuckGo, **Wikipedia**, **HN**, **StackOverflow**, **OpenWeb**
- **Project**: planning + scaffolding
- **Exec**: Python/Bash
- **Image**: generation via free backends
- **RE**: objdump/radare2, Hex, Patterns (YARA), GDB, Ghidra
- **Audio**: waveform, synth, analysis, *described SFX in key*

Use `/tools` or see `TOOLS.md` for full details.

## 🧰 Supervisor
`auto-restart-server_patched.sh` — single instance via `flock`, health checks, backoff, optional client monitoring.
A `demonology-auto-restart.service` (systemd) is provided for unattended runs.

## 🧑‍💻 Development
```bash
pip install -r requirements-dev.txt
pytest
black demonology/ && isort demonology/
```

Happy conjuring! 🔮



Copyright (c) 2025 Kelsi Davis

Permission is granted to use, copy, modify, and distribute this software 
for personal, educational, or research purposes only. Commercial use, 
including selling, offering paid services, or integrating into 
commercial products, requires prior written permission from the author.

