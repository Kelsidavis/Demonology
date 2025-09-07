
# AI Agent Tool Capabilities (Updated)

> This document summarizes what the Demonology agent can do **right now** in your workspace, including the
> latest tooling and runtime upgrades (free web search backends, CLI history, TUI scrollback, config flags, etc.).

## 🚨 Critical mindset for agents
**You have broad tool access through the framework. Use tools directly instead of guessing.**
Framework-level safety checks exist (timeouts, basic command guards), but **you are responsible** for acting safely
and ethically in this workspace.

---

## Available Tool Categories

### 1) 📁 File Operations (`file_operations`)
- Full filesystem access (absolute paths allowed)
- Create / read / write / delete; list directories; move/copy
- Works across all file types

**Examples**
```json
{"operation": "read", "path": "/etc/hosts"}
{"operation": "write", "path": "./notes/todo.md", "content": "- [ ] Ship patch\n"}
{"operation": "list", "path": ".", "recursive": false}
```

---

### 2) 🔍 Codebase Analysis (`codebase_analysis`)
- File tree / indexing (size & binary-aware)
- Regex/substring grep across large codebases

**Examples**
```json
{"operation": "tree", "path": ".", "depth": 2}
{"operation": "grep", "query": "class .*Tool\(", "regex": true, "path": "demonology/tools"}
```

---

### 3) ⚡ Code Execution (`code_execution`)
- Execute Python or Bash in this environment
- Install packages / run scripts with timeouts

**Examples**
```json
{"language": "python", "code": "import sys; print(sys.version)"}
{"language": "bash", "code": "ls -la && echo ok"}
```

---

### 4) 🏗️ Project Planning (`project_planning`)
- Analyze a repo and propose next steps
- Generate plans / scaffolds / workflow suggestions

**Examples**
```json
{"action": "analyze_existing", "project_name": "."}
{"action": "create_new", "project_name": "web_app", "technology_stack": "React+Node.js"}
```

---

### 5) 🎨 Media Tools
#### Image Generation (`image_generation`)
- Text-to-image with style & size controls

#### Image Analysis (`image_analysis`)
- OCR and UI/mockup/diagram understanding

**Examples**
```json
{"prompt": "a cozy reading nook, watercolor", "size": "1024x1024"}
{"operation": "analyze", "path": "screenshots/login.png"}
```

---

### 6) 🔧 Reverse Engineering Suite
- **Disassembler** (`disassembler`): objdump/radare2
- **Hex Editor** (`hex_editor`): dump/search/patch/info
- **Pattern Search** (`pattern_search`): strings/regex/hex/yara
- **Debugger** (`debugger`): GDB info/disas/script
- **Ghidra** (`ghidra_analysis`): headless analysis & JSON export

**Examples**
```json
{"binary_path": "./bin/app", "tool": "objdump", "architecture": "auto", "section": ".text"}
{"file_path": "./bin/app", "operation": "search", "search_pattern": "7f454c46"}
{"binary_path": "./bin/app", "analysis_type": "functions", "output_format": "json"}
```

---

### 7) 🎵 Audio Synthesis & Sheet Music
- **Waveform Generator** & **Synthesizer**
- **Audio Analysis** for spectral features
- **Described SFX**: create sweeps/noises from plain-English prompts; can quantize pitch ranges to a musical key (e.g. *A minor*) when requested
- **Sheet Music OMR**: convert sheet music images/PDFs to MusicXML/MIDI/WAV using Audiveris

**Examples**
```json
{"tool": "described_sfx", "text": "a rising whoosh that lands on A4, in A minor", "duration": 2.5}
{"tool": "synthesizer", "wave": "saw", "notes": ["A4","C5","E5"], "envelope": {"attack": 0.01, "release": 0.6}}
{"tool": "sheet_music_omr", "input_path": "score.png", "make_midi": true, "make_wav": true, "tempo_bpm": 120}
```

---

### 8) 🌐 Web Integration
- **DuckDuckGo IA** (`web_search`)
- **Reddit** (`reddit_search`): via PRAW or public JSON fallback
- **Free Extras** (no key required):
  - **WikipediaSearchTool** (`wikipedia_search`)
  - **HackerNewsSearchTool** (`hackernews_search`)
  - **StackOverflowSearchTool** (`stackoverflow_search`)
  - **OpenWebSearchTool** (`open_web_search`) – fans out to the above, optionally including DuckDuckGo IA

**Examples**
```json
{"name":"wikipedia_search","arguments":{"query":"A minor chord","limit":5}}
{"name":"hackernews_search","arguments":{"query":"LLM retrieval","tags":"story","hits_per_page":10}}
{"name":"stackoverflow_search","arguments":{"query":"python asyncio timeout","pagesize":8}}
{"name":"open_web_search","arguments":{"query":"vector search evaluation","limit_per_source":5,"include_ddg":true}}
```

---

## Runtime & UX Enhancements

### CLI
- **Input history** with arrow keys on Unix (readline)
- **`/history [N]`** command (works on all platforms)
- Writes history to `history.txt` in your config dir
- More resilient streaming (SSE headers, heartbeat timeout), retry on `429` with `Retry-After`

**Env**
- `DEMONOLOGY_HISTORY_MAX` (default `1000`)

### TUI
- **Scrollback limit** to keep UI snappy during long streams
- Safer theme switching & Windows-safe imports
- Throttled layout updates (less flicker)

**Env**
- `DEMONOLOGY_SCROLLBACK_LIMIT` (default `400`)

### Config & Client
- New API flags aligned with client:
  - `sse_heartbeat_timeout` (default `120.0` seconds)
  - `allow_server_restart` (opt-in; disabled by default)
- Safe, atomic YAML saves with backups; schema versioning
- Minimal env overrides: `DEMONOLOGY_API_BASE_URL`, `DEMONOLOGY_API_MODEL`

### HTTP (web tools)
- Timeouts & retries with exponential backoff
- Env knobs:
  - `DEMONOLOGY_HTTP_TIMEOUT` (default `15`)
  - `DEMONOLOGY_HTTP_RETRIES` (default `2`)
  - `DEMONOLOGY_HTTP_BACKOFF` (default `0.6`)

---

## Using the Registry

```python
from demonology.tools import create_default_registry, to_openai_tools_format, load_report

registry = create_default_registry()
print(registry.list())                 # names of registered tools
print(load_report(registry))           # what loaded vs skipped (with reasons)
openai_tools = to_openai_tools_format(registry)  # function-call schemas
```

**Calling a tool**
```python
res = await registry.call("wikipedia_search", query="A minor chord", limit=5)
if res.get("success"):
    print(res["results"][:3])
```

---

## Safety & Good Practices
1. Prefer tools over speculation; chain tools for complex tasks.
2. Keep workspace safety in mind even with broad local access.
3. For long sessions, use `/history` and adjust scrollback/history envs.
4. If a tool isn’t present on the host (e.g., `objdump`, `ghidra`), it will be skipped automatically and reported by `load_report()`.

---

## Slash Commands (CLI)
- `/help` — show command help
- `/status` — runtime health and last request info
- `/tools` — list registered tools
- `/history [N]` — show last N inputs (default 20)
- `/context` — show/troubleshoot conversation context
- `/logs` — print recent log entries

---

WoW pipeline: MPQ extract → WDT/ADT terrain → M2/M3/WMO models → scene bundling (terrain tiles, model library, placements)
