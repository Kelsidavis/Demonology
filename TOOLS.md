
# 🛠️ Demonology Tools Reference (Updated)

This document lists the currently available tools in the Demonology CLI and how to use them.
It expands the original reference with **free web search backends**, a **reverse‑engineering suite**,
and **audio synthesis / described SFX** tools.

---

## 🔍 Search & Information Tools

### DuckDuckGo Web Search (`web_search`)
**Purpose**: General web results via DuckDuckGo Instant Answer.
**Params**: `query` (str, required), `num_results` (int=5)

**Example**
```json
{"name":"web_search","arguments":{"query":"Python 3.12 features","num_results":5}}
```

### Reddit Search (`reddit_search`)
**Purpose**: Community discussions / opinions.
**Params**: `query` (required), `subreddit` (opt), `sort` (relevance|hot|top|new|comments), `time_filter` (all|day|week|month|year), `limit` (1-25)

**Example**
```json
{"name":"reddit_search","arguments":{"query":"FastAPI vs Django","time_filter":"year","limit":8}}
```

### Wikipedia Search (`wikipedia_search`) — **NEW**
**Purpose**: MediaWiki search with snippets and canonical links.
**Params**: `query` (required), `limit` (int=5)

**Example**
```json
{"name":"wikipedia_search","arguments":{"query":"A minor chord","limit":5}}
```

### Hacker News Search (`hackernews_search`) — **NEW**
**Purpose**: HN stories/comments via Algolia.
**Params**: `query` (required), `tags` ("story"|"comment"|null), `hits_per_page` (int=10), `page` (int=0)

**Example**
```json
{"name":"hackernews_search","arguments":{"query":"LLM retrieval","tags":"story","hits_per_page":10}}
```

### Stack Overflow Search (`stackoverflow_search`) — **NEW**
**Purpose**: Stack Exchange Q&A.
**Params**: `query` (required), `tags` (opt list), `accepted` (bool), `score_min` (int), `pagesize` (int<=30)

**Example**
```json
{"name":"stackoverflow_search","arguments":{"query":"python asyncio timeout","pagesize":8}}
```

### Open Web Search (`open_web_search`) — **NEW (Aggregator)**
**Purpose**: Fan out to multiple free sources (Wikipedia, HN, SO, and optionally `web_search`). Returns merged, deduped results.
**Params**: `query` (required), `limit_per_source` (int=5), `include_ddg` (bool=false)

**Example**
```json
{"name":"open_web_search","arguments":{"query":"vector search evaluation","limit_per_source":5,"include_ddg":true}}
```

---

## 🧱 Project Creation (`project_planning`)
**Purpose**: Generate a structured project plan and (optionally) scaffold files.
**Params**: `project_name` (required), `project_description` (required), `technology_stack` (opt), `complexity` (simple|medium|complex), `save_to_file` (bool=true), `execute_plan` (bool=true)

**Example**
```json
{"name":"project_planning","arguments":{"project_name":"web_scraper","project_description":"Scrape product prices with rotation & retries","technology_stack":"Python","execute_plan":true}}
```

---

## 📁 File Operations (`file_operations`)
**Purpose**: Create/read/write/list/delete within the workspace-safe root.
**Params**: `operation` (required), `path` (opt), `content` (opt), `recursive` (opt)

**Examples**
```json
{"name":"file_operations","arguments":{"operation":"create_or_write_file","path":"README.md","content":"hello"}}
{"name":"file_operations","arguments":{"operation":"list","path":"./", "recursive":false}}
```

---

## ⚡ Code Execution (`code_execution`)
**Purpose**: Run snippets in sandboxed Python/Bash with timeouts.
**Params**: `language` ("python"|"bash"), `code`, `timeout` (s, default 15)

**Examples**
```json
{"name":"code_execution","arguments":{"language":"python","code":"print(2+2)"}}
{"name":"code_execution","arguments":{"language":"bash","code":"ls -la && echo ok"}} 
```

---

## 🎨 Image Generation (`image_generation`)
**Purpose**: Generate images (Pollinations, HF FLUX.1, Craiyon fallback).
**Params**: `prompt` (required), `style` (realistic|artistic|anime|fantasy|pixel-art|concept-art), `size` (512x512|768x768|1024x1024), `filename` (opt), `save_image` (bool=true)

**Example**
```json
{"name":"image_generation","arguments":{"prompt":"fantasy sword icon","style":"fantasy","size":"768x768"}}
```

---

## 🧩 Reverse Engineering Suite — **NEW**

> These tools are auto‑detected and registered only if their underlying binaries exist on the host.
> Use `load_report()` or `/tools` to see which ones are active.

### Disassembler (`disassembler`)
**Backends**: `objdump`, `radare2` (r2)
**Params**: `binary_path` (required), `tool` (objdump|radare2), `architecture` (x86|x86_64|arm|arm64|mips|auto), `section`, `start_address`, `end_address`

**Example**
```json
{"name":"disassembler","arguments":{"binary_path":"./bin/app","tool":"objdump","section":".text"}}
```

### Hex Editor (`hex_editor`)
**Ops**: `dump`, `search`, `patch`, `info`
**Params**: `file_path` (required), `operation`, `offset` (int), `length` (int), `search_pattern` (hex), `patch_data` (hex)

**Examples**
```json
{"name":"hex_editor","arguments":{"file_path":"./bin/app","operation":"dump","offset":0,"length":128}}
{"name":"hex_editor","arguments":{"file_path":"./bin/app","operation":"patch","offset":4096,"patch_data":"DE AD BE EF"}} 
```

### Pattern Search (`pattern_search`)
**Modes**: `strings`, `regex`, `hex_pattern`, `yara`
**Params**: `file_path` (required), `search_type`, `pattern` (depends), `min_length` (int, for strings), `encoding` (ascii|unicode|both)

**Example**
```json
{"name":"pattern_search","arguments":{"file_path":"./bin/app","search_type":"strings","min_length":6,"encoding":"both"}}
```

### Debugger (`debugger`)
**Backend**: GDB (if installed)
**Ops**: `info`, `disas`, `script`
**Params**: `binary_path` (required), `operation`, `address`, `function`, `gdb_commands`, `args`

**Example**
```json
{"name":"debugger","arguments":{"binary_path":"./bin/app","operation":"disas","function":"main"}}
```

### Ghidra Analysis (`ghidra_analysis`)
**Backend**: Ghidra headless (`analyzeHeadless`), auto‑located or via `GHIDRA_INSTALL_DIR`
**Params**: `binary_path` (required), `analysis_type` (basic|functions|strings|imports|exports|decompile|full), `output_format` (text|json|xml), `script_path` (opt), `timeout` (s)

**Example**
```json
{"name":"ghidra_analysis","arguments":{"binary_path":"./bin/app","analysis_type":"functions","output_format":"json"}}
```

**Notes**
- Install hints: `objdump` (binutils), `r2/radare2`, `gdb`, `ghidra`
- YARA requires: `pip install yara-python`

---

## 🎵 Audio Synthesis & Effects — **NEW**

### Waveform Generator (`waveform_generator`)
**Purpose**: Basic tones/noise (sine/square/saw/triangle/noise).
**Params**: `frequency`, `duration`, `amplitude`, `sample_rate`

### Synthesizer (`synthesizer`)
**Purpose**: Multi‑note synth with envelope.
**Params**: `wave`, `notes` (["A4","C5"]), `bpm`, `envelope` (`attack/decay/sustain/release`), `sample_rate`

### Audio Analysis (`audio_analysis`)
**Purpose**: Spectral / RMS / peak analysis for WAV files.
**Params**: `path` (audio file), `operation`

### Described SFX (`described_sfx`) — *text‑to‑sound*
**Purpose**: Create whooshes/risers/sweeps/noises from plain English, with optional **musical key** quantization (e.g., *A minor*).
**Params**: `text` (required), `duration` (s), `key` (e.g., "A minor"), `scale_degrees` (opt), `start_pitch_hz` (opt), `end_pitch_hz` (opt), `sample_rate`

**Examples**
```json
{"name":"described_sfx","arguments":{"text":"a rising whoosh that lands on A4", "key":"A minor","duration":2.5}}
{"name":"synthesizer","arguments":{"wave":"saw","notes":["A4","C5","E5"],"envelope":{"attack":0.01,"release":0.6}}}
```

---

## 🔧 Configuration

Edit `config.yaml`:
```yaml
tools:
  enabled: true
  allowed_tools:
    - "file_operations"
    - "web_search"
    - "reddit_search"
    - "wikipedia_search"        # NEW
    - "hackernews_search"       # NEW
    - "stackoverflow_search"    # NEW
    - "open_web_search"         # NEW
    - "project_planning"
    - "code_execution"
    - "image_generation"
    - "disassembler"            # NEW
    - "hex_editor"              # NEW
    - "pattern_search"          # NEW
    - "debugger"                # NEW
    - "ghidra_analysis"         # NEW
    - "waveform_generator"      # NEW
    - "synthesizer"             # NEW
    - "audio_analysis"          # NEW
    - "described_sfx"           # NEW

api:
  sse_heartbeat_timeout: 120.0   # aligns with client
  allow_server_restart: false     # opt-in
```

Optional env for web requests:
```
DEMONOLOGY_HTTP_TIMEOUT=15
DEMONOLOGY_HTTP_RETRIES=2
DEMONOLOGY_HTTP_BACKOFF=0.6
```

---

## 🧪 Tool Development (Quick Start)
```python
class MyCustomTool(Tool):
    def __init__(self):
        super().__init__("my_tool", "What it does")
    def to_openai_function(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": {"type":"object","properties":{}}}
    async def execute(self, **kwargs) -> Dict[str, Any]:
        return {"success": True, "echo": kwargs}
```
Register via the guarded loader in `demonology/tools/__init__.py`.

---

## 🐛 Troubleshooting
- **Tool missing?** Use `/tools` or `load_report()` to see what loaded/skipped and why.
- **Reverse‑engineering backends**: install `objdump/radare2/gdb/ghidra`. Set `GHIDRA_INSTALL_DIR` if not auto‑found.
- **YARA**: `pip install yara-python`.
- **Reddit auth (optional)**: `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`.

---

**Happy tool crafting with Demonology!** 🔮✨
