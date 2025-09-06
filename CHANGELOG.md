# 📜 Demonology CLI Changelog (Updated)

All notable changes are documented here. Follows **Keep a Changelog** and **SemVer**.

## [0.3.0] - 2025-09-06

### Added
- **CLI history** with ↑/↓ (Unix) via readline; `/history [N]` popover (all platforms)
- **TUI scrollback limit** (env `DEMONOLOGY_SCROLLBACK_LIMIT`) for long sessions
- **Free web tools**: Wikipedia, Hacker News, Stack Overflow, and an Open Web aggregator
- **Reverse‑engineering suite**: disassembler, hex editor, pattern search (strings/regex/hex/YARA), GDB debugger, headless Ghidra
- **Audio synthesis + described SFX** with optional key quantization (e.g., *A minor*)
- **Auto‑restart supervisor (patched)** with health checks, rate limiting, exponential backoff
- **Config flags** aligned with client: `sse_heartbeat_timeout`, `allow_server_restart`

### Changed
- Safer atomic config writes with backups and schema versioning
- More resilient streaming (SSE heartbeat, 429 retry with `Retry-After`)
- Theme switching safety and Windows‑friendly imports

### Fixed
- UI update flicker via throttled layout refresh
- Metadata corruption risks with atomic conversation/metadata writes

## [0.2.0] - 2024-09-03
- See previous release notes for Web Search, Reddit Search, Project Planning, and improved tool system.

## [0.1.0] - 2024-01-XX
- Initial release with streaming UI, theming, tool framework, and conversation management.
