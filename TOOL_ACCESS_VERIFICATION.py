#!/usr/bin/env python3
"""
Tool Access Verification Script (patched)

- Aligns with the current ToolRegistry API.
- Prints a truthful, workspace-aware capability summary.
- Gracefully skips tools that aren't registered on this host.
"""

import asyncio
import sys
from pathlib import Path

try:
    # Core registry helpers you ship
    from demonology.tools import create_default_registry, load_report
except Exception as e:
    print(f"❌ Cannot import tool registry: {e}")
    sys.exit(1)


async def verify_tool_access() -> int:
    print("🚀 DEMONOLOGY TOOL ACCESS VERIFICATION")
    print("=" * 54)

    # Create tool registry
    registry = create_default_registry()

    # Load report (what loaded vs. skipped)
    try:
        report = load_report(registry)
    except Exception:
        report = None

    # List tools
    try:
        tool_names = list(registry.list())
    except Exception:
        # fallback if registry exposes a different method
        tool_names = getattr(registry, "list_available_tools", lambda: [])()

    print(f"✅ Registry initialized with {len(tool_names)} tool(s) available\n")
    if report:
        print("🔎 Loader report (summary):")
        for k, v in report.items():
            print(f"  • {k}: {v}")
        print()

    # Helper to run a tool if present
    async def maybe_call(name: str, **kwargs):
        if name not in tool_names:
            print(f"   ◌ Skipping {name} (not registered)")
            return None
        try:
            res = await registry.call(name, **kwargs)
            ok = bool(res.get("success"))
            status = "✅" if ok else "❌"
            print(f"   {status} {name}")
            if not ok:
                print(f"      ↳ error: {res.get('error')}")
            return res
        except Exception as e:
            print(f"   ❌ {name} raised: {e}")
            return None

    print("🧪 CORE CAPABILITIES")
    print("-" * 22)

    print("📁 File operations…")
    await maybe_call("file_operations", operation="list", path=".")

    print("⚡ Code execution (Python)…")
    await maybe_call("code_execution", language="python",
                     code="print('Hello from Demonology'); import os; print(os.listdir('.'))")

    print("🔍 Codebase analysis (tree)…")
    await maybe_call("codebase_analysis", operation="tree", path=".", depth=2, max_entries=10)

    # Optional web queries (skip silently if not registered)
    print("🌐 Web search (optional)…")
    await maybe_call("wikipedia_search", query="A minor chord", limit=1)
    await maybe_call("open_web_search", query="vector search evaluation", limit_per_source=2, include_ddg=False)

    # Optional RE checks
    print("🧩 Reverse‑engineering (optional)…")
    await maybe_call("disassembler", binary_path="./bin/app", tool="objdump", section=".text")
    await maybe_call("hex_editor", file_path="./bin/app", operation="info")

    # Optional audio check (will no-op if not enabled)
    print("🎵 Audio synthesis (optional)…")
    await maybe_call("described_sfx", text="a short soft whoosh", duration=1.2)

    print("\n🎯 CAPABILITY SUMMARY (current host)")
    print("-" * 36)
    print("• Filesystem: access within your configured workspace root (or wider, if configured).")
    print("• Execution: Python/Bash snippets with timeouts.")
    print("• Codebase: tree/grep utilities for large repos.")
    print("• Web: free backends (Wikipedia/HN/SO/OpenWeb) when registered.")
    print("• RE: objdump/r2/GDB/Ghidra (if installed on host and registered).")
    print("• Audio: waveform/synth/analysis/described SFX.")
    print("\n📖 See AGENT_CAPABILITIES.md and TOOLS.md for details.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(verify_tool_access()))
    except KeyboardInterrupt:
        print("\n⚠️  Verification interrupted by user")
        raise SystemExit(130)
