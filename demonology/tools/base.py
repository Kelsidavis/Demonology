# demonology/tools/base.py
from __future__ import annotations

import abc
import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)

# Workspace confinement
WORKSPACE_ROOT = Path(os.environ.get("DEMONOLOGY_ROOT", os.getcwd())).resolve()


def _confine(path: Path) -> Path:
    """Resolve path and ensure it remains under WORKSPACE_ROOT; disallow symlink traversal."""
    p = (WORKSPACE_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        p.relative_to(WORKSPACE_ROOT)
    except Exception:
        raise PermissionError(f"Path escapes workspace root: {p}")
    # Disallow symlink traversal for target and parents (best-effort)
    for parent in [p] + list(p.parents):
        try:
            if parent.is_symlink():
                raise PermissionError(f"Symlinked path not allowed: {parent}")
        except FileNotFoundError:
            # Missing parent during creation is okay
            pass
    return p


def _ok_path(path: Path) -> bool:
    """Check if path is valid and within workspace bounds."""
    try:
        _confine(path)
        return True
    except (PermissionError, ValueError, OSError):
        return False

# ---------------------------
# Safety filter (command guard)
# ---------------------------

# A conservative, extensible set of patterns for obviously destructive/system-disruptive actions.
# Keep these high-signal only to avoid false positives; expand cautiously.
_IMMUTABLE_BLOCK_PATTERNS: List[str] = [
    r"rm\s+-rf\s+/(?:\s|$)",                      # nuke root
    r"mkfs\.",                                    # format filesystems
    r"dd\s+if=",                                  # raw disk writes are often 'dd if=/dev/zero of=/dev/sdX'
    r">\s*/dev/sd[a-z]\b",                        # direct disk redirection
    r"shutdown\b", r"reboot\b", r"halt\b",        # immediate system disruption
    r"\bsystemctl\s+(reboot|poweroff|halt|rescue)\b",
    r"\bpoweroff\b",
    r"mount\b.*\s/dev/",                          # mounting devices
    r"\bumount\b",
    r":\(\)\s*\{.*\};:",                          # classic fork bomb
    r"\bchown\s+-R\s+root\b",                     # suspicious mass ownership changes
    r"\bchmod\s+-R\s+0\b",                        # blanket permission nuking
]

_IMMUTABLE_BLOCK_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _IMMUTABLE_BLOCK_PATTERNS]


def _blocked(cmd: str) -> Optional[str]:
    """
    Return a human-readable reason string when the provided command/snippet
    matches a dangerous pattern, otherwise None.

    NOTE: Keep return type as 'Optional[str]' to maintain compatibility with callers.
    """
    if not cmd:
        return None
    for pat in _IMMUTABLE_BLOCK_RE:
        if pat.search(cmd):
            return f"Command blocked by safety policy (pattern: {pat.pattern})"
    return None


# ---------------------------
# Tool base + registry
# ---------------------------

class Tool(abc.ABC):
    """
    Base class for agent tools.

    Safety & Scope:
      - Tools SHOULD operate within a sandboxed, workspace-scoped environment.
      - Tools MUST validate/normalize user-provided paths and block symlink escape.
      - Tools SHOULD apply reasonable resource/time/output limits to subprocess work.
      - Tools MUST honor `_blocked()` where applicable (e.g., shell/python execution).

    Implementations:
      - Provide a clear `description`.
      - Implement `to_openai_function()` with name/description/parameters schema.
      - Implement `execute()` as an async method returning a JSON-serializable dict.
      - Optionally override `is_available()` to detect runtime prerequisites.
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abc.abstractmethod
    def to_openai_function(self) -> Dict[str, Any]:
        """Return a JSON schema description for this tool (for function calling)."""
        raise NotImplementedError

    def is_available(self) -> bool:
        """Override to signal whether the tool can be used on this host."""
        return True

    @abc.abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Run the tool. Must be implemented by subclasses."""
        raise NotImplementedError


class ToolRegistry:
    """
    Lightweight registry/runner for Tool instances.

    - Registers tools by unique name.
    - Provides a unified async `call` with availability checks.
    - Returns structured failure dicts instead of raising, unless cancelled.
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            logger.warning("Overwriting tool registration for '%s'", tool.name)
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list(self) -> List[str]:
        return sorted(self._tools.keys())

    async def call(self, name: str, **kwargs) -> Dict[str, Any]:
        tool = self._tools.get(name)
        if not tool:
            return {"success": False, "error": f"Unknown tool: {name}"}

        if not tool.is_available():
            return {"success": False, "error": f"Tool '{name}' is not available in this environment"}

        try:
            result = await tool.execute(**kwargs)
            # Ensure a consistent envelope with success key
            if not isinstance(result, dict) or "success" not in result:
                return {"success": False, "error": "Tool returned invalid result payload"}
            return result
        except asyncio.CancelledError:
            # Preserve cancellation semantics for cooperative cancellation
            raise
        except Exception as e:
            logger.exception("Tool '%s' execution error", name)
            return {"success": False, "error": str(e)}

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return a list of tools with their details."""
        tools = []
        for name, tool in self._tools.items():
            tools.append({
                "name": name,
                "description": tool.description,
                "enabled": True,
                "available": tool.is_available()
            })
        return tools

    def to_openai_tools_format(self) -> List[Dict[str, Any]]:
        """Convert registered tools to OpenAI function calling format."""
        payload: List[Dict[str, Any]] = []
        for name in self.list():
            tool = self.get(name)
            if not tool:
                continue
            try:
                payload.append({"type": "function", "function": tool.to_openai_function()})
            except Exception as e:
                logger.debug("Schema export failed for %s: %s", name, e, exc_info=True)
        return payload

    async def execute_tool(self, name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name with the given kwargs."""
        return await self.call(name, **kwargs)

