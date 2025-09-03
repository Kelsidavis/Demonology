# demonology/tools/base.py
from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Safety fence
# ---------------------------------------------------------------------

def _get_default_safe_root() -> Path:
    """Get the default safe root, checking environment first."""
    env_root = os.environ.get("GRIMOIRE_SAFE_ROOT", "")
    if env_root:
        return Path(env_root).resolve()
    return Path.cwd().resolve()

SAFE_ROOT = _get_default_safe_root()

IMMUTABLE_BLOCK_PATTERNS = [
    r"rm\s+-rf\s+/(?:\s|$)",
    r"mkfs\.",
    r"dd\s+if=",
    r"shutdown\b",
    r"reboot\b",
    r"halt\b",
    r"mount\b.*\s/dev/",
    r"umount\b",
]
IMMUTABLE_BLOCK_RE = re.compile("|".join(IMMUTABLE_BLOCK_PATTERNS),
                                re.IGNORECASE | re.MULTILINE)


def _is_within(base: Path, target: Path) -> bool:
    try:
        base = base.resolve()
        target = target.resolve()
        return str(target).startswith(str(base))
    except Exception:
        return False


def _safe_path(p: str, want_dir: bool = False) -> Path:
    if not p:
        raise ValueError("Empty path")
    
    path = Path(p)
    
    # Handle absolute paths
    if path.is_absolute():
        path = path.resolve()
        if not _is_within(SAFE_ROOT, path):
            raise PermissionError(f"Refusing to operate outside SAFE_ROOT: {path}")
    else:
        # Handle relative paths
        path = (SAFE_ROOT / path).resolve()
        if not _is_within(SAFE_ROOT, path):
            raise PermissionError(f"Refusing to operate outside SAFE_ROOT: {path}")
    
    if want_dir and not str(path).endswith(os.sep):
        path = Path(str(path) + os.sep)
    return path


def _blocked(cmd: str) -> Optional[str]:
    if IMMUTABLE_BLOCK_RE.search(cmd or ""):
        return "Command blocked by safety policy"
    return None


# ---------------------------------------------------------------------
# Base Tool
# ---------------------------------------------------------------------

@dataclass
class Tool:
    name: str
    description: str
    enabled: bool = True

    def is_available(self) -> bool:
        return True

    def to_openai_function(self) -> Dict[str, Any]:
        raise NotImplementedError

    async def execute(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


# ---------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------

class ToolRegistry:
    def __init__(self, safe_root: Optional[Path] = None):
        self.safe_root = safe_root or SAFE_ROOT
        self.tools: Dict[str, Tool] = {}
        self.toolsets: Dict[str, List[Tool]] = {}

    def register_tool(self, tool: Tool, toolset: str = "default"):
        """Register a tool in the registry."""
        self.tools[tool.name] = tool
        if toolset not in self.toolsets:
            self.toolsets[toolset] = []
        self.toolsets[toolset].append(tool)

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def get_toolset(self, name: str) -> List[Tool]:
        """Get all tools in a toolset."""
        return self.toolsets.get(name, [])

    def list_tools(self) -> List[Tool]:
        """List all registered tools."""
        return list(self.tools.values())

    def list_available_tools(self) -> List[Tool]:
        """List all available tools (those that pass is_available check)."""
        return [tool for tool in self.tools.values() if tool.enabled and tool.is_available()]

    def to_openai_functions(self) -> List[Dict[str, Any]]:
        """Convert all available tools to OpenAI function format."""
        return [tool.to_openai_function() for tool in self.list_available_tools()]

    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        tool = self.get_tool(tool_name)
        if not tool:
            return {"success": False, "error": f"Tool '{tool_name}' not found"}
        
        if not tool.enabled:
            return {"success": False, "error": f"Tool '{tool_name}' is disabled"}
        
        if not tool.is_available():
            return {"success": False, "error": f"Tool '{tool_name}' is not available"}
        
        try:
            return await tool.execute(**kwargs)
        except Exception as e:
            logger.exception(f"Error executing tool '{tool_name}'")
            return {"success": False, "error": f"Tool execution failed: {str(e)}"}