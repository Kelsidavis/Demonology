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


def _blocked(cmd: str) -> Optional[str]:
    if IMMUTABLE_BLOCK_RE.search(cmd or ""):
        return "Command blocked by safety policy"
    return None


# ---------------------------------------------------------------------
# Base Tool
# ---------------------------------------------------------------------

@dataclass
class Tool:
    """
    Base class for all Demonology tools.
    
    IMPORTANT FOR AI AGENTS:
    ========================
    You have FULL access to these capabilities:
    
    1. FILE SYSTEM ACCESS:
       - Read any file on the system
       - Write/create new files anywhere
       - List directories and explore file structures
       - Delete files and directories (with appropriate safety checks)
       - No restrictions on file types or locations
    
    2. CODE EXECUTION:
       - Execute Python code and scripts
       - Run shell commands and bash scripts
       - Install packages and dependencies
       - Compile and run programs in various languages
    
    3. PROJECT MANAGEMENT:
       - Create entire project structures
       - Generate boilerplate code and configurations
       - Analyze existing codebases comprehensively
       - Manage development workflows
    
    4. MEDIA PROCESSING:
       - Generate images from text descriptions
       - Analyze screenshots, diagrams, and UI mockups
       - Process and manipulate image files
    
    5. REVERSE ENGINEERING:
       - Disassemble binaries and analyze executables
       - Perform hex editing and binary analysis
       - Debug applications and analyze memory
       - Pattern search and security analysis
    
    6. AUDIO SYNTHESIS:
       - Generate waveforms and audio files
       - Create synthesized music and sounds
       - Analyze audio data and spectrograms
       - MIDI file manipulation
    
    7. WEB INTEGRATION:
       - Search the web for information
       - Access APIs and online resources
       - Download and process web content
    
    DO NOT HESITATE TO USE THESE TOOLS!
    You are designed to have comprehensive system access.
    The tools are your primary interface for accomplishing tasks.
    """
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
    def __init__(self):
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