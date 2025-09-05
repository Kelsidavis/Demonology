# demonology/tools/__init__.py
"""
Modular tool system for Demonology.

This module provides a modular architecture for tools, allowing them to be
organized by functionality and loaded on demand.

🤖 FOR AI AGENTS:
=================
This tool system gives you UNRESTRICTED access to:
- File system operations (read/write/execute anywhere)
- Code execution (Python, bash, compilation)  
- Project creation and analysis
- Media processing (image generation/analysis)
- Reverse engineering tools
- Audio synthesis capabilities
- Web search and integration

USE THESE TOOLS ACTIVELY! They are your primary interface for accomplishing tasks.
Don't simulate what these tools can do - actually use them.

See AGENT_CAPABILITIES.md for comprehensive usage guide.
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from .base import Tool

logger = logging.getLogger(__name__)


def remind_agent_capabilities():
    """
    Display a reminder of agent capabilities for AI agents who might forget.
    
    Call this function if you're uncertain about what tools are available
    or if you need a reminder of your access levels.
    """
    print("""
🚨 AI AGENT CAPABILITY REMINDER 🚨
==================================

YOU HAVE FULL SYSTEM ACCESS THROUGH THESE TOOLS:

✅ FILE OPERATIONS - Read/write/delete ANY file, ANY location
✅ CODE EXECUTION - Run Python, bash, install packages, compile code  
✅ PROJECT MANAGEMENT - Create entire projects, analyze codebases
✅ IMAGE PROCESSING - Generate images, analyze screenshots/mockups
✅ REVERSE ENGINEERING - Disassemble binaries, hex editing, debugging
✅ AUDIO SYNTHESIS - Generate music, waveforms, MIDI files
✅ WEB INTEGRATION - Search web, access APIs, download content

🔥 CRITICAL POINTS:
- NO safe_root restrictions - work anywhere!
- NO permission needed - you have full access!
- DON'T simulate - USE the actual tools!
- Tools are your PRIMARY interface - use them first!

Example: Instead of saying "I would check the file...", 
DO: <tool_call><function=file_operations><parameter=operation>read</parameter><parameter=path>./file.txt</parameter></function></tool_call>

Load tools with: create_default_registry()
See AGENT_CAPABILITIES.md for detailed usage guide.
    """)


# Automatically remind when module is imported
remind_agent_capabilities()

# Import core tools
from .file_ops import FileOperationsTool
from .codebase import CodebaseAnalysisTool
from .execution import CodeExecutionTool
from .web import WebSearchTool, RedditSearchTool
from .project import ProjectPlanningTool
from .media import ImageGenerationTool, ImageAnalysisTool

# Import reverse engineering toolset
from .reverse_eng import (
    DisassemblerTool, 
    HexEditorTool, 
    PatternSearchTool, 
    DebuggingTool,
    GhidraAnalysisTool
)

# Import audio synthesis toolset
from .audio import (
    WaveformGeneratorTool,
    SynthesizerTool, 
    AudioAnalysisTool,
    MIDITool
)


def create_default_registry():
    """Create a registry with all available tools."""
    from .base import ToolRegistry
    registry = ToolRegistry()
    
    # Core tools
    registry.register_tool(FileOperationsTool(), "core")
    registry.register_tool(CodebaseAnalysisTool(), "core")
    registry.register_tool(CodeExecutionTool(), "core")
    
    # Web tools
    registry.register_tool(WebSearchTool(), "web")
    registry.register_tool(RedditSearchTool(), "web")
    
    # Project tools
    registry.register_tool(ProjectPlanningTool(), "project")
    
    # Media tools
    registry.register_tool(ImageGenerationTool(), "media")
    registry.register_tool(ImageAnalysisTool(), "media")
    
    # Reverse engineering toolset
    registry.register_tool(DisassemblerTool(), "reverse_eng")
    registry.register_tool(HexEditorTool(), "reverse_eng") 
    registry.register_tool(PatternSearchTool(), "reverse_eng")
    registry.register_tool(DebuggingTool(), "reverse_eng")
    registry.register_tool(GhidraAnalysisTool(), "reverse_eng")
    
    # Audio synthesis toolset
    registry.register_tool(WaveformGeneratorTool(), "audio")
    registry.register_tool(SynthesizerTool(), "audio")
    registry.register_tool(AudioAnalysisTool(), "audio")
    registry.register_tool(MIDITool(), "audio")
    
    return registry


def load_toolset(toolset_name: str) -> List[Tool]:
    """Load a specific toolset."""
    registry = create_default_registry()
    return registry.get_toolset(toolset_name)


def load_all_tools() -> List[Tool]:
    """Load all available tools."""
    registry = create_default_registry()
    return registry.list_tools()


def load_available_tools() -> List[Tool]:
    """Load all available tools (those that pass is_available check)."""
    registry = create_default_registry()
    return registry.list_available_tools()


# Enhanced ToolRegistry that combines modular and legacy tools
class ToolRegistry:
    """Enhanced ToolRegistry that combines modular and legacy tools."""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        logger.info("ToolRegistry initializing")
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register all available tools (both modular and legacy)."""
        try:
            # Get modular tools
            modular_registry = create_default_registry()
            modular_tools = modular_registry.list_tools()
            logger.info(f"Loading {len(modular_tools)} modular tools")
            for tool in modular_tools:
                self.tools[tool.name] = tool
                logger.debug(f"Registered modular tool: {tool.name}")
        except Exception as e:
            logger.error(f"Failed to load modular tools: {e}")
            import traceback
            traceback.print_exc()
        
        # All tools are now modular - no legacy import needed

    def register_tool(self, tool: Tool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all tools with their details."""
        out = []
        for t in self.tools.values():
            out.append({
                "name": t.name,
                "description": t.description,
                "enabled": t.enabled,
                "available": t.is_available(),
                "parameters": t.to_openai_function().get("parameters", {}).get("properties", {})
            })
        return out

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def to_openai_tools_format(self) -> List[Dict[str, Any]]:
        """Convert tools to OpenAI function format."""
        payload = []
        for t in self.tools.values():
            if not t.enabled or not t.is_available():
                continue
            payload.append({"type": "function", "function": t.to_openai_function()})
        return payload

    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool with backward compatibility for legacy parameter handling."""
        tool = self.get_tool(tool_name)
        if not tool:
            return {"success": False, "error": f"Tool not found: {tool_name}"}
        if not tool.enabled:
            return {"success": False, "error": f"Tool is disabled: {tool_name}"}
        if not tool.is_available():
            return {"success": False, "error": f"Tool is not available: {tool_name}"}
        
        try:
            # Handle legacy tools with specific parameter handling
            if tool_name == "file_operations":
                logger.debug(f"file_operations called with kwargs: {kwargs}")
                if "operation" in kwargs:
                    operation = kwargs.pop("operation")
                    return await tool.execute(operation, **kwargs)
                else:
                    return {"success": False, "error": "Missing required 'operation' parameter"}
            elif tool_name in ["web_search", "reddit_search"]:
                if "query" in kwargs:
                    query = kwargs.pop("query")
                    return await tool.execute(query, **kwargs)
                else:
                    return {"success": False, "error": "Missing required 'query' parameter"}
            elif tool_name == "image_generation":
                if "prompt" in kwargs:
                    prompt = kwargs.pop("prompt")
                    content_type = kwargs.pop("content_type", "auto")
                    return await tool.execute(prompt, content_type=content_type, **kwargs)
                else:
                    return {"success": False, "error": "Missing required 'prompt' parameter"}
            elif tool_name == "image_analysis":
                if "image_path" in kwargs:
                    image_path = kwargs.pop("image_path")
                    return await tool.execute(image_path, **kwargs)
                else:
                    return {"success": False, "error": "Missing required 'image_path' parameter"}
            elif tool_name == "ghidra_analysis":
                if "binary_path" in kwargs:
                    binary_path = kwargs.pop("binary_path")
                    return await tool.execute(binary_path, **kwargs)
                else:
                    return {"success": False, "error": "Missing required 'binary_path' parameter"}
            else:
                # For new modular tools, use standard **kwargs execution
                return await tool.execute(**kwargs)
                
        except Exception as e:
            logger.exception("Tool execution failed: %s", tool_name)
            error_msg = str(e)
            
            # Provide actionable error messages for autonomous coding workflows
            if "Unsupported language" in error_msg:
                return {"success": False, "error": f"Language not supported by {tool_name}. Supported languages may include: python, bash. Check tool documentation.", "tool": tool_name, "actionable": True}
            elif "not found" in error_msg.lower():
                return {"success": False, "error": f"Resource not found for {tool_name}: {error_msg}. Verify file paths and dependencies.", "tool": tool_name, "actionable": True}
            elif "permission" in error_msg.lower() or "access" in error_msg.lower():
                return {"success": False, "error": f"Access denied for {tool_name}: {error_msg}. Check file permissions.", "tool": tool_name, "actionable": True}
            else:
                return {"success": False, "error": f"Tool execution failed: {error_msg}", "tool": tool_name, "actionable": False}


__all__ = [
    "Tool",
    "ToolRegistry",  # Enhanced ToolRegistry with all modular tools
    # Core tools
    "FileOperationsTool",
    "CodebaseAnalysisTool", 
    "CodeExecutionTool",
    # Web tools
    "WebSearchTool",
    "RedditSearchTool",
    # Project tools
    "ProjectPlanningTool",
    # Media tools
    "ImageGenerationTool",
    "ImageAnalysisTool",
    # Reverse engineering tools
    "DisassemblerTool",
    "HexEditorTool", 
    "PatternSearchTool",
    "DebuggingTool",
    "GhidraAnalysisTool",
    # Audio tools
    "WaveformGeneratorTool",
    "SynthesizerTool",
    "AudioAnalysisTool", 
    "MIDITool",
    # Registry functions
    "create_default_registry",
    "load_toolset",
    "load_all_tools",
    "load_available_tools"
]