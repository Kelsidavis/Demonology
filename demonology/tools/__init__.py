from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from .base import Tool, ToolRegistry

logger = logging.getLogger(__name__)

# ---------------- Utilities ----------------

def _try_register(reg: ToolRegistry, cls: type, report: List[Dict[str, Any]], *, name_hint: Optional[str] = None):
    tool_name = name_hint or getattr(cls, "__name__", str(cls))
    try:
        tool: Tool = cls()  # type: ignore[call-arg]
        if hasattr(tool, "is_available"):
            avail = bool(tool.is_available())
        else:
            avail = True
        if avail:
            reg.register(tool)
            report.append({"tool": tool.name, "status": "registered"})
        else:
            report.append({"tool": tool_name, "status": "skipped", "reason": "is_available() == False"})
    except Exception as e:
        report.append({"tool": tool_name, "status": "error", "reason": str(e)})
        logger.debug("Failed to register %s: %s", tool_name, e, exc_info=True)

def _optional_import(module: str, names: List[str]) -> Tuple[Optional[object], Dict[str, Any]]:
    try:
        mod = __import__(f"{__name__}.{module}", fromlist=names)
        return mod, {"module": module, "status": "ok"}
    except Exception as e:
        return None, {"module": module, "status": "missing", "reason": str(e)}

# ---------------- Public helpers ----------------

def remind_agent_capabilities() -> str:
    return (
        "AI Agent Tooling (workspace-scoped):\n"
        "- File ops, code analysis/execution (sandboxed)\n"
        "- Web search: DuckDuckGo IA, Reddit, Wikipedia, HackerNews, StackOverflow\n"
        "- Media: image generation/analysis\n"
        "- Audio: synthesis, analysis, described SFX (key-aware)\n"
        "- Music: sheet OMR → MusicXML/MIDI/WAV (Audiveris/music21)\n"
        "- 3D: model generator/export (OBJ/STL/GLB/PLY via trimesh)\n"
        "- Heightmaps: PNG/JPG → 3D terrain, procedural generation (game engines)\n"
        "- Reverse engineering: objdump/r2/gdb/ghidra (if available)\n"
        "Use ToolRegistry.call(name, **kwargs)."
    )

def create_default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    report: List[Dict[str, Any]] = []

    # Core modules
    mod, _ = _optional_import("file_ops", ["FileOperationsTool"])
    if mod and hasattr(mod, "FileOperationsTool"): 
        _try_register(reg, getattr(mod, "FileOperationsTool"), report)

    mod, _ = _optional_import("codebase", ["CodebaseAnalysisTool"])
    if mod and hasattr(mod, "CodebaseAnalysisTool"): 
        _try_register(reg, getattr(mod, "CodebaseAnalysisTool"), report)

    mod, _ = _optional_import("execution", ["CodeExecutionTool"])
    if mod and hasattr(mod, "CodeExecutionTool"): 
        _try_register(reg, getattr(mod, "CodeExecutionTool"), report)

    # Web tools
    mod, _ = _optional_import("web", ["WebSearchTool", "RedditSearchTool"])
    if mod:
        if hasattr(mod, "WebSearchTool"): 
            _try_register(reg, getattr(mod, "WebSearchTool"), report)
        if hasattr(mod, "RedditSearchTool"): 
            _try_register(reg, getattr(mod, "RedditSearchTool"), report)

    # Free extras: Wikipedia, HackerNews, StackOverflow, OpenWebSearch
    mod, _ = _optional_import("web_free_extras", [
        "WikipediaSearchTool", "HackerNewsSearchTool", "StackOverflowSearchTool", "OpenWebSearchTool"
    ])
    if mod:
        for cls_name in ["WikipediaSearchTool", "HackerNewsSearchTool", "StackOverflowSearchTool", "OpenWebSearchTool"]:
            if hasattr(mod, cls_name):
                _try_register(reg, getattr(mod, cls_name), report)

    # Project planning
    mod, _ = _optional_import("project", ["ProjectPlanningTool"])
    if mod and hasattr(mod, "ProjectPlanningTool"): 
        _try_register(reg, getattr(mod, "ProjectPlanningTool"), report)

    # Media
    mod, _ = _optional_import("media", ["ImageGenerationTool", "ImageAnalysisTool"])
    if mod:
        if hasattr(mod, "ImageGenerationTool"): 
            _try_register(reg, getattr(mod, "ImageGenerationTool"), report)
        if hasattr(mod, "ImageAnalysisTool"): 
            _try_register(reg, getattr(mod, "ImageAnalysisTool"), report)

    # Reverse engineering
    mod, _ = _optional_import("reverse_eng", [
        "DisassemblerTool", "HexEditorTool", "PatternSearchTool", "DebuggingTool", "GhidraAnalysisTool"
    ])
    if mod:
        for cls_name in ["DisassemblerTool", "HexEditorTool", "PatternSearchTool", "DebuggingTool", "GhidraAnalysisTool"]:
            if hasattr(mod, cls_name):
                _try_register(reg, getattr(mod, cls_name), report)

    # Audio
    mod, _ = _optional_import("audio", [
        "WaveformGeneratorTool", "SynthesizerTool", "AudioAnalysisTool", "DescribedSFXTool"
    ])
    if mod:
        for cls_name in ["WaveformGeneratorTool", "SynthesizerTool", "AudioAnalysisTool", "DescribedSFXTool"]:
            if hasattr(mod, cls_name):
                _try_register(reg, getattr(mod, cls_name), report)

    # Music / Sheet OMR
    mod, _ = _optional_import("sheet_music_omr", ["SheetMusicOMRTool"])
    if mod and hasattr(mod, "SheetMusicOMRTool"):
        _try_register(reg, getattr(mod, "SheetMusicOMRTool"), report)

    # 3D Model Generator
    mod, _ = _optional_import("model3d_generator", ["Model3DGeneratorTool"])
    if mod and hasattr(mod, "Model3DGeneratorTool"):
        _try_register(reg, getattr(mod, "Model3DGeneratorTool"), report)

    # Heightmap Generator
    mod, _ = _optional_import("heightmap_generator", ["HeightmapGeneratorTool"])
    if mod and hasattr(mod, "HeightmapGeneratorTool"):
        _try_register(reg, getattr(mod, "HeightmapGeneratorTool"), report)

    reg._load_report = report  # type: ignore[attr-defined]
    return reg

def load_report(reg: Optional[ToolRegistry] = None) -> List[Dict[str, Any]]:
    reg = reg or create_default_registry()
    return getattr(reg, "_load_report", [])  # type: ignore[no-any-return]

def load_all_tools() -> List[str]:
    reg = create_default_registry()
    return reg.list()

def load_available_tools() -> List[str]:
    reg = create_default_registry()
    return reg.list()

def to_openai_tools_format(reg: Optional[ToolRegistry] = None) -> List[Dict[str, Any]]:
    reg = reg or create_default_registry()
    payload: List[Dict[str, Any]] = []
    for name in reg.list():
        t = reg.get(name)
        if not t:
            continue
        try:
            payload.append({"type": "function", "function": t.to_openai_function()})
        except Exception as e:
            logger.debug("Schema export failed for %s: %s", name, e, exc_info=True)
    return payload

__all__ = [
    "Tool", "ToolRegistry",
    "create_default_registry", "load_all_tools", "load_available_tools",
    "to_openai_tools_format", "load_report", "remind_agent_capabilities",
]
