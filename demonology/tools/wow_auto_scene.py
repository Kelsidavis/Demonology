
from __future__ import annotations

import logging, os, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from .base import Tool
except Exception:
    class Tool:
        def __init__(self, name: str, description: str):
            self.name, self.description = name, description

# Optional sibling tools
try:
    from .mpq_extractor import MPQExtractorTool
except Exception:
    MPQExtractorTool = None  # type: ignore

try:
    from .wow_world_converter import WoWWorldConverterTool
except Exception:
    WoWWorldConverterTool = None  # type: ignore

try:
    from .wow_model_converter import WoWModelConverterTool
except Exception:
    WoWModelConverterTool = None  # type: ignore

try:
    from .wow_scene_bundler import WoWSceneBundlerTool
except Exception:
    WoWSceneBundlerTool = None  # type: ignore

logger = logging.getLogger(__name__)

MAP_PREF_ORDER = ["Azeroth","Kalimdor","EasternKingdoms","Northrend","Outland","Draenor","Pandaria","BrokenIsles","Zandalar","KulTiras"]

def _is_mpq_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() == ".mpq"

def _find_files_case_insensitive(root: Path, pattern: str) -> List[Path]:
    """Return files whose name matches pattern case-insensitively (glob-like only for filename)."""
    target = pattern.lower()
    out = []
    for f in root.rglob("*"):
        if f.is_file() and f.name.lower() == target:
            out.append(f)
    return out

def _discover_map_names(extracted_root: Path) -> List[str]:
    maps_dir = extracted_root / "World" / "Maps"
    if not maps_dir.is_dir():
        return []
    return sorted([d.name for d in maps_dir.iterdir() if d.is_dir()])

def _choose_map_name(candidates: List[str]) -> Optional[str]:
    if not candidates:
        return None
    # prefer common names
    for pref in MAP_PREF_ORDER:
        if pref in candidates:
            return pref
    return candidates[0]

class WoWAutoSceneTool(Tool):
    """
    High-level convenience: accept a filename like 'terrain.MPQ' or a directory.
    - If MPQ(s): extract to ./extracted
    - Discover map_name from extracted data
    - Export terrain + (optionally) models
    - Bundle into scene_manifest.json
    All paths are kept inside the current workspace ('.') to satisfy sandbox guards.
    """
    def __init__(self):
        super().__init__("wow_auto_scene", "One-call auto pipeline for WoW assets (MPQ or extracted).")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Path to an MPQ file/dir or an already extracted root (contains World/Maps)"},
                    "workspace": {"type": "string", "description": "Workspace root (must be inside allowed root)", "default": "."},
                    "map_name": {"type": "string", "description": "Override autodetected map name (e.g., 'Azeroth')"},
                    "downsample": {"type": "integer", "default": 2},
                    "export_models": {"type": "boolean", "default": True},
                    "export_terrain": {"type": "boolean", "default": True}
                },
                "required": ["input"]
            }
        }

    def is_available(self) -> bool:
        return True

    async def execute(self,
                      input: str,
                      workspace: str = ".",
                      map_name: Optional[str] = None,
                      downsample: int = 2,
                      export_models: bool = True,
                      export_terrain: bool = True,
                      **_) -> Dict[str, Any]:
        ws = Path(workspace).resolve()
        ws.mkdir(parents=True, exist_ok=True)

        inp = Path(input)
        if not inp.is_absolute():
            # try workspace-relative first
            cand = (ws / input).resolve()
            if cand.exists():
                inp = cand
            else:
                # case-insensitive search by filename inside workspace
                matches = _find_files_case_insensitive(ws, inp.name)
                if matches:
                    inp = matches[0]

        # Directories where we will write outputs
        extract_root = ws / "extracted"
        exports_root = ws / "exports"
        bundles_root = ws / "bundles"
        for p in (extract_root, exports_root, bundles_root):
            p.mkdir(parents=True, exist_ok=True)

        # Stage A: Ensure we have an extracted root
        extracted_ready = False
        notes: List[str] = []

        if inp.exists():
            if _is_mpq_file(inp) or (inp.is_dir() and any(f.suffix.lower()==".mpq" for f in inp.glob("*.mpq"))):
                if MPQExtractorTool is None:
                    return {"success": False, "error": "MPQExtractorTool not available (install backend or register tool)."}
                mpq = MPQExtractorTool()
                # Accept file or directory
                mpq_inputs = [str(inp)]
                res = await mpq.execute(mpq_paths=mpq_inputs, operation="extract_all", dest_dir=str(extract_root))
                if not res.get("success"):
                    return {"success": False, "stage": "mpq_extract", "details": res}
                extracted_ready = True
                notes.append("Extracted MPQ(s) to ./extracted")
            elif inp.is_dir() and (inp / "World" / "Maps").exists():
                # Already-extracted tree
                extract_root = inp
                extracted_ready = True
                notes.append("Using provided extracted root.")
            else:
                # If it's a filename in workspace that doesn't exist, search recursively
                if inp.name and not inp.exists():
                    matches = _find_files_case_insensitive(ws, inp.name)
                    if matches:
                        inp = matches[0]
                        if _is_mpq_file(inp):
                            if MPQExtractorTool is None:
                                return {"success": False, "error": "MPQExtractorTool not available."}
                            res = await MPQExtractorTool().execute(mpq_paths=[str(inp)], operation="extract_all", dest_dir=str(extract_root))
                            if not res.get("success"):
                                return {"success": False, "stage": "mpq_extract", "details": res}
                            extracted_ready = True
                            notes.append(f"Found case-insensitive file match: {inp.name}")
        else:
            return {"success": False, "error": f"Input not found in workspace: {input}"}

        if not extracted_ready:
            return {"success": False, "error": "Could not prepare extracted assets (no MPQ match and no World/Maps found)."}

        # Stage B: Choose map_name
        maps = _discover_map_names(extract_root)
        chosen = map_name or _choose_map_name(maps)
        if not chosen:
            return {"success": False, "stage": "map_discovery", "error": "No map directories found under World/Maps.", "maps_root": str(extract_root / 'World' / 'Maps')}
        if chosen not in maps:
            notes.append(f"Map '{chosen}' not found; using first available: '{maps[0]}'")
            chosen = maps[0]

        # Stage C: Terrain export
        terrain_manifest = None
        if export_terrain:
            if WoWWorldConverterTool is None:
                return {"success": False, "error": "WoWWorldConverterTool not available."}
            world = WoWWorldConverterTool()
            res2 = await world.execute(
                maps_root=str(extract_root),
                map_name=chosen,
                tiles=None,
                merge_tiles=False,
                export=["glb","height_png"],
                output_dir=str(exports_root / chosen),
                downsample=downsample,
                bundle_dir=str(bundles_root)
            )
            if not res2.get("success"):
                return {"success": False, "stage": "terrain", "details": res2}
            terrain_manifest = res2.get("bundle_manifest")

        # Stage D: Model export
        models_manifest = None
        if export_models and WoWModelConverterTool is not None:
            models = WoWModelConverterTool()
            res3 = await models.execute(
                input_path=str(extract_root),
                output_path=str(exports_root / "models"),
                kind="auto",
                lod=0,
                bundle_dir=str(bundles_root)
            )
            if res3.get("success"):
                models_manifest = res3.get("bundle_manifest")

        # Stage E: Scene bundling
        scene_manifest_path = None
        if WoWSceneBundlerTool is not None and terrain_manifest:
            bundler = WoWSceneBundlerTool()
            res4 = await bundler.execute(
                terrain_manifest=terrain_manifest,
                models_manifest=models_manifest,
                maps_root=str(extract_root),
                include_placements=True,
                output_path=str(ws / "scene_manifest.json")
            )
            if not res4.get("success"):
                return {"success": False, "stage": "scene", "details": res4}
            scene_manifest_path = res4.get("output")

        return {
            "success": True,
            "notes": notes,
            "workspace": str(ws),
            "extract_root": str(extract_root),
            "exports_root": str(exports_root),
            "bundles_root": str(bundles_root),
            "map_name": chosen,
            "terrain_manifest": terrain_manifest,
            "models_manifest": models_manifest,
            "scene_manifest": scene_manifest_path
        }
