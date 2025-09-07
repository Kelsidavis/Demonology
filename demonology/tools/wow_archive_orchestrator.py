
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .base import Tool  # type: ignore
except Exception:
    class Tool:
        def __init__(self, name: str, description: str):
            self.name, self.description = name, description

logger = logging.getLogger(__name__)

# Import sibling tools (runtime-optional; we check attributes)
try:
    from .wow_world_converter import WoWWorldConverterTool  # type: ignore
except Exception:
    WoWWorldConverterTool = None  # type: ignore
try:
    from .wow_model_converter import WoWModelConverterTool  # type: ignore
except Exception:
    WoWModelConverterTool = None  # type: ignore
try:
    from .wow_scene_bundler import WoWSceneBundlerTool  # type: ignore
except Exception:
    WoWSceneBundlerTool = None  # type: ignore
try:
    from .mpq_extractor import MPQExtractorTool  # type: ignore
except Exception:
    MPQExtractorTool = None  # type: ignore
try:
    from .wow_to_unreal import WoWToUnrealTool  # type: ignore
except Exception:
    WoWToUnrealTool = None  # type: ignore

class WoWArchiveOrchestratorTool(Tool):
    """
    One-shot pipeline:
      1) Extract MPQ archives to a workspace folder
      2) Convert terrain (WDT/ADT) to meshes/heightmaps with bundle manifest
      3) Convert models (M2/M3/WMO) to GLB with bundle manifest
      4) Build a unified scene_manifest.json

    Requires: wow_world_converter, wow_model_converter, wow_scene_bundler, mpq_extractor
    """

    def __init__(self):
        super().__init__("wow_archive_orchestrator", "Extract MPQs and build a complete scene bundle.")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "mpq_inputs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Paths to .mpq files or directories containing them. If empty, auto-discovers MPQ files.",
                        "default": []
                    },
                    "workspace": {
                        "type": "string",
                        "description": "Working directory for extraction and outputs",
                        "default": "wow_workspace"
                    },
                    "map_name": {
                        "type": "string",
                        "description": "Map to export terrain for (e.g., 'Azeroth'). If empty, auto-discovers available maps.",
                        "default": ""
                    },
                    "downsample": {
                        "type": "integer",
                        "default": 2,
                        "description": "Heightmap downsample factor (1=full)"
                    },
                    "export_models": {
                        "type": "boolean",
                        "default": True
                    },
                    "export_terrain": {
                        "type": "boolean",
                        "default": True
                    },
                    "build_scene": {
                        "type": "boolean",
                        "default": True
                    },
                    "create_unreal_project": {
                        "type": "boolean",
                        "description": "Create a complete Unreal Engine project with all converted assets",
                        "default": False
                    },
                    "unreal_project_name": {
                        "type": "string",
                        "description": "Name for the Unreal Engine project",
                        "default": "WoWUnrealProject"
                    }
                },
                "required": []
            }
        }

    def is_available(self) -> bool:
        return all([
            MPQExtractorTool is not None,
            (WoWWorldConverterTool is not None) or (WoWModelConverterTool is not None) or (WoWSceneBundlerTool is not None)
        ])

    async def execute(self,
                      mpq_inputs: List[str] = None,
                      map_name: str = "",
                      workspace: str = "wow_workspace",
                      downsample: int = 2,
                      export_models: bool = True,
                      export_terrain: bool = True,
                      build_scene: bool = True,
                      create_unreal_project: bool = False,
                      unreal_project_name: str = "WoWUnrealProject",
                      **_) -> Dict[str, Any]:
        
        # Handle auto-discovery
        if mpq_inputs is None:
            mpq_inputs = []
        ws = Path(workspace).expanduser().resolve()
        extract_root = ws / "extracted"
        exports_root = ws / "exports"
        bundles_root = ws / "bundles"
        for p in (extract_root, exports_root, bundles_root):
            p.mkdir(parents=True, exist_ok=True)

        # 1) Extract MPQs
        if MPQExtractorTool is None:
            return {"success": False, "error": "MPQExtractorTool not available."}
        mpq = MPQExtractorTool()
        res1 = await mpq.execute(mpq_paths=mpq_inputs, operation="extract_all", dest_dir=str(extract_root))
        if not res1.get("success"):
            return {"success": False, "stage": "extract", "details": res1}

        # Auto-discover map name if not provided
        if not map_name:
            maps_dir = extract_root / "World" / "Maps"
            if maps_dir.exists():
                available_maps = [d.name for d in maps_dir.iterdir() if d.is_dir()]
                if available_maps:
                    # Prefer common maps or take the first one
                    priority_maps = ["Azeroth", "Kalimdor", "Outland", "Northrend", "Pandaria", "Draenor", "BrokenIsles", "Zandalar", "KulTiras"]
                    map_name = next((m for m in priority_maps if m in available_maps), available_maps[0])
                    logger.info(f"Auto-discovered map: {map_name} from {available_maps}")
                else:
                    logger.warning("No maps found in extracted data")
                    map_name = "UnknownMap"
            else:
                logger.warning("No World/Maps directory found in extracted data")
                map_name = "UnknownMap"

        terrain_manifest = None
        models_manifest = None

        # 2) Terrain
        if export_terrain and WoWWorldConverterTool is not None:
            world = WoWWorldConverterTool()
            res2 = await world.execute(
                maps_root=str(extract_root),
                map_name=map_name,
                tiles=None,
                merge_tiles=False,
                export=["glb", "height_png"],
                output_dir=str(exports_root / map_name),
                downsample=downsample,
                bundle_dir=str(bundles_root)
            )
            if not res2.get("success"):
                return {"success": False, "stage": "terrain", "details": res2}
            terrain_manifest = res2.get("bundle_manifest")

        # 3) Models
        if export_models and WoWModelConverterTool is not None:
            models = WoWModelConverterTool()
            res3 = await models.execute(
                input_path=str(extract_root),
                output_path=str(exports_root / "models"),
                kind="auto",
                lod=0,
                bundle_dir=str(bundles_root)
            )
            if not res3.get("success"):
                # not fatal; continue without models
                logger.warning("Models export failed: %s", res3)
            else:
                # The batched export returns output_dir + manifest path
                models_manifest = res3.get("bundle_manifest")

        # 4) Scene bundling
        scene_manifest_path = None
        if build_scene and WoWSceneBundlerTool is not None and terrain_manifest:
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

        # 5) Unreal Engine Project Creation
        unreal_project_path = None
        if create_unreal_project and WoWToUnrealTool is not None:
            unreal_tool = WoWToUnrealTool()
            unreal_res = await unreal_tool.execute(
                operation="create_project",
                wow_data_path=str(extract_root),
                unreal_project_path=str(ws / unreal_project_name),
                asset_types=["all"] if export_models else ["terrain"],
                conversion_settings={
                    "model_format": "fbx",
                    "texture_format": "png",
                    "create_materials": True,
                    "optimize_meshes": True
                },
                unreal_settings={
                    "project_name": unreal_project_name,
                    "generate_blueprints": True,
                    "create_game_mode": True
                }
            )
            if unreal_res.get("success"):
                unreal_project_path = unreal_res.get("project_path")
                logger.info(f"Created Unreal Engine project at: {unreal_project_path}")
            else:
                logger.warning(f"Unreal Engine project creation failed: {unreal_res.get('error')}")

        return {
            "success": True,
            "workspace": str(ws),
            "extract_root": str(extract_root),
            "exports_root": str(exports_root),
            "bundles_root": str(bundles_root),
            "terrain_manifest": terrain_manifest,
            "models_manifest": models_manifest,  # may be None if model export failed
            "scene_manifest": scene_manifest_path,
            "unreal_project": unreal_project_path,
            "discovered_map": map_name
        }
