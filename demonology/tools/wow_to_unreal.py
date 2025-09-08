# demonology/tools/wow_to_unreal.py
from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import Tool, _confine

class WoWToUnrealTool(Tool):
    """
    Complete World of Warcraft to Unreal Engine asset pipeline.
    Orchestrates the conversion of WoW assets (M2, BLP, BLS, DBC) into Unreal Engine projects.
    """

    def __init__(self):
        super().__init__("wow_to_unreal", "Complete WoW to Unreal Engine asset conversion pipeline")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["convert_assets", "create_project", "import_world", "batch_convert", "analyze_assets"],
                        "description": "Pipeline operation to perform"
                    },
                    "wow_data_path": {
                        "type": "string",
                        "description": "Path to WoW game data directory or extracted MPQ files"
                    },
                    "unreal_project_path": {
                        "type": "string",
                        "description": "Path to target Unreal Engine project",
                        "default": "./WoWUnrealProject"
                    },
                    "asset_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["models", "textures", "materials", "data", "all"]
                        },
                        "description": "Types of assets to convert",
                        "default": ["all"]
                    },
                    "world_filter": {
                        "type": "object",
                        "description": "Filter for specific WoW content",
                        "properties": {
                            "zones": {"type": "array", "items": {"type": "string"}, "description": "Specific zone names"},
                            "content_types": {"type": "array", "items": {"type": "string"}, "description": "Content types (creatures, items, spells)"},
                            "level_range": {"type": "object", "properties": {"min": {"type": "integer"}, "max": {"type": "integer"}}},
                            "include_instances": {"type": "boolean", "default": False},
                            "include_battlegrounds": {"type": "boolean", "default": False}
                        }
                    },
                    "conversion_settings": {
                        "type": "object",
                        "description": "Asset conversion settings",
                        "properties": {
                            "model_format": {"type": "string", "enum": ["fbx", "obj"], "default": "fbx"},
                            "texture_format": {"type": "string", "enum": ["png", "tga", "dds"], "default": "png"},
                            "scale_factor": {"type": "number", "default": 1.0},
                            "optimize_meshes": {"type": "boolean", "default": True},
                            "generate_lods": {"type": "boolean", "default": True},
                            "compress_textures": {"type": "boolean", "default": True},
                            "create_materials": {"type": "boolean", "default": True}
                        }
                    },
                    "unreal_settings": {
                        "type": "object", 
                        "description": "Unreal Engine specific settings",
                        "properties": {
                            "project_name": {"type": "string", "default": "WoWProject"},
                            "engine_version": {"type": "string", "default": "5.4"},
                            "template": {"type": "string", "default": "third_person"},
                            "generate_blueprints": {"type": "boolean", "default": True},
                            "create_game_mode": {"type": "boolean", "default": True},
                            "setup_lighting": {"type": "boolean", "default": True}
                        }
                    }
                },
                "required": ["operation", "wow_data_path"]
            }
        }

    async def _analyze_wow_assets(self, wow_data_path: Path) -> Dict[str, Any]:
        """Analyze available WoW assets in the data path."""
        analysis = {
            "models": [],
            "textures": [],
            "shaders": [],
            "database_files": [],
            "total_size": 0,
            "directory_structure": {}
        }
        
        if not wow_data_path.exists():
            return {"error": f"WoW data path not found: {wow_data_path}"}
        
        try:
            # Scan for M2 models
            m2_files = list(wow_data_path.rglob("*.m2"))
            analysis["models"] = [{"file": str(f), "size": f.stat().st_size} for f in m2_files[:100]]  # Limit for performance
            
            # Scan for BLP textures
            blp_files = list(wow_data_path.rglob("*.blp"))
            analysis["textures"] = [{"file": str(f), "size": f.stat().st_size} for f in blp_files[:200]]
            
            # Scan for BLS shaders
            bls_files = list(wow_data_path.rglob("*.bls"))
            analysis["shaders"] = [{"file": str(f), "size": f.stat().st_size} for f in bls_files[:50]]
            
            # Scan for DBC files
            dbc_files = list(wow_data_path.rglob("*.dbc"))
            analysis["database_files"] = [{"file": str(f), "size": f.stat().st_size} for f in dbc_files[:50]]
            
            # Calculate total size
            analysis["total_size"] = (
                len(analysis["models"]) * 1000000 +  # Estimate
                len(analysis["textures"]) * 500000 +
                len(analysis["shaders"]) * 10000 +
                len(analysis["database_files"]) * 50000
            )
            
            # Directory structure
            directories = {}
            for item in wow_data_path.rglob("*"):
                if item.is_dir():
                    directories[str(item.relative_to(wow_data_path))] = len(list(item.iterdir()))
            analysis["directory_structure"] = directories
            
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis

    async def _setup_unreal_project(self, project_path: Path, unreal_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Set up the Unreal Engine project structure."""
        try:
            from .unreal_project import UnrealProjectTool
            
            project_tool = UnrealProjectTool()
            if not project_tool.is_available():
                return {"success": False, "error": "Unreal Engine not found"}
            
            # Create project
            result = await project_tool.execute(
                operation="create",
                project_name=unreal_settings.get("project_name", "WoWProject"),
                project_path=str(project_path.parent),
                template=unreal_settings.get("template", "third_person"),
                engine_version=unreal_settings.get("engine_version", "5.4")
            )
            
            if not result.get("success", False):
                return result
            
            # Create WoW-specific directory structure
            content_dir = project_path / "Content"
            wow_dirs = [
                "WoW/Models",
                "WoW/Textures", 
                "WoW/Materials",
                "WoW/Data",
                "WoW/Blueprints",
                "WoW/Maps",
                "WoW/Audio",
                "WoW/UI"
            ]
            
            for dir_path in wow_dirs:
                (content_dir / dir_path).mkdir(parents=True, exist_ok=True)
            
            # Generate project configuration
            config = {
                "WoWConversionSettings": {
                    "SourceDataPath": "",
                    "ConversionDate": "",
                    "AssetCounts": {
                        "Models": 0,
                        "Textures": 0,
                        "Materials": 0,
                        "DataTables": 0
                    },
                    "ConversionSettings": unreal_settings
                }
            }
            
            config_file = project_path / "Config" / "WoWConversion.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            return {
                "success": True,
                "project_path": str(project_path),
                "content_directories": wow_dirs,
                "config_file": str(config_file)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _convert_models(self, wow_data_path: Path, output_dir: Path, 
                            conversion_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Convert WoW M2 models to Unreal format."""
        try:
            from .wow_m2_converter import WoWM2ModelTool
            
            m2_tool = WoWM2ModelTool()
            
            # Find M2 files
            m2_files = list(wow_data_path.rglob("*.m2"))
            if not m2_files:
                return {"success": True, "converted": 0, "message": "No M2 files found"}
            
            # Batch convert models
            result = await m2_tool.execute(
                operation="batch_convert",
                input_path=str(wow_data_path),
                output_path=str(output_dir),
                output_format=conversion_settings.get("model_format", "fbx"),
                scale_factor=conversion_settings.get("scale_factor", 1.0),
                optimize_mesh=conversion_settings.get("optimize_meshes", True),
                generate_lod=conversion_settings.get("generate_lods", True)
            )
            
            return result
            
        except ImportError:
            return {"success": False, "error": "WoWM2ModelTool not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _convert_textures(self, wow_data_path: Path, output_dir: Path,
                              conversion_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Convert WoW BLP textures."""
        try:
            from .wow_blp_texture import WoWBLPTextureTool
            
            blp_tool = WoWBLPTextureTool()
            
            # Find BLP files
            blp_files = list(wow_data_path.rglob("*.blp"))
            if not blp_files:
                return {"success": True, "converted": 0, "message": "No BLP files found"}
            
            # Batch convert textures
            result = await blp_tool.execute(
                operation="batch_convert",
                input_path=str(wow_data_path),
                output_path=str(output_dir),
                output_format=conversion_settings.get("texture_format", "png"),
                unreal_settings={
                    "generate_unreal_asset": True,
                    "srgb": True,
                    "generate_mipmaps": True,
                    "compression": "TC_Default" if conversion_settings.get("compress_textures", True) else "TC_Displacementmap"
                }
            )
            
            return result
            
        except ImportError:
            return {"success": False, "error": "WoWBLPTextureTool not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _convert_materials(self, wow_data_path: Path, output_dir: Path,
                               conversion_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Convert WoW BLS shaders to Unreal materials."""
        try:
            from .wow_bls_shader import WoWBLSShaderTool
            
            bls_tool = WoWBLSShaderTool()
            
            # Find BLS files
            bls_files = list(wow_data_path.rglob("*.bls"))
            if not bls_files:
                return {"success": True, "converted": 0, "message": "No BLS files found"}
            
            # Batch convert materials
            result = await bls_tool.execute(
                operation="batch_convert",
                input_path=str(wow_data_path),
                output_path=str(output_dir),
                material_type="standard",
                shader_features={
                    "enable_transparency": True,
                    "enable_normal_mapping": True,
                    "enable_specular": True
                }
            )
            
            return result
            
        except ImportError:
            return {"success": False, "error": "WoWBLSShaderTool not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _convert_database_files(self, wow_data_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Convert WoW DBC database files."""
        try:
            from .wow_dbc_parser import WoWDBCParserTool
            
            dbc_tool = WoWDBCParserTool()
            
            # Find DBC files
            dbc_files = list(wow_data_path.rglob("*.dbc"))
            if not dbc_files:
                return {"success": True, "converted": 0, "message": "No DBC files found"}
            
            # Batch parse DBC files
            result = await dbc_tool.execute(
                operation="batch_parse",
                input_path=str(wow_data_path),
                output_path=str(output_dir),
                output_format="json",
                include_strings=True
            )
            
            return result
            
        except ImportError:
            return {"success": False, "error": "WoWDBCParserTool not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _create_integration_blueprints(self, project_path: Path, 
                                           asset_counts: Dict[str, int]) -> Dict[str, Any]:
        """Create Unreal Blueprints for WoW asset integration."""
        try:
            from .unreal_blueprint import UnrealBlueprintTool
            
            blueprint_tool = UnrealBlueprintTool()
            blueprints_created = []
            
            # Create WoW Asset Manager Blueprint
            asset_manager_result = await blueprint_tool.execute(
                operation="create_actor",
                project_path=str(project_path),
                blueprint_name="BP_WoWAssetManager",
                blueprint_type="Actor",
                parent_class="Actor",
                blueprint_path="WoW/Blueprints",
                variables=[
                    {"name": "ModelCount", "type": "int", "default_value": str(asset_counts.get("models", 0)), "is_public": True},
                    {"name": "TextureCount", "type": "int", "default_value": str(asset_counts.get("textures", 0)), "is_public": True},
                    {"name": "MaterialCount", "type": "int", "default_value": str(asset_counts.get("materials", 0)), "is_public": True}
                ]
            )
            
            if asset_manager_result.get("success"):
                blueprints_created.append("BP_WoWAssetManager")
            
            # Create WoW Game Mode if requested
            game_mode_result = await blueprint_tool.execute(
                operation="create_game_mode",
                project_path=str(project_path),
                blueprint_name="BP_WoWGameMode",
                blueprint_path="WoW/Blueprints"
            )
            
            if game_mode_result.get("success"):
                blueprints_created.append("BP_WoWGameMode")
            
            return {
                "success": True,
                "blueprints_created": blueprints_created,
                "asset_manager": asset_manager_result,
                "game_mode": game_mode_result
            }
            
        except ImportError:
            return {"success": False, "error": "UnrealBlueprintTool not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def execute(self, operation: str, wow_data_path: str, 
                     unreal_project_path: str = "./WoWUnrealProject",
                     asset_types: List[str] = None, world_filter: Optional[Dict] = None,
                     conversion_settings: Optional[Dict[str, Any]] = None,
                     unreal_settings: Optional[Dict[str, Any]] = None, **_) -> Dict[str, Any]:
        
        asset_types = asset_types or ["all"]
        conversion_settings = conversion_settings or {}
        unreal_settings = unreal_settings or {}
        
        try:
            wow_path = _confine(Path(wow_data_path))
            project_path = _confine(Path(unreal_project_path))
            
            if not wow_path.exists():
                return {"success": False, "error": f"WoW data path not found: {wow_data_path}"}
            
            if operation == "analyze_assets":
                analysis = await self._analyze_wow_assets(wow_path)
                return {"success": True, "analysis": analysis}
            
            elif operation == "create_project":
                result = await self._setup_unreal_project(project_path, unreal_settings)
                return result
            
            elif operation == "convert_assets" or operation == "batch_convert":
                # First analyze available assets
                analysis = await self._analyze_wow_assets(wow_path)
                
                if "error" in analysis:
                    return {"success": False, "error": analysis["error"]}
                
                # Set up project if it doesn't exist
                if not project_path.exists():
                    setup_result = await self._setup_unreal_project(project_path, unreal_settings)
                    if not setup_result.get("success"):
                        return setup_result
                
                # Prepare output directories
                content_dir = project_path / "Content" / "WoW"
                results = {
                    "asset_analysis": analysis,
                    "conversions": {}
                }
                
                # Convert assets based on types requested
                if "all" in asset_types or "models" in asset_types:
                    models_dir = content_dir / "Models"
                    models_result = await self._convert_models(wow_path, models_dir, conversion_settings)
                    results["conversions"]["models"] = models_result
                
                if "all" in asset_types or "textures" in asset_types:
                    textures_dir = content_dir / "Textures"
                    textures_result = await self._convert_textures(wow_path, textures_dir, conversion_settings)
                    results["conversions"]["textures"] = textures_result
                
                if "all" in asset_types or "materials" in asset_types:
                    materials_dir = content_dir / "Materials"
                    materials_result = await self._convert_materials(wow_path, materials_dir, conversion_settings)
                    results["conversions"]["materials"] = materials_result
                
                if "all" in asset_types or "data" in asset_types:
                    data_dir = content_dir / "Data"
                    data_result = await self._convert_database_files(wow_path, data_dir)
                    results["conversions"]["data"] = data_result
                
                # Create integration Blueprints
                asset_counts = {
                    "models": len(analysis.get("models", [])),
                    "textures": len(analysis.get("textures", [])),
                    "materials": len(analysis.get("shaders", [])),
                    "data": len(analysis.get("database_files", []))
                }
                
                if unreal_settings.get("generate_blueprints", True):
                    blueprints_result = await self._create_integration_blueprints(project_path, asset_counts)
                    results["blueprints"] = blueprints_result
                
                # Update project configuration
                config_dir = project_path / "Config"
                config_dir.mkdir(exist_ok=True)  # Ensure Config directory exists
                
                config_file = config_dir / "WoWConversion.json"
                
                config = {
                    "WoWConversionSettings": {
                        "SourceDataPath": str(wow_path),
                        "ConversionDate": "2024-01-01",
                        "AssetCounts": asset_counts,
                        "ConversionSettings": conversion_settings
                    }
                }
                
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                
                return {
                    "success": True,
                    "operation": operation,
                    "project_path": str(project_path),
                    "wow_data_path": str(wow_path),
                    "asset_counts": asset_counts,
                    "results": results
                }
            
            elif operation == "import_world":
                # TODO: Implement world/zone import functionality
                return {
                    "success": False,
                    "error": "World import not yet implemented",
                    "message": "This feature will import specific WoW zones/worlds into Unreal Engine"
                }
            
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            return {"success": False, "error": f"WoW to Unreal conversion failed: {str(e)}"}