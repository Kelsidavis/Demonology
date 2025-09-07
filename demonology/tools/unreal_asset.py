# demonology/tools/unreal_asset.py
from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Tool, _confine

class UnrealAssetTool(Tool):
    """
    Unreal Engine asset management tool.
    Handles import, export, and management of game assets including models, textures, sounds, etc.
    """

    def __init__(self):
        super().__init__("unreal_asset", "Import, export, and manage Unreal Engine assets")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["import", "export", "organize", "convert", "optimize", "list", "analyze"],
                        "description": "Asset operation to perform"
                    },
                    "project_path": {
                        "type": "string",
                        "description": "Path to Unreal project root"
                    },
                    "asset_path": {
                        "type": "string",
                        "description": "Path to asset file(s) or directory"
                    },
                    "asset_type": {
                        "type": "string",
                        "enum": ["mesh", "texture", "sound", "animation", "material", "particle", "font", "data"],
                        "description": "Type of asset to work with"
                    },
                    "target_path": {
                        "type": "string",
                        "description": "Target path within Content folder",
                        "default": "Assets"
                    },
                    "import_settings": {
                        "type": "object",
                        "description": "Import-specific settings",
                        "properties": {
                            "auto_generate_materials": {"type": "boolean", "default": True},
                            "auto_generate_collision": {"type": "boolean", "default": False},
                            "combine_meshes": {"type": "boolean", "default": False},
                            "import_textures": {"type": "boolean", "default": True},
                            "texture_resolution": {"type": "integer", "enum": [256, 512, 1024, 2048, 4096], "default": 1024},
                            "compression_quality": {"type": "string", "enum": ["Low", "Medium", "High", "Lossless"], "default": "High"}
                        }
                    },
                    "optimization_settings": {
                        "type": "object",
                        "description": "Asset optimization settings",
                        "properties": {
                            "reduce_polycount": {"type": "boolean", "default": False},
                            "target_polycount": {"type": "integer", "default": 1000},
                            "compress_textures": {"type": "boolean", "default": True},
                            "generate_lods": {"type": "boolean", "default": True},
                            "lod_levels": {"type": "integer", "default": 3}
                        }
                    }
                },
                "required": ["operation", "project_path"]
            }
        }

    def _get_supported_extensions(self, asset_type: str) -> List[str]:
        """Get supported file extensions for each asset type."""
        extensions_map = {
            "mesh": [".fbx", ".obj", ".3ds", ".dae", ".blend"],
            "texture": [".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tiff", ".exr", ".hdr"],
            "sound": [".wav", ".ogg", ".mp3", ".aiff"],
            "animation": [".fbx", ".bvh", ".dae"],
            "material": [".mtl", ".mat"],
            "font": [".ttf", ".otf"],
            "data": [".json", ".xml", ".csv"]
        }
        return extensions_map.get(asset_type, [])

    def _analyze_asset_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze an asset file and return metadata."""
        try:
            stat = file_path.stat()
            file_size = stat.st_size
            
            # Basic file info
            info = {
                "name": file_path.stem,
                "extension": file_path.suffix.lower(),
                "size_bytes": file_size,
                "size_mb": round(file_size / (1024 * 1024), 2),
                "path": str(file_path)
            }
            
            # Asset type detection
            if file_path.suffix.lower() in [".fbx", ".obj", ".3ds", ".dae", ".blend"]:
                info["type"] = "mesh"
                info["estimated_polygons"] = "unknown"  # Would need actual mesh analysis
            elif file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tiff"]:
                info["type"] = "texture"
                # Try to get image dimensions if PIL is available
                try:
                    from PIL import Image
                    with Image.open(file_path) as img:
                        info["dimensions"] = {"width": img.width, "height": img.height}
                        info["format"] = img.format
                except:
                    info["dimensions"] = "unknown"
            elif file_path.suffix.lower() in [".wav", ".ogg", ".mp3", ".aiff"]:
                info["type"] = "sound"
                info["duration"] = "unknown"  # Would need audio analysis
            else:
                info["type"] = "unknown"
            
            return info
            
        except Exception as e:
            return {"error": str(e), "path": str(file_path)}

    def _create_import_metadata(self, asset_info: Dict[str, Any], import_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Create import metadata for Unreal Engine."""
        metadata = {
            "AssetImportData": {
                "SourceFilePath": asset_info.get("path", ""),
                "SourceFileTimestamp": "2024-01-01T00:00:00.000Z",
                "ImportSettings": import_settings
            },
            "AssetMetaData": {
                "ImportedSize": asset_info.get("size_bytes", 0),
                "AssetType": asset_info.get("type", "unknown"),
                "OriginalFormat": asset_info.get("extension", ""),
                "ImportTime": "2024-01-01T00:00:00.000Z"
            }
        }
        
        # Add type-specific metadata
        if asset_info.get("type") == "texture":
            metadata["TextureSettings"] = {
                "CompressionSettings": "TC_Default",
                "Filter": "TF_Default", 
                "MipGenSettings": "TMGS_FromTextureGroup",
                "PowerOfTwoMode": "ETexturePowerOfTwoSetting::None",
                "Dimensions": asset_info.get("dimensions", {"width": 0, "height": 0})
            }
        elif asset_info.get("type") == "mesh":
            metadata["MeshSettings"] = {
                "bAutoGenerateCollision": import_settings.get("auto_generate_collision", False),
                "bCombineMeshes": import_settings.get("combine_meshes", False),
                "bImportMaterials": import_settings.get("auto_generate_materials", True),
                "bImportTextures": import_settings.get("import_textures", True)
            }
        
        return metadata

    async def _import_assets(self, project_path: Path, asset_path: Path, target_path: str, 
                           asset_type: str, import_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Import assets into the Unreal project."""
        try:
            content_dir = _confine(project_path / "Content" / target_path)
            content_dir.mkdir(parents=True, exist_ok=True)
            
            imported_assets = []
            errors = []
            
            # Handle single file or directory
            files_to_import = []
            if asset_path.is_file():
                files_to_import = [asset_path]
            elif asset_path.is_dir():
                supported_exts = self._get_supported_extensions(asset_type)
                for ext in supported_exts:
                    files_to_import.extend(asset_path.rglob(f"*{ext}"))
            
            for file_path in files_to_import:
                try:
                    # Analyze the asset
                    asset_info = self._analyze_asset_file(file_path)
                    
                    if "error" in asset_info:
                        errors.append(f"Failed to analyze {file_path.name}: {asset_info['error']}")
                        continue
                    
                    # Create target filename
                    target_file = content_dir / file_path.name
                    
                    # Copy the asset file
                    shutil.copy2(file_path, target_file)
                    
                    # Create metadata file
                    metadata = self._create_import_metadata(asset_info, import_settings)
                    metadata_file = content_dir / f"{file_path.stem}.uasset.json"
                    
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    # Create placeholder .uasset file
                    uasset_file = content_dir / f"{file_path.stem}.uasset"
                    with open(uasset_file, 'w') as f:
                        f.write(f"// Asset placeholder: {file_path.stem}\n")
                        f.write(f"// Source: {file_path}\n")
                        f.write(f"// Metadata: {metadata_file.name}\n")
                    
                    imported_assets.append({
                        "name": file_path.stem,
                        "type": asset_info.get("type", "unknown"),
                        "source": str(file_path),
                        "target": str(target_file),
                        "uasset": str(uasset_file),
                        "metadata": str(metadata_file),
                        "size_mb": asset_info.get("size_mb", 0)
                    })
                    
                except Exception as e:
                    errors.append(f"Failed to import {file_path.name}: {str(e)}")
            
            return {
                "success": True,
                "imported_assets": imported_assets,
                "import_count": len(imported_assets),
                "errors": errors,
                "target_directory": str(content_dir)
            }
            
        except Exception as e:
            return {"success": False, "error": f"Import failed: {str(e)}"}

    async def _organize_assets(self, project_path: Path) -> Dict[str, Any]:
        """Organize assets into appropriate folders by type."""
        try:
            content_dir = _confine(project_path / "Content")
            if not content_dir.exists():
                return {"success": False, "error": "Content directory not found"}
            
            # Asset organization structure
            organization = {
                "mesh": "Meshes",
                "texture": "Textures", 
                "sound": "Audio",
                "animation": "Animations",
                "material": "Materials",
                "font": "Fonts",
                "data": "Data"
            }
            
            organized_files = []
            errors = []
            
            # Create organization directories
            for folder in organization.values():
                (content_dir / folder).mkdir(exist_ok=True)
            
            # Find and organize assets
            for asset_file in content_dir.rglob("*"):
                if asset_file.is_file() and asset_file.suffix.lower() not in ['.json', '.uasset']:
                    try:
                        asset_info = self._analyze_asset_file(asset_file)
                        asset_type = asset_info.get("type", "unknown")
                        
                        if asset_type in organization:
                            target_dir = content_dir / organization[asset_type]
                            target_file = target_dir / asset_file.name
                            
                            # Only move if not already in correct location
                            if asset_file.parent != target_dir:
                                shutil.move(str(asset_file), str(target_file))
                                
                                # Move associated metadata files
                                metadata_file = asset_file.parent / f"{asset_file.stem}.uasset.json"
                                uasset_file = asset_file.parent / f"{asset_file.stem}.uasset"
                                
                                if metadata_file.exists():
                                    shutil.move(str(metadata_file), str(target_dir / metadata_file.name))
                                if uasset_file.exists():
                                    shutil.move(str(uasset_file), str(target_dir / uasset_file.name))
                                
                                organized_files.append({
                                    "name": asset_file.name,
                                    "type": asset_type,
                                    "moved_to": str(target_file)
                                })
                                
                    except Exception as e:
                        errors.append(f"Failed to organize {asset_file.name}: {str(e)}")
            
            return {
                "success": True,
                "organized_files": organized_files,
                "organization_count": len(organized_files),
                "folder_structure": organization,
                "errors": errors
            }
            
        except Exception as e:
            return {"success": False, "error": f"Organization failed: {str(e)}"}

    async def _list_assets(self, project_path: Path, asset_type: Optional[str] = None) -> Dict[str, Any]:
        """List all assets in the project."""
        try:
            content_dir = _confine(project_path / "Content")
            if not content_dir.exists():
                return {"success": False, "error": "Content directory not found"}
            
            assets = []
            total_size = 0
            
            for asset_file in content_dir.rglob("*"):
                if asset_file.is_file() and asset_file.suffix.lower() not in ['.json', '.uasset']:
                    asset_info = self._analyze_asset_file(asset_file)
                    
                    if asset_type is None or asset_info.get("type") == asset_type:
                        assets.append(asset_info)
                        total_size += asset_info.get("size_bytes", 0)
            
            # Group by type
            by_type = {}
            for asset in assets:
                asset_type_key = asset.get("type", "unknown")
                if asset_type_key not in by_type:
                    by_type[asset_type_key] = []
                by_type[asset_type_key].append(asset)
            
            return {
                "success": True,
                "assets": assets,
                "asset_count": len(assets),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "by_type": by_type,
                "type_counts": {k: len(v) for k, v in by_type.items()}
            }
            
        except Exception as e:
            return {"success": False, "error": f"Asset listing failed: {str(e)}"}

    async def execute(self, operation: str, project_path: str, asset_path: Optional[str] = None,
                     asset_type: str = "mesh", target_path: str = "Assets",
                     import_settings: Optional[Dict[str, Any]] = None,
                     optimization_settings: Optional[Dict[str, Any]] = None, **_) -> Dict[str, Any]:
        
        try:
            proj_path = _confine(Path(project_path))
            
            # Verify this is an Unreal project
            if not any(proj_path.glob("*.uproject")):
                return {"success": False, "error": "Not an Unreal Engine project (no .uproject file found)"}
            
            import_settings = import_settings or {}
            optimization_settings = optimization_settings or {}
            
            if operation == "import":
                if not asset_path:
                    return {"success": False, "error": "asset_path is required for import operation"}
                
                asset_source = _confine(Path(asset_path))
                if not asset_source.exists():
                    return {"success": False, "error": f"Asset path not found: {asset_path}"}
                
                return await self._import_assets(proj_path, asset_source, target_path, asset_type, import_settings)
            
            elif operation == "organize":
                return await self._organize_assets(proj_path)
            
            elif operation == "list":
                return await self._list_assets(proj_path, asset_type if asset_type != "mesh" else None)
            
            elif operation == "analyze":
                if not asset_path:
                    return {"success": False, "error": "asset_path is required for analyze operation"}
                
                asset_source = _confine(Path(asset_path))
                if not asset_source.exists():
                    return {"success": False, "error": f"Asset path not found: {asset_path}"}
                
                if asset_source.is_file():
                    analysis = self._analyze_asset_file(asset_source)
                    return {"success": True, "analysis": analysis}
                else:
                    # Analyze directory
                    files_analyzed = []
                    for file_path in asset_source.rglob("*"):
                        if file_path.is_file():
                            analysis = self._analyze_asset_file(file_path)
                            files_analyzed.append(analysis)
                    
                    return {
                        "success": True,
                        "directory": str(asset_source),
                        "files_analyzed": files_analyzed,
                        "file_count": len(files_analyzed)
                    }
            
            else:
                return {"success": False, "error": f"Unsupported operation: {operation}"}
                
        except Exception as e:
            return {"success": False, "error": f"Asset operation failed: {str(e)}"}