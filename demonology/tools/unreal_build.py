# demonology/tools/unreal_build.py
from __future__ import annotations

import asyncio
import json
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Tool, _confine

class UnrealBuildTool(Tool):
    """
    Unreal Engine build, compilation, and packaging tool.
    Handles project compilation, packaging, and deployment.
    """

    def __init__(self):
        super().__init__("unreal_build", "Build, compile, and package Unreal Engine projects")
        
        # Find Unreal Engine installation
        self.ue_path = self._find_unreal_engine()
        self.ubt_path = self._find_unreal_build_tool()
        self.uat_path = self._find_automation_tool()

    def _find_unreal_engine(self) -> Optional[Path]:
        """Find Unreal Engine installation directory."""
        common_paths = [
            Path("/home/k/UnrealEngine"),
            Path.home() / "UnrealEngine", 
            Path("/opt/UnrealEngine"),
            Path("/usr/local/UnrealEngine")
        ]
        
        for path in common_paths:
            if path.exists() and (path / "Engine").exists():
                return path
        return None

    def _find_unreal_build_tool(self) -> Optional[Path]:
        """Find UnrealBuildTool executable."""
        if not self.ue_path:
            return None
            
        system = platform.system().lower()
        if system == "windows":
            ubt_path = self.ue_path / "Engine/Binaries/DotNET/UnrealBuildTool/UnrealBuildTool.exe"
        elif system == "linux":
            ubt_path = self.ue_path / "Engine/Binaries/Linux/UnrealBuildTool"
        else:  # macOS
            ubt_path = self.ue_path / "Engine/Binaries/Mac/UnrealBuildTool"
        
        return ubt_path if ubt_path.exists() else None

    def _find_automation_tool(self) -> Optional[Path]:
        """Find UnrealAutomationTool."""
        if not self.ue_path:
            return None
            
        system = platform.system().lower()
        if system == "windows":
            uat_path = self.ue_path / "Engine/Binaries/DotNET/AutomationTool/AutomationTool.exe"
        else:
            uat_path = self.ue_path / "Engine/Binaries/DotNET/AutomationTool/AutomationTool.dll"
        
        return uat_path if uat_path.exists() else None

    def is_available(self) -> bool:
        """Check if Unreal Engine build tools are available."""
        return self.ue_path is not None and self.ubt_path is not None

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["build", "package", "cook", "clean", "rebuild", "generate_project_files", "analyze"],
                        "description": "Build operation to perform"
                    },
                    "project_path": {
                        "type": "string",
                        "description": "Path to Unreal project root"
                    },
                    "configuration": {
                        "type": "string",
                        "enum": ["Debug", "DebugGame", "Development", "Shipping", "Test"],
                        "description": "Build configuration",
                        "default": "Development"
                    },
                    "platform": {
                        "type": "string", 
                        "enum": ["Win64", "Linux", "Mac", "Android", "IOS"],
                        "description": "Target platform",
                        "default": "Linux"
                    },
                    "target": {
                        "type": "string",
                        "description": "Build target (usually project name + 'Editor' or 'Game')"
                    },
                    "package_settings": {
                        "type": "object",
                        "description": "Packaging-specific settings",
                        "properties": {
                            "archive_directory": {"type": "string", "description": "Output directory for packaged game"},
                            "staging_directory": {"type": "string", "description": "Temporary staging directory"},
                            "pak": {"type": "boolean", "default": True, "description": "Package assets into PAK files"},
                            "compressed": {"type": "boolean", "default": True, "description": "Compress packaged data"},
                            "for_distribution": {"type": "boolean", "default": False, "description": "Package for distribution"},
                            "include_prerequisites": {"type": "boolean", "default": True, "description": "Include engine prerequisites"},
                            "include_app_local_prerequisites": {"type": "boolean", "default": False}
                        }
                    },
                    "build_options": {
                        "type": "object", 
                        "description": "Advanced build options",
                        "properties": {
                            "parallel": {"type": "boolean", "default": True, "description": "Use parallel compilation"},
                            "max_parallel_actions": {"type": "integer", "default": 0, "description": "Max parallel processes (0=auto)"},
                            "verbose": {"type": "boolean", "default": False, "description": "Verbose output"},
                            "ignore_engine_version": {"type": "boolean", "default": False},
                            "no_engine_changes": {"type": "boolean", "default": False}
                        }
                    }
                },
                "required": ["operation", "project_path"]
            }
        }

    async def _run_command(self, cmd: List[str], cwd: Optional[Path] = None, timeout: int = 3600) -> Dict[str, Any]:
        """Execute a build command asynchronously."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd) if cwd else None
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return {
                    "success": False,
                    "error": f"Command timed out after {timeout} seconds",
                    "stdout": "",
                    "stderr": ""
                }
            
            return {
                "success": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace")
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": ""
            }

    async def _build_project(self, project_path: Path, target: str, platform: str, 
                           configuration: str, build_options: Dict[str, Any]) -> Dict[str, Any]:
        """Build the Unreal project using UnrealBuildTool."""
        if not self.ubt_path:
            return {"success": False, "error": "UnrealBuildTool not found"}
        
        # Find .uproject file
        uproject_files = list(project_path.glob("*.uproject"))
        if not uproject_files:
            return {"success": False, "error": "No .uproject file found"}
        
        project_name = uproject_files[0].stem
        if not target:
            target = f"{project_name}Editor"  # Default to Editor target
        
        # Build UnrealBuildTool command
        cmd = [str(self.ubt_path), target, platform, configuration]
        
        # Add project file
        cmd.extend(["-project", str(uproject_files[0])])
        
        # Add build options
        if build_options.get("parallel", True):
            cmd.append("-parallel")
        
        max_actions = build_options.get("max_parallel_actions", 0)
        if max_actions > 0:
            cmd.extend(["-MaxParallelActions", str(max_actions)])
        
        if build_options.get("verbose", False):
            cmd.append("-verbose")
        
        if build_options.get("ignore_engine_version", False):
            cmd.append("-IgnoreEngineVersion")
        
        if build_options.get("no_engine_changes", False):
            cmd.append("-NoEngineChanges")
        
        # Execute build
        result = await self._run_command(cmd, cwd=project_path, timeout=7200)  # 2 hour timeout
        
        return {
            "success": result["success"],
            "target": target,
            "platform": platform,
            "configuration": configuration,
            "build_output": result["stdout"][-2000:],  # Last 2000 chars
            "build_errors": result["stderr"][-1000:] if result["stderr"] else None,
            "command": " ".join(cmd)
        }

    async def _package_project(self, project_path: Path, platform: str, configuration: str,
                             package_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Package the project using UnrealAutomationTool."""
        if not self.uat_path:
            return {"success": False, "error": "UnrealAutomationTool not found"}
        
        # Find .uproject file
        uproject_files = list(project_path.glob("*.uproject"))
        if not uproject_files:
            return {"success": False, "error": "No .uproject file found"}
        
        # Setup packaging directories
        archive_dir = package_settings.get("archive_directory", "Packaged")
        staging_dir = package_settings.get("staging_directory", "Staging")
        
        archive_path = _confine(project_path / archive_dir)
        staging_path = _confine(project_path / staging_dir)
        
        archive_path.mkdir(parents=True, exist_ok=True)
        staging_path.mkdir(parents=True, exist_ok=True)
        
        # Build packaging command
        cmd = [
            "dotnet" if platform.system().lower() != "windows" else str(self.uat_path),
            str(self.uat_path) if platform.system().lower() != "windows" else "",
            "BuildCookRun",
            f"-project={uproject_files[0]}",
            f"-targetplatform={platform}",
            f"-clientconfig={configuration}",
            f"-archivedirectory={archive_path}",
            f"-stagingdirectory={staging_path}",
            "-build",
            "-cook",
            "-stage",
            "-archive"
        ]
        
        # Remove empty strings
        cmd = [c for c in cmd if c]
        
        # Add packaging options
        if package_settings.get("pak", True):
            cmd.append("-pak")
        
        if package_settings.get("compressed", True):
            cmd.append("-compressed")
        
        if package_settings.get("for_distribution", False):
            cmd.append("-distribution")
        
        if package_settings.get("include_prerequisites", True):
            cmd.append("-prereqs")
        
        if package_settings.get("include_app_local_prerequisites", False):
            cmd.append("-applocal")
        
        # Execute packaging
        result = await self._run_command(cmd, cwd=project_path, timeout=10800)  # 3 hour timeout
        
        return {
            "success": result["success"],
            "platform": platform,
            "configuration": configuration,
            "archive_directory": str(archive_path),
            "staging_directory": str(staging_path),
            "package_output": result["stdout"][-2000:],  # Last 2000 chars
            "package_errors": result["stderr"][-1000:] if result["stderr"] else None,
            "command": " ".join(cmd)
        }

    async def _cook_content(self, project_path: Path, platform: str) -> Dict[str, Any]:
        """Cook content for the specified platform."""
        if not self.uat_path:
            return {"success": False, "error": "UnrealAutomationTool not found"}
        
        uproject_files = list(project_path.glob("*.uproject"))
        if not uproject_files:
            return {"success": False, "error": "No .uproject file found"}
        
        cmd = [
            "dotnet" if platform.system().lower() != "windows" else str(self.uat_path),
            str(self.uat_path) if platform.system().lower() != "windows" else "",
            "BuildCookRun",
            f"-project={uproject_files[0]}",
            f"-targetplatform={platform}",
            "-cook",
            "-cookflavor=ASTC",
            "-skipcook" if platform.lower() == "win64" else "-cook"
        ]
        
        cmd = [c for c in cmd if c]
        
        result = await self._run_command(cmd, cwd=project_path, timeout=3600)
        
        return {
            "success": result["success"],
            "platform": platform,
            "cook_output": result["stdout"][-2000:],
            "cook_errors": result["stderr"][-1000:] if result["stderr"] else None
        }

    async def _clean_project(self, project_path: Path) -> Dict[str, Any]:
        """Clean build artifacts."""
        try:
            import shutil
            
            # Directories to clean
            clean_dirs = ["Binaries", "Intermediate", "Build", "Saved"]
            cleaned = []
            
            for dir_name in clean_dirs:
                dir_path = project_path / dir_name
                if dir_path.exists():
                    shutil.rmtree(dir_path)
                    cleaned.append(dir_name)
            
            return {
                "success": True,
                "cleaned_directories": cleaned,
                "message": f"Cleaned {len(cleaned)} directories"
            }
        except Exception as e:
            return {"success": False, "error": f"Clean failed: {str(e)}"}

    async def execute(self, operation: str, project_path: str, configuration: str = "Development",
                     platform: str = "Linux", target: Optional[str] = None,
                     package_settings: Optional[Dict[str, Any]] = None,
                     build_options: Optional[Dict[str, Any]] = None, **_) -> Dict[str, Any]:
        
        if not self.is_available():
            return {
                "success": False,
                "error": "Unreal Engine build tools not found",
                "hints": [
                    "Install Unreal Engine 5.4+",
                    "Ensure UnrealBuildTool is available",
                    "Check UNREAL_ENGINE_PATH environment variable"
                ]
            }
        
        try:
            proj_path = _confine(Path(project_path))
            
            # Verify this is an Unreal project
            if not any(proj_path.glob("*.uproject")):
                return {"success": False, "error": "Not an Unreal Engine project (no .uproject file found)"}
            
            package_settings = package_settings or {}
            build_options = build_options or {}
            
            if operation == "build":
                return await self._build_project(proj_path, target, platform, configuration, build_options)
            
            elif operation == "package":
                return await self._package_project(proj_path, platform, configuration, package_settings)
            
            elif operation == "cook":
                return await self._cook_content(proj_path, platform)
            
            elif operation == "clean":
                return await self._clean_project(proj_path)
            
            elif operation == "rebuild":
                # Clean then build
                clean_result = await self._clean_project(proj_path)
                if not clean_result["success"]:
                    return clean_result
                
                build_result = await self._build_project(proj_path, target, platform, configuration, build_options)
                build_result["clean_result"] = clean_result
                return build_result
            
            elif operation == "analyze":
                # Analyze project structure and build requirements
                uproject_files = list(proj_path.glob("*.uproject"))
                project_name = uproject_files[0].stem if uproject_files else "Unknown"
                
                with open(uproject_files[0]) as f:
                    project_data = json.load(f)
                
                # Check for source files
                source_dir = proj_path / "Source"
                has_cpp = source_dir.exists() and any(source_dir.rglob("*.cpp"))
                
                return {
                    "success": True,
                    "project_name": project_name,
                    "engine_version": project_data.get("EngineAssociation", "Unknown"),
                    "modules": project_data.get("Modules", []),
                    "plugins": project_data.get("Plugins", []),
                    "has_cpp_code": has_cpp,
                    "build_targets": [f"{project_name}Editor", f"{project_name}Game", f"{project_name}Server"],
                    "supported_platforms": project_data.get("TargetPlatforms", ["Windows", "Linux", "Mac"]),
                    "unreal_path": str(self.ue_path),
                    "build_tools": {
                        "ubt_available": self.ubt_path is not None,
                        "uat_available": self.uat_path is not None,
                        "ubt_path": str(self.ubt_path) if self.ubt_path else None,
                        "uat_path": str(self.uat_path) if self.uat_path else None
                    }
                }
            
            else:
                return {"success": False, "error": f"Unsupported operation: {operation}"}
                
        except Exception as e:
            return {"success": False, "error": f"Build operation failed: {str(e)}"}