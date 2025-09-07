# demonology/tools/unreal_project.py
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Tool, _confine, _ok_path

class UnrealProjectTool(Tool):
    """
    Unreal Engine project management tool for autonomous game development.
    Handles project creation, configuration, and lifecycle management.
    """

    def __init__(self):
        super().__init__("unreal_project", "Create and manage Unreal Engine projects")
        
        # Try to find Unreal Engine installation
        self.ue_path = self._find_unreal_engine()
        self.ubt_path = self._find_unreal_build_tool()

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
            
        ubt_paths = [
            self.ue_path / "Engine/Binaries/DotNET/UnrealBuildTool/UnrealBuildTool.exe",
            self.ue_path / "Engine/Binaries/DotNET/UnrealBuildTool/UnrealBuildTool.dll",
            self.ue_path / "Engine/Binaries/Linux/UnrealBuildTool",
        ]
        
        for path in ubt_paths:
            if path.exists():
                return path
        return None

    def is_available(self) -> bool:
        """Check if Unreal Engine is available."""
        return self.ue_path is not None

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["create", "configure", "info", "clean", "generate_project_files"],
                        "description": "Project operation to perform"
                    },
                    "project_name": {
                        "type": "string", 
                        "description": "Name of the project"
                    },
                    "project_path": {
                        "type": "string",
                        "description": "Path where project should be created/managed",
                        "default": "."
                    },
                    "template": {
                        "type": "string",
                        "enum": ["blank", "first_person", "third_person", "top_down", "puzzle", "racing", "strategy"],
                        "description": "Project template to use for creation",
                        "default": "blank"
                    },
                    "engine_version": {
                        "type": "string",
                        "description": "Target Unreal Engine version",
                        "default": "5.4"
                    },
                    "project_settings": {
                        "type": "object",
                        "description": "Additional project configuration settings",
                        "properties": {
                            "with_starter_content": {"type": "boolean", "default": False},
                            "target_platforms": {
                                "type": "array",
                                "items": {"type": "string"},
                                "default": ["Windows"]
                            },
                            "plugins": {
                                "type": "array", 
                                "items": {"type": "string"},
                                "description": "Plugins to enable by default"
                            }
                        }
                    }
                },
                "required": ["operation", "project_name"]
            }
        }

    async def _run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> Dict[str, Any]:
        """Execute a command asynchronously."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd) if cwd else None
            )
            
            stdout, stderr = await proc.communicate()
            
            return {
                "returncode": proc.returncode,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace")
            }
        except Exception as e:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": str(e)
            }

    async def _create_project(self, project_name: str, project_path: Path, template: str, 
                            project_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new Unreal Engine project."""
        if not self.ue_path:
            return {"success": False, "error": "Unreal Engine not found"}

        # Ensure project directory
        project_dir = _confine(project_path / project_name)
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Create .uproject file
        uproject_data = {
            "FileVersion": 3,
            "EngineAssociation": project_settings.get("engine_version", "5.4"),
            "Category": "",
            "Description": f"Auto-generated project: {project_name}",
            "Modules": [
                {
                    "Name": project_name,
                    "Type": "Runtime",
                    "LoadingPhase": "Default",
                    "AdditionalDependencies": [
                        "Engine",
                        "CoreUObject"
                    ]
                }
            ],
            "Plugins": [
                {"Name": plugin, "Enabled": True} 
                for plugin in project_settings.get("plugins", [])
            ],
            "TargetPlatforms": project_settings.get("target_platforms", ["Windows"])
        }

        uproject_path = project_dir / f"{project_name}.uproject"
        with open(uproject_path, 'w') as f:
            json.dump(uproject_data, f, indent=2)

        # Create basic directory structure
        directories = [
            "Source", "Source/" + project_name,
            "Content", "Content/Blueprints", "Content/Maps", 
            "Content/Materials", "Content/Meshes", "Content/Textures",
            "Config", "Binaries", "Intermediate"
        ]
        
        for dir_name in directories:
            (project_dir / dir_name).mkdir(parents=True, exist_ok=True)

        # Create basic C++ class files if needed
        await self._create_basic_cpp_files(project_dir, project_name)
        
        # Create basic config files
        await self._create_config_files(project_dir, project_name, project_settings)

        return {
            "success": True,
            "project_path": str(project_dir),
            "uproject_path": str(uproject_path),
            "template": template,
            "message": f"Created Unreal Engine project '{project_name}'"
        }

    async def _create_basic_cpp_files(self, project_dir: Path, project_name: str):
        """Create basic C++ source files for the project."""
        source_dir = project_dir / "Source" / project_name
        
        # Create main module header
        module_header = f"""// {project_name}.h
#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

class F{project_name}Module : public IModuleInterface
{{
public:
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;
}};
"""
        
        # Create main module implementation
        module_cpp = f"""// {project_name}.cpp
#include "{project_name}.h"
#include "Modules/ModuleManager.h"

IMPLEMENT_PRIMARY_GAME_MODULE(F{project_name}Module, {project_name}, "{project_name}");

void F{project_name}Module::StartupModule()
{{
    // This code will execute after your module is loaded into memory
}}

void F{project_name}Module::ShutdownModule()
{{
    // This function may be called during shutdown to clean up your module
}}
"""

        # Create Build.cs file
        build_cs = f"""// {project_name}.Build.cs
using UnrealBuildTool;

public class {project_name} : ModuleRules
{{
    public {project_name}(ReadOnlyTargetRules Target) : base(Target)
    {{
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        
        PublicDependencyModuleNames.AddRange(new string[] {{ "Core", "CoreUObject", "Engine", "InputCore" }});
        
        PrivateDependencyModuleNames.AddRange(new string[] {{ }});
    }}
}}
"""

        # Write files
        with open(source_dir / f"{project_name}.h", 'w') as f:
            f.write(module_header)
        with open(source_dir / f"{project_name}.cpp", 'w') as f:
            f.write(module_cpp)
        with open(source_dir / f"{project_name}.Build.cs", 'w') as f:
            f.write(build_cs)

    async def _create_config_files(self, project_dir: Path, project_name: str, settings: Dict[str, Any]):
        """Create basic configuration files."""
        config_dir = project_dir / "Config"
        
        # DefaultEngine.ini
        engine_config = f"""[/Script/EngineSettings.GameMapsSettings]
GameDefaultMap=/Game/Maps/DefaultMap
EditorStartupMap=/Game/Maps/DefaultMap

[/Script/HardwareTargeting.HardwareTargetingSettings]
TargetedHardwareClass=Desktop
AppliedTargetedHardwareClass=Desktop
DefaultGraphicsPerformance=Maximum
AppliedDefaultGraphicsPerformance=Maximum

[/Script/Engine.Engine]
+ActiveGameNameRedirects=(OldGameName="TP_Blank",NewGameName="/Script/{project_name}")
+ActiveGameNameRedirects=(OldGameName="/Script/TP_Blank",NewGameName="/Script/{project_name}")
+ActiveClassRedirects=(OldClassName="TP_BlankGameModeBase",NewClassName="{project_name}GameModeBase")

[/Script/Engine.RendererSettings]
r.ReflectionMethod=1
r.GenerateMeshDistanceFields=True
r.DynamicGlobalIlluminationMethod=1
r.Lumen.TraceMeshSDFs=0
"""
        
        with open(config_dir / "DefaultEngine.ini", 'w') as f:
            f.write(engine_config)

        # DefaultGame.ini
        game_config = f"""[/Script/EngineSettings.GeneralProjectSettings]
ProjectID={{{project_name.upper()}}}
ProjectName={project_name}
"""
        
        with open(config_dir / "DefaultGame.ini", 'w') as f:
            f.write(game_config)

    async def _generate_project_files(self, project_path: Path) -> Dict[str, Any]:
        """Generate project files using Unreal's build system."""
        if not self.ue_path:
            return {"success": False, "error": "Unreal Engine not found"}
        
        # Find .uproject file
        uproject_files = list(project_path.glob("*.uproject"))
        if not uproject_files:
            return {"success": False, "error": "No .uproject file found"}
        
        uproject_file = uproject_files[0]
        
        # Run GenerateProjectFiles
        gen_script = self.ue_path / "GenerateProjectFiles.sh"
        if not gen_script.exists():
            gen_script = self.ue_path / "GenerateProjectFiles.bat"
            
        if not gen_script.exists():
            return {"success": False, "error": "GenerateProjectFiles script not found"}
        
        result = await self._run_command([str(gen_script), str(uproject_file)], cwd=self.ue_path)
        
        return {
            "success": result["returncode"] == 0,
            "output": result["stdout"],
            "error": result["stderr"] if result["returncode"] != 0 else None
        }

    async def execute(self, operation: str, project_name: str, project_path: str = ".", 
                     template: str = "blank", engine_version: str = "5.4",
                     project_settings: Optional[Dict[str, Any]] = None, **_) -> Dict[str, Any]:
        
        if not self.is_available():
            return {
                "success": False, 
                "error": "Unreal Engine not found. Please install Unreal Engine 5.4+",
                "hints": ["Install Unreal Engine from Epic Games Launcher or build from source"]
            }

        project_settings = project_settings or {}
        project_settings.setdefault("engine_version", engine_version)
        
        base_path = _confine(Path(project_path))
        
        try:
            if operation == "create":
                return await self._create_project(project_name, base_path, template, project_settings)
                
            elif operation == "generate_project_files":
                proj_path = base_path / project_name if (base_path / project_name).exists() else base_path
                return await self._generate_project_files(proj_path)
                
            elif operation == "info":
                proj_path = base_path / project_name if (base_path / project_name).exists() else base_path
                uproject_files = list(proj_path.glob("*.uproject"))
                if not uproject_files:
                    return {"success": False, "error": f"No Unreal project found at {proj_path}"}
                
                with open(uproject_files[0]) as f:
                    project_data = json.load(f)
                
                return {
                    "success": True,
                    "project_name": uproject_files[0].stem,
                    "project_path": str(proj_path),
                    "uproject_file": str(uproject_files[0]),
                    "engine_version": project_data.get("EngineAssociation", "Unknown"),
                    "modules": project_data.get("Modules", []),
                    "plugins": project_data.get("Plugins", []),
                    "target_platforms": project_data.get("TargetPlatforms", [])
                }
                
            elif operation == "clean":
                proj_path = base_path / project_name if (base_path / project_name).exists() else base_path
                # Clean build artifacts
                for clean_dir in ["Binaries", "Intermediate", "Build"]:
                    clean_path = proj_path / clean_dir
                    if clean_path.exists():
                        shutil.rmtree(clean_path)
                
                return {
                    "success": True,
                    "message": f"Cleaned build artifacts for {project_name}",
                    "cleaned_directories": ["Binaries", "Intermediate", "Build"]
                }
                
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            return {"success": False, "error": f"Operation failed: {str(e)}"}