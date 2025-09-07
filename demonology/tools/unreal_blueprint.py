# demonology/tools/unreal_blueprint.py
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import Tool, _confine

class UnrealBlueprintTool(Tool):
    """
    Unreal Engine Blueprint visual scripting tool.
    Creates and manages Blueprint assets, visual scripts, and game logic.
    """

    def __init__(self):
        super().__init__("unreal_blueprint", "Create and manage Unreal Engine Blueprints and visual scripts")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["create_actor", "create_pawn", "create_game_mode", "create_widget", 
                               "create_component", "create_function", "create_variable", "connect_nodes"],
                        "description": "Blueprint operation to perform"
                    },
                    "project_path": {
                        "type": "string",
                        "description": "Path to Unreal project root"
                    },
                    "blueprint_name": {
                        "type": "string",
                        "description": "Name of the Blueprint to create/modify"
                    },
                    "blueprint_type": {
                        "type": "string",
                        "enum": ["Actor", "Pawn", "Character", "GameMode", "Widget", "Component", "Function", "Macro"],
                        "description": "Type of Blueprint to create"
                    },
                    "parent_class": {
                        "type": "string",
                        "description": "Parent class for the Blueprint",
                        "default": "Actor"
                    },
                    "blueprint_path": {
                        "type": "string",
                        "description": "Relative path within Content folder",
                        "default": "Blueprints"
                    },
                    "nodes": {
                        "type": "array",
                        "description": "Visual script nodes to add",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "description": "Node type (e.g., 'Event', 'Function', 'Variable')"},
                                "name": {"type": "string", "description": "Node name"},
                                "inputs": {"type": "array", "items": {"type": "string"}},
                                "outputs": {"type": "array", "items": {"type": "string"}},
                                "properties": {"type": "object", "description": "Node-specific properties"}
                            }
                        }
                    },
                    "variables": {
                        "type": "array",
                        "description": "Blueprint variables to create",
                        "items": {
                            "type": "object", 
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string", "description": "Variable type (e.g., 'float', 'int', 'bool', 'string')"},
                                "default_value": {"type": "string"},
                                "is_public": {"type": "boolean", "default": False}
                            }
                        }
                    },
                    "functions": {
                        "type": "array",
                        "description": "Custom functions to create",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "inputs": {"type": "array", "items": {"type": "object"}},
                                "outputs": {"type": "array", "items": {"type": "object"}},
                                "is_pure": {"type": "boolean", "default": False},
                                "logic": {"type": "array", "description": "Function logic nodes"}
                            }
                        }
                    }
                },
                "required": ["operation", "project_path", "blueprint_name"]
            }
        }

    def _generate_blueprint_guid(self) -> str:
        """Generate a unique GUID for Blueprint assets."""
        return str(uuid.uuid4()).upper().replace('-', '')

    def _create_blueprint_metadata(self, blueprint_name: str, blueprint_type: str, parent_class: str) -> Dict[str, Any]:
        """Create Blueprint metadata structure."""
        return {
            "Type": "Blueprint",
            "Name": blueprint_name,
            "Class": blueprint_type,
            "ParentClass": parent_class,
            "GUID": self._generate_blueprint_guid(),
            "Version": {
                "Major": 5,
                "Minor": 4,
                "Patch": 0
            },
            "Timestamp": "2024-01-01T00:00:00.000Z"
        }

    def _create_variable_node(self, var_name: str, var_type: str, default_value: Any = None, is_public: bool = False) -> Dict[str, Any]:
        """Create a Blueprint variable node."""
        return {
            "NodeGuid": self._generate_blueprint_guid(),
            "NodeType": "Variable",
            "VariableName": var_name,
            "VariableType": var_type,
            "DefaultValue": default_value,
            "bIsPublic": is_public,
            "bIsEditable": is_public,
            "Category": "Default",
            "Position": {"X": 0, "Y": 0}
        }

    def _create_event_node(self, event_name: str, position: Tuple[int, int] = (0, 0)) -> Dict[str, Any]:
        """Create a Blueprint event node."""
        return {
            "NodeGuid": self._generate_blueprint_guid(),
            "NodeType": "Event",
            "EventName": event_name,
            "Position": {"X": position[0], "Y": position[1]},
            "OutputPins": [
                {
                    "PinName": "exec",
                    "PinType": "exec"
                }
            ]
        }

    def _create_function_call_node(self, function_name: str, target_class: str, position: Tuple[int, int] = (0, 0)) -> Dict[str, Any]:
        """Create a function call node."""
        return {
            "NodeGuid": self._generate_blueprint_guid(),
            "NodeType": "FunctionCall",
            "FunctionName": function_name,
            "TargetClass": target_class,
            "Position": {"X": position[0], "Y": position[1]},
            "InputPins": [
                {
                    "PinName": "exec",
                    "PinType": "exec"
                }
            ],
            "OutputPins": [
                {
                    "PinName": "exec",
                    "PinType": "exec"
                }
            ]
        }

    def _create_basic_actor_blueprint(self, blueprint_name: str, parent_class: str = "Actor") -> Dict[str, Any]:
        """Create a basic Actor Blueprint with common events."""
        blueprint = {
            "Metadata": self._create_blueprint_metadata(blueprint_name, "Actor", parent_class),
            "Variables": [],
            "Functions": [],
            "EventGraph": {
                "Nodes": [
                    self._create_event_node("BeginPlay", (100, 100)),
                    self._create_event_node("Tick", (100, 300)),
                    self._create_event_node("EndPlay", (100, 500))
                ],
                "Connections": []
            },
            "Components": [
                {
                    "ComponentName": "DefaultSceneRoot",
                    "ComponentClass": "SceneComponent",
                    "bIsRoot": True
                }
            ]
        }
        return blueprint

    def _create_pawn_blueprint(self, blueprint_name: str, parent_class: str = "Pawn") -> Dict[str, Any]:
        """Create a Pawn Blueprint with movement capabilities."""
        blueprint = self._create_basic_actor_blueprint(blueprint_name, parent_class)
        
        # Add Pawn-specific events
        blueprint["EventGraph"]["Nodes"].extend([
            self._create_event_node("SetupPlayerInputComponent", (100, 700)),
            self._create_event_node("PossessedBy", (100, 900)),
            self._create_event_node("UnPossessed", (100, 1100))
        ])
        
        # Add movement component
        blueprint["Components"].append({
            "ComponentName": "MovementComponent",
            "ComponentClass": "PawnMovementComponent",
            "bIsRoot": False
        })
        
        return blueprint

    def _create_widget_blueprint(self, blueprint_name: str) -> Dict[str, Any]:
        """Create a UI Widget Blueprint."""
        return {
            "Metadata": self._create_blueprint_metadata(blueprint_name, "Widget", "UserWidget"),
            "Variables": [],
            "Functions": [
                {
                    "FunctionName": "Construct",
                    "bIsOverride": True,
                    "Nodes": [
                        self._create_event_node("Construct", (100, 100))
                    ]
                }
            ],
            "WidgetTree": {
                "RootWidget": {
                    "WidgetClass": "CanvasPanel",
                    "Children": []
                }
            }
        }

    def _create_game_mode_blueprint(self, blueprint_name: str) -> Dict[str, Any]:
        """Create a GameMode Blueprint."""
        blueprint = self._create_basic_actor_blueprint(blueprint_name, "GameModeBase")
        
        # Add GameMode-specific events and functions
        blueprint["EventGraph"]["Nodes"].extend([
            self._create_event_node("InitGame", (100, 700)),
            self._create_event_node("PreLogin", (100, 900)),
            self._create_event_node("PostLogin", (100, 1100)),
            self._create_event_node("StartPlay", (100, 1300))
        ])
        
        # Add default classes variables
        blueprint["Variables"].extend([
            self._create_variable_node("DefaultPawnClass", "TSubclassOf<Pawn>", None, True),
            self._create_variable_node("PlayerControllerClass", "TSubclassOf<PlayerController>", None, True)
        ])
        
        return blueprint

    async def _create_blueprint_file(self, project_path: Path, blueprint_path: str, blueprint_name: str, 
                                   blueprint_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create the actual Blueprint file on disk."""
        try:
            content_path = _confine(project_path / "Content" / blueprint_path)
            content_path.mkdir(parents=True, exist_ok=True)
            
            blueprint_file = content_path / f"{blueprint_name}.uasset"
            
            # Create a JSON representation (in real UE, this would be binary)
            # For our purposes, we'll create a readable JSON version
            json_file = content_path / f"{blueprint_name}.json"
            
            with open(json_file, 'w') as f:
                json.dump(blueprint_data, f, indent=2)
            
            # Create a placeholder .uasset file to indicate Blueprint presence
            with open(blueprint_file, 'w') as f:
                f.write(f"// Blueprint asset placeholder: {blueprint_name}\n")
                f.write(f"// Actual Blueprint data in: {blueprint_name}.json\n")
            
            return {
                "success": True,
                "blueprint_file": str(blueprint_file),
                "json_file": str(json_file),
                "blueprint_path": str(content_path)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def execute(self, operation: str, project_path: str, blueprint_name: str,
                     blueprint_type: str = "Actor", parent_class: str = "Actor",
                     blueprint_path: str = "Blueprints", nodes: Optional[List[Dict]] = None,
                     variables: Optional[List[Dict]] = None, functions: Optional[List[Dict]] = None,
                     **_) -> Dict[str, Any]:
        
        try:
            proj_path = _confine(Path(project_path))
            
            # Verify this is an Unreal project
            if not any(proj_path.glob("*.uproject")):
                return {"success": False, "error": "Not an Unreal Engine project (no .uproject file found)"}
            
            if operation == "create_actor":
                blueprint_data = self._create_basic_actor_blueprint(blueprint_name, parent_class)
                
            elif operation == "create_pawn":
                blueprint_data = self._create_pawn_blueprint(blueprint_name, parent_class)
                
            elif operation == "create_game_mode":
                blueprint_data = self._create_game_mode_blueprint(blueprint_name)
                
            elif operation == "create_widget":
                blueprint_data = self._create_widget_blueprint(blueprint_name)
                
            elif operation == "create_component":
                blueprint_data = self._create_basic_actor_blueprint(blueprint_name, "ActorComponent")
                
            else:
                return {"success": False, "error": f"Unsupported operation: {operation}"}
            
            # Add custom variables if provided
            if variables:
                for var in variables:
                    var_node = self._create_variable_node(
                        var.get("name", "NewVariable"),
                        var.get("type", "float"),
                        var.get("default_value"),
                        var.get("is_public", False)
                    )
                    blueprint_data["Variables"].append(var_node)
            
            # Add custom functions if provided
            if functions:
                for func in functions:
                    func_data = {
                        "FunctionName": func.get("name", "NewFunction"),
                        "Description": func.get("description", ""),
                        "bIsPublic": True,
                        "bIsPure": func.get("is_pure", False),
                        "Inputs": func.get("inputs", []),
                        "Outputs": func.get("outputs", []),
                        "Nodes": func.get("logic", [])
                    }
                    blueprint_data["Functions"].append(func_data)
            
            # Create the Blueprint file
            result = await self._create_blueprint_file(proj_path, blueprint_path, blueprint_name, blueprint_data)
            
            if result["success"]:
                return {
                    "success": True,
                    "operation": operation,
                    "blueprint_name": blueprint_name,
                    "blueprint_type": blueprint_type,
                    "blueprint_file": result["blueprint_file"],
                    "json_representation": result["json_file"],
                    "variables_count": len(blueprint_data.get("Variables", [])),
                    "functions_count": len(blueprint_data.get("Functions", [])),
                    "nodes_count": len(blueprint_data.get("EventGraph", {}).get("Nodes", [])),
                    "message": f"Created {blueprint_type} Blueprint: {blueprint_name}"
                }
            else:
                return result
                
        except Exception as e:
            return {"success": False, "error": f"Blueprint creation failed: {str(e)}"}