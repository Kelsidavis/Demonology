# demonology/tools/unreal_gameplay.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Tool, _confine

class UnrealGameplayTool(Tool):
    """
    Unreal Engine gameplay mechanics and logic tool.
    Creates game systems, mechanics, AI, and interactive elements.
    """

    def __init__(self):
        super().__init__("unreal_gameplay", "Create and manage Unreal Engine gameplay mechanics and systems")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["create_game_system", "create_ai_behavior", "create_interaction", "create_inventory", 
                               "create_combat_system", "create_level", "create_ui_system", "create_save_system"],
                        "description": "Gameplay operation to perform"
                    },
                    "project_path": {
                        "type": "string",
                        "description": "Path to Unreal project root"
                    },
                    "system_name": {
                        "type": "string",
                        "description": "Name of the gameplay system to create"
                    },
                    "system_type": {
                        "type": "string",
                        "enum": ["player_controller", "game_mode", "ai_controller", "inventory", "combat", "dialogue", 
                               "quest", "level", "ui", "save_load"],
                        "description": "Type of gameplay system"
                    },
                    "game_genre": {
                        "type": "string",
                        "enum": ["fps", "rpg", "platformer", "strategy", "puzzle", "racing", "adventure"],
                        "description": "Game genre to tailor mechanics for",
                        "default": "adventure"
                    },
                    "complexity": {
                        "type": "string",
                        "enum": ["simple", "intermediate", "advanced"],
                        "description": "Complexity level of the system",
                        "default": "intermediate"
                    },
                    "features": {
                        "type": "array",
                        "description": "Specific features to include in the system",
                        "items": {"type": "string"}
                    },
                    "parameters": {
                        "type": "object",
                        "description": "System-specific parameters",
                        "properties": {
                            "player_health": {"type": "number", "default": 100},
                            "movement_speed": {"type": "number", "default": 600},
                            "jump_height": {"type": "number", "default": 420},
                            "inventory_size": {"type": "integer", "default": 20},
                            "max_level": {"type": "integer", "default": 50},
                            "ai_sight_range": {"type": "number", "default": 1500},
                            "interaction_distance": {"type": "number", "default": 200}
                        }
                    }
                },
                "required": ["operation", "project_path", "system_name"]
            }
        }

    def _create_player_controller_system(self, system_name: str, genre: str, complexity: str, 
                                       features: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a player controller system with input handling and movement."""
        
        # Base player controller structure
        controller = {
            "ClassName": f"{system_name}PlayerController",
            "BaseClass": "PlayerController",
            "Components": [],
            "Variables": [
                {
                    "Name": "MovementSpeed",
                    "Type": "float",
                    "DefaultValue": params.get("movement_speed", 600),
                    "Description": "Player movement speed"
                },
                {
                    "Name": "JumpHeight", 
                    "Type": "float",
                    "DefaultValue": params.get("jump_height", 420),
                    "Description": "Player jump height"
                }
            ],
            "Functions": [
                {
                    "Name": "BeginPlay",
                    "Type": "Event",
                    "Description": "Initialize player controller"
                },
                {
                    "Name": "SetupInputComponent", 
                    "Type": "Override",
                    "Description": "Setup input bindings"
                }
            ],
            "InputActions": []
        }

        # Add genre-specific features
        if genre == "fps":
            controller["Variables"].extend([
                {"Name": "MouseSensitivity", "Type": "float", "DefaultValue": 1.0},
                {"Name": "bInvertYAxis", "Type": "bool", "DefaultValue": False}
            ])
            controller["InputActions"].extend([
                {"Name": "MoveForward", "Key": "W"},
                {"Name": "MoveRight", "Key": "D"}, 
                {"Name": "Turn", "Type": "Axis"},
                {"Name": "LookUp", "Type": "Axis"},
                {"Name": "Fire", "Key": "LeftMouseButton"},
                {"Name": "Aim", "Key": "RightMouseButton"}
            ])
            
        elif genre == "platformer":
            controller["Variables"].extend([
                {"Name": "bCanDoubleJump", "Type": "bool", "DefaultValue": True},
                {"Name": "DoubleJumpHeight", "Type": "float", "DefaultValue": 300}
            ])
            controller["InputActions"].extend([
                {"Name": "MoveRight", "Key": "D"},
                {"Name": "Jump", "Key": "Space"},
                {"Name": "Dash", "Key": "LeftShift"}
            ])
            
        elif genre == "rpg":
            controller["Variables"].extend([
                {"Name": "bIsInCombat", "Type": "bool", "DefaultValue": False},
                {"Name": "TargetActor", "Type": "AActor*", "DefaultValue": None}
            ])
            controller["InputActions"].extend([
                {"Name": "MoveForward", "Key": "W"},
                {"Name": "MoveRight", "Key": "D"},
                {"Name": "Interact", "Key": "E"},
                {"Name": "OpenInventory", "Key": "Tab"},
                {"Name": "Attack", "Key": "LeftMouseButton"},
                {"Name": "Block", "Key": "RightMouseButton"}
            ])

        # Add complexity-based features
        if complexity in ["intermediate", "advanced"]:
            controller["Variables"].append({
                "Name": "InputBuffer",
                "Type": "TArray<FInputAction>",
                "Description": "Input buffering for complex moves"
            })
            
        if complexity == "advanced":
            controller["Functions"].extend([
                {"Name": "HandleAdvancedInput", "Type": "Custom"},
                {"Name": "ProcessInputBuffer", "Type": "Custom"},
                {"Name": "UpdateCameraSystem", "Type": "Custom"}
            ])

        return controller

    def _create_ai_behavior_system(self, system_name: str, complexity: str, 
                                 features: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Create an AI behavior system with decision making."""
        
        ai_system = {
            "ClassName": f"{system_name}AIController", 
            "BaseClass": "AIController",
            "BehaviorTree": f"BT_{system_name}",
            "BlackBoard": f"BB_{system_name}",
            "Variables": [
                {
                    "Name": "SightRange",
                    "Type": "float", 
                    "DefaultValue": params.get("ai_sight_range", 1500),
                    "Description": "AI sight detection range"
                },
                {
                    "Name": "TargetPlayer",
                    "Type": "APawn*",
                    "Description": "Current player target"
                },
                {
                    "Name": "PatrolPoints",
                    "Type": "TArray<FVector>",
                    "Description": "Patrol waypoints"
                }
            ],
            "States": [
                {
                    "Name": "Idle",
                    "Description": "AI is idle, not engaged",
                    "Transitions": ["Patrol", "Alert"]
                },
                {
                    "Name": "Patrol",
                    "Description": "AI is patrolling area",
                    "Transitions": ["Idle", "Alert", "Chase"]
                },
                {
                    "Name": "Alert",
                    "Description": "AI detected something suspicious", 
                    "Transitions": ["Patrol", "Chase", "Idle"]
                },
                {
                    "Name": "Chase",
                    "Description": "AI is chasing target",
                    "Transitions": ["Attack", "Alert", "Patrol"]
                }
            ],
            "BehaviorNodes": [
                {
                    "Type": "Sequence",
                    "Name": "MainBehavior",
                    "Children": [
                        {"Type": "Service", "Name": "UpdateTarget"},
                        {"Type": "Selector", "Name": "ActionSelector", "Children": [
                            {"Type": "Task", "Name": "ChasePlayer"},
                            {"Type": "Task", "Name": "PatrolArea"},
                            {"Type": "Task", "Name": "IdleWait"}
                        ]}
                    ]
                }
            ]
        }

        if complexity in ["intermediate", "advanced"]:
            ai_system["States"].extend([
                {"Name": "Attack", "Description": "AI is attacking target", "Transitions": ["Chase", "Alert"]},
                {"Name": "Investigate", "Description": "AI investigating last known position", "Transitions": ["Patrol", "Alert", "Chase"]}
            ])

        if complexity == "advanced":
            ai_system["Variables"].extend([
                {"Name": "EmotionalState", "Type": "EEmotionType", "DefaultValue": "Neutral"},
                {"Name": "MemorySystem", "Type": "UAIMemoryComponent*"},
                {"Name": "CommunicationRadius", "Type": "float", "DefaultValue": 1000}
            ])
            
        return ai_system

    def _create_inventory_system(self, system_name: str, complexity: str, 
                               features: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Create an inventory management system."""
        
        inventory = {
            "ClassName": f"{system_name}InventoryComponent",
            "BaseClass": "ActorComponent",
            "Variables": [
                {
                    "Name": "MaxSlots",
                    "Type": "int32",
                    "DefaultValue": params.get("inventory_size", 20),
                    "Description": "Maximum inventory slots"
                },
                {
                    "Name": "Items",
                    "Type": "TArray<FInventoryItem>",
                    "Description": "Current inventory items"
                },
                {
                    "Name": "Currency",
                    "Type": "int32",
                    "DefaultValue": 0,
                    "Description": "Player currency/gold"
                }
            ],
            "Structs": [
                {
                    "Name": "FInventoryItem",
                    "Fields": [
                        {"Name": "ItemID", "Type": "FString"},
                        {"Name": "DisplayName", "Type": "FText"},
                        {"Name": "Description", "Type": "FText"}, 
                        {"Name": "Icon", "Type": "UTexture2D*"},
                        {"Name": "Quantity", "Type": "int32"},
                        {"Name": "MaxStack", "Type": "int32"},
                        {"Name": "ItemType", "Type": "EItemType"},
                        {"Name": "Rarity", "Type": "EItemRarity"},
                        {"Name": "Value", "Type": "int32"}
                    ]
                }
            ],
            "Functions": [
                {"Name": "AddItem", "Parameters": ["FInventoryItem"], "ReturnType": "bool"},
                {"Name": "RemoveItem", "Parameters": ["FString", "int32"], "ReturnType": "bool"},
                {"Name": "GetItem", "Parameters": ["FString"], "ReturnType": "FInventoryItem"},
                {"Name": "HasItem", "Parameters": ["FString", "int32"], "ReturnType": "bool"},
                {"Name": "UseItem", "Parameters": ["FString"], "ReturnType": "bool"},
                {"Name": "DropItem", "Parameters": ["FString", "int32"], "ReturnType": "bool"}
            ]
        }

        if complexity in ["intermediate", "advanced"]:
            inventory["Functions"].extend([
                {"Name": "SortInventory", "ReturnType": "void"},
                {"Name": "FilterItems", "Parameters": ["EItemType"], "ReturnType": "TArray<FInventoryItem>"},
                {"Name": "GetTotalValue", "ReturnType": "int32"}
            ])

        if complexity == "advanced":
            inventory["Variables"].extend([
                {"Name": "EquippedItems", "Type": "TMap<EEquipSlot, FInventoryItem>"},
                {"Name": "QuickSlots", "Type": "TArray<FInventoryItem>"}
            ])
            inventory["Functions"].extend([
                {"Name": "EquipItem", "Parameters": ["FString"], "ReturnType": "bool"},
                {"Name": "UnequipItem", "Parameters": ["EEquipSlot"], "ReturnType": "bool"},
                {"Name": "SwapItems", "Parameters": ["int32", "int32"], "ReturnType": "bool"}
            ])

        return inventory

    def _create_level_system(self, system_name: str, genre: str, complexity: str) -> Dict[str, Any]:
        """Create a complete level with gameplay elements."""
        
        level = {
            "LevelName": f"{system_name}Level",
            "LevelBlueprint": f"BP_{system_name}Level",
            "Geometry": {
                "Terrain": {
                    "Type": "Landscape",
                    "Size": "1009x1009",
                    "Materials": ["M_Grass", "M_Stone", "M_Dirt"]
                },
                "StaticMeshes": []
            },
            "Lighting": {
                "SkyLight": {"Intensity": 1.0, "Color": "White"},
                "DirectionalLight": {"Intensity": 3.0, "Color": "Warm"}
            },
            "GameplayElements": [],
            "SpawnPoints": [
                {"Type": "PlayerStart", "Location": "0,0,100", "Rotation": "0,0,0"}
            ],
            "Objectives": []
        }

        # Add genre-specific level elements
        if genre == "fps":
            level["GameplayElements"].extend([
                {"Type": "WeaponPickup", "Count": 5},
                {"Type": "HealthPack", "Count": 10},
                {"Type": "AmmoCrate", "Count": 8},
                {"Type": "EnemySpawner", "Count": 3}
            ])
            
        elif genre == "platformer":
            level["GameplayElements"].extend([
                {"Type": "MovingPlatform", "Count": 5},
                {"Type": "Collectible", "Count": 20},
                {"Type": "Checkpoint", "Count": 3},
                {"Type": "Hazard", "Count": 8}
            ])
            
        elif genre == "rpg":
            level["GameplayElements"].extend([
                {"Type": "NPCVendor", "Count": 2},
                {"Type": "QuestGiver", "Count": 3},
                {"Type": "Treasure", "Count": 10},
                {"Type": "Enemy", "Count": 15}
            ])

        if complexity in ["intermediate", "advanced"]:
            level["Objectives"] = [
                {"Name": "Primary", "Description": "Complete main objective"},
                {"Name": "Secondary", "Description": "Find all collectibles"},
                {"Name": "Bonus", "Description": "Complete without taking damage"}
            ]

        return level

    async def execute(self, operation: str, project_path: str, system_name: str,
                     system_type: str = "player_controller", game_genre: str = "adventure",
                     complexity: str = "intermediate", features: Optional[List[str]] = None,
                     parameters: Optional[Dict[str, Any]] = None, **_) -> Dict[str, Any]:
        
        try:
            proj_path = _confine(Path(project_path))
            
            # Verify this is an Unreal project
            if not any(proj_path.glob("*.uproject")):
                return {"success": False, "error": "Not an Unreal Engine project (no .uproject file found)"}
            
            features = features or []
            parameters = parameters or {}
            
            # Create appropriate directory structure
            blueprints_dir = _confine(proj_path / "Content" / "Blueprints" / "Gameplay")
            blueprints_dir.mkdir(parents=True, exist_ok=True)
            
            system_data = None
            
            if operation == "create_game_system":
                if system_type == "player_controller":
                    system_data = self._create_player_controller_system(system_name, game_genre, complexity, features, parameters)
                elif system_type == "ai_controller":
                    system_data = self._create_ai_behavior_system(system_name, complexity, features, parameters)
                elif system_type == "inventory":
                    system_data = self._create_inventory_system(system_name, complexity, features, parameters)
                else:
                    return {"success": False, "error": f"Unsupported system type: {system_type}"}
                    
            elif operation == "create_level":
                system_data = self._create_level_system(system_name, game_genre, complexity)
                
            elif operation == "create_ai_behavior":
                system_data = self._create_ai_behavior_system(system_name, complexity, features, parameters)
                
            elif operation == "create_inventory":
                system_data = self._create_inventory_system(system_name, complexity, features, parameters)
                
            else:
                return {"success": False, "error": f"Unsupported operation: {operation}"}
            
            if not system_data:
                return {"success": False, "error": "Failed to generate system data"}
            
            # Save the system definition
            system_file = blueprints_dir / f"{system_name}.json"
            with open(system_file, 'w') as f:
                json.dump(system_data, f, indent=2)
            
            # Create Blueprint placeholder
            blueprint_file = blueprints_dir / f"BP_{system_name}.uasset"
            with open(blueprint_file, 'w') as f:
                f.write(f"// Gameplay System Blueprint: {system_name}\n")
                f.write(f"// Type: {system_type}\n")
                f.write(f"// Genre: {game_genre}\n")
                f.write(f"// Complexity: {complexity}\n")
                f.write(f"// System Definition: {system_name}.json\n")
            
            return {
                "success": True,
                "operation": operation,
                "system_name": system_name,
                "system_type": system_type,
                "genre": game_genre,
                "complexity": complexity,
                "system_file": str(system_file),
                "blueprint_file": str(blueprint_file),
                "features_count": len(system_data.get("Variables", []) + system_data.get("Functions", [])),
                "message": f"Created {system_type} gameplay system: {system_name}"
            }
            
        except Exception as e:
            return {"success": False, "error": f"Gameplay system creation failed: {str(e)}"}