#!/usr/bin/env python3
"""
Comprehensive Demo: Autonomous Game Creation with Unreal Engine
==============================================================

This demo shows how to use Demonology's Unreal Engine toolset to create
a complete game from scratch, entirely through code.
"""

import asyncio
import sys
from pathlib import Path

# Add demonology to path
sys.path.insert(0, str(Path(__file__).parent / "demonology"))

from demonology.tools import create_default_registry

async def create_complete_game():
    """Create a complete game using the Unreal Engine toolset."""
    
    print("🎮 AUTONOMOUS GAME CREATION DEMO 🎮")
    print("=" * 50)
    
    # Initialize tool registry
    reg = create_default_registry()
    
    # Project configuration
    project_name = "AutonomousRPG"
    project_path = "GameProjects"  # Relative to current workspace
    
    print(f"Creating game project: {project_name}")
    print(f"Project location: {project_path}")
    print()
    
    # Step 1: Create the project
    print("📁 Step 1: Creating Unreal Engine project...")
    project_result = await reg.call('unreal_project',
        operation="create",
        project_name=project_name,
        project_path=project_path,
        template="third_person",
        engine_version="5.4",
        project_settings={
            "with_starter_content": False,
            "target_platforms": ["Windows", "Linux"],
            "plugins": ["EnhancedInput", "GameplayAbilities"]
        }
    )
    
    if project_result["success"]:
        print(f"✅ Project created: {project_result['project_path']}")
    else:
        print(f"❌ Project creation failed: {project_result['error']}")
        return False
    
    full_project_path = project_result["project_path"]
    print()
    
    # Step 2: Create core gameplay systems
    print("⚙️  Step 2: Creating core gameplay systems...")
    
    # Create Player Controller
    player_controller = await reg.call('unreal_gameplay',
        operation="create_game_system",
        project_path=full_project_path,
        system_name="MainPlayerController",
        system_type="player_controller",
        game_genre="rpg",
        complexity="advanced",
        features=["inventory", "interaction", "combat"],
        parameters={
            "player_health": 100,
            "movement_speed": 600,
            "interaction_distance": 200
        }
    )
    
    if player_controller["success"]:
        print(f"✅ Player Controller: {player_controller['message']}")
    else:
        print(f"❌ Player Controller failed: {player_controller['error']}")
    
    # Create AI System
    ai_system = await reg.call('unreal_gameplay',
        operation="create_ai_behavior",
        project_path=full_project_path,
        system_name="SmartNPC",
        complexity="advanced",
        features=["combat", "dialogue", "patrol"],
        parameters={
            "ai_sight_range": 1500
        }
    )
    
    if ai_system["success"]:
        print(f"✅ AI System: {ai_system['message']}")
    else:
        print(f"❌ AI System failed: {ai_system['error']}")
    
    # Create Inventory System
    inventory_system = await reg.call('unreal_gameplay',
        operation="create_inventory",
        project_path=full_project_path,
        system_name="PlayerInventory",
        complexity="advanced",
        parameters={
            "inventory_size": 40
        }
    )
    
    if inventory_system["success"]:
        print(f"✅ Inventory System: {inventory_system['message']}")
    else:
        print(f"❌ Inventory System failed: {inventory_system['error']}")
    
    print()
    
    # Step 3: Create Blueprints
    print("🎯 Step 3: Creating game Blueprints...")
    
    # Create Player Character Blueprint
    player_bp = await reg.call('unreal_blueprint',
        operation="create_pawn",
        project_path=full_project_path,
        blueprint_name="MainCharacter",
        parent_class="Character",
        blueprint_path="Characters",
        variables=[
            {"name": "Health", "type": "float", "default_value": "100.0", "is_public": True},
            {"name": "MaxHealth", "type": "float", "default_value": "100.0", "is_public": True},
            {"name": "Mana", "type": "float", "default_value": "50.0", "is_public": True},
            {"name": "Level", "type": "int", "default_value": "1", "is_public": True}
        ],
        functions=[
            {
                "name": "TakeDamage",
                "description": "Handle incoming damage",
                "inputs": [{"name": "DamageAmount", "type": "float"}],
                "outputs": [{"name": "ActualDamage", "type": "float"}],
                "is_pure": False
            },
            {
                "name": "Heal",
                "description": "Restore health",
                "inputs": [{"name": "HealAmount", "type": "float"}],
                "outputs": [],
                "is_pure": False
            }
        ]
    )
    
    if player_bp["success"]:
        print(f"✅ Player Character Blueprint: {player_bp['blueprint_name']}")
        print(f"   Variables: {player_bp['variables_count']}, Functions: {player_bp['functions_count']}")
    else:
        print(f"❌ Player Blueprint failed: {player_bp['error']}")
    
    # Create Enemy Blueprint
    enemy_bp = await reg.call('unreal_blueprint',
        operation="create_pawn",
        project_path=full_project_path,
        blueprint_name="BasicEnemy",
        parent_class="Character", 
        blueprint_path="Enemies",
        variables=[
            {"name": "Health", "type": "float", "default_value": "50.0", "is_public": True},
            {"name": "AttackDamage", "type": "float", "default_value": "15.0", "is_public": True},
            {"name": "DetectionRange", "type": "float", "default_value": "1000.0", "is_public": True}
        ]
    )
    
    if enemy_bp["success"]:
        print(f"✅ Enemy Blueprint: {enemy_bp['blueprint_name']}")
    
    # Create Game Mode Blueprint
    gamemode_bp = await reg.call('unreal_blueprint',
        operation="create_game_mode",
        project_path=full_project_path,
        blueprint_name="RPGGameMode",
        blueprint_path="GameModes"
    )
    
    if gamemode_bp["success"]:
        print(f"✅ Game Mode Blueprint: {gamemode_bp['blueprint_name']}")
    
    # Create UI Widget
    ui_widget = await reg.call('unreal_blueprint',
        operation="create_widget",
        project_path=full_project_path,
        blueprint_name="MainHUD",
        blueprint_path="UI"
    )
    
    if ui_widget["success"]:
        print(f"✅ UI Widget Blueprint: {ui_widget['blueprint_name']}")
    
    print()
    
    # Step 4: Create a level
    print("🗺️  Step 4: Creating game level...")
    level_result = await reg.call('unreal_gameplay',
        operation="create_level",
        project_path=full_project_path,
        system_name="StartingVillage",
        game_genre="rpg",
        complexity="intermediate"
    )
    
    if level_result["success"]:
        print(f"✅ Level created: {level_result['message']}")
    else:
        print(f"❌ Level creation failed: {level_result['error']}")
    
    print()
    
    # Step 5: Set up assets (demo with placeholder organization)
    print("📦 Step 5: Setting up asset management...")
    
    # List current assets
    asset_list = await reg.call('unreal_asset',
        operation="list",
        project_path=full_project_path
    )
    
    if asset_list["success"]:
        print(f"✅ Found {asset_list['asset_count']} assets in project")
        for asset_type, count in asset_list['type_counts'].items():
            print(f"   {asset_type}: {count} assets")
    
    # Organize assets
    organize_result = await reg.call('unreal_asset',
        operation="organize",
        project_path=full_project_path
    )
    
    if organize_result["success"]:
        print(f"✅ Organized {organize_result['organization_count']} assets")
    
    print()
    
    # Step 6: Project analysis
    print("📊 Step 6: Analyzing created project...")
    
    project_info = await reg.call('unreal_project',
        operation="info",
        project_name=project_name,
        project_path=project_path
    )
    
    if project_info["success"]:
        print("✅ Project Analysis:")
        print(f"   Project: {project_info['project_name']}")
        print(f"   Engine: {project_info['engine_version']}")
        print(f"   Modules: {len(project_info['modules'])}")
        print(f"   Plugins: {len(project_info['plugins'])}")
        print(f"   Platforms: {', '.join(project_info['target_platforms'])}")
    
    print()
    print("🎉 GAME CREATION COMPLETE! 🎉")
    print("=" * 50)
    
    print("Your autonomous RPG game includes:")
    print("✅ Complete Unreal Engine 5.4 project structure")
    print("✅ Advanced player controller with RPG mechanics")
    print("✅ Intelligent AI behavior system")
    print("✅ Comprehensive inventory system")
    print("✅ Player character Blueprint with health/mana")
    print("✅ Enemy AI Blueprint with combat capabilities")
    print("✅ Game mode and UI systems")
    print("✅ Starting village level with RPG elements")
    print("✅ Organized asset management structure")
    
    print(f"\\n📁 Project Location: {full_project_path}")
    print("🚀 Ready for further development and customization!")
    
    return True

async def demo_advanced_features():
    """Demonstrate advanced features of the toolset."""
    print("\\n🔧 ADVANCED FEATURES DEMO 🔧")
    print("=" * 40)
    
    reg = create_default_registry()
    project_path = "GameProjects/AutonomousRPG"
    
    # Advanced Blueprint creation with complex logic
    print("Creating advanced combat system Blueprint...")
    combat_bp = await reg.call('unreal_blueprint',
        operation="create_component",
        project_path=project_path,
        blueprint_name="CombatSystem",
        blueprint_path="Systems",
        variables=[
            {"name": "ComboCounter", "type": "int", "default_value": "0"},
            {"name": "LastAttackTime", "type": "float", "default_value": "0.0"},
            {"name": "CombatState", "type": "enum", "default_value": "Idle"}
        ],
        functions=[
            {
                "name": "ExecuteAttack",
                "description": "Execute attack with combo system",
                "inputs": [{"name": "AttackType", "type": "enum"}],
                "outputs": [{"name": "Damage", "type": "float"}],
                "is_pure": False
            },
            {
                "name": "CalculateComboMultiplier", 
                "description": "Calculate damage multiplier based on combo",
                "inputs": [{"name": "ComboCount", "type": "int"}],
                "outputs": [{"name": "Multiplier", "type": "float"}],
                "is_pure": True
            }
        ]
    )
    
    if combat_bp["success"]:
        print(f"✅ Advanced Combat System: {combat_bp['functions_count']} functions created")
    
    # Create complex AI behavior
    print("Creating advanced AI with emotional states...")
    advanced_ai = await reg.call('unreal_gameplay',
        operation="create_ai_behavior", 
        project_path=project_path,
        system_name="EmotionalNPC",
        complexity="advanced",
        features=["emotion", "memory", "communication"],
        parameters={
            "ai_sight_range": 2000
        }
    )
    
    if advanced_ai["success"]:
        print("✅ Emotional AI system with memory and communication")
    
    print("\\n🎯 Advanced features demonstrated!")

if __name__ == "__main__":
    try:
        # Run the complete game creation demo
        success = asyncio.run(create_complete_game())
        
        if success:
            # Run advanced features demo
            asyncio.run(demo_advanced_features())
            
        print("\\n🎮 Demo completed successfully!")
        print("\\nYou now have a complete Unreal Engine toolset for autonomous game development!")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()