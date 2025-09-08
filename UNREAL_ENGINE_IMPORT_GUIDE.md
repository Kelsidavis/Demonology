# 🎮 WoW to Unreal Engine Import Guide

## 📁 Project Structure

After running the conversion, you'll find:

```
wow_unreal_workspace/
├── WoWUnrealProject/          # Complete Unreal Engine project
│   ├── WoWUnrealProject.uproject  # Double-click to open in UE5
│   ├── Content/               # All imported assets
│   │   ├── WoW/
│   │   │   ├── Terrain/       # Landscape materials & heightmaps
│   │   │   ├── Models/        # M2/WMO converted meshes
│   │   │   ├── Textures/      # BLP textures converted to PNG/TGA
│   │   │   └── Materials/     # Generated material instances
│   │   └── Blueprints/        # Game logic & spawn systems
│   └── Source/                # C++ source (if needed)
├── exports/                   # Raw converted assets
├── bundles/                   # Manifest files for batch import
└── scene_manifest.json        # Complete scene description
```

## 🚀 Quick Start Guide

### 1. Open the Project
1. Navigate to `wow_unreal_workspace/WoWUnrealProject/`
2. Double-click `WoWUnrealProject.uproject`
3. Choose your Unreal Engine version (5.1+ recommended)
4. Wait for initial compilation (5-10 minutes first time)

### 2. Import WoW Assets

#### Automatic Import (Recommended)
The project includes a **WoW Asset Importer** blueprint that reads the scene manifest:

1. Open `Content/WoW/Blueprints/BP_WoWAssetImporter`
2. Click "Import All Assets" to batch import everything
3. Progress will be shown in the Output Log

#### Manual Import
If automatic import fails, import manually:

**Terrain:**
1. Content Browser → Right-click → Import to /Game/WoW/Terrain/
2. Select all `.png` heightmaps from `exports/Azeroth/`
3. Import Settings:
   - **Texture Group:** World
   - **Compression:** TC_Normalmap (for heightmaps)
   - **Generate Mipmaps:** True

**Models:**
1. Content Browser → Right-click → Import to /Game/WoW/Models/
2. Select all `.fbx`/`.glb` files from `exports/models/`
3. Import Settings:
   - **Import Mesh:** True
   - **Import Materials:** True
   - **Import Textures:** True
   - **Convert Scene:** True
   - **Force Front X Axis:** True (WoW coordinate fix)

**Textures:**
1. Import all `.png`/`.tga` files from `exports/textures/`
2. Auto-generate materials for each texture

### 3. Create the World

#### Method A: Landscape Tool (Terrain)
1. **Landscape Mode** → Create New
2. **Import from File:** Select heightmap PNG
3. **Size:** Use detected size (typically 512x512 per tile)
4. **Scale:** Z=100, XY=100 (adjust for desired world scale)
5. **Material:** Assign auto-generated landscape material

#### Method B: Static Mesh Placement (Pre-built terrain)
1. Drag `.fbx` terrain meshes directly into the level
2. Position using the `scene_manifest.json` coordinates
3. Scale appropriately (WoW uses yards, UE5 uses cm)

### 4. Object Placement System

The converted data includes **precise object placements**:

#### Blueprint Spawner System
1. Use `BP_WoWObjectSpawner` blueprint
2. Reads placement data from scene manifest
3. Automatically spawns:
   - Trees and rocks (M2 doodads)
   - Buildings (WMO objects)  
   - NPCs and creatures
   - Interactive objects

#### Manual Placement
Reference `scene_manifest.json` for exact coordinates:
```json
{
  "placements": [
    {
      "ref": "asset_00001",
      "kind": "m2", 
      "translation": [1234.5, 2345.6, 100.0],
      "rotation_deg": [0, 45, 0],
      "scale": [1.0, 1.0, 1.0],
      "tile": [32, 32]
    }
  ]
}
```

### 5. Animation System

#### Character Animations (M2)
- Skeletal meshes include full animation data
- Use **Animation Blueprint** to control:
  - Walk/Run cycles
  - Combat animations  
  - Idle poses
  - Death sequences
  - Emotes and gestures

#### Environment Animations
- Animated water
- Moving platforms
- Rotating mechanisms
- Particle effects

### 6. Coordinate System Conversion

**WoW → Unreal Engine:**
- WoW: Z-up, right-handed, yards
- UE5: Z-up, left-handed, centimeters  
- Conversion: `UE_pos = WoW_pos * 91.44` (yards to cm)
- Rotation: Mirror Y-axis for correct handedness

### 7. Performance Optimization

#### LOD (Level of Detail)
- Models include multiple LOD levels
- Configure **LOD Groups** in mesh import settings:
  - LOD0: Full detail (near)
  - LOD1: Reduced (medium)  
  - LOD2: Minimal (far)

#### Culling & Streaming
1. **Hierarchical LOD (HLOD):** Combine distant objects
2. **World Partition:** Stream large worlds efficiently
3. **Occlusion Culling:** Hide objects behind terrain

#### Lighting
1. **Static Lighting:** Bake lightmaps for terrain
2. **Dynamic Lighting:** Use for characters/effects
3. **Sky Lighting:** Import WoW skybox textures

## 🛠 Troubleshooting

### Common Issues

**Import Errors:**
- Ensure Unreal Engine 5.1+
- Check file paths don't exceed 260 characters
- Verify sufficient disk space (100+ GB for full world)

**Coordinate Problems:**
- Check coordinate system conversion settings
- Verify scale factors (yards → centimeters)
- Ensure proper axis conversion (Y-axis mirroring)

**Performance Issues:**
- Enable World Partition for large maps
- Configure proper LOD distances
- Use texture streaming for large texture sets

**Missing Textures:**
- Verify BLP to PNG conversion completed
- Check texture import paths match material references
- Regenerate materials if needed

### Advanced Configuration

**World Scale Adjustment:**
```cpp
// In C++ or Blueprint
float WoWToUnrealScale = 91.44f; // yards to cm
FVector UnrealPos = WoWPos * WoWToUnrealScale;
```

**Custom Shaders:**
- WoW uses specialized shaders for water, terrain blending
- Import custom materials from `Materials/` folder
- Modify in Material Editor for UE5 compatibility

## 🎯 Final Steps

1. **Test the World:**
   - Play in Viewport to verify everything loads
   - Check object placements and scaling
   - Test animation playback

2. **Package for Distribution:**
   - File → Package Project → Platform
   - Configure packaging settings for target platform
   - Build lighting before packaging (Project → Build Lighting)

3. **Multiplayer Setup (Optional):**
   - Configure GameMode for multiplayer
   - Set up player spawning system
   - Network replication for dynamic objects

---

## 🏆 Result

You now have a **complete, playable Unreal Engine 5 project** with:
- ✅ Fully rendered WoW terrain
- ✅ All character models with animations  
- ✅ Environmental objects in correct positions
- ✅ Materials and textures
- ✅ Optimized for performance
- ✅ Ready for gameplay development

**Enjoy exploring your favorite WoW zones in Unreal Engine! 🎮**