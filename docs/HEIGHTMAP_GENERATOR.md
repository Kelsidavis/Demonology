# Heightmap Generator & Converter Tool

## Overview

The Heightmap Generator is a powerful tool for creating and converting heightmaps for 3D terrain generation. It supports importing PNG/JPG heightmaps, generating procedural terrain, and exporting to various game engine formats.

## Features

### **Input Formats**
- **PNG/JPG Images** - Import existing heightmaps from any image format
- **Procedural Generation** - Create terrain using noise algorithms and terrain shaping
- **Multiple Terrain Types** - Mountains, hills, plains, deserts, canyons, islands, valleys

### **Output Formats** 
- **OBJ** - Wavefront OBJ for universal 3D software compatibility
- **STL** - STereoLithography for 3D printing
- **GLB/GLTF** - Modern game engine format with PBR support
- **Unity RAW** - Unity's native terrain heightfield format
- **R16** - 16-bit RAW heightfield format

### **Game Engine Integration**
- **Unity** - RAW heightfield export with proper scaling
- **Unreal Engine** - Optimized mesh export with LOD support
- **Godot** - glTF export with texture mapping
- **Blender** - Full mesh export with materials
- **Generic** - Universal OBJ format for any 3D software

### **Advanced Features**
- **Terrain Analysis** - Automatic terrain type detection and statistics
- **Texture Generation** - Diffuse, normal, and splat maps
- **Mesh Optimization** - Automatic mesh cleanup and optimization
- **Terrain Features** - Rivers, roads, plateaus, craters, ridges
- **Multi-Scale Support** - From small props to massive landscapes

## Usage Examples

### **Convert PNG Heightmap**
```python
# Convert a PNG heightmap to 3D mesh
result = await tool.execute(
    mode="convert",
    input_path="my_heightmap.png",
    output_path="terrain.obj",
    scale={"x": 1000, "y": 100, "z": 1000},
    generate_textures=True,
    optimize=True
)
```

### **Generate Procedural Mountains**
```python
# Create mountainous terrain with rivers and ridges
result = await tool.execute(
    mode="generate",
    width=512,
    height=512,
    terrain_type="mountains",
    scale={"x": 2000, "y": 500, "z": 2000},
    features=["rivers", "ridges"],
    game_engine="unity",
    generate_textures=True
)
```

### **Analyze Existing Heightmap**
```python
# Analyze heightmap characteristics
result = await tool.execute(
    mode="analyze",
    input_path="unknown_terrain.png"
)
```

### **Game Engine Specific Export**
```python
# Export for Unity with RAW heightfield
result = await tool.execute(
    mode="convert",
    input_path="heightmap.png", 
    output_path="unity_terrain.raw",
    game_engine="unity",
    scale={"x": 500, "y": 50, "z": 500}
)

# Export for Unreal Engine with optimization
result = await tool.execute(
    mode="generate",
    terrain_type="canyon",
    width=1024,
    height=1024,
    output_path="unreal_canyon.obj",
    game_engine="unreal",
    optimize=True
)
```

## Parameter Reference

### **Mode Selection**
- `"convert"` - Convert existing PNG/JPG heightmap to 3D mesh
- `"generate"` - Create procedural terrain from scratch
- `"analyze"` - Analyze existing heightmap properties

### **Terrain Types**
- `"mountains"` - High peaks with realistic mountain distribution
- `"hills"` - Rolling hills with gentle slopes
- `"plains"` - Flat terrain with minimal elevation variation
- `"desert"` - Sand dunes and desert landscapes
- `"canyon"` - Deep canyons and cliff formations
- `"islands"` - Island chains with water boundaries
- `"valleys"` - Valley systems with natural drainage

### **Scale Parameters**
```python
scale = {
    "x": 1000,    # World width in units
    "y": 100,     # Maximum height in units  
    "z": 1000     # World depth in units
}
```

### **Noise Parameters**
```python
noise_params = {
    "octaves": 6,        # Number of noise layers (detail)
    "frequency": 0.01,   # Base noise frequency (scale)
    "amplitude": 0.8,    # Maximum noise strength
    "lacunarity": 2.0,   # Frequency multiplier per octave
    "persistence": 0.5   # Amplitude multiplier per octave
}
```

### **Terrain Features**
- `"rivers"` - Meandering river systems that carve through terrain
- `"roads"` - Winding roads that follow terrain contours
- `"plateaus"` - Flat elevated areas for strategic placement
- `"craters"` - Impact craters and depressions
- `"ridges"` - Mountain ridges and elevated spine formations

### **Game Engine Targets**
- `"unity"` - Unity 3D with RAW heightfield support
- `"unreal"` - Unreal Engine with landscape optimization
- `"godot"` - Godot Engine with glTF format
- `"blender"` - Blender 3D with full material support
- `"generic"` - Universal OBJ format

## Response Formats

### **Successful Generation**
```python
{
    "success": True,
    "mode": "generate",
    "terrain_type": "mountains",
    "heightmap_path": "/path/to/heightmap.png",
    "output_path": "/path/to/terrain.obj",
    "mesh_stats": {
        "vertices": 65536,
        "faces": 130050,
        "bounds": [[-500, 0, -500], [500, 100, 500]],
        "volume": 2500000
    },
    "generation_params": {
        "dimensions": [256, 256],
        "scale": {"x": 1000, "y": 100, "z": 1000},
        "noise_params": {...},
        "features": ["rivers", "ridges"]
    },
    "texture_maps": [
        "/path/to/terrain_diffuse.png",
        "/path/to/terrain_normal.png", 
        "/path/to/terrain_splat.png"
    ],
    "game_engine": "unity"
}
```

### **Successful Conversion**
```python
{
    "success": True,
    "mode": "convert",
    "input_path": "/path/to/input.png",
    "output_path": "/path/to/output.obj",
    "mesh_stats": {
        "vertices": 16384,
        "faces": 32256,
        "bounds": [[-250, 0, -250], [250, 75, 250]],
        "volume": 1875000
    },
    "heightmap_info": {
        "dimensions": [128, 128],
        "min_height": 0.1,
        "max_height": 0.9,
        "scale": {"x": 500, "y": 75, "z": 500}
    },
    "texture_maps": [...],
    "game_engine": "unity"
}
```

### **Analysis Results**
```python
{
    "success": True,
    "mode": "analyze",
    "input_path": "/path/to/heightmap.png",
    "heightmap_stats": {
        "dimensions": [512, 512],
        "min_height": 0.0,
        "max_height": 1.0,
        "mean_height": 0.45,
        "height_range": 1.0,
        "rough_estimate": "mountains",
        "recommended_scale": {"x": 1024, "y": 200, "z": 1024}
    },
    "terrain_features": {
        "has_water_areas": True,
        "has_mountains": True, 
        "has_flat_areas": False,
        "height_variation": "high",
        "dominant_elevation": "medium"
    },
    "conversion_suggestions": {
        "recommended_resolution": 2,
        "needs_optimization": False,
        "suitable_for_games": True
    }
}
```

## Workflow Examples

### **Complete Terrain Pipeline**
```python
# 1. Generate base terrain
terrain_result = await tool.execute(
    mode="generate",
    width=512,
    height=512, 
    terrain_type="mountains",
    features=["rivers", "plateaus"],
    output_path="base_terrain.png"
)

# 2. Analyze the generated terrain
analysis = await tool.execute(
    mode="analyze",
    input_path="base_terrain.png"
)

# 3. Convert to game-ready mesh with recommended settings
final_result = await tool.execute(
    mode="convert",
    input_path="base_terrain.png",
    output_path="game_terrain.obj", 
    scale=analysis["heightmap_stats"]["recommended_scale"],
    game_engine="unity",
    generate_textures=True,
    optimize=True
)
```

### **Multi-Format Export**
```python
# Export same terrain for different engines
heightmap = "my_terrain.png"

formats = [
    {"path": "unity_terrain.raw", "engine": "unity"},
    {"path": "unreal_terrain.obj", "engine": "unreal"}, 
    {"path": "godot_terrain.glb", "engine": "godot"},
    {"path": "blender_terrain.obj", "engine": "blender"}
]

for fmt in formats:
    await tool.execute(
        mode="convert",
        input_path=heightmap,
        output_path=fmt["path"],
        game_engine=fmt["engine"],
        optimize=True
    )
```

## Technical Details

### **Mesh Generation**
- **Triangulation** - Quad-based mesh split into triangles for maximum compatibility
- **UV Mapping** - Automatic texture coordinate generation for material support
- **Normal Calculation** - Proper face normals for correct lighting
- **Optimization** - Duplicate removal, hole filling, and mesh cleanup

### **Texture Maps**

**Diffuse Map** - Color based on elevation:
- Blue (0-30%) - Water/low areas
- Green (30-50%) - Grasslands/plains
- Brown (50-80%) - Hills/mountains
- Gray/White (80%+) - Snow peaks

**Normal Map** - Surface detail from height gradients:
- Calculated from neighboring pixel height differences
- Properly formatted RGB values for game engines
- Adjustable intensity based on terrain scale

**Splat Map** - Texture blending weights (RGBA):
- R: Grass texture weight
- G: Dirt/soil texture weight  
- B: Rock/stone texture weight
- A: Snow/ice texture weight

### **Game Engine Integration**

**Unity**
- RAW heightfield format (16-bit little-endian)
- Proper terrain scaling and resolution
- Compatible with Unity's built-in terrain system
- Automatic texture map generation for splatting

**Unreal Engine**
- Optimized mesh export with proper LOD structure
- Material-ready UV coordinates
- Landscape-compatible vertex layout
- Performance optimization for large terrains

**Godot**
- glTF format with embedded textures
- Node-ready mesh structure
- Material export with PBR properties
- Collision mesh generation

## Performance Guidelines

### **Memory Usage**
- 512x512 terrain ≈ 250K vertices, 500K triangles
- 1024x1024 terrain ≈ 1M vertices, 2M triangles
- Consider mesh optimization for large terrains

### **Quality vs Performance**
- **Low Detail**: 128x128 for distant terrain, props
- **Medium Detail**: 256x256 for standard gameplay areas
- **High Detail**: 512x512 for hero/showcase areas
- **Ultra Detail**: 1024x1024+ for cinematic/architectural use

### **Optimization Tips**
1. **Enable optimization** - Always use `optimize=True` for game use
2. **Appropriate resolution** - Don't over-tessellate distant terrain
3. **Texture size** - Match texture resolution to mesh detail
4. **LOD planning** - Generate multiple resolutions for distance-based LOD

## Integration Examples

### **Unity Integration**
```csharp
// Unity C# script to load generated RAW heightfield
public void LoadHeightmapTerrain(string rawPath) {
    TerrainData terrainData = new TerrainData();
    
    // Load RAW file
    byte[] rawBytes = File.ReadAllBytes(rawPath);
    ushort[] heights = new ushort[rawBytes.Length / 2];
    Buffer.BlockCopy(rawBytes, 0, heights, 0, rawBytes.Length);
    
    // Convert to Unity height array
    int resolution = (int)Mathf.Sqrt(heights.Length);
    float[,] heightArray = new float[resolution, resolution];
    
    for (int y = 0; y < resolution; y++) {
        for (int x = 0; x < resolution; x++) {
            heightArray[y, x] = heights[y * resolution + x] / 65535f;
        }
    }
    
    terrainData.heightmapResolution = resolution;
    terrainData.SetHeights(0, 0, heightArray);
}
```

### **Blender Python Integration**
```python
# Blender Python script to import generated OBJ terrain
import bpy
import bmesh

def import_terrain(obj_path):
    # Import OBJ file
    bpy.ops.import_scene.obj(filepath=obj_path)
    
    # Get the imported object
    terrain_obj = bpy.context.selected_objects[0]
    
    # Add subdivision modifier for smoothing
    modifier = terrain_obj.modifiers.new(name="Subsurf", type='SUBSURF')
    modifier.levels = 1
    
    # Apply material if texture maps exist
    material = bpy.data.materials.new(name="TerrainMaterial")
    terrain_obj.data.materials.append(material)
    
    return terrain_obj
```

## Troubleshooting

### **Common Issues**

**Import Errors**
```bash
# Missing dependencies
pip install numpy trimesh pillow

# For advanced features
pip install open3d pymeshlab  # Optional optimization
```

**Memory Issues**
- Use smaller terrain sizes (256x256 instead of 1024x1024)
- Enable optimization to reduce vertex count
- Generate multiple smaller tiles instead of one large terrain

**Export Problems**
- Check file permissions for output directory
- Verify output format is supported by target engine
- Use absolute paths for input/output files

**Quality Issues**
- Increase noise octaves for more detail
- Adjust scale parameters for proper proportions
- Add terrain features for visual interest
- Generate normal maps for surface detail

### **Performance Optimization**

**Large Terrains**
```python
# Generate terrain in tiles for better performance
tile_size = 256
total_size = 1024
tiles = []

for x in range(0, total_size, tile_size):
    for z in range(0, total_size, tile_size):
        tile = await tool.execute(
            mode="generate",
            width=tile_size,
            height=tile_size,
            terrain_type="mountains",
            offset=(x, z),  # Custom parameter for tile positioning
            optimize=True
        )
        tiles.append(tile)
```

**LOD Generation**
```python
# Generate multiple LOD levels
lod_levels = [512, 256, 128, 64]  # Decreasing detail

for i, resolution in enumerate(lod_levels):
    await tool.execute(
        mode="convert",
        input_path="master_heightmap.png",
        output_path=f"terrain_lod{i}.obj",
        width=resolution,
        height=resolution,
        optimize=True
    )
```

## Best Practices

1. **Start Small** - Test with 128x128 or 256x256 terrains before scaling up
2. **Analyze First** - Use analyze mode to understand heightmap characteristics
3. **Scale Appropriately** - Match world scale to intended use case
4. **Generate Textures** - Always include texture maps for realistic materials
5. **Optimize for Target** - Use appropriate game engine settings
6. **Test Import** - Verify exported files work in target 3D software
7. **Version Control** - Keep original heightmaps for future modifications

This tool bridges the gap between 2D heightmap images and fully game-ready 3D terrain, supporting the complete pipeline from concept to implementation across multiple game engines and 3D software packages.