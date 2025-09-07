# 3D Model Generator Tool

## Overview

The 3D Model Generator creates procedural 3D models from text prompts or explicit part specifications. It supports various primitive shapes, design templates, materials, and export formats.

## Features

### **Primitive Types**
- **box** - Rectangular boxes with configurable dimensions
- **cylinder** - Cylindrical shapes with radius, height, and segment control  
- **sphere** - Spherical shapes with subdivision control
- **cone** - Conical shapes tapering to a point
- **torus** - Donut shapes with major/minor radius
- **wedge** - Triangular prisms (ramps, roofs, blades)
- **tube** - Hollow cylinders with inner/outer radius
- **helix** - Spring/spiral shapes with configurable turns

### **Design Templates**
- **spaceship** - Sci-fi spacecraft with engines, wings, cockpit
- **rocket** - Missiles/rockets with fins and nose cone
- **house** - Basic residential structures with roofs
- **car** - Vehicles with body, wheels, windows
- **building** - Office buildings/skyscrapers with floors
- **tree** - Various tree types (oak, pine, palm) with realistic proportions
- **furniture** - Chairs, tables, desks, bookshelves
- **weapon** - Swords, spears, axes with proper handles

### **Materials & Colors**
- **Color support** - Parse colors from prompts (`"red spaceship"`, `"green tree"`)
- **Named colors** - red, green, blue, white, black, yellow, purple, cyan, gray, orange, pink
- **RGB values** - Specify custom colors as `color: 255,128,64` or `color: 1.0,0.5,0.2`
- **Vertex colors** - Applied to mesh for colored rendering
- **UV mapping** - Basic texture coordinate generation

### **Mesh Operations**
- **Boolean operations** - Union, difference, intersection between shapes
- **Extrusion** - Convert 2D outlines to 3D shapes
- **Optimization** - Remove duplicates, fix normals, fill holes, smoothing
- **Validation** - Mesh integrity checks and cleanup

### **Export Formats**
- **OBJ** - Wavefront OBJ (default, widely supported)
- **STL** - STereoLithography (3D printing)
- **PLY** - Polygon File Format
- **GLB** - Binary glTF (web/game engines)
- **GLTF** - Text glTF with separate assets

## Usage Examples

### Text Prompt Generation
```python
# Simple shapes with colors
await tool.execute(prompt="red spaceship length:8 width:3")
await tool.execute(prompt="green oak tree height:12") 
await tool.execute(prompt="blue chair")
await tool.execute(prompt="steel sword length:3")

# Buildings and structures  
await tool.execute(prompt="office building height:25 width:10")
await tool.execute(prompt="wooden table")
await tool.execute(prompt="pine tree height:15")
```

### Part-Based Construction
```python
# Custom torus
await tool.execute(mode='merge', parts=[{
    'mode': 'primitive',
    'primitive': 'torus',
    'radius': 2.0,
    'inner_radius': 0.8,
    'segments': 24,
    'transform': {
        'translate': [0, 1, 0],
        'rotate_deg': [0, 45, 0],
        'material': {
            'color': [1.0, 0.5, 0.2]  # Orange
        }
    }
}])

# Complex helix spring
await tool.execute(mode='merge', parts=[{
    'mode': 'primitive', 
    'primitive': 'helix',
    'radius': 1.5,
    'height': 4.0,
    'turns': 2.5,
    'segments': 48
}])
```

### Advanced Features
```python
# With optimization and custom format
await tool.execute(
    prompt="smooth red car", 
    format="stl",
    optimize=True
)

# Custom output path
await tool.execute(
    prompt="spaceship",
    output_path="models/my_ship.obj"
)
```

## Parameter Reference

### **Common Prompt Parameters**
- `length:X` / `width:X` / `height:X` - Dimensions
- `radius:X` / `inner_radius:X` - Circular measurements  
- `scale:X` - Overall size multiplier
- `segments:X` - Mesh resolution for curved shapes
- `color: R,G,B` - RGB color values (0-255 or 0.0-1.0)
- Named colors: `red`, `blue`, `green`, etc.

### **Primitive-Specific Parameters**
```python
# Box
{'primitive': 'box', 'size': [width, height, depth]}

# Cylinder  
{'primitive': 'cylinder', 'radius': 1.0, 'height': 2.0, 'segments': 24}

# Torus
{'primitive': 'torus', 'radius': 2.0, 'inner_radius': 0.5, 'segments': 32, 'minor_segments': 16}

# Helix
{'primitive': 'helix', 'radius': 1.0, 'height': 3.0, 'turns': 2.0, 'segments': 48}

# Tube (hollow cylinder)
{'primitive': 'tube', 'radius': 1.0, 'inner_radius': 0.7, 'height': 2.0, 'segments': 24}
```

### **Transform Parameters**
```python
'transform': {
    'translate': [x, y, z],           # Position offset
    'rotate_deg': [rx, ry, rz],       # Rotation in degrees
    'scale': 2.0,                     # Uniform scaling
    'material': {
        'color': [r, g, b, a],        # RGBA color (0.0-1.0)
        'generate_uvs': True          # Generate texture coordinates
    }
}
```

## Output Statistics

The tool returns detailed statistics about generated models:

```python
{
    "success": True,
    "path": "/path/to/output.obj",
    "mode": "auto",
    "prompt": "red spaceship",
    "parts_count": 7,
    "stats": {
        "vertices": 1466,              # Number of vertices
        "faces": 2896,                 # Number of triangular faces  
        "volume": 46.86,               # Internal volume (if closed mesh)
        "surface_area": 168.65,        # Total surface area
        "seed": 2041269456,            # Random seed used
        "format": "obj",               # Export format
        "bounds": [[-2.5, -0.4, -4.5], [2.5, 4.0, 4.0]]  # Bounding box
    }
}
```

## Dependencies

```bash
pip install trimesh>=4.4.0 numpy>=1.26.0 shapely>=2.0.4 networkx>=3.0 rtree>=1.0.0
```

## Tips & Best Practices

1. **Start simple** - Use text prompts for quick prototyping
2. **Be specific** - Include dimensions, colors, and style keywords
3. **Use parts mode** - For precise control over complex models
4. **Enable optimization** - For cleaner meshes and smaller file sizes
5. **Check output stats** - Verify vertex/face counts for performance
6. **Experiment with formats** - OBJ for general use, STL for 3D printing, GLB for web

## Troubleshooting

- **Empty results**: Check that primitives have valid dimensions
- **Boolean failures**: Complex boolean operations may fall back to original shapes
- **Large files**: Use optimization and consider reducing segment counts
- **Color issues**: Ensure colors are in 0.0-1.0 range for proper rendering