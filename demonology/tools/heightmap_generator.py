# heightmap_generator.py
from __future__ import annotations
import math, os, random, re, time, logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing as mp
try:
    import numpy as np
    import trimesh
    from PIL import Image
except Exception:
    np = None; trimesh = None; Image = None

try:
    from numba import jit, njit, prange
    HAS_NUMBA = True
except Exception:
    # Fallback decorators that do nothing
    def jit(*args, **kwargs): 
        def decorator(func): return func
        return decorator
    def njit(*args, **kwargs): 
        def decorator(func): return func
        return decorator
    prange = range
    HAS_NUMBA = False

try:
    from .base import Tool
except Exception:
    class Tool:
        def __init__(self, name: str, description: str):
            self.name, self.description = name, description
        def to_openai_function(self) -> Dict[str, Any]: ...
        async def execute(self, **kwargs) -> Dict[str, Any]: ...

logger = logging.getLogger(__name__)

# JIT-compiled helper functions for performance
@njit(parallel=True, fastmath=True, cache=True)
def _generate_noise_chunk_numba(
    width: int,
    height: int, 
    offset_x: int,
    offset_y: int,
    total_width: int,
    total_height: int,
    octaves: int,
    frequency: float,
    amplitude: float,
    lacunarity: float,
    persistence: float
) -> np.ndarray:
    """JIT-compiled noise generation for a chunk"""
    
    chunk = np.zeros((height, width), dtype=np.float32)
    
    for octave in prange(octaves):
        octave_frequency = frequency * (lacunarity ** octave)
        octave_amplitude = amplitude * (persistence ** octave)
        
        for i in prange(height):
            for j in prange(width):
                # Global coordinates
                global_x = (offset_x + j) * octave_frequency / total_width
                global_y = (offset_y + i) * octave_frequency / total_height
                
                # Multi-wave noise
                noise_value = (
                    math.sin(global_x * 2 * math.pi) * math.cos(global_y * 2 * math.pi) +
                    math.sin(global_x * 4 * math.pi + 1.5) * math.cos(global_y * 4 * math.pi + 1.5) * 0.5 +
                    math.sin(global_x * 8 * math.pi + 3) * math.cos(global_y * 8 * math.pi + 3) * 0.25
                ) / 1.75
                
                chunk[i, j] += noise_value * octave_amplitude
    
    return chunk

@njit(parallel=True, fastmath=True, cache=True)
def _apply_mountain_shaping_numba(
    heightmap: np.ndarray,
    center_x: float,
    center_y: float,
    width: int,
    height: int
) -> np.ndarray:
    """JIT-compiled mountain terrain shaping"""
    
    result = np.zeros_like(heightmap)
    norm_factor = min(width, height)
    
    for i in prange(height):
        for j in prange(width):
            # Multiple peak centers with distance calculations
            dist1 = math.sqrt((j - center_x)**2 + (i - center_y)**2) / norm_factor
            dist2 = math.sqrt((j - center_x*0.3)**2 + (i - center_y*1.5)**2) / norm_factor
            dist3 = math.sqrt((j - center_x*1.7)**2 + (i - center_y*0.4)**2) / norm_factor
            
            mountain_factor = (
                math.exp(-dist1 * 3) * 0.8 + 
                math.exp(-dist2 * 4) * 0.6 +
                math.exp(-dist3 * 4) * 0.5
            )
            result[i, j] = heightmap[i, j] * 0.5 + mountain_factor
    
    return result

class HeightmapGeneratorTool(Tool):
    """Generate and convert heightmaps for 3D terrain generation
    
    Features:
    - Import PNG/JPG heightmaps and convert to 3D meshes
    - Generate procedural heightmaps (noise, fractals, terracing)
    - Export to game engine formats (Unity RAW, Unreal, OBJ/FBX)
    - Terrain optimization with LOD support
    - Multi-layer terrain with textures and materials
    """
    
    def __init__(self):
        super().__init__(
            "heightmap_generator", 
            "Generate procedural heightmaps and convert PNG/JPG heightmaps to 3D terrain meshes for game engines"
        )
    
    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string", 
                        "enum": ["convert", "generate", "analyze"],
                        "description": "Operation mode: convert existing heightmap, generate new one, or analyze existing"
                    },
                    "input_path": {
                        "type": "string",
                        "description": "Path to input heightmap image (PNG/JPG) for conversion or analysis"
                    },
                    "output_path": {
                        "type": "string", 
                        "description": "Output file path (auto-detects format from extension: .obj, .raw, .r16, .fbx)"
                    },
                    "width": {
                        "type": "integer",
                        "description": "Width in pixels for generated heightmaps (default: 2048, max: 16384)",
                        "default": 2048
                    },
                    "height": {
                        "type": "integer", 
                        "description": "Height in pixels for generated heightmaps (default: 2048, max: 16384)",
                        "default": 2048
                    },
                    "terrain_type": {
                        "type": "string",
                        "enum": ["mountains", "hills", "plains", "desert", "canyon", "islands", "valleys"],
                        "description": "Type of terrain to generate"
                    },
                    "scale": {
                        "type": "object",
                        "description": "Terrain scaling: {x: world_width, y: max_height, z: world_depth}",
                        "properties": {
                            "x": {"type": "number", "description": "World width in units"},
                            "y": {"type": "number", "description": "Maximum height in units"}, 
                            "z": {"type": "number", "description": "World depth in units"}
                        }
                    },
                    "resolution": {
                        "type": "integer",
                        "description": "Mesh resolution - higher = more detailed (default: 2, max: 8)"
                    },
                    "noise_params": {
                        "type": "object",
                        "description": "Noise generation parameters",
                        "properties": {
                            "octaves": {"type": "integer", "description": "Noise octaves (detail layers)"},
                            "frequency": {"type": "number", "description": "Base noise frequency"},
                            "amplitude": {"type": "number", "description": "Noise amplitude"},
                            "lacunarity": {"type": "number", "description": "Frequency multiplier per octave"},
                            "persistence": {"type": "number", "description": "Amplitude multiplier per octave"}
                        }
                    },
                    "features": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Terrain features to add: ['rivers', 'roads', 'plateaus', 'craters', 'ridges']"
                    },
                    "game_engine": {
                        "type": "string",
                        "enum": ["unity", "unreal", "godot", "blender", "generic"],
                        "description": "Target game engine for optimized export"
                    },
                    "generate_textures": {
                        "type": "boolean", 
                        "description": "Generate texture maps (diffuse, normal, splat) alongside heightmap"
                    },
                    "optimize": {
                        "type": "boolean",
                        "description": "Apply mesh optimization (decimation, smoothing)"
                    },
                    "tile_size": {
                        "type": "integer",
                        "description": "Size of individual tiles for massive terrains (default: 1024)"
                    },
                    "streaming": {
                        "type": "boolean",
                        "description": "Enable memory-efficient streaming for large heightmaps (default: auto)"
                    },
                    "lod_levels": {
                        "type": "integer",
                        "description": "Number of LOD levels to generate (1-5, default: 1)"
                    },
                    "max_vertices": {
                        "type": "integer", 
                        "description": "Maximum vertices per mesh (auto-tiles if exceeded, default: 1000000)"
                    },
                    "world_size": {
                        "type": "number",
                        "description": "Total world size in kilometers for massive worlds (auto-scales)"
                    }
                },
                "required": ["mode"]
            }
        }
    
    async def execute(
        self,
        mode: str,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        width: int = 2048,
        height: int = 2048,
        terrain_type: str = "mountains",
        scale: Optional[Dict[str, float]] = None,
        resolution: int = 2,
        noise_params: Optional[Dict[str, float]] = None,
        features: Optional[List[str]] = None,
        game_engine: str = "generic",
        generate_textures: bool = False,
        optimize: bool = False,
        tile_size: int = 1024,
        streaming: Optional[bool] = None,
        lod_levels: int = 1,
        max_vertices: int = 1000000,
        world_size: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        
        if not np or not trimesh or not Image:
            return {"success": False, "error": "Missing dependencies: numpy, trimesh, PIL required"}
        
        # Auto-enable streaming for large terrains
        if streaming is None:
            streaming = (width * height > 4096 * 4096)  # 16MP+ threshold
        
        # Auto-scale world size if provided
        if world_size is not None:
            if scale is None:
                scale = {"x": world_size * 1000, "y": world_size * 50, "z": world_size * 1000}
        
        # Validate massive terrain parameters
        if width > 16384 or height > 16384:
            return {"success": False, "error": "Maximum size is 16384x16384. Use tile_size for larger worlds."}
        
        try:
            if mode == "convert":
                return await self._convert_heightmap_massive(
                    input_path, output_path, scale, resolution, game_engine, 
                    generate_textures, optimize, streaming, lod_levels, max_vertices, tile_size
                )
            elif mode == "generate":
                return await self._generate_heightmap_massive(
                    width, height, terrain_type, output_path, scale, noise_params,
                    features, game_engine, generate_textures, optimize, streaming, 
                    lod_levels, max_vertices, tile_size, world_size
                )
            elif mode == "analyze":
                return await self._analyze_heightmap(input_path)
            else:
                return {"success": False, "error": f"Invalid mode: {mode}"}
                
        except Exception as e:
            logger.exception("Heightmap operation failed")
            return {"success": False, "error": str(e)}
    
    async def _convert_heightmap(
        self, 
        input_path: str, 
        output_path: Optional[str],
        scale: Optional[Dict[str, float]],
        resolution: int,
        game_engine: str,
        generate_textures: bool,
        optimize: bool
    ) -> Dict[str, Any]:
        """Convert PNG/JPG heightmap to 3D terrain mesh"""
        
        if not input_path:
            return {"success": False, "error": "input_path required for convert mode"}
        
        input_file = Path(input_path)
        if not input_file.exists():
            return {"success": False, "error": f"Input file not found: {input_path}"}
        
        # Load heightmap image
        try:
            # Temporarily increase PIL's decompression bomb limit for massive heightmaps
            old_max_pixels = Image.MAX_IMAGE_PIXELS
            Image.MAX_IMAGE_PIXELS = None  # Disable limit
            
            img = Image.open(input_file).convert('L')  # Convert to grayscale
            width, height = img.size
            
            # Restore original limit
            Image.MAX_IMAGE_PIXELS = old_max_pixels
            
            heightmap = np.array(img, dtype=np.float32) / 255.0  # Normalize to 0-1
        except Exception as e:
            # Restore original limit on error
            Image.MAX_IMAGE_PIXELS = old_max_pixels
            return {"success": False, "error": f"Failed to load heightmap image: {e}"}
        
        # Set default scale if not provided
        if not scale:
            scale = {"x": width * 0.1, "y": 50.0, "z": height * 0.1}
        
        # Generate output path if not provided
        if not output_path:
            output_path = str(input_file.with_suffix('.obj'))
        
        # Convert heightmap to mesh
        mesh_result = self._heightmap_to_mesh(heightmap, scale, resolution)
        if not mesh_result["success"]:
            return mesh_result
        
        mesh = mesh_result["mesh"]
        
        # Apply optimization
        if optimize:
            mesh = self._optimize_mesh(mesh)
        
        # Export mesh
        export_result = self._export_mesh(mesh, output_path, game_engine)
        if not export_result["success"]:
            return export_result
        
        # Generate texture maps if requested
        texture_paths = []
        if generate_textures:
            texture_result = self._generate_texture_maps(heightmap, Path(output_path).stem, scale)
            if texture_result["success"]:
                texture_paths = texture_result["paths"]
        
        return {
            "success": True,
            "mode": "convert",
            "input_path": str(input_file),
            "output_path": output_path,
            "mesh_stats": {
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "bounds": mesh.bounds.tolist(),
                "volume": mesh.volume if mesh.is_volume else 0
            },
            "heightmap_info": {
                "dimensions": [width, height],
                "min_height": float(heightmap.min()),
                "max_height": float(heightmap.max()),
                "scale": scale
            },
            "texture_maps": texture_paths,
            "game_engine": game_engine
        }
    
    async def _generate_heightmap(
        self,
        width: int,
        height: int, 
        terrain_type: str,
        output_path: Optional[str],
        scale: Optional[Dict[str, float]],
        noise_params: Optional[Dict[str, float]],
        features: Optional[List[str]],
        game_engine: str,
        generate_textures: bool,
        optimize: bool
    ) -> Dict[str, Any]:
        """Generate procedural heightmap and convert to mesh"""
        
        # Set default parameters
        if not scale:
            scale = {"x": width * 0.1, "y": 100.0, "z": height * 0.1}
        
        if not noise_params:
            noise_params = self._get_default_noise_params(terrain_type)
        
        if not output_path:
            output_path = f"heightmaps/generated_{terrain_type}_{width}x{height}.obj"
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Generate heightmap
        heightmap = self._generate_procedural_heightmap(
            width, height, terrain_type, noise_params, features or []
        )
        
        # Save heightmap as PNG
        heightmap_path = str(Path(output_path).with_suffix('.png'))
        heightmap_img = (heightmap * 255).astype(np.uint8)
        Image.fromarray(heightmap_img, 'L').save(heightmap_path)
        
        # Convert to mesh
        mesh_result = self._heightmap_to_mesh(heightmap, scale, 2)
        if not mesh_result["success"]:
            return mesh_result
        
        mesh = mesh_result["mesh"]
        
        # Apply optimization
        if optimize:
            mesh = self._optimize_mesh(mesh)
        
        # Export mesh
        export_result = self._export_mesh(mesh, output_path, game_engine)
        if not export_result["success"]:
            return export_result
        
        # Generate texture maps
        texture_paths = []
        if generate_textures:
            texture_result = self._generate_texture_maps(heightmap, Path(output_path).stem, scale)
            if texture_result["success"]:
                texture_paths = texture_result["paths"]
        
        return {
            "success": True,
            "mode": "generate",
            "terrain_type": terrain_type,
            "heightmap_path": heightmap_path,
            "output_path": output_path,
            "mesh_stats": {
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces), 
                "bounds": mesh.bounds.tolist(),
                "volume": mesh.volume if mesh.is_volume else 0
            },
            "generation_params": {
                "dimensions": [width, height],
                "scale": scale,
                "noise_params": noise_params,
                "features": features or []
            },
            "texture_maps": texture_paths,
            "game_engine": game_engine
        }
    
    async def _analyze_heightmap(self, input_path: str) -> Dict[str, Any]:
        """Analyze existing heightmap and return statistics"""
        
        if not input_path:
            return {"success": False, "error": "input_path required for analyze mode"}
        
        input_file = Path(input_path)
        if not input_file.exists():
            return {"success": False, "error": f"Input file not found: {input_path}"}
        
        try:
            # Temporarily increase PIL's decompression bomb limit for massive heightmaps
            old_max_pixels = Image.MAX_IMAGE_PIXELS
            Image.MAX_IMAGE_PIXELS = None  # Disable limit
            
            img = Image.open(input_file).convert('L')
            width, height = img.size
            
            # Restore original limit
            Image.MAX_IMAGE_PIXELS = old_max_pixels
            
            heightmap = np.array(img, dtype=np.float32) / 255.0
        except Exception as e:
            # Restore original limit on error
            Image.MAX_IMAGE_PIXELS = old_max_pixels
            return {"success": False, "error": f"Failed to load heightmap: {e}"}
        
        # Calculate statistics
        stats = {
            "dimensions": [width, height],
            "min_height": float(heightmap.min()),
            "max_height": float(heightmap.max()),
            "mean_height": float(heightmap.mean()),
            "height_range": float(heightmap.max() - heightmap.min()),
            "rough_estimate": self._estimate_terrain_type(heightmap),
            "recommended_scale": self._recommend_scale(heightmap, width, height)
        }
        
        # Analyze terrain features
        features = self._analyze_terrain_features(heightmap)
        
        return {
            "success": True,
            "mode": "analyze", 
            "input_path": str(input_file),
            "heightmap_stats": stats,
            "terrain_features": features,
            "conversion_suggestions": {
                "recommended_resolution": 2 if max(width, height) > 1024 else 1,
                "needs_optimization": max(width, height) > 2048,
                "suitable_for_games": height < 4096 and width < 4096
            }
        }
    
    def _heightmap_to_mesh(
        self, 
        heightmap: np.ndarray, 
        scale: Dict[str, float], 
        resolution: int
    ) -> Dict[str, Any]:
        """Convert 2D heightmap array to 3D mesh"""
    
        # Patch: interpret `resolution` as downsample factor (1=full, 2=half, 4=quarter)
    
        try:
    
            step = max(1, int(resolution))
    
        except Exception:
    
            step = 1
    
        if step > 1:
    
            heightmap = heightmap[::step, ::step]
    
        

        
        try:
            height, width = heightmap.shape
            
            # Downsample if resolution < 1 (for performance)
            if resolution < 1:
                step = int(1 / resolution)
                heightmap = heightmap[::step, ::step]
                height, width = heightmap.shape
            
            # Calculate world coordinates
            x_scale = scale["x"] / (width - 1)
            z_scale = scale["z"] / (height - 1) 
            y_scale = scale["y"]
            
            # Use optimized mesh generation based on size
            if height * width > 50000:  # Use parallel processing for large meshes
                return self._heightmap_to_mesh_parallel(heightmap, scale, resolution)
            
            # Generate vertices using vectorized operations
            i_coords, j_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
            
            # Vectorized vertex calculation
            x = j_coords * x_scale - scale["x"] / 2  # Center on origin
            z = i_coords * z_scale - scale["z"] / 2
            y = heightmap * y_scale
            
            # Stack coordinates and reshape to vertex array
            vertices = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
            
            # Generate faces using vectorized operations
            faces = self._generate_faces_vectorized(height, width)
            
            # Create mesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Generate UV coordinates using vectorized operations
            uvs = np.stack([
                j_coords.ravel() / (width - 1),
                i_coords.ravel() / (height - 1)
            ], axis=1)
            
            # Add UV coordinates to mesh
            try:
                mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
            except:
                pass  # UV mapping optional
            
            return {"success": True, "mesh": mesh}
            
        except Exception as e:
            return {"success": False, "error": f"Mesh generation failed: {e}"}
    
    def _generate_faces_vectorized(self, height: int, width: int) -> np.ndarray:
        """Generate mesh faces using vectorized operations"""
        
        # Create index arrays for quads
        i_indices = np.arange(height - 1)
        j_indices = np.arange(width - 1)
        
        # Create meshgrid of quad indices
        i_grid, j_grid = np.meshgrid(i_indices, j_indices, indexing='ij')
        
        # Calculate vertex indices for each quad
        v0 = i_grid * width + j_grid
        v1 = i_grid * width + (j_grid + 1)
        v2 = (i_grid + 1) * width + (j_grid + 1)
        v3 = (i_grid + 1) * width + j_grid
        
        # Create triangular faces (2 per quad)
        faces1 = np.stack([v0.ravel(), v1.ravel(), v2.ravel()], axis=1)
        faces2 = np.stack([v0.ravel(), v2.ravel(), v3.ravel()], axis=1)
        
        # Combine all faces
        faces = np.vstack([faces1, faces2])
        return faces
    
    def _heightmap_to_mesh_parallel(
        self, 
        heightmap: np.ndarray, 
        scale: Dict[str, float], 
        resolution: int
    ) -> Dict[str, Any]:
        """Convert large heightmap to mesh using parallel processing"""
        
        try:
            height, width = heightmap.shape
            
            # Calculate world coordinates
            x_scale = scale["x"] / (width - 1)
            z_scale = scale["z"] / (height - 1) 
            y_scale = scale["y"]
            
            # Determine optimal chunk size for parallel processing
            num_cores = min(mp.cpu_count(), 8)  # Limit to 8 cores max
            chunk_size = max(100, height // num_cores)
            
            # Split heightmap into chunks for parallel processing
            chunks = []
            for i in range(0, height, chunk_size):
    
        # Patch: UVs based on world x/z extents (0..1)
    
        try:
    
            import numpy as _np
    
            uvs = _np.empty((len(vertices), 2), dtype=_np.float32)
    
            denom_x = max(1e-9, scale.get('x', 1.0))
    
            denom_z = max(1e-9, scale.get('z', 1.0))
    
            uvs[:, 0] = (vertices[:, 0] + denom_x / 2.0) / denom_x
    
            uvs[:, 1] = (vertices[:, 2] + denom_z / 2.0) / denom_z
    
            mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
    
        except Exception:
    
            pass
    
        
                end_i = min(i + chunk_size, height)
                chunk_data = {
                    'heightmap_chunk': heightmap[i:end_i, :],
                    'start_row': i,
                    'x_scale': x_scale,
                    'z_scale': z_scale,
                    'y_scale': y_scale,
                    'scale': scale,
                    'width': width
                }
                chunks.append(chunk_data)
            
            # Process chunks in parallel
            with ThreadPoolExecutor(max_workers=num_cores) as executor:
                chunk_results = list(executor.map(self._process_mesh_chunk, chunks))
            
            # Combine results
            all_vertices = []
            all_faces = []
            vertex_offset = 0
            
            for vertices, faces in chunk_results:
                all_vertices.append(vertices)
                # Adjust face indices for vertex offset
                adjusted_faces = faces + vertex_offset
                all_faces.append(adjusted_faces)
                vertex_offset += len(vertices)
            
            # Combine all vertices and faces
            vertices = np.vstack(all_vertices)
            faces = np.vstack(all_faces)
            
            # Create mesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            
    
        # Patch: UVs based on world x/z extents (0..1)
    
        try:
    
            import numpy as _np
    
            uvs = _np.empty((len(vertices), 2), dtype=_np.float32)
    
            denom_x = max(1e-9, scale.get('x', 1.0))
    
            denom_z = max(1e-9, scale.get('z', 1.0))
    
            uvs[:, 0] = (vertices[:, 0] + denom_x / 2.0) / denom_x
    
            uvs[:, 1] = (vertices[:, 2] + denom_z / 2.0) / denom_z
    
            mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
    
        except Exception:
    
            pass
    
        
# Generate UV coordinates
            total_vertices = len(vertices)
            uvs = np.zeros((total_vertices, 2))
            idx = 0
            for i in range(height):
                for j in range(width):
                    if idx < total_vertices:
                        uvs[idx] = [j / (width - 1), i / (height - 1)]
                        idx += 1
            
            # Add UV coordinates to mesh
            try:
                mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
            except:
                pass  # UV mapping optional
            
            return {"success": True, "mesh": mesh}
            
        except Exception as e:
            return {"success": False, "error": f"Parallel mesh generation failed: {e}"}
    
    def _process_mesh_chunk(self, chunk_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Process a chunk of heightmap into vertices and faces"""
        
        heightmap_chunk = chunk_data['heightmap_chunk']
        start_row = chunk_data['start_row']
        x_scale = chunk_data['x_scale']
        z_scale = chunk_data['z_scale']
        y_scale = chunk_data['y_scale']
        scale = chunk_data['scale']
        total_width = chunk_data['width']
        
        chunk_height, chunk_width = heightmap_chunk.shape
        
        # Generate vertices for this chunk
        i_coords, j_coords = np.meshgrid(
            np.arange(chunk_height) + start_row, 
            np.arange(chunk_width), 
            indexing='ij'
        )
        
        # Vectorized vertex calculation
        x = j_coords * x_scale - scale["x"] / 2
        z = i_coords * z_scale - scale["z"] / 2
        y = heightmap_chunk * y_scale
        
        vertices = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
        
        # Generate faces for this chunk (only if not the last row chunk)
        if start_row + chunk_height < heightmap_chunk.shape[0] + start_row:
            faces = self._generate_faces_vectorized(chunk_height, chunk_width)
        else:
            # For the last chunk, generate faces only for complete quads
            if chunk_height > 1:
                faces = self._generate_faces_vectorized(chunk_height, chunk_width)
            else:
                faces = np.array([]).reshape(0, 3)
        
        return vertices, faces
    
    
    def _generate_procedural_heightmap(
        self,
        width: int,
        height: int,
        terrain_type: str,
        noise_params: Dict[str, float],
        features: List[str]
    ) -> np.ndarray:
        """Generate procedural heightmap using noise and terrain features"""
        
        # Initialize heightmap
        heightmap = np.zeros((height, width), dtype=np.float32)
        
        # Generate base noise
        heightmap = self._generate_noise(
            width, height, 
            noise_params["octaves"],
            noise_params["frequency"],
            noise_params["amplitude"], 
            noise_params["lacunarity"],
            noise_params["persistence"]
        )
        
        # Apply terrain-specific shaping
        heightmap = self._apply_terrain_shaping(heightmap, terrain_type)
        
        # Add terrain features
        for feature in features:
            heightmap = self._add_terrain_feature(heightmap, feature)
        
        # Normalize to 0-1 range
        heightmap = np.clip(heightmap, 0, 1)
        
        return heightmap
    
    def _generate_noise(
        self,
        width: int, 
        height: int,
        octaves: int,
        frequency: float,
        amplitude: float,
        lacunarity: float,
        persistence: float
    ) -> np.ndarray:
        """Generate multi-octave Perlin-like noise using vectorized operations"""
        
        heightmap = np.zeros((height, width), dtype=np.float32)
        
        # Pre-compute coordinate meshgrids for vectorization
        j_coords, i_coords = np.meshgrid(np.arange(width), np.arange(height))
        
        for octave in range(octaves):
            octave_frequency = frequency * (lacunarity ** octave)
            octave_amplitude = amplitude * (persistence ** octave)
            
            # Vectorized coordinate calculations
            x = j_coords * octave_frequency / width
            y = i_coords * octave_frequency / height
            
            # Vectorized multi-wave noise generation
            noise_layer = (
                np.sin(x * 2 * np.pi) * np.cos(y * 2 * np.pi) +
                np.sin(x * 4 * np.pi + 1.5) * np.cos(y * 4 * np.pi + 1.5) * 0.5 +
                np.sin(x * 8 * np.pi + 3) * np.cos(y * 8 * np.pi + 3) * 0.25
            ) / 1.75
            
            # Add octave contribution
            heightmap += noise_layer * octave_amplitude
        
        return heightmap
    
    def _apply_terrain_shaping(self, heightmap: np.ndarray, terrain_type: str) -> np.ndarray:
        """Apply terrain-specific shaping to heightmap using vectorized operations"""
        
        height, width = heightmap.shape
        center_x, center_y = width // 2, height // 2
        
        # Pre-compute coordinate meshgrids for vectorization
        j_coords, i_coords = np.meshgrid(np.arange(width), np.arange(height))
        
        if terrain_type == "mountains":
            # Use JIT-compiled version if available, otherwise vectorized version
            if HAS_NUMBA and width * height > 10000:
                heightmap = _apply_mountain_shaping_numba(
                    heightmap, float(center_x), float(center_y), width, height
                )
            else:
                # Vectorized mountain peak generation
                norm_factor = min(width, height)
                
                # Multiple peak centers with vectorized distance calculations
                dist1 = np.sqrt((j_coords - center_x)**2 + (i_coords - center_y)**2) / norm_factor
                dist2 = np.sqrt((j_coords - center_x*0.3)**2 + (i_coords - center_y*1.5)**2) / norm_factor
                dist3 = np.sqrt((j_coords - center_x*1.7)**2 + (i_coords - center_y*0.4)**2) / norm_factor
                
                mountain_factor = (
                    np.exp(-dist1 * 3) * 0.8 + 
                    np.exp(-dist2 * 4) * 0.6 +
                    np.exp(-dist3 * 4) * 0.5
                )
                heightmap = heightmap * 0.5 + mountain_factor
        
        elif terrain_type == "hills":
            # Already vectorized - no change needed
            heightmap = heightmap * 0.6 + 0.2
            
        elif terrain_type == "plains":
            # Already vectorized - no change needed
            heightmap = heightmap * 0.3 + 0.1
            
        elif terrain_type == "canyon":
            # Vectorized canyon generation
            center_line_dist = np.abs(j_coords - center_x) / width
            canyon_depth = np.exp(-center_line_dist * 8) * 0.8
            heightmap = heightmap * 0.7 + 0.3 - canyon_depth
        
        elif terrain_type == "islands":
            # Vectorized island shaping
            edge_dist = np.minimum(
                np.minimum(i_coords, j_coords),
                np.minimum(height - 1 - i_coords, width - 1 - j_coords)
            ) / min(width, height) * 2
            island_factor = np.tanh(edge_dist * 3) * 0.8
            heightmap = heightmap * island_factor
        
        elif terrain_type == "desert":
            # Vectorized sand dunes
            dune_wave = np.sin(j_coords * 0.02) * np.cos(i_coords * 0.015) * 0.3
            heightmap = heightmap * 0.4 + dune_wave + 0.2
        
        elif terrain_type == "valleys":
            # Vectorized valley system
            valley_x = np.abs(np.sin(j_coords * 0.01)) * 0.5
            valley_y = np.abs(np.sin(i_coords * 0.008)) * 0.4
            heightmap = heightmap * 0.6 + valley_x + valley_y
        
        return heightmap
    
    def _add_terrain_feature(self, heightmap: np.ndarray, feature: str) -> np.ndarray:
        """Add specific terrain features to heightmap"""
        
        height, width = heightmap.shape
        
        if feature == "rivers":
            # Add meandering river
            river_y = height // 2
            for j in range(width):
                river_offset = int(math.sin(j * 0.02) * height * 0.2)
                river_center = river_y + river_offset
                
                # River depression
                for i in range(max(0, river_center - 10), min(height, river_center + 10)):
                    river_depth = math.exp(-((i - river_center) / 5)**2) * 0.3
                    if 0 <= i < height:
                        heightmap[i, j] -= river_depth
        
        elif feature == "roads":
            # Add winding road
            for j in range(0, width, 20):
                road_height = int(height * (0.3 + 0.4 * math.sin(j * 0.01)))
                # Flatten area for road
                for i in range(max(0, road_height - 3), min(height, road_height + 3)):
                    if j < width:
                        heightmap[i, j] = heightmap[i, j] * 0.5 + heightmap[road_height, j] * 0.5
        
        elif feature == "plateaus": 
            # Add flat plateaus
            center_x, center_y = width // 3, height // 3
            plateau_radius = min(width, height) // 8
            plateau_height = 0.7
            
            for i in range(height):
                for j in range(width):
                    dist = math.sqrt((j - center_x)**2 + (i - center_y)**2)
                    if dist < plateau_radius:
                        heightmap[i, j] = max(heightmap[i, j], plateau_height)
        
        elif feature == "craters":
            # Add impact craters
            crater_x, crater_y = width // 4, height * 3 // 4
            crater_radius = min(width, height) // 10
            
            for i in range(height):
                for j in range(width):
                    dist = math.sqrt((j - crater_x)**2 + (i - crater_y)**2)
                    if dist < crater_radius:
                        crater_depth = (1 - dist / crater_radius) * 0.4
                        heightmap[i, j] -= crater_depth
        
        elif feature == "ridges":
            # Add mountain ridges
            for i in range(0, height, height // 4):
                for j in range(width):
                    ridge_height = 0.6 + 0.2 * math.sin(j * 0.03)
                    # Create ridge line
                    for k in range(max(0, i - 5), min(height, i + 5)):
                        ridge_factor = math.exp(-((k - i) / 3)**2)
                        heightmap[k, j] = max(heightmap[k, j], ridge_height * ridge_factor)
        
        return heightmap
    
    def _get_default_noise_params(self, terrain_type: str) -> Dict[str, float]:
        """Get default noise parameters for terrain type"""
        
        params = {
            "mountains": {"octaves": 6, "frequency": 0.01, "amplitude": 0.8, "lacunarity": 2.0, "persistence": 0.5},
            "hills": {"octaves": 4, "frequency": 0.02, "amplitude": 0.6, "lacunarity": 2.0, "persistence": 0.6},
            "plains": {"octaves": 3, "frequency": 0.05, "amplitude": 0.3, "lacunarity": 2.0, "persistence": 0.4},
            "desert": {"octaves": 4, "frequency": 0.03, "amplitude": 0.5, "lacunarity": 1.8, "persistence": 0.5},
            "canyon": {"octaves": 5, "frequency": 0.015, "amplitude": 0.7, "lacunarity": 2.2, "persistence": 0.4},
            "islands": {"octaves": 5, "frequency": 0.02, "amplitude": 0.6, "lacunarity": 2.0, "persistence": 0.5},
            "valleys": {"octaves": 4, "frequency": 0.025, "amplitude": 0.5, "lacunarity": 1.9, "persistence": 0.55}
        }
        
        return params.get(terrain_type, params["mountains"])
    
    def _optimize_mesh(self, mesh: 'trimesh.Trimesh') -> 'trimesh.Trimesh':
        """Apply mesh optimization techniques"""
        
        try:
            # Remove duplicate vertices
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
            
            # Fix mesh issues
            mesh.fix_normals()
            mesh.fill_holes()
            
            # Optional: Simplify mesh (commented out to preserve detail)
            # if len(mesh.vertices) > 10000:
            #     mesh = mesh.simplify_quadric_decimation(5000)
            
        except Exception as e:
            logger.warning(f"Mesh optimization partially failed: {e}")
        
        return mesh
    
    def _export_mesh(self, mesh: 'trimesh.Trimesh', output_path: str, game_engine: str) -> Dict[str, Any]:
        """Export mesh to various formats for game engines"""
        
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            ext = output_file.suffix.lower()
            
            if ext == '.obj':
                # Wavefront OBJ - Universal format
                mesh.export(output_path)
                
            elif ext == '.stl':
                # STL format
                mesh.export(output_path)
                
            elif ext == '.ply':
                # PLY format
                mesh.export(output_path)
                
            elif ext in ['.glb', '.gltf']:
                # glTF format for web/modern engines
                mesh.export(output_path)
                
            elif ext == '.raw':
                # Unity RAW heightfield format
                return self._export_unity_raw(mesh, output_path)
                
            elif ext == '.r16':
                # 16-bit RAW format
                return self._export_r16(mesh, output_path)
                
            else:
                # Default to OBJ
                obj_path = str(output_file.with_suffix('.obj'))
                mesh.export(obj_path)
                output_path = obj_path
            
            return {"success": True, "output_path": output_path}
            
        except Exception as e:
            return {"success": False, "error": f"Export failed: {e}"}
    
    def _export_unity_raw(self, mesh: 'trimesh.Trimesh', output_path: str) -> Dict[str, Any]:
        """Export Unity-compatible RAW heightfield"""
        try:
            # Extract height values from mesh vertices
            vertices = mesh.vertices
            y_values = vertices[:, 1]  # Y coordinates = height
            
            # Determine grid size (assuming square grid)
            unique_x = len(np.unique(vertices[:, 0]))
            unique_z = len(np.unique(vertices[:, 2]))
            
            # Reshape to heightmap grid
            heightmap = y_values.reshape(unique_z, unique_x)
            
            # Normalize to Unity's expected range (0-65535)
            min_height, max_height = heightmap.min(), heightmap.max()
            if max_height > min_height:
                normalized = ((heightmap - min_height) / (max_height - min_height) * 65535).astype(np.uint16)
            else:
                normalized = np.zeros_like(heightmap, dtype=np.uint16)
            
            # Save as binary RAW file
            with open(output_path, 'wb') as f:
                normalized.tobytes()
                f.write(normalized.tobytes())
            
            return {
                "success": True,
                "output_path": output_path,
                "unity_info": {
                    "resolution": [unique_x, unique_z],
                    "height_range": [float(min_height), float(max_height)]
                }
            }
            
        except Exception as e:
            return {"success": False, "error": f"Unity RAW export failed: {e}"}
    
    def _export_r16(self, mesh: 'trimesh.Trimesh', output_path: str) -> Dict[str, Any]:
        """Export 16-bit RAW format"""
        # Similar to Unity RAW but different byte order
        return self._export_unity_raw(mesh, output_path)
    
    def _generate_texture_maps(
        self, 
        heightmap: np.ndarray, 
        basename: str,
        scale: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate texture maps based on heightmap"""
        
        try:
            height, width = heightmap.shape
            paths = []
            
            # Diffuse map (color based on height)
            diffuse = np.zeros((height, width, 3), dtype=np.uint8)
            for i in range(height):
    
        import numpy as np
    
        h, w = heightmap.shape
    
        # Diffuse map
    
        diffuse = np.zeros((h, w, 3), dtype=np.uint8)
    
        m1 = heightmap < 0.3
    
        m2 = (heightmap >= 0.3) & (heightmap < 0.5)
    
        m3 = (heightmap >= 0.5) & (heightmap < 0.8)
    
        m4 = heightmap >= 0.8
    
        diffuse[m1] = [50, 100, 200]
    
        diffuse[m2] = [100, 180, 80]
    
        diffuse[m3] = [140, 120, 80]
    
        diffuse[m4] = [200, 200, 220]
    
        
    
        # Normal map via gradient
    
        dz, dx = np.gradient(heightmap.astype(np.float32))
    
        scale_x = scale.get('y', 1.0) / max(1e-9, scale.get('x', 1.0))
    
        scale_z = scale.get('y', 1.0) / max(1e-9, scale.get('z', 1.0))
    
        dx *= scale_x; dz *= scale_z
    
        normal = np.dstack((-dx, np.full_like(dx, 2.0), -dz))
    
        norm = np.linalg.norm(normal, axis=2, keepdims=True)
    
        normal = normal / np.maximum(norm, 1e-9)
    
        normal_map = ((normal + 1.0) * 127.5).astype(np.uint8)
    
        
    
        # Splat map
    
        splat = np.zeros((h, w, 4), dtype=np.uint8)
    
        s1 = heightmap < 0.4
    
        s2 = (heightmap >= 0.4) & (heightmap < 0.7)
    
        s3 = (heightmap >= 0.7) & (heightmap < 0.9)
    
        s4 = heightmap >= 0.9
    
        splat[s1] = [255, 50, 0, 0]
    
        splat[s2] = [100, 255, 100, 0]
    
        splat[s3] = [0, 100, 255, 0]
    
        splat[s4] = [0, 0, 50, 255]
    
        return {'diffuse': diffuse, 'normal': normal_map, 'splat': splat}
    
        
    
    def _calculate_normal_map(self, heightmap: np.ndarray, scale: Dict[str, float]) -> np.ndarray:
        """Calculate normal map from heightmap gradients"""
        
        height, width = heightmap.shape
        normal_map = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Scale factors for gradient calculation
        scale_x = scale["y"] / scale["x"]  # Height per unit distance
        scale_z = scale["y"] / scale["z"]
        
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                # Calculate gradients using neighboring pixels
                dx = (heightmap[i, j + 1] - heightmap[i, j - 1]) * scale_x
                dz = (heightmap[i + 1, j] - heightmap[i - 1, j]) * scale_z
                
                # Calculate normal vector
                normal = np.array([-dx, 2.0, -dz])  # 2.0 controls normal intensity
                normal = normal / np.linalg.norm(normal)  # Normalize
                
                # Convert to 0-255 range (with 128 as neutral)
                normal_map[i, j] = ((normal + 1) * 127.5).astype(np.uint8)
        
        return normal_map
    
    def _estimate_terrain_type(self, heightmap: np.ndarray) -> str:
        """Estimate terrain type from heightmap characteristics"""
        
        height_range = heightmap.max() - heightmap.min()
        mean_height = heightmap.mean()
        std_height = heightmap.std()
        
        if height_range > 0.7 and std_height > 0.2:
            return "mountains"
        elif height_range > 0.4 and std_height > 0.15:
            return "hills"  
        elif height_range < 0.2:
            return "plains"
        elif mean_height < 0.3:
            return "valleys"
        else:
            return "mixed"
    
    def _recommend_scale(self, heightmap: np.ndarray, width: int, height: int) -> Dict[str, float]:
        """Recommend appropriate world scale for heightmap"""
        
        # Base scale on heightmap dimensions and height range
        height_range = heightmap.max() - heightmap.min()
        
        world_width = width * 2  # 2 units per pixel
        world_depth = height * 2
        world_height = height_range * 100  # Scale height variation
        
        return {
            "x": world_width,
            "y": world_height,
            "z": world_depth
        }
    
    def _analyze_terrain_features(self, heightmap: np.ndarray) -> Dict[str, Any]:
        """Analyze heightmap for terrain features"""
        
        features = {
            "has_water_areas": bool(np.sum(heightmap < 0.2) > len(heightmap.flat) * 0.1),
            "has_mountains": bool(np.sum(heightmap > 0.8) > len(heightmap.flat) * 0.05),
            "has_flat_areas": bool(np.sum(np.abs(heightmap - heightmap.mean()) < 0.1) > len(heightmap.flat) * 0.2),
            "height_variation": "high" if heightmap.std() > 0.2 else "medium" if heightmap.std() > 0.1 else "low",
            "dominant_elevation": "high" if heightmap.mean() > 0.6 else "medium" if heightmap.mean() > 0.4 else "low"
        }
        
        return features
    
    # =============== MASSIVE TERRAIN METHODS ===============
    
    async def _convert_heightmap_massive(
        self, 
        input_path: str, 
        output_path: Optional[str],
        scale: Optional[Dict[str, float]],
        resolution: int,
        game_engine: str,
        generate_textures: bool,
        optimize: bool,
        streaming: bool,
        lod_levels: int,
        max_vertices: int,
        tile_size: int
    ) -> Dict[str, Any]:
        """Convert massive PNG/JPG heightmap using streaming and tiling"""
        
        if not input_path:
            return {"success": False, "error": "input_path required for convert mode"}
        
        input_file = Path(input_path)
        if not input_file.exists():
            return {"success": False, "error": f"Input file not found: {input_path}"}
        
        # Get image dimensions without loading full image into memory
        try:
            # Temporarily increase PIL's decompression bomb limit for massive heightmaps
            old_max_pixels = Image.MAX_IMAGE_PIXELS
            Image.MAX_IMAGE_PIXELS = None  # Disable limit
            
            with Image.open(input_file) as img:
                width, height = img.size
                img_format = img.format
                
            # Restore original limit
            Image.MAX_IMAGE_PIXELS = old_max_pixels
        except Exception as e:
            # Restore original limit on error
            Image.MAX_IMAGE_PIXELS = old_max_pixels
            return {"success": False, "error": f"Failed to read heightmap header: {e}"}
        
        logger.info(f"Processing massive heightmap: {width}x{height} ({width*height/1000000:.1f}MP)")
        
        # Set default scale for massive terrain
        if not scale:
            # Auto-scale based on size - larger heightmaps get larger world scales
            world_scale = max(100, min(50000, width * 0.5))  # 0.5 units per pixel
            scale = {"x": world_scale, "y": world_scale * 0.1, "z": world_scale}
        
        # Generate output path if not provided
        if not output_path:
            output_path = str(input_file.with_suffix('.obj'))
        
        output_dir = Path(output_path).parent
        output_stem = Path(output_path).stem
        
        # Check if we need tiling based on size or vertex count
        estimated_vertices = width * height
        needs_tiling = estimated_vertices > max_vertices or streaming
        
        if needs_tiling:
            return await self._convert_heightmap_tiled(
                input_file, output_dir, output_stem, width, height, scale, 
                tile_size, lod_levels, game_engine, generate_textures, optimize
            )
        else:
            # Use optimized single-file processing
            return await self._convert_heightmap_optimized(
                input_file, output_path, scale, resolution, lod_levels,
                game_engine, generate_textures, optimize
            )
    
    async def _generate_heightmap_massive(
        self,
        width: int,
        height: int, 
        terrain_type: str,
        output_path: Optional[str],
        scale: Optional[Dict[str, float]],
        noise_params: Optional[Dict[str, float]],
        features: Optional[List[str]],
        game_engine: str,
        generate_textures: bool,
        optimize: bool,
        streaming: bool,
        lod_levels: int,
        max_vertices: int,
        tile_size: int,
        world_size: Optional[float]
    ) -> Dict[str, Any]:
        """Generate massive procedural heightmap with streaming support"""
        
        logger.info(f"Generating massive terrain: {width}x{height} ({width*height/1000000:.1f}MP)")
        
        # Set default parameters for massive terrain
        if not scale:
            if world_size:
                scale = {"x": world_size * 1000, "y": world_size * 50, "z": world_size * 1000}
            else:
                # Auto-scale based on resolution
                world_scale = max(1000, width * 2)  # 2 units per pixel minimum
                scale = {"x": world_scale, "y": world_scale * 0.05, "z": world_scale}
        
        if not noise_params:
            noise_params = self._get_default_noise_params(terrain_type)
            # Adjust for massive terrain - more octaves for detail
            noise_params["octaves"] = min(8, noise_params["octaves"] + 2)
        
        if not output_path:
            size_desc = f"{width//1000}K" if width >= 1000 else str(width)
            output_path = f"massive_terrain/{terrain_type}_{size_desc}x{height//1000 if height >= 1000 else height}.obj"
        
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_stem = Path(output_path).stem
        
        # Check if we need streaming generation
        estimated_vertices = width * height
        needs_streaming = estimated_vertices > max_vertices or streaming
        
        if needs_streaming:
            return await self._generate_heightmap_streaming(
                width, height, terrain_type, output_dir, output_stem, scale,
                noise_params, features or [], tile_size, lod_levels, 
                game_engine, generate_textures, optimize
            )
        else:
            # Use optimized single-generation
            return await self._generate_heightmap_optimized(
                width, height, terrain_type, output_path, scale, noise_params,
                features or [], lod_levels, game_engine, generate_textures, optimize
            )
    
    async def _convert_heightmap_tiled(
        self,
        input_file: Path,
        output_dir: Path,
        output_stem: str,
        width: int,
        height: int,
        scale: Dict[str, float],
        tile_size: int,
        lod_levels: int,
        game_engine: str,
        generate_textures: bool,
        optimize: bool
    ) -> Dict[str, Any]:
        """Convert massive heightmap using tile-based approach"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate tile grid
        tiles_x = (width + tile_size - 1) // tile_size
        tiles_y = (height + tile_size - 1) // tile_size
        
        logger.info(f"Processing {tiles_x}x{tiles_y} = {tiles_x * tiles_y} tiles of {tile_size}x{tile_size}")
        
        tile_results = []
        total_vertices = 0
        total_faces = 0
        
        # Open image once for reading tiles
        # Temporarily increase PIL's decompression bomb limit for massive heightmaps
        old_max_pixels = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None  # Disable limit
        
        try:
            with Image.open(input_file) as full_img:
                full_img = full_img.convert('L')  # Grayscale
                
                for tile_y in range(tiles_y):
                    for tile_x in range(tiles_x):
                        # Calculate tile bounds
                        x_start = tile_x * tile_size
                        y_start = tile_y * tile_size
                        x_end = min(x_start + tile_size, width)
                        y_end = min(y_start + tile_size, height)
                        
                        # Extract tile
                        tile_img = full_img.crop((x_start, y_start, x_end, y_end))
                        tile_width, tile_height = tile_img.size
                        heightmap = np.array(tile_img, dtype=np.float32) / 255.0
                        
                        # Calculate tile world position
                        world_x = (x_start / width - 0.5) * scale["x"]
                        world_z = (y_start / height - 0.5) * scale["z"]
                        
                        # Adjust scale for tile size
                        tile_scale = {
                            "x": scale["x"] * (tile_width / width),
                            "y": scale["y"],
                            "z": scale["z"] * (tile_height / height)
                        }
                        
                        # Generate mesh for tile
                        mesh_result = self._heightmap_to_mesh(heightmap, tile_scale, 1)
                        if not mesh_result["success"]:
                            continue
                        
                        mesh = mesh_result["mesh"]
                        
                        # Translate mesh to world position
                        mesh.vertices[:, 0] += world_x
                        mesh.vertices[:, 2] += world_z
                        
                        # Optimize if requested
                        if optimize:
                            mesh = self._optimize_mesh(mesh)
                        
                        # Export tile
                        tile_name = f"{output_stem}_tile_{tile_x:03d}_{tile_y:03d}"
                        tile_path = output_dir / f"{tile_name}.obj"
                        
                        try:
                            mesh.export(str(tile_path))
                            tile_results.append({
                                "tile_id": f"{tile_x}_{tile_y}",
                                "path": str(tile_path),
                                "vertices": len(mesh.vertices),
                                "faces": len(mesh.faces),
                                "world_position": [world_x, 0, world_z],
                                "world_bounds": mesh.bounds.tolist()
                            })
                            
                            total_vertices += len(mesh.vertices)
                            total_faces += len(mesh.faces)
                            
                        except Exception as e:
                            logger.warning(f"Failed to export tile {tile_x},{tile_y}: {e}")
            
        except Exception as e:
            # Restore original limit on error
            Image.MAX_IMAGE_PIXELS = old_max_pixels
            return {"success": False, "error": f"Failed to process heightmap: {e}"}
        finally:
            # Always restore original limit
            Image.MAX_IMAGE_PIXELS = old_max_pixels
        
        # Generate LOD levels if requested
        lod_results = []
        if lod_levels > 1:
            lod_results = await self._generate_lod_levels(
                input_file, output_dir, output_stem, lod_levels, scale, game_engine
            )
        
        # Generate master index file
        index_data = {
            "terrain_info": {
                "source_file": str(input_file),
                "total_size": [width, height],
                "tile_size": tile_size,
                "tile_grid": [tiles_x, tiles_y],
                "world_scale": scale,
                "game_engine": game_engine
            },
            "tiles": tile_results,
            "lod_levels": lod_results,
            "total_stats": {
                "vertices": total_vertices,
                "faces": total_faces,
                "tiles_generated": len(tile_results)
            }
        }
        
        index_path = output_dir / f"{output_stem}_index.json"
        import json
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        return {
            "success": True,
            "mode": "convert_massive",
            "approach": "tiled",
            "input_path": str(input_file),
            "output_directory": str(output_dir),
            "index_file": str(index_path),
            "mesh_stats": {
                "total_vertices": total_vertices,
                "total_faces": total_faces,
                "tiles_generated": len(tile_results),
                "tile_grid": [tiles_x, tiles_y]
            },
            "heightmap_info": {
                "dimensions": [width, height],
                "scale": scale,
                "tiles": tile_results[:5]  # First 5 tiles as sample
            },
            "lod_levels": len(lod_results),
            "game_engine": game_engine
        }
    
    async def _generate_heightmap_streaming(
        self,
        width: int,
        height: int,
        terrain_type: str,
        output_dir: Path,
        output_stem: str,
        scale: Dict[str, float],
        noise_params: Dict[str, float],
        features: List[str],
        tile_size: int,
        lod_levels: int,
        game_engine: str,
        generate_textures: bool,
        optimize: bool
    ) -> Dict[str, Any]:
        """Generate massive terrain using streaming approach"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate tile grid
        tiles_x = (width + tile_size - 1) // tile_size  
        tiles_y = (height + tile_size - 1) // tile_size
        
        logger.info(f"Generating {tiles_x}x{tiles_y} = {tiles_x * tiles_y} terrain tiles")
        
        tile_results = []
        total_vertices = 0
        total_faces = 0
        
        # Generate master heightmap first (for features that span tiles)
        master_heightmap_path = output_dir / f"{output_stem}_master.png"
        logger.info("Generating master heightmap for feature consistency...")
        
        # Use chunked generation to avoid memory issues
        master_heightmap = self._generate_chunked_heightmap(
            width, height, terrain_type, noise_params, features, chunk_size=2048
        )
        
        # Save master heightmap
        heightmap_img = (master_heightmap * 255).astype(np.uint8)
        Image.fromarray(heightmap_img, 'L').save(master_heightmap_path)
        
        # Generate tiles from master heightmap
        for tile_y in range(tiles_y):
            for tile_x in range(tiles_x):
                # Calculate tile bounds
                x_start = tile_x * tile_size
                y_start = tile_y * tile_size
                x_end = min(x_start + tile_size, width)
                y_end = min(y_start + tile_size, height)
                
                # Extract tile from master heightmap
                tile_heightmap = master_heightmap[y_start:y_end, x_start:x_end]
                tile_height, tile_width = tile_heightmap.shape
                
                # Calculate tile world position  
                world_x = (x_start / width - 0.5) * scale["x"]
                world_z = (y_start / height - 0.5) * scale["z"]
                
                # Adjust scale for tile size
                tile_scale = {
                    "x": scale["x"] * (tile_width / width),
                    "y": scale["y"], 
                    "z": scale["z"] * (tile_height / height)
                }
                
                # Generate mesh for tile
                mesh_result = self._heightmap_to_mesh(tile_heightmap, tile_scale, 1)
                if not mesh_result["success"]:
                    continue
                
                mesh = mesh_result["mesh"]
                
                # Translate mesh to world position
                mesh.vertices[:, 0] += world_x
                mesh.vertices[:, 2] += world_z
                
                # Optimize if requested
                if optimize:
                    mesh = self._optimize_mesh(mesh)
                
                # Export tile
                tile_name = f"{output_stem}_tile_{tile_x:03d}_{tile_y:03d}"
                tile_path = output_dir / f"{tile_name}.obj"
                
                try:
                    mesh.export(str(tile_path))
                    
                    # Generate texture tile if requested
                    texture_paths = []
                    if generate_textures:
                        texture_result = self._generate_texture_maps(
                            tile_heightmap, str(output_dir / tile_name), tile_scale
                        )
                        if texture_result["success"]:
                            texture_paths = texture_result["paths"]
                    
                    tile_results.append({
                        "tile_id": f"{tile_x}_{tile_y}",
                        "path": str(tile_path),
                        "vertices": len(mesh.vertices),
                        "faces": len(mesh.faces),
                        "world_position": [world_x, 0, world_z],
                        "world_bounds": mesh.bounds.tolist(),
                        "textures": texture_paths
                    })
                    
                    total_vertices += len(mesh.vertices)
                    total_faces += len(mesh.faces)
                    
                    # Clean up mesh from memory
                    del mesh
                    
                except Exception as e:
                    logger.warning(f"Failed to export tile {tile_x},{tile_y}: {e}")
        
        # Clean up master heightmap from memory
        del master_heightmap
        
        # Generate LOD levels if requested
        lod_results = []
        if lod_levels > 1:
            lod_results = await self._generate_lod_levels_streaming(
                master_heightmap_path, output_dir, output_stem, lod_levels, scale, game_engine
            )
        
        # Generate master index file
        index_data = {
            "terrain_info": {
                "generation_type": "procedural_streaming",
                "terrain_type": terrain_type,
                "total_size": [width, height],
                "tile_size": tile_size,
                "tile_grid": [tiles_x, tiles_y],
                "world_scale": scale,
                "noise_params": noise_params,
                "features": features,
                "game_engine": game_engine
            },
            "files": {
                "master_heightmap": str(master_heightmap_path),
                "tiles": tile_results,
                "lod_levels": lod_results
            },
            "total_stats": {
                "vertices": total_vertices,
                "faces": total_faces,
                "tiles_generated": len(tile_results),
                "memory_efficient": True
            }
        }
        
        index_path = output_dir / f"{output_stem}_index.json"
        import json
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        return {
            "success": True,
            "mode": "generate_massive",
            "approach": "streaming",
            "terrain_type": terrain_type,
            "master_heightmap": str(master_heightmap_path),
            "output_directory": str(output_dir),
            "index_file": str(index_path),
            "mesh_stats": {
                "total_vertices": total_vertices,
                "total_faces": total_faces,
                "tiles_generated": len(tile_results),
                "tile_grid": [tiles_x, tiles_y]
            },
            "generation_params": {
                "dimensions": [width, height],
                "scale": scale,
                "noise_params": noise_params,
                "features": features
            },
            "lod_levels": len(lod_results),
            "game_engine": game_engine,
            "performance": {
                "memory_efficient": True,
                "streaming_enabled": True
            }
        }
    
    def _generate_chunked_heightmap(
        self,
        width: int,
        height: int,
        terrain_type: str,
        noise_params: Dict[str, float],
        features: List[str],
        chunk_size: int = 2048
    ) -> np.ndarray:
        """Generate massive heightmap in memory-efficient chunks"""
        
        # Create output array
        heightmap = np.zeros((height, width), dtype=np.float32)
        
        # Process in chunks to manage memory
        for y_start in range(0, height, chunk_size):
            y_end = min(y_start + chunk_size, height)
            
            for x_start in range(0, width, chunk_size):
                x_end = min(x_start + chunk_size, width)
                
                # Generate chunk
                chunk_width = x_end - x_start
                chunk_height = y_end - y_start
                
                chunk = self._generate_noise_chunk(
                    chunk_width, chunk_height, x_start, y_start, width, height,
                    noise_params["octaves"], noise_params["frequency"],
                    noise_params["amplitude"], noise_params["lacunarity"], 
                    noise_params["persistence"]
                )
                
                # Apply terrain shaping to chunk
                chunk = self._apply_terrain_shaping_chunk(
                    chunk, terrain_type, x_start, y_start, width, height
                )
                
                # Store chunk in master heightmap
                heightmap[y_start:y_end, x_start:x_end] = chunk
                
                # Clean up chunk memory
                del chunk
        
        # Apply features that span multiple chunks
        for feature in features:
            heightmap = self._add_terrain_feature(heightmap, feature)
        
        # Normalize
        heightmap = np.clip(heightmap, 0, 1)
        
        return heightmap
    
    def _generate_noise_chunk(
        self,
        chunk_width: int,
        chunk_height: int, 
        offset_x: int,
        offset_y: int,
        total_width: int,
        total_height: int,
        octaves: int,
        frequency: float,
        amplitude: float,
        lacunarity: float,
        persistence: float
    ) -> np.ndarray:
        """Generate noise for a specific chunk with proper global coordinates"""
        
        # Use JIT-compiled version if available for better performance
        if HAS_NUMBA:
            return _generate_noise_chunk_numba(
                chunk_width, chunk_height, offset_x, offset_y,
                total_width, total_height, octaves, frequency,
                amplitude, lacunarity, persistence
            )
        
        # Fallback to regular implementation
        chunk = np.zeros((chunk_height, chunk_width), dtype=np.float32)
        
        for octave in range(octaves):
            octave_frequency = frequency * (lacunarity ** octave)
            octave_amplitude = amplitude * (persistence ** octave)
            
            for i in range(chunk_height):
                for j in range(chunk_width):
                    # Global coordinates
                    global_x = (offset_x + j) * octave_frequency / total_width
                    global_y = (offset_y + i) * octave_frequency / total_height
                    
                    # Multi-wave noise
                    noise_value = (
                        math.sin(global_x * 2 * math.pi) * math.cos(global_y * 2 * math.pi) +
                        math.sin(global_x * 4 * math.pi + 1.5) * math.cos(global_y * 4 * math.pi + 1.5) * 0.5 +
                        math.sin(global_x * 8 * math.pi + 3) * math.cos(global_y * 8 * math.pi + 3) * 0.25
                    ) / 1.75
                    
                    chunk[i, j] += noise_value * octave_amplitude
        
        return chunk
    
    def _apply_terrain_shaping_chunk(
        self,
        chunk: np.ndarray,
        terrain_type: str,
        offset_x: int,
        offset_y: int,
        total_width: int,
        total_height: int
    ) -> np.ndarray:
        """Apply terrain shaping to chunk using global coordinates"""
        
        chunk_height, chunk_width = chunk.shape
        center_x, center_y = total_width // 2, total_height // 2
        
        if terrain_type == "mountains":
            for i in range(chunk_height):
                for j in range(chunk_width):
                    # Global coordinates
                    global_x = offset_x + j
                    global_y = offset_y + i
                    
                    # Multiple mountain peaks
                    dist1 = math.sqrt((global_x - center_x)**2 + (global_y - center_y)**2) / min(total_width, total_height)
                    dist2 = math.sqrt((global_x - center_x*0.3)**2 + (global_y - center_y*1.5)**2) / min(total_width, total_height)
                    dist3 = math.sqrt((global_x - center_x*1.7)**2 + (global_y - center_y*0.4)**2) / min(total_width, total_height)
                    
                    mountain_factor = (
                        math.exp(-dist1 * 3) * 0.8 + 
                        math.exp(-dist2 * 4) * 0.6 +
                        math.exp(-dist3 * 4) * 0.5
                    )
                    chunk[i, j] = chunk[i, j] * 0.5 + mountain_factor
        
        # Add other terrain types as needed...
        
        return chunk
    
    async def _generate_lod_levels(
        self,
        input_file: Path,
        output_dir: Path,
        output_stem: str,
        lod_levels: int,
        scale: Dict[str, float],
        game_engine: str
    ) -> List[Dict[str, Any]]:
        """Generate multiple LOD levels for massive terrain"""
        
        lod_results = []
        
        # Temporarily increase PIL's decompression bomb limit for massive heightmaps
        old_max_pixels = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None  # Disable limit
        
        try:
            with Image.open(input_file) as img:
                img = img.convert('L')
                original_width, original_height = img.size
                
                for lod in range(1, lod_levels):
                    # Calculate LOD resolution (half each level)
                    lod_factor = 2 ** lod
                    lod_width = max(64, original_width // lod_factor)
                    lod_height = max(64, original_height // lod_factor)
                    
                    # Resize heightmap
                    lod_img = img.resize((lod_width, lod_height), Image.Resampling.LANCZOS)
                    heightmap = np.array(lod_img, dtype=np.float32) / 255.0
                    
                    # Generate mesh
                    mesh_result = self._heightmap_to_mesh(heightmap, scale, 1)
                    if not mesh_result["success"]:
                        continue
                    
                    mesh = mesh_result["mesh"]
                    mesh = self._optimize_mesh(mesh)
                    
                    # Export LOD level
                    lod_path = output_dir / f"{output_stem}_lod{lod}.obj"
                    mesh.export(str(lod_path))
                    
                    lod_results.append({
                        "lod_level": lod,
                        "resolution": [lod_width, lod_height],
                        "path": str(lod_path),
                        "vertices": len(mesh.vertices),
                        "faces": len(mesh.faces),
                        "reduction_factor": lod_factor
                    })
                    
        except Exception as e:
            # Restore original limit on error
            Image.MAX_IMAGE_PIXELS = old_max_pixels
            logger.warning(f"LOD generation failed: {e}")
            return []
        finally:
            # Always restore original limit
            Image.MAX_IMAGE_PIXELS = old_max_pixels
        
        return lod_results
    
    async def _generate_lod_levels_streaming(
        self,
        master_heightmap_path: Path,
        output_dir: Path,
        output_stem: str,
        lod_levels: int,
        scale: Dict[str, float],
        game_engine: str
    ) -> List[Dict[str, Any]]:
        """Generate LOD levels from master heightmap"""
        
        return await self._generate_lod_levels(
            master_heightmap_path, output_dir, output_stem, lod_levels, scale, game_engine
        )
    
    async def _convert_heightmap_optimized(
        self,
        input_file: Path,
        output_path: str,
        scale: Dict[str, float],
        resolution: int,
        lod_levels: int,
        game_engine: str,
        generate_textures: bool,
        optimize: bool
    ) -> Dict[str, Any]:
        """Optimized single-file conversion for large (but manageable) heightmaps"""
        
        # This is a more optimized version of the original convert method
        # Load image efficiently
        try:
            # Temporarily increase PIL's decompression bomb limit for massive heightmaps
            old_max_pixels = Image.MAX_IMAGE_PIXELS
            Image.MAX_IMAGE_PIXELS = None  # Disable limit
            
            img = Image.open(input_file).convert('L')
            width, height = img.size
            
            # Restore original limit
            Image.MAX_IMAGE_PIXELS = old_max_pixels
            
            # Use memory mapping for very large images
            if width * height > 8192 * 8192:  # 64MP threshold
                # Process in chunks
                heightmap = self._load_heightmap_chunked(img, width, height)
            else:
                heightmap = np.array(img, dtype=np.float32) / 255.0
        except Exception as e:
            # Restore original limit on error
            Image.MAX_IMAGE_PIXELS = old_max_pixels
            return {"success": False, "error": f"Failed to load heightmap: {e}"}
        
        # Generate mesh
        mesh_result = self._heightmap_to_mesh(heightmap, scale, resolution)
        if not mesh_result["success"]:
            return mesh_result
        
        mesh = mesh_result["mesh"]
        
        # Apply optimization
        if optimize:
            mesh = self._optimize_mesh_advanced(mesh)
        
        # Export mesh
        export_result = self._export_mesh(mesh, output_path, game_engine)
        if not export_result["success"]:
            return export_result
        
        # Generate texture maps if requested
        texture_paths = []
        if generate_textures:
            texture_result = self._generate_texture_maps(heightmap, Path(output_path).stem, scale)
            if texture_result["success"]:
                texture_paths = texture_result["paths"]
        
        return {
            "success": True,
            "mode": "convert_optimized",
            "input_path": str(input_file),
            "output_path": output_path,
            "mesh_stats": {
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "bounds": mesh.bounds.tolist(),
                "volume": mesh.volume if mesh.is_volume else 0
            },
            "heightmap_info": {
                "dimensions": [width, height],
                "min_height": float(heightmap.min()),
                "max_height": float(heightmap.max()),
                "scale": scale
            },
            "texture_maps": texture_paths,
            "game_engine": game_engine,
            "performance": {
                "optimized": optimize,
                "memory_efficient": width * height > 8192 * 8192
            }
        }
    
    def _load_heightmap_chunked(self, img: Image.Image, width: int, height: int, chunk_size: int = 4096) -> np.ndarray:
        """Load very large heightmap in chunks to manage memory"""
        
        heightmap = np.zeros((height, width), dtype=np.float32)
        
        for y in range(0, height, chunk_size):
            y_end = min(y + chunk_size, height)
            for x in range(0, width, chunk_size):
                x_end = min(x + chunk_size, width)
                
                # Load chunk
                chunk_img = img.crop((x, y, x_end, y_end))
                chunk_array = np.array(chunk_img, dtype=np.float32) / 255.0
                
                heightmap[y:y_end, x:x_end] = chunk_array
                
                del chunk_img, chunk_array
        
        return heightmap
    
    def _optimize_mesh_advanced(self, mesh: 'trimesh.Trimesh') -> 'trimesh.Trimesh':
        """Advanced mesh optimization for massive terrains"""
        
        try:
            # Basic cleanup
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
            mesh.fix_normals()
            mesh.fill_holes()
            
            # Advanced optimization for large meshes
            if len(mesh.vertices) > 100000:
                # Simplify very large meshes
                target_faces = min(len(mesh.faces), 500000)  # Max 500K faces
                if len(mesh.faces) > target_faces:
                    try:
                        mesh = mesh.simplify_quadric_decimation(target_faces)
                    except:
                        logger.warning("Quadric decimation failed, using basic simplification")
                
                # Smooth if very detailed
                try:
                    mesh = mesh.smoothed()
                except:
                    pass
            
        except Exception as e:
            logger.warning(f"Advanced mesh optimization partially failed: {e}")
        
        return mesh
    
    async def _generate_heightmap_optimized(
        self,
        width: int,
        height: int,
        terrain_type: str,
        output_path: str,
        scale: Dict[str, float],
        noise_params: Dict[str, float],
        features: List[str],
        lod_levels: int,
        game_engine: str,
        generate_textures: bool,
        optimize: bool
    ) -> Dict[str, Any]:
        """Optimized single-file terrain generation for large (but manageable) sizes"""
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Generate heightmap
        heightmap = self._generate_procedural_heightmap(
            width, height, terrain_type, noise_params, features
        )
        
        # Save heightmap as PNG
        heightmap_path = str(Path(output_path).with_suffix('.png'))
        heightmap_img = (heightmap * 255).astype(np.uint8)
        Image.fromarray(heightmap_img, 'L').save(heightmap_path)
        
        # Convert to mesh
        mesh_result = self._heightmap_to_mesh(heightmap, scale, 1)
        if not mesh_result["success"]:
            return mesh_result
        
        mesh = mesh_result["mesh"]
        
        # Apply optimization
        if optimize:
            mesh = self._optimize_mesh_advanced(mesh)
        
        # Export mesh
        export_result = self._export_mesh(mesh, output_path, game_engine)
        if not export_result["success"]:
            return export_result
        
        # Generate texture maps if requested
        texture_paths = []
        if generate_textures:
            texture_result = self._generate_texture_maps(heightmap, Path(output_path).stem, scale)
            if texture_result["success"]:
                texture_paths = texture_result["paths"]
        
        # Generate LOD levels if requested
        lod_results = []
        if lod_levels > 1:
            lod_results = await self._generate_lod_levels(
                Path(heightmap_path), Path(output_path).parent, Path(output_path).stem, 
                lod_levels, scale, game_engine
            )
        
        return {
            "success": True,
            "mode": "generate_optimized",
            "terrain_type": terrain_type,
            "heightmap_path": heightmap_path,
            "output_path": output_path,
            "mesh_stats": {
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "bounds": mesh.bounds.tolist(),
                "volume": mesh.volume if mesh.is_volume else 0
            },
            "generation_params": {
                "dimensions": [width, height],
                "scale": scale,
                "noise_params": noise_params,
                "features": features
            },
            "texture_maps": texture_paths,
            "lod_levels": len(lod_results),
            "game_engine": game_engine,
            "performance": {
                "optimized": optimize,
                "memory_efficient": width * height > 4096 * 4096
            }
        }
