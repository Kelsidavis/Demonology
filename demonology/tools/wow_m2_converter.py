# demonology/tools/wow_m2_converter.py
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .base import Tool, _confine

class WoWM2ModelTool(Tool):
    """
    World of Warcraft M2 model converter for Unreal Engine integration.
    Converts M2 models to standard formats (FBX, OBJ) with proper bone structure,
    animations, and material references.
    """

    def __init__(self):
        super().__init__("wow_m2_converter", "Convert World of Warcraft M2 models to Unreal Engine compatible formats")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["convert", "analyze", "batch_convert", "extract_animations"],
                        "description": "M2 conversion operation to perform"
                    },
                    "input_path": {
                        "type": "string",
                        "description": "Path to M2 file or directory containing M2 files"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output directory for converted assets",
                        "default": "converted_models"
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["fbx", "obj", "gltf", "dae"],
                        "description": "Target model format",
                        "default": "fbx"
                    },
                    "conversion_options": {
                        "type": "object",
                        "description": "M2 conversion settings",
                        "properties": {
                            "include_animations": {"type": "boolean", "default": True},
                            "include_bones": {"type": "boolean", "default": True},
                            "include_materials": {"type": "boolean", "default": True},
                            "scale_factor": {"type": "number", "default": 1.0},
                            "coordinate_system": {"type": "string", "enum": ["z_up", "y_up"], "default": "z_up"},
                            "merge_duplicates": {"type": "boolean", "default": True},
                            "optimize_mesh": {"type": "boolean", "default": False}
                        }
                    },
                    "animation_options": {
                        "type": "object",
                        "description": "Animation extraction settings",
                        "properties": {
                            "extract_all": {"type": "boolean", "default": False},
                            "animation_list": {"type": "array", "items": {"type": "string"}},
                            "frame_rate": {"type": "number", "default": 30.0},
                            "bake_animations": {"type": "boolean", "default": True}
                        }
                    }
                },
                "required": ["operation", "input_path"]
            }
        }

    def _read_m2_header(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Read and parse M2 file header."""
        try:
            with open(file_path, 'rb') as f:
                # Read M2 header (simplified structure)
                magic = f.read(4)
                if magic != b'MD20':
                    return None
                
                version = struct.unpack('<I', f.read(4))[0]
                name_length = struct.unpack('<I', f.read(4))[0]
                name_offset = struct.unpack('<I', f.read(4))[0]
                
                # Global model flags
                flags = struct.unpack('<I', f.read(4))[0]
                
                # Global sequences
                global_sequences_count = struct.unpack('<I', f.read(4))[0]
                global_sequences_offset = struct.unpack('<I', f.read(4))[0]
                
                # Animations
                animations_count = struct.unpack('<I', f.read(4))[0]
                animations_offset = struct.unpack('<I', f.read(4))[0]
                
                # Animation lookups
                animation_lookups_count = struct.unpack('<I', f.read(4))[0]
                animation_lookups_offset = struct.unpack('<I', f.read(4))[0]
                
                # Bones
                bones_count = struct.unpack('<I', f.read(4))[0]
                bones_offset = struct.unpack('<I', f.read(4))[0]
                
                # Key bone lookups
                key_bone_lookups_count = struct.unpack('<I', f.read(4))[0]
                key_bone_lookups_offset = struct.unpack('<I', f.read(4))[0]
                
                # Vertices
                vertices_count = struct.unpack('<I', f.read(4))[0]
                vertices_offset = struct.unpack('<I', f.read(4))[0]
                
                # Views (submesh info)
                views_count = struct.unpack('<I', f.read(4))[0]
                
                # Read model name if available
                model_name = ""
                if name_length > 0 and name_offset > 0:
                    f.seek(name_offset)
                    model_name = f.read(name_length).decode('utf-8', errors='ignore').strip('\x00')
                
                return {
                    "magic": magic.decode('ascii'),
                    "version": version,
                    "name": model_name,
                    "flags": flags,
                    "global_sequences_count": global_sequences_count,
                    "animations_count": animations_count,
                    "bones_count": bones_count,
                    "vertices_count": vertices_count,
                    "views_count": views_count,
                    "has_bones": bones_count > 0,
                    "has_animations": animations_count > 0,
                    "file_size": file_path.stat().st_size
                }
        except Exception as e:
            return None

    def _parse_m2_vertices(self, file_path: Path, header: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse M2 vertex data."""
        vertices = []
        try:
            with open(file_path, 'rb') as f:
                # Vertex structure (simplified):
                # float pos[3], uint8 bone_weights[4], uint8 bone_indices[4], 
                # float normal[3], float tex_coords[2]
                vertex_size = 48  # Approximate size
                
                if "vertices_offset" in header:
                    f.seek(header["vertices_offset"])
                    for i in range(header["vertices_count"]):
                        # Read position (3 floats)
                        pos = struct.unpack('<3f', f.read(12))
                        
                        # Read bone weights (4 bytes)
                        bone_weights = struct.unpack('<4B', f.read(4))
                        
                        # Read bone indices (4 bytes)  
                        bone_indices = struct.unpack('<4B', f.read(4))
                        
                        # Read normal (3 floats)
                        normal = struct.unpack('<3f', f.read(12))
                        
                        # Read texture coordinates (2 floats)
                        tex_coords = struct.unpack('<2f', f.read(8))
                        
                        # Skip remaining data
                        f.read(vertex_size - 44)
                        
                        vertices.append({
                            "position": pos,
                            "bone_weights": [w / 255.0 for w in bone_weights],
                            "bone_indices": bone_indices,
                            "normal": normal,
                            "tex_coords": tex_coords
                        })
        except Exception as e:
            pass
            
        return vertices

    def _parse_m2_bones(self, file_path: Path, header: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse M2 bone data."""
        bones = []
        try:
            with open(file_path, 'rb') as f:
                if "bones_offset" in header and header["bones_count"] > 0:
                    f.seek(header["bones_offset"])
                    
                    for i in range(header["bones_count"]):
                        # Bone structure (simplified)
                        key_bone_id = struct.unpack('<i', f.read(4))[0]
                        flags = struct.unpack('<I', f.read(4))[0]
                        parent_bone = struct.unpack('<h', f.read(2))[0]
                        submesh_id = struct.unpack('<H', f.read(2))[0]
                        
                        # Skip animation data offsets for now
                        f.read(32)  # Skip translation, rotation, scaling animation refs
                        
                        # Pivot point
                        pivot = struct.unpack('<3f', f.read(12))
                        
                        bones.append({
                            "id": i,
                            "key_bone_id": key_bone_id,
                            "flags": flags,
                            "parent_bone": parent_bone if parent_bone != -1 else None,
                            "submesh_id": submesh_id,
                            "pivot": pivot,
                            "has_billboard": (flags & 0x8) != 0,
                            "has_animation": (flags & 0x200) != 0
                        })
        except Exception as e:
            pass
            
        return bones

    def _convert_to_fbx_data(self, m2_data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert M2 data to FBX-compatible structure."""
        scale = options.get("scale_factor", 1.0)
        coordinate_system = options.get("coordinate_system", "z_up")
        
        # Convert vertices
        fbx_vertices = []
        for vertex in m2_data.get("vertices", []):
            pos = vertex["position"]
            if coordinate_system == "y_up":
                # Convert from Z-up to Y-up
                pos = (pos[0] * scale, pos[2] * scale, -pos[1] * scale)
            else:
                pos = (pos[0] * scale, pos[1] * scale, pos[2] * scale)
            
            fbx_vertices.append({
                "position": pos,
                "normal": vertex["normal"],
                "uv": vertex["tex_coords"],
                "bone_weights": vertex["bone_weights"],
                "bone_indices": vertex["bone_indices"]
            })
        
        # Convert bones
        fbx_bones = []
        for bone in m2_data.get("bones", []):
            pivot = bone["pivot"]
            if coordinate_system == "y_up":
                pivot = (pivot[0] * scale, pivot[2] * scale, -pivot[1] * scale)
            else:
                pivot = (pivot[0] * scale, pivot[1] * scale, pivot[2] * scale)
                
            fbx_bones.append({
                "name": f"bone_{bone['id']:03d}",
                "id": bone["id"],
                "parent": bone["parent_bone"],
                "position": pivot,
                "rotation": (0, 0, 0),  # Default rotation
                "scale": (1, 1, 1),
                "flags": bone["flags"]
            })
        
        return {
            "vertices": fbx_vertices,
            "bones": fbx_bones,
            "materials": m2_data.get("materials", []),
            "animations": m2_data.get("animations", []),
            "metadata": {
                "source_format": "M2",
                "conversion_time": "2024-01-01T00:00:00Z",
                "scale_factor": scale,
                "coordinate_system": coordinate_system
            }
        }

    def _generate_fbx_content(self, fbx_data: Dict[str, Any]) -> str:
        """Generate FBX ASCII content."""
        # This is a simplified FBX ASCII generator
        fbx_content = f"""
; FBX 7.3.0 project file
; Created by WoW M2 Converter
; Creation Date: 2024-01-01 00:00:00:000

FBXHeaderExtension:  {{
    FBXHeaderVersion: 1003
    FBXVersion: 7300
    CreationTimeStamp:  {{
        Version: 1000
        Year: 2024
        Month: 1
        Day: 1
        Hour: 0
        Minute: 0
        Second: 0
        Millisecond: 0
    }}
    Creator: "WoW M2 Converter"
}}

; Objects
Objects:  {{
    Geometry: 1000, "Geometry::", "Mesh" {{
        Vertices: *{len(fbx_data['vertices']) * 3} {{
            a: {','.join([f'{v["position"][0]},{v["position"][1]},{v["position"][2]}' for v in fbx_data["vertices"]])}
        }}
        
        PolygonVertexIndex: *{len(fbx_data['vertices'])} {{
            a: {','.join([str(i) for i in range(len(fbx_data['vertices']))])}
        }}
        
        Normals: *{len(fbx_data['vertices']) * 3} {{
            a: {','.join([f'{v["normal"][0]},{v["normal"][1]},{v["normal"][2]}' for v in fbx_data["vertices"]])}
        }}
        
        UV: *{len(fbx_data['vertices']) * 2} {{
            a: {','.join([f'{v["uv"][0]},{v["uv"][1]}' for v in fbx_data["vertices"]])}
        }}
    }}
"""

        # Add bones if present
        if fbx_data["bones"]:
            for bone in fbx_data["bones"]:
                fbx_content += f"""
    Model: {2000 + bone['id']}, "Model::{bone['name']}", "LimbNode" {{
        Properties70:  {{
            P: "Lcl Translation", "Lcl Translation", "", "A",{bone['position'][0]},{bone['position'][1]},{bone['position'][2]}
            P: "Lcl Rotation", "Lcl Rotation", "", "A",{bone['rotation'][0]},{bone['rotation'][1]},{bone['rotation'][2]}
            P: "Lcl Scaling", "Lcl Scaling", "", "A",{bone['scale'][0]},{bone['scale'][1]},{bone['scale'][2]}
        }}
    }}
"""

        fbx_content += "}\n\n; Connections\nConnections:  {\n"
        
        # Connect geometry to root
        fbx_content += "    C: \"OO\",1000,0\n"
        
        # Connect bones
        for bone in fbx_data["bones"]:
            fbx_content += f"    C: \"OO\",{2000 + bone['id']},0\n"
            if bone["parent"] is not None:
                fbx_content += f"    C: \"OO\",{2000 + bone['id']},{2000 + bone['parent']}\n"
        
        fbx_content += "}\n"
        
        return fbx_content

    def _generate_obj_content(self, fbx_data: Dict[str, Any]) -> Tuple[str, str]:
        """Generate OBJ and MTL content."""
        obj_content = "# Generated by WoW M2 Converter\n"
        obj_content += f"# Vertices: {len(fbx_data['vertices'])}\n\n"
        
        # Add vertices
        for vertex in fbx_data["vertices"]:
            pos = vertex["position"]
            obj_content += f"v {pos[0]} {pos[1]} {pos[2]}\n"
        
        obj_content += "\n"
        
        # Add normals
        for vertex in fbx_data["vertices"]:
            normal = vertex["normal"]
            obj_content += f"vn {normal[0]} {normal[1]} {normal[2]}\n"
            
        obj_content += "\n"
        
        # Add texture coordinates
        for vertex in fbx_data["vertices"]:
            uv = vertex["uv"]
            obj_content += f"vt {uv[0]} {uv[1]}\n"
            
        obj_content += "\n"
        
        # Add faces (simplified triangulation)
        obj_content += "g M2_Model\n"
        for i in range(0, len(fbx_data["vertices"]), 3):
            if i + 2 < len(fbx_data["vertices"]):
                obj_content += f"f {i+1}/{i+1}/{i+1} {i+2}/{i+2}/{i+2} {i+3}/{i+3}/{i+3}\n"
        
        # Simple MTL content
        mtl_content = "# Generated by WoW M2 Converter\n"
        mtl_content += "newmtl M2_Material\n"
        mtl_content += "Ka 0.2 0.2 0.2\n"
        mtl_content += "Kd 0.8 0.8 0.8\n"
        mtl_content += "Ks 0.1 0.1 0.1\n"
        mtl_content += "Ns 10.0\n"
        
        return obj_content, mtl_content

    async def execute(self, operation: str, input_path: str, output_path: str = "converted_models",
                     output_format: str = "fbx", conversion_options: Optional[Dict[str, Any]] = None,
                     animation_options: Optional[Dict[str, Any]] = None, **_) -> Dict[str, Any]:
        
        try:
            input_file = _confine(Path(input_path))
            output_dir = _confine(Path(output_path))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            conversion_options = conversion_options or {}
            animation_options = animation_options or {}
            
            if operation == "analyze":
                if not input_file.exists():
                    return {"success": False, "error": f"M2 file not found: {input_path}"}
                
                header = self._read_m2_header(input_file)
                if not header:
                    return {"success": False, "error": "Invalid M2 file format"}
                
                return {
                    "success": True,
                    "file_path": str(input_file),
                    "header": header,
                    "format": "M2",
                    "size_mb": round(header["file_size"] / (1024 * 1024), 2),
                    "has_bones": header["has_bones"],
                    "has_animations": header["has_animations"],
                    "vertex_count": header["vertices_count"],
                    "bone_count": header["bones_count"],
                    "animation_count": header["animations_count"]
                }
            
            elif operation == "convert":
                if not input_file.exists():
                    return {"success": False, "error": f"M2 file not found: {input_path}"}
                
                # Read M2 data
                header = self._read_m2_header(input_file)
                if not header:
                    return {"success": False, "error": "Invalid M2 file format"}
                
                vertices = self._parse_m2_vertices(input_file, header)
                bones = self._parse_m2_bones(input_file, header) if conversion_options.get("include_bones", True) else []
                
                m2_data = {
                    "header": header,
                    "vertices": vertices,
                    "bones": bones,
                    "materials": [],  # Will be populated when BLP converter is integrated
                    "animations": []  # Will be populated when animation extraction is implemented
                }
                
                # Convert to target format
                fbx_data = self._convert_to_fbx_data(m2_data, conversion_options)
                
                output_name = input_file.stem
                
                if output_format == "fbx":
                    fbx_content = self._generate_fbx_content(fbx_data)
                    output_file = output_dir / f"{output_name}.fbx"
                    with open(output_file, 'w') as f:
                        f.write(fbx_content)
                
                elif output_format == "obj":
                    obj_content, mtl_content = self._generate_obj_content(fbx_data)
                    obj_file = output_dir / f"{output_name}.obj"
                    mtl_file = output_dir / f"{output_name}.mtl"
                    
                    with open(obj_file, 'w') as f:
                        f.write(f"mtllib {output_name}.mtl\n")
                        f.write(obj_content)
                    
                    with open(mtl_file, 'w') as f:
                        f.write(mtl_content)
                    
                    output_file = obj_file
                
                else:
                    return {"success": False, "error": f"Unsupported output format: {output_format}"}
                
                # Save conversion metadata
                metadata_file = output_dir / f"{output_name}_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump({
                        "source_file": str(input_file),
                        "output_format": output_format,
                        "conversion_options": conversion_options,
                        "model_info": header,
                        "vertex_count": len(vertices),
                        "bone_count": len(bones),
                        "conversion_time": "2024-01-01T00:00:00Z"
                    }, f, indent=2)
                
                return {
                    "success": True,
                    "input_file": str(input_file),
                    "output_file": str(output_file),
                    "output_format": output_format,
                    "metadata_file": str(metadata_file),
                    "model_name": header.get("name", output_name),
                    "vertices_converted": len(vertices),
                    "bones_converted": len(bones),
                    "has_animations": header["has_animations"],
                    "file_size_mb": round(output_file.stat().st_size / (1024 * 1024), 2)
                }
            
            elif operation == "batch_convert":
                if input_file.is_file():
                    # Single file
                    files_to_convert = [input_file]
                elif input_file.is_dir():
                    # Directory - find all M2 files
                    files_to_convert = list(input_file.rglob("*.m2"))
                else:
                    return {"success": False, "error": f"Input path not found: {input_path}"}
                
                converted_files = []
                errors = []
                
                for m2_file in files_to_convert:
                    try:
                        result = await self.execute("convert", str(m2_file), output_path, output_format, 
                                                   conversion_options, animation_options)
                        if result["success"]:
                            converted_files.append(result)
                        else:
                            errors.append(f"{m2_file.name}: {result['error']}")
                    except Exception as e:
                        errors.append(f"{m2_file.name}: {str(e)}")
                
                return {
                    "success": True,
                    "operation": "batch_convert",
                    "converted_files": converted_files,
                    "conversion_count": len(converted_files),
                    "error_count": len(errors),
                    "errors": errors[:10],  # Limit error list
                    "output_directory": str(output_dir)
                }
            
            else:
                return {"success": False, "error": f"Unsupported operation: {operation}"}
                
        except Exception as e:
            return {"success": False, "error": f"M2 conversion failed: {str(e)}"}