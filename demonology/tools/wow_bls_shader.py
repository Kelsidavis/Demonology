# demonology/tools/wow_bls_shader.py
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import Tool, _confine

class WoWBLSShaderTool(Tool):
    """
    World of Warcraft BLS shader file converter.
    Converts BLS shader files to Unreal Engine material format.
    """

    def __init__(self):
        super().__init__("wow_bls_shader", "Convert World of Warcraft BLS shader files to Unreal materials")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["convert", "batch_convert", "analyze", "extract_textures"],
                        "description": "Operation to perform on BLS files"
                    },
                    "input_path": {
                        "type": "string",
                        "description": "Path to BLS file or directory containing BLS files"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output directory for converted materials",
                        "default": "./converted_materials"
                    },
                    "material_type": {
                        "type": "string",
                        "enum": ["standard", "terrain", "water", "character", "ui"],
                        "description": "Type of material to generate",
                        "default": "standard"
                    },
                    "unreal_version": {
                        "type": "string",
                        "enum": ["5.0", "5.1", "5.2", "5.3", "5.4"],
                        "description": "Target Unreal Engine version",
                        "default": "5.4"
                    },
                    "shader_features": {
                        "type": "object",
                        "description": "Shader features to enable",
                        "properties": {
                            "enable_transparency": {"type": "boolean", "default": False},
                            "enable_normal_mapping": {"type": "boolean", "default": True},
                            "enable_specular": {"type": "boolean", "default": True},
                            "enable_emission": {"type": "boolean", "default": False},
                            "two_sided": {"type": "boolean", "default": False},
                            "alpha_test": {"type": "boolean", "default": False}
                        }
                    }
                },
                "required": ["operation", "input_path"]
            }
        }

    def _parse_bls_header(self, data: bytes) -> Dict[str, Any]:
        """Parse BLS file header."""
        if len(data) < 32:
            raise ValueError("Invalid BLS file: too short")
        
        # BLS header structure (simplified)
        signature = data[:4]
        if signature != b'BLS\x00':
            raise ValueError(f"Invalid BLS signature: {signature}")
        
        version, shader_type, num_textures, num_uniforms = struct.unpack('<IIII', data[4:20])
        
        return {
            "signature": signature.decode('ascii', errors='ignore'),
            "version": version,
            "shader_type": shader_type,
            "num_textures": num_textures,
            "num_uniforms": num_uniforms,
            "header_size": 32
        }

    def _extract_shader_properties(self, data: bytes, header: Dict[str, Any]) -> Dict[str, Any]:
        """Extract shader properties from BLS data."""
        properties = {
            "textures": [],
            "uniforms": [],
            "blend_mode": "Opaque",
            "shading_model": "DefaultLit",
            "two_sided": False,
            "alpha_test": False
        }
        
        offset = header["header_size"]
        
        # Extract texture references
        for i in range(header["num_textures"]):
            if offset + 64 > len(data):
                break
                
            tex_data = data[offset:offset + 64]
            # Extract texture name (null-terminated string)
            tex_name_bytes = tex_data[:32].rstrip(b'\x00')
            tex_name = tex_name_bytes.decode('ascii', errors='ignore')
            
            # Extract texture properties
            tex_type, tex_flags = struct.unpack('<II', tex_data[32:40])
            
            texture_info = {
                "name": tex_name,
                "type": tex_type,
                "flags": tex_flags,
                "slot": i
            }
            
            # Determine texture usage based on name patterns
            if "diffuse" in tex_name.lower() or "color" in tex_name.lower():
                texture_info["usage"] = "BaseColor"
            elif "normal" in tex_name.lower() or "bump" in tex_name.lower():
                texture_info["usage"] = "Normal"
            elif "specular" in tex_name.lower() or "spec" in tex_name.lower():
                texture_info["usage"] = "Specular"
            elif "rough" in tex_name.lower():
                texture_info["usage"] = "Roughness"
            elif "metal" in tex_name.lower():
                texture_info["usage"] = "Metallic"
            elif "emis" in tex_name.lower() or "glow" in tex_name.lower():
                texture_info["usage"] = "EmissiveColor"
            elif "alpha" in tex_name.lower() or "opacity" in tex_name.lower():
                texture_info["usage"] = "Opacity"
            else:
                texture_info["usage"] = "Custom"
            
            properties["textures"].append(texture_info)
            offset += 64
        
        # Extract uniform/constant values
        for i in range(header["num_uniforms"]):
            if offset + 48 > len(data):
                break
                
            uniform_data = data[offset:offset + 48]
            uniform_name_bytes = uniform_data[:32].rstrip(b'\x00')
            uniform_name = uniform_name_bytes.decode('ascii', errors='ignore')
            
            # Extract uniform value (assuming float4)
            values = struct.unpack('<ffff', uniform_data[32:48])
            
            uniform_info = {
                "name": uniform_name,
                "values": values,
                "type": "float4"
            }
            
            properties["uniforms"].append(uniform_info)
            offset += 48
        
        # Analyze shader type and flags to determine material properties
        if header["shader_type"] & 0x1:  # Alpha blending
            properties["blend_mode"] = "Translucent"
        if header["shader_type"] & 0x2:  # Two-sided
            properties["two_sided"] = True
        if header["shader_type"] & 0x4:  # Alpha test
            properties["alpha_test"] = True
            properties["blend_mode"] = "Masked"
        
        return properties

    def _generate_unreal_material(self, material_name: str, properties: Dict[str, Any],
                                shader_features: Dict[str, Any]) -> str:
        """Generate Unreal Engine material asset."""
        
        # Build material nodes
        material_nodes = []
        connections = []
        node_id = 1
        
        # Texture sample nodes
        for tex in properties["textures"]:
            if tex["usage"] in ["BaseColor", "Normal", "Specular", "Roughness", "Metallic", "EmissiveColor", "Opacity"]:
                texture_node = {
                    "NodeId": node_id,
                    "NodeType": "TextureSample",
                    "MaterialExpression": "MaterialExpressionTextureSample",
                    "Properties": {
                        "Texture": f"Texture2D'/Game/Textures/{tex['name']}.{tex['name']}'",
                        "SamplerType": "SAMPLERTYPE_Color" if tex["usage"] != "Normal" else "SAMPLERTYPE_Normal"
                    },
                    "Position": {"X": -400, "Y": node_id * 200}
                }
                material_nodes.append(texture_node)
                
                # Connect to material output
                if tex["usage"] == "BaseColor":
                    connections.append({
                        "OutputNodeId": node_id,
                        "OutputPin": "RGB",
                        "InputPin": "Base Color"
                    })
                elif tex["usage"] == "Normal" and shader_features.get("enable_normal_mapping", True):
                    connections.append({
                        "OutputNodeId": node_id,
                        "OutputPin": "RGB",
                        "InputPin": "Normal"
                    })
                elif tex["usage"] == "Specular" and shader_features.get("enable_specular", True):
                    connections.append({
                        "OutputNodeId": node_id,
                        "OutputPin": "RGB",
                        "InputPin": "Specular"
                    })
                elif tex["usage"] == "Roughness":
                    connections.append({
                        "OutputNodeId": node_id,
                        "OutputPin": "R",
                        "InputPin": "Roughness"
                    })
                elif tex["usage"] == "Metallic":
                    connections.append({
                        "OutputNodeId": node_id,
                        "OutputPin": "R",
                        "InputPin": "Metallic"
                    })
                elif tex["usage"] == "EmissiveColor" and shader_features.get("enable_emission", False):
                    connections.append({
                        "OutputNodeId": node_id,
                        "OutputPin": "RGB",
                        "InputPin": "Emissive Color"
                    })
                elif tex["usage"] == "Opacity" and shader_features.get("enable_transparency", False):
                    connections.append({
                        "OutputNodeId": node_id,
                        "OutputPin": "R",
                        "InputPin": "Opacity"
                    })
                
                node_id += 1
        
        # Constant nodes for uniform values
        for uniform in properties["uniforms"]:
            if "color" in uniform["name"].lower() or "tint" in uniform["name"].lower():
                const_node = {
                    "NodeId": node_id,
                    "NodeType": "VectorParameter",
                    "MaterialExpression": "MaterialExpressionVectorParameter",
                    "Properties": {
                        "ParameterName": uniform["name"],
                        "DefaultValue": {
                            "R": uniform["values"][0],
                            "G": uniform["values"][1],
                            "B": uniform["values"][2],
                            "A": uniform["values"][3]
                        }
                    },
                    "Position": {"X": -400, "Y": node_id * 200}
                }
                material_nodes.append(const_node)
                node_id += 1
            elif "scale" in uniform["name"].lower() or "intensity" in uniform["name"].lower():
                scalar_node = {
                    "NodeId": node_id,
                    "NodeType": "ScalarParameter",
                    "MaterialExpression": "MaterialExpressionScalarParameter",
                    "Properties": {
                        "ParameterName": uniform["name"],
                        "DefaultValue": uniform["values"][0]
                    },
                    "Position": {"X": -400, "Y": node_id * 200}
                }
                material_nodes.append(scalar_node)
                node_id += 1
        
        # Generate material asset JSON
        material_asset = {
            "Type": "Material",
            "Name": material_name,
            "Properties": {
                "BlendMode": properties["blend_mode"],
                "ShadingModel": properties["shading_model"],
                "TwoSided": properties["two_sided"] or shader_features.get("two_sided", False),
                "DitheredLODTransition": False,
                "TranslucencyLightingMode": "TLM_VolumetricNonDirectional" if properties["blend_mode"] == "Translucent" else "TLM_Surface"
            },
            "MaterialGraph": {
                "Nodes": material_nodes,
                "Connections": connections
            },
            "TextureDependencies": [tex["name"] for tex in properties["textures"]],
            "ParameterGroups": [
                {
                    "GroupName": "WoW Material Parameters",
                    "Parameters": [uniform["name"] for uniform in properties["uniforms"]]
                }
            ]
        }
        
        return json.dumps(material_asset, indent=2)

    def _generate_material_instance(self, base_material: str, instance_name: str,
                                  properties: Dict[str, Any]) -> str:
        """Generate material instance for specific texture combinations."""
        
        instance_data = {
            "Type": "MaterialInstance",
            "Name": instance_name,
            "Parent": base_material,
            "TextureParameters": {},
            "ScalarParameters": {},
            "VectorParameters": {}
        }
        
        # Set texture parameters
        for tex in properties["textures"]:
            param_name = f"{tex['usage']}Texture"
            instance_data["TextureParameters"][param_name] = f"Texture2D'/Game/Textures/{tex['name']}.{tex['name']}'"
        
        # Set scalar parameters
        for uniform in properties["uniforms"]:
            if len(uniform["values"]) == 1:
                instance_data["ScalarParameters"][uniform["name"]] = uniform["values"][0]
            else:
                instance_data["VectorParameters"][uniform["name"]] = {
                    "R": uniform["values"][0],
                    "G": uniform["values"][1],
                    "B": uniform["values"][2],
                    "A": uniform["values"][3]
                }
        
        return json.dumps(instance_data, indent=2)

    async def _convert_bls_file(self, input_file: Path, output_dir: Path,
                              material_type: str, shader_features: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a single BLS file to Unreal material."""
        try:
            with open(input_file, 'rb') as f:
                bls_data = f.read()
            
            # Parse BLS file
            header = self._parse_bls_header(bls_data)
            properties = self._extract_shader_properties(bls_data, header)
            
            material_name = input_file.stem
            
            # Generate base material
            material_content = self._generate_unreal_material(material_name, properties, shader_features)
            
            # Save material file
            material_file = output_dir / f"{material_name}.uasset"
            with open(material_file, 'w') as f:
                f.write(material_content)
            
            # Generate material instance for easier tweaking
            instance_content = self._generate_material_instance(material_name, f"{material_name}_Inst", properties)
            instance_file = output_dir / f"{material_name}_Inst.uasset"
            with open(instance_file, 'w') as f:
                f.write(instance_content)
            
            return {
                "success": True,
                "input_file": str(input_file),
                "material_file": str(material_file),
                "instance_file": str(instance_file),
                "shader_type": header["shader_type"],
                "num_textures": header["num_textures"],
                "num_uniforms": header["num_uniforms"],
                "blend_mode": properties["blend_mode"],
                "textures": [tex["name"] for tex in properties["textures"]]
            }
            
        except Exception as e:
            return {
                "success": False,
                "input_file": str(input_file),
                "error": str(e)
            }

    async def execute(self, operation: str, input_path: str, output_path: str = "./converted_materials",
                     material_type: str = "standard", unreal_version: str = "5.4",
                     shader_features: Optional[Dict[str, Any]] = None, **_) -> Dict[str, Any]:
        
        shader_features = shader_features or {}
        
        try:
            input_path_obj = _confine(Path(input_path))
            output_dir = _confine(Path(output_path))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if operation == "convert":
                if not input_path_obj.exists():
                    return {"success": False, "error": f"Input file not found: {input_path}"}
                
                if not input_path_obj.suffix.lower() == '.bls':
                    return {"success": False, "error": "Input file must be a BLS file"}
                
                result = await self._convert_bls_file(
                    input_path_obj, output_dir, material_type, shader_features
                )
                
                return result
            
            elif operation == "batch_convert":
                if not input_path_obj.exists():
                    return {"success": False, "error": f"Input directory not found: {input_path}"}
                
                bls_files = list(input_path_obj.rglob("*.bls"))
                if not bls_files:
                    return {"success": False, "error": "No BLS files found in directory"}
                
                results = []
                successful = 0
                failed = 0
                
                for bls_file in bls_files:
                    result = await self._convert_bls_file(
                        bls_file, output_dir, material_type, shader_features
                    )
                    results.append(result)
                    
                    if result["success"]:
                        successful += 1
                    else:
                        failed += 1
                
                return {
                    "success": True,
                    "operation": "batch_convert",
                    "total_files": len(bls_files),
                    "successful": successful,
                    "failed": failed,
                    "output_directory": str(output_dir),
                    "results": results
                }
            
            elif operation == "analyze":
                if not input_path_obj.exists() or not input_path_obj.suffix.lower() == '.bls':
                    return {"success": False, "error": "Invalid BLS file"}
                
                with open(input_path_obj, 'rb') as f:
                    bls_data = f.read()
                
                header = self._parse_bls_header(bls_data)
                properties = self._extract_shader_properties(bls_data, header)
                
                return {
                    "success": True,
                    "file": str(input_path_obj),
                    "version": header["version"],
                    "shader_type": header["shader_type"],
                    "num_textures": header["num_textures"],
                    "num_uniforms": header["num_uniforms"],
                    "blend_mode": properties["blend_mode"],
                    "textures": properties["textures"],
                    "uniforms": properties["uniforms"],
                    "file_size": len(bls_data)
                }
            
            elif operation == "extract_textures":
                if not input_path_obj.exists() or not input_path_obj.suffix.lower() == '.bls':
                    return {"success": False, "error": "Invalid BLS file"}
                
                with open(input_path_obj, 'rb') as f:
                    bls_data = f.read()
                
                header = self._parse_bls_header(bls_data)
                properties = self._extract_shader_properties(bls_data, header)
                
                # Extract texture references
                texture_list_file = output_dir / f"{input_path_obj.stem}_textures.txt"
                with open(texture_list_file, 'w') as f:
                    for tex in properties["textures"]:
                        f.write(f"{tex['name']}.blp\n")
                
                return {
                    "success": True,
                    "file": str(input_path_obj),
                    "texture_count": len(properties["textures"]),
                    "texture_list_file": str(texture_list_file),
                    "textures": [tex["name"] for tex in properties["textures"]]
                }
            
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            return {"success": False, "error": f"BLS conversion failed: {str(e)}"}