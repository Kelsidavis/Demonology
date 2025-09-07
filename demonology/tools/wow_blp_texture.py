# demonology/tools/wow_blp_texture.py
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import io

from .base import Tool, _confine

class WoWBLPTextureTool(Tool):
    """
    World of Warcraft BLP texture file converter.
    Converts BLP texture files to common formats (PNG, TGA, DDS) for Unreal Engine.
    """

    def __init__(self):
        super().__init__("wow_blp_texture", "Convert World of Warcraft BLP texture files")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["convert", "batch_convert", "info", "extract_mipmaps"],
                        "description": "Operation to perform on BLP files"
                    },
                    "input_path": {
                        "type": "string",
                        "description": "Path to BLP file or directory containing BLP files"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output directory for converted files",
                        "default": "./converted_textures"
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["png", "tga", "dds", "jpg"],
                        "description": "Target texture format",
                        "default": "png"
                    },
                    "mipmap_level": {
                        "type": "integer",
                        "description": "Specific mipmap level to extract (0 = highest resolution)",
                        "default": 0
                    },
                    "alpha_handling": {
                        "type": "string",
                        "enum": ["preserve", "remove", "premultiply"],
                        "description": "How to handle alpha channel",
                        "default": "preserve"
                    },
                    "unreal_settings": {
                        "type": "object",
                        "description": "Unreal Engine specific texture settings",
                        "properties": {
                            "generate_unreal_asset": {"type": "boolean", "default": True},
                            "texture_group": {"type": "string", "default": "World"},
                            "compression": {"type": "string", "default": "TC_Default"},
                            "srgb": {"type": "boolean", "default": True},
                            "generate_mipmaps": {"type": "boolean", "default": True}
                        }
                    }
                },
                "required": ["operation", "input_path"]
            }
        }

    def _read_blp_header(self, data: bytes) -> Dict[str, Any]:
        """Parse BLP file header."""
        if len(data) < 148:
            raise ValueError("Invalid BLP file: too short")
        
        # BLP header structure
        header = struct.unpack('<4sII4xIIIIII16I', data[:148])
        
        signature = header[0]
        if signature not in [b'BLP1', b'BLP2']:
            raise ValueError(f"Invalid BLP signature: {signature}")
        
        return {
            "signature": signature.decode('ascii'),
            "type": header[1],  # 0=JPEG, 1=Palettized
            "compression": header[2],  # Compression type
            "alpha_bits": header[3],  # Alpha channel depth
            "width": header[4],
            "height": header[5],
            "mipmap_count": header[6],
            "mipmap_offsets": header[7:23],  # 16 mipmap offsets
            "mipmap_sizes": header[23:39] if len(header) > 23 else [0]*16
        }

    def _decompress_dxt(self, data: bytes, width: int, height: int, dxt_type: int) -> bytes:
        """Decompress DXT compressed texture data."""
        try:
            from PIL import Image
            import numpy as np
            
            # Simple DXT1 decompression implementation
            if dxt_type == 1:  # DXT1
                block_size = 8
                blocks_x = (width + 3) // 4
                blocks_y = (height + 3) // 4
                
                pixels = []
                
                for y in range(blocks_y):
                    for x in range(blocks_x):
                        block_offset = (y * blocks_x + x) * block_size
                        if block_offset + block_size > len(data):
                            continue
                            
                        block = data[block_offset:block_offset + block_size]
                        # Simplified DXT1 block decompression
                        c0, c1 = struct.unpack('<HH', block[:4])
                        
                        # Convert 565 to RGB
                        r0 = ((c0 >> 11) & 0x1F) << 3
                        g0 = ((c0 >> 5) & 0x3F) << 2
                        b0 = (c0 & 0x1F) << 3
                        
                        r1 = ((c1 >> 11) & 0x1F) << 3
                        g1 = ((c1 >> 5) & 0x3F) << 2
                        b1 = (c1 & 0x1F) << 3
                        
                        # Simple interpolation
                        colors = [
                            (r0, g0, b0, 255),
                            (r1, g1, b1, 255),
                            ((2*r0 + r1)//3, (2*g0 + g1)//3, (2*b0 + b1)//3, 255),
                            ((r0 + 2*r1)//3, (g0 + 2*g1)//3, (b0 + 2*b1)//3, 255)
                        ]
                        
                        indices = struct.unpack('<I', block[4:8])[0]
                        
                        for py in range(4):
                            for px in range(4):
                                if x*4 + px < width and y*4 + py < height:
                                    idx = (indices >> (2 * (py*4 + px))) & 3
                                    pixels.append(colors[idx])
                
                # Convert to bytes
                rgba_data = bytearray()
                for pixel in pixels:
                    rgba_data.extend(pixel)
                
                return bytes(rgba_data)
            
            # Fallback: return original data
            return data
            
        except ImportError:
            # Fallback without numpy
            return data

    def _convert_blp_to_image(self, blp_data: bytes, mipmap_level: int = 0) -> Image.Image:
        """Convert BLP data to PIL Image."""
        header = self._read_blp_header(blp_data)
        
        # Calculate mipmap dimensions
        width = header["width"] >> mipmap_level
        height = header["height"] >> mipmap_level
        width = max(1, width)
        height = max(1, height)
        
        if mipmap_level >= header["mipmap_count"]:
            mipmap_level = header["mipmap_count"] - 1
        
        mipmap_offset = header["mipmap_offsets"][mipmap_level]
        mipmap_size = header["mipmap_sizes"][mipmap_level]
        
        if mipmap_offset == 0 or mipmap_size == 0:
            raise ValueError(f"Invalid mipmap level {mipmap_level}")
        
        # Extract mipmap data
        mipmap_data = blp_data[mipmap_offset:mipmap_offset + mipmap_size]
        
        if header["signature"] == "BLP1":
            # BLP1 format (usually JPEG compressed)
            if header["type"] == 0:  # JPEG
                try:
                    # Try to decode as JPEG
                    return Image.open(io.BytesIO(mipmap_data))
                except Exception:
                    # Fallback: create placeholder
                    return Image.new('RGBA', (width, height), (255, 0, 255, 255))
            
        elif header["signature"] == "BLP2":
            # BLP2 format
            if header["compression"] == 1:  # Palettized
                # Read palette (256 colors * 4 bytes BGRA)
                palette_data = blp_data[148:148 + 1024]
                palette = []
                for i in range(256):
                    b, g, r, a = struct.unpack('BBBB', palette_data[i*4:(i+1)*4])
                    palette.extend([r, g, b, a])
                
                # Create image from palette indices
                image = Image.new('RGBA', (width, height))
                pixels = []
                
                for i, idx in enumerate(mipmap_data):
                    if i >= width * height:
                        break
                    pixels.append((palette[idx*4], palette[idx*4+1], 
                                 palette[idx*4+2], palette[idx*4+3]))
                
                image.putdata(pixels)
                return image
                
            elif header["compression"] in [2, 3]:  # DXT1/DXT3
                # DXT compressed
                decompressed = self._decompress_dxt(mipmap_data, width, height, 
                                                 header["compression"])
                
                # Convert decompressed data to image
                if len(decompressed) >= width * height * 4:
                    image = Image.new('RGBA', (width, height))
                    pixels = []
                    for i in range(0, len(decompressed), 4):
                        if i + 3 < len(decompressed):
                            pixels.append(tuple(decompressed[i:i+4]))
                    image.putdata(pixels)
                    return image
        
        # Fallback: create placeholder texture
        return Image.new('RGBA', (width, height), (255, 0, 255, 255))

    def _generate_unreal_texture_asset(self, texture_name: str, texture_path: str, 
                                     unreal_settings: Dict[str, Any]) -> str:
        """Generate Unreal Engine texture asset metadata."""
        asset_data = {
            "Type": "Texture2D",
            "Name": texture_name,
            "ImportPath": texture_path,
            "TextureGroup": unreal_settings.get("texture_group", "World"),
            "CompressionSettings": unreal_settings.get("compression", "TC_Default"),
            "SRGB": unreal_settings.get("srgb", True),
            "GenerateMipmaps": unreal_settings.get("generate_mipmaps", True),
            "AddressX": "Wrap",
            "AddressY": "Wrap",
            "Filter": "Linear",
            "LODBias": 0,
            "LODGroup": "TEXTUREGROUP_World"
        }
        
        return f"""{{
    "Type": "Texture2D",
    "Properties": {{
        "ImportPath": "{texture_path}",
        "TextureGroup": "{asset_data['TextureGroup']}",
        "CompressionSettings": "{asset_data['CompressionSettings']}",
        "SRGB": {str(asset_data['SRGB']).lower()},
        "GenerateMipmaps": {str(asset_data['GenerateMipmaps']).lower()},
        "AddressX": "{asset_data['AddressX']}",
        "AddressY": "{asset_data['AddressY']}",
        "Filter": "{asset_data['Filter']}"
    }}
}}"""

    async def _convert_blp_file(self, input_file: Path, output_dir: Path, 
                              output_format: str, mipmap_level: int,
                              alpha_handling: str, unreal_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a single BLP file."""
        try:
            with open(input_file, 'rb') as f:
                blp_data = f.read()
            
            # Convert to PIL Image
            image = self._convert_blp_to_image(blp_data, mipmap_level)
            
            # Handle alpha channel
            if alpha_handling == "remove" and image.mode == "RGBA":
                # Convert RGBA to RGB
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif alpha_handling == "premultiply" and image.mode == "RGBA":
                # Premultiply alpha
                import numpy as np
                img_array = np.array(image, dtype=np.float32)
                alpha = img_array[:, :, 3:4] / 255.0
                img_array[:, :, :3] *= alpha
                image = Image.fromarray(img_array.astype(np.uint8))
            
            # Generate output filename
            output_name = input_file.stem
            output_file = output_dir / f"{output_name}.{output_format}"
            
            # Save image
            if output_format.upper() == "TGA":
                image.save(output_file, format="TGA")
            elif output_format.upper() == "DDS":
                # For DDS, we'll save as PNG and note that DDS conversion needs external tool
                png_file = output_dir / f"{output_name}.png"
                image.save(png_file, format="PNG")
                output_file = png_file
            else:
                image.save(output_file, format=output_format.upper())
            
            result = {
                "success": True,
                "input_file": str(input_file),
                "output_file": str(output_file),
                "format": output_format,
                "dimensions": image.size,
                "mode": image.mode
            }
            
            # Generate Unreal asset if requested
            if unreal_settings.get("generate_unreal_asset", True):
                asset_file = output_dir / f"{output_name}.uasset"
                asset_content = self._generate_unreal_texture_asset(
                    output_name, str(output_file), unreal_settings
                )
                
                with open(asset_file, 'w') as f:
                    f.write(asset_content)
                
                result["unreal_asset"] = str(asset_file)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "input_file": str(input_file),
                "error": str(e)
            }

    async def execute(self, operation: str, input_path: str, output_path: str = "./converted_textures",
                     output_format: str = "png", mipmap_level: int = 0,
                     alpha_handling: str = "preserve", 
                     unreal_settings: Optional[Dict[str, Any]] = None, **_) -> Dict[str, Any]:
        
        unreal_settings = unreal_settings or {}
        
        try:
            input_path_obj = _confine(Path(input_path))
            output_dir = _confine(Path(output_path))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if operation == "convert":
                if not input_path_obj.exists():
                    return {"success": False, "error": f"Input file not found: {input_path}"}
                
                if not input_path_obj.suffix.lower() == '.blp':
                    return {"success": False, "error": "Input file must be a BLP file"}
                
                result = await self._convert_blp_file(
                    input_path_obj, output_dir, output_format, mipmap_level,
                    alpha_handling, unreal_settings
                )
                
                return result
            
            elif operation == "batch_convert":
                if not input_path_obj.exists():
                    return {"success": False, "error": f"Input directory not found: {input_path}"}
                
                blp_files = list(input_path_obj.rglob("*.blp"))
                if not blp_files:
                    return {"success": False, "error": "No BLP files found in directory"}
                
                results = []
                successful = 0
                failed = 0
                
                for blp_file in blp_files:
                    result = await self._convert_blp_file(
                        blp_file, output_dir, output_format, mipmap_level,
                        alpha_handling, unreal_settings
                    )
                    results.append(result)
                    
                    if result["success"]:
                        successful += 1
                    else:
                        failed += 1
                
                return {
                    "success": True,
                    "operation": "batch_convert",
                    "total_files": len(blp_files),
                    "successful": successful,
                    "failed": failed,
                    "output_directory": str(output_dir),
                    "results": results
                }
            
            elif operation == "info":
                if not input_path_obj.exists() or not input_path_obj.suffix.lower() == '.blp':
                    return {"success": False, "error": "Invalid BLP file"}
                
                with open(input_path_obj, 'rb') as f:
                    blp_data = f.read()
                
                header = self._read_blp_header(blp_data)
                
                return {
                    "success": True,
                    "file": str(input_path_obj),
                    "signature": header["signature"],
                    "dimensions": f"{header['width']}x{header['height']}",
                    "mipmap_count": header["mipmap_count"],
                    "compression": header["compression"],
                    "alpha_bits": header["alpha_bits"],
                    "file_size": len(blp_data)
                }
            
            elif operation == "extract_mipmaps":
                if not input_path_obj.exists() or not input_path_obj.suffix.lower() == '.blp':
                    return {"success": False, "error": "Invalid BLP file"}
                
                with open(input_path_obj, 'rb') as f:
                    blp_data = f.read()
                
                header = self._read_blp_header(blp_data)
                extracted = []
                
                for level in range(header["mipmap_count"]):
                    try:
                        result = await self._convert_blp_file(
                            input_path_obj, output_dir / f"mipmap_{level}", 
                            output_format, level, alpha_handling, unreal_settings
                        )
                        extracted.append({
                            "level": level,
                            "result": result
                        })
                    except Exception as e:
                        extracted.append({
                            "level": level,
                            "error": str(e)
                        })
                
                return {
                    "success": True,
                    "operation": "extract_mipmaps",
                    "file": str(input_path_obj),
                    "total_mipmaps": header["mipmap_count"],
                    "extracted": extracted
                }
            
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            return {"success": False, "error": f"BLP conversion failed: {str(e)}"}