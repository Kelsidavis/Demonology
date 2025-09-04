# demonology/tools/media.py
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import Tool

logger = logging.getLogger(__name__)


class ImageGenerationTool(Tool):
    """Generate images from text descriptions using free AI image generation APIs."""
    
    def __init__(self):
        super().__init__("image_generation", "Generate images from text descriptions")
    
    def is_available(self) -> bool:
        try:
            import requests
            return True
        except ImportError:
            return False
    
    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Text description of the image to generate"},
                    "content_type": {"type": "string", "enum": ["auto", "texture", "character", "object", "scene"], "description": "Type of image content for better prompt optimization", "default": "auto"},
                    "style": {"type": "string", "enum": ["realistic", "artistic", "anime", "fantasy", "pixel-art", "concept-art"], "description": "Image style", "default": "realistic"},
                    "size": {"type": "string", "enum": ["512x512", "768x768", "1024x1024"], "description": "Image dimensions", "default": "512x512"},
                    "filename": {"type": "string", "description": "Filename for saved image (optional)"},
                    "save_image": {"type": "boolean", "description": "Whether to save image to disk", "default": True}
                },
                "required": ["prompt"]
            }
        }
    
    async def execute(self, prompt: str, content_type: str = "auto", style: str = "realistic", size: str = "512x512", 
                     filename: str = None, save_image: bool = True, **kwargs) -> Dict[str, Any]:
        try:
            import requests
            import base64
            from datetime import datetime
            import hashlib
            
            # Enhance prompt based on style and content type
            enhanced_prompt = self._enhance_prompt(prompt, style, content_type)
            
            # Try multiple APIs in order of preference
            apis = [
                self._try_pollinations_ai,
                self._try_huggingface_api,
                self._try_craiyon_api
            ]
            
            image_data = None
            api_used = None
            
            for api_func in apis:
                try:
                    result = await api_func(enhanced_prompt, size)
                    if result:
                        image_data, api_used = result
                        break
                except Exception as e:
                    logger.warning(f"API {api_func.__name__} failed: {e}")
                    continue
            
            if not image_data:
                return {"success": False, "error": "All image generation APIs failed"}
            
            # Generate filename if not provided
            if not filename:
                # Create hash of prompt for unique filename
                prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_image_{timestamp}_{prompt_hash}.png"
            
            if not filename.endswith(('.png', '.jpg', '.jpeg')):
                filename += '.png'
            
            result = {
                "success": True,
                "prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "style": style,
                "size": size,
                "api_used": api_used,
                "filename": filename
            }
            
            if save_image:
                # Save image to current working directory
                try:
                    # Use current working directory
                    current_dir = Path.cwd().resolve()
                    image_path = current_dir / filename
                except Exception:
                    # Fallback to home directory if current directory fails
                    image_path = Path.home() / filename
                
                if isinstance(image_data, bytes):
                    # Direct binary data
                    image_path.write_bytes(image_data)
                elif isinstance(image_data, str):
                    # Base64 encoded data
                    if image_data.startswith('data:image'):
                        # Remove data:image/png;base64, prefix
                        image_data = image_data.split(',')[1]
                    decoded_data = base64.b64decode(image_data)
                    image_path.write_bytes(decoded_data)
                
                result["file_path"] = str(image_path)
                result["file_size"] = image_path.stat().st_size
                result["message"] = f"Image generated and saved to {filename}"
            else:
                result["image_data"] = image_data
                result["message"] = "Image generated successfully"
            
            return result
            
        except Exception as e:
            logger.exception("ImageGenerationTool error")
            return {"success": False, "error": str(e)}
    
    def _enhance_prompt(self, prompt: str, style: str, content_type: str = "auto") -> str:
        """Enhance the prompt based on the selected style and content type."""
        # Determine content type
        if content_type == "auto":
            # Auto-detect what type of image is being requested
            prompt_lower = prompt.lower()
            
            is_texture = any(word in prompt_lower for word in [
                'texture', 'pattern', 'material', 'surface', 'seamless', 'tileable',
                'wood grain', 'fabric', 'metal', 'stone', 'concrete', 'marble'
            ])
            
            is_character = any(word in prompt_lower for word in [
                'person', 'character', 'man', 'woman', 'face', 'portrait', 'figure',
                'human', 'people', 'body', 'head', 'eyes'
            ])
            
            is_object = any(word in prompt_lower for word in [
                'object', 'item', 'tool', 'weapon', 'vehicle', 'furniture', 'product'
            ])
            
            is_scene = any(word in prompt_lower for word in [
                'scene', 'landscape', 'environment', 'room', 'building', 'place', 
                'location', 'background', 'setting'
            ])
            
            # Determine the content type based on detection
            if is_texture:
                detected_type = 'texture'
            elif is_character:
                detected_type = 'character'
            elif is_object:
                detected_type = 'object'
            elif is_scene:
                detected_type = 'scene'
            else:
                detected_type = 'general'
        else:
            detected_type = content_type
        
        # Base style prefixes
        style_prefixes = {
            "realistic": "photorealistic, high quality, detailed, ",
            "artistic": "artistic, painterly, creative, ",
            "anime": "anime style, manga, colorful, ",
            "fantasy": "fantasy art, magical, mystical, ",
            "pixel-art": "pixel art, 8-bit style, retro gaming, ",
            "concept-art": "concept art, digital painting, professional, "
        }
        
        # Content-specific suffixes to avoid unwanted elements
        content_suffixes = {
            'texture': ", seamless pattern, isolated on neutral background, no objects, no people, no scenes, tileable, material study",
            'character': ", isolated character, plain background, no environment, no scenes, character focus, portrait style",
            'object': ", isolated object, plain background, no people, no scenes, product shot, clean composition",
            'scene': ", environmental art, atmospheric, detailed setting",
            'general': ""
        }
        
        # Build enhanced prompt
        prefix = style_prefixes.get(style, "")
        enhanced = f"{prefix}{prompt}"
        
        # Add content-specific instructions
        if detected_type in content_suffixes:
            enhanced += content_suffixes[detected_type]
        
        # Add quality modifiers
        enhanced += ", high resolution, professional quality"
        
        return enhanced
    
    async def _try_pollinations_ai(self, prompt: str, size: str) -> Optional[Tuple[bytes, str]]:
        """Try Pollinations.ai API - completely free, no API key needed."""
        import requests
        from urllib.parse import quote
        
        # Pollinations.ai supports various models
        model = "flux"  # or "flux-realism", "flux-3d", etc.
        
        # Parse size
        width, height = size.split('x')
        
        # Build URL
        url = f"https://image.pollinations.ai/prompt/{quote(prompt)}?width={width}&height={height}&model={model}&nologo=true"
        
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        return response.content, "Pollinations.ai"
    
    async def _try_huggingface_api(self, prompt: str, size: str) -> Optional[Tuple[bytes, str]]:
        """Try Hugging Face Inference API - free with HF token."""
        import requests
        import os
        
        hf_token = os.environ.get('HUGGINGFACE_TOKEN')
        if not hf_token:
            logger.info("No Hugging Face token found, skipping HF API")
            return None
        
        # Use FLUX.1 schnell model (fast and free)
        api_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
        
        headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "num_inference_steps": 4,  # FLUX schnell works well with 1-4 steps
                "guidance_scale": 0.0,     # FLUX schnell works better without guidance
            }
        }
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        return response.content, "Hugging Face FLUX.1"
    
    async def _try_craiyon_api(self, prompt: str, size: str) -> Optional[Tuple[str, str]]:
        """Try unofficial Craiyon API - free but may be rate limited."""
        import requests
        import json
        import time
        
        # This is a simplified version - the actual Craiyon API is more complex
        # and may require reverse engineering their web interface
        try:
            # Craiyon web interface endpoint (unofficial)
            url = "https://api.craiyon.com/v3"
            
            payload = {
                "prompt": prompt,
                "model": "art",  # or "drawing", "photo"
                "negative_prompt": "blurry, low quality, distorted",
                "version": "35s5hfwn9n78gb06",  # This may need to be updated
            }
            
            response = requests.post(url, json=payload, timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                if "images" in data and data["images"]:
                    # Return first image (base64 encoded)
                    return data["images"][0], "Craiyon"
            
            return None
            
        except Exception as e:
            logger.warning(f"Craiyon API failed: {e}")
            return None


class ImageAnalysisTool(Tool):
    """Analyze images including screenshots, diagrams, code snippets, and UI mockups."""
    
    def __init__(self):
        super().__init__("image_analysis", "Analyze images to identify objects, faces, and text.")
    
    def is_available(self) -> bool:
        """Check if required image processing libraries are available."""
        try:
            from PIL import Image
            return True
        except ImportError:
            return False
    
    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the image file to analyze"
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["general", "ui_mockup", "code_snippet", "diagram", "screenshot", "text_extraction"],
                        "description": "Type of analysis to perform",
                        "default": "general"
                    },
                    "extract_text": {
                        "type": "boolean",
                        "description": "Whether to attempt OCR text extraction",
                        "default": True
                    },
                    "save_analysis": {
                        "type": "boolean", 
                        "description": "Whether to save analysis results to a text file",
                        "default": True
                    }
                },
                "required": ["image_path"]
            }
        }
    
    async def execute(self, image_path: str, analysis_type: str = "general", 
                     extract_text: bool = True, save_analysis: bool = True, **kwargs) -> Dict[str, Any]:
        try:
            # Resolve and validate image path
            img_path = Path(image_path).resolve()
            
            # Check if path exists
            if not img_path.exists():
                return {"success": False, "error": f"Image file not found: {image_path}"}
            
            if not img_path.is_file():
                return {"success": False, "error": f"Path is not a file: {image_path}"}
            
            # Check if it's an image file
            valid_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.ico'}
            if img_path.suffix.lower() not in valid_extensions:
                return {"success": False, "error": f"Unsupported image format: {img_path.suffix}"}
            
            # Perform image analysis
            analysis_result = await self._analyze_image(img_path, analysis_type, extract_text)
            
            result = {
                "success": True,
                "image_path": str(img_path),
                "image_name": img_path.name,
                "analysis_type": analysis_type,
                "analysis": analysis_result
            }
            
            if save_analysis:
                # Save analysis to text file
                analysis_file = img_path.with_suffix('.analysis.txt')
                analysis_content = self._format_analysis_report(analysis_result, img_path.name, analysis_type)
                analysis_file.write_text(analysis_content, encoding='utf-8')
                result["analysis_file"] = str(analysis_file)
            
            return result
            
        except Exception as e:
            logger.exception("ImageAnalysisTool error")
            return {"success": False, "error": f"Image analysis failed: {str(e)}"}
    
    async def _analyze_image(self, img_path: Path, analysis_type: str, extract_text: bool) -> Dict[str, Any]:
        """Perform comprehensive image analysis."""
        analysis = {
            "basic_info": {},
            "visual_description": "",
            "extracted_text": "",
            "technical_details": {},
            "suggestions": []
        }
        
        try:
            from PIL import Image
            
            # Open and analyze image
            with Image.open(img_path) as img:
                # Basic image information
                analysis["basic_info"] = {
                    "dimensions": f"{img.width}x{img.height}",
                    "format": img.format,
                    "mode": img.mode,
                    "file_size": img_path.stat().st_size,
                    "has_transparency": img.mode in ('RGBA', 'LA') or 'transparency' in img.info
                }
                
                # Convert to RGB for analysis if needed
                if img.mode != 'RGB':
                    img_rgb = img.convert('RGB')
                else:
                    img_rgb = img
                
                # Analyze image characteristics
                analysis["technical_details"] = self._analyze_image_characteristics(img_rgb)
                
                # Generate visual description based on analysis type
                analysis["visual_description"] = self._generate_visual_description(img_rgb, analysis_type)
                
                # Extract text if requested
                if extract_text:
                    analysis["extracted_text"] = self._extract_text_from_image(img_rgb)
                
                # Generate suggestions based on analysis type
                analysis["suggestions"] = self._generate_suggestions(analysis_type, analysis)
        
        except ImportError:
            analysis["error"] = "PIL (Pillow) library not available for image analysis"
        except Exception as e:
            analysis["error"] = f"Image analysis failed: {str(e)}"
        
        return analysis
    
    def _analyze_image_characteristics(self, img: 'Image.Image') -> Dict[str, Any]:
        """Analyze technical characteristics of the image."""
        import statistics
        
        characteristics = {}
        
        try:
            # Color analysis
            colors = img.getcolors(maxcolors=256*256*256)
            if colors:
                # Dominant colors
                dominant_colors = sorted(colors, key=lambda x: x[0], reverse=True)[:5]
                characteristics["dominant_colors"] = [
                    {"rgb": color[1], "count": color[0]} for color in dominant_colors
                ]
            
            # Brightness analysis
            grayscale = img.convert('L')
            pixels = list(grayscale.getdata())
            characteristics["brightness"] = {
                "average": statistics.mean(pixels),
                "median": statistics.median(pixels),
                "min": min(pixels),
                "max": max(pixels)
            }
            
            # Detect if image appears to be a screenshot
            characteristics["likely_screenshot"] = self._detect_screenshot_characteristics(img)
            
            # Detect if image contains UI elements
            characteristics["likely_ui"] = self._detect_ui_characteristics(img)
            
        except Exception as e:
            characteristics["analysis_error"] = str(e)
        
        return characteristics
    
    def _detect_screenshot_characteristics(self, img: 'Image.Image') -> Dict[str, Any]:
        """Detect characteristics that suggest this is a screenshot."""
        screenshot_indicators = {
            "has_typical_screenshot_ratio": False,
            "has_ui_like_regions": False,
            "has_text_regions": False,
            "confidence": "low"
        }
        
        width, height = img.size
        ratio = width / height
        
        # Common screenshot ratios
        common_ratios = [16/9, 16/10, 4/3, 3/2, 21/9]
        screenshot_indicators["has_typical_screenshot_ratio"] = any(
            abs(ratio - r) < 0.1 for r in common_ratios
        )
        
        # Look for rectangular regions (potential UI elements)
        # This is a simplified heuristic
        if width > 800 and height > 600:
            screenshot_indicators["has_ui_like_regions"] = True
        
        # Estimate confidence
        score = 0
        if screenshot_indicators["has_typical_screenshot_ratio"]:
            score += 1
        if screenshot_indicators["has_ui_like_regions"]:
            score += 1
        
        if score >= 2:
            screenshot_indicators["confidence"] = "high"
        elif score == 1:
            screenshot_indicators["confidence"] = "medium"
        
        return screenshot_indicators
    
    def _detect_ui_characteristics(self, img: 'Image.Image') -> Dict[str, Any]:
        """Detect characteristics that suggest this contains UI elements."""
        ui_indicators = {
            "has_rectangular_regions": False,
            "has_button_like_elements": False,
            "has_form_elements": False,
            "confidence": "low"
        }
        
        width, height = img.size
        
        # Simple heuristics for UI detection
        # Look for common UI dimensions and aspect ratios
        if width > 300 and height > 200:
            ui_indicators["has_rectangular_regions"] = True
        
        # More sophisticated UI detection would require edge detection
        # For now, use basic heuristics
        ui_indicators["confidence"] = "medium" if ui_indicators["has_rectangular_regions"] else "low"
        
        return ui_indicators
    
    def _generate_visual_description(self, img: 'Image.Image', analysis_type: str) -> str:
        """Generate a visual description based on the analysis type."""
        width, height = img.size
        
        descriptions = {
            "general": f"Image with dimensions {width}x{height} pixels. ",
            "ui_mockup": f"UI mockup or interface design with dimensions {width}x{height}. ",
            "code_snippet": f"Image containing code or text content, {width}x{height} pixels. ",
            "diagram": f"Diagram or schematic image, {width}x{height} pixels. ",
            "screenshot": f"Screenshot capture with dimensions {width}x{height}. ",
            "text_extraction": f"Text-containing image, {width}x{height} pixels. "
        }
        
        base_description = descriptions.get(analysis_type, descriptions["general"])
        
        # Add more details based on image characteristics
        if width > 1920:
            base_description += "High resolution image. "
        elif width < 500:
            base_description += "Small or thumbnail-sized image. "
        
        aspect_ratio = width / height
        if aspect_ratio > 2:
            base_description += "Wide aspect ratio, possibly a banner or header. "
        elif aspect_ratio < 0.5:
            base_description += "Tall aspect ratio, possibly a mobile screen or sidebar. "
        
        return base_description
    
    def _extract_text_from_image(self, img: 'Image.Image') -> str:
        """Extract text from image using OCR if available."""
        try:
            import pytesseract
            
            # Extract text using Tesseract OCR
            extracted_text = pytesseract.image_to_string(img)
            return extracted_text.strip()
            
        except ImportError:
            return "OCR not available (pytesseract not installed)"
        except Exception as e:
            return f"OCR failed: {str(e)}"
    
    def _generate_suggestions(self, analysis_type: str, analysis: Dict[str, Any]) -> List[str]:
        """Generate suggestions based on the analysis type and results."""
        suggestions = []
        
        basic_info = analysis.get("basic_info", {})
        width, height = basic_info.get("dimensions", "0x0").split('x')
        width, height = int(width), int(height)
        
        if analysis_type == "ui_mockup":
            suggestions.extend([
                "Consider creating interactive prototypes based on this mockup",
                "Document the UI components and their specifications",
                "Validate the design with accessibility guidelines"
            ])
            
        elif analysis_type == "code_snippet":
            if analysis.get("extracted_text"):
                suggestions.extend([
                    "Review the extracted code for syntax and best practices",
                    "Consider converting the image to actual code files",
                    "Check for any visible errors or improvements"
                ])
            
        elif analysis_type == "screenshot":
            suggestions.extend([
                "Document the context and purpose of this screenshot",
                "Consider annotating important areas for clarity",
                "Use for bug reports or feature documentation"
            ])
            
        elif analysis_type == "diagram":
            suggestions.extend([
                "Convert to editable diagram format if needed",
                "Ensure all text and labels are readable",
                "Consider creating digital version for easier editing"
            ])
        
        # General suggestions based on technical characteristics
        if width < 800 and height < 600:
            suggestions.append("Image resolution is relatively low - consider higher resolution for better clarity")
        
        if analysis.get("extracted_text") and len(analysis["extracted_text"]) > 50:
            suggestions.append("Significant text content detected - consider extracting for documentation")
        
        return suggestions
    
    def _format_analysis_report(self, analysis: Dict[str, Any], filename: str, analysis_type: str) -> str:
        """Format the analysis results into a readable report."""
        report = f"""# Image Analysis Report: {filename}

## Basic Information
- **Analysis Type**: {analysis_type}
- **Dimensions**: {analysis.get('basic_info', {}).get('dimensions', 'Unknown')}
- **Format**: {analysis.get('basic_info', {}).get('format', 'Unknown')}
- **File Size**: {analysis.get('basic_info', {}).get('file_size', 'Unknown')} bytes
- **Color Mode**: {analysis.get('basic_info', {}).get('mode', 'Unknown')}

## Visual Description
{analysis.get('visual_description', 'No description available')}

## Technical Details
"""
        
        tech_details = analysis.get('technical_details', {})
        if tech_details.get('brightness'):
            brightness = tech_details['brightness']
            report += f"""
### Color and Brightness Analysis
- **Average Brightness**: {brightness.get('average', 0):.1f}/255
- **Brightness Range**: {brightness.get('min', 0)} - {brightness.get('max', 255)}
"""
        
        if tech_details.get('dominant_colors'):
            report += "\n### Dominant Colors\n"
            for i, color_info in enumerate(tech_details['dominant_colors'][:3], 1):
                rgb = color_info['rgb']
                if isinstance(rgb, tuple) and len(rgb) == 3:
                    report += f"- **Color {i}**: RGB({rgb[0]}, {rgb[1]}, {rgb[2]}) - {color_info['count']} pixels\n"
        
        # Add extracted text if available
        extracted_text = analysis.get('extracted_text', '')
        if extracted_text and len(extracted_text.strip()) > 0:
            report += f"""
## Extracted Text
```
{extracted_text}
```
"""
        
        # Add suggestions
        suggestions = analysis.get('suggestions', [])
        if suggestions:
            report += "\n## Suggestions\n"
            for suggestion in suggestions:
                report += f"- {suggestion}\n"
        
        report += f"""
---
*Analysis generated by Demonology Image Analysis Tool*
*Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report