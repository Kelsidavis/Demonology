
# demonology/tools/media.py (patched)
from __future__ import annotations

import io
import os
import re
import json
import time
import math
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import requests
from PIL import Image, ImageStat, ImageOps, ImageFilter

try:
    import pytesseract  # optional
    _HAS_TESS = True
except Exception:
    _HAS_TESS = False

from .base import Tool

logger = logging.getLogger(__name__)

# ---------- Workspace confinement (mirrors audio.py) ----------
WORKSPACE_ROOT = Path(os.environ.get("DEMONOLOGY_ROOT", os.getcwd())).resolve()

def _safe_path(path: str, want_dir: bool = False) -> Path:
    if not path:
        raise ValueError("Empty path")
    p = Path(path)
    p = (WORKSPACE_ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    try:
        p.relative_to(WORKSPACE_ROOT)
    except Exception:
        raise PermissionError(f"Path escapes workspace root: {p}")
    # disallow symlink traversal (best effort)
    for parent in [p] + list(p.parents):
        try:
            if parent.is_symlink():
                raise PermissionError(f"Symlinked path not allowed: {parent}")
        except FileNotFoundError:
            pass
    if want_dir:
        p.mkdir(parents=True, exist_ok=True)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p

# ---------- HTTP helpers ----------
DEFAULT_TIMEOUT = float(os.environ.get("DEMONOLOGY_HTTP_TIMEOUT", "30"))

def _http_get(url: str, **kwargs) -> requests.Response:
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
    return requests.get(url, **kwargs)

def _http_post(url: str, **kwargs) -> requests.Response:
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
    return requests.post(url, **kwargs)

# ============================================================
# Image Generation Tool
# ============================================================

class ImageGenerationTool(Tool):
    """
    Generate images via free + tokened backends with safe workspace outputs.
    Backends (in order): Pollinations (no token), Hugging Face (HF_TOKEN), Craiyon.
    """

    def __init__(self):
        super().__init__("image_generate", "Generate an image from a text prompt using multiple backends with fallbacks.")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Short visual description"},
                    "style": {"type": "string", "description": "Optional style tag, e.g., 'photorealistic', 'watercolor'"},
                    "size": {"type": "string", "description": "WxH like '768x768' (some backends ignore)", "default": "768x768"},
                    "content_type": {"type": "string", "enum": ["scene","object","texture","icon","ui"], "default": "scene"},
                    "output_file": {"type": "string", "description": "Relative path under workspace", "default": "images/generated.png"}
                },
                "required": ["prompt", "output_file"]
            }
        }

    async def execute(self, prompt: str, style: Optional[str] = None, size: str = "768x768",
                      content_type: str = "scene", output_file: str = "images/generated.png", **_) -> Dict[str, Any]:
        try:
            out_path = _safe_path(output_file, want_dir=False)
        except Exception as e:
            return {"success": False, "error": f"Invalid output path: {e}"}

        prompt2 = self._enhance_prompt(prompt, style, content_type)
        backends_report: List[Dict[str, Any]] = []

        # Try Pollinations
        img = None
        try:
            img, meta = self._pollinations(prompt2, size)
            backends_report.append(meta)
        except Exception as e:
            backends_report.append({"api": "pollinations", "ok": False, "error": str(e)})
            img = None

        # Try Hugging Face if Pollinations failed
        if img is None:
            try:
                img, meta = self._huggingface(prompt2, size)
                backends_report.append(meta)
            except Exception as e:
                backends_report.append({"api": "huggingface", "ok": False, "error": str(e)})
                img = None

        # Try Craiyon if still None
        if img is None:
            try:
                img, meta = self._craiyon(prompt2)
                backends_report.append(meta)
            except Exception as e:
                backends_report.append({"api": "craiyon", "ok": False, "error": str(e)})
                img = None

        if img is None:
            return {
                "success": False,
                "error": "All image generation backends failed",
                "backends": backends_report
            }

        try:
            img.save(str(out_path))
        except Exception as e:
            return {"success": False, "error": f"Failed to save image: {e}"}

        return {
            "success": True,
            "operation": "image_generate",
            "prompt": prompt,
            "prompt_enhanced": prompt2,
            "size": size,
            "output_file": str(out_path.relative_to(WORKSPACE_ROOT)),
            "backends": backends_report
        }

    # ----- prompt shaping -----
    def _enhance_prompt(self, prompt: str, style: Optional[str], content_type: str) -> str:
        p = prompt.strip()
        if style:
            p = f"{style} {p}"
        suffix_map = {
            "object": "isolated object, studio background",
            "texture": "tileable seamless texture",
            "icon": "vector-like icon, simple flat design",
            "ui": "clean UI screenshot, sharp text, high contrast",
            "scene": "cinematic lighting, detailed scene"
        }
        suffix = suffix_map.get(content_type, "detailed")
        return f"{p}, {suffix}"

    # ----- backends -----
    def _pollinations(self, prompt: str, size: str) -> Tuple[Image.Image, Dict[str, Any]]:
        # Pollinations simple GET
        url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}?nologo=true&size={size}"
        r = _http_get(url, stream=True, headers={"Accept": "image/*"})
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        meta = {"api": "pollinations", "ok": True, "status": r.status_code, "bytes": len(r.content)}
        return img, meta

    def _huggingface(self, prompt: str, size: str) -> Tuple[Image.Image, Dict[str, Any]]:
        token = os.environ.get("HUGGINGFACE_TOKEN")
        if not token:
            raise RuntimeError("HUGGINGFACE_TOKEN not set")
        # Use a public SDXL-like inference endpoint (user-configurable via env if desired)
        model_url = os.environ.get("HF_INFERENCE_URL", "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1")
        headers = {"Authorization": f"Bearer {token}", "Accept": "image/png"}
        payload = {"inputs": prompt, "options": {"wait_for_model": True}}
        r = _http_post(model_url, headers=headers, data=json.dumps(payload))
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        meta = {"api": "huggingface", "ok": True, "status": r.status_code, "bytes": len(r.content)}
        return img, meta

    def _craiyon(self, prompt: str) -> Tuple[Image.Image, Dict[str, Any]]:
        # Craiyon legacy public API
        r = _http_post("https://backend.craiyon.com/generate", json={"prompt": prompt})
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        data = r.json()
        # Craiyon returns multiple base64 images; pick the first
        import base64
        if not data or "images" not in data or not data["images"]:
            raise RuntimeError("No images in response")
        b64 = data["images"][0]
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        meta = {"api": "craiyon", "ok": True}
        return img, meta

# ============================================================
# Image Analysis Tool
# ============================================================

class ImageAnalysisTool(Tool):
    """
    Analyze an image with safe pathing and resilient sub-steps.
    Returns compact JSON and writes a .analysis.txt next to the image (in workspace).
    """

    def __init__(self):
        super().__init__("image_analyze", "Analyze images: meta, colors, brightness, simple UI heuristics, OCR (optional).")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path under workspace to an image file"},
                    "run_ocr": {"type": "boolean", "description": "Try OCR (if pytesseract present)", "default": True},
                    "save_report": {"type": "boolean", "description": "Write a .analysis.txt file near the image", "default": True}
                },
                "required": ["image_path"]
            }
        }

    async def execute(self, image_path: str, run_ocr: bool = True, save_report: bool = True, **_) -> Dict[str, Any]:
        try:
            img_path = _safe_path(image_path, want_dir=False)
        except Exception as e:
            return {"success": False, "error": f"Invalid input path: {e}"}
        if not img_path.exists():
            return {"success": False, "error": f"File not found: {img_path}"}

        analysis: Dict[str, Any] = {"success": True, "operation": "image_analyze", "file": str(img_path.relative_to(WORKSPACE_ROOT))}
        tech: Dict[str, Any] = {"errors": []}
        analysis["technical_details"] = tech

        # Load
        try:
            with Image.open(str(img_path)) as im:
                img = im.convert("RGB")
        except Exception as e:
            return {"success": False, "error": f"Could not open image: {e}"}

        # Basic metadata
        try:
            analysis["meta"] = {
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "format": Image.open(str(img_path)).format or "PNG"
            }
        except Exception as e:
            tech["errors"].append(f"meta: {e}")

        # Dominant colors (coarse k-means-ish via resize + histogram)
        try:
            small = img.resize((64, 64), Image.BILINEAR)
            colors = small.getcolors(maxcolors=64*64)
            colors = sorted(colors or [], key=lambda x: x[0], reverse=True)[:5]
            analysis["dominant_colors"] = [{"count": c, "rgb": rgb} for c, rgb in colors]
        except Exception as e:
            tech["errors"].append(f"colors: {e}")

        # Brightness / contrast
        try:
            stat = ImageStat.Stat(img)
            mean = sum(stat.mean) / len(stat.mean)
            rms = sum(stat.rms) / len(stat.rms)
            analysis["luminance"] = {"mean": mean, "rms": rms}
        except Exception as e:
            tech["errors"].append(f"luminance: {e}")

        # Simple UI / screenshot heuristic
        try:
            w, h = img.width, img.height
            ratio = w / max(1, h)
            is_rectilinear = (w % 2 == 0) and (h % 2 == 0)
            likely_ui = (ratio > 1.2 and ratio < 2.0 and is_rectilinear) or (w >= 1000 and h >= 700)
            analysis["ui_heuristic"] = {"likely_ui": bool(likely_ui), "ratio": ratio, "even_dims": is_rectilinear}
        except Exception as e:
            tech["errors"].append(f"ui_heuristic: {e}")

        # OCR (optional)
        ocr_text = None
        if run_ocr:
            if not _HAS_TESS:
                tech["errors"].append("OCR requested but pytesseract not installed")
            else:
                try:
                    gray = ImageOps.grayscale(img)
                    sharp = gray.filter(ImageFilter.SHARPEN)
                    ocr_text = pytesseract.image_to_string(sharp) or ""
                    analysis["ocr"] = {"text": ocr_text.strip()[:5000], "truncated": len(ocr_text) > 5000}
                except Exception as e:
                    tech["errors"].append(f"OCR: {e}")

        # Visual description (very light heuristic text)
        try:
            desc = self._describe(img)
            analysis["description"] = desc
        except Exception as e:
            tech["errors"].append(f"description: {e}")

        # Save a lightweight report (optional)
        if save_report:
            try:
                report_name = img_path.with_suffix(img_path.suffix + ".analysis.txt").name
                report_path = _safe_path(str(Path(analysis["file"]).with_suffix(Path(analysis["file"]).suffix + ".analysis.txt")), want_dir=False)
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(self._format_report(analysis))
                analysis["report_file"] = str(report_path.relative_to(WORKSPACE_ROOT))
            except Exception as e:
                tech["errors"].append(f"report: {e}")

        return analysis

    # ----- helpers -----
    def _describe(self, img: Image.Image) -> str:
        w, h = img.width, img.height
        stat = ImageStat.Stat(img)
        mean = sum(stat.mean)/len(stat.mean)
        palette_hint = "vivid" if mean > 140 else "muted" if mean < 100 else "balanced"
        aspect = "wide" if w > h else "tall" if h > w else "square"
        return f"{aspect} image, {palette_hint} tones, {w}×{h}px"

    def _format_report(self, a: Dict[str, Any]) -> str:
        lines = []
        lines.append(f"Image: {a.get('file','?')}")
        m = a.get("meta", {})
        lines.append(f"Size: {m.get('width','?')}x{m.get('height','?')}  Mode: {m.get('mode','?')}  Format: {m.get('format','?')}")
        if "dominant_colors" in a:
            cols = ", ".join([str(x["rgb"]) for x in a["dominant_colors"]])
            lines.append(f"Dominant colors: {cols}")
        if "luminance" in a:
            lines.append(f"Luminance mean: {a['luminance'].get('mean','?'):.2f}  rms: {a['luminance'].get('rms','?'):.2f}")
        if "ui_heuristic" in a:
            uh = a["ui_heuristic"]
            lines.append(f"UI-like: {uh.get('likely_ui')} (ratio {uh.get('ratio'):.2f}, even_dims={uh.get('even_dims')})")
        if "ocr" in a:
            lines.append("OCR snippet:")
            txt = a["ocr"].get("text","").strip().splitlines()
            lines.extend(("  " + line) for line in txt[:6])
            if a["ocr"].get("truncated"):
                lines.append("  ... [truncated]")
        if a.get("technical_details", {}).get("errors"):
            lines.append("Errors:")
            for e in a["technical_details"]["errors"]:
                lines.append(f"  - {e}")
        return "\n".join(lines)
