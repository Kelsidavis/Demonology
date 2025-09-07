from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import Tool

logger = logging.getLogger(__name__)

# ---------- optional deps ----------
try:
    import numpy as np  # type: ignore
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False

try:
    import trimesh  # type: ignore
    from trimesh.creation import box as _box, cylinder as _cylinder, icosphere as _icosphere
    _HAS_TRIMESH = True
except Exception:
    _HAS_TRIMESH = False

try:
    from PIL import Image  # type: ignore
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

try:
    from shapely.geometry import Polygon, MultiPolygon  # type: ignore
    from shapely.ops import unary_union  # type: ignore
    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False

try:
    # only used when extruding from SVG
    from svgpathtools import svg2paths  # type: ignore
    _HAS_SVG = True
except Exception:
    _HAS_SVG = False

# ---------- workspace safety ----------
WORKSPACE_ROOT = Path(os.environ.get("DEMONOLOGY_ROOT", os.getcwd())).resolve()

def _safe_path(path: str, *, ensure_dir: bool = False) -> Path:
    if not path:
        raise ValueError("Empty path")
    p = Path(path)
    p = (WORKSPACE_ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    try:
        p.relative_to(WORKSPACE_ROOT)
    except Exception:
        raise PermissionError(f"Path escapes workspace: {p}")
    if ensure_dir:
        p.mkdir(parents=True, exist_ok=True)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p

# ---------- helpers ----------
@dataclass
class Transform:
    translate: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotate_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # XYZ in degrees
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)

def _apply_transform(mesh, tf: Transform):
    import math
    if tf is None:
        return mesh
    # scale
    S = np.diag([tf.scale[0], tf.scale[1], tf.scale[2], 1.0])
    mesh.apply_transform(S)
    # rotation (XYZ degrees)
    rx, ry, rz = [math.radians(a) for a in tf.rotate_deg]
    Rx = trimesh.transformations.rotation_matrix(rx, (1,0,0))
    Ry = trimesh.transformations.rotation_matrix(ry, (0,1,0))
    Rz = trimesh.transformations.rotation_matrix(rz, (0,0,1))
    mesh.apply_transform(Rx @ Ry @ Rz)
    # translation
    T = trimesh.transformations.translation_matrix(tf.translate)
    mesh.apply_transform(T)
    return mesh

def _merge_meshes(meshes: List["trimesh.Trimesh"]) -> "trimesh.Trimesh":
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)

def _svg_to_polygons(svg_path: Path) -> List[Polygon]:
    if not _HAS_SVG:
        raise RuntimeError("svgpathtools not installed")
    # load paths, approximate with line segments, create polygons
    paths, attrs = svg2paths(str(svg_path))
    polys: List[Polygon] = []
    for path in paths:
        # sample each path into points
        length = path.length()
        n = max(32, int(length / 4.0))
        pts = [path.point(i / n) for i in range(n+1)]
        coords = [(p.real, p.imag) for p in pts]
        poly = Polygon(coords).buffer(0)
        if poly and not poly.is_empty:
            if isinstance(poly, (Polygon,)):
                polys.append(poly)
            elif isinstance(poly, MultiPolygon):
                polys.extend(list(poly.geoms))
    if not polys:
        raise RuntimeError("No polygons could be formed from SVG")
    return polys

def _polygon_to_trimesh(poly: Polygon, height: float) -> "trimesh.Trimesh":
    # triangulate polygon exterior/interiors via trimesh.path
    # shapely -> dict of paths
    from trimesh.path import Path2D
    from shapely.geometry import LinearRing
    def ring_to_np(ring: LinearRing):
        xs, ys = ring.xy
        return np.vstack([xs, ys]).T

    entities = []
    vertices = []

    ext = ring_to_np(poly.exterior)
    # Build Path2D from polylines
    # We'll rely on Path2D.polygons_full to triangulate
    p2 = Path2D(entities=[], vertices=ext)
    try:
        # Use trimesh.creation.extrude_polygon if available (requires shapely)
        mesh = trimesh.creation.extrude_polygon(poly, height)
    except Exception:
        # Fallback: simple prism box bounds (not ideal)
        minx, miny, maxx, maxy = poly.bounds
        w = maxx - minx
        h = maxy - miny
        mesh = _box(extents=(w, h, height))
        mesh.apply_translation(((minx+maxx)/2.0, (miny+maxy)/2.0, height/2.0))
    return mesh

def _heightmap_to_mesh(img: Path, z_scale: float = 1.0) -> "trimesh.Trimesh":
    if not _HAS_PIL:
        raise RuntimeError("Pillow not installed")
    im = Image.open(str(img)).convert("L")
    arr = np.array(im, dtype=np.float32) / 255.0
    h, w = arr.shape
    # grid
    xs, ys = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    zs = arr * float(z_scale)
    # vertices
    vertices = np.column_stack([xs.flatten(), ys.flatten(), zs.flatten()])
    # faces (two triangles per quad)
    faces = []
    def vid(x, y): return y * w + x
    for y in range(h - 1):
        for x in range(w - 1):
            v0 = vid(x, y)
            v1 = vid(x + 1, y)
            v2 = vid(x, y + 1)
            v3 = vid(x + 1, y + 1)
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    faces = np.array(faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh

# ---------- tool ----------
class Model3DGeneratorTool(Tool):
    """
    Generate 3D models procedurally and export to OBJ/STL/GLB/PLY.
    Primitives: cube/box, sphere (icosphere), cylinder, plane
    Builders: heightmap image -> mesh, extrude SVG/profile, merge multiple parts
    """

    def __init__(self):
        super().__init__("model3d_generator", "Generate 3D models (primitives/heightmap/extrude) and export to OBJ/STL/GLB/PLY.")

    def is_available(self) -> bool:
        return _HAS_TRIMESH and _HAS_NUMPY

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["primitive", "heightmap", "extrude_svg", "merge"],
                        "description": "What to generate"
                    },
                    "primitive": {
                        "type": "string",
                        "enum": ["box", "sphere", "cylinder", "plane"],
                        "description": "Primitive to create (when mode=primitive)"
                    },
                    "size": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "Dimensions for box/plane as [x,y,z]"
                    },
                    "radius": {"type": "number", "description": "Radius for sphere/cylinder"},
                    "height": {"type": "number", "description": "Height for cylinder/extrude"},
                    "subdivisions": {"type": "integer", "description": "Sphere subdivisions (2-6 recommended)", "default": 3},
                    "segments": {"type": "integer", "description": "Cylinder segments around axis", "default": 32},
                    "heightmap_path": {"type": "string", "description": "Image path for heightmap (when mode=heightmap)"},
                    "z_scale": {"type": "number", "description": "Scale factor for heightmap elevation", "default": 1.0},
                    "svg_path": {"type": "string", "description": "SVG path to extrude (when mode=extrude_svg)"},
                    "output_path": {"type": "string", "description": "Where to save the mesh (.obj/.stl/.glb/.ply)"},
                    "transform": {
                        "type": "object",
                        "properties": {
                            "translate": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "default": [0,0,0]},
                            "rotate_deg": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "default": [0,0,0]},
                            "scale": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "default": [1,1,1]}
                        }
                    },
                    "parts": {
                        "type": "array",
                        "description": "For mode=merge: array of meshes to build and combine",
                        "items": {
                            "type": "object",
                            "properties": {
                                "mode": {"type": "string", "enum": ["primitive", "heightmap", "extrude_svg"]},
                                "primitive": {"type": "string", "enum": ["box", "sphere", "cylinder", "plane"]},
                                "size": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                                "radius": {"type": "number"},
                                "height": {"type": "number"},
                                "subdivisions": {"type": "integer"},
                                "segments": {"type": "integer"},
                                "heightmap_path": {"type": "string"},
                                "z_scale": {"type": "number"},
                                "svg_path": {"type": "string"},
                                "transform": {
                                    "type": "object",
                                    "properties": {
                                        "translate": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                                        "rotate_deg": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                                        "scale": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
                                    }
                                }
                            },
                            "required": ["mode"]
                        }
                    }
                },
                "required": ["mode", "output_path"]
            }
        }

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        if not self.is_available():
            return {"success": False, "error": "trimesh/numpy not installed"}

        mode = kwargs.get("mode")
        output_path = kwargs.get("output_path")
        try:
            out = _safe_path(output_path)
        except Exception as e:
            return {"success": False, "error": f"output_path invalid: {e}"}

        try:
            mesh = None
            if mode == "primitive":
                mesh = self._build_primitive(kwargs)
            elif mode == "heightmap":
                mesh = self._build_heightmap(kwargs)
            elif mode == "extrude_svg":
                mesh = self._build_extrude_svg(kwargs)
            elif mode == "merge":
                mesh = self._build_merge(kwargs)
            else:
                return {"success": False, "error": f"Unknown mode: {mode}"}

            if mesh is None:
                return {"success": False, "error": "Mesh build returned None"}

            # export
            fmt = out.suffix.lower().lstrip(".")
            if fmt not in {"obj", "stl", "ply", "glb"}:
                return {"success": False, "error": f"Unsupported export format: .{fmt}"}

            mesh.export(str(out))
            return {
                "success": True,
                "mode": mode,
                "output": str(out.relative_to(WORKSPACE_ROOT)),
                "vertices": int(mesh.vertices.shape[0]),
                "faces": int(mesh.faces.shape[0])
            }
        except Exception as e:
            logger.exception("Model3DGeneratorTool error")
            return {"success": False, "error": str(e)}

    # -------- builders --------
    def _build_primitive(self, kw: Dict[str, Any]):
        primitive = kw.get("primitive")
        tf = self._parse_tf(kw.get("transform"))
        if primitive == "box":
            size = kw.get("size") or [1.0, 1.0, 1.0]
            mesh = _box(extents=size)
        elif primitive == "sphere":
            radius = float(kw.get("radius") or 0.5)
            subs = int(kw.get("subdivisions") or 3)
            mesh = _icosphere(subdivisions=max(1, subs), radius=radius)
        elif primitive == "cylinder":
            radius = float(kw.get("radius") or 0.5)
            height = float(kw.get("height") or 1.0)
            segs = int(kw.get("segments") or 32)
            mesh = _cylinder(radius=radius, height=height, sections=max(3, segs))
        elif primitive == "plane":
            size = kw.get("size") or [1.0, 1.0, 0.0]
            mesh = _box(extents=[size[0], size[1], 0.001])
        else:
            raise ValueError(f"Unknown primitive: {primitive}")
        return _apply_transform(mesh, tf)

    def _build_heightmap(self, kw: Dict[str, Any]):
        if not (_HAS_PIL and _HAS_NUMPY):
            raise RuntimeError("Pillow/numpy not installed for heightmap")
        path = kw.get("heightmap_path")
        if not path:
            raise ValueError("heightmap_path required")
        z_scale = float(kw.get("z_scale") or 1.0)
        mesh = _heightmap_to_mesh(_safe_path(path), z_scale=z_scale)
        tf = self._parse_tf(kw.get("transform"))
        return _apply_transform(mesh, tf)

    def _build_extrude_svg(self, kw: Dict[str, Any]):
        if not _HAS_SHAPELY:
            raise RuntimeError("shapely not installed for extrusion")
        path = kw.get("svg_path")
        height = float(kw.get("height") or 1.0)
        if not path:
            raise ValueError("svg_path required")
        svg = _safe_path(path)
        polys = _svg_to_polygons(svg)
        # union overlapping
        shape = unary_union(polys)
        meshes = []
        if isinstance(shape, Polygon):
            meshes.append(_polygon_to_trimesh(shape, height))
        else:
            for g in shape.geoms:
                meshes.append(_polygon_to_trimesh(g, height))
        mesh = _merge_meshes(meshes)
        tf = self._parse_tf(kw.get("transform"))
        return _apply_transform(mesh, tf)

    def _build_merge(self, kw: Dict[str, Any]):
        parts = kw.get("parts") or []
        if not isinstance(parts, list) or not parts:
            raise ValueError("parts array required for merge")
        built = []
        for p in parts:
            mode = p.get("mode")
            if mode == "primitive":
                built.append(self._build_primitive(p))
            elif mode == "heightmap":
                built.append(self._build_heightmap(p))
            elif mode == "extrude_svg":
                built.append(self._build_extrude_svg(p))
            else:
                raise ValueError(f"Unsupported part mode: {mode}")
        return _merge_meshes(built)

    # -------- utils --------
    def _parse_tf(self, data: Optional[Dict[str, Any]]) -> Transform:
        if not data:
            return Transform()
        t = data.get("translate") or (0,0,0)
        r = data.get("rotate_deg") or (0,0,0)
        s = data.get("scale") or (1,1,1)
        return Transform(tuple(map(float, t)), tuple(map(float, r)), tuple(map(float, s)))
