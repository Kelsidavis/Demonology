# model3d_generator.py
from __future__ import annotations
import math, os, random, re, time, logging
from pathlib import Path
from typing import Any, Dict, List, Optional
try:
    import numpy as np
    import trimesh
except Exception:
    np = None; trimesh = None
try:
    from .base import Tool
except Exception:
    class Tool:
        def __init__(self, name: str, description: str):
            self.name, self.description = name, description
logger = logging.getLogger(__name__)

def _slug(s: str, default: str = "model") -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (s or "").strip()).strip("_").lower()
    return s or default
def _workspace_root() -> Path:
    return Path(os.environ.get("DEMONOLOGY_ROOT") or os.getcwd()).resolve()
def _is_within(root: Path, p: Path) -> bool:
    try:
        return str(p.resolve()).startswith(str(root.resolve()) + os.sep)
    except Exception:
        return False
def _choose_output_path(prompt: Optional[str], ext: Optional[str]) -> Path:
    stem = _slug(prompt or "model"); ext = (("." + ext) if ext and not str(ext).startswith(".") else (ext or ".obj"))
    return (_workspace_root() / "models" / f"{stem}_{int(time.time())}{ext}")
def _infer_format_from_ext(p: Path) -> str:
    return {".obj":"obj",".stl":"stl",".ply":"ply",".glb":"glb",".gltf":"gltf"}.get(p.suffix.lower(), "obj")
def _deg2rad(v: List[float]) -> List[float]: return [math.radians(x) for x in v]
def _seed_from_prompt(prompt: str) -> int: return abs(hash(prompt)) % (2**32)

def _extract_numbers(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in ("length","width","height","radius","scale","inner_radius","segments","twist"):
        m = re.search(rf"{key}\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
        if m: out[key] = float(m.group(1))
    m = re.search(r"\b(size|scale)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
    if m and "scale" not in out: out["scale"] = float(m.group(2))
    return out

def _extract_colors(text: str) -> Dict[str, List[float]]:
    """Extract color specifications from text"""
    colors: Dict[str, List[float]] = {}
    # Look for RGB values like "color: 255,128,64" or "red: 1.0,0.5,0.2"
    color_patterns = [("color", r"color\s*[:=]?\s*([0-9.,\s]+)"), 
                     ("primary", r"primary\s*[:=]?\s*([0-9.,\s]+)"),
                     ("secondary", r"secondary\s*[:=]?\s*([0-9.,\s]+)")]
    for name, pattern in color_patterns:
        m = re.search(pattern, text, re.I)
        if m:
            try:
                vals = [float(x.strip()) for x in m.group(1).split(',') if x.strip()]
                if len(vals) >= 3:
                    # Normalize if values are 0-255 range
                    if any(v > 1.0 for v in vals[:3]):
                        vals = [v/255.0 for v in vals]
                    colors[name] = vals[:4]  # Support RGBA
            except ValueError:
                pass
    # Look for named colors
    named_colors = {"red":[1,0,0], "green":[0,1,0], "blue":[0,0,1], "white":[1,1,1], 
                   "black":[0,0,0], "yellow":[1,1,0], "purple":[1,0,1], "cyan":[0,1,1],
                   "gray":[0.5,0.5,0.5], "orange":[1,0.5,0], "pink":[1,0.7,0.7]}
    for color_name, rgb in named_colors.items():
        if color_name in text.lower():
            colors["color"] = rgb
            break
    return colors
def _keyword(text: str, *keys: str) -> bool:
    t = text.lower(); return any(k in t for k in keys)

def _design_spaceship(rng: random.Random, knobs: Dict[str, float]) -> List[Dict[str, Any]]:
    L = knobs.get("length", 10.0); W = knobs.get("width", 4.0); H = knobs.get("height", 2.5)
    if _keyword(knobs.get("text",""), "sleek","slim","fighter"): W*=0.7; H*=0.8
    if _keyword(knobs.get("text",""), "cargo","freighter","bulky"): W*=1.3; H*=1.2
    parts = []
    parts.append({"mode":"primitive","primitive":"box","size":[L*0.6,H,W*0.9],
                  "transform":{"translate":[0,H*0.5,0],"rotate_deg":[0,rng.uniform(-10,15),0]}})
    parts.append({"mode":"primitive","primitive":"cylinder","radius":max(W*0.12,0.3),"height":L*0.25,"segments":24,
                  "transform":{"translate":[0,H*0.25,-L*0.35]}})
    parts.append({"mode":"primitive","primitive":"sphere","radius":max(W*0.15,0.4),"subdivisions":3,
                  "transform":{"translate":[0,H*0.3,-L*0.5]}})
    wing_span = W*1.8; wing_thk=max(0.15,H*0.12); wing_len=L*0.35
    parts += [
        {"mode":"primitive","primitive":"box","size":[wing_len,wing_thk,wing_span],
         "transform":{"translate":[-L*0.10,H*0.1,0.0],"rotate_deg":[0,0,8]}},
        {"mode":"primitive","primitive":"box","size":[wing_len,wing_thk,wing_span],
         "transform":{"translate":[ L*0.10,H*0.1,0.0],"rotate_deg":[0,0,-8]}}
    ]
    parts.append({"mode":"primitive","primitive":"box","size":[W*0.15,H*1.2,W*0.4],
                  "transform":{"translate":[0,H*1.0,L*0.25]}})
    eng_r=max(W*0.12,0.25); eng_h=L*0.3
    parts += [
        {"mode":"primitive","primitive":"cylinder","radius":eng_r,"height":eng_h,"segments":24,
         "transform":{"translate":[-W*0.45,H*0.0,L*0.35]}},
        {"mode":"primitive","primitive":"cylinder","radius":eng_r,"height":eng_h,"segments":24,
         "transform":{"translate":[ W*0.45,H*0.0,L*0.35]}}
    ]
    parts.append({"mode":"primitive","primitive":"sphere","radius":max(W*0.18,0.35),"subdivisions":3,
                  "transform":{"translate":[0,H*0.2,-L*0.05]}})
    return parts
def _design_rocket(rng, k):
    H=k.get("height",12.0); R=k.get("radius",1.2); parts=[]
    parts.append({"mode":"primitive","primitive":"cylinder","radius":R,"height":H*0.7,"segments":24,
                  "transform":{"translate":[0,H*0.35,0]}})
    parts.append({"mode":"primitive","primitive":"cone","radius":R*0.9,"height":H*0.25,"segments":24,
                  "transform":{"translate":[0,H*0.85,0]}})
    fin_h=H*0.18; fin_w=R*0.15; fin_l=R*1.2
    for ang,dx,dz in ((0,R*1.05,0),(90,0,R*1.05),(180,-R*1.05,0),(270,0,-R*1.05)):
        parts.append({"mode":"primitive","primitive":"box","size":[fin_l,fin_h,fin_w],
                      "transform":{"translate":[dx,fin_h*0.5,dz],"rotate_deg":[0,ang,0]}})
    return parts
def _design_house(rng,k):
    W=k.get("width",8.0); D=k.get("length",10.0); H=k.get("height",4.5)
    return [
        {"mode":"primitive","primitive":"box","size":[D,H,W],"transform":{"translate":[0,H*0.5,0]}},
        {"mode":"primitive","primitive":"prism","size":[D*1.05,H*0.8,W*1.05],"transform":{"translate":[0,H*1.1,0]}}
    ]
def _design_car(rng,k):
    L=k.get("length",4.2); W=k.get("width",1.8); H=k.get("height",1.5)
    parts=[{"mode":"primitive","primitive":"box","size":[L,H*0.55,W],"transform":{"translate":[0,H*0.275,0]}},
           {"mode":"primitive","primitive":"box","size":[L*0.6,H*0.5,W*0.95],"transform":{"translate":[0,H*0.75,0]}}]
    wheel_r=min(W,H)*0.25
    for x in (-L*0.35,L*0.35):
        for z in (-W*0.45,W*0.45):
            parts.append({"mode":"primitive","primitive":"cylinder","radius":wheel_r,"height":0.3,"segments":24,
                          "transform":{"translate":[x,wheel_r,z],"rotate_deg":[0,0,90]}})
    return parts
def _design_building(rng: random.Random, knobs: Dict[str, float]) -> List[Dict[str, Any]]:
    W = knobs.get("width", 12.0); D = knobs.get("length", 15.0); H = knobs.get("height", 20.0)
    parts = []
    
    # Main building structure
    parts.append({"mode":"primitive","primitive":"box","size":[D,H,W],
                  "transform":{"translate":[0,H*0.5,0]}})
    
    # Add floors/windows pattern
    floor_height = H / max(1, int(H / 3.5))  # About 3.5m per floor
    num_floors = int(H / floor_height)
    
    # Entrance
    door_w, door_h = min(W*0.2, 2.5), min(H*0.15, 3.0)
    parts.append({"mode":"primitive","primitive":"box","size":[D*0.1,door_h,door_w],
                  "transform":{"translate":[D*0.45,door_h*0.5,-W*0.01]}})
    
    # Roof
    if _keyword(knobs.get("text",""), "peaked", "gabled", "roof"):
        parts.append({"mode":"primitive","primitive":"wedge","size":[D*1.1,H*0.3,W*1.1],
                      "transform":{"translate":[0,H+H*0.15,0]}})
    else:
        parts.append({"mode":"primitive","primitive":"box","size":[D*1.05,H*0.1,W*1.05],
                      "transform":{"translate":[0,H+H*0.05,0]}})
    
    return parts

def _design_tree(rng: random.Random, knobs: Dict[str, float]) -> List[Dict[str, Any]]:
    H = knobs.get("height", 8.0); trunk_r = knobs.get("radius", 0.3)
    crown_r = max(H*0.4, trunk_r*3)
    parts = []
    
    # Trunk
    trunk_h = H * rng.uniform(0.3, 0.5)
    parts.append({"mode":"primitive","primitive":"cylinder","radius":trunk_r,"height":trunk_h,"segments":12,
                  "transform":{"translate":[0,trunk_h*0.5,0]}})
    
    # Crown - different styles based on keywords
    crown_y = trunk_h + crown_r * 0.7
    if _keyword(knobs.get("text",""), "pine", "spruce", "evergreen"):
        # Conical tree
        for i in range(3):
            cone_r = crown_r * (1.2 - i*0.3)
            cone_h = crown_r * 0.8
            parts.append({"mode":"primitive","primitive":"cone","radius":cone_r,"height":cone_h,"segments":16,
                          "transform":{"translate":[0,crown_y + i*cone_h*0.4,0]}})
    elif _keyword(knobs.get("text",""), "palm"):
        # Palm tree - thin trunk, spherical top
        parts.append({"mode":"primitive","primitive":"sphere","radius":crown_r*0.6,"subdivisions":2,
                      "transform":{"translate":[0,crown_y,0]}})
    else:
        # Regular leafy tree
        parts.append({"mode":"primitive","primitive":"sphere","radius":crown_r,"subdivisions":3,
                      "transform":{"translate":[0,crown_y,0]}})
        # Additional crown layers
        for i in range(2):
            offset_x, offset_z = rng.uniform(-crown_r*0.3, crown_r*0.3), rng.uniform(-crown_r*0.3, crown_r*0.3)
            parts.append({"mode":"primitive","primitive":"sphere","radius":crown_r*rng.uniform(0.6,0.8),"subdivisions":2,
                          "transform":{"translate":[offset_x,crown_y+rng.uniform(-crown_r*0.2,crown_r*0.2),offset_z]}})
    
    return parts

def _design_furniture(rng: random.Random, knobs: Dict[str, float]) -> List[Dict[str, Any]]:
    scale = knobs.get("scale", 1.0)
    text = knobs.get("text", "")
    parts = []
    
    if _keyword(text, "chair", "seat"):
        # Chair
        seat_w, seat_d, seat_h = 1.8*scale, 1.8*scale, 1.8*scale
        back_h = seat_h * 1.8
        leg_r = 0.15*scale
        
        # Seat
        parts.append({"mode":"primitive","primitive":"box","size":[seat_w,0.2*scale,seat_d],
                      "transform":{"translate":[0,seat_h,0]}})
        
        # Backrest
        parts.append({"mode":"primitive","primitive":"box","size":[seat_w,back_h-seat_h,0.2*scale],
                      "transform":{"translate":[0,(seat_h+back_h)*0.5,-seat_d*0.4]}})
        
        # Legs
        for x in [-seat_w*0.4, seat_w*0.4]:
            for z in [-seat_d*0.4, seat_d*0.4]:
                parts.append({"mode":"primitive","primitive":"cylinder","radius":leg_r,"height":seat_h,"segments":12,
                              "transform":{"translate":[x,seat_h*0.5,z]}})
    
    elif _keyword(text, "table", "desk"):
        # Table
        table_w, table_d, table_h = 4.0*scale, 2.5*scale, 3.0*scale
        top_thick = 0.2*scale
        leg_r = 0.2*scale
        
        # Table top
        parts.append({"mode":"primitive","primitive":"box","size":[table_w,top_thick,table_d],
                      "transform":{"translate":[0,table_h,0]}})
        
        # Legs
        for x in [-table_w*0.45, table_w*0.45]:
            for z in [-table_d*0.45, table_d*0.45]:
                parts.append({"mode":"primitive","primitive":"cylinder","radius":leg_r,"height":table_h-top_thick*0.5,"segments":12,
                              "transform":{"translate":[x,(table_h-top_thick*0.5)*0.5,z]}})
    
    elif _keyword(text, "bookshelf", "shelf"):
        # Bookshelf
        shelf_w, shelf_d, shelf_h = 3.0*scale, 1.2*scale, 6.0*scale
        shelf_thick = 0.15*scale
        num_shelves = 4
        
        # Sides
        for x in [-shelf_w*0.5, shelf_w*0.5]:
            parts.append({"mode":"primitive","primitive":"box","size":[shelf_thick,shelf_h,shelf_d],
                          "transform":{"translate":[x,shelf_h*0.5,0]}})
        
        # Shelves
        for i in range(num_shelves + 1):
            y = i * shelf_h / num_shelves
            parts.append({"mode":"primitive","primitive":"box","size":[shelf_w,shelf_thick,shelf_d],
                          "transform":{"translate":[0,y,0]}})
    
    else:
        # Generic furniture piece
        parts.append({"mode":"primitive","primitive":"box","size":[2*scale,2*scale,1*scale],
                      "transform":{"translate":[0,1*scale,0]}})
    
    return parts

def _design_weapon(rng: random.Random, knobs: Dict[str, float]) -> List[Dict[str, Any]]:
    length = knobs.get("length", 4.0)
    text = knobs.get("text", "")
    parts = []
    
    if _keyword(text, "sword", "blade"):
        # Sword
        blade_w, blade_h, handle_l = 0.2, length*0.7, length*0.25
        grip_r, guard_w = 0.15, 0.8
        
        # Blade
        parts.append({"mode":"primitive","primitive":"wedge","size":[blade_h,blade_w*2,blade_w],
                      "transform":{"translate":[0,0,blade_h*0.5],"rotate_deg":[90,0,0]}})
        
        # Handle
        parts.append({"mode":"primitive","primitive":"cylinder","radius":grip_r,"height":handle_l,"segments":12,
                      "transform":{"translate":[0,0,-handle_l*0.5]}})
        
        # Crossguard
        parts.append({"mode":"primitive","primitive":"box","size":[blade_w,guard_w,blade_w*0.5],
                      "transform":{"translate":[0,0,0]}})
    
    elif _keyword(text, "spear", "lance"):
        # Spear
        shaft_l, shaft_r = length*0.85, 0.08
        point_l = length*0.15
        
        # Shaft
        parts.append({"mode":"primitive","primitive":"cylinder","radius":shaft_r,"height":shaft_l,"segments":12,
                      "transform":{"translate":[0,0,shaft_l*0.5]}})
        
        # Spearhead
        parts.append({"mode":"primitive","primitive":"cone","radius":shaft_r*2,"height":point_l,"segments":12,
                      "transform":{"translate":[0,0,shaft_l+point_l*0.5]}})
    
    elif _keyword(text, "axe"):
        # Axe
        handle_l, handle_r = length*0.8, 0.1
        head_w, head_h = 0.8, 0.5
        
        # Handle
        parts.append({"mode":"primitive","primitive":"cylinder","radius":handle_r,"height":handle_l,"segments":12,
                      "transform":{"translate":[0,0,handle_l*0.5]}})
        
        # Axe head
        parts.append({"mode":"primitive","primitive":"wedge","size":[head_w,head_h,0.3],
                      "transform":{"translate":[head_w*0.25,0,handle_l*0.85],"rotate_deg":[0,90,0]}})
    
    return parts

def _design_generic(rng,k):
    base=k.get("scale",1.0); parts=[]
    for _ in range(3 + rng.randint(0,3)):
        prim=rng.choice(["box","cylinder","sphere","torus","wedge"])
        if prim=="box":
            size=[rng.uniform(1,3)*base, rng.uniform(0.5,2)*base, rng.uniform(1,3)*base]
            parts.append({"mode":"primitive","primitive":prim,"size":size,
                          "transform":{"translate":[rng.uniform(-2,2),rng.uniform(0,2),rng.uniform(-2,2)],
                                       "rotate_deg":[rng.uniform(0,15),rng.uniform(0,180),rng.uniform(0,15)]}})
        elif prim=="cylinder":
            parts.append({"mode":"primitive","primitive":prim,"radius":rng.uniform(0.3,1)*base,"height":rng.uniform(1,3)*base,"segments":24,
                          "transform":{"translate":[rng.uniform(-2,2),rng.uniform(0,2),rng.uniform(-2,2)]}})
        elif prim=="torus":
            parts.append({"mode":"primitive","primitive":prim,"radius":rng.uniform(0.5,1.2)*base,"inner_radius":rng.uniform(0.1,0.4)*base,"segments":24,
                          "transform":{"translate":[rng.uniform(-2,2),rng.uniform(0,2),rng.uniform(-2,2)]}})
        elif prim=="wedge":
            size=[rng.uniform(1,3)*base, rng.uniform(0.5,2)*base, rng.uniform(1,3)*base]
            parts.append({"mode":"primitive","primitive":prim,"size":size,
                          "transform":{"translate":[rng.uniform(-2,2),rng.uniform(0,2),rng.uniform(-2,2)]}})
        else:
            parts.append({"mode":"primitive","primitive":prim,"radius":rng.uniform(0.5,1.2)*base,"subdivisions":3,
                          "transform":{"translate":[rng.uniform(-2,2),rng.uniform(0,2),rng.uniform(-2,2)]}})
    return parts
def propose_design_from_prompt(prompt: str) -> List[Dict[str, Any]]:
    text=(prompt or "").strip(); rng=random.Random(abs(hash(text))%(2**32)); knobs=_extract_numbers(text); knobs["text"]=text
    if _keyword(text,"spaceship","space ship","starfighter","craft"): return _design_spaceship(rng,knobs)
    if _keyword(text,"rocket","missile"): return _design_rocket(rng,knobs)
    if _keyword(text,"house","hut","cabin"): return _design_house(rng,knobs)
    if _keyword(text,"car","vehicle","auto"): return _design_car(rng,knobs)
    if _keyword(text,"building","skyscraper","tower","office"): return _design_building(rng,knobs)
    if _keyword(text,"tree","oak","pine","palm","birch","maple"): return _design_tree(rng,knobs)
    if _keyword(text,"chair","table","desk","furniture","bookshelf","shelf"): return _design_furniture(rng,knobs)
    if _keyword(text,"sword","blade","weapon","spear","axe","knife"): return _design_weapon(rng,knobs)
    return _design_generic(rng,knobs)

def _primitive_mesh(spec: Dict[str, Any]):
    prim=spec.get("primitive")
    if prim=="box":
        sx,sy,sz=spec.get("size",[1,1,1]); return trimesh.creation.box(extents=[sx,sy,sz])
    if prim=="cylinder":
        r=float(spec.get("radius",0.5)); h=float(spec.get("height",1.0)); seg=int(spec.get("segments",24))
        return trimesh.creation.cylinder(radius=r, height=h, sections=seg)
    if prim=="sphere":
        r=float(spec.get("radius",0.75)); sub=int(spec.get("subdivisions",3)); return trimesh.creation.icosphere(subdivisions=sub, radius=r)
    if prim=="cone":
        r=float(spec.get("radius",0.5)); h=float(spec.get("height",1.0)); seg=int(spec.get("segments",24))
        return trimesh.creation.cone(radius=r, height=h, sections=seg)
    if prim=="prism":
        sx,sy,sz=spec.get("size",[2,1,2]); return trimesh.creation.box(extents=[sx,sy,sz])
    # New primitives
    if prim=="torus":
        major_r=float(spec.get("radius",1.0)); minor_r=float(spec.get("inner_radius",0.3))
        seg_major=int(spec.get("segments",24)); seg_minor=int(spec.get("minor_segments",12))
        return _create_torus(major_r, minor_r, seg_major, seg_minor)
    if prim=="wedge":
        sx,sy,sz=spec.get("size",[2,1,2]); return _create_wedge(sx,sy,sz)
    if prim=="tube":
        r=float(spec.get("radius",0.5)); inner_r=float(spec.get("inner_radius",0.3))
        h=float(spec.get("height",1.0)); seg=int(spec.get("segments",24))
        return _create_tube(r, inner_r, h, seg)
    if prim=="helix":
        r=float(spec.get("radius",0.5)); h=float(spec.get("height",2.0))
        turns=float(spec.get("turns",3.0)); seg=int(spec.get("segments",64))
        return _create_helix(r, h, turns, seg)
    return None

def _create_torus(major_radius: float, minor_radius: float, major_segments: int, minor_segments: int):
    """Create a torus mesh"""
    vertices, faces = [], []
    for i in range(major_segments):
        theta = 2 * math.pi * i / major_segments
        for j in range(minor_segments):
            phi = 2 * math.pi * j / minor_segments
            x = (major_radius + minor_radius * math.cos(phi)) * math.cos(theta)
            y = minor_radius * math.sin(phi)
            z = (major_radius + minor_radius * math.cos(phi)) * math.sin(theta)
            vertices.append([x, y, z])
    
    for i in range(major_segments):
        for j in range(minor_segments):
            v1 = i * minor_segments + j
            v2 = i * minor_segments + (j + 1) % minor_segments
            v3 = ((i + 1) % major_segments) * minor_segments + j
            v4 = ((i + 1) % major_segments) * minor_segments + (j + 1) % minor_segments
            faces.extend([[v1,v2,v4], [v1,v4,v3]])
    
    return trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))

def _create_wedge(width: float, height: float, depth: float):
    """Create a triangular wedge/prism"""
    vertices = np.array([
        [0, 0, 0], [width, 0, 0], [width/2, height, 0],  # Front face
        [0, 0, depth], [width, 0, depth], [width/2, height, depth]  # Back face
    ])
    faces = np.array([
        [0,1,2], [3,5,4],  # Front and back triangular faces
        [0,3,4], [0,4,1],  # Bottom face
        [1,4,5], [1,5,2],  # Right face
        [2,5,3], [2,3,0]   # Left face
    ])
    return trimesh.Trimesh(vertices=vertices, faces=faces)

def _create_tube(outer_radius: float, inner_radius: float, height: float, segments: int):
    """Create a hollow tube/pipe"""
    outer_cyl = trimesh.creation.cylinder(radius=outer_radius, height=height, sections=segments)
    inner_cyl = trimesh.creation.cylinder(radius=inner_radius, height=height, sections=segments)
    try:
        return outer_cyl.difference(inner_cyl)
    except Exception:
        return outer_cyl  # Fallback to solid cylinder

def _create_helix(radius: float, height: float, turns: float, segments: int):
    """Create a helix/spring shape"""
    vertices, faces = [], []
    tube_radius = radius * 0.1  # Thickness of the helix
    
    for i in range(segments + 1):
        t = i / segments
        angle = 2 * math.pi * turns * t
        y = height * t - height/2
        
        # Center point of tube cross-section
        center_x = radius * math.cos(angle)
        center_z = radius * math.sin(angle)
        
        # Create circular cross-section
        for j in range(8):  # 8 points around tube circumference
            cross_angle = 2 * math.pi * j / 8
            # Local coordinate system for tube
            normal_x = -math.sin(angle)
            normal_z = math.cos(angle)
            binormal_x = math.cos(angle)
            binormal_z = math.sin(angle)
            
            offset_x = tube_radius * (math.cos(cross_angle) * normal_x)
            offset_y = tube_radius * math.sin(cross_angle)
            offset_z = tube_radius * (math.cos(cross_angle) * normal_z)
            
            vertices.append([center_x + offset_x, y + offset_y, center_z + offset_z])
    
    # Generate faces
    cross_sections = segments + 1
    points_per_section = 8
    for i in range(segments):
        for j in range(points_per_section):
            v1 = i * points_per_section + j
            v2 = i * points_per_section + (j + 1) % points_per_section
            v3 = (i + 1) * points_per_section + j
            v4 = (i + 1) * points_per_section + (j + 1) % points_per_section
            faces.extend([[v1,v2,v4], [v1,v4,v3]])
    
    return trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
def _apply_material(mesh, material_spec: Dict[str, Any]):
    """Apply material properties to mesh"""
    if not material_spec:
        return mesh
    
    # Apply vertex colors if specified
    color = material_spec.get("color")
    if color and len(color) >= 3:
        # Ensure color is in [0,1] range
        rgb = [max(0, min(1, float(c))) for c in color[:3]]
        alpha = color[3] if len(color) > 3 else 1.0
        
        # Set vertex colors (RGBA format)
        vertex_colors = np.full((len(mesh.vertices), 4), [rgb[0], rgb[1], rgb[2], alpha], dtype=np.float64)
        mesh.visual.vertex_colors = vertex_colors
    
    # Apply texture coordinates if needed
    if material_spec.get("generate_uvs", False):
        try:
            # Simple planar UV mapping
            vertices = mesh.vertices
            bounds = mesh.bounds
            size = bounds[1] - bounds[0]
            
            # Project onto XY plane for UV coordinates
            if size[0] > 0 and size[1] > 0:
                uv = np.zeros((len(vertices), 2))
                uv[:, 0] = (vertices[:, 0] - bounds[0][0]) / size[0]
                uv[:, 1] = (vertices[:, 1] - bounds[0][1]) / size[1]
                mesh.visual.uv = uv
        except Exception:
            pass  # UV generation failed, continue without UVs
    
    return mesh

def _apply_transform(mesh, transform: Dict[str, Any]):
    T = np.eye(4); rot=transform.get("rotate_deg"); trans=transform.get("translate"); scale=transform.get("scale")
    if rot:
        rx,ry,rz=_deg2rad([float(x) for x in rot])
        Rx=trimesh.transformations.rotation_matrix(rx,[1,0,0]); Ry=trimesh.transformations.rotation_matrix(ry,[0,1,0]); Rz=trimesh.transformations.rotation_matrix(rz,[0,0,1])
        T=trimesh.transformations.concatenate_matrices(T,Rx,Ry,Rz)
    if trans:
        tx,ty,tz=[float(x) for x in trans]
        T=trimesh.transformations.concatenate_matrices(T, trimesh.transformations.translation_matrix([tx,ty,tz]))
    if scale:
        s=float(scale); S=np.eye(4); S[:3,:3]*=s; T=trimesh.transformations.concatenate_matrices(T,S)
    
    mesh=mesh.copy(); mesh.apply_transform(T)
    
    # Apply material if specified
    material = transform.get("material")
    if material:
        mesh = _apply_material(mesh, material)
    
    return mesh
def _apply_mesh_operation(base_mesh, operation: Dict[str, Any]):
    """Apply mesh operation like boolean ops, extrusion, etc."""
    op_type = operation.get("type")
    
    if op_type == "extrude" and hasattr(trimesh, "path"):
        # Simple extrusion along Z-axis
        height = float(operation.get("height", 1.0))
        try:
            # Get 2D outline and extrude
            outline = base_mesh.outline()
            if outline is not None:
                return outline.extrude(height)
        except Exception:
            pass
        return base_mesh
    
    elif op_type in ["union", "difference", "intersection"] and "target" in operation:
        target_spec = operation["target"]
        if isinstance(target_spec, dict) and target_spec.get("mode") == "primitive":
            target_mesh = _primitive_mesh(target_spec)
            if target_mesh is not None:
                target_mesh = _apply_transform(target_mesh, target_spec.get("transform", {}))
                try:
                    if op_type == "union":
                        return base_mesh.union(target_mesh)
                    elif op_type == "difference":
                        return base_mesh.difference(target_mesh)
                    elif op_type == "intersection":
                        return base_mesh.intersection(target_mesh)
                except Exception:
                    pass  # Boolean ops can fail, fallback to original
    
    elif op_type == "smooth" and hasattr(base_mesh, "smoothed"):
        try:
            return base_mesh.smoothed()
        except Exception:
            pass
    
    elif op_type == "subdivide" and hasattr(base_mesh, "subdivide"):
        try:
            base_mesh.subdivide()
            return base_mesh
        except Exception:
            pass
    
    return base_mesh

def _optimize_mesh(mesh, operations: List[str] = None):
    """Apply mesh optimization operations"""
    operations = operations or []
    
    if "remove_duplicates" in operations:
        try:
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
        except Exception:
            pass
    
    if "fix_normals" in operations:
        try:
            mesh.fix_normals()
        except Exception:
            pass
    
    if "fill_holes" in operations:
        try:
            mesh.fill_holes()
        except Exception:
            pass
    
    if "smooth" in operations and hasattr(mesh, "smoothed"):
        try:
            mesh = mesh.smoothed()
        except Exception:
            pass
    
    return mesh

def _build_mesh_from_parts(parts: List[Dict[str, Any]]):
    meshes=[]
    for p in parts:
        if p.get("mode")=="primitive":
            m=_primitive_mesh(p); 
            if m is None: continue
            m=_apply_transform(m, p.get("transform", {}))
            
            # Apply any operations
            operations = p.get("operations", [])
            for op in operations:
                m = _apply_mesh_operation(m, op)
            
            meshes.append(m)
        
        elif p.get("mode")=="operation":
            # Handle complex operations between existing meshes
            continue  # Skip for now, would need more complex state tracking
    
    if not meshes: raise RuntimeError("No valid primitives produced")
    
    # Combine all meshes
    combined = trimesh.util.concatenate(meshes)
    
    # Apply global optimizations
    optimizations = ["remove_duplicates", "fix_normals"]
    combined = _optimize_mesh(combined, optimizations)
    
    return combined

class Model3DGeneratorTool(Tool):
    def __init__(self):
        super().__init__("model3d_generator","Generate 3D models procedurally or from a short prompt.")
    def is_available(self)->bool:
        return (np is not None) and (trimesh is not None)
    def to_openai_function(self)->Dict[str, Any]:
        return {
            "name": self.name,
            "description": "Generate 3D models from prompts or part specifications. Supports primitives: box, cylinder, sphere, cone, torus, wedge, tube, helix. Templates: spaceship, rocket, house, car, building, tree, furniture, weapon. Colors and materials supported.",
            "parameters": {
                "type":"object",
                "properties":{
                    "mode":{"type":"string","enum":["merge","auto"],"default":"auto","description":"'auto' generates from prompt, 'merge' combines provided parts"},
                    "prompt":{"type":"string","description":"Description of model to generate (e.g. 'red spaceship', 'oak tree height:10', 'blue chair')"},
                    "parts":{"type":"array","items":{"type":"object"},"description":"Array of part specifications for merge mode"},
                    "output_path":{"type":"string","description":"Optional output file path (relative to workspace)"},
                    "format":{"type":"string","enum":["obj","stl","ply","glb","gltf"],"description":"Output format (default: obj)"},
                    "seed":{"type":"integer","description":"Random seed for reproducible generation"},
                    "optimize":{"type":"boolean","default":True,"description":"Apply mesh optimization (remove duplicates, fix normals)"}
                },
                "required":[]
            }
        }
    async def execute(self, mode: str="auto", prompt: Optional[str]=None, parts: Optional[List[Dict[str,Any]]]=None,
                      output_path: Optional[str]=None, format: Optional[str]=None, seed: Optional[int]=None, 
                      optimize: bool=True, **_):
        if not self.is_available():
            missing=[]; 
            if np is None: missing.append("numpy")
            if trimesh is None: missing.append("trimesh")
            return {"success":False,"error":f"Dependencies missing: {', '.join(missing)}. pip install numpy trimesh"}
        root=_workspace_root()
        out_path = (root / output_path).resolve() if (output_path and output_path.strip() and not os.path.isabs(output_path)) else \
                   (Path(output_path).resolve() if output_path and os.path.isabs(output_path) else _choose_output_path(prompt or mode, ext=(format or None)))
        if not _is_within(root, out_path):
            return {"success":False,"error":f"output_path outside workspace: {out_path} (root: {root})"}
        fmt = (format or "").lower().strip() or _infer_format_from_ext(out_path)
        if mode=="merge":
            design=parts or []
            if not design: return {"success":False,"error":"mode=merge requires non-empty 'parts'"}
        else:
            text=(prompt or "").strip() or "generic model"
            if seed is None: seed=_seed_from_prompt(text)
            random.seed(seed); design=propose_design_from_prompt(text)
        try:
            mesh=_build_mesh_from_parts(design)
            
            # Apply global colors from prompt if specified and in auto mode
            if mode == "auto" and text:
                colors = _extract_colors(text)
                if colors and "color" in colors:
                    mesh = _apply_material(mesh, {"color": colors["color"]})
                
                # Apply additional optimizations if requested
                if optimize:
                    extra_opts = []
                    if _keyword(text, "smooth"): extra_opts.append("smooth")
                    if _keyword(text, "clean", "fix"): extra_opts.extend(["fill_holes", "remove_duplicates"])
                    if extra_opts:
                        mesh = _optimize_mesh(mesh, extra_opts)
                
        except Exception as e:
            return {"success":False,"error":f"mesh build failed: {e}"}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            mesh.export(str(out_path), file_type=fmt)
        except Exception as e:
            return {"success":False,"error":f"export failed: {e}","path":str(out_path)}
        # Calculate additional stats
        volume = 0.0
        surface_area = 0.0
        try:
            if mesh.is_volume:
                volume = float(mesh.volume)
            surface_area = float(mesh.area)
        except Exception:
            pass
        
        return {"success":True,"path":str(out_path),"mode":mode,"prompt":prompt,"parts_count":len(design),
                "stats":{"vertices":int(mesh.vertices.shape[0]),"faces":int(mesh.faces.shape[0]),
                        "volume":volume,"surface_area":surface_area,"seed":seed,"format":fmt,
                        "bounds":mesh.bounds.tolist() if hasattr(mesh.bounds, 'tolist') else None}}

