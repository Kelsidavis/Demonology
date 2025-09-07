# model3d_generator.py
from __future__ import annotations
import math, os, random, re, time, logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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

def _create_textured_trunk(rng: random.Random, base_r: float, top_r: float, height: float, 
                          segments: int = 16, bark_type: str = "oak") -> Dict[str, Any]:
    """Create a realistic trunk with bark texture variations"""
    
    # Create trunk with slight taper and surface variations
    vertices = []
    faces = []
    
    # Generate vertices with bark texture displacement
    for level in range(segments + 1):
        y = height * level / segments
        radius = base_r + (top_r - base_r) * (level / segments)
        
        # Add bark texture variations
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            
            # Bark displacement based on type
            if bark_type in ["oak", "maple", "birch"]:
                # Vertical ridges
                displacement = 0.02 + 0.01 * math.sin(8 * angle) * math.cos(4 * y / height)
            elif bark_type in ["pine", "spruce"]:
                # Scaly bark pattern  
                displacement = 0.01 + 0.008 * math.sin(12 * angle + 2 * y) * math.cos(6 * angle)
            elif bark_type == "palm":
                # Ring patterns
                displacement = 0.005 + 0.003 * math.sin(20 * y / height)
            else:
                # Default smooth with slight variation
                displacement = 0.005 + 0.002 * rng.uniform(-1, 1)
            
            actual_radius = radius + displacement
            x = actual_radius * math.cos(angle)
            z = actual_radius * math.sin(angle)
            vertices.append([x, y, z])
    
    # Generate faces
    for level in range(segments):
        for i in range(segments):
            v1 = level * segments + i
            v2 = level * segments + (i + 1) % segments
            v3 = (level + 1) * segments + i  
            v4 = (level + 1) * segments + (i + 1) % segments
            faces.extend([[v1, v2, v4], [v1, v4, v3]])
    
    return {
        "mode": "custom",
        "type": "trunk", 
        "vertices": vertices,
        "faces": faces,
        "material": {"color": _get_bark_color(bark_type), "generate_uvs": True}
    }

def _get_bark_color(bark_type: str) -> List[float]:
    """Get realistic bark colors for different tree types"""
    colors = {
        "oak": [0.4, 0.3, 0.2],      # Dark brown
        "birch": [0.9, 0.9, 0.85],   # White/cream  
        "pine": [0.5, 0.4, 0.3],     # Reddish brown
        "maple": [0.45, 0.35, 0.25], # Medium brown
        "cherry": [0.6, 0.3, 0.2],   # Reddish
        "palm": [0.6, 0.5, 0.4],     # Tan/brown
        "willow": [0.3, 0.3, 0.25],  # Gray-brown
    }
    return colors.get(bark_type, [0.4, 0.3, 0.2])

def _generate_branch_system(rng: random.Random, trunk_h: float, trunk_r: float, 
                           tree_type: str, complexity: int = 3) -> List[Dict[str, Any]]:
    """Generate realistic branching using simplified L-system principles"""
    branches = []
    
    # Branch parameters by tree type
    if tree_type in ["oak", "maple", "elm"]:
        branch_angle = 45  # Degrees from vertical
        branch_levels = complexity
        branch_density = 0.8
    elif tree_type in ["pine", "spruce", "fir"]:
        branch_angle = 25  # More upward
        branch_levels = complexity + 1
        branch_density = 1.2  # Denser branching
    elif tree_type == "willow":
        branch_angle = 60  # Drooping
        branch_levels = complexity
        branch_density = 1.0
    else:
        branch_angle = 40
        branch_levels = complexity 
        branch_density = 0.9
    
    # Generate primary branches
    primary_branches = max(3, int(6 * branch_density))
    branch_start_height = trunk_h * 0.4  # Start 40% up the trunk
    
    for i in range(primary_branches):
        # Position around trunk
        angle = 2 * math.pi * i / primary_branches + rng.uniform(-0.3, 0.3)
        height = branch_start_height + (trunk_h * 0.5) * (i / primary_branches)
        
        # Primary branch dimensions
        branch_length = trunk_h * rng.uniform(0.3, 0.6)
        branch_radius = trunk_r * rng.uniform(0.2, 0.4)
        
        # Branch direction
        direction = [
            math.cos(angle) * math.sin(math.radians(branch_angle)),
            math.cos(math.radians(branch_angle)), 
            math.sin(angle) * math.sin(math.radians(branch_angle))
        ]
        
        # Position branch
        branch_center = [
            direction[0] * branch_length * 0.5,
            height + direction[1] * branch_length * 0.5,
            direction[2] * branch_length * 0.5
        ]
        
        # Calculate rotation to align with direction
        pitch = math.degrees(math.asin(direction[1]))
        yaw = math.degrees(math.atan2(direction[2], direction[0]))
        
        branches.append({
            "mode": "primitive",
            "primitive": "cylinder",
            "radius": branch_radius,
            "height": branch_length,
            "segments": 8,
            "transform": {
                "translate": branch_center,
                "rotate_deg": [pitch, yaw, 0],
                "material": {"color": _get_bark_color(tree_type)}
            }
        })
        
        # Generate secondary branches
        if branch_levels > 1:
            sub_branches = rng.randint(2, 4)
            for j in range(sub_branches):
                sub_length = branch_length * rng.uniform(0.3, 0.7)
                sub_radius = branch_radius * rng.uniform(0.4, 0.7)
                
                # Position along primary branch
                t = 0.3 + 0.6 * j / sub_branches
                sub_base = [
                    direction[0] * branch_length * t,
                    height + direction[1] * branch_length * t,
                    direction[2] * branch_length * t
                ]
                
                # Sub-branch direction (spread out from primary)
                sub_angle = angle + rng.uniform(-1.0, 1.0)
                sub_pitch = branch_angle + rng.uniform(-15, 15)
                
                sub_direction = [
                    math.cos(sub_angle) * math.sin(math.radians(sub_pitch)),
                    math.cos(math.radians(sub_pitch)),
                    math.sin(sub_angle) * math.sin(math.radians(sub_pitch))
                ]
                
                sub_center = [
                    sub_base[0] + sub_direction[0] * sub_length * 0.5,
                    sub_base[1] + sub_direction[1] * sub_length * 0.5,
                    sub_base[2] + sub_direction[2] * sub_length * 0.5
                ]
                
                sub_yaw = math.degrees(math.atan2(sub_direction[2], sub_direction[0]))
                sub_pitch_deg = math.degrees(math.asin(sub_direction[1]))
                
                branches.append({
                    "mode": "primitive",
                    "primitive": "cylinder", 
                    "radius": sub_radius,
                    "height": sub_length,
                    "segments": 6,
                    "transform": {
                        "translate": sub_center,
                        "rotate_deg": [sub_pitch_deg, sub_yaw, 0],
                        "material": {"color": _get_bark_color(tree_type)}
                    }
                })
    
    return branches

def _generate_foliage_clusters(rng: random.Random, branches: List[Dict[str, Any]], 
                              tree_type: str, season: str = "summer", age: str = "mature") -> List[Dict[str, Any]]:
    """Generate realistic foliage clusters at branch endpoints"""
    foliage = []
    
    # Foliage parameters by tree type and season
    if tree_type in ["oak", "maple", "elm"]:
        if season == "spring":
            leaf_color = [0.4, 0.8, 0.2]  # Light green
            density = 0.7
        elif season == "summer":
            leaf_color = [0.2, 0.6, 0.1]  # Dark green
            density = 1.0
        elif season == "fall":
            leaf_color = [0.8, 0.5, 0.1] if tree_type == "maple" else [0.6, 0.4, 0.1]  # Orange/brown
            density = 0.8
        else:  # winter
            return foliage  # No leaves
            
        cluster_size = 0.8 if age == "young" else 1.2 if age == "mature" else 1.5
        
    elif tree_type in ["pine", "spruce", "fir"]:
        leaf_color = [0.1, 0.4, 0.1]  # Dark green (evergreen)
        density = 1.2
        cluster_size = 0.6 if age == "young" else 1.0 if age == "mature" else 1.3
        
    elif tree_type == "palm":
        leaf_color = [0.2, 0.7, 0.2]  # Tropical green
        density = 0.6  
        cluster_size = 2.0  # Large fronds
        
    else:
        leaf_color = [0.3, 0.6, 0.2]  # Default green
        density = 0.9
        cluster_size = 1.0
    
    # Generate foliage clusters at branch endpoints
    for branch in branches:
        transform = branch.get("transform", {})
        pos = transform.get("translate", [0, 0, 0])
        
        # Skip if too close to trunk (lower branches)
        if pos[1] < branches[0].get("transform", {}).get("translate", [0, 0, 0])[1]:
            continue
            
        # Create foliage cluster
        num_clusters = max(1, int(density * rng.randint(1, 3)))
        
        for i in range(num_clusters):
            cluster_offset = [
                rng.uniform(-cluster_size*0.3, cluster_size*0.3),
                rng.uniform(-cluster_size*0.2, cluster_size*0.4),
                rng.uniform(-cluster_size*0.3, cluster_size*0.3)
            ]
            
            cluster_pos = [
                pos[0] + cluster_offset[0],
                pos[1] + cluster_offset[1], 
                pos[2] + cluster_offset[2]
            ]
            
            if tree_type == "palm":
                # Palm fronds - elongated shapes
                foliage.append({
                    "mode": "primitive",
                    "primitive": "box",  # Will be stretched for frond shape
                    "size": [cluster_size*2, 0.1, cluster_size*0.5],
                    "transform": {
                        "translate": cluster_pos,
                        "rotate_deg": [rng.uniform(-20, 20), rng.uniform(0, 360), rng.uniform(-10, 10)],
                        "material": {"color": leaf_color}
                    }
                })
            elif tree_type in ["pine", "spruce", "fir"]:
                # Needle clusters - smaller, denser
                foliage.append({
                    "mode": "primitive", 
                    "primitive": "sphere",
                    "radius": cluster_size * rng.uniform(0.4, 0.7),
                    "subdivisions": 2,
                    "transform": {
                        "translate": cluster_pos,
                        "scale": [1.0, 0.6, 1.0],  # Flatten slightly
                        "material": {"color": leaf_color}
                    }
                })
            else:
                # Deciduous leaves - varied cluster sizes
                foliage.append({
                    "mode": "primitive",
                    "primitive": "sphere",
                    "radius": cluster_size * rng.uniform(0.6, 1.2),
                    "subdivisions": 3,
                    "transform": {
                        "translate": cluster_pos,
                        "material": {"color": leaf_color}
                    }
                })
    
    return foliage

def _generate_fibonacci_sequence(n: int) -> List[int]:
    """Generate Fibonacci sequence up to n terms"""
    if n <= 0:
        return []
    elif n == 1:
        return [1]
    elif n == 2:
        return [1, 1]
    
    fib = [1, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

def _fibonacci_spiral_points(num_points: int, height: float, radius: float) -> List[Tuple[float, float, float]]:
    """Generate points along a Fibonacci spiral (golden spiral)"""
    points = []
    golden_ratio = (1 + math.sqrt(5)) / 2  # φ = 1.618...
    golden_angle = 2 * math.pi / (golden_ratio * golden_ratio)  # ≈ 137.5°
    
    for i in range(num_points):
        # Height progression
        y = height * (i / max(1, num_points - 1))
        
        # Spiral angle based on golden angle
        theta = i * golden_angle
        
        # Radius varies with height (taper effect)
        r = radius * (1.0 - 0.3 * (i / max(1, num_points - 1)))
        
        # Convert to Cartesian coordinates
        x = r * math.cos(theta)
        z = r * math.sin(theta)
        
        points.append((x, y, z))
    
    return points

def _generate_fibonacci_branches(rng: random.Random, trunk_h: float, trunk_r: float, 
                               tree_type: str, complexity: int = 3) -> List[Dict[str, Any]]:
    """Generate branches using Fibonacci spiral patterns for natural distribution"""
    branches = []
    
    # Calculate number of branch points based on tree type and complexity
    if tree_type in ["oak", "maple", "elm"]:
        base_points = 8 + complexity * 3
        branch_thickness = trunk_r * 0.3
        branch_length_factor = 0.6
    elif tree_type in ["pine", "spruce", "fir"]:
        base_points = 12 + complexity * 4  # More branches for conifers
        branch_thickness = trunk_r * 0.2
        branch_length_factor = 0.8
    elif tree_type == "willow":
        base_points = 15 + complexity * 5  # Very branchy
        branch_thickness = trunk_r * 0.15
        branch_length_factor = 1.2  # Longer drooping branches
    else:
        base_points = 10 + complexity * 3
        branch_thickness = trunk_r * 0.25
        branch_length_factor = 0.7
    
    # Generate Fibonacci spiral points for branch attachment
    fib_points = _fibonacci_spiral_points(base_points, trunk_h * 0.7, trunk_r * 1.2)
    
    # Create branches at Fibonacci points
    for i, (x, y, z) in enumerate(fib_points):
        if y < trunk_h * 0.3:  # Skip lower third of trunk
            continue
            
        # Branch properties based on position in sequence
        fib_seq = _generate_fibonacci_sequence(10)
        fib_index = i % len(fib_seq)
        fib_weight = fib_seq[fib_index] / max(fib_seq)
        
        # Branch length varies with Fibonacci weight and height
        height_factor = (trunk_h - y) / trunk_h  # Longer branches higher up
        base_length = trunk_h * branch_length_factor * height_factor
        branch_length = base_length * (0.5 + 0.5 * fib_weight)
        
        # Branch thickness decreases with distance from trunk
        thickness = branch_thickness * (0.3 + 0.7 * height_factor) * fib_weight
        
        # Calculate branch direction (outward from trunk)
        branch_angle = math.atan2(z, x)
        
        # Add natural variation
        branch_angle += rng.uniform(-0.3, 0.3)
        elevation_angle = rng.uniform(-0.2, 0.4)  # Slight upward bias
        
        # Branch end point
        end_x = x + branch_length * math.cos(branch_angle) * math.cos(elevation_angle)
        end_y = y + branch_length * math.sin(elevation_angle)
        end_z = z + branch_length * math.sin(branch_angle) * math.cos(elevation_angle)
        
        # Create branch cylinder
        branch_center_x = x + (end_x - x) * 0.5
        branch_center_y = y + (end_y - y) * 0.5
        branch_center_z = z + (end_z - z) * 0.5
        
        # Calculate rotation to align with branch direction
        branch_dir = np.array([end_x - x, end_y - y, end_z - z])
        branch_dir = branch_dir / np.linalg.norm(branch_dir)
        
        # Convert direction to rotation angles
        rot_y = math.degrees(math.atan2(branch_dir[0], branch_dir[2]))
        rot_x = math.degrees(-math.asin(branch_dir[1]))
        
        branches.append({
            "mode": "primitive",
            "primitive": "cylinder",
            "radius": thickness,
            "height": branch_length,
            "segments": 12,
            "transform": {
                "translate": [branch_center_x, branch_center_y, branch_center_z],
                "rotate_deg": [rot_x, rot_y, 0],
                "material": {
                    "color": [0.4, 0.25, 0.1],  # Brown bark
                    "generate_uvs": True
                }
            }
        })
        
        # Add secondary branches for higher complexity
        if complexity > 2 and fib_weight > 0.6:
            # Generate 2-3 smaller secondary branches
            num_secondary = min(3, int(fib_weight * 4))
            for j in range(num_secondary):
                sec_angle = rng.uniform(0, 2 * math.pi)
                sec_elevation = rng.uniform(-0.1, 0.3)
                sec_length = branch_length * rng.uniform(0.3, 0.6)
                sec_thickness = thickness * rng.uniform(0.4, 0.7)
                
                sec_end_x = end_x + sec_length * math.cos(sec_angle) * math.cos(sec_elevation)
                sec_end_y = end_y + sec_length * math.sin(sec_elevation)
                sec_end_z = end_z + sec_length * math.sin(sec_angle) * math.cos(sec_elevation)
                
                sec_center_x = end_x + (sec_end_x - end_x) * 0.5
                sec_center_y = end_y + (sec_end_y - end_y) * 0.5
                sec_center_z = end_z + (sec_end_z - end_z) * 0.5
                
                sec_dir = np.array([sec_end_x - end_x, sec_end_y - end_y, sec_end_z - end_z])
                if np.linalg.norm(sec_dir) > 0:
                    sec_dir = sec_dir / np.linalg.norm(sec_dir)
                    sec_rot_y = math.degrees(math.atan2(sec_dir[0], sec_dir[2]))
                    sec_rot_x = math.degrees(-math.asin(max(-1, min(1, sec_dir[1]))))
                    
                    branches.append({
                        "mode": "primitive",
                        "primitive": "cylinder",
                        "radius": sec_thickness,
                        "height": sec_length,
                        "segments": 8,
                        "transform": {
                            "translate": [sec_center_x, sec_center_y, sec_center_z],
                            "rotate_deg": [sec_rot_x, sec_rot_y, 0],
                            "material": {
                                "color": [0.4, 0.25, 0.1],
                                "generate_uvs": True
                            }
                        }
                    })
    
    return branches

def _add_bark_texture_uvs(mesh: 'trimesh.Trimesh', primitive_type: str = "cylinder") -> 'trimesh.Trimesh':
    """Add UV texture coordinates for bark texturing"""
    if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
        return mesh
        
    vertices = mesh.vertices
    uvs = np.zeros((len(vertices), 2))
    
    if primitive_type == "cylinder":
        # Cylindrical UV mapping for trunk/branches
        center = np.mean(vertices, axis=0)
        
        for i, vertex in enumerate(vertices):
            # Translate to center
            relative = vertex - center
            
            # Calculate cylindrical coordinates
            theta = math.atan2(relative[2], relative[0])  # Angle around Y-axis
            height = relative[1]  # Y coordinate
            
            # Normalize to 0-1 range
            u = (theta + math.pi) / (2 * math.pi)  # 0-1 around circumference
            
            # V coordinate based on height (normalized)
            min_y, max_y = np.min(vertices[:, 1]), np.max(vertices[:, 1])
            if max_y > min_y:
                v = (height - min_y) / (max_y - min_y)
            else:
                v = 0.5
            
            uvs[i] = [u, v]
            
    elif primitive_type == "sphere":
        # Spherical UV mapping for foliage
        center = np.mean(vertices, axis=0)
        
        for i, vertex in enumerate(vertices):
            relative = vertex - center
            length = np.linalg.norm(relative)
            
            if length > 0:
                # Normalize
                relative = relative / length
                
                # Spherical coordinates
                theta = math.atan2(relative[2], relative[0])
                phi = math.acos(max(-1, min(1, relative[1])))
                
                u = (theta + math.pi) / (2 * math.pi)
                v = phi / math.pi
                
                uvs[i] = [u, v]
            else:
                uvs[i] = [0.5, 0.5]
    
    # Add UV coordinates to mesh
    if hasattr(mesh.visual, 'uv'):
        mesh.visual.uv = uvs
    else:
        try:
            # Create texture visual if it doesn't exist
            mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
        except:
            # Fallback: store as vertex attribute
            pass
    
    return mesh

def _generate_root_system(rng: random.Random, trunk_r: float, tree_type: str, 
                         environment: str = "normal") -> List[Dict[str, Any]]:
    """Generate visible root systems"""
    roots = []
    
    # Root parameters by tree type
    if tree_type in ["oak", "maple"]:
        root_spread = trunk_r * 4
        root_depth = trunk_r * 0.5
        num_major_roots = 5
    elif tree_type in ["pine", "spruce"]:
        root_spread = trunk_r * 3  
        root_depth = trunk_r * 0.3
        num_major_roots = 4
    elif tree_type == "willow":
        root_spread = trunk_r * 6  # Wide spreading
        root_depth = trunk_r * 0.8
        num_major_roots = 7
    else:
        root_spread = trunk_r * 3.5
        root_depth = trunk_r * 0.4
        num_major_roots = 4
    
    # Environmental adaptations
    if environment == "dry":
        root_depth *= 1.5  # Deeper roots
        root_spread *= 0.8
    elif environment == "wet":
        root_spread *= 1.3  # Wider surface roots
        root_depth *= 0.7
    
    # Generate major surface roots
    for i in range(num_major_roots):
        angle = 2 * math.pi * i / num_major_roots + rng.uniform(-0.5, 0.5)
        
        # Root extends outward and downward
        root_length = root_spread * rng.uniform(0.8, 1.2)
        root_radius = trunk_r * rng.uniform(0.15, 0.25)
        
        # Root position and direction
        end_x = math.cos(angle) * root_length
        end_z = math.sin(angle) * root_length  
        end_y = -root_depth * rng.uniform(0.5, 1.0)
        
        # Root segments (curved)
        segments = 3
        for seg in range(segments):
            t = seg / segments
            seg_start = [end_x * t, end_y * (t**1.5), end_z * t]  # Curved downward
            seg_end = [end_x * (t + 1/segments), end_y * ((t + 1/segments)**1.5), end_z * (t + 1/segments)]
            
            seg_center = [(seg_start[0] + seg_end[0])/2, (seg_start[1] + seg_end[1])/2, (seg_start[2] + seg_end[2])/2]
            seg_length = math.sqrt(sum((seg_end[i] - seg_start[i])**2 for i in range(3)))
            
            # Calculate rotation to align with segment direction
            direction = [(seg_end[i] - seg_start[i])/seg_length for i in range(3)]
            yaw = math.degrees(math.atan2(direction[2], direction[0]))
            pitch = math.degrees(math.asin(-direction[1]))  # Negative because roots go down
            
            roots.append({
                "mode": "primitive",
                "primitive": "cylinder",
                "radius": root_radius * (1 - 0.3 * t),  # Taper
                "height": seg_length,
                "segments": 6,
                "transform": {
                    "translate": seg_center,
                    "rotate_deg": [pitch, yaw, 0],
                    "material": {"color": [0.3, 0.2, 0.1]}  # Root brown
                }
            })
    
    return roots

def _design_tree(rng: random.Random, knobs: Dict[str, float]) -> List[Dict[str, Any]]:
    """Generate realistic trees with advanced branching, foliage, and root systems"""
    text = knobs.get("text", "")
    H = knobs.get("height", 8.0)
    trunk_base_r = knobs.get("radius", H * 0.04)  # More realistic trunk scaling
    
    # Determine tree type from keywords
    tree_type = "oak"  # default
    if _keyword(text, "oak"): tree_type = "oak"
    elif _keyword(text, "maple"): tree_type = "maple"  
    elif _keyword(text, "pine", "spruce", "evergreen", "fir"): tree_type = "pine"
    elif _keyword(text, "birch"): tree_type = "birch"
    elif _keyword(text, "palm"): tree_type = "palm"
    elif _keyword(text, "willow"): tree_type = "willow"
    elif _keyword(text, "cherry"): tree_type = "cherry"
    elif _keyword(text, "elm"): tree_type = "elm"
    
    # Determine season and age
    season = "summer"
    if _keyword(text, "spring"): season = "spring"
    elif _keyword(text, "fall", "autumn"): season = "fall" 
    elif _keyword(text, "winter"): season = "winter"
    
    age = "mature"
    if _keyword(text, "young", "sapling"): age = "young"
    elif _keyword(text, "old", "ancient", "giant"): age = "old"
    
    # Environment
    environment = "normal"
    if _keyword(text, "dry", "desert", "arid"): environment = "dry"
    elif _keyword(text, "wet", "swamp", "marsh"): environment = "wet"
    
    parts = []
    
    # Trunk dimensions based on tree type and age
    if tree_type == "palm":
        trunk_h = H * 0.8
        trunk_top_r = trunk_base_r * 0.9  # Palm trunks don't taper much
    else:
        trunk_h = H * (0.35 if age == "young" else 0.45 if age == "mature" else 0.5)
        trunk_top_r = trunk_base_r * (0.8 if age == "young" else 0.7 if age == "mature" else 0.6)
    
    # Generate textured trunk
    trunk = _create_textured_trunk(rng, trunk_base_r, trunk_top_r, trunk_h, 16, tree_type)
    parts.append(trunk)
    
    # Generate branch system (skip for palm trees)
    if tree_type != "palm":
        complexity = 2 if age == "young" else 3 if age == "mature" else 4
        
        # Use Fibonacci branching for more natural distribution
        use_fibonacci = _keyword(text, "fibonacci", "spiral", "natural", "golden") or True  # Default to Fibonacci
        if use_fibonacci:
            branches = _generate_fibonacci_branches(rng, trunk_h, trunk_base_r, tree_type, complexity)
        else:
            branches = _generate_branch_system(rng, trunk_h, trunk_base_r, tree_type, complexity)
        parts.extend(branches)
        
        # Generate foliage
        foliage = _generate_foliage_clusters(rng, branches, tree_type, season, age)
        parts.extend(foliage)
    else:
        # Palm tree crown - fronds at top
        crown_y = trunk_h * 0.9
        num_fronds = rng.randint(8, 12)
        for i in range(num_fronds):
            angle = 2 * math.pi * i / num_fronds + rng.uniform(-0.3, 0.3)
            frond_length = H * 0.4
            
            frond_end = [
                math.cos(angle) * frond_length * 0.7,
                crown_y + frond_length * 0.3,  # Slightly upward
                math.sin(angle) * frond_length * 0.7
            ]
            
            parts.append({
                "mode": "primitive",
                "primitive": "box",
                "size": [frond_length, 0.1, frond_length * 0.3],
                "transform": {
                    "translate": frond_end,
                    "rotate_deg": [0, math.degrees(angle), rng.uniform(-15, 15)],
                    "material": {"color": [0.2, 0.7, 0.2]}
                }
            })
    
    # Generate root system if requested  
    if _keyword(text, "roots", "root") or environment in ["dry", "wet"]:
        roots = _generate_root_system(rng, trunk_base_r, tree_type, environment)
        parts.extend(roots)
    
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

def _design_bridge(rng: random.Random, knobs: Dict[str, float]) -> List[Dict[str, Any]]:
    L = knobs.get("length", 20.0); W = knobs.get("width", 4.0); H = knobs.get("height", 8.0)
    parts = []
    
    # Main deck
    deck_thick = H * 0.15
    parts.append({"mode":"primitive","primitive":"box","size":[L,deck_thick,W],
                  "transform":{"translate":[0,H*0.4,0]}})
    
    # Support pillars
    pillar_r = W * 0.1
    pillar_h = H * 0.5
    num_pillars = max(2, int(L / 6))
    for i in range(num_pillars):
        x = -L*0.4 + i * (L*0.8 / (num_pillars-1))
        for z in [-W*0.4, W*0.4]:
            parts.append({"mode":"primitive","primitive":"cylinder","radius":pillar_r,"height":pillar_h,"segments":16,
                          "transform":{"translate":[x,pillar_h*0.5,z]}})
    
    # Suspension cables (decorative)
    if _keyword(knobs.get("text",""), "suspension", "cable"):
        tower_h = H * 1.5
        for x in [-L*0.3, L*0.3]:
            parts.append({"mode":"primitive","primitive":"box","size":[W*0.2,tower_h,W*0.15],
                          "transform":{"translate":[x,tower_h*0.5,0]}})
    
    return parts

def _design_terrain(rng: random.Random, knobs: Dict[str, float]) -> List[Dict[str, Any]]:
    W = knobs.get("width", 15.0); D = knobs.get("length", 15.0)
    scale = knobs.get("scale", 1.0)
    parts = []
    
    # Base terrain
    base_h = scale * 0.5
    parts.append({"mode":"primitive","primitive":"box","size":[D,base_h,W],
                  "transform":{"translate":[0,base_h*0.5,0]}})
    
    # Hills/mounds
    num_hills = rng.randint(3, 6)
    for i in range(num_hills):
        hill_r = rng.uniform(1.0, 3.0) * scale
        hill_h = rng.uniform(0.8, 2.5) * scale
        x = rng.uniform(-D*0.4, D*0.4)
        z = rng.uniform(-W*0.4, W*0.4)
        parts.append({"mode":"primitive","primitive":"sphere","radius":hill_r,"subdivisions":2,
                      "transform":{"translate":[x,base_h+hill_h*0.3,z],"scale":1.0}})
    
    return parts

def _design_mech(rng: random.Random, knobs: Dict[str, float]) -> List[Dict[str, Any]]:
    H = knobs.get("height", 8.0); scale = knobs.get("scale", 1.0)
    parts = []
    
    # Main torso
    torso_w, torso_h, torso_d = H*0.4, H*0.3, H*0.25
    parts.append({"mode":"primitive","primitive":"box","size":[torso_w,torso_h,torso_d],
                  "transform":{"translate":[0,H*0.6,0]}})
    
    # Head
    head_r = H * 0.12
    parts.append({"mode":"primitive","primitive":"sphere","radius":head_r,"subdivisions":3,
                  "transform":{"translate":[0,H*0.85,0]}})
    
    # Arms with symmetry
    arm_l, arm_r = H*0.35, H*0.08
    parts.append({"mode":"primitive","primitive":"cylinder","radius":arm_r,"height":arm_l,"segments":12,
                  "transform":{"translate":[torso_w*0.7,H*0.65,0],"rotate_deg":[0,0,20]},
                  "symmetry":{"axis":"x"}})
    
    # Legs with symmetry  
    leg_l, leg_r = H*0.45, H*0.1
    parts.append({"mode":"primitive","primitive":"cylinder","radius":leg_r,"height":leg_l,"segments":12,
                  "transform":{"translate":[torso_w*0.15,leg_l*0.5,0]},
                  "symmetry":{"axis":"x"}})
    
    # Weapons (optional)
    if _keyword(knobs.get("text",""), "armed", "weapon", "gun"):
        parts.append({"mode":"primitive","primitive":"cylinder","radius":H*0.04,"height":H*0.6,"segments":8,
                      "transform":{"translate":[torso_w*0.8,H*0.65,0],"rotate_deg":[0,0,45]}})
    
    return parts

def _design_drone(rng: random.Random, knobs: Dict[str, float]) -> List[Dict[str, Any]]:
    scale = knobs.get("scale", 1.0); R = scale * 2.0
    parts = []
    
    # Main body
    body_r = R * 0.3
    parts.append({"mode":"primitive","primitive":"sphere","radius":body_r,"subdivisions":3,
                  "transform":{"translate":[0,0,0],"scale":1.0}})
    
    # Rotors with array
    rotor_r = R * 0.8
    num_rotors = 4 if not _keyword(knobs.get("text",""), "hex", "six") else 6
    angle_step = 360 / num_rotors
    
    for i in range(num_rotors):
        angle = math.radians(i * angle_step)
        x = R * math.cos(angle)
        z = R * math.sin(angle)
        # Rotor arm
        parts.append({"mode":"primitive","primitive":"cylinder","radius":body_r*0.2,"height":R*0.8,"segments":8,
                      "transform":{"translate":[x*0.6,0,z*0.6],"rotate_deg":[0,math.degrees(angle),0]}})
        # Rotor disc
        parts.append({"mode":"primitive","primitive":"cylinder","radius":rotor_r,"height":0.05,"segments":16,
                      "transform":{"translate":[x,body_r*0.3,z]}})
    
    return parts

def _design_lamp(rng: random.Random, knobs: Dict[str, float]) -> List[Dict[str, Any]]:
    H = knobs.get("height", 6.0); scale = knobs.get("scale", 1.0)
    parts = []
    
    # Base
    base_r = H * 0.2
    parts.append({"mode":"primitive","primitive":"cylinder","radius":base_r,"height":H*0.1,"segments":16,
                  "transform":{"translate":[0,H*0.05,0]}})
    
    # Pole
    pole_r = H * 0.03
    pole_h = H * 0.7
    parts.append({"mode":"primitive","primitive":"cylinder","radius":pole_r,"height":pole_h,"segments":12,
                  "transform":{"translate":[0,H*0.1+pole_h*0.5,0]}})
    
    # Shade (different styles based on keywords)
    shade_r = H * 0.25
    shade_h = H * 0.2
    shade_y = H * 0.9
    
    if _keyword(knobs.get("text",""), "round", "globe"):
        parts.append({"mode":"primitive","primitive":"sphere","radius":shade_r,"subdivisions":3,
                      "transform":{"translate":[0,shade_y,0]}})
    elif _keyword(knobs.get("text",""), "cone", "conical"):
        parts.append({"mode":"primitive","primitive":"cone","radius":shade_r,"height":shade_h,"segments":16,
                      "transform":{"translate":[0,shade_y,0],"rotate_deg":[180,0,0]}})
    else:  # Default cylindrical shade
        parts.append({"mode":"primitive","primitive":"cylinder","radius":shade_r,"height":shade_h,"segments":16,
                      "transform":{"translate":[0,shade_y,0]}})
    
    return parts

def _design_vase(rng: random.Random, knobs: Dict[str, float]) -> List[Dict[str, Any]]:
    H = knobs.get("height", 4.0); R = knobs.get("radius", 1.0)
    parts = []
    
    # Base
    base_r = R * 0.8
    base_h = H * 0.15
    parts.append({"mode":"primitive","primitive":"cylinder","radius":base_r,"height":base_h,"segments":24,
                  "transform":{"translate":[0,base_h*0.5,0]}})
    
    # Main body (different profiles based on style)
    if _keyword(knobs.get("text",""), "round", "bulb", "bulbous"):
        # Bulbous vase
        bulge_r = R * 1.2
        bulge_h = H * 0.4
        parts.append({"mode":"primitive","primitive":"sphere","radius":bulge_r,"subdivisions":3,
                      "transform":{"translate":[0,base_h+bulge_h*0.7,0],"scale":1.0}})
        
        # Neck
        neck_r = R * 0.6
        neck_h = H * 0.3
        parts.append({"mode":"primitive","primitive":"cylinder","radius":neck_r,"height":neck_h,"segments":20,
                      "transform":{"translate":[0,base_h+bulge_h+neck_h*0.5,0]}})
    else:
        # Cylindrical vase with taper
        main_h = H * 0.7
        top_r = R * 0.9
        # Use cone for tapered effect
        parts.append({"mode":"primitive","primitive":"cone","radius":base_r,"height":main_h,"segments":24,
                      "transform":{"translate":[0,base_h+main_h*0.5,0]}})
        
        # Rim
        rim_h = H * 0.08
        parts.append({"mode":"primitive","primitive":"cylinder","radius":top_r,"height":rim_h,"segments":24,
                      "transform":{"translate":[0,base_h+main_h+rim_h*0.5,0]}})
    
    return parts

def _apply_style_modifiers(parts: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    """Apply style-based modifications to parts"""
    
    # Sleek/streamlined - reduce dimensions, smoother shapes
    if _keyword(text, "sleek", "streamlined", "smooth", "aerodynamic"):
        for p in parts:
            transform = p.get("transform", {})
            if "scale" not in transform:
                transform["scale"] = 0.85
            # Increase subdivision for smoother shapes
            if p.get("primitive") == "sphere":
                p["subdivisions"] = max(p.get("subdivisions", 3), 4)
            elif p.get("primitive") in ["cylinder", "cone", "torus"]:
                p["segments"] = max(p.get("segments", 24), 32)
    
    # Bulky/robust - increase dimensions, blockier shapes
    elif _keyword(text, "bulky", "robust", "heavy", "thick", "chunky"):
        for p in parts:
            transform = p.get("transform", {})
            if "scale" not in transform:
                transform["scale"] = 1.3
            # Reduce subdivisions for blockier shapes
            if p.get("primitive") == "sphere":
                p["subdivisions"] = min(p.get("subdivisions", 3), 2)
            elif p.get("primitive") in ["cylinder", "cone", "torus"]:
                p["segments"] = min(p.get("segments", 24), 16)
    
    # Organic - add randomization, more subdivision
    elif _keyword(text, "organic", "natural", "flowing", "curved"):
        rng = random.Random(abs(hash(text)) % (2**32))
        for p in parts:
            transform = p.get("transform", {})
            # Add slight random variations
            translate = transform.get("translate", [0, 0, 0])
            for i in range(3):
                translate[i] += rng.uniform(-0.1, 0.1)
            transform["translate"] = translate
            
            # Add slight random rotation
            rotate = transform.get("rotate_deg", [0, 0, 0])
            for i in range(3):
                rotate[i] += rng.uniform(-5, 5)
            transform["rotate_deg"] = rotate
            
            # Increase subdivisions for organic feel
            if p.get("primitive") == "sphere":
                p["subdivisions"] = max(p.get("subdivisions", 3), 4)
    
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
def _design_forest(rng: random.Random, knobs: Dict[str, float]) -> List[Dict[str, Any]]:
    """Generate a collection of trees forming a forest scene"""
    text = knobs.get("text", "")
    area_size = knobs.get("width", knobs.get("length", 20.0))
    tree_count = int(knobs.get("count", max(5, area_size // 3)))
    
    parts = []
    
    # Determine forest type
    if _keyword(text, "pine", "evergreen", "coniferous"):
        tree_types = ["pine", "spruce", "fir"]
    elif _keyword(text, "deciduous", "hardwood"):
        tree_types = ["oak", "maple", "birch", "elm"]
    elif _keyword(text, "mixed"):
        tree_types = ["oak", "maple", "pine", "birch"]
    else:
        tree_types = ["oak", "maple", "pine"]
    
    # Generate trees with natural distribution
    for i in range(tree_count):
        # Random position within area
        x = rng.uniform(-area_size*0.4, area_size*0.4)
        z = rng.uniform(-area_size*0.4, area_size*0.4)
        
        # Ensure some spacing between trees
        min_distance = 2.0
        attempts = 0
        while attempts < 10:
            too_close = False
            for j in range(i):
                if j < len(parts):
                    other_pos = parts[j*10].get("transform", {}).get("translate", [0,0,0])  # Rough estimate
                    distance = math.sqrt((x - other_pos[0])**2 + (z - other_pos[2])**2)
                    if distance < min_distance:
                        too_close = True
                        break
            if not too_close:
                break
            x = rng.uniform(-area_size*0.4, area_size*0.4)
            z = rng.uniform(-area_size*0.4, area_size*0.4)
            attempts += 1
        
        # Select tree type and parameters
        tree_type = rng.choice(tree_types)
        tree_height = rng.uniform(6.0, 12.0)
        # Weighted choice using random selection
        rand_val = rng.random()
        if rand_val < 0.3:
            tree_age = "young"
        elif rand_val < 0.8:  # 0.3 + 0.5
            tree_age = "mature"
        else:
            tree_age = "old"
        
        # Create tree-specific prompt
        tree_prompt = f"{tree_age} {tree_type} tree height:{tree_height}"
        if _keyword(text, "spring", "fall", "winter"):
            season = next(word for word in ["spring", "fall", "winter"] if word in text.lower())
            tree_prompt += f" {season}"
        if _keyword(text, "roots"):
            tree_prompt += " roots"
            
        # Generate individual tree
        tree_knobs = _extract_numbers(tree_prompt)
        tree_knobs["text"] = tree_prompt
        tree_knobs["height"] = tree_height
        
        tree_parts = _design_tree(rng, tree_knobs)
        
        # Translate tree to position
        for part in tree_parts:
            transform = part.get("transform", {})
            translate = transform.get("translate", [0, 0, 0])
            transform["translate"] = [translate[0] + x, translate[1], translate[2] + z]
            part["transform"] = transform
        
        parts.extend(tree_parts)
    
    return parts

def _design_orchard(rng: random.Random, knobs: Dict[str, float]) -> List[Dict[str, Any]]:
    """Generate an organized orchard with fruit trees"""
    text = knobs.get("text", "")
    rows = int(knobs.get("rows", 3))
    cols = int(knobs.get("columns", 4))
    spacing = knobs.get("spacing", 4.0)
    
    parts = []
    
    # Determine fruit tree type
    if _keyword(text, "apple"): tree_type = "apple"
    elif _keyword(text, "cherry"): tree_type = "cherry" 
    elif _keyword(text, "peach"): tree_type = "peach"
    elif _keyword(text, "pear"): tree_type = "pear"
    else: tree_type = "apple"
    
    # Generate trees in organized grid
    for row in range(rows):
        for col in range(cols):
            x = (col - cols/2) * spacing
            z = (row - rows/2) * spacing
            
            # Tree parameters
            tree_height = rng.uniform(4.0, 8.0)  # Smaller fruit trees
            
            # Create tree prompt
            tree_prompt = f"{tree_type} tree height:{tree_height}"
            
            # Generate tree
            tree_knobs = _extract_numbers(tree_prompt)
            tree_knobs["text"] = tree_prompt
            tree_knobs["height"] = tree_height
            
            tree_parts = _design_tree(rng, tree_knobs)
            
            # Position tree
            for part in tree_parts:
                transform = part.get("transform", {})
                translate = transform.get("translate", [0, 0, 0])
                transform["translate"] = [translate[0] + x, translate[1], translate[2] + z]
                part["transform"] = transform
            
            parts.extend(tree_parts)
    
    return parts

def propose_design_from_prompt(prompt: str) -> List[Dict[str, Any]]:
    text=(prompt or "").strip(); rng=random.Random(abs(hash(text))%(2**32)); knobs=_extract_numbers(text); knobs["text"]=text
    
    # Select base design
    design = None
    if _keyword(text,"spaceship","space ship","starfighter","craft"): design = _design_spaceship(rng,knobs)
    elif _keyword(text,"rocket","missile"): design = _design_rocket(rng,knobs)
    elif _keyword(text,"house","hut","cabin"): design = _design_house(rng,knobs)
    elif _keyword(text,"car","vehicle","auto"): design = _design_car(rng,knobs)
    elif _keyword(text,"building","skyscraper","tower","office"): design = _design_building(rng,knobs)
    elif _keyword(text,"forest","woods","grove"): design = _design_forest(rng,knobs)
    elif _keyword(text,"orchard","fruit trees"): design = _design_orchard(rng,knobs)
    elif _keyword(text,"tree","oak","pine","palm","birch","maple","willow","cherry","elm"): design = _design_tree(rng,knobs)
    elif _keyword(text,"chair","table","desk","furniture","bookshelf","shelf"): design = _design_furniture(rng,knobs)
    elif _keyword(text,"sword","blade","weapon","spear","axe","knife"): design = _design_weapon(rng,knobs)
    elif _keyword(text,"bridge","overpass","viaduct"): design = _design_bridge(rng,knobs)
    elif _keyword(text,"terrain","landscape","hills","ground"): design = _design_terrain(rng,knobs)
    elif _keyword(text,"mech","robot","android","bot"): design = _design_mech(rng,knobs)
    elif _keyword(text,"drone","quadcopter","multirotor"): design = _design_drone(rng,knobs)
    elif _keyword(text,"lamp","light","fixture"): design = _design_lamp(rng,knobs)
    elif _keyword(text,"vase","pot","vessel","urn"): design = _design_vase(rng,knobs)
    else: design = _design_generic(rng,knobs)
    
    # Apply style modifiers
    design = _apply_style_modifiers(design, text)
    
    return design

def _primitive_mesh(spec: Dict[str, Any]):
    prim=spec.get("primitive")
    mesh = None
    
    if prim=="box":
        sx,sy,sz=spec.get("size",[1,1,1]); mesh = trimesh.creation.box(extents=[sx,sy,sz])
    elif prim=="cylinder":
        r=float(spec.get("radius",0.5)); h=float(spec.get("height",1.0)); seg=int(spec.get("segments",24))
        mesh = trimesh.creation.cylinder(radius=r, height=h, sections=seg)
    elif prim=="sphere":
        r=float(spec.get("radius",0.75)); sub=int(spec.get("subdivisions",3))
        mesh = trimesh.creation.icosphere(subdivisions=sub, radius=r)
    elif prim=="cone":
        r=float(spec.get("radius",0.5)); h=float(spec.get("height",1.0)); seg=int(spec.get("segments",24))
        mesh = trimesh.creation.cone(radius=r, height=h, sections=seg)
    elif prim=="prism":
        sx,sy,sz=spec.get("size",[2,1,2]); mesh = trimesh.creation.box(extents=[sx,sy,sz])
    # New primitives  
    elif prim=="torus":
        major_r=float(spec.get("radius",1.0)); minor_r=float(spec.get("inner_radius",0.3))
        seg_major=int(spec.get("segments",24)); seg_minor=int(spec.get("minor_segments",12))
        mesh = _create_torus(major_r, minor_r, seg_major, seg_minor)
    elif prim=="wedge":
        sx,sy,sz=spec.get("size",[2,1,2]); mesh = _create_wedge(sx,sy,sz)
    elif prim=="tube":
        r=float(spec.get("radius",0.5)); inner_r=float(spec.get("inner_radius",0.3))
        h=float(spec.get("height",1.0)); seg=int(spec.get("segments",24))
        mesh = _create_tube(r, inner_r, h, seg)
    elif prim=="helix":
        r=float(spec.get("radius",0.5)); h=float(spec.get("height",2.0))
        turns=float(spec.get("turns",3.0)); seg=int(spec.get("segments",64))
        mesh = _create_helix(r, h, turns, seg)
    
    # Add UV texture coordinates if requested
    if mesh is not None:
        material = spec.get("transform", {}).get("material", {})
        if material.get("generate_uvs", False):
            try:
                if prim in ["cylinder", "cone", "tube"]:
                    mesh = _add_bark_texture_uvs(mesh, "cylinder")
                elif prim in ["sphere"]:
                    mesh = _add_bark_texture_uvs(mesh, "sphere")
                elif prim in ["torus"]:
                    mesh = _add_bark_texture_uvs(mesh, "sphere")  # Use spherical mapping for torus
            except Exception as e:
                # UV mapping failed, continue without UVs
                pass
    
    return mesh

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
        if isinstance(scale, (list, tuple)) and len(scale) >= 3:
            # Non-uniform scaling
            sx, sy, sz = float(scale[0]), float(scale[1]), float(scale[2])
            S = np.eye(4); S[0,0] = sx; S[1,1] = sy; S[2,2] = sz
            T=trimesh.transformations.concatenate_matrices(T,S)
        else:
            # Uniform scaling
            s=float(scale); S=np.eye(4); S[:3,:3]*=s; T=trimesh.transformations.concatenate_matrices(T,S)
    
    mesh=mesh.copy(); mesh.apply_transform(T)
    
    # Apply material if specified
    material = transform.get("material")
    if material:
        mesh = _apply_material(mesh, material)
    
    return mesh
def _apply_mesh_operation(base_mesh, operation: Dict[str, Any], boolean_engine: str = "auto"):
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
                
                # Try different boolean engines
                success = False
                result = base_mesh
                
                # OpenSCAD backend if requested and available
                if boolean_engine in ["auto", "scad"]:
                    try:
                        if hasattr(trimesh, "interfaces") and hasattr(trimesh.interfaces, "scad"):
                            if op_type == "union":
                                result = trimesh.interfaces.scad.union([base_mesh, target_mesh])
                            elif op_type == "difference":
                                result = trimesh.interfaces.scad.difference(base_mesh, [target_mesh])
                            elif op_type == "intersection":
                                result = trimesh.interfaces.scad.intersection([base_mesh, target_mesh])
                            success = True
                    except Exception:
                        pass
                
                # Native trimesh boolean ops as fallback
                if not success and boolean_engine in ["auto", "native"]:
                    try:
                        if op_type == "union":
                            result = base_mesh.union(target_mesh)
                        elif op_type == "difference":
                            result = base_mesh.difference(target_mesh)
                        elif op_type == "intersection":
                            result = base_mesh.intersection(target_mesh)
                        success = True
                    except Exception:
                        pass
                
                return result if success else base_mesh
    
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

def _expand_instances(parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand parts with array and symmetry specifications"""
    expanded = []
    for p in parts:
        instances = [p]
        
        # Handle array duplication
        arr = p.get("array")
        if arr:
            instances = []
            n = max(1, int(arr.get("count", 1)))
            dx, dy, dz = arr.get("offset", [0, 0, 0])
            for i in range(n):
                q = dict(p)
                t = dict(q.get("transform", {}))
                off = t.get("translate", [0, 0, 0])
                t["translate"] = [off[0] + i*dx, off[1] + i*dy, off[2] + i*dz]
                q["transform"] = t
                # Remove array spec from instance to avoid re-expansion
                if "array" in q:
                    del q["array"]
                instances.append(q)
        
        # Handle symmetry mirroring
        sym = p.get("symmetry")
        if sym and sym.get("axis") in ("x", "y", "z"):
            axis = sym["axis"]
            idx = {"x": 0, "y": 1, "z": 2}[axis]
            mirrored_instances = []
            for q in instances:
                # Create mirrored copy
                m = dict(q)
                t = dict(m.get("transform", {}))
                tr = list(t.get("translate", [0, 0, 0]))
                tr[idx] *= -1
                t["translate"] = tr
                
                # Also mirror rotation if present
                rot = t.get("rotate_deg")
                if rot:
                    rot = list(rot)
                    # Mirror rotation around the symmetry axis
                    if axis == "x":
                        rot[1] *= -1  # Mirror Y rotation
                        rot[2] *= -1  # Mirror Z rotation
                    elif axis == "y":
                        rot[0] *= -1  # Mirror X rotation
                        rot[2] *= -1  # Mirror Z rotation
                    elif axis == "z":
                        rot[0] *= -1  # Mirror X rotation
                        rot[1] *= -1  # Mirror Y rotation
                    t["rotate_deg"] = rot
                
                m["transform"] = t
                # Remove symmetry spec to avoid re-mirroring
                if "symmetry" in m:
                    del m["symmetry"]
                mirrored_instances.append(m)
            
            instances.extend(mirrored_instances)
        
        expanded.extend(instances)
    
    return expanded

def _apply_post_optimization(mesh, quality: str = "standard"):
    """Apply post-processing optimization with optional external libraries"""
    if quality == "draft":
        return mesh  # Skip optimization for draft quality
    
    # Try open3d mesh simplification
    if quality == "high":
        try:
            import open3d as o3d
            # Convert to open3d format
            vertices = mesh.vertices.astype(np.float64)
            faces = mesh.faces.astype(np.int32)
            
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
            
            # Apply filters
            o3d_mesh.remove_degenerate_triangles()
            o3d_mesh.remove_duplicated_triangles()
            o3d_mesh.remove_duplicated_vertices()
            o3d_mesh.remove_non_manifold_edges()
            
            # Convert back to trimesh
            new_vertices = np.asarray(o3d_mesh.vertices)
            new_faces = np.asarray(o3d_mesh.triangles)
            mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, visual=mesh.visual)
            
        except ImportError:
            pass  # open3d not available
        except Exception:
            pass  # open3d processing failed
    
    # Try pymeshlab cleanup if available
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        
        # Create pymeshlab mesh
        ml_mesh = pymeshlab.Mesh(
            vertex_matrix=mesh.vertices.astype(np.float64),
            face_matrix=mesh.faces.astype(np.int32)
        )
        ms.add_mesh(ml_mesh)
        
        # Apply cleanup filters based on quality
        if quality in ["standard", "high"]:
            ms.meshing_remove_duplicate_faces()
            ms.meshing_remove_duplicate_vertices()
            ms.meshing_remove_unreferenced_vertices()
            
        if quality == "high":
            ms.meshing_close_holes(maxholesize=30)
            ms.meshing_repair_non_manifold_edges()
            
        # Convert back to trimesh
        cleaned = ms.current_mesh()
        mesh = trimesh.Trimesh(
            vertices=cleaned.vertex_matrix(),
            faces=cleaned.face_matrix(),
            visual=mesh.visual
        )
        
    except ImportError:
        pass  # pymeshlab not available
    except Exception:
        pass  # pymeshlab processing failed
    
    return mesh

def _apply_quality_settings(spec: Dict[str, Any], quality: str) -> Dict[str, Any]:
    """Apply quality-based parameter adjustments"""
    if quality == "draft":
        # Reduce resolution for faster generation
        if spec.get("primitive") == "cylinder":
            spec["segments"] = min(spec.get("segments", 24), 12)
        elif spec.get("primitive") == "sphere":
            spec["subdivisions"] = min(spec.get("subdivisions", 3), 2)
        elif spec.get("primitive") == "torus":
            spec["segments"] = min(spec.get("segments", 24), 12)
            spec["minor_segments"] = min(spec.get("minor_segments", 12), 8)
        elif spec.get("primitive") == "helix":
            spec["segments"] = min(spec.get("segments", 64), 32)
            
    elif quality == "high":
        # Increase resolution for better quality
        if spec.get("primitive") == "cylinder":
            spec["segments"] = max(spec.get("segments", 24), 32)
        elif spec.get("primitive") == "sphere":
            spec["subdivisions"] = max(spec.get("subdivisions", 3), 4)
        elif spec.get("primitive") == "torus":
            spec["segments"] = max(spec.get("segments", 24), 32)
            spec["minor_segments"] = max(spec.get("minor_segments", 12), 16)
        elif spec.get("primitive") == "helix":
            spec["segments"] = max(spec.get("segments", 64), 96)
    
    return spec

def _build_mesh_from_parts(parts: List[Dict[str, Any]], quality: str = "standard", boolean_engine: str = "auto"):
    meshes=[]
    for p in parts:
        if p.get("mode")=="primitive":
            # Apply quality settings to primitive specs
            p_with_quality = _apply_quality_settings(dict(p), quality)
            m=_primitive_mesh(p_with_quality); 
            if m is None: continue
            m=_apply_transform(m, p.get("transform", {}))
            
            # Apply any operations
            operations = p.get("operations", [])
            for op in operations:
                m = _apply_mesh_operation(m, op, boolean_engine)
            
            meshes.append(m)
            
        elif p.get("mode")=="custom":
            # Handle custom mesh types (like textured trunks)
            custom_type = p.get("type")
            if custom_type == "trunk":
                vertices = np.array(p.get("vertices", []))
                faces = np.array(p.get("faces", []))
                if len(vertices) > 0 and len(faces) > 0:
                    m = trimesh.Trimesh(vertices=vertices, faces=faces)
                    # Apply material if specified
                    material = p.get("material", {})
                    if material:
                        m = _apply_material(m, material)
                    meshes.append(m)
            # Add more custom types as needed
        
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
                    "optimize":{"type":"boolean","default":True,"description":"Apply mesh optimization (remove duplicates, fix normals)"},
                    "material":{"type":"object","description":"Global PBR material for GLB/GLTF (baseColor, metallic, roughness)"},
                    "boolean_engine":{"type":"string","enum":["auto","scad","native"],"default":"auto","description":"Boolean operation engine selection"},
                    "units":{"type":"string","enum":["m","cm","mm","in","ft"],"default":"m","description":"Output units"},
                    "scale_to_height":{"type":"number","description":"Scale model to specific height in chosen units"},
                    "scale_to_width":{"type":"number","description":"Scale model to specific width in chosen units"},
                    "scale_to_length":{"type":"number","description":"Scale model to specific length in chosen units"},
                    "quality":{"type":"string","enum":["draft","standard","high"],"default":"standard","description":"Mesh quality level"}
                },
                "required":[]
            }
        }
    async def execute(self, mode: str="auto", prompt: Optional[str]=None, parts: Optional[List[Dict[str,Any]]]=None,
                      output_path: Optional[str]=None, format: Optional[str]=None, seed: Optional[int]=None, 
                      optimize: bool=True, material: Optional[Dict[str,Any]]=None, boolean_engine: str="auto",
                      units: str="m", scale_to_height: Optional[float]=None, scale_to_width: Optional[float]=None,
                      scale_to_length: Optional[float]=None, quality: str="standard", **_):
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
            # Expand arrays and symmetry before building
            design = _expand_instances(design)
            mesh=_build_mesh_from_parts(design, quality, boolean_engine)
            
            # Apply global colors from prompt if specified and in auto mode
            if mode == "auto" and text:
                colors = _extract_colors(text)
                if colors and "color" in colors:
                    mesh = _apply_material(mesh, {"color": colors["color"]})
            
            # Apply global PBR material for GLB/GLTF formats
            if material and fmt in ("glb", "gltf"):
                rgba = material.get("baseColor", [1, 1, 1, 1])
                metallic = float(material.get("metallic", 0.0))
                roughness = float(material.get("roughness", 0.9))
                try:
                    if hasattr(trimesh.visual, "material") and hasattr(trimesh.visual.material, "PBRMaterial"):
                        mesh.visual.material = trimesh.visual.material.PBRMaterial(
                            baseColorFactor=rgba,
                            metallicFactor=metallic,
                            roughnessFactor=roughness
                        )
                except Exception:
                    pass  # Fall back to vertex colors if PBR fails
            
            # Apply dimensional scaling
            if scale_to_height or scale_to_width or scale_to_length:
                bounds = mesh.bounds
                size = bounds[1] - bounds[0]
                scale_factor = 1.0
                
                if scale_to_height and size[1] > 1e-9:
                    scale_factor = float(scale_to_height) / float(size[1])
                elif scale_to_width and size[0] > 1e-9:
                    scale_factor = float(scale_to_width) / float(size[0])
                elif scale_to_length and size[2] > 1e-9:
                    scale_factor = float(scale_to_length) / float(size[2])
                
                if scale_factor != 1.0:
                    mesh.apply_scale(scale_factor)
                
            # Apply additional optimizations if requested
            if optimize:
                extra_opts = []
                if mode == "auto" and text:
                    if _keyword(text, "smooth"): extra_opts.append("smooth")
                    if _keyword(text, "clean", "fix"): extra_opts.extend(["fill_holes", "remove_duplicates"])
                if extra_opts:
                    mesh = _optimize_mesh(mesh, extra_opts)
                
                # Apply post-export optimization if available
                mesh = _apply_post_optimization(mesh, quality)
                
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

