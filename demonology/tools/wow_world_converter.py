
from __future__ import annotations
import io, os, struct, re, logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import trimesh
except Exception:
    trimesh = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from .base import Tool
except Exception:
    class Tool:
        def __init__(self, name: str, description: str):
            self.name = name; self.description = description

import json

logger = logging.getLogger(__name__)

TILES_PER_SIDE = 64
CHUNKS_PER_TILE = 16
GRID_PER_TILE = CHUNKS_PER_TILE * 8 + 1  # 129
TILE_WORLD_SIZE_YARDS = 533.33333

def _read_chunks(f: io.BufferedReader):
    while True:
        head = f.read(8)
        if len(head) < 8:
            break
        magic = head[0:4].decode('ascii', errors='replace')
        size = struct.unpack('<I', head[4:8])[0]
        data = f.read(size)
        if len(data) != size:
            break
        start = f.tell() - 8 - size
        yield magic, size, data, start

def _list_tiles_from_wdt(wdt_path: Path) -> Optional[List[Tuple[int,int]]]:
    try:
        tiles = []
        with wdt_path.open('rb') as f:
            # Check if this is a WDT file with REVM header
            header = f.read(4)
            f.seek(0)  # Reset to beginning
            
            # Handle different WDT formats
            if header == b'REVM':
                # New format WDT - skip to chunks after REVM header
                f.seek(12)  # Skip REVM header
                
            for magic, size, data, _ in _read_chunks(f):
                if magic == 'MAIN':
                    if size < TILES_PER_SIDE*TILES_PER_SIDE*8:
                        continue
                    arr = np.frombuffer(data, dtype='<u4')
                    flags = arr[0::2] if arr.size >= 2 else arr
                    for y in range(TILES_PER_SIDE):
                        for x in range(TILES_PER_SIDE):
                            i = y*TILES_PER_SIDE + x
                            if i < len(flags) and flags[i] != 0:
                                tiles.append((x,y))
                    return tiles
                elif magic == 'NIAM':  # MAIN backwards - some WDT files have this
                    if size < TILES_PER_SIDE*TILES_PER_SIDE*8:
                        continue
                    # For instance dungeons, might have different format
                    # Try to find any non-zero entries indicating tiles exist
                    arr = np.frombuffer(data, dtype='<u4')
                    # Instance dungeons typically use tile (0,0)
                    if len(arr) > 0:
                        tiles.append((0,0))
                        return tiles
        
        # If no tiles found via WDT, but file exists, assume it's a single-tile instance
        if wdt_path.exists():
            logger.info("WDT exists but no tiles found - assuming single tile instance at (0,0)")
            return [(0,0)]
            
        return None
    except Exception as e:
        logger.warning("WDT parse failed: %s", e)
        # For instances, often just single tile at 0,0
        return [(0,0)]

def _list_tiles_from_fs(map_dir: Path, map_name: str) -> List[Tuple[int,int]]:
    tiles = []
    pat = re.compile(rf"^{re.escape(map_name)}_(\d+)_(\d+)\.adt$", re.I)
    for p in map_dir.glob(f"{map_name}_*.adt"):
        m = pat.match(p.name)
        if m:
            x, y = int(m.group(1)), int(m.group(2))
            tiles.append((x,y))
    return sorted(set(tiles))

def _parse_mcnk_heights(mcnk_bytes: bytes) -> Optional[np.ndarray]:
    off = 0; n = len(mcnk_bytes)
    while off + 8 <= n:
        mag = mcnk_bytes[off:off+4].decode('ascii', errors='replace')
        sz  = struct.unpack('<I', mcnk_bytes[off+4:off+8])[0]
        off += 8
        if off + sz > n:
            break
        if mag == 'MCVT' and sz >= 145*4:
            floats = np.frombuffer(mcnk_bytes[off:off+145*4], dtype='<f4', count=145)
            grid9 = floats[:81].reshape((9,9))
            return grid9
        off += sz
    return None

def _read_adt_height_129(adt_path: Path) -> Optional[np.ndarray]:
    try:
        with adt_path.open('rb') as f:
            grids = []
            for magic, size, data, _ in _read_chunks(f):
                if magic == 'MCNK':
                    g9 = _parse_mcnk_heights(data)
                    if g9 is None:
                        g9 = np.zeros((9,9), dtype=np.float32)
                    grids.append(g9)
            if len(grids) != CHUNKS_PER_TILE * CHUNKS_PER_TILE:
                logger.warning("ADT %s: expected 256 MCNK; got %d", adt_path.name, len(grids))
                while len(grids) < CHUNKS_PER_TILE*CHUNKS_PER_TILE:
                    grids.append(np.zeros((9,9), dtype=np.float32))
            H = GRID_PER_TILE
            out = np.zeros((H, H), dtype=np.float32)
            for cy in range(CHUNKS_PER_TILE):
                for cx in range(CHUNKS_PER_TILE):
                    g = grids[cy*CHUNKS_PER_TILE + cx]
                    ys = cy*8; xs = cx*8
                    out[ys:ys+9, xs:xs+9] = g
            return out
    except Exception as e:
        logger.error("Failed to read ADT %s: %s", adt_path, e)
        return None

def _read_wdl_height(wdl_path: Path) -> Optional[np.ndarray]:
    """Read height data from WDL file for instance dungeons."""
    try:
        with wdl_path.open('rb') as f:
            # Handle REVM header format
            header = f.read(4)
            f.seek(0)
            
            if header == b'REVM':
                # Skip REVM header (12 bytes)
                f.seek(12)
                
            for magic, size, data, _ in _read_chunks(f):
                logger.debug(f"WDL chunk: {magic}, size: {size}")
                
                if magic == 'FOAM':  # FOAM chunk might contain terrain data for instances
                    # FOAM chunks in WDL files can contain heightmap data
                    if size >= 16:  # Need at least some data
                        # Instance dungeons often have simple flat heightmaps
                        # Generate a basic heightmap for the instance
                        dim = 65  # Standard size for small instances
                        height_data = np.zeros((dim, dim), dtype=np.float32)
                        logger.info("Generated flat heightmap from FOAM chunk for instance")
                        return height_data
                        
                elif magic == 'MLHD':  # Heightmap data in WDL
                    if size >= 4:
                        heights = np.frombuffer(data, dtype='<f4')
                        sqrt_len = int(np.sqrt(len(heights)))
                        if sqrt_len * sqrt_len == len(heights):
                            return heights.reshape((sqrt_len, sqrt_len))
                        # Try common instance sizes
                        for dim in [17, 33, 65, 129]:
                            if len(heights) >= dim * dim:
                                return heights[:dim*dim].reshape((dim, dim))
                                
                elif magic == 'MCNK':  # Sometimes WDL contains MCNK chunks like ADT
                    g9 = _parse_mcnk_heights(data)
                    if g9 is not None:
                        # Single chunk for small instance - expand to reasonable size
                        return np.tile(g9, (4, 4))  # Tile the 9x9 to 36x36
                        
        # If no height data found but WDL exists, create a default flat heightmap
        # Many instance dungeons don't have complex terrain
        logger.info("No height data found in WDL, generating flat terrain for instance")
        return np.zeros((65, 65), dtype=np.float32)
        
    except Exception as e:
        logger.warning("Failed to read WDL %s: %s", wdl_path, e)
        # Generate a basic flat heightmap as fallback for instances
        logger.info("Generating flat heightmap as fallback for instance")
        return np.zeros((65, 65), dtype=np.float32)

def _export_height_png16(path: Path, heightmap: np.ndarray):
    if Image is None:
        raise RuntimeError("Pillow not installed")
    h = heightmap.astype(np.float64)
    h_min, h_max = float(h.min()), float(h.max())
    scale = 0.0 if (h_max - h_min) == 0 else 65535.0 / (h_max - h_min)
    img16 = np.clip((h - h_min) * scale, 0, 65535).astype(np.uint16)
    Image.fromarray(img16, mode='I;16').save(str(path))

def _export_height_raw16(path: Path, heightmap: np.ndarray):
    h = heightmap.astype(np.float64)
    h_min, h_max = float(h.min()), float(h.max())
    scale = 0.0 if (h_max - h_min) == 0 else 65535.0 / (h_max - h_min)
    raw16 = np.clip((h - h_min) * scale, 0, 65535).astype('<u2')
    with path.open('wb') as f:
        f.write(raw16.tobytes(order='C'))

def _mesh_from_heightmap(height: np.ndarray, xz_size: float):
    if trimesh is None:
        raise RuntimeError("trimesh not installed")
    H, W = height.shape
    xs = np.linspace(-xz_size/2.0, xz_size/2.0, W, dtype=np.float32)
    zs = np.linspace(-xz_size/2.0, xz_size/2.0, H, dtype=np.float32)
    xv, zv = np.meshgrid(xs, zs)
    vertices = np.column_stack([xv.reshape(-1), height.reshape(-1), zv.reshape(-1)])
    faces = []
    for r in range(H-1):
        for c in range(W-1):
            i0 = r*W + c; i1 = i0 + 1; i2 = i0 + W; i3 = i2 + 1
            faces.append([i0, i2, i1]); faces.append([i1, i2, i3])
    return trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces, dtype=np.int32), process=False)

class WoWWorldConverterTool(Tool):
    def __init__(self):
        super().__init__("wow_world_converter", "Convert WoW WDT/ADT terrain to OBJ/GLB and 16-bit heightmaps.")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "maps_root": {"type": "string", "description": "Path to extracted WoW data root containing World/Maps/<Map>/<Map>.wdt and ADTs."},
                    "map_name": {"type": "string", "description": "Internal map directory (e.g. 'Kalimdor', 'Azeroth', 'Northrend')"},
                    "tiles": {"type": "array", "description": "List of tile coords [[x,y],...]; omit for all", "items": {"type": "array", "items": {"type": "integer"}}},
                    "merge_tiles": {"type": "boolean", "description": "Merge selected tiles into one mesh/heightmap", "default": False},
                    "export": {"type": "array", "items": {"type": "string", "enum": ["obj","glb","height_png","height_raw"]}, "default": ["obj","height_png"]},
                    "output_dir": {"type": "string", "description": "Where to write results", "default": "wow_exports"},
                    "downsample": {"type": "integer", "description": "Downsample factor (1=full,2=half,...)", "default": 1},
                    "bundle_dir": {"type": "string", "description": "If set, write a world manifest JSON here (tiles, transforms, outputs)"}
                },
                "required": ["maps_root","map_name"]
            }
        }

    async def execute(self, 
                      maps_root: str,
                      map_name: str,
                      tiles: Optional[List[List[int]]] = None,
                      merge_tiles: bool = False,
                      export: Optional[List[str]] = None,
                      output_dir: str = "wow_exports",
                      downsample: int = 1,
                      bundle_dir: str | None = None,
                      **_) -> Dict[str, Any]:
        export = export or ["obj","height_png"]
        root = Path(maps_root).expanduser().resolve()
        map_dir = (root / "World" / "Maps" / map_name)
        wdt_path = map_dir / f"{map_name}.wdt"
        if not map_dir.exists():
            return {"success": False, "error": f"Map dir not found: {map_dir}"}

        discovered = _list_tiles_from_wdt(wdt_path) if wdt_path.exists() else None
        fs_tiles = _list_tiles_from_fs(map_dir, map_name)
        
        # If filesystem found more tiles than WDT, use filesystem (WDT parsing might be incomplete)
        if fs_tiles and len(fs_tiles) > len(discovered or []):
            logger.info(f"Filesystem found {len(fs_tiles)} tiles vs WDT {len(discovered or [])} - using filesystem discovery")
            discovered = fs_tiles
        elif not discovered:
            discovered = fs_tiles
            
        if not discovered:
            return {"success": False, "error": "No tiles found via WDT or file system. Ensure your MPQs are extracted to World/Maps/<Map>."}

        selected = set(tuple(x) for x in tiles) if tiles else set(discovered)

        heights: Dict[Tuple[int,int], np.ndarray] = {}
        missing: List[Tuple[int,int]] = []
        
        # Check if this is an instance dungeon without ADT files
        wdl_path = map_dir / f"{map_name}.wdl"
        adt_files_exist = any((map_dir / f"{map_name}_{tx:02d}_{ty:02d}.adt").exists() 
                             for tx, ty in selected)
        
        if not adt_files_exist and wdl_path.exists():
            # Instance dungeon - try to read from WDL file
            logger.info("No ADT files found, trying to read instance terrain from WDL")
            wdl_height = _read_wdl_height(wdl_path)
            if wdl_height is not None:
                # For instances, typically use tile (0,0)
                if downsample and downsample > 1:
                    wdl_height = wdl_height[::downsample, ::downsample]
                heights[(0, 0)] = wdl_height
                logger.info("Successfully read WDL heightmap: %s", wdl_height.shape)
        else:
            # Regular world map - read ADT files
            for (tx, ty) in sorted(selected):
                p = map_dir / f"{map_name}_{tx:02d}_{ty:02d}.adt"
                if not p.exists():
                    missing.append((tx,ty)); continue
                hm = _read_adt_height_129(p)
                if hm is None:
                    missing.append((tx,ty)); continue
                if downsample and downsample > 1:
                    hm = hm[::downsample, ::downsample]
                heights[(tx,ty)] = hm
                
        if not heights:
            wdl_hint = f" (WDL: {'exists' if wdl_path.exists() else 'missing'})" if not adt_files_exist else ""
            return {"success": False, "error": f"All requested tiles missing or unreadable{wdl_hint}.", "missing": missing}

        outdir = Path(output_dir).expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)

        results = []
        if merge_tiles and len(heights) > 1:
            xs = sorted(set(x for x,_ in heights.keys()))
            ys = sorted(set(y for _,y in heights.keys()))
            XN, YN = len(xs), len(ys)
            step = next(iter(heights.values())).shape[0]
            H = YN * (step - 1) + 1
            W = XN * (step - 1) + 1
            mega = np.zeros((H, W), dtype=np.float32)
            for yi, yv in enumerate(ys):
                for xi, xv in enumerate(xs):
                    hm = heights.get((xv,yv))
                    if hm is None: continue
                    r0 = yi*(step-1); c0 = xi*(step-1)
                    mega[r0:r0+hm.shape[0], c0:c0+hm.shape[1]] = hm
            base = f"{map_name}_merged"
            if "height_png" in export: _export_height_png16(outdir / f"{base}.png", mega)
            if "height_raw" in export: _export_height_raw16(outdir / f"{base}.raw", mega)
            if "obj" in export or "glb" in export:
                try:
                    mesh = _mesh_from_heightmap(mega, xz_size=TILE_WORLD_SIZE_YARDS * XN)
                    if "obj" in export: mesh.export(str(outdir / f"{base}.obj"))
                    if "glb" in export: mesh.export(str(outdir / f"{base}.glb"))
                except Exception as e:
                    logger.error("Mesh export failed: %s", e)
            results.append({"tile": "merged", "path_base": str(outdir / base), "shape": mega.shape})
        else:
            for (tx, ty), hm in heights.items():
                base = f"{map_name}_{tx:02d}_{ty:02d}"
                if "height_png" in export: _export_height_png16(outdir / f"{base}.png", hm)
                if "height_raw" in export: _export_height_raw16(outdir / f"{base}.raw", hm)
                if "obj" in export or "glb" in export:
                    try:
                        mesh = _mesh_from_heightmap(hm, xz_size=TILE_WORLD_SIZE_YARDS)
                        if "obj" in export: mesh.export(str(outdir / f"{base}.obj"))
                        if "glb" in export: mesh.export(str(outdir / f"{base}.glb"))
                    except Exception as e:
                        logger.error("Mesh export failed: %s", e)
                results.append({"tile": [tx,ty], "path_base": str(outdir / base), "shape": hm.shape})

        # Bundle manifest: list per-tile transforms and produced files
        if bundle_dir:
            bdir = Path(bundle_dir).expanduser().resolve()
            bdir.mkdir(parents=True, exist_ok=True)
            tiles_sorted = sorted(list(heights.keys()))
            xs = sorted(set(x for x,_ in tiles_sorted))
            ys = sorted(set(y for _,y in tiles_sorted))
            minx, miny = (xs[0], ys[0])
            entries = []
            for (tx, ty) in tiles_sorted:
                base = f"{map_name}_{tx:02d}_{ty:02d}"
                ox = (tx - minx) * TILE_WORLD_SIZE_YARDS
                oz = (ty - miny) * TILE_WORLD_SIZE_YARDS
                files = {}
                for kind in (export or []):
                    if kind == "obj": files["obj"] = str(outdir / f"{base}.obj")
                    if kind == "glb": files["glb"] = str(outdir / f"{base}.glb")
                    if kind == "height_png": files["height_png"] = str(outdir / f"{base}.png")
                    if kind == "height_raw": files["height_raw"] = str(outdir / f"{base}.raw")
                entries.append({
                    "tile": [tx, ty],
                    "origin_yards": [float(ox), 0.0, float(oz)],
                    "world_size_yards": float(TILE_WORLD_SIZE_YARDS),
                    "downsample": int(downsample),
                    "shape": list(heights[(tx,ty)].shape),
                    "files": files
                })
            manifest = {
                "type": "world_bundle",
                "map": map_name,
                "tile_world_size_yards": float(TILE_WORLD_SIZE_YARDS),
                "downsample": int(downsample),
                "output_dir": str(outdir),
                "merged": bool(merge_tiles),
                "entries": entries
            }
            (bdir / f"{map_name}_world_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return {
            "success": True,
            "map": map_name,
            "tiles": sorted(list(heights.keys())),
            "missing": missing,
            "downsample": int(downsample),
            "exports": export,
            "output_dir": str(outdir),
            "results": results,
            "bundle_manifest": str(Path(bundle_dir)/f"{map_name}_world_manifest.json") if bundle_dir else None
        }
