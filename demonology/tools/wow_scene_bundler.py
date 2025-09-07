
from __future__ import annotations
import io, struct, json, logging, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from .base import Tool  # type: ignore
except Exception:
    class Tool:
        def __init__(self, name: str, description: str):
            self.name, self.description = name, description

# ---- ADT placement readers (best-effort classic layout) ----

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
        yield magic, size, data

def _parse_zero_strings(blob: bytes) -> List[str]:
    # NUL-separated C strings, often with trailing NUL
    parts = blob.split(b'\x00')
    out = []
    for p in parts:
        if not p:
            continue
        try:
            out.append(p.decode('utf-8', errors='ignore'))
        except Exception:
            out.append(p.decode('latin-1', errors='ignore'))
    return out

def _read_placements_from_adt(adt_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Best-effort extraction of placement records from ADT:
    - MMDX (model filenames) + MMID (uint32 indices) + MDDF (doodads: M2 placements)
    - MWMO (wmo filenames)   + MWID (uint32 indices) + MODF (wmo placements)
    We infer struct sizes; if unexpected, skip with a warning.
    Returns: {"m2": [...], "wmo": [...]}
    """
    res = {"m2": [], "wmo": []}
    try:
        with adt_path.open('rb') as f:
            mmdx = []; mwmo = []
            mmid = []; mwid = []
            mddf_bytes = b''; modf_bytes = b''
            for magic, size, data in _read_chunks(f):
                if magic == 'MMDX':
                    mmdx = _parse_zero_strings(data)
                elif magic == 'MMID':
                    mmid = list(struct.unpack('<' + 'I'*(size//4), data))
                elif magic == 'MDDF':
                    mddf_bytes = data
                elif magic == 'MWMO':
                    mwmo = _parse_zero_strings(data)
                elif magic == 'MWID':
                    mwid = list(struct.unpack('<' + 'I'*(size//4), data))
                elif magic == 'MODF':
                    modf_bytes = data
            # MDDF entries (classic 36 bytes): nameId, uniqueId, posX/Y/Z, rotX/Y/Z, scale(uint16), flags(uint16)
            if mddf_bytes:
                if len(mddf_bytes) % 36 == 0:
                    n = len(mddf_bytes)//36
                    for i in range(n):
                        (nameId, uniqueId,
                         posX, posY, posZ,
                         rotX, rotY, rotZ,
                         scale_u16, flags_u16) = struct.unpack_from('<IIffffffHH', mddf_bytes, i*36)
                        scale = float(scale_u16) / 1024.0 if scale_u16 else 1.0
                        path = None
                        if nameId < len(mmid):
                            off = mmid[nameId]
                            # find which string matches this offset (approximate by index)
                            # mmid points to byte offsets into MMDX blob; our parsed list lost offsets
                            # use fallback: if nameId < len(mmdx), map directly
                            if nameId < len(mmdx):
                                path = mmdx[nameId]
                        elif nameId < len(mmdx):
                            path = mmdx[nameId]
                        res["m2"].append({
                            "nameId": int(nameId),
                            "uniqueId": int(uniqueId),
                            "path": path,
                            "position": [float(posX), float(posY), float(posZ)],
                            "rotation_deg": [float(rotX), float(rotY), float(rotZ)],
                            "scale": float(scale),
                            "flags": int(flags_u16)
                        })
                else:
                    logger.warning("MDDF size not multiple of 36; unknown variant for %s", adt_path.name)
            # MODF entries (classic 64 bytes). We'll parse a minimal subset.
            if modf_bytes:
                if len(modf_bytes) % 64 == 0:
                    n = len(modf_bytes)//64
                    for i in range(n):
                        (nameId, uniqueId,
                         posX, posY, posZ,
                         rotX, rotY, rotZ,
                         ext0x, ext0y, ext0z,
                         ext1x, ext1y, ext1z,
                         flags_u16, doodadSet_u16, nameSet_u16, pad_u16) = struct.unpack_from('<IIffffff fff fff HHHH', modf_bytes, i*64)
                        path = None
                        if nameId < len(mwid):
                            off = mwid[nameId]
                            if nameId < len(mwmo):
                                path = mwmo[nameId]
                        elif nameId < len(mwmo):
                            path = mwmo[nameId]
                        res["wmo"].append({
                            "nameId": int(nameId),
                            "uniqueId": int(uniqueId),
                            "path": path,
                            "position": [float(posX), float(posY), float(posZ)],
                            "rotation_deg": [float(rotX), float(rotY), float(rotZ)],
                            "flags": int(flags_u16),
                            "sets": {"doodad": int(doodadSet_u16), "name": int(nameSet_u16)}
                        })
                else:
                    logger.warning("MODF size not multiple of 64; unknown variant for %s", adt_path.name)
    except Exception as e:
        logger.warning("Failed to read placements from %s: %s", adt_path, e)
    return res

def _normpath(p: str) -> str:
    return re.sub(r'[\\/]+', '/', p.strip()).lower()

def _build_model_lookup(models_manifest: Dict[str, Any]) -> Dict[str, str]:
    """Return mapping from original source path basename (lower) -> exported file path."""
    mapping = {}
    for e in models_manifest.get("entries", []):
        src = _normpath(e.get("input",""))
        out = e.get("output")
        if not src or not out: 
            continue
        mapping[Path(src).name.lower()] = out
        mapping[_normpath(src)] = out
    return mapping

class WoWSceneBundlerTool(Tool):
    """
    Build a top-level scene manifest that merges:
    - Terrain world manifest (tiles & transforms)
    - Models manifest (exported M2/M3/WMO assets)
    Optionally, scan ADT files for MDDF/MODF placements and create nodes referencing exported assets.
    """
    def __init__(self):
        super().__init__("wow_scene_bundler", "Merge terrain+models into a re-assemblable scene manifest and optional placements.")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "terrain_manifest": {"type": "string", "description": "Path to *_world_manifest.json from wow_world_converter"},
                    "models_manifest": {"type": "string", "description": "Path to models_manifest.json from wow_model_converter (optional)"},
                    "maps_root": {"type": "string", "description": "Root to extracted WoW data (optional; enables placement scan)"},
                    "output_path": {"type": "string", "description": "Where to write scene_manifest.json", "default": "exports/scene_manifest.json"},
                    "include_placements": {"type": "boolean", "description": "Scan ADT files for MDDF/MODF and include placement nodes", "default": False}
                },
                "required": ["terrain_manifest","output_path"]
            }
        }

    def is_available(self) -> bool:
        return True

    async def execute(self,
                      terrain_manifest: str,
                      output_path: str,
                      models_manifest: Optional[str] = None,
                      maps_root: Optional[str] = None,
                      include_placements: bool = False,
                      **_) -> Dict[str, Any]:
        tpath = Path(terrain_manifest).expanduser().resolve()
        if not tpath.exists():
            return {"success": False, "error": f"Terrain manifest not found: {terrain_manifest}"}
        terrain = json.loads(tpath.read_text(encoding="utf-8"))

        models = {}
        if models_manifest:
            mpath = Path(models_manifest).expanduser().resolve()
            if not mpath.exists():
                return {"success": False, "error": f"Models manifest not found: {models_manifest}"}
            models = json.loads(mpath.read_text(encoding="utf-8"))

        # Scene header
        scene = {
            "type": "wow_scene",
            "coordinate_system": "Z_UP",
            "units": {"name": "yard", "meters_per_unit": 0.9144},
            "terrain": [],             # list of nodes with file + transform
            "models_library": [],      # distinct assets
            "placements": [],          # nodes referencing library ids
            "sources": {
                "terrain_manifest": str(tpath),
                "models_manifest": str(Path(models_manifest).resolve()) if models_manifest else None
            }
        }

        # Terrain nodes
        entries = terrain.get("entries", [])
        for e in entries:
            files = e.get("files", {})
            mesh = files.get("glb") or files.get("obj") or files.get("height_raw") or files.get("height_png")
            if not mesh:
                continue
            scene["terrain"].append({
                "tile": e.get("tile"),
                "file": mesh,
                "translation": e.get("origin_yards", [0.0,0.0,0.0]),
                "scale": [1.0, 1.0, 1.0],
                "rotation_deg": [0.0, 0.0, 0.0],
                "size_yards": e.get("world_size_yards", terrain.get("tile_world_size_yards"))
            })

        # Models library (flat list)
        model_id_map: Dict[str, str] = {}
        if models:
            lib = []
            for idx, me in enumerate(models.get("entries", []), start=1):
                asset_id = f"asset_{idx:05d}"
                lib.append({
                    "id": asset_id,
                    "kind": me.get("kind"),
                    "source": me.get("output"),
                    "original": me.get("input"),
                    "mesh_stats": me.get("mesh_stats"),
                    "coord_sys": me.get("coord_sys", "Z_UP"),
                    "scale_meters": me.get("scale_meters", 1.0)
                })
                # Map by basename for placement lookup
                if me.get("input"):
                    model_id_map[Path(me["input"]).name.lower()] = asset_id
            scene["models_library"] = lib

        # Optional placements from ADT files
        placed_count = 0
        errors = []
        if include_placements and maps_root:
            maps_root_p = Path(maps_root).expanduser().resolve()
            map_name = terrain.get("map")
            if map_name:
                # Build lookup from model path -> exported file
                export_lookup = {}
                if models:
                    for me in models.get("entries", []):
                        src = me.get("input"); out = me.get("output")
                        if not src or not out: continue
                        export_lookup[Path(src).name.lower()] = {"id": None, "out": out}
                # For each tile in terrain manifest, open ADT and read placement chunks
                for e in entries:
                    tv = e.get("tile")
                    if not (isinstance(tv, list) and len(tv)==2):
                        continue
                    tx, ty = tv
                    adt_path = maps_root_p / "World" / "Maps" / map_name / f"{map_name}_{tx:02d}_{ty:02d}.adt"
                    if not adt_path.exists():
                        continue
                    plc = _read_placements_from_adt(adt_path)
                    def _add_node(kind: str, rec: Dict[str, Any]):
                        nonlocal placed_count
                        src_path = rec.get("path") or ""
                        base = Path(src_path).name.lower()
                        asset_id = model_id_map.get(base)
                        if not asset_id and base in export_lookup:
                            # we have an exported path but no library id; leave ref None, keep hint
                            pass
                        node = {
                            "ref": asset_id,
                            "kind": kind,
                            "source_hint": src_path,
                            "translation": rec.get("position", [0,0,0]),
                            "rotation_deg": rec.get("rotation_deg", [0,0,0]),
                            "scale": [rec.get("scale", 1.0)]*3 if kind=="m2" else [1.0,1.0,1.0],
                            "tile": [tx, ty]
                        }
                        scene["placements"].append(node)
                        placed_count += 1
                    for rec in plc.get("m2", []):
                        _add_node("m2", rec)
                    for rec in plc.get("wmo", []):
                        _add_node("wmo", rec)
            else:
                errors.append("Terrain manifest missing 'map' field; cannot infer ADT paths.")

        # Write scene manifest
        outp = Path(output_path).expanduser().resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(scene, indent=2), encoding='utf-8')

        return {
            "success": True,
            "output": str(outp),
            "terrain_nodes": len(scene["terrain"]),
            "model_assets": len(scene["models_library"]),
            "placements": placed_count,
            "errors": errors or None
        }
