
from __future__ import annotations

import struct
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh

try:
    # Optional helper for WoW formats (M2/WMO)
    # Search env var or vendored third_party path before plain import
    import os, sys
    p = os.getenv("PYWOWLIB_PATH")
    if p and p not in sys.path:
        sys.path.insert(0, p)
    else:
        # try vendored third_party/pywowlib
        here = Path(__file__).resolve()
        vendored = here.parents[1] / "third_party" / "pywowlib"
        if vendored.is_dir() and str(vendored) not in sys.path:
            sys.path.insert(0, str(vendored))
    from pywowlib import m2_file, wmo_file  # type: ignore
    _PYWOWLIB = True
except Exception:
    _PYWOWLIB = False

# Local base class
try:
    from .base import Tool, _blocked  # type: ignore
except Exception:
    # fallback for standalone testing
    class Tool:
        def __init__(self, name: str, description: str):
            self.name = name
            self.description = description

        def to_openai_function(self) -> Dict[str, Any]:
            return {}

        def is_available(self) -> bool:
            return True

        async def execute(self, **kwargs) -> Dict[str, Any]:
            raise NotImplementedError

logger = logging.getLogger(__name__)


# -----------------------------
# Utility: write OBJ/GLB safely
# -----------------------------

def _ensure_outdir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _export_mesh(mesh: trimesh.Trimesh, output_path: Path) -> Dict[str, Any]:
    _ensure_outdir(output_path)
    ext = output_path.suffix.lower()
    if ext == ".obj":
        mesh.export(output_path)
    elif ext in (".glb", ".gltf"):
        # export to glb for compactness
        mesh.export(output_path, file_type="glb" if ext == ".glb" else "gltf")
    else:
        raise ValueError(f"Unsupported output extension: {ext}")
    return {
        "vertices": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0])
    }


# -----------------------------------
# Minimal M3 (The War Within) parser
# -----------------------------------

@dataclass
class _Chunk:
    fourcc: str
    size: int
    prop_a: int
    prop_b: int
    start: int   # start of payload in file
    end: int     # end of payload


class M3Parser:
    """
    Very small subset M3 reader (geometry only). Based on wowdev.wiki:
    - Chunked format (16 byte header), e.g. VPOS, VNML, VUV0, VINX, VGEO, VSTR.
    - Geosets define index/vertex ranges; we combine across geosets.
    Limitations: ignores materials, bones, instances/shaders, LOD selection.
    """

    def __init__(self, data: bytes):
        self.data = data
        self.stream = io.BytesIO(data)
        self.chunks: Dict[str, List[_Chunk]] = {}
        self._scan_chunks()

    def _scan_chunks(self):
        s = self.stream
        s.seek(0, io.SEEK_END)
        total = s.tell()
        s.seek(0)

        while s.tell() + 16 <= total:
            hdr = s.read(16)
            if len(hdr) < 16:
                break
            fourcc, size, prop_a, prop_b = struct.unpack("<4sIII", hdr)
            fourcc = fourcc.decode("ascii", errors="replace")
            payload_start = s.tell()
            payload_end = payload_start + size
            if payload_end > total:
                # broken file
                break
            chunk = _Chunk(fourcc=fourcc, size=size, prop_a=prop_a, prop_b=prop_b,
                           start=payload_start, end=payload_end)
            self.chunks.setdefault(fourcc, []).append(chunk)
            s.seek(payload_end)

    def _read_vertex_buffer(self, chunk: _Chunk, fmt: str) -> np.ndarray:
        """Read a flat buffer according to wowdev's simple formats: 3F32, 2F32, 4F32, 2F16, 4F16, 4U8N, U10N, 1U16/1U32, 4U16."""
        s = self.stream
        s.seek(chunk.start)
        buf = s.read(chunk.size)
        view = memoryview(buf)

        def _from_half(v: np.ndarray) -> np.ndarray:
            # numpy supports float16 natively
            return v.astype(np.float32)

        if fmt == "3F32":
            arr = np.frombuffer(view, dtype="<f4")
            return arr.reshape((-1, 3))
        if fmt == "2F32":
            arr = np.frombuffer(view, dtype="<f4")
            return arr.reshape((-1, 2))
        if fmt == "4F32":
            arr = np.frombuffer(view, dtype="<f4")
            return arr.reshape((-1, 4))
        if fmt == "2F16":
            arr = np.frombuffer(view, dtype="<f2")
            return _from_half(arr).reshape((-1, 2))
        if fmt == "4F16":
            arr = np.frombuffer(view, dtype="<f2")
            return _from_half(arr).reshape((-1, 4))
        if fmt == "4U8N":
            arr = np.frombuffer(view, dtype=np.uint8).astype(np.float32).reshape((-1, 4))
            return arr / 255.0
        if fmt == "1U16":
            arr = np.frombuffer(view, dtype="<u2")
            return arr.reshape((-1, 1))
        if fmt == "4U16":
            arr = np.frombuffer(view, dtype="<u2").reshape((-1, 4))
            return arr.astype(np.float32)
        if fmt == "1U32":
            arr = np.frombuffer(view, dtype="<u4")
            return arr.reshape((-1, 1))
        if fmt == "U10N":
            # packed 10:10:10 + 2bits padding
            raw = np.frombuffer(view, dtype="<u4")
            x = (raw & 0x3FF).astype(np.float32) / 1023.0
            y = ((raw >> 10) & 0x3FF).astype(np.float32) / 1023.0
            z = ((raw >> 20) & 0x3FF).astype(np.float32) / 1023.0
            return np.stack([x, y, z], axis=1)
        # fallback: raw bytes
        return np.frombuffer(view, dtype=np.uint8)

    def extract_mesh(self, lod: int | None = None) -> trimesh.Trimesh:
        # Get buffers
        def _first_fmt(ch_name: str, default_fmt: str | None = None) -> Tuple[Optional[_Chunk], Optional[str]]:
            chunks = self.chunks.get(ch_name, [])
            if not chunks:
                return None, None
            ch = chunks[0]
            fmt_bytes = struct.pack("<I", ch.prop_a)[:4]
            try:
                # prop_a is used as 4-char code like "3F32"
                fmt = ch.prop_a.to_bytes(4, "little").decode("ascii", errors="ignore")
                if not fmt.strip() and default_fmt:
                    fmt = default_fmt
            except Exception:
                fmt = default_fmt
            # Some files store the ascii chars directly in prop_a; ensure printable
            fmt = fmt if fmt and all(32 <= ord(c) <= 126 for c in fmt) else (default_fmt or "3F32")
            return ch, fmt

        pos_chunk, pos_fmt = _first_fmt("VPOS", "3F32")
        nrm_chunk, nrm_fmt = _first_fmt("VNML", "3F32")
        uv0_chunk, uv0_fmt = _first_fmt("VUV0", "2F32")
        idx_chunk, idx_fmt = _first_fmt("VINX", "1U16")

        if pos_chunk is None or idx_chunk is None:
            raise ValueError("M3 missing required geometry chunks (VPOS and/or VINX)")

        positions = self._read_vertex_buffer(pos_chunk, pos_fmt)
        normals = self._read_vertex_buffer(nrm_chunk, nrm_fmt) if nrm_chunk else None
        uvs = self._read_vertex_buffer(uv0_chunk, uv0_fmt) if uv0_chunk else None

        indices_buf = self._read_vertex_buffer(idx_chunk, idx_fmt).ravel().astype(np.int64)
        if idx_fmt == "1U16":
            # ok
            pass
        elif idx_fmt == "1U32":
            pass
        else:
            # Best effort: treat as 32-bit
            indices_buf = np.frombuffer(self.stream.getbuffer()[idx_chunk.start:idx_chunk.end], dtype="<u4").astype(np.int64)

        # Geosets define slices
        geosets = self.chunks.get("VGEO", [])
        s = self.stream
        tris_list = []
        vert_list = []
        offset = 0

        if geosets:
            # read geoset table
            g = geosets[0]
            s.seek(g.start)
            count = g.prop_a  # geosetCount
            # Each geoset is 0x24 bytes as per wiki
            geos = []
            for i in range(count):
                entry = s.read(0x24)
                if len(entry) < 0x24:
                    break
                unk0, nameStart, nameCount, iStart, iCount, vStart, vCount, unk1, unk2 = struct.unpack("<9I", entry)
                geos.append((iStart, iCount, vStart, vCount))
            # Combine all geosets (optionally only LOD0 later)
            all_faces = []
            all_verts = []
            vert_offset = 0
            for (iStart, iCount, vStart, vCount) in geos:
                # gather vertices slice
                v_slice = positions[vStart:vStart+vCount]
                all_verts.append(v_slice)
                # indices are relative to vertexStart
                inds = indices_buf[iStart:iStart+iCount] - vStart + vert_offset
                all_faces.append(inds.reshape((-1, 3)))
                vert_offset += v_slice.shape[0]
            V = np.vstack(all_verts) if all_verts else positions
            F = np.vstack(all_faces) if all_faces else indices_buf.reshape((-1, 3))
        else:
            # No geosets table? assume whole buffer is a single mesh
            V = positions
            F = indices_buf.reshape((-1, 3))

        # Basic attribute packaging
        mesh = trimesh.Trimesh(vertices=V, faces=F, process=True)
        if uvs is not None and len(uvs) >= len(V):
            # Attach as primary UVs
            mesh.visual.uv = uvs[:len(V)]
        if normals is not None and len(normals) >= len(V):
            mesh.vertex_normals = normals[:len(V)]
        return mesh


# ---------------------------------------
# Tool: WoWModelConverter (M2/M3/WMO->OBJ)
# ---------------------------------------

class WoWModelConverterTool(Tool):
    def __init__(self):
        super().__init__("wow_model_converter", "Convert WoW M2/M3/WMO to OBJ/GLB")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Path to a .m2/.m3/.wmo file or a directory containing them"
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["auto", "m2", "m3", "wmo_root", "wmo_group"],
                        "description": "Force a specific parser; default auto",
                        "default": "auto"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Destination .obj/.glb file OR output directory when input_path is a folder"
                    },
                    "bundle_dir": {
                        "type": "string",
                        "description": "If set, write a JSON manifest with all exported assets here"
                    },
                    "lod": {
                        "type": "integer",
                        "description": "Desired LOD (where applicable)",
                        "default": 0
                    }
                },
                "required": ["input_path", "output_path"]
            }
        }

    def is_available(self) -> bool:
        # trimesh + numpy are required; pywowlib is optional for M2/WMO
        try:
            _ = np.zeros((1,1))
            _ = trimesh.creation.box()
            return True
        except Exception:
            return False

    async def execute(self, input_path: str, output_path: str, kind: str = "auto", lod: int = 0, bundle_dir: str | None = None, **_) -> Dict[str, Any]:
        try:
            in_path = Path(input_path).resolve()
            out_path = Path(output_path).resolve()
            if not in_path.exists():
                return {"success": False, "error": f"Input path not found: {input_path}"}
            # Validate output path based on input type
            if in_path.is_dir():
                # Directory input: output should be a directory
                if out_path.exists() and out_path.is_file():
                    return {"success": False, "error": "When input_path is a directory, output_path must be a directory."}
                # Continue with batch processing
            else:
                # Single file input: output must have proper extension
                if not out_path.suffix.lower() in (".obj", ".glb", ".gltf"):
                    return {"success": False, "error": "output_path must end with .obj, .glb or .gltf"}

            if kind == "auto":
                kind = self._infer_kind(in_path)

            
            manifest_entries = []
            if in_path.is_dir():
                # Determine output directory
                out_dir = out_path if (not out_path.suffix) else out_path.parent
                out_dir.mkdir(parents=True, exist_ok=True)
                supported_exts = {".m2": "m2", ".m3": "m3", ".wmo": "wmo_root"}
                for f in sorted(in_path.rglob("*")):
                    if f.suffix.lower() not in supported_exts:
                        continue
                    k = supported_exts[f.suffix.lower()]
                    try:
                        if k == "m3":
                            mesh = self._convert_m3(f, lod)
                        elif k == "m2":
                            if not _PYWOWLIB:
                                continue
                            mesh = self._convert_m2(f, lod)
                        else:
                            if not _PYWOWLIB:
                                continue
                            # Guess root vs group by name
                            is_root = "_" not in f.stem
                            mesh = self._convert_wmo(f, is_root=is_root)
                        out_file = out_dir / (f.stem + ".glb")
                        stats = _export_mesh(mesh, out_file)
                        manifest_entries.append({
                            "input": str(f),
                            "kind": k if k != "wmo_root" else ("wmo_group" if "_" in f.stem else "wmo_root"),
                            "output": str(out_file),
                            "mesh_stats": stats,
                            "coord_sys": "Z_UP",
                            "scale_meters": 1.0
                        })
                    except Exception as e:
                        manifest_entries.append({
                            "input": str(f),
                            "kind": k,
                            "error": str(e)
                        })
                # Write bundle manifest if requested
                if bundle_dir:
                    bdir = Path(bundle_dir).expanduser().resolve()
                    bdir.mkdir(parents=True, exist_ok=True)
                    (bdir / "models_manifest.json").write_text(json.dumps({
                        "type": "models_bundle",
                        "root": str(in_path),
                        "output_dir": str(out_dir),
                        "entries": manifest_entries
                    }, indent=2), encoding="utf-8")
                return {"success": True, "batch": True, "count": len(manifest_entries),
                        "output_dir": str(out_dir),
                        "bundle_manifest": str(Path(bundle_dir)/"models_manifest.json") if bundle_dir else None}

            if kind == "m3":
                mesh = self._convert_m3(in_path, lod)
            elif kind == "m2":
                if not _PYWOWLIB:
                    return {"success": False, "error": "pywowlib not available; install with 'pip install pywowlib' to read M2/WMO."}
                mesh = self._convert_m2(in_path, lod)
            elif kind in ("wmo_root", "wmo_group"):
                if not _PYWOWLIB:
                    return {"success": False, "error": "pywowlib not available; install with 'pip install pywowlib' to read WMO."}
                mesh = self._convert_wmo(in_path, is_root=(kind=="wmo_root"))
            else:
                return {"success": False, "error": f"Unknown kind: {kind}"}

            stats = _export_mesh(mesh, out_path)
            return {
                "success": True,
                "kind": kind,
                "input": str(in_path),
                "output": str(out_path),
                "mesh_stats": stats
            }

        except Exception as e:
            logger.exception("WoWModelConverter error")
            return {"success": False, "error": str(e)}

    # -------- helpers ----------

    def _infer_kind(self, path: Path) -> str:
        ext = path.suffix.lower()
        name = path.name.lower()
        if ext == ".m3":
            return "m3"
        if ext == ".m2":
            return "m2"
        if ext == ".wmo":
            # group files are NAME_###.wmo; root is NAME.wmo
            return "wmo_group" if "_" in name[:-4] else "wmo_root"
        # directories: try to guess
        for p in path.glob("*.m3"):
            return "m3"
        for p in path.glob("*.m2"):
            return "m2"
        for p in path.glob("*.wmo"):
            return "wmo_root"
        return "m3"  # default

    def _convert_m3(self, file_path: Path, lod: int) -> trimesh.Trimesh:
        data = file_path.read_bytes()
        parser = M3Parser(data)
        mesh = parser.extract_mesh(lod=lod)
        return mesh

    def _convert_m2(self, file_path: Path, lod: int) -> trimesh.Trimesh:
        """
        Best-effort M2 -> mesh using pywowlib. We pick the first skin profile.
        """
        mf = m2_file.M2File(str(file_path))  # type: ignore
        mf.read()
        # Get main vertices (pos, normal, uv) from M2; indices from skin profile 0
        vertices = np.array([[v.pos.x, v.pos.y, v.pos.z] for v in mf.vertices], dtype=np.float32)
        # WoW uses Z-up; convert to Y-up if desired (here we keep as-is)
        # UVs: two sets in M2; we take the first
        uvs = np.array([[v.tex_coords[0].x, v.tex_coords[0].y] for v in mf.vertices], dtype=np.float32)

        # Choose a skin (LOD/view); 0 if exists
        if mf.skins:
            skin = mf.skins[min(lod, len(mf.skins)-1)]
            indices = np.array(skin.indices, dtype=np.int64)
            faces = indices.reshape((-1, 3))
        else:
            # Fallback: try triangles directly
            idx = np.array(mf.triangles, dtype=np.int64)
            faces = idx.reshape((-1, 3))

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        if uvs.shape[0] >= vertices.shape[0]:
            mesh.visual.uv = uvs[:vertices.shape[0]]
        return mesh

    def _convert_wmo(self, path: Path, is_root: bool) -> trimesh.Trimesh:
        """
        Combine all groups into a single mesh for the root file; or read one group.
        """
        if is_root:
            wmo = wmo_file.WMOFile(str(path))  # type: ignore
            wmo.read()
            group_paths = [Path(g.filepath) if getattr(g, "filepath", "") else None for g in wmo.groups]
            group_paths = [p if p and p.exists() else path.with_name(f"{path.stem}_{i:03d}.wmo") for i, p in enumerate(group_paths)]
            meshes = []
            for gp in group_paths:
                if gp.exists():
                    meshes.append(self._read_wmo_group(gp))
            if not meshes:
                raise ValueError("No WMO groups found")
            return trimesh.util.concatenate(meshes)
        else:
            return self._read_wmo_group(path)

    def _read_wmo_group(self, group_path: Path) -> trimesh.Trimesh:
        grp = wmo_file.WMOGroupFile(str(group_path))  # type: ignore
        grp.read()
        # Extract vertices & triangles
        vertices = np.array([[v.x, v.y, v.z] for v in grp.vertices], dtype=np.float32)
        indices = np.array(grp.indices, dtype=np.int64)
        faces = indices.reshape((-1, 3))
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        # UVs per vertex if provided
        if hasattr(grp, "tex_coords") and grp.tex_coords:
            try:
                uvs = np.array([[t.x, t.y] for t in grp.tex_coords], dtype=np.float32)
                if len(uvs) >= len(vertices):
                    mesh.visual.uv = uvs[:len(vertices)]
            except Exception:
                pass
        return mesh
