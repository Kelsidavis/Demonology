# demonology/tools/mpq_extractor.py
from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Tool, _confine, _ok_path

# Optional Python backends (best-effort imports; tool still works without them)
try:
    import pylibmpq  # type: ignore
except Exception:
    pylibmpq = None  # type: ignore

class MPQExtractorTool(Tool):
    """
    Safe MPQ listing/extraction with multiple backends:
      1) Python backend via pylibmpq (if installed)
      2) External CLI 'mpq' (if allowed & present)
    """

    def __init__(self):
        super().__init__("mpq_extractor", "List/extract World of Warcraft MPQ archives.")

    def is_available(self) -> bool:
        """Check if MPQ extraction backends are available."""
        return self._backend_can_python() or self._backend_can_cli()

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "mpq_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of .MPQ files or directories containing them. If empty, auto-discovers MPQ files in common locations.",
                        "default": []
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["list", "extract_all"],
                        "default": "list"
                    },
                    "dest_dir": {
                        "type": "string",
                        "description": "Destination directory for extraction (extract_all only)",
                        "default": "extracted"
                    },
                    "overwrite": {
                        "type": "boolean",
                        "default": False
                    },
                    "backend": {
                        "type": "string",
                        "enum": ["auto", "python", "cli"],
                        "default": "auto"
                    },
                    "parallel": {
                        "type": "boolean",
                        "description": "Extract multiple MPQ files in parallel (CLI backend only)",
                        "default": False
                    },
                    "quiet": {
                        "type": "boolean",
                        "description": "Reduce output verbosity for faster extraction",
                        "default": True
                    }
                },
                "required": []
            }
        }

    def _discover_inputs(self, mpq_paths: List[str]) -> List[Path]:
        found: List[Path] = []
        
        # If no paths provided, auto-discover MPQ files
        if not mpq_paths:
            print("🔍 Auto-discovering MPQ files in common locations...")
            search_paths = [
                Path.cwd(),  # Current directory
                Path.home() / "Desktop",  # Desktop
                Path("/home") / Path.cwd().parts[2] / "Desktop" if len(Path.cwd().parts) > 2 else Path.home() / "Desktop",
                Path.cwd() / "Data",  # WoW Data directory
                Path.cwd() / "data",
                Path.cwd() / "mpqs",
                Path.cwd() / "MPQs"
            ]
            
            for search_path in search_paths:
                try:
                    if search_path.exists() and search_path.is_dir():
                        # Search for MPQ files (case-insensitive)
                        mpq_files = list(search_path.glob("**/*.MPQ")) + list(search_path.glob("**/*.mpq"))
                        found.extend([x for x in mpq_files if x.is_file()])
                        
                        # Limit to first 50 files to avoid overwhelming
                        if len(found) > 50:
                            found = found[:50]
                            break
                except (PermissionError, OSError):
                    continue
            
            if found:
                print(f"🎯 Auto-discovered {len(found)} MPQ files")
            else:
                print("❌ No MPQ files found in auto-discovery")
        else:
            # Use provided paths
            for raw in mpq_paths:
                p = _confine(Path(raw))
                if p.is_dir():
                    found.extend([x for x in p.glob("**/*.MPQ") if x.is_file()])
                    found.extend([x for x in p.glob("**/*.mpq") if x.is_file()])
                elif p.is_file() and p.suffix.lower() == ".mpq":
                    found.append(p)
        
        return found

    def _backend_can_python(self) -> bool:
        return pylibmpq is not None

    def _backend_can_cli(self) -> bool:
        # Respect security whitelist (handled by base; here we only detect)
        return shutil.which("mpq") is not None

    async def _list_python(self, path: Path) -> List[str]:
        # pylibmpq doesn’t standardize a directory listing across all forks;
        # some builds expose .files / iter-like. We try common patterns.
        # If unavailable, we gracefully degrade to CLI.
        try:
            archive = pylibmpq.Archive(str(path))  # type: ignore
        except Exception as e:
            raise RuntimeError(f"pylibmpq open failed ({e})")

        names: List[str] = []
        # Try best-known attributes
        for attr in ("files", "namelist", "filelist"):
            files = getattr(archive, attr, None)
            if files:
                try:
                    for f in files:
                        name = f if isinstance(f, str) else getattr(f, "filename", None)
                        if name:
                            names.append(name)
                    return sorted(set(names))
                except Exception:
                    pass

        # Fallback: try numeric iteration (some forks expose __len__ and __getitem__)
        try:
            count = len(archive)  # type: ignore
            for i in range(count):
                try:
                    names.append(archive[i].filename)  # type: ignore
                except Exception:
                    pass
            return sorted(set([n for n in names if n]))
        except Exception:
            raise RuntimeError("pylibmpq cannot enumerate files on this build")

    async def _extract_all_python(self, path: Path, dest_dir: Path, overwrite: bool) -> Dict[str, Any]:
        # Minimal, defensive extractor; not all pylibmpq forks expose unified API.
        archive = pylibmpq.Archive(str(path))  # type: ignore
        extracted = 0
        errors: List[str] = []

        # Try to obtain names via the same helper
        try:
            names = await self._list_python(path)
        except Exception as e:
            return {"success": False, "error": f"Python backend listing failed: {e}"}

        for name in names:
            try:
                # create output path
                out_path = _confine(dest_dir / name)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if out_path.exists() and not overwrite:
                    continue

                # read bytes (APIs vary: some provide read_file / extractfile / getmember)
                data = None
                for cand in ("read_file", "extractfile", "open", "getmember"):
                    fn = getattr(archive, cand, None)
                    if fn:
                        try:
                            obj = fn(name)
                            if hasattr(obj, "read"):
                                data = obj.read()
                            elif isinstance(obj, (bytes, bytearray)):
                                data = bytes(obj)
                            else:
                                # Maybe returns index; try archive[index].read()
                                pass
                            break
                        except Exception:
                            continue
                if data is None:
                    # Final hail mary: some forks expose archive[name]
                    try:
                        item = archive[name]  # type: ignore
                        data = item.read() if hasattr(item, "read") else None
                    except Exception:
                        pass

                if not data:
                    errors.append(f"skip {name}: unreadable (backend limitation)")
                    continue

                with open(out_path, "wb") as f:
                    f.write(data)
                extracted += 1
            except Exception as e:
                errors.append(f"{name}: {e}")

        return {
            "success": True,
            "backend": "python",
            "extracted": extracted,
            "skipped": len(names) - extracted,
            "errors": errors[:20],
        }

    async def _run_cli(self, args: List[str], quiet: bool = True) -> Dict[str, Any]:
        # Use DEVNULL for stdout to reduce output processing overhead during extraction when quiet=True
        use_devnull = quiet and any("extract" in arg for arg in args)
        
        proc = await asyncio.create_subprocess_exec(
            *args, 
            stdout=asyncio.subprocess.DEVNULL if use_devnull else asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        if not use_devnull:
            stdout, stderr = await proc.communicate()
            return {
                "code": proc.returncode,
                "stdout": stdout.decode("utf-8", "replace"),
                "stderr": stderr.decode("utf-8", "replace"),
            }
        else:
            # For quiet extract operations, only capture stderr
            _, stderr = await proc.communicate()
            return {
                "code": proc.returncode,
                "stdout": "",
                "stderr": stderr.decode("utf-8", "replace"),
            }

    async def _list_cli(self, path: Path, quiet: bool = False) -> List[str]:
        # The `mpq` CLI syntax: mpq list <archive>
        cmd = ["mpq", "list", str(path)]
        res = await self._run_cli(cmd, quiet=quiet)
        if res["code"] == 0 and res["stdout"].strip():
            lines = [ln.strip() for ln in res["stdout"].splitlines() if ln.strip()]
            return lines
        raise RuntimeError(f"CLI 'mpq list' failed; stderr: {res['stderr']}")

    async def _extract_all_cli(self, path: Path, dest_dir: Path, overwrite: bool, quiet: bool = True) -> Dict[str, Any]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # The `mpq` CLI syntax: mpq extract <archive> -o <output_dir> -k (keep folder structure)
        cmd = ["mpq", "extract", str(path), "-o", str(dest_dir), "-k"]
        
        res = await self._run_cli(cmd, quiet=quiet)
        if res["code"] == 0:
            # Count extracted files by scanning the output directory instead of parsing stdout
            try:
                extracted_count = sum(1 for _ in dest_dir.rglob("*") if _.is_file() and not _.name.startswith("("))
            except Exception:
                extracted_count = "unknown"
            
            return {
                "success": True, 
                "backend": "cli", 
                "extracted_files": extracted_count,
                "stderr": res["stderr"][-500:] if res["stderr"] else ""
            }
        
        return {"success": False, "error": f"CLI extraction failed: {res['stderr'] or 'Unknown error'}"}

    async def execute(
        self,
        mpq_paths: List[str] = None,
        operation: str = "list",
        dest_dir: str = "extracted",
        overwrite: bool = False,
        backend: str = "auto",
        parallel: bool = False,
        quiet: bool = True,
        **_: Any
    ) -> Dict[str, Any]:
        # Handle None input (auto-discovery mode)
        if mpq_paths is None:
            mpq_paths = []
        # Resolve candidates
        paths = self._discover_inputs(mpq_paths)
        if not paths:
            return {"success": False, "error": "No MPQ files found in provided paths."}

        # Confinement & outputs
        out_root = _confine(Path(dest_dir)) if dest_dir else _confine(Path("extracted"))
        results: Dict[str, Any] = {"success": True, "items": []}

        # Pick backend with smart fallback
        python_available = self._backend_can_python()
        cli_available = self._backend_can_cli()
        
        if backend == "python":
            if python_available:
                use_python, use_cli = True, False
            elif cli_available:
                # Fall back to CLI when Python is explicitly requested but not available
                use_python, use_cli = False, True
            else:
                use_python, use_cli = False, False
        elif backend == "cli":
            if cli_available:
                use_python, use_cli = False, True
            else:
                use_python, use_cli = False, False
        else:  # "auto"
            use_python = python_available
            use_cli = cli_available

        if not (use_python or use_cli):
            error_msg = "No MPQ backend available."
            hints = []
            if backend == "python":
                error_msg = "Python MPQ backend (pylibmpq) not available."
                hints.append("pip install pylibmpq  # if available for your platform")
                if cli_available:
                    hints.append("Consider using 'auto' or 'cli' backend instead")
            elif backend == "cli":
                error_msg = "CLI MPQ backend ('mpq' command) not available."
                hints.append("Install a CLI like 'mpq' (StormLib-based) and ensure it's on PATH")
                if python_available:
                    hints.append("Consider using 'auto' or 'python' backend instead")
            else:
                hints = [
                    "pip install pylibmpq  # if available for your platform",
                    "OR install a CLI like 'mpq' (StormLib-based) and ensure it's on PATH",
                ]
            
            return {
                "success": False,
                "error": error_msg,
                "hints": hints,
            }

        # Handle parallel extraction for CLI backend
        if operation == "extract_all" and parallel and use_cli and len(paths) > 1:
            tasks = []
            for p in paths:
                # Create separate output directories for each MPQ when extracting in parallel
                mpq_dest = out_root / p.stem
                tasks.append(self._extract_all_cli(p, mpq_dest, overwrite, quiet))
            
            try:
                parallel_results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, (p, result) in enumerate(zip(paths, parallel_results)):
                    if isinstance(result, Exception):
                        return {"success": False, "error": f"Parallel extraction failed for {p.name}: {result}"}
                    elif not result.get("success"):
                        return {"success": False, "error": f"Parallel extraction failed for {p.name}: {result.get('error')}"}
                    results["items"].append({"path": str(p), **result})
            except Exception as e:
                return {"success": False, "error": f"Parallel extraction error: {e}"}
        else:
            # Sequential processing (original logic)
            for p in paths:
                if operation == "list":
                    if use_python:
                        try:
                            names = await self._list_python(p)
                            results["items"].append({"path": str(p), "files": names})
                            continue
                        except Exception:
                            # fall through to CLI
                            pass
                    if use_cli:
                        try:
                            names = await self._list_cli(p, quiet)
                            results["items"].append({"path": str(p), "files": names})
                            continue
                        except Exception as e:
                            return {"success": False, "error": f"List failed for {p.name}: {e}"}
                    return {"success": False, "error": "No working backend could list files."}

                elif operation == "extract_all":
                    if use_python:
                        try:
                            r = await self._extract_all_python(p, out_root, overwrite)
                            if not r.get("success"):
                                raise RuntimeError(r.get("error", "python backend failed"))
                            results["items"].append({"path": str(p), **r})
                            continue
                        except Exception:
                            pass  # fall back to CLI

                    if use_cli:
                        r = await self._extract_all_cli(p, out_root, overwrite, quiet)
                        if not r.get("success"):
                            return {"success": False, "error": r.get("error", "CLI failed")}
                        results["items"].append({"path": str(p), **r})
                        continue

                    return {"success": False, "error": "No working backend could extract files."}
                else:
                    return {"success": False, "error": f"Unknown operation: {operation}"}

        results["output_dir"] = str(out_root)
        return results

