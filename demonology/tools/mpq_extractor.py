
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

try:
    from .base import Tool  # type: ignore
except Exception:
    class Tool:
        def __init__(self, name: str, description: str):
            self.name, self.description = name, description

logger = logging.getLogger(__name__)

# Optional python backend(s)
_BACKENDS = {}

try:
    import pympq  # type: ignore
    _BACKENDS['pympq'] = 'pympq'
except Exception:
    pass

def _which(*names: str) -> Optional[str]:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None

def _has_cli() -> Optional[str]:
    # Common CLIs from mpq-tools packages
    return _which("mpq")

async def _run(cmd: List[str]) -> Tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out, err = await proc.communicate()
    return proc.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")

class MPQExtractorTool(Tool):
    """
    Work with Blizzard MPQ archives:
    - List files inside (.list)
    - Extract all contents (or best-effort patterns via CLI)

    Backends:
      * Python: pympq (if installed)
      * CLI: 'mpq' from mpq-tools (if found in PATH)
    """

    def __init__(self):
        super().__init__("mpq_extractor", "List and extract Blizzard MPQ archives (pympq/CLI).")

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
                        "description": "List of .mpq paths (or a directory containing them)"
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["list", "extract_all"],
                        "default": "list",
                        "description": "List archive contents or extract all files"
                    },
                    "dest_dir": {
                        "type": "string",
                        "description": "Destination directory for extraction (required for extract_all)"
                    },
                    "overwrite": {
                        "type": "boolean",
                        "default": True,
                        "description": "Overwrite existing files on extraction"
                    }
                },
                "required": ["mpq_paths"]
            }
        }

    def is_available(self) -> bool:
        return bool(_BACKENDS) or bool(_has_cli())

    async def execute(self,
                      mpq_paths: List[str],
                      operation: str = "list",
                      dest_dir: Optional[str] = None,
                      overwrite: bool = True,
                      **_) -> Dict[str, Any]:
        archives = self._collect_archives(mpq_paths)
        if not archives:
            return {"success": False, "error": "No MPQ files found in provided paths."}

        py_backend = 'pympq' if 'pympq' in _BACKENDS else None
        cli = _has_cli()

        if operation == "list":
            listing = {}
            for arc in archives:
                if py_backend:
                    try:
                        import pympq  # type: ignore
                        with pympq.MPQArchive(str(arc)) as a:
                            listing[str(arc)] = a.namelist()
                        continue
                    except Exception as e:
                        logger.debug("pympq list failed for %s: %s", arc, e, exc_info=True)
                if cli:
                    code, out, err = await _run([cli, "l", str(arc)])
                    if code == 0:
                        files = self._parse_cli_list(out)
                        listing[str(arc)] = files
                    else:
                        listing[str(arc)] = {"error": err.strip() or out.strip()}
                else:
                    listing[str(arc)] = {"error": "No backend available (install 'pympq' or 'mpq-tools')."}
            return {"success": True, "archives": [str(x) for x in archives], "listing": listing}

        if operation == "extract_all":
            if not dest_dir:
                return {"success": False, "error": "dest_dir is required for extract_all"}
            dest = Path(dest_dir).expanduser().resolve()
            dest.mkdir(parents=True, exist_ok=True)
            extracted = []
            errors = []
            for arc in archives:
                if py_backend:
                    try:
                        import pympq  # type: ignore
                        with pympq.MPQArchive(str(arc)) as a:
                            for name in a.namelist():
                                out_path = dest / name
                                out_path.parent.mkdir(parents=True, exist_ok=True)
                                if out_path.exists() and not overwrite:
                                    continue
                                with a.open(name) as src, open(out_path, "wb") as dst:
                                    dst.write(src.read())
                        extracted.append(str(arc))
                        continue
                    except Exception as e:
                        logger.debug("pympq extract failed for %s: %s", arc, e, exc_info=True)
                if cli:
                    # mpq x <archive> -o <dest>
                    code, out, err = await _run([cli, "x", str(arc), "-o", str(dest)])
                    if code == 0:
                        extracted.append(str(arc))
                    else:
                        errors.append({"archive": str(arc), "error": err.strip() or out.strip()})
                else:
                    errors.append({"archive": str(arc), "error": "No backend available (install 'pympq' or 'mpq-tools')."})
            return {"success": len(extracted) > 0 and not errors, "extracted": extracted, "errors": errors}

        return {"success": False, "error": f"Unknown operation: {operation}"}

    def _collect_archives(self, inputs: List[str]) -> List[Path]:
        out: List[Path] = []
        for p in inputs:
            pp = Path(p).expanduser().resolve()
            if pp.is_file() and pp.suffix.lower() == ".mpq":
                out.append(pp)
            elif pp.is_dir():
                out.extend(sorted(pp.rglob("*.mpq")))
        return out

    def _parse_cli_list(self, out: str) -> List[str]:
        files: List[str] = []
        for line in out.splitlines():
            line = line.strip()
            if not line or line.startswith("Archive:") or line.startswith("Listing"):
                continue
            # mpq-tools usually prints file names one per line
            files.append(line)
        return files
