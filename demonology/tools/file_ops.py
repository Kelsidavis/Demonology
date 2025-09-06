# demonology/tools/file_ops.py
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, List

from .base import Tool

class FileOperationsTool(Tool):
    """
    Safeguarded file operations within a workspace root.

    Root is resolved from env DEMONOLOGY_ROOT or current working directory.
    - Blocks symlink traversal/escapes.
    - Caps read size (default 5 MiB), configurable per call via max_bytes.
    - Refuses obviously dangerous deletions.
    """

    _ALLOWED_EXTS: Optional[set[str]] = None
    _ROOT: Path = Path(os.environ.get("DEMONOLOGY_ROOT", os.getcwd())).resolve()
    _MAX_READ_BYTES_DEFAULT = 5 * 1024 * 1024  # 5 MiB

    def __init__(self):
        super().__init__(
            "file_operations",
            "Create/read/write/delete/copy/move files & directories within the safeguarded workspace.",
        )

    # ---------- Schema ----------
    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "create_or_write_file",
                            "append",
                            "append_file",
                            "read",
                            "read_file",
                            "read_bytes",
                            "delete_file",
                            "create_directory",
                            "delete_directory",
                            "list_dir",
                            "move",
                            "copy",
                            "info",
                            "get_file_info",
                        ],
                        "description": "File operation to perform",
                    },
                    "path": {"type": "string", "description": "Target file/directory path"},
                    "content": {
                        "type": "string",
                        "description": "Content to write/append (UTF-8 text)",
                        "default": "",
                    },
                    "source": {"type": "string", "description": "Source path for copy/move"},
                    "destination": {"type": "string", "description": "Destination path for copy/move"},
                    "recursive": {
                        "type": "boolean",
                        "description": "Recursive delete for delete_directory",
                        "default": False,
                    },
                    "max_bytes": {
                        "type": "integer",
                        "description": "Max bytes to read for read/read_bytes (default 5 MiB)",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Optional filename pattern for list_dir (glob, e.g. *.txt)",
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": "Include dotfiles in list_dir results",
                        "default": False,
                    },
                },
                "required": ["operation"],
            },
        }

    # ---------- Dispatcher ----------
    async def execute(self, **kwargs) -> Dict[str, Any]:
        op = (kwargs.get("operation") or "").lower().strip()

        try:
            if op in ("create_or_write_file", "write", "write_file"):
                return await self._create_or_write_file(kwargs.get("path"), kwargs.get("content", ""))

            if op in ("append", "append_file"):
                return await self._append_file(kwargs.get("path"), kwargs.get("content", ""))

            if op in ("read", "read_file"):
                return await self._read_text(kwargs.get("path"), kwargs.get("max_bytes"))

            if op == "read_bytes":
                return await self._read_bytes(kwargs.get("path"), kwargs.get("max_bytes"))

            if op == "delete_file":
                return await self._delete_file(kwargs.get("path"))

            if op == "create_directory":
                return await self._create_directory(kwargs.get("path"))

            if op == "delete_directory":
                return await self._delete_directory(kwargs.get("path"), kwargs.get("recursive", False))

            if op == "list_dir":
                return await self._list_dir(
                    kwargs.get("path"),
                    kwargs.get("pattern"),
                    bool(kwargs.get("include_hidden", False)),
                )

            if op == "move":
                return await self._move(kwargs.get("source"), kwargs.get("destination"))

            if op == "copy":
                return await self._copy(kwargs.get("source"), kwargs.get("destination"))

            if op in ("info", "get_file_info"):
                return await self._get_file_info(kwargs.get("path"))

            return {"success": False, "error": f"Unknown operation: {op}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    # ---------- Helpers ----------
    def _resolve_in_root(self, p: str) -> Path:
        if not p:
            raise ValueError("Empty path")
        if "\x00" in p:
            raise ValueError("Invalid path")
        cand = Path(p)
        # Normalize relative to root (and normalize absolute too)
        cand = (self._ROOT / cand).resolve() if not cand.is_absolute() else cand.resolve()

        # Enforce root containment
        try:
            cand.relative_to(self._ROOT)
        except Exception:
            raise PermissionError(f"Path escapes workspace root: {cand}")

        # Block symlink traversal for the target and all parents
        # We try best-effort lstat; missing parents are tolerated (on create).
        for parent in [cand] + list(cand.parents):
            try:
                if parent.is_symlink():
                    raise PermissionError(f"Symlinked path not allowed: {parent}")
            except FileNotFoundError:
                pass
        return cand

    def _dangerous_dir(self, p: Path) -> bool:
        # Obvious dangerous directories to refuse deletion
        dangerous: List[Path] = [
            self._ROOT,
            self._ROOT.parent,
            Path("/"),
            Path.home(),
            Path("/etc"),
            Path("/proc"),
            Path("/sys"),
            Path("/dev"),
            Path("/run"),
            Path("/var"),
            Path("/boot"),
            Path("/lib"),
            Path("/lib64"),
        ]
        return any(p == d for d in dangerous)

    # ---------- Operations ----------
    async def _create_or_write_file(self, path: Optional[str], content: str) -> Dict[str, Any]:
        if not path:
            return {"success": False, "error": "Missing 'path' for create_or_write_file"}
        p = self._resolve_in_root(path)
        if p.exists() and p.is_dir():
            return {"success": False, "error": f"Path is a directory: {p}"}
        p.parent.mkdir(parents=True, exist_ok=True)
        # Text write (UTF-8)
        with p.open("w", encoding="utf-8") as f:
            f.write(content or "")
        size = p.stat().st_size
        return {"success": True, "operation": "create_or_write_file", "path": str(p), "size": size}

    async def _append_file(self, path: Optional[str], content: str) -> Dict[str, Any]:
        if not path:
            return {"success": False, "error": "Missing 'path' for append"}
        p = self._resolve_in_root(path)
        if p.exists() and p.is_dir():
            return {"success": False, "error": f"Path is a directory: {p}"}
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(content or "")
        return {"success": True, "operation": "append", "path": str(p), "size": p.stat().st_size}

    async def _read_text(self, path: Optional[str], max_bytes: Optional[int]) -> Dict[str, Any]:
        if not path:
            return {"success": False, "error": "Missing 'path' for read"}
        p = self._resolve_in_root(path)
        if not p.exists() or not p.is_file():
            return {"success": False, "error": f"File not found: {p}"}
        limit = int(max_bytes) if (max_bytes and int(max_bytes) > 0) else self._MAX_READ_BYTES_DEFAULT
        data = p.open("rb").read(limit + 1)
        truncated = len(data) > limit
        if truncated:
            data = data[:limit]
        content = data.decode("utf-8", errors="replace")
        return {
            "success": True,
            "operation": "read",
            "path": str(p),
            "content": content,
            "truncated": truncated,
            "bytes": len(data),
        }

    async def _read_bytes(self, path: Optional[str], max_bytes: Optional[int]) -> Dict[str, Any]:
        if not path:
            return {"success": False, "error": "Missing 'path' for read_bytes"}
        p = self._resolve_in_root(path)
        if not p.exists() or not p.is_file():
            return {"success": False, "error": f"File not found: {p}"}
        limit = int(max_bytes) if (max_bytes and int(max_bytes) > 0) else self._MAX_READ_BYTES_DEFAULT
        data = p.open("rb").read(limit + 1)
        truncated = len(data) > limit
        if truncated:
            data = data[:limit]
        return {
            "success": True,
            "operation": "read_bytes",
            "path": str(p),
            "bytes": len(data),
            "truncated": truncated,
            "content_hex": data.hex(),
        }

    async def _delete_file(self, path: Optional[str]) -> Dict[str, Any]:
        if not path:
            return {"success": False, "error": "Missing 'path' for delete_file"}
        p = self._resolve_in_root(path)
        if not p.exists():
            return {"success": False, "error": f"File not found: {p}"}
        if p.is_dir():
            return {"success": False, "error": f"Path is a directory (use delete_directory): {p}"}
        # refuse obviously dangerous file names at root (paranoid: e.g., workspace root marker)
        if p == self._ROOT:
            return {"success": False, "error": f"Refusing to delete dangerous path: {p}"}
        p.unlink(missing_ok=False)
        return {"success": True, "operation": "delete_file", "path": str(p), "deleted": True}

    async def _create_directory(self, path: Optional[str]) -> Dict[str, Any]:
        if not path:
            return {"success": False, "error": "Missing 'path' for create_directory"}
        d = self._resolve_in_root(path)
        d.mkdir(parents=True, exist_ok=True)
        return {"success": True, "operation": "create_directory", "path": str(d), "created": True}

    async def _delete_directory(self, path: Optional[str], recursive: bool) -> Dict[str, Any]:
        if not path:
            return {"success": False, "error": "Missing 'path' for delete_directory"}
        d = self._resolve_in_root(path)
        if not d.exists():
            return {"success": False, "error": f"Directory not found: {d}"}
        if not d.is_dir():
            return {"success": False, "error": f"Path is not a directory: {d}"}
        if self._dangerous_dir(d):
            return {"success": False, "error": f"Refusing to delete dangerous directory: {d}"}
        if recursive:
            shutil.rmtree(d)
            return {
                "success": True,
                "operation": "delete_directory",
                "path": str(d),
                "recursive": True,
                "deleted": True,
            }
        else:
            try:
                d.rmdir()
            except OSError as e:
                return {"success": False, "error": f"Directory not empty (use recursive): {d} ({e})"}
            return {"success": True, "operation": "delete_directory", "path": str(d), "deleted": True}

    async def _list_dir(self, path: Optional[str], pattern: Optional[str], include_hidden: bool) -> Dict[str, Any]:
        if not path:
            return {"success": False, "error": "Missing 'path' for list_dir"}
        d = self._resolve_in_root(path)
        if not d.exists() or not d.is_dir():
            return {"success": False, "error": f"Directory not found: {d}"}
        # glob if pattern supplied, else iterate children
        entries = list(d.glob(pattern)) if pattern else list(d.iterdir())
        items = []
        for e in entries:
            name = e.name
            if not include_hidden and name.startswith("."):
                continue
            try:
                st = e.stat()
                items.append(
                    {
                        "name": name,
                        "path": str(e),
                        "is_dir": e.is_dir(),
                        "is_file": e.is_file(),
                        "size": st.st_size if e.is_file() else None,
                        "modified_time": st.st_mtime,
                    }
                )
            except FileNotFoundError:
                # entry might vanish between list and stat
                continue
        return {"success": True, "operation": "list_dir", "path": str(d), "items": items, "count": len(items)}

    async def _move(self, source: Optional[str], destination: Optional[str]) -> Dict[str, Any]:
        if not source or not destination:
            return {"success": False, "error": "Missing 'source' or 'destination' for move"}
        src = self._resolve_in_root(source)
        dst = self._resolve_in_root(destination)
        if not src.exists():
            return {"success": False, "error": f"Source not found: {src}"}
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return {"success": True, "operation": "move", "source": str(src), "destination": str(dst)}

    async def _copy(self, source: Optional[str], destination: Optional[str]) -> Dict[str, Any]:
        if not source or not destination:
            return {"success": False, "error": "Missing 'source' or 'destination' for copy"}
        src = self._resolve_in_root(source)
        dst = self._resolve_in_root(destination)
        if not src.exists():
            return {"success": False, "error": f"Source not found: {src}"}
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        return {"success": True, "operation": "copy", "source": str(src), "destination": str(dst)}

    async def _get_file_info(self, path: Optional[str]) -> Dict[str, Any]:
        if not path:
            return {"success": False, "error": "Missing 'path' for info"}
        p = self._resolve_in_root(path)
        if not p.exists():
            return {"success": False, "error": f"Path not found: {p}"}
        st = p.stat()
        return {
            "success": True,
            "operation": "info",
            "info": {
                "name": p.name,
                "path": str(p),
                "is_dir": p.is_dir(),
                "is_file": p.is_file(),
                "size": st.st_size if p.is_file() else None,
                "modified_time": st.st_mtime,
                "mode_octal": oct(st.st_mode)[-4:],
            },
        }

