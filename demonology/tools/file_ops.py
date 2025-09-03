# demonology/tools/file_ops.py
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Tool, _safe_path

logger = logging.getLogger(__name__)


class FileOperationsTool(Tool):
    """
    Filesystem operations (fenced to SAFE_ROOT):
      - create_or_write_file(path, content="")
      - create_directory(path)
      - read(path)
      - list(path=".")
      - delete_file(path)
      - delete_directory(path, recursive: bool)
    """
    # Allow ANY extension (including .cpp, .c, .go, etc).
    # Set to a set like {'.txt', '.py'} if you want to re-enable filtering.
    _ALLOWED_EXTS: Optional[set[str]] = None

    def __init__(self, safe_root: Optional[Path] = None):
        super().__init__("file_operations", "Create/read/list/delete files and directories (SAFE_ROOT-fenced).")
        from .base import SAFE_ROOT
        self.safe_root = (safe_root or SAFE_ROOT).resolve()

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "One of: create_or_write_file, create_directory, read, list, delete_file, delete_directory"
                    },
                    "path": {"type": "string", "description": "Path relative to SAFE_ROOT"},
                    "content": {"type": "string", "description": "File content for create_or_write_file"},
                    "recursive": {"type": "boolean", "description": "Recursive delete for delete_directory"}
                },
                "required": ["operation"]
            }
        }

    async def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        try:
            op = (operation or "").strip().lower()
            if op == "create_or_write_file":
                return await self._create_or_write_file(kwargs.get("path"), kwargs.get("content", ""))
            if op == "create_directory":
                return await self._create_directory(kwargs.get("path"))
            if op == "read":
                return await self._read_file(kwargs.get("path"))
            if op == "list":
                return await self._list_directory(kwargs.get("path", "."))
            if op == "delete_file":
                return await self._delete_file(kwargs.get("path"))
            if op == "delete_directory":
                return await self._delete_directory(kwargs.get("path"), bool(kwargs.get("recursive", False)))
            return {"success": False, "error": f"Unknown operation: {operation}"}
        except Exception as e:
            logger.exception("FileOperationsTool error")
            return {"success": False, "error": str(e)}

    async def _create_or_write_file(self, path: Optional[str], content: str) -> Dict[str, Any]:
        if not path:
            return {"success": False, "error": "Missing 'path' for create_or_write_file"}
        p = _safe_path(path)
        if p.exists() and p.is_dir():
            return {"success": False, "error": f"Path is a directory: {p}"}
        # Extension filter disabled unless _ALLOWED_EXTS is a non-empty set
        if "." in p.name and self._ALLOWED_EXTS:
            ext = p.suffix.lower()
            if ext and ext not in self._ALLOWED_EXTS:
                return {"success": False, "error": f"Extension not allowed: {ext}"}
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content or "", encoding="utf-8")
        ok = p.exists() and p.is_file()
        size = p.stat().st_size if ok else 0
        return {
            "success": bool(ok),
            "operation": "create_or_write_file",
            "path": str(p),
            "size": size,
            "exists": ok,
            "absolute_path": str(p.resolve()),
        }

    async def _create_directory(self, path: Optional[str]) -> Dict[str, Any]:
        if not path:
            return {"success": False, "error": "Missing 'path' for create_directory"}
        d = _safe_path(path, want_dir=True)
        d.mkdir(parents=True, exist_ok=True)
        ok = d.exists() and d.is_dir()
        return {
            "success": bool(ok),
            "operation": "create_directory",
            "path": str(d),
            "exists": ok,
            "absolute_path": str(d.resolve()),
        }

    async def _read_file(self, path: Optional[str]) -> Dict[str, Any]:
        if not path:
            return {"success": False, "error": "Missing 'path' for read"}
        p = _safe_path(path)
        if not p.exists() or not p.is_file():
            return {"success": False, "error": f"File not found: {p}"}
        content = p.read_text(encoding="utf-8", errors="replace")
        return {"success": True, "operation": "read", "path": str(p), "content": content}

    async def _list_directory(self, path: Optional[str]) -> Dict[str, Any]:
        d = _safe_path(path or ".", want_dir=True)
        if not d.exists() or not d.is_dir():
            return {"success": False, "error": f"Directory not found: {d}"}
        items: List[Dict[str, Any]] = []
        for child in sorted(d.iterdir(), key=lambda x: x.name.lower()):
            try:
                items.append({
                    "name": child.name,
                    "is_dir": child.is_dir(),
                    "size": (child.stat().st_size if child.is_file() else None),
                })
            except Exception:
                items.append({"name": child.name, "is_dir": child.is_dir(), "size": None})
        return {"success": True, "operation": "list", "path": str(d), "items": items}

    async def _delete_file(self, path: Optional[str]) -> Dict[str, Any]:
        if not path:
            return {"success": False, "error": "Missing 'path' for delete_file"}
        p = _safe_path(path)
        if not p.exists():
            return {"success": False, "error": f"File not found: {p}"}
        if p.is_dir():
            return {"success": False, "error": f"Path is a directory (use delete_directory): {p}"}
        p.unlink()
        return {"success": True, "operation": "delete_file", "path": str(p), "deleted": True}

    async def _delete_directory(self, path: Optional[str], recursive: bool) -> Dict[str, Any]:
        if not path:
            return {"success": False, "error": "Missing 'path' for delete_directory"}
        d = _safe_path(path, want_dir=True)
        if not d.exists() or not d.is_dir():
            return {"success": False, "error": f"Directory not found: {d}"}
        if recursive:
            shutil.rmtree(d)
            return {"success": True, "operation": "delete_directory", "path": str(d), "recursive": True, "deleted": True}
        # non-recursive: only delete if empty
        if any(d.iterdir()):
            return {"success": False, "error": "Directory is not empty", "path": str(d)}
        d.rmdir()
        return {"success": True, "operation": "delete_directory", "path": str(d), "recursive": False, "deleted": True}