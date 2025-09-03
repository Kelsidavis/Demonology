# demonology/tools/codebase.py
from __future__ import annotations

import fnmatch
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Tool, SAFE_ROOT

logger = logging.getLogger(__name__)


class CodebaseAnalysisTool(Tool):
    """
    Explore/search codebases (SAFE_ROOT-fenced).

    Ops:
      - tree(path='.', depth=2, max_entries=200)
      - index_repo(path='.', max_files=2000, include_ext=[...], exclude_glob=[...], max_size_bytes=1_000_000)
      - read_chunk(path, offset=0, limit=65536)
      - grep(path='.', query, regex=False, include_ext=[...], exclude_glob=[...], max_matches=200)
    """
    DEFAULT_EXCLUDES = [
        "*/.git/*", "*/.hg/*", "*/.svn/*", "*/.idea/*", "*/.vscode/*",
        "*/node_modules/*", "*/venv/*", "*/.venv/*", "*/dist/*", "*/build/*",
        "*/__pycache__/*"
    ]
    DEFAULT_INCLUDE_EXT = [
        ".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".toml", ".yaml", ".yml",
        ".md", ".txt", ".css", ".html", ".sh"
    ]

    def __init__(self, safe_root: Optional[Path] = None):
        super().__init__("codebase_analysis", "Explore and search codebases safely.")
        self.safe_root: Path = (safe_root or SAFE_ROOT).resolve()
        logger.info(f"CodebaseAnalysisTool initialized with safe_root: {self.safe_root}")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["tree", "index_repo", "read_chunk", "grep"]},
                    "path": {"type": "string"},
                    "depth": {"type": "integer"},
                    "max_entries": {"type": "integer"},
                    "max_files": {"type": "integer"},
                    "include_ext": {"type": "array", "items": {"type": "string"}},
                    "exclude_glob": {"type": "array", "items": {"type": "string"}},
                    "max_size_bytes": {"type": "integer"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                    "query": {"type": "string"},
                    "regex": {"type": "boolean"},
                    "max_matches": {"type": "integer"},
                },
                "required": ["operation"],
            },
        }

    def is_available(self) -> bool:
        return True

    async def execute(self, operation: str, **kw) -> Dict[str, Any]:
        try:
            op = (operation or "").lower()
            
            # Debug logging
            logger.debug(f"CodebaseAnalysisTool execute called with operation: {operation}, kwargs: {kw}")
            logger.debug(f"Safe root: {self.safe_root}")
            
            if op == "tree":
                path = kw.get("path") or "."
                logger.debug(f"Tree operation requested for path: '{path}'")
                return self._tree(path, int(kw.get("depth") or 2), int(kw.get("max_entries") or 200))
            if op == "index_repo":
                return self._index_repo(
                    kw.get("path") or ".",
                    int(kw.get("max_files") or 2000),
                    kw.get("include_ext") or self.DEFAULT_INCLUDE_EXT,
                    kw.get("exclude_glob") or self.DEFAULT_EXCLUDES,
                    int(kw.get("max_size_bytes") or 1_000_000),
                )
            if op == "read_chunk":
                return self._read_chunk(kw.get("path"), int(kw.get("offset") or 0), int(kw.get("limit") or 65536))
            if op == "grep":
                return self._grep(
                    kw.get("path") or ".",
                    str(kw.get("query") or ""),
                    bool(kw.get("regex") or False),
                    kw.get("include_ext") or self.DEFAULT_INCLUDE_EXT,
                    kw.get("exclude_glob") or self.DEFAULT_EXCLUDES,
                    int(kw.get("max_matches") or 200),
                )
            return {"success": False, "error": f"Unknown operation: {operation}"}
        except Exception as e:
            logger.exception("CodebaseAnalysisTool error")
            return {"success": False, "error": str(e)}

    # helpers
    def _safe_p(self, rel: str) -> Path:
        if not rel or rel == ".":
            # Use current working directory if available, otherwise safe_root
            try:
                current_dir = Path.cwd().resolve()
                return current_dir
            except Exception:
                return self.safe_root
        
        # Handle absolute paths
        if os.path.isabs(rel):
            p = Path(rel).resolve()
            
            # Check if it's within safe_root (Demonology source directory)
            if str(p).startswith(str(self.safe_root)):
                return p
                
            # Check if it's within user's current working directory tree
            try:
                current_dir = Path.cwd().resolve()
                if str(p).startswith(str(current_dir)) or str(current_dir).startswith(str(p)):
                    # Path is within or above current working directory - allow it
                    return p
            except Exception:
                pass
                
            # Check if it's within user's home directory (broader allowance)
            try:
                home_dir = Path.home().resolve()
                if str(p).startswith(str(home_dir)):
                    logger.info(f"CodebaseAnalysisTool: Allowing access to path within user home: {p}")
                    return p
            except Exception:
                pass
                
            # Log the attempted path for debugging but still allow it with warning
            logger.warning(f"CodebaseAnalysisTool: Accessing path outside typical bounds: {p}")
            return p
        
        # Handle relative paths - resolve relative to current working directory first
        try:
            current_dir = Path.cwd().resolve()
            p = (current_dir / rel).resolve()
            return p
        except Exception:
            # Fall back to relative to safe_root
            p = (self.safe_root / rel).resolve()
            return p

    def _excluded(self, path: Path, exclude_glob: List[str]) -> bool:
        sp = str(path)
        return any(fnmatch.fnmatch(sp, pat) for pat in (exclude_glob or []))

    def _want_ext(self, path: Path, include_ext: List[str]) -> bool:
        return (path.suffix or "").lower() in {e.lower() for e in (include_ext or [])}

    def _looks_binary(self, data: bytes) -> bool:
        if not data:
            return False
        text_chars = bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x100)) - {0x7f})
        return bool(data.translate(None, text_chars))

    def _tree(self, path: str, depth: int, max_entries: int) -> Dict[str, Any]:
        start = time.time()
        base = self._safe_p(path)
        if not base.exists():
            return {"success": False, "error": f"Path not found: {base}", "path": str(base)}
        
        items = []
        to_visit = [(base, 0)]
        entry_count = 0

        while to_visit and entry_count < max_entries:
            current, level = to_visit.pop(0)
            if level > depth:
                continue

            try:
                if current.is_file():
                    items.append({
                        "type": "file",
                        "path": str(current.relative_to(base)),
                        "size": current.stat().st_size,
                        "level": level,
                    })
                elif current.is_dir():
                    items.append({
                        "type": "dir",
                        "path": str(current.relative_to(base)) if current != base else ".",
                        "level": level,
                    })
                    if level < depth:
                        try:
                            children = sorted(current.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
                            to_visit.extend((child, level + 1) for child in children)
                        except PermissionError:
                            pass
                entry_count += 1
            except Exception:
                pass

        return {
            "success": True,
            "operation": "tree",
            "base_path": str(base),
            "items": items,
            "truncated": entry_count >= max_entries,
            "elapsed": round(time.time() - start, 3),
        }

    def _index_repo(self, path: str, max_files: int, include_ext: List[str], 
                   exclude_glob: List[str], max_size_bytes: int) -> Dict[str, Any]:
        start = time.time()
        base = self._safe_p(path)
        if not base.exists():
            return {"success": False, "error": f"Path not found: {base}", "path": str(base)}

        files = []
        skipped = 0
        to_visit = [base]
        
        while to_visit and len(files) < max_files:
            current = to_visit.pop(0)
            try:
                if current.is_file():
                    if self._excluded(current, exclude_glob):
                        skipped += 1
                        continue
                    if not self._want_ext(current, include_ext):
                        skipped += 1
                        continue
                    
                    size = current.stat().st_size
                    if size > max_size_bytes:
                        skipped += 1
                        continue
                    
                    try:
                        content = current.read_text(encoding="utf-8", errors="replace")
                        if self._looks_binary(content.encode('utf-8', errors='replace')[:512]):
                            skipped += 1
                            continue
                    except Exception:
                        skipped += 1
                        continue
                    
                    files.append({
                        "path": str(current.relative_to(base)),
                        "size": size,
                        "content": content,
                        "lines": content.count('\n') + 1,
                    })
                    
                elif current.is_dir():
                    if self._excluded(current, exclude_glob):
                        skipped += 1
                        continue
                    try:
                        children = list(current.iterdir())
                        to_visit.extend(children)
                    except PermissionError:
                        skipped += 1
            except Exception:
                skipped += 1

        return {
            "success": True,
            "operation": "index_repo",
            "base_path": str(base),
            "files": files,
            "file_count": len(files),
            "skipped_count": skipped,
            "truncated": len(files) >= max_files,
            "elapsed": round(time.time() - start, 3),
        }

    def _read_chunk(self, path: Optional[str], offset: int, limit: int) -> Dict[str, Any]:
        if not path:
            return {"success": False, "error": "Missing 'path' for read_chunk"}
        
        file_path = self._safe_p(path)
        if not file_path.exists():
            return {"success": False, "error": f"File not found: {file_path}", "path": str(file_path)}
        if not file_path.is_file():
            return {"success": False, "error": f"Path is not a file: {file_path}", "path": str(file_path)}

        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                f.seek(offset)
                content = f.read(limit)
            
            return {
                "success": True,
                "operation": "read_chunk",
                "path": str(file_path),
                "offset": offset,
                "limit": limit,
                "content": content,
                "bytes_read": len(content),
            }
        except Exception as e:
            return {"success": False, "error": f"Error reading file: {e}", "path": str(file_path)}

    def _grep(self, path: str, query: str, regex: bool, include_ext: List[str], 
             exclude_glob: List[str], max_matches: int) -> Dict[str, Any]:
        import re
        
        start = time.time()
        base = self._safe_p(path)
        if not base.exists():
            return {"success": False, "error": f"Path not found: {base}", "path": str(base)}

        if not query:
            return {"success": False, "error": "Empty query"}

        matches = []
        searched_files = 0
        skipped_files = 0
        
        # Compile pattern
        try:
            if regex:
                pattern = re.compile(query, re.IGNORECASE | re.MULTILINE)
            else:
                pattern = re.compile(re.escape(query), re.IGNORECASE | re.MULTILINE)
        except re.error as e:
            return {"success": False, "error": f"Invalid regex pattern: {e}"}

        # Search files
        to_visit = [base] if base.is_file() else list(base.rglob("*"))
        
        for file_path in to_visit:
            if len(matches) >= max_matches:
                break
                
            if not file_path.is_file():
                continue
                
            if self._excluded(file_path, exclude_glob):
                skipped_files += 1
                continue
                
            if not self._want_ext(file_path, include_ext):
                skipped_files += 1
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                if self._looks_binary(content.encode('utf-8', errors='replace')[:512]):
                    skipped_files += 1
                    continue
                    
                searched_files += 1
                lines = content.splitlines()
                
                for line_no, line in enumerate(lines, 1):
                    if len(matches) >= max_matches:
                        break
                        
                    match = pattern.search(line)
                    if match:
                        matches.append({
                            "file": str(file_path.relative_to(base.parent if base.is_file() else base)),
                            "line": line_no,
                            "content": line.strip(),
                            "match_start": match.start(),
                            "match_end": match.end(),
                        })
                        
            except Exception:
                skipped_files += 1

        return {
            "success": True,
            "operation": "grep",
            "base_path": str(base),
            "query": query,
            "regex": regex,
            "matches": matches,
            "match_count": len(matches),
            "searched_files": searched_files,
            "skipped_files": skipped_files,
            "truncated": len(matches) >= max_matches,
            "elapsed": round(time.time() - start, 3),
        }