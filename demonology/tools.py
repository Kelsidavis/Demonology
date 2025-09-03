# demonology/tools.py
from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Safety fence
# ---------------------------------------------------------------------

def _get_default_safe_root() -> Path:
    """Get the default safe root, checking environment first."""
    env_root = os.environ.get("GRIMOIRE_SAFE_ROOT", "")
    if env_root:
        return Path(env_root).resolve()
    return Path.cwd().resolve()

SAFE_ROOT = _get_default_safe_root()

IMMUTABLE_BLOCK_PATTERNS = [
    r"rm\s+-rf\s+/(?:\s|$)",
    r"mkfs\.",
    r"dd\s+if=",
    r"shutdown\b",
    r"reboot\b",
    r"halt\b",
    r"mount\b.*\s/dev/",
    r"umount\b",
]
IMMUTABLE_BLOCK_RE = re.compile("|".join(IMMUTABLE_BLOCK_PATTERNS),
                                re.IGNORECASE | re.MULTILINE)


def _is_within(base: Path, target: Path) -> bool:
    try:
        base = base.resolve()
        target = target.resolve()
        return str(target).startswith(str(base))
    except Exception:
        return False


def _safe_path(p: str, want_dir: bool = False) -> Path:
    if not p:
        raise ValueError("Empty path")
    
    path = Path(p)
    
    # Handle absolute paths
    if path.is_absolute():
        path = path.resolve()
        if not _is_within(SAFE_ROOT, path):
            raise PermissionError(f"Refusing to operate outside SAFE_ROOT: {path}")
    else:
        # Handle relative paths
        path = (SAFE_ROOT / path).resolve()
        if not _is_within(SAFE_ROOT, path):
            raise PermissionError(f"Refusing to operate outside SAFE_ROOT: {path}")
    
    if want_dir and not str(path).endswith(os.sep):
        path = Path(str(path) + os.sep)
    return path


def _blocked(cmd: str) -> Optional[str]:
    if IMMUTABLE_BLOCK_RE.search(cmd or ""):
        return "Command blocked by safety policy"
    return None


# ---------------------------------------------------------------------
# Base Tool
# ---------------------------------------------------------------------

@dataclass
class Tool:
    name: str
    description: str
    enabled: bool = True

    def is_available(self) -> bool:
        return True

    def to_openai_function(self) -> Dict[str, Any]:
        raise NotImplementedError

    async def execute(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


# ---------------------------------------------------------------------
# File Operations Tool
# ---------------------------------------------------------------------

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

# ---------------------------------------------------------------------
# Codebase Analysis Tool
# ---------------------------------------------------------------------

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
        nontext = data.translate(None, text_chars)
        return float(len(nontext)) / float(len(data)) > 0.30

    # ops
    def _tree(self, rel: str, depth: int, max_entries: int) -> Dict[str, Any]:
        base = self._safe_p(rel)
        if not base.exists():
            raise FileNotFoundError("Base path does not exist.")
        if not base.is_dir():
            raise NotADirectoryError("Path is not a directory.")
        out: List[Dict[str, Any]] = []
        entries = 0

        def walk(d: Path, lvl: int):
            nonlocal entries
            if entries >= max_entries:
                return
            try:
                for item in sorted(d.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                    if entries >= max_entries:
                        return
                    # Calculate relative path from the base directory being scanned
                    try:
                        rel_path = str(item.relative_to(base))
                        if not rel_path:  # If it's the base directory itself
                            rel_path = "."
                    except ValueError:
                        # Fallback to absolute path if relative calculation fails
                        rel_path = str(item)
                    out.append({"path": rel_path, "type": "directory" if item.is_dir() else "file"})
                    entries += 1
                    if item.is_dir() and lvl < depth:
                        walk(item, lvl + 1)
            except PermissionError:
                pass

        walk(base, 0)
        # Calculate root path for return value
        try:
            root_path = str(base.relative_to(self.safe_root))
        except ValueError:
            # If base is outside safe_root, use absolute path or current directory name
            root_path = base.name if base.name else str(base)
        
        return {"success": True, "operation": "tree", "root": root_path, "depth": depth, "count": len(out), "nodes": out}

    def _index_repo(self, rel: str, max_files: int, include_ext: List[str], exclude_glob: List[str], max_size_bytes: int) -> Dict[str, Any]:
        base = self._safe_p(rel)
        if not base.exists() or not base.is_dir():
            raise NotADirectoryError("Path is not a directory.")
        items: List[Dict[str, Any]] = []
        count = 0
        for root, dirs, files in os.walk(base):
            root_p = Path(root)
            dirs[:] = [d for d in dirs if not self._excluded(root_p / d, exclude_glob)]
            for f in files:
                p = root_p / f
                if self._excluded(p, exclude_glob):
                    continue
                if not self._want_ext(p, include_ext):
                    continue
                try:
                    size = p.stat().st_size
                except OSError:
                    continue
                if size > max_size_bytes:
                    continue
                # Calculate relative path from the base directory
                try:
                    rel_path = str(p.relative_to(base))
                except ValueError:
                    rel_path = str(p)
                items.append({"path": rel_path, "size": int(size), "ext": p.suffix.lower()})
                count += 1
                if count >= max_files:
                    break
            if count >= max_files:
                break
        # Calculate root path for return value
        try:
            root_path = str(base.relative_to(self.safe_root))
        except ValueError:
            root_path = base.name if base.name else str(base)
        
        return {"success": True, "operation": "index_repo", "root": root_path, "count": len(items), "files": items}

    def _read_chunk(self, rel: Optional[str], offset: int, limit: int) -> Dict[str, Any]:
        if not rel:
            raise ValueError("path is required for read_chunk")
        p = self._safe_p(rel)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError("Path is not a file.")
        data = p.read_bytes()
        if self._looks_binary(data[:4096]):
            return {"success": False, "operation": "read_chunk", "error": "Likely binary file; refusing to return raw bytes.", "path": str(p)}
        start = max(0, int(offset))
        end = max(start, start + max(1, int(limit)))
        slice_bytes = data[start:end]
        try:
            text = slice_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = slice_bytes.decode("latin-1")
        eof = end >= len(data)
        return {"success": True, "operation": "read_chunk", "path": str(p), "offset": start, "limit": end - start, "eof": eof, "content": text}

    def _grep(self, rel: str, query: str, regex: bool, include_ext: List[str], exclude_glob: List[str], max_matches: int) -> Dict[str, Any]:
        if not query:
            raise ValueError("query is required for grep")
        base = self._safe_p(rel)
        if not base.exists() or not base.is_dir():
            raise NotADirectoryError("Path is not a directory.")
        hits: List[Dict[str, Any]] = []
        total = 0
        pattern = re.compile(query, re.IGNORECASE) if regex else None

        for root, dirs, files in os.walk(base):
            root_p = Path(root)
            dirs[:] = [d for d in dirs if not self._excluded(root_p / d, exclude_glob)]
            for f in files:
                p = root_p / f
                if self._excluded(p, exclude_glob):
                    continue
                if not self._want_ext(p, include_ext):
                    continue
                try:
                    with p.open("rb") as fh:
                        sample = fh.read(4096)
                        if self._looks_binary(sample):
                            continue
                        fh.seek(0)
                        for i, raw in enumerate(fh, start=1):
                            try:
                                line = raw.decode("utf-8")
                            except UnicodeDecodeError:
                                line = raw.decode("latin-1")
                            found = bool(pattern.search(line)) if pattern else (query.lower() in line.lower())
                            if found:
                                # Calculate relative path from base directory
                                try:
                                    rel_path = str(p.relative_to(base))
                                except ValueError:
                                    rel_path = str(p)
                                hits.append({"path": rel_path, "line": i, "preview": line.rstrip()[:300]})
                                total += 1
                                if total >= max_matches:
                                    try:
                                        root_path = str(base.relative_to(self.safe_root))
                                    except ValueError:
                                        root_path = base.name if base.name else str(base)
                                    return {"success": True, "operation": "grep", "root": root_path, "count": len(hits), "results": hits}
                except OSError:
                    continue
        # Calculate root path for return value
        try:
            root_path = str(base.relative_to(self.safe_root))
        except ValueError:
            root_path = base.name if base.name else str(base)
        
        return {"success": True, "operation": "grep", "root": root_path, "count": len(hits), "results": hits}



# ---------------------------------------------------------------------
# Code Execution Tool (minimal, guarded)
# ---------------------------------------------------------------------

class CodeExecutionTool(Tool):
    def __init__(self, safe_root: Optional[Path] = None):
        super().__init__("code_execution", "Execute small code snippets in a sandboxed subprocess.")
        self.safe_root = safe_root or SAFE_ROOT

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {"type": "string", "description": "python or bash"},
                    "code": {"type": "string", "description": "Code to execute"},
                    "timeout": {"type": "integer", "description": "Seconds before kill", "default": 15}
                },
                "required": ["language", "code"]
            }
        }

    async def execute(self, language: str, code: str, timeout: int = 15, **_) -> Dict[str, Any]:
        try:
            language = (language or "").strip().lower()
            if language not in {"python", "bash"}:
                return {"success": False, "error": f"Unsupported language: {language}"}

            if language == "bash":
                import asyncio as aio
                blocked = _blocked(code)
                if blocked:
                    return {"success": False, "error": blocked}
                proc = await aio.create_subprocess_shell(
                    code,
                    stdout=aio.subprocess.PIPE,
                    stderr=aio.subprocess.PIPE,
                    cwd=str(self.safe_root),
                )
            else:
                import asyncio as aio
                proc = await aio.create_subprocess_exec(
                    "python3", "-c", code,
                    stdout=aio.subprocess.PIPE,
                    stderr=aio.subprocess.PIPE,
                    cwd=str(self.safe_root),
                )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                return {"success": False, "error": "Execution timed out"}

            rc = proc.returncode
            return {
                "success": rc == 0,
                "returncode": rc,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
            }
        except Exception as e:
            logger.exception("CodeExecutionTool error")
            return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------
# Web Search Tool
# ---------------------------------------------------------------------

class WebSearchTool(Tool):
    """Search the web using DuckDuckGo or similar search engines."""
    
    def __init__(self):
        super().__init__("web_search", "Search the web for information")
    
    def is_available(self) -> bool:
        try:
            import requests
            return True
        except ImportError:
            return False
    
    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "description": "Number of results (default 5)", "default": 5}
                },
                "required": ["query"]
            }
        }
    
    async def execute(self, query: str, num_results: int = 5, **kwargs) -> Dict[str, Any]:
        try:
            import requests
            from urllib.parse import quote_plus
            import re
            
            # Use DuckDuckGo instant answer API
            search_url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_redirect=1"
            
            response = requests.get(search_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            
            # Get instant answer
            if data.get("Abstract"):
                results.append({
                    "title": data.get("AbstractText", "")[:100] + "..." if len(data.get("AbstractText", "")) > 100 else data.get("AbstractText", ""),
                    "url": data.get("AbstractURL", ""),
                    "snippet": data.get("Abstract", "")
                })
            
            # Get related topics
            for topic in data.get("RelatedTopics", [])[:num_results-1]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("Text", "")[:100] + "..." if len(topic.get("Text", "")) > 100 else topic.get("Text", ""),
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", "")
                    })
            
            return {
                "success": True,
                "query": query,
                "results": results[:num_results],
                "total_results": len(results)
            }
            
        except Exception as e:
            logger.exception("WebSearchTool error")
            return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------
# Project Planning Tool
# ---------------------------------------------------------------------

class ProjectPlanningTool(Tool):
    """Analyze existing projects or generate new project plans and task breakdowns for development projects."""
    
    def __init__(self, safe_root: Optional[Path] = None):
        super().__init__("project_planning", "Analyze existing projects and continue work, or generate new project plans and task breakdowns")
        self.safe_root = safe_root or SAFE_ROOT
        # Debug logging removed for cleaner output
    
    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["analyze_existing", "create_new", "continue"], "description": "Action to perform: analyze existing project, create new project, or continue existing work", "default": "analyze_existing"},
                    "project_name": {"type": "string", "description": "Name of the project (for new projects) or path to existing project directory"},
                    "project_description": {"type": "string", "description": "Description of what the project should do (for new projects)"},
                    "technology_stack": {"type": "string", "description": "Preferred technologies (e.g., Python, React, C++)"},
                    "complexity": {"type": "string", "enum": ["simple", "medium", "complex"], "description": "Project complexity level"},
                    "save_to_file": {"type": "boolean", "description": "Whether to save the plan to a file", "default": True},
                    "execute_plan": {"type": "boolean", "description": "Whether to automatically create the project structure", "default": True}
                },
                "required": ["action", "project_name"]
            }
        }
    
    async def execute(self, action: str = "analyze_existing", project_name: str = "", project_description: str = "", 
                     technology_stack: str = "", complexity: str = "medium", save_to_file: bool = True, 
                     execute_plan: bool = True, **kwargs) -> Dict[str, Any]:
        try:
            if action == "analyze_existing":
                return await self._analyze_existing_project(project_name, save_to_file)
            elif action == "continue":
                return await self._continue_existing_project(project_name, save_to_file)
            elif action == "create_new":
                return await self._create_new_project(project_name, project_description, technology_stack, 
                                                    complexity, save_to_file, execute_plan)
            else:
                # Default behavior - try to analyze existing first, then create new
                existing_analysis = await self._analyze_existing_project(project_name, False)
                if existing_analysis["success"] and existing_analysis.get("project_found", False):
                    return existing_analysis
                else:
                    return await self._create_new_project(project_name, project_description, technology_stack, 
                                                        complexity, save_to_file, execute_plan)
                
        except Exception as e:
            logger.exception("ProjectPlanningTool error")
            return {"success": False, "error": str(e)}
    
    async def _analyze_existing_project(self, project_name: str, save_to_file: bool = True) -> Dict[str, Any]:
        """Analyze an existing project and provide continuation suggestions."""
        try:
            # Look for existing project directory
            project_path = None
            
            # Try different approaches to find the project
            if "/" in project_name or "\\" in project_name:
                # Path provided - check if it's a file or directory
                try:
                    input_path = Path(project_name).resolve()
                    
                    # If it's a file, use its parent directory
                    if input_path.exists() and input_path.is_file():
                        project_path = input_path.parent
                        # If it's a plan file, try to infer the project directory name
                        if input_path.name.endswith('_plan.md'):
                            # Look for directory with similar name
                            inferred_name = input_path.name.replace('_plan.md', '')
                            inferred_dir = input_path.parent / inferred_name
                            if inferred_dir.exists() and inferred_dir.is_dir():
                                project_path = inferred_dir
                    # If it's a directory, use it directly
                    elif input_path.exists() and input_path.is_dir():
                        project_path = input_path
                    else:
                        project_path = None
                        
                except (PermissionError, ValueError, OSError):
                    project_path = None
            else:
                # Try to find project in current directory or subdirectories
                try:
                    search_dir = Path.cwd().resolve()
                except Exception:
                    search_dir = self.safe_root
                
                # Look for exact match
                potential_path = search_dir / project_name.replace(' ', '_').lower()
                if potential_path.exists() and potential_path.is_dir():
                    project_path = potential_path
                else:
                    # Look for similar names
                    for item in search_dir.iterdir():
                        if item.is_dir() and project_name.lower().replace(' ', '_') in item.name.lower():
                            project_path = item
                            break
            
            if not project_path or not project_path.exists():
                try:
                    current_dir = Path.cwd().resolve()
                    search_locations = [str(current_dir)]
                    if current_dir != self.safe_root:
                        search_locations.append(str(self.safe_root))
                except Exception:
                    search_locations = [str(self.safe_root)]
                    
                return {
                    "success": True,
                    "project_found": False,
                    "message": f"No existing project found for '{project_name}'. Consider creating a new project.",
                    "search_locations": search_locations
                }
            
            # Analyze the existing project
            analysis = await self._perform_project_analysis(project_path)
            
            # Generate continuation plan
            continuation_plan = await self._generate_continuation_plan(project_path, analysis)
            
            result = {
                "success": True,
                "project_found": True,
                "project_path": str(project_path),
                "project_name": project_path.name,
                "analysis": analysis,
                "continuation_plan": continuation_plan,
                "message": f"Analyzed existing project '{project_path.name}' at {project_path}"
            }
            
            if save_to_file:
                # Save continuation plan
                plan_filename = f"{project_path.name}_continuation_plan.md"
                plan_file = project_path / plan_filename
                plan_file.write_text(continuation_plan, encoding="utf-8")
                result["plan_file"] = str(plan_file)
            
            return result
            
        except Exception as e:
            logger.exception("Project analysis error")
            return {"success": False, "error": str(e)}
    
    async def _continue_existing_project(self, project_name: str, save_to_file: bool = True) -> Dict[str, Any]:
        """Continue work on an existing project."""
        analysis_result = await self._analyze_existing_project(project_name, False)
        
        if not analysis_result["success"] or not analysis_result.get("project_found", False):
            return analysis_result
        
        project_path = Path(analysis_result["project_path"])
        analysis = analysis_result["analysis"]
        
        # Generate specific next steps
        next_steps = await self._generate_next_steps(project_path, analysis)
        
        result = {
            "success": True,
            "project_path": str(project_path),
            "project_name": project_path.name,
            "analysis": analysis,
            "next_steps": next_steps,
            "message": f"Ready to continue work on '{project_path.name}'"
        }
        
        if save_to_file:
            # Save next steps
            steps_filename = f"{project_path.name}_next_steps.md"
            steps_file = project_path / steps_filename
            steps_file.write_text(next_steps, encoding="utf-8")
            result["steps_file"] = str(steps_file)
        
        return result
    
    async def _create_new_project(self, project_name: str, project_description: str, technology_stack: str, 
                                complexity: str, save_to_file: bool, execute_plan: bool) -> Dict[str, Any]:
        """Create a new project (original functionality)."""
        # Generate project structure based on complexity and technology
        plan = self._generate_project_plan(project_name, project_description, technology_stack, complexity)
        
        result = {
            "success": True,
            "project_name": project_name,
            "plan": plan,
            "file_saved": None,
            "project_created": False,
            "files_created": []
        }
        
        if save_to_file:
            # Save to project plan file in the current working directory
            plan_filename = f"{project_name.replace(' ', '_').lower()}_plan.md"
            
            # Always save in current working directory, not safe_root
            try:
                current_dir = Path.cwd().resolve()
                plan_file = current_dir / plan_filename
            except Exception:
                # If we can't get current directory, use safe_root as fallback
                plan_file = self.safe_root / plan_filename
            
            plan_file.write_text(plan, encoding="utf-8")
            result["file_saved"] = str(plan_file)
        
        if execute_plan:
            # Create the actual project structure
            created_files = await self._execute_project_plan(project_name, technology_stack, complexity)
            result["project_created"] = True
            result["files_created"] = created_files
            result["message"] = f"Project '{project_name}' created with {len(created_files)} files"
        else:
            result["message"] = f"Project plan generated for '{project_name}'"
        
        return result
    
    async def _perform_project_analysis(self, project_path: Path) -> Dict[str, Any]:
        """Analyze an existing project's structure and contents."""
        analysis = {
            "project_type": "unknown",
            "files_found": [],
            "key_files": {},
            "dependencies": [],
            "structure": {},
            "last_modified": None,
            "size_info": {}
        }
        
        try:
            # Get basic info
            if project_path.exists():
                analysis["last_modified"] = project_path.stat().st_mtime
                
            # Analyze file structure
            total_files = 0
            total_size = 0
            file_types = {}
            
            for item in project_path.rglob("*"):
                if item.is_file():
                    total_files += 1
                    try:
                        size = item.stat().st_size
                        total_size += size
                        ext = item.suffix.lower()
                        file_types[ext] = file_types.get(ext, 0) + 1
                    except:
                        pass
                    
                    # Record first 50 files
                    if len(analysis["files_found"]) < 50:
                        analysis["files_found"].append(str(item.relative_to(project_path)))
            
            analysis["size_info"] = {
                "total_files": total_files,
                "total_size_bytes": total_size,
                "file_types": file_types
            }
            
            # Detect project type and key files
            key_files = {
                "package.json": project_path / "package.json",
                "requirements.txt": project_path / "requirements.txt", 
                "Cargo.toml": project_path / "Cargo.toml",
                "Makefile": project_path / "Makefile",
                "CMakeLists.txt": project_path / "CMakeLists.txt",
                "setup.py": project_path / "setup.py",
                "pyproject.toml": project_path / "pyproject.toml",
                "README.md": project_path / "README.md",
                "README.txt": project_path / "README.txt",
                "main.py": project_path / "main.py",
                "src/main.py": project_path / "src" / "main.py",
                "main.cpp": project_path / "main.cpp",
                "src/main.cpp": project_path / "src" / "main.cpp"
            }
            
            for name, path in key_files.items():
                if path.exists():
                    analysis["key_files"][name] = str(path.relative_to(project_path))
                    
            # Determine project type
            if "package.json" in analysis["key_files"]:
                analysis["project_type"] = "nodejs/javascript"
                # Try to read package.json for dependencies
                try:
                    import json
                    pkg_json = json.loads((project_path / "package.json").read_text())
                    deps = list(pkg_json.get("dependencies", {}).keys())
                    dev_deps = list(pkg_json.get("devDependencies", {}).keys())
                    analysis["dependencies"] = deps + dev_deps
                except:
                    pass
            elif "requirements.txt" in analysis["key_files"] or "setup.py" in analysis["key_files"]:
                analysis["project_type"] = "python"
                # Try to read requirements
                try:
                    if "requirements.txt" in analysis["key_files"]:
                        req_content = (project_path / "requirements.txt").read_text()
                        analysis["dependencies"] = [line.strip() for line in req_content.split('\n') if line.strip() and not line.startswith('#')]
                except:
                    pass
            elif "Cargo.toml" in analysis["key_files"]:
                analysis["project_type"] = "rust"
            elif "CMakeLists.txt" in analysis["key_files"] or "Makefile" in analysis["key_files"]:
                analysis["project_type"] = "cpp/c"
            
        except Exception as e:
            logger.warning(f"Error analyzing project {project_path}: {e}")
        
        return analysis
    
    async def _generate_continuation_plan(self, project_path: Path, analysis: Dict[str, Any]) -> str:
        """Generate a plan for continuing work on an existing project."""
        project_name = project_path.name
        project_type = analysis.get("project_type", "unknown")
        key_files = analysis.get("key_files", {})
        
        plan = f"""# {project_name} - Project Continuation Analysis

## Project Overview
- **Location**: {project_path}
- **Type**: {project_type}
- **Total Files**: {analysis.get('size_info', {}).get('total_files', 'Unknown')}
- **Total Size**: {analysis.get('size_info', {}).get('total_size_bytes', 0)} bytes

## Project Structure Analysis
"""
        
        if key_files:
            plan += "\n### Key Files Found:\n"
            for file_name, file_path in key_files.items():
                plan += f"- **{file_name}**: `{file_path}`\n"
        
        if analysis.get("dependencies"):
            plan += f"\n### Dependencies ({len(analysis['dependencies'])}):\n"
            for dep in analysis["dependencies"][:10]:  # Show first 10
                plan += f"- {dep}\n"
            if len(analysis["dependencies"]) > 10:
                plan += f"- ... and {len(analysis['dependencies']) - 10} more\n"
        
        plan += f"""
## Recommended Next Steps

Based on the project analysis, here are suggested actions to continue development:

### 1. Environment Setup
"""
        
        if project_type == "python":
            plan += """- Set up virtual environment: `python -m venv venv`
- Activate environment: `source venv/bin/activate` (Linux/Mac) or `venv\\Scripts\\activate` (Windows)
- Install dependencies: `pip install -r requirements.txt`
"""
        elif project_type == "nodejs/javascript":
            plan += """- Install dependencies: `npm install`
- Check available scripts: `npm run`
"""
        elif project_type == "rust":
            plan += """- Build project: `cargo build`
- Run tests: `cargo test`
"""
        elif project_type == "cpp/c":
            plan += """- Build project: `make` or `cmake .` then `make`
"""
        
        plan += """
### 2. Code Review
- Review recent changes and TODOs
- Check for incomplete features or bug fixes
- Review documentation for outdated information

### 3. Testing & Quality
- Run existing tests to ensure current state is stable
- Check for linting or formatting issues
- Review error logs if any

### 4. Development Priorities
Based on the project structure, consider:
- Completing any incomplete features
- Adding missing documentation
- Improving test coverage
- Optimizing performance bottlenecks
- Adding new features as needed

---
*Analysis generated by Demonology Project Planning Tool*
"""
        
        return plan
    
    async def _generate_next_steps(self, project_path: Path, analysis: Dict[str, Any]) -> str:
        """Generate specific actionable next steps for continuing the project."""
        project_name = project_path.name
        project_type = analysis.get("project_type", "unknown")
        
        steps = f"""# Next Steps for {project_name}

## Immediate Actions

### 1. Project Setup
"""
        
        if project_type == "python":
            steps += """```bash
cd """ + str(project_path) + """
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR venv\\Scripts\\activate  # Windows
pip install -r requirements.txt
python -m pytest  # Run tests if available
```
"""
        elif project_type == "nodejs/javascript":
            steps += """```bash
cd """ + str(project_path) + """
npm install
npm test  # Run tests if available
npm start  # Start development server
```
"""
        elif project_type == "rust":
            steps += """```bash
cd """ + str(project_path) + """
cargo build
cargo test
cargo run
```
"""
        
        steps += """
### 2. Code Analysis
- [ ] Read through main entry points and core modules
- [ ] Identify any TODO comments or incomplete features
- [ ] Check for any compilation or runtime errors
- [ ] Review recent git history (if available)

### 3. Development Focus Areas
- [ ] Complete any unfinished features
- [ ] Fix any identified bugs or issues  
- [ ] Add missing error handling
- [ ] Improve documentation where needed
- [ ] Add or improve tests for critical functionality

### 4. Quality Improvements
- [ ] Run linting tools and fix style issues
- [ ] Check for security vulnerabilities
- [ ] Optimize performance if needed
- [ ] Update dependencies if outdated

## Development Workflow
1. Make small, incremental changes
2. Test frequently during development
3. Commit changes regularly with clear messages
4. Document new features or significant changes
5. Consider adding integration tests for new functionality

---
*Generated by Demonology Project Planning Tool*
"""
        
        return steps
    
    def _generate_project_plan(self, name: str, description: str, tech_stack: str, complexity: str) -> str:
        """Generate a detailed project plan."""
        
        # Determine project phases based on complexity
        phases = {
            "simple": ["Planning", "Implementation", "Testing"],
            "medium": ["Planning", "Design", "Implementation", "Testing", "Documentation"],
            "complex": ["Planning", "Research", "Design", "Implementation", "Testing", "Documentation", "Deployment"]
        }
        
        # Generate file structure suggestions based on technology
        file_structure = self._suggest_file_structure(tech_stack.lower(), complexity)
        
        plan = f"""# {name} - Project Plan

## Project Overview
{description}

**Technology Stack:** {tech_stack or 'Not specified'}
**Complexity Level:** {complexity.title()}

## Project Phases

"""
        
        for i, phase in enumerate(phases.get(complexity, phases["medium"]), 1):
            plan += f"### {i}. {phase}\n"
            plan += self._get_phase_details(phase, complexity) + "\n\n"
        
        plan += f"""## Suggested File Structure

```
{name.replace(' ', '_').lower()}/
{file_structure}
```

## Development Milestones

- [ ] Project setup and initial structure
- [ ] Core functionality implementation
- [ ] User interface/interaction layer
- [ ] Testing and validation
- [ ] Documentation and README
- [ ] Final review and optimization

## Resources and Dependencies

Based on the technology stack, you may need:
{self._suggest_dependencies(tech_stack.lower())}

---
*Generated by Demonology Project Planning Tool*
"""
        
        return plan
    
    def _get_phase_details(self, phase: str, complexity: str) -> str:
        """Get detailed description for each phase."""
        details = {
            "Planning": "- Define project scope and requirements\n- Set up development environment\n- Create project timeline",
            "Research": "- Research existing solutions\n- Evaluate libraries and frameworks\n- Prototype key features",
            "Design": "- Create system architecture\n- Design user interface mockups\n- Plan database schema (if applicable)",
            "Implementation": "- Set up project structure\n- Implement core functionality\n- Build user interface",
            "Testing": "- Write unit tests\n- Perform integration testing\n- Manual testing and bug fixes",
            "Documentation": "- Write API documentation\n- Create user guides\n- Add inline code comments",
            "Deployment": "- Set up production environment\n- Configure CI/CD pipeline\n- Deploy and monitor application"
        }
        return details.get(phase, f"- Complete {phase.lower()} phase")
    
    def _suggest_file_structure(self, tech_stack: str, complexity: str) -> str:
        """Suggest file structure based on technology stack."""
        if "python" in tech_stack:
            return """├── src/
│   ├── __init__.py
│   ├── main.py
│   └── modules/
├── tests/
│   └── test_main.py
├── requirements.txt
├── README.md
└── setup.py"""
        elif "javascript" in tech_stack or "node" in tech_stack:
            return """├── src/
│   ├── index.js
│   └── components/
├── tests/
├── package.json
├── README.md
└── .gitignore"""
        elif "c++" in tech_stack or "cpp" in tech_stack:
            return """├── src/
│   ├── main.cpp
│   └── headers/
├── tests/
├── Makefile
├── README.md
└── CMakeLists.txt"""
        elif "react" in tech_stack:
            return """├── public/
├── src/
│   ├── components/
│   ├── App.js
│   └── index.js
├── package.json
├── README.md
└── .gitignore"""
        else:
            return """├── src/
├── tests/
├── docs/
├── README.md
└── .gitignore"""
    
    def _suggest_dependencies(self, tech_stack: str) -> str:
        """Suggest dependencies based on technology stack."""
        if "python" in tech_stack:
            return "- Python 3.8+\n- pip for package management\n- Virtual environment (venv/conda)\n- Testing framework (pytest)"
        elif "javascript" in tech_stack or "node" in tech_stack:
            return "- Node.js and npm\n- Testing framework (Jest/Mocha)\n- Linting tools (ESLint)"
        elif "c++" in tech_stack:
            return "- C++ compiler (GCC/Clang)\n- Build system (Make/CMake)\n- Testing framework (Google Test)"
        elif "react" in tech_stack:
            return "- Node.js and npm\n- React development tools\n- Testing library (React Testing Library)"
        else:
            return "- Development environment setup\n- Version control (Git)\n- Testing framework\n- Documentation tools"
    
    async def _execute_project_plan(self, project_name: str, tech_stack: str, complexity: str) -> List[str]:
        """Create the actual project structure and files."""
        created_files = []
        
        # Clean up project name for directory use
        clean_name = project_name.replace(' ', '_').lower()
        
        # Create project in current working directory, not safe_root
        try:
            current_dir = Path.cwd().resolve()
            
            # If project_name looks like a path, use it directly
            if "/" in project_name or "\\" in project_name:
                # Try to interpret as relative path from current directory
                project_dir = current_dir / project_name
            else:
                project_dir = current_dir / clean_name
                
        except Exception:
            # If we can't get current directory, fall back to safe_root
            if "/" in project_name or "\\" in project_name:
                try:
                    project_dir = _safe_path(project_name, want_dir=True)
                except (PermissionError, ValueError):
                    project_dir = self.safe_root / clean_name
            else:
                project_dir = self.safe_root / clean_name
        
        # Create project directory
        project_dir.mkdir(parents=True, exist_ok=True)
        created_files.append(str(project_dir))
        
        # Create technology-specific project structure
        if "python" in tech_stack.lower():
            created_files.extend(await self._create_python_project(project_dir, project_name, complexity))
        elif "c++" in tech_stack.lower() or "cpp" in tech_stack.lower():
            created_files.extend(await self._create_cpp_project(project_dir, project_name, complexity))
        elif "javascript" in tech_stack.lower() or "node" in tech_stack.lower():
            created_files.extend(await self._create_js_project(project_dir, project_name, complexity))
        elif "react" in tech_stack.lower():
            created_files.extend(await self._create_react_project(project_dir, project_name, complexity))
        else:
            created_files.extend(await self._create_generic_project(project_dir, project_name, complexity))
        
        return created_files
    
    async def _create_python_project(self, project_dir: Path, project_name: str, complexity: str) -> List[str]:
        """Create a Python project structure."""
        created = []
        
        # Create directories
        src_dir = project_dir / "src"
        src_dir.mkdir(exist_ok=True)
        created.append(str(src_dir))
        
        tests_dir = project_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        created.append(str(tests_dir))
        
        # Create main Python files
        main_py = src_dir / "main.py"
        main_py.write_text(f'''#!/usr/bin/env python3
"""
{project_name} - Main application entry point.
"""

def main():
    """Main function."""
    print("Hello from {project_name}!")
    # TODO: Implement your application logic here

if __name__ == "__main__":
    main()
''', encoding="utf-8")
        created.append(str(main_py))
        
        # Create __init__.py
        init_py = src_dir / "__init__.py"
        init_py.write_text('"""Main package."""\n', encoding="utf-8")
        created.append(str(init_py))
        
        # Create requirements.txt
        req_txt = project_dir / "requirements.txt"
        req_txt.write_text("# Add your dependencies here\n", encoding="utf-8")
        created.append(str(req_txt))
        
        # Create README.md
        readme = project_dir / "README.md"
        readme.write_text(f'''# {project_name}

## Description
TODO: Add project description

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/main.py
```

## Testing
```bash
python -m pytest tests/
```
''', encoding="utf-8")
        created.append(str(readme))
        
        # Create test file
        test_main = tests_dir / "test_main.py"
        test_main.write_text(f'''"""
Tests for {project_name}.
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import main

def test_main():
    """Test main function runs without error."""
    # TODO: Add meaningful tests
    assert True
''', encoding="utf-8")
        created.append(str(test_main))
        
        return created
    
    async def _create_cpp_project(self, project_dir: Path, project_name: str, complexity: str) -> List[str]:
        """Create a C++ project structure."""
        created = []
        
        # Create directories
        src_dir = project_dir / "src"
        src_dir.mkdir(exist_ok=True)
        created.append(str(src_dir))
        
        # Create main.cpp
        main_cpp = src_dir / "main.cpp"
        main_cpp.write_text(f'''#include <iostream>
#include <string>

int main() {{
    std::cout << "Hello from {project_name}!" << std::endl;
    // TODO: Implement your application logic here
    return 0;
}}
''', encoding="utf-8")
        created.append(str(main_cpp))
        
        # Create Makefile
        makefile = project_dir / "Makefile"
        makefile.write_text(f'''CXX=g++
CXXFLAGS=-std=c++17 -Wall -Wextra -O2
TARGET={project_name.replace(' ', '_').lower()}
SRCDIR=src
SOURCES=$(wildcard $(SRCDIR)/*.cpp)

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET)

.PHONY: all clean
''', encoding="utf-8")
        created.append(str(makefile))
        
        # Create README.md
        readme = project_dir / "README.md"
        readme.write_text(f'''# {project_name}

## Description
TODO: Add project description

## Build
```bash
make
```

## Run
```bash
./{project_name.replace(' ', '_').lower()}
```

## Clean
```bash
make clean
```
''', encoding="utf-8")
        created.append(str(readme))
        
        return created
    
    async def _create_js_project(self, project_dir: Path, project_name: str, complexity: str) -> List[str]:
        """Create a JavaScript/Node.js project structure."""
        created = []
        
        # Create directories
        src_dir = project_dir / "src"
        src_dir.mkdir(exist_ok=True)
        created.append(str(src_dir))
        
        # Create main.js
        main_js = src_dir / "index.js"
        main_js.write_text(f'''#!/usr/bin/env node
/**
 * {project_name} - Main application entry point
 */

function main() {{
    console.log("Hello from {project_name}!");
    // TODO: Implement your application logic here
}}

if (require.main === module) {{
    main();
}}

module.exports = {{ main }};
''', encoding="utf-8")
        created.append(str(main_js))
        
        # Create package.json
        package_json = project_dir / "package.json"
        package_json.write_text(f'''{{
  "name": "{project_name.replace(' ', '-').lower()}",
  "version": "1.0.0",
  "description": "TODO: Add project description",
  "main": "src/index.js",
  "scripts": {{
    "start": "node src/index.js",
    "test": "jest"
  }},
  "keywords": [],
  "author": "",
  "license": "MIT",
  "devDependencies": {{
    "jest": "^29.0.0"
  }}
}}
''', encoding="utf-8")
        created.append(str(package_json))
        
        # Create README.md
        readme = project_dir / "README.md"
        readme.write_text(f'''# {project_name}

## Description
TODO: Add project description

## Installation
```bash
npm install
```

## Usage
```bash
npm start
```

## Testing
```bash
npm test
```
''', encoding="utf-8")
        created.append(str(readme))
        
        return created
    
    async def _create_react_project(self, project_dir: Path, project_name: str, complexity: str) -> List[str]:
        """Create a React project structure."""
        created = []
        
        # Create directories
        src_dir = project_dir / "src"
        src_dir.mkdir(exist_ok=True)
        created.append(str(src_dir))
        
        public_dir = project_dir / "public"
        public_dir.mkdir(exist_ok=True)
        created.append(str(public_dir))
        
        # Create App.js
        app_js = src_dir / "App.js"
        app_js.write_text(f'''import React from 'react';
import './App.css';

function App() {{
  return (
    <div className="App">
      <header className="App-header">
        <h1>{project_name}</h1>
        <p>Welcome to your new React application!</p>
      </header>
    </div>
  );
}}

export default App;
''', encoding="utf-8")
        created.append(str(app_js))
        
        # Create index.js
        index_js = src_dir / "index.js"
        index_js.write_text('''import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
''', encoding="utf-8")
        created.append(str(index_js))
        
        # Create basic CSS
        app_css = src_dir / "App.css"
        app_css.write_text('''.App {
  text-align: center;
}

.App-header {
  background-color: #282c34;
  padding: 20px;
  color: white;
  min-height: 50vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}
''', encoding="utf-8")
        created.append(str(app_css))
        
        index_css = src_dir / "index.css"
        index_css.write_text('''body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
}
''', encoding="utf-8")
        created.append(str(index_css))
        
        # Create index.html
        index_html = public_dir / "index.html"
        index_html.write_text(f'''<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{project_name}</title>
  </head>
  <body>
    <div id="root"></div>
  </body>
</html>
''', encoding="utf-8")
        created.append(str(index_html))
        
        # Create package.json
        package_json = project_dir / "package.json"
        package_json.write_text(f'''{{
  "name": "{project_name.replace(' ', '-').lower()}",
  "version": "0.1.0",
  "private": true,
  "dependencies": {{
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1"
  }},
  "scripts": {{
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  }},
  "browserslist": {{
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }}
}}
''', encoding="utf-8")
        created.append(str(package_json))
        
        return created
    
    async def _create_generic_project(self, project_dir: Path, project_name: str, complexity: str) -> List[str]:
        """Create a generic project structure."""
        created = []
        
        # Create basic directories
        src_dir = project_dir / "src"
        src_dir.mkdir(exist_ok=True)
        created.append(str(src_dir))
        
        docs_dir = project_dir / "docs"
        docs_dir.mkdir(exist_ok=True)
        created.append(str(docs_dir))
        
        # Create README.md
        readme = project_dir / "README.md"
        readme.write_text(f'''# {project_name}

## Description
TODO: Add project description

## Getting Started
TODO: Add setup and usage instructions

## Contributing
TODO: Add contribution guidelines
''', encoding="utf-8")
        created.append(str(readme))
        
        # Create basic source file
        main_file = src_dir / "main.txt"
        main_file.write_text(f"Main file for {project_name}\nTODO: Implement your project here\n", encoding="utf-8")
        created.append(str(main_file))
        
        return created


# ---------------------------------------------------------------------
# Reddit Search Tool
# ---------------------------------------------------------------------

class RedditSearchTool(Tool):
    """Search Reddit for posts and comments using the Reddit API."""
    
    def __init__(self):
        super().__init__("reddit_search", "Search Reddit posts and discussions")
    
    def is_available(self) -> bool:
        try:
            # Check if we have requests at minimum for public API
            import requests
            return True
        except ImportError:
            return False
    
    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "subreddit": {"type": "string", "description": "Subreddit to search in (optional, searches all if not specified)"},
                    "sort": {"type": "string", "enum": ["relevance", "hot", "top", "new", "comments"], "description": "Sort method", "default": "relevance"},
                    "time_filter": {"type": "string", "enum": ["all", "day", "week", "month", "year"], "description": "Time filter", "default": "all"},
                    "limit": {"type": "integer", "description": "Number of results (1-25)", "default": 5, "minimum": 1, "maximum": 25}
                },
                "required": ["query"]
            }
        }
    
    async def execute(self, query: str, subreddit: str = None, sort: str = "relevance", 
                     time_filter: str = "all", limit: int = 5, **kwargs) -> Dict[str, Any]:
        try:
            import os
            
            # Check for Reddit API credentials in environment variables
            client_id = os.environ.get('REDDIT_CLIENT_ID')
            client_secret = os.environ.get('REDDIT_CLIENT_SECRET')
            user_agent = os.environ.get('REDDIT_USER_AGENT', f'Demonology:v1.0 (by /u/demonology_user)')
            
            # Try to use PRAW if available and credentials exist
            if client_id and client_secret:
                try:
                    import praw
                    from datetime import datetime
                    
                    # Initialize Reddit API client
                    reddit = praw.Reddit(
                        client_id=client_id,
                        client_secret=client_secret,
                        user_agent=user_agent
                    )
                    
                    results = []
                    
                    if subreddit:
                        # Search within a specific subreddit
                        sub = reddit.subreddit(subreddit)
                        if sort == "relevance":
                            posts = sub.search(query, sort=sort, time_filter=time_filter, limit=limit)
                        elif sort == "hot":
                            posts = sub.hot(limit=limit)
                        elif sort == "top":
                            posts = sub.top(time_filter=time_filter, limit=limit)
                        elif sort == "new":
                            posts = sub.new(limit=limit)
                        else:
                            posts = sub.search(query, sort=sort, time_filter=time_filter, limit=limit)
                    else:
                        # Search all of Reddit
                        if sort == "relevance":
                            posts = reddit.subreddit("all").search(query, sort=sort, time_filter=time_filter, limit=limit)
                        else:
                            posts = reddit.subreddit("all").search(query, sort=sort, time_filter=time_filter, limit=limit)
                    
                    for post in posts:
                        # Filter by query if not using search
                        if sort != "relevance" and query.lower() not in (post.title.lower() + " " + post.selftext.lower()):
                            continue
                            
                        results.append({
                            "title": post.title,
                            "author": str(post.author) if post.author else "[deleted]",
                            "subreddit": str(post.subreddit),
                            "score": post.score,
                            "url": f"https://reddit.com{post.permalink}",
                            "created_utc": datetime.fromtimestamp(post.created_utc).isoformat(),
                            "text": post.selftext[:500] + "..." if len(post.selftext) > 500 else post.selftext,
                            "num_comments": post.num_comments
                        })
                    
                    return {
                        "success": True,
                        "query": query,
                        "subreddit": subreddit,
                        "results": results[:limit],
                        "total_results": len(results),
                        "method": "reddit_api"
                    }
                    
                except ImportError:
                    logger.info("PRAW not available, falling back to public API")
                    pass
                except Exception as api_error:
                    logger.warning(f"Reddit API error: {api_error}")
                    pass
            
            # Fall back to using requests for public Reddit data
            return await self._search_reddit_public(query, subreddit, sort, time_filter, limit)
                
        except Exception as e:
            logger.exception("RedditSearchTool error")
            return {"success": False, "error": str(e)}
    
    async def _search_reddit_public(self, query: str, subreddit: str = None, 
                                   sort: str = "relevance", time_filter: str = "all", 
                                   limit: int = 5) -> Dict[str, Any]:
        """Fallback method using public Reddit JSON API."""
        try:
            import requests
            from urllib.parse import quote_plus
            import json
            from datetime import datetime
            
            # Build Reddit search URL
            if subreddit:
                base_url = f"https://www.reddit.com/r/{subreddit}/search.json"
                params = {
                    "q": query,
                    "restrict_sr": "on",
                    "sort": sort if sort != "relevance" else "top",
                    "t": time_filter,
                    "limit": min(limit, 25)
                }
            else:
                base_url = "https://www.reddit.com/search.json"
                params = {
                    "q": query,
                    "sort": sort if sort != "relevance" else "top", 
                    "t": time_filter,
                    "limit": min(limit, 25)
                }
            
            headers = {
                "User-Agent": "Demonology:v1.0 (by /u/demonology_user)"
            }
            
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            if "data" in data and "children" in data["data"]:
                for post_data in data["data"]["children"]:
                    post = post_data["data"]
                    results.append({
                        "title": post.get("title", ""),
                        "author": post.get("author", "[deleted]"),
                        "subreddit": post.get("subreddit", ""),
                        "score": post.get("score", 0),
                        "url": f"https://reddit.com{post.get('permalink', '')}",
                        "created_utc": datetime.fromtimestamp(post.get("created_utc", 0)).isoformat(),
                        "text": (post.get("selftext", "")[:500] + "...") if len(post.get("selftext", "")) > 500 else post.get("selftext", ""),
                        "num_comments": post.get("num_comments", 0)
                    })
            
            return {
                "success": True,
                "query": query,
                "subreddit": subreddit,
                "results": results[:limit],
                "total_results": len(results),
                "method": "public_api"
            }
            
        except Exception as e:
            logger.exception("Reddit public search error")
            return {"success": False, "error": f"Reddit search failed: {str(e)}"}


# ---------------------------------------------------------------------
# Image Generation Tool
# ---------------------------------------------------------------------

class ImageGenerationTool(Tool):
    """Generate images from text descriptions using free AI image generation APIs."""
    
    def __init__(self, safe_root: Optional[Path] = None):
        super().__init__("image_generation", "Generate images from text descriptions")
        self.safe_root = safe_root or SAFE_ROOT
    
    def is_available(self) -> bool:
        try:
            import requests
            return True
        except ImportError:
            return False
    
    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Text description of the image to generate"},
                    "content_type": {"type": "string", "enum": ["auto", "texture", "character", "object", "scene"], "description": "Type of image content for better prompt optimization", "default": "auto"},
                    "style": {"type": "string", "enum": ["realistic", "artistic", "anime", "fantasy", "pixel-art", "concept-art"], "description": "Image style", "default": "realistic"},
                    "size": {"type": "string", "enum": ["512x512", "768x768", "1024x1024"], "description": "Image dimensions", "default": "512x512"},
                    "filename": {"type": "string", "description": "Filename for saved image (optional)"},
                    "save_image": {"type": "boolean", "description": "Whether to save image to disk", "default": True}
                },
                "required": ["prompt"]
            }
        }
    
    async def execute(self, prompt: str, content_type: str = "auto", style: str = "realistic", size: str = "512x512", 
                     filename: str = None, save_image: bool = True, **kwargs) -> Dict[str, Any]:
        try:
            import requests
            import base64
            from datetime import datetime
            import hashlib
            
            # Enhance prompt based on style and content type
            enhanced_prompt = self._enhance_prompt(prompt, style, content_type)
            
            # Try multiple APIs in order of preference
            apis = [
                self._try_pollinations_ai,
                self._try_huggingface_api,
                self._try_craiyon_api
            ]
            
            image_data = None
            api_used = None
            
            for api_func in apis:
                try:
                    result = await api_func(enhanced_prompt, size)
                    if result:
                        image_data, api_used = result
                        break
                except Exception as e:
                    logger.warning(f"API {api_func.__name__} failed: {e}")
                    continue
            
            if not image_data:
                return {"success": False, "error": "All image generation APIs failed"}
            
            # Generate filename if not provided
            if not filename:
                # Create hash of prompt for unique filename
                prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_image_{timestamp}_{prompt_hash}.png"
            
            if not filename.endswith(('.png', '.jpg', '.jpeg')):
                filename += '.png'
            
            result = {
                "success": True,
                "prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "style": style,
                "size": size,
                "api_used": api_used,
                "filename": filename
            }
            
            if save_image:
                # Save image to current working directory
                try:
                    # Use current working directory instead of safe_root
                    current_dir = Path.cwd().resolve()
                    image_path = current_dir / filename
                except Exception:
                    # Fallback to safe_root if current directory fails
                    image_path = self.safe_root / filename
                
                if isinstance(image_data, bytes):
                    # Direct binary data
                    image_path.write_bytes(image_data)
                elif isinstance(image_data, str):
                    # Base64 encoded data
                    if image_data.startswith('data:image'):
                        # Remove data:image/png;base64, prefix
                        image_data = image_data.split(',')[1]
                    decoded_data = base64.b64decode(image_data)
                    image_path.write_bytes(decoded_data)
                
                result["file_path"] = str(image_path)
                result["file_size"] = image_path.stat().st_size
                result["message"] = f"Image generated and saved to {filename}"
            else:
                result["image_data"] = image_data
                result["message"] = "Image generated successfully"
            
            return result
            
        except Exception as e:
            logger.exception("ImageGenerationTool error")
            return {"success": False, "error": str(e)}
    
    def _enhance_prompt(self, prompt: str, style: str, content_type: str = "auto") -> str:
        """Enhance the prompt based on the selected style and content type."""
        # Determine content type
        if content_type == "auto":
            # Auto-detect what type of image is being requested
            prompt_lower = prompt.lower()
            
            is_texture = any(word in prompt_lower for word in [
                'texture', 'pattern', 'material', 'surface', 'seamless', 'tileable',
                'wood grain', 'fabric', 'metal', 'stone', 'concrete', 'marble'
            ])
            
            is_character = any(word in prompt_lower for word in [
                'person', 'character', 'man', 'woman', 'face', 'portrait', 'figure',
                'human', 'people', 'body', 'head', 'eyes'
            ])
            
            is_object = any(word in prompt_lower for word in [
                'object', 'item', 'tool', 'weapon', 'vehicle', 'furniture', 'product'
            ])
            
            is_scene = any(word in prompt_lower for word in [
                'scene', 'landscape', 'environment', 'room', 'building', 'place', 
                'location', 'background', 'setting'
            ])
            
            # Determine the content type based on detection
            if is_texture:
                detected_type = 'texture'
            elif is_character:
                detected_type = 'character'
            elif is_object:
                detected_type = 'object'
            elif is_scene:
                detected_type = 'scene'
            else:
                detected_type = 'general'
        else:
            detected_type = content_type
        
        # Base style prefixes
        style_prefixes = {
            "realistic": "photorealistic, high quality, detailed, ",
            "artistic": "artistic, painterly, creative, ",
            "anime": "anime style, manga, colorful, ",
            "fantasy": "fantasy art, magical, mystical, ",
            "pixel-art": "pixel art, 8-bit style, retro gaming, ",
            "concept-art": "concept art, digital painting, professional, "
        }
        
        # Content-specific suffixes to avoid unwanted elements
        content_suffixes = {
            'texture': ", seamless pattern, isolated on neutral background, no objects, no people, no scenes, tileable, material study",
            'character': ", isolated character, plain background, no environment, no scenes, character focus, portrait style",
            'object': ", isolated object, plain background, no people, no scenes, product shot, clean composition",
            'scene': ", environmental art, atmospheric, detailed setting",
            'general': ""
        }
        
        # Build enhanced prompt
        prefix = style_prefixes.get(style, "")
        enhanced = f"{prefix}{prompt}"
        
        # Add content-specific instructions
        if detected_type in content_suffixes:
            enhanced += content_suffixes[detected_type]
        
        # Add quality modifiers
        enhanced += ", high resolution, professional quality"
        
        return enhanced
    
    async def _try_pollinations_ai(self, prompt: str, size: str) -> Optional[Tuple[bytes, str]]:
        """Try Pollinations.ai API - completely free, no API key needed."""
        import requests
        from urllib.parse import quote
        
        # Pollinations.ai supports various models
        model = "flux"  # or "flux-realism", "flux-3d", etc.
        
        # Parse size
        width, height = size.split('x')
        
        # Build URL
        url = f"https://image.pollinations.ai/prompt/{quote(prompt)}?width={width}&height={height}&model={model}&nologo=true"
        
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        return response.content, "Pollinations.ai"
    
    async def _try_huggingface_api(self, prompt: str, size: str) -> Optional[Tuple[bytes, str]]:
        """Try Hugging Face Inference API - free with HF token."""
        import requests
        import os
        
        hf_token = os.environ.get('HUGGINGFACE_TOKEN')
        if not hf_token:
            logger.info("No Hugging Face token found, skipping HF API")
            return None
        
        # Use FLUX.1 schnell model (fast and free)
        api_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
        
        headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "num_inference_steps": 4,  # FLUX schnell works well with 1-4 steps
                "guidance_scale": 0.0,     # FLUX schnell works better without guidance
            }
        }
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        return response.content, "Hugging Face FLUX.1"
    
    async def _try_craiyon_api(self, prompt: str, size: str) -> Optional[Tuple[str, str]]:
        """Try unofficial Craiyon API - free but may be rate limited."""
        import requests
        import json
        import time
        
        # This is a simplified version - the actual Craiyon API is more complex
        # and may require reverse engineering their web interface
        try:
            # Craiyon web interface endpoint (unofficial)
            url = "https://api.craiyon.com/v3"
            
            payload = {
                "prompt": prompt,
                "model": "art",  # or "drawing", "photo"
                "negative_prompt": "blurry, low quality, distorted",
                "version": "35s5hfwn9n78gb06",  # This may need to be updated
            }
            
            response = requests.post(url, json=payload, timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                if "images" in data and data["images"]:
                    # Return first image (base64 encoded)
                    return data["images"][0], "Craiyon"
            
            return None
            
        except Exception as e:
            logger.warning(f"Craiyon API failed: {e}")
            return None


# ---------------------------------------------------------------------
# Image Analysis Tool
# ---------------------------------------------------------------------

class ImageAnalysisTool(Tool):
    """Analyze images including screenshots, diagrams, code snippets, and UI mockups."""
    
    def __init__(self, safe_root: Optional[Path] = None):
        super().__init__("image_analysis", "Analyze and describe images including screenshots, diagrams, UI mockups, and visual content")
        self.safe_root = safe_root or SAFE_ROOT
    
    def is_available(self) -> bool:
        """Check if required image processing libraries are available."""
        try:
            from PIL import Image
            return True
        except ImportError:
            return False
    
    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the image file to analyze"
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["general", "ui_mockup", "code_snippet", "diagram", "screenshot", "text_extraction"],
                        "description": "Type of analysis to perform",
                        "default": "general"
                    },
                    "extract_text": {
                        "type": "boolean",
                        "description": "Whether to attempt OCR text extraction",
                        "default": True
                    },
                    "save_analysis": {
                        "type": "boolean", 
                        "description": "Whether to save analysis results to a text file",
                        "default": True
                    }
                },
                "required": ["image_path"]
            }
        }
    
    async def execute(self, image_path: str, analysis_type: str = "general", 
                     extract_text: bool = True, save_analysis: bool = True, **kwargs) -> Dict[str, Any]:
        try:
            # Resolve and validate image path
            img_path = Path(image_path).resolve()
            
            # Check if path exists
            if not img_path.exists():
                return {"success": False, "error": f"Image file not found: {image_path}"}
            
            if not img_path.is_file():
                return {"success": False, "error": f"Path is not a file: {image_path}"}
            
            # Check if it's an image file
            valid_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.ico'}
            if img_path.suffix.lower() not in valid_extensions:
                return {"success": False, "error": f"Unsupported image format: {img_path.suffix}"}
            
            # Perform image analysis
            analysis_result = await self._analyze_image(img_path, analysis_type, extract_text)
            
            result = {
                "success": True,
                "image_path": str(img_path),
                "image_name": img_path.name,
                "analysis_type": analysis_type,
                "analysis": analysis_result
            }
            
            if save_analysis:
                # Save analysis to text file
                analysis_file = img_path.with_suffix('.analysis.txt')
                analysis_content = self._format_analysis_report(analysis_result, img_path.name, analysis_type)
                analysis_file.write_text(analysis_content, encoding='utf-8')
                result["analysis_file"] = str(analysis_file)
            
            return result
            
        except Exception as e:
            logger.exception("ImageAnalysisTool error")
            return {"success": False, "error": f"Image analysis failed: {str(e)}"}
    
    async def _analyze_image(self, img_path: Path, analysis_type: str, extract_text: bool) -> Dict[str, Any]:
        """Perform comprehensive image analysis."""
        analysis = {
            "basic_info": {},
            "visual_description": "",
            "extracted_text": "",
            "technical_details": {},
            "suggestions": []
        }
        
        try:
            from PIL import Image
            
            # Open and analyze image
            with Image.open(img_path) as img:
                # Basic image information
                analysis["basic_info"] = {
                    "dimensions": f"{img.width}x{img.height}",
                    "format": img.format,
                    "mode": img.mode,
                    "file_size": img_path.stat().st_size,
                    "has_transparency": img.mode in ('RGBA', 'LA') or 'transparency' in img.info
                }
                
                # Convert to RGB for analysis if needed
                if img.mode != 'RGB':
                    img_rgb = img.convert('RGB')
                else:
                    img_rgb = img
                
                # Analyze image characteristics
                analysis["technical_details"] = self._analyze_image_characteristics(img_rgb)
                
                # Generate visual description based on analysis type
                analysis["visual_description"] = self._generate_visual_description(img_rgb, analysis_type)
                
                # Extract text if requested
                if extract_text:
                    analysis["extracted_text"] = self._extract_text_from_image(img_rgb)
                
                # Generate suggestions based on analysis type
                analysis["suggestions"] = self._generate_suggestions(analysis_type, analysis)
        
        except ImportError:
            analysis["error"] = "PIL (Pillow) library not available for image analysis"
        except Exception as e:
            analysis["error"] = f"Image analysis failed: {str(e)}"
        
        return analysis
    
    def _analyze_image_characteristics(self, img: 'Image.Image') -> Dict[str, Any]:
        """Analyze technical characteristics of the image."""
        import statistics
        
        characteristics = {}
        
        try:
            # Color analysis
            colors = img.getcolors(maxcolors=256*256*256)
            if colors:
                # Dominant colors
                dominant_colors = sorted(colors, key=lambda x: x[0], reverse=True)[:5]
                characteristics["dominant_colors"] = [
                    {"rgb": color[1], "count": color[0]} for color in dominant_colors
                ]
            
            # Brightness analysis
            grayscale = img.convert('L')
            pixels = list(grayscale.getdata())
            characteristics["brightness"] = {
                "average": statistics.mean(pixels),
                "median": statistics.median(pixels),
                "min": min(pixels),
                "max": max(pixels)
            }
            
            # Detect if image appears to be a screenshot
            characteristics["likely_screenshot"] = self._detect_screenshot_characteristics(img)
            
            # Detect if image contains UI elements
            characteristics["likely_ui"] = self._detect_ui_characteristics(img)
            
        except Exception as e:
            characteristics["analysis_error"] = str(e)
        
        return characteristics
    
    def _detect_screenshot_characteristics(self, img: 'Image.Image') -> Dict[str, Any]:
        """Detect characteristics that suggest this is a screenshot."""
        screenshot_indicators = {
            "has_typical_screenshot_ratio": False,
            "has_ui_like_regions": False,
            "has_text_regions": False,
            "confidence": "low"
        }
        
        width, height = img.size
        ratio = width / height
        
        # Common screenshot ratios
        common_ratios = [16/9, 16/10, 4/3, 3/2, 21/9]
        screenshot_indicators["has_typical_screenshot_ratio"] = any(
            abs(ratio - r) < 0.1 for r in common_ratios
        )
        
        # Look for rectangular regions (potential UI elements)
        # This is a simplified heuristic
        if width > 800 and height > 600:
            screenshot_indicators["has_ui_like_regions"] = True
        
        # Estimate confidence
        score = 0
        if screenshot_indicators["has_typical_screenshot_ratio"]:
            score += 1
        if screenshot_indicators["has_ui_like_regions"]:
            score += 1
        
        if score >= 2:
            screenshot_indicators["confidence"] = "high"
        elif score == 1:
            screenshot_indicators["confidence"] = "medium"
        
        return screenshot_indicators
    
    def _detect_ui_characteristics(self, img: 'Image.Image') -> Dict[str, Any]:
        """Detect characteristics that suggest this contains UI elements."""
        ui_indicators = {
            "has_rectangular_regions": False,
            "has_button_like_elements": False,
            "has_form_elements": False,
            "confidence": "low"
        }
        
        width, height = img.size
        
        # Simple heuristics for UI detection
        # Look for common UI dimensions and aspect ratios
        if width > 300 and height > 200:
            ui_indicators["has_rectangular_regions"] = True
        
        # More sophisticated UI detection would require edge detection
        # For now, use basic heuristics
        ui_indicators["confidence"] = "medium" if ui_indicators["has_rectangular_regions"] else "low"
        
        return ui_indicators
    
    def _generate_visual_description(self, img: 'Image.Image', analysis_type: str) -> str:
        """Generate a visual description based on the analysis type."""
        width, height = img.size
        
        descriptions = {
            "general": f"Image with dimensions {width}x{height} pixels. ",
            "ui_mockup": f"UI mockup or interface design with dimensions {width}x{height}. ",
            "code_snippet": f"Image containing code or text content, {width}x{height} pixels. ",
            "diagram": f"Diagram or schematic image, {width}x{height} pixels. ",
            "screenshot": f"Screenshot capture with dimensions {width}x{height}. ",
            "text_extraction": f"Text-containing image, {width}x{height} pixels. "
        }
        
        base_description = descriptions.get(analysis_type, descriptions["general"])
        
        # Add more details based on image characteristics
        if width > 1920:
            base_description += "High resolution image. "
        elif width < 500:
            base_description += "Small or thumbnail-sized image. "
        
        aspect_ratio = width / height
        if aspect_ratio > 2:
            base_description += "Wide aspect ratio, possibly a banner or header. "
        elif aspect_ratio < 0.5:
            base_description += "Tall aspect ratio, possibly a mobile screen or sidebar. "
        
        return base_description
    
    def _extract_text_from_image(self, img: 'Image.Image') -> str:
        """Extract text from image using OCR if available."""
        try:
            import pytesseract
            
            # Extract text using Tesseract OCR
            extracted_text = pytesseract.image_to_string(img)
            return extracted_text.strip()
            
        except ImportError:
            return "OCR not available (pytesseract not installed)"
        except Exception as e:
            return f"OCR failed: {str(e)}"
    
    def _generate_suggestions(self, analysis_type: str, analysis: Dict[str, Any]) -> List[str]:
        """Generate suggestions based on the analysis type and results."""
        suggestions = []
        
        basic_info = analysis.get("basic_info", {})
        width, height = basic_info.get("dimensions", "0x0").split('x')
        width, height = int(width), int(height)
        
        if analysis_type == "ui_mockup":
            suggestions.extend([
                "Consider creating interactive prototypes based on this mockup",
                "Document the UI components and their specifications",
                "Validate the design with accessibility guidelines"
            ])
            
        elif analysis_type == "code_snippet":
            if analysis.get("extracted_text"):
                suggestions.extend([
                    "Review the extracted code for syntax and best practices",
                    "Consider converting the image to actual code files",
                    "Check for any visible errors or improvements"
                ])
            
        elif analysis_type == "screenshot":
            suggestions.extend([
                "Document the context and purpose of this screenshot",
                "Consider annotating important areas for clarity",
                "Use for bug reports or feature documentation"
            ])
            
        elif analysis_type == "diagram":
            suggestions.extend([
                "Convert to editable diagram format if needed",
                "Ensure all text and labels are readable",
                "Consider creating digital version for easier editing"
            ])
        
        # General suggestions based on technical characteristics
        if width < 800 and height < 600:
            suggestions.append("Image resolution is relatively low - consider higher resolution for better clarity")
        
        if analysis.get("extracted_text") and len(analysis["extracted_text"]) > 50:
            suggestions.append("Significant text content detected - consider extracting for documentation")
        
        return suggestions
    
    def _format_analysis_report(self, analysis: Dict[str, Any], filename: str, analysis_type: str) -> str:
        """Format the analysis results into a readable report."""
        report = f"""# Image Analysis Report: {filename}

## Basic Information
- **Analysis Type**: {analysis_type}
- **Dimensions**: {analysis.get('basic_info', {}).get('dimensions', 'Unknown')}
- **Format**: {analysis.get('basic_info', {}).get('format', 'Unknown')}
- **File Size**: {analysis.get('basic_info', {}).get('file_size', 'Unknown')} bytes
- **Color Mode**: {analysis.get('basic_info', {}).get('mode', 'Unknown')}

## Visual Description
{analysis.get('visual_description', 'No description available')}

## Technical Details
"""
        
        tech_details = analysis.get('technical_details', {})
        if tech_details.get('brightness'):
            brightness = tech_details['brightness']
            report += f"""
### Color and Brightness Analysis
- **Average Brightness**: {brightness.get('average', 0):.1f}/255
- **Brightness Range**: {brightness.get('min', 0)} - {brightness.get('max', 255)}
"""
        
        if tech_details.get('dominant_colors'):
            report += "\n### Dominant Colors\n"
            for i, color_info in enumerate(tech_details['dominant_colors'][:3], 1):
                rgb = color_info['rgb']
                if isinstance(rgb, tuple) and len(rgb) == 3:
                    report += f"- **Color {i}**: RGB({rgb[0]}, {rgb[1]}, {rgb[2]}) - {color_info['count']} pixels\n"
        
        # Add extracted text if available
        extracted_text = analysis.get('extracted_text', '')
        if extracted_text and len(extracted_text.strip()) > 0:
            report += f"""
## Extracted Text
```
{extracted_text}
```
"""
        
        # Add suggestions
        suggestions = analysis.get('suggestions', [])
        if suggestions:
            report += "\n## Suggestions\n"
            for suggestion in suggestions:
                report += f"- {suggestion}\n"
        
        report += f"""
---
*Analysis generated by Demonology Image Analysis Tool*
*Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report


# ---------------------------------------------------------------------
# Ghidra Headless Analysis Tool
# ---------------------------------------------------------------------

class GhidraAnalysisTool(Tool):
    def __init__(self, safe_root: Optional[Path] = None):
        super().__init__("ghidra_analysis", "Perform headless binary analysis using Ghidra.")
        self.safe_root = safe_root or SAFE_ROOT
    
    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "binary_path": {
                        "type": "string", 
                        "description": "Path to the binary file to analyze"
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["basic", "functions", "strings", "imports", "exports", "decompile", "full"],
                        "description": "Type of analysis to perform",
                        "default": "basic"
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["text", "json", "xml"],
                        "description": "Output format for analysis results",
                        "default": "text"
                    },
                    "script_path": {
                        "type": "string",
                        "description": "Optional path to custom Ghidra script to run"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Analysis timeout in seconds",
                        "default": 300
                    }
                },
                "required": ["binary_path"]
            }
        }
    
    def is_available(self) -> bool:
        """Check if Ghidra is available in the system."""
        # Check for ghidra in common locations
        ghidra_paths = [
            "/opt/ghidra/support/analyzeHeadless",
            "/usr/local/ghidra/support/analyzeHeadless",
            shutil.which("analyzeHeadless")
        ]
        
        for path in ghidra_paths:
            if path and Path(path).exists():
                return True
        
        # Check environment variable
        ghidra_install = os.environ.get("GHIDRA_INSTALL_DIR")
        if ghidra_install:
            headless_path = Path(ghidra_install) / "support" / "analyzeHeadless"
            if headless_path.exists():
                return True
        
        return False
    
    def _find_ghidra_headless(self) -> Optional[str]:
        """Find the Ghidra headless analyzer executable."""
        # Try common paths
        common_paths = [
            "/opt/ghidra/support/analyzeHeadless",
            "/usr/local/ghidra/support/analyzeHeadless",
            "/Applications/ghidra/support/analyzeHeadless"  # macOS
        ]
        
        for path in common_paths:
            if Path(path).exists():
                return path
        
        # Try which command
        which_result = shutil.which("analyzeHeadless")
        if which_result:
            return which_result
        
        # Check environment variable
        ghidra_install = os.environ.get("GHIDRA_INSTALL_DIR")
        if ghidra_install:
            headless_path = Path(ghidra_install) / "support" / "analyzeHeadless"
            if headless_path.exists():
                return str(headless_path)
        
        return None
    
    async def execute(self, binary_path: str, analysis_type: str = "basic", 
                     output_format: str = "text", script_path: Optional[str] = None,
                     timeout: int = 300, **_) -> Dict[str, Any]:
        try:
            # Find Ghidra headless analyzer
            ghidra_headless = self._find_ghidra_headless()
            if not ghidra_headless:
                return {
                    "success": False,
                    "error": "Ghidra headless analyzer not found. Please install Ghidra or set GHIDRA_INSTALL_DIR environment variable."
                }
            
            # Validate binary path
            binary_file = _safe_path(binary_path)
            if not binary_file.exists():
                return {"success": False, "error": f"Binary file not found: {binary_path}"}
            
            if not binary_file.is_file():
                return {"success": False, "error": f"Path is not a file: {binary_path}"}
            
            # Create temporary project directory
            import tempfile
            with tempfile.TemporaryDirectory(prefix="ghidra_analysis_") as temp_dir:
                temp_path = Path(temp_dir)
                project_dir = temp_path / "project"
                project_name = "analysis_project"
                
                # Prepare Ghidra command
                cmd = [
                    ghidra_headless,
                    str(project_dir),
                    project_name,
                    "-import", str(binary_file),
                    "-noanalysis" if analysis_type == "basic" else "-analyse"
                ]
                
                # Add output format options
                if output_format == "json":
                    cmd.extend(["-postScript", "ExportJson.py"])
                elif output_format == "xml":
                    cmd.extend(["-postScript", "ExportXml.py"])
                
                # Add custom script if provided
                if script_path:
                    script_file = _safe_path(script_path)
                    if script_file.exists():
                        cmd.extend(["-postScript", str(script_file)])
                
                # Add analysis-specific options
                if analysis_type == "functions":
                    cmd.extend(["-scriptPath", str(temp_path / "scripts"), "-postScript", "ListFunctions.py"])
                elif analysis_type == "strings":
                    cmd.extend(["-scriptPath", str(temp_path / "scripts"), "-postScript", "ListStrings.py"])
                elif analysis_type == "imports":
                    cmd.extend(["-scriptPath", str(temp_path / "scripts"), "-postScript", "ListImports.py"])
                elif analysis_type == "exports":
                    cmd.extend(["-scriptPath", str(temp_path / "scripts"), "-postScript", "ListExports.py"])
                elif analysis_type == "decompile":
                    cmd.extend(["-scriptPath", str(temp_path / "scripts"), "-postScript", "DecompileAll.py"])
                
                # Create basic analysis scripts if they don't exist
                await self._create_analysis_scripts(temp_path, analysis_type)
                
                # Execute Ghidra headless analysis
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self.safe_root)
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(), 
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    return {
                        "success": False,
                        "error": f"Analysis timed out after {timeout} seconds"
                    }
                
                stdout_text = stdout.decode('utf-8', errors='replace')
                stderr_text = stderr.decode('utf-8', errors='replace')
                
                # Parse results based on analysis type
                results = await self._parse_analysis_results(
                    stdout_text, stderr_text, analysis_type, temp_path
                )
                
                return {
                    "success": process.returncode == 0,
                    "binary": str(binary_file),
                    "analysis_type": analysis_type,
                    "output_format": output_format,
                    "results": results,
                    "stdout": stdout_text[-1000:] if len(stdout_text) > 1000 else stdout_text,
                    "stderr": stderr_text[-500:] if len(stderr_text) > 500 else stderr_text,
                    "return_code": process.returncode
                }
                
        except Exception as e:
            logger.exception("Ghidra analysis failed")
            return {"success": False, "error": f"Analysis failed: {str(e)}"}
    
    async def _create_analysis_scripts(self, temp_path: Path, analysis_type: str):
        """Create basic Ghidra analysis scripts."""
        scripts_dir = temp_path / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Basic function listing script
        if analysis_type in ["functions", "full"]:
            func_script = scripts_dir / "ListFunctions.py"
            func_script.write_text('''
# List all functions in the binary
from ghidra.program.model.listing import *

program = getCurrentProgram()
listing = program.getListing()
functions = listing.getFunctions(True)

print("=== FUNCTIONS ===")
for func in functions:
    print("Function: {} at {}".format(func.getName(), func.getEntryPoint()))
    print("  Size: {} bytes".format(func.getBody().getNumAddresses()))
    print("  Parameters: {}".format(func.getParameterCount()))
    print("")
''')
        
        # String listing script
        if analysis_type in ["strings", "full"]:
            strings_script = scripts_dir / "ListStrings.py"
            strings_script.write_text('''
# List all strings in the binary
from ghidra.program.model.data import *

program = getCurrentProgram()
listing = program.getListing()
memory = program.getMemory()

print("=== STRINGS ===")
data_iter = listing.getDefinedData(True)
for data in data_iter:
    if data.hasStringValue():
        print("String at {}: {}".format(data.getAddress(), data.getValue()))
''')
    
    async def _parse_analysis_results(self, stdout: str, stderr: str, 
                                    analysis_type: str, temp_path: Path) -> Dict[str, Any]:
        """Parse Ghidra analysis results."""
        results = {
            "analysis_type": analysis_type,
            "summary": {},
            "details": []
        }
        
        # Extract basic information from stdout
        lines = stdout.split('\n')
        
        # Look for common Ghidra output patterns
        functions_found = []
        strings_found = []
        imports_found = []
        exports_found = []
        
        current_section = None
        for line in lines:
            line = line.strip()
            
            if "=== FUNCTIONS ===" in line:
                current_section = "functions"
                continue
            elif "=== STRINGS ===" in line:
                current_section = "strings"
                continue
            elif "Function:" in line and current_section == "functions":
                functions_found.append(line)
            elif "String at" in line and current_section == "strings":
                strings_found.append(line)
            elif "INFO" in line and "functions" in line.lower():
                # Extract function count from info messages
                import re
                match = re.search(r'(\d+)\s+functions?', line)
                if match:
                    results["summary"]["function_count"] = int(match.group(1))
        
        # Add parsed data to results
        if functions_found:
            results["details"].append({
                "type": "functions",
                "count": len(functions_found),
                "items": functions_found[:50]  # Limit output
            })
        
        if strings_found:
            results["details"].append({
                "type": "strings", 
                "count": len(strings_found),
                "items": strings_found[:50]  # Limit output
            })
        
        # Extract summary statistics
        if "Analysis complete" in stdout:
            results["summary"]["status"] = "completed"
        elif "Error" in stderr or "Exception" in stderr:
            results["summary"]["status"] = "error"
        else:
            results["summary"]["status"] = "partial"
        
        return results


# ---------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------

class ToolRegistry:
    def __init__(self, safe_root: Optional[Path] = None):
        self.tools: Dict[str, Tool] = {}
        logger.info(f"ToolRegistry initializing with safe_root: {safe_root}")
        self._register_default_tools(safe_root)

    def _register_default_tools(self, safe_root: Optional[Path]) -> None:
        self.register_tool(FileOperationsTool(safe_root=safe_root))
        self.register_tool(CodebaseAnalysisTool(safe_root=safe_root))
        self.register_tool(CodeExecutionTool(safe_root=safe_root))
        self.register_tool(WebSearchTool())
        self.register_tool(ProjectPlanningTool(safe_root=safe_root))
        self.register_tool(RedditSearchTool())
        self.register_tool(ImageGenerationTool(safe_root=safe_root))
        self.register_tool(ImageAnalysisTool(safe_root=safe_root))
        self.register_tool(GhidraAnalysisTool(safe_root=safe_root))

    def register_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def list_tools(self) -> List[Dict[str, Any]]:
        out = []
        for t in self.tools.values():
            out.append({
                "name": t.name,
                "description": t.description,
                "enabled": t.enabled,
                "available": t.is_available(),
                "parameters": t.to_openai_function().get("parameters", {}).get("properties", {})
            })
        return out

    def get_tool(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def to_openai_tools_format(self) -> List[Dict[str, Any]]:
        payload = []
        for t in self.tools.values():
            if not t.enabled or not t.is_available():
                continue
            payload.append({"type": "function", "function": t.to_openai_function()})
        return payload

    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        tool = self.get_tool(tool_name)
        if not tool:
            return {"success": False, "error": f"Tool not found: {tool_name}"}
        if not tool.enabled:
            return {"success": False, "error": f"Tool is disabled: {tool_name}"}
        if not tool.is_available():
            return {"success": False, "error": f"Tool is not available: {tool_name}"}
        try:
            # Handle FileOperationsTool's positional operation parameter
            if tool_name == "file_operations":
                logger.debug(f"file_operations called with kwargs: {kwargs}")
                if "operation" in kwargs:
                    operation = kwargs.pop("operation")
                    return await tool.execute(operation, **kwargs)
                else:
                    logger.error(f"file_operations called without 'operation' parameter. Available keys: {list(kwargs.keys())}")
                    return {"success": False, "error": "Missing required 'operation' parameter"}
            # Handle other tools with positional parameters
            elif tool_name == "web_search":
                if "query" in kwargs:
                    query = kwargs.pop("query")
                    return await tool.execute(query, **kwargs)
                else:
                    return {"success": False, "error": "Missing required 'query' parameter"}
            elif tool_name == "project_planning":
                if "action" in kwargs and "project_name" in kwargs:
                    action = kwargs.pop("action")
                    project_name = kwargs.pop("project_name")
                    project_description = kwargs.pop("project_description", "")
                    return await tool.execute(action=action, project_name=project_name, project_description=project_description, **kwargs)
                else:
                    return {"success": False, "error": "Missing required 'action' and/or 'project_name' parameters"}
            elif tool_name == "reddit_search":
                if "query" in kwargs:
                    query = kwargs.pop("query")
                    return await tool.execute(query, **kwargs)
                else:
                    return {"success": False, "error": "Missing required 'query' parameter"}
            elif tool_name == "image_generation":
                if "prompt" in kwargs:
                    prompt = kwargs.pop("prompt")
                    content_type = kwargs.pop("content_type", "auto")
                    return await tool.execute(prompt, content_type=content_type, **kwargs)
                else:
                    return {"success": False, "error": "Missing required 'prompt' parameter"}
            elif tool_name == "image_analysis":
                if "image_path" in kwargs:
                    image_path = kwargs.pop("image_path")
                    return await tool.execute(image_path, **kwargs)
                else:
                    return {"success": False, "error": "Missing required 'image_path' parameter"}
            elif tool_name == "ghidra_analysis":
                if "binary_path" in kwargs:
                    binary_path = kwargs.pop("binary_path")
                    return await tool.execute(binary_path, **kwargs)
                else:
                    return {"success": False, "error": "Missing required 'binary_path' parameter"}
            else:
                return await tool.execute(**kwargs)
        except Exception as e:
            logger.exception("Tool execution failed: %s", tool_name)
            return {"success": False, "error": f"Tool execution failed: {e}", "tool": tool_name}

