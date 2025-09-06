# demonology/tools/codebase.py
from __future__ import annotations

import io
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .base import Tool

# Workspace confinement
WORKSPACE_ROOT = Path(os.environ.get("DEMONOLOGY_ROOT", os.getcwd())).resolve()

# Reasonable project excludes
DEFAULT_EXCLUDES = {
    ".git", ".hg", ".svn", ".DS_Store",
    "__pycache__", ".pytest_cache", ".mypy_cache",
    "node_modules", "dist", "build", ".venv", "venv", ".tox",
    ".idea", ".vscode", ".cache", ".coverage", "coverage", "target",
}

# Soft limits
DEFAULT_MAX_MATCHES = 500
DEFAULT_MAX_FILE_BYTES = 1_000_000       # 1 MB per file read cap (index/peek)
DEFAULT_MAX_TOTAL_BYTES = 50_000_000     # 50 MB cap across entire walk
DEFAULT_MAX_FILES = 25_000               # max files visited
READ_CHUNK_BYTES = 64 * 1024             # streaming chunk size for grep


def _confine(path: Path) -> Path:
    """Resolve path and ensure it remains under WORKSPACE_ROOT; disallow symlink traversal."""
    p = (WORKSPACE_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        p.relative_to(WORKSPACE_ROOT)
    except Exception:
        raise PermissionError(f"Path escapes workspace root: {p}")
    # Disallow symlink traversal for target and parents (best-effort)
    for parent in [p] + list(p.parents):
        try:
            if parent.is_symlink():
                raise PermissionError(f"Symlinked path not allowed: {parent}")
        except FileNotFoundError:
            # Missing parent during creation is okay
            pass
    return p


def _is_probably_binary(sample: bytes) -> bool:
    """Heuristic: treat as binary if NUL present or too many non-text bytes."""
    if not sample:
        return False
    sample = sample[:512]
    if b"\x00" in sample:
        return True
    # printable ASCII + common control chars
    text_set = set(range(7, 14)) | {27} | set(range(0x20, 0x7F))
    nontext = sum(1 for b in sample if b not in text_set)
    return (nontext / len(sample)) > 0.30


def _should_exclude(path: Path, excludes: Iterable[str], include_hidden: bool) -> bool:
    name = path.name
    if not include_hidden and name.startswith("."):
        return True
    if name in excludes:
        return True
    return False


@dataclass
class GrepOptions:
    regex: bool = True
    case_sensitive: bool = True
    include_hidden: bool = False
    include_extensions: Optional[List[str]] = None   # e.g. [".py", ".ts"]
    exclude_extensions: Optional[List[str]] = None
    max_matches: int = DEFAULT_MAX_MATCHES


class CodebaseAnalysisTool(Tool):
    """Inspect and query codebases safely within a workspace root."""

    def __init__(self):
        super().__init__(
            "codebase_analysis",
            "Explore files in the workspace: tree, grep, index. Scoped to the workspace root."
        )

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["tree", "grep", "index"],
                        "description": "Action to perform"
                    },
                    "path": {"type": "string", "description": "Directory or file to operate on"},
                    "pattern": {"type": "string", "description": "Pattern for grep (regex by default)"},
                    "regex": {"type": "boolean", "description": "Pattern is regex (default true)"},
                    "case_sensitive": {"type": "boolean", "description": "Case-sensitive match (default true)"},
                    "include_hidden": {"type": "boolean", "description": "Include dotfiles/dirs"},
                    "include_extensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Only search files with these extensions (e.g. ['.py','.js'])"
                    },
                    "exclude_extensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Skip files with these extensions"
                    },
                    "depth": {"type": "integer", "description": "Max depth for tree (0 = only the dir itself)"},
                    "max_matches": {"type": "integer", "description": "Max matches to return (grep)"},
                    "max_file_bytes": {"type": "integer", "description": "Per-file read cap (index)"},
                    "max_total_bytes": {"type": "integer", "description": "Total read cap across repo (index/grep)"},
                    "max_files": {"type": "integer", "description": "Max files to visit"},
                },
                "required": ["operation", "path"]
            }
        }

    async def execute(self, **kwargs) -> Dict[str, Any]:
        op = (kwargs.get("operation") or "").strip().lower()
        base = _confine(Path(kwargs.get("path") or "."))

        try:
            if op == "tree":
                return self._tree(
                    base=base,
                    depth=kwargs.get("depth"),
                    include_hidden=bool(kwargs.get("include_hidden", False)),
                )

            if op == "grep":
                opts = GrepOptions(
                    regex=kwargs.get("regex", True),
                    case_sensitive=kwargs.get("case_sensitive", True),
                    include_hidden=bool(kwargs.get("include_hidden", False)),
                    include_extensions=kwargs.get("include_extensions"),
                    exclude_extensions=kwargs.get("exclude_extensions"),
                    max_matches=int(kwargs.get("max_matches") or DEFAULT_MAX_MATCHES),
                )
                return await self._grep(
                    base=base,
                    pattern_str=kwargs.get("pattern") or "",
                    opts=opts,
                    max_total_bytes=int(kwargs.get("max_total_bytes") or DEFAULT_MAX_TOTAL_BYTES),
                    max_files=int(kwargs.get("max_files") or DEFAULT_MAX_FILES),
                )

            if op == "index":
                return await self._index_repo(
                    base=base,
                    include_hidden=bool(kwargs.get("include_hidden", False)),
                    include_exts=kwargs.get("include_extensions"),
                    exclude_exts=kwargs.get("exclude_extensions"),
                    max_file_bytes=int(kwargs.get("max_file_bytes") or DEFAULT_MAX_FILE_BYTES),
                    max_total_bytes=int(kwargs.get("max_total_bytes") or DEFAULT_MAX_TOTAL_BYTES),
                    max_files=int(kwargs.get("max_files") or DEFAULT_MAX_FILES),
                )

            return {"success": False, "error": f"Unknown operation: {op}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ---------- tree ----------

    def _tree(self, base: Path, depth: Optional[int], include_hidden: bool) -> Dict[str, Any]:
        if not base.exists():
            return {"success": False, "error": f"Path not found: {base}"}

        max_depth = depth if (isinstance(depth, int) and depth >= 0) else None
        items: List[Dict[str, Any]] = []

        def _walk(dirpath: Path, current_depth: int):
            if max_depth is not None and current_depth > max_depth:
                return
            try:
                for entry in sorted(dirpath.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
                    if _should_exclude(entry, DEFAULT_EXCLUDES, include_hidden):
                        continue
                    rel = entry.relative_to(base if base.is_dir() else base.parent)
                    try:
                        st = entry.stat()
                    except FileNotFoundError:
                        continue
                    items.append({
                        "path": str(rel),
                        "is_dir": entry.is_dir(),
                        "is_file": entry.is_file(),
                        "size": st.st_size if entry.is_file() else None,
                        "modified_time": st.st_mtime,
                        "mode_octal": oct(st.st_mode)[-4:],
                    })
                    if entry.is_dir():
                        _walk(entry, current_depth + 1)
            except PermissionError:
                pass

        if base.is_dir():
            _walk(base, 0)
        else:
            # single file
            st = base.stat()
            items.append({
                "path": base.name,
                "is_dir": False,
                "is_file": True,
                "size": st.st_size,
                "modified_time": st.st_mtime,
                "mode_octal": oct(st.st_mode)[-4:],
            })

        return {"success": True, "operation": "tree", "root": str(base), "items": items, "count": len(items)}

    # ---------- grep ----------

    async def _grep(
        self,
        base: Path,
        pattern_str: str,
        opts: GrepOptions,
        max_total_bytes: int,
        max_files: int,
    ) -> Dict[str, Any]:
        if not pattern_str:
            return {"success": False, "error": "Missing 'pattern' for grep"}

        flags = 0
        if not opts.case_sensitive:
            flags |= re.IGNORECASE

        try:
            pattern = re.compile(pattern_str if opts.regex else re.escape(pattern_str), flags)
        except re.error as e:
            return {"success": False, "error": f"Invalid regex: {e}"}

        matches: List[Dict[str, Any]] = []
        visited_files = 0
        total_bytes = 0
        skipped_binaries = 0
        skipped_excluded = 0

        def want_ext(path: Path) -> bool:
            if opts.include_extensions:
                if path.suffix.lower() not in {e.lower() for e in opts.include_extensions}:
                    return False
            if opts.exclude_extensions:
                if path.suffix.lower() in {e.lower() for e in opts.exclude_extensions}:
                    return False
            return True

        def iter_files(root: Path) -> Iterable[Path]:
            if root.is_file():
                yield root
                return
            for dpath, dnames, fnames in os.walk(root, followlinks=False):
                d = Path(dpath)
                # prune directories in-place
                pruned = []
                for dn in list(dnames):
                    sub = d / dn
                    if _should_exclude(sub, DEFAULT_EXCLUDES, opts.include_hidden):
                        pruned.append(dn)
                for dn in pruned:
                    dnames.remove(dn)
                for fn in fnames:
                    fpath = d / fn
                    if _should_exclude(fpath, DEFAULT_EXCLUDES, opts.include_hidden):
                        continue
                    yield fpath

        base_rel_anchor = base if base.is_dir() else base.parent

        for f in iter_files(base):
            if visited_files >= max_files:
                break
            try:
                st = f.stat()
                if not stat.S_ISREG(st.st_mode):
                    continue
                if not want_ext(f):
                    skipped_excluded += 1
                    continue

                # quick binary sniff
                with open(f, "rb") as fh:
                    head = fh.read(512)
                    if _is_probably_binary(head):
                        skipped_binaries += 1
                        continue

                visited_files += 1

                # stream read line-by-line
                line_no = 0
                carried = ""
                with io.open(f, "r", encoding="utf-8", errors="replace") as fh:
                    while True:
                        chunk = fh.read(READ_CHUNK_BYTES)
                        if not chunk:
                            # process remaining carried line
                            if carried:
                                line_no += 1
                                self._maybe_match_line(matches, pattern, carried, f, base_rel_anchor)
                            break
                        total_bytes += len(chunk)
                        if total_bytes > max_total_bytes or len(matches) >= opts.max_matches:
                            break
                        chunk = carried + chunk
                        lines = chunk.splitlines(keepends=False)
                        if chunk and not chunk.endswith(("\n", "\r")):
                            carried = lines.pop() if lines else chunk
                        else:
                            carried = ""
                        for ln in lines:
                            line_no += 1
                            if len(matches) >= opts.max_matches:
                                break
                            self._maybe_match_line(matches, pattern, ln, f, base_rel_anchor)
            except (IOError, OSError):
                continue

            if total_bytes > max_total_bytes or len(matches) >= opts.max_matches:
                break

        return {
            "success": True,
            "operation": "grep",
            "root": str(base),
            "pattern": pattern_str,
            "regex": opts.regex,
            "case_sensitive": opts.case_sensitive,
            "matches": matches,
            "match_count": len(matches),
            "visited_files": visited_files,
            "skipped_binaries": skipped_binaries,
            "skipped_excluded": skipped_excluded,
            "truncated": len(matches) >= opts.max_matches or total_bytes >= max_total_bytes,
        }

    def _maybe_match_line(
        self,
        matches: List[Dict[str, Any]],
        pattern: re.Pattern,
        line: str,
        file_path: Path,
        base_rel_anchor: Path,
    ) -> None:
        m = pattern.search(line)
        if m:
            rel = file_path.relative_to(base_rel_anchor)
            matches.append({
                "file": str(rel),
                "line": line.strip(),
                "line_number": None,  # not tracking exact line number in streaming split; can add if needed
                "match_start": m.start(),
                "match_end": m.end(),
            })

    # ---------- index ----------

    async def _index_repo(
        self,
        base: Path,
        include_hidden: bool,
        include_exts: Optional[List[str]],
        exclude_exts: Optional[List[str]],
        max_file_bytes: int,
        max_total_bytes: int,
        max_files: int,
    ) -> Dict[str, Any]:
        if not base.exists():
            return {"success": False, "error": f"Path not found: {base}"}

        documents: List[Dict[str, Any]] = []
        total_bytes = 0
        visited_files = 0
        skipped_binaries = 0
        skipped_excluded = 0

        def want_ext(path: Path) -> bool:
            if include_exts:
                if path.suffix.lower() not in {e.lower() for e in include_exts}:
                    return False
            if exclude_exts:
                if path.suffix.lower() in {e.lower() for e in exclude_exts}:
                    return False
            return True

        def iter_files(root: Path) -> Iterable[Path]:
            if root.is_file():
                yield root
                return
            for dpath, dnames, fnames in os.walk(root, followlinks=False):
                d = Path(dpath)
                pruned = []
                for dn in list(dnames):
                    sub = d / dn
                    if _should_exclude(sub, DEFAULT_EXCLUDES, include_hidden):
                        pruned.append(dn)
                for dn in pruned:
                    dnames.remove(dn)
                for fn in fnames:
                    fpath = d / fn
                    if _should_exclude(fpath, DEFAULT_EXCLUDES, include_hidden):
                        continue
                    yield fpath

        base_rel_anchor = base if base.is_dir() else base.parent

        for f in iter_files(base):
            if visited_files >= max_files or total_bytes >= max_total_bytes:
                break
            try:
                st = f.stat()
                if not stat.S_ISREG(st.st_mode):
                    continue
                if not want_ext(f):
                    skipped_excluded += 1
                    continue
                with open(f, "rb") as fh:
                    head = fh.read(512)
                    if _is_probably_binary(head):
                        skipped_binaries += 1
                        continue
                    rest = fh.read(max(0, max_file_bytes - len(head)))
                    blob = head + rest
                    total_bytes += len(blob)
                    visited_files += 1
                text = blob.decode("utf-8", errors="replace")
                rel = f.relative_to(base_rel_anchor)
                documents.append({
                    "path": str(rel),
                    "bytes": len(blob),
                    "truncated": len(blob) >= max_file_bytes,
                    "preview": text if len(text) <= 10_000 else text[:10_000],  # keep response reasonable
                })
            except (IOError, OSError):
                continue

        return {
            "success": True,
            "operation": "index",
            "root": str(base),
            "documents": documents,
            "count": len(documents),
            "visited_files": visited_files,
            "skipped_binaries": skipped_binaries,
            "skipped_excluded": skipped_excluded,
            "total_bytes": total_bytes,
            "truncated": total_bytes >= max_total_bytes or visited_files >= max_files,
        }

