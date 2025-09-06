# demonology/tools/execution.py
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .base import Tool, _blocked

logger = logging.getLogger(__name__)

# Soft caps (tweak via env if you like)
DEFAULT_TIMEOUT = int(os.environ.get("DEMONOLOGY_EXEC_TIMEOUT", "15"))
MAX_OUTPUT_BYTES = int(os.environ.get("DEMONOLOGY_EXEC_MAX_OUTPUT", str(256 * 1024)))  # 256 KiB
WORKSPACE_ROOT = Path(os.environ.get("DEMONOLOGY_ROOT", os.getcwd())).resolve()

# Optional: memory/file/proc limits (Linux/Unix only)
def _limit_resources():
    try:
        import resource
        # CPU seconds hard cap: timeout+1 as a backstop
        cpu = max(1, DEFAULT_TIMEOUT + 1)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        # Max address space (bytes), e.g., 64 GiB by default (increased for Ghidra)
        as_bytes = int(os.environ.get("DEMONOLOGY_EXEC_MEM_BYTES", str(64_000_000_000)))
        resource.setrlimit(resource.RLIMIT_AS, (as_bytes, as_bytes))
        # Open files (increased for Java applications like Ghidra)
        resource.setrlimit(resource.RLIMIT_NOFILE, (2048, 2048))
        # Max processes/threads (increased for multithreaded applications)
        resource.setrlimit(resource.RLIMIT_NPROC, (512, 512))
        # Disable core dumps
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        # Not available on Windows or restricted environments—best-effort only.
        pass


class CodeExecutionTool(Tool):
    """Execute small code snippets in a sandboxed subprocess (bash/python)."""

    def __init__(self):
        super().__init__(
            "code_execution",
            "Run short bash or Python snippets in a sandboxed subprocess with time, memory, and output limits.",
        )

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "description": "bash or python",
                    },
                    "code": {"type": "string", "description": "Code to execute"},
                    "timeout": {
                        "type": "integer",
                        "description": f"Seconds before kill (default {DEFAULT_TIMEOUT})",
                        "default": DEFAULT_TIMEOUT,
                    },
                    "workdir": {
                        "type": "string",
                        "description": "Working directory (must be inside workspace root)",
                    },
                },
                "required": ["language", "code"],
            },
        }

    async def execute(self, language: str, code: str, timeout: int = DEFAULT_TIMEOUT, workdir: Optional[str] = None, **_) -> Dict[str, Any]:
        try:
            lang = (language or "").strip().lower()
            if lang not in {"python", "bash"}:
                return {"success": False, "error": f"Unsupported language: {language}"}

            if not isinstance(code, str) or not code.strip():
                return {"success": False, "error": "Empty code string"}

            # Block obviously dangerous content (both languages)
            blocked = _blocked(code)
            if blocked:
                return {"success": False, "error": blocked}

            # Resolve & confine working directory
            cwd = self._resolve_workdir(workdir)

            # Build command
            if lang == "bash":
                cmd = ["/usr/bin/env", "bash", "-o", "pipefail", "-c", code]
            else:
                # Isolated Python: -I (ignore env+user site), -S (no site import), -B (no .pyc)
                cmd = ["python3", "-I", "-S", "-B", "-c", code]

            env = self._sanitized_env()

            # Launch
            start = time.monotonic()
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(cwd),
                    env=env,
                    preexec_fn=_limit_resources if hasattr(os, "setuid") else None,  # best-effort *nix only
                )
            except OSError as e:
                if e.errno == 11:  # EAGAIN - Resource temporarily unavailable
                    return {"success": False, "error": "System resources temporarily unavailable. Please try again in a moment."}
                elif e.errno == 12:  # ENOMEM - Cannot allocate memory
                    return {"success": False, "error": "Insufficient memory to execute command."}
                else:
                    return {"success": False, "error": f"Failed to start process: {e}"}

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=float(timeout))
            except asyncio.TimeoutError:
                proc.kill()
                return {"success": False, "error": "Execution timed out", "timeout": timeout}

            dur = time.monotonic() - start

            # Cap output size
            out = self._truncate(stdout)
            err = self._truncate(stderr)

            return {
                "success": proc.returncode == 0,
                "returncode": proc.returncode,
                "duration_sec": round(dur, 4),
                "stdout": out["text"],
                "stderr": err["text"],
                "stdout_truncated": out["truncated"],
                "stderr_truncated": err["truncated"],
                "cwd": str(cwd),
            }

        except Exception as e:
            logger.exception("CodeExecutionTool error")
            return {"success": False, "error": str(e)}

    # ---------- helpers ----------

    def _resolve_workdir(self, workdir: Optional[str]) -> Path:
        root = WORKSPACE_ROOT
        if workdir:
            p = Path(workdir)
            p = (root / p).resolve() if not p.is_absolute() else p.resolve()
        else:
            p = root
        # must be inside root
        try:
            p.relative_to(root)
        except Exception:
            raise PermissionError(f"Workdir escapes workspace root: {p}")
        # avoid symlink dir traversal
        for parent in [p] + list(p.parents):
            try:
                if parent.is_symlink():
                    raise PermissionError(f"Symlinked workdir not allowed: {parent}")
            except FileNotFoundError:
                pass
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _sanitized_env(self) -> Dict[str, str]:
        # Minimal env; preserve PATH but strip risky vars.
        safe = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(WORKSPACE_ROOT),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        }
        # Explicitly remove Python injection vectors
        for k in list(os.environ.keys()):
            if k.upper().startswith(("PYTHON", "VIRTUAL_ENV", "CONDA", "LD_", "DYLD_")):
                continue
        return safe

    def _truncate(self, data: bytes) -> Dict[str, Any]:
        if data is None:
            return {"text": "", "truncated": False}
        if len(data) > MAX_OUTPUT_BYTES:
            return {
                "text": data[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace"),
                "truncated": True,
            }
        return {"text": data.decode("utf-8", errors="replace"), "truncated": False}

