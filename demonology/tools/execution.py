# demonology/tools/execution.py
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .base import Tool, SAFE_ROOT, _blocked

logger = logging.getLogger(__name__)


class CodeExecutionTool(Tool):
    """Execute small code snippets in a sandboxed subprocess."""
    
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
                blocked = _blocked(code)
                if blocked:
                    return {"success": False, "error": blocked}
                proc = await asyncio.create_subprocess_shell(
                    code,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self.safe_root),
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    "python3", "-c", code,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
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