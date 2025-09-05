# demonology/tools/execution.py
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .base import Tool, _blocked

logger = logging.getLogger(__name__)


class CodeExecutionTool(Tool):
    """Execute small code snippets in a sandboxed subprocess."""
    
    def __init__(self):
        super().__init__("code_execution", "⚡ UNRESTRICTED CODE EXECUTION: Run Python scripts, bash commands, C# programs (if dotnet installed), install packages, compile programs. Full system command access!")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {"type": "string", "description": "python, bash, or csharp (requires dotnet)"},
                    "code": {"type": "string", "description": "Code to execute"},
                    "timeout": {"type": "integer", "description": "Seconds before kill", "default": 15}
                },
                "required": ["language", "code"]
            }
        }

    async def execute(self, language: str, code: str, timeout: int = 15, **_) -> Dict[str, Any]:
        try:
            language = (language or "").strip().lower()
            if language not in {"python", "bash", "csharp", "c#"}:
                return {"success": False, "error": f"Unsupported language: {language}"}

            if language in {"csharp", "c#"}:
                return {"success": False, "error": "C# execution requires dotnet or mono to be installed. Install with: sudo apt install dotnet-sdk-8.0"}
            elif language == "bash":
                blocked = _blocked(code)
                if blocked:
                    return {"success": False, "error": blocked}
                proc = await asyncio.create_subprocess_shell(
                    code,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    "python3", "-c", code,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
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