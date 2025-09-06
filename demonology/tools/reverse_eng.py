# demonology/tools/reverse_eng.py
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Tool, _blocked

logger = logging.getLogger(__name__)


class DisassemblerTool(Tool):
    """Disassemble binaries using objdump, radare2, or other tools."""

    def __init__(self):
        super().__init__("disassembler", "Disassemble binary files using various disassemblers.")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "binary_path": {
                        "type": "string",
                        "description": "Path to the binary file to disassemble"
                    },
                    "tool": {
                        "type": "string",
                        "enum": ["objdump", "radare2"],  # dropped unsupported 'capstone'
                        "description": "Disassembler tool to use",
                        "default": "objdump"
                    },
                    "architecture": {
                        "type": "string",
                        "enum": ["x86", "x86_64", "arm", "arm64", "mips", "auto"],
                        "description": "Target architecture",
                        "default": "auto"
                    },
                    "section": {
                        "type": "string",
                        "description": "Specific section to disassemble (e.g. .text)"
                    },
                    "start_address": {
                        "type": "string",
                        "description": "Start address for disassembly (hex format)"
                    },
                    "end_address": {
                        "type": "string",
                        "description": "End address for disassembly (hex format)"
                    }
                },
                "required": ["binary_path"]
            }
        }

    def is_available(self) -> bool:
        """Check if any disassembler tool is available."""
        tools = ["objdump", "r2", "radare2"]
        return any(shutil.which(tool) for tool in tools)

    async def execute(
        self,
        binary_path: str,
        tool: str = "objdump",
        architecture: str = "auto",
        section: Optional[str] = None,
        start_address: Optional[str] = None,
        end_address: Optional[str] = None,
        **_,
    ) -> Dict[str, Any]:
        try:
            # Validate binary path
            binary_file = Path(binary_path).resolve()
            if not binary_file.exists():
                return {"success": False, "error": f"Binary file not found: {binary_path}"}

            if tool == "objdump":
                return await self._disassemble_objdump(binary_file, architecture, section, start_address, end_address)
            elif tool in ["radare2", "r2"]:
                return await self._disassemble_radare2(binary_file, architecture, section, start_address, end_address)
            else:
                return {"success": False, "error": f"Unsupported disassembler tool: {tool}"}

        except Exception as e:
            logger.exception("DisassemblerTool error")
            return {"success": False, "error": str(e)}

    async def _disassemble_objdump(
        self,
        binary_file: Path,
        arch: str,
        section: Optional[str],
        start_addr: Optional[str],
        end_addr: Optional[str],
    ) -> Dict[str, Any]:
        """Disassemble using objdump."""
        if not shutil.which("objdump"):
            return {"success": False, "error": "objdump not found"}

        cmd = ["objdump", "-d"]

        # Architecture specific options (objdump uses -m for machine)
        if arch != "auto":
            if arch == "x86":
                cmd.extend(["-m", "i386"])
            elif arch == "x86_64":
                cmd.extend(["-m", "i386:x86-64"])
            elif arch == "arm":
                cmd.extend(["-m", "arm"])
            elif arch == "arm64":
                cmd.extend(["-m", "aarch64"])
            elif arch == "mips":
                cmd.extend(["-m", "mips"])

        # Section specific
        if section:
            # objdump wants the raw section name, e.g. .text
            cmd.extend(["-j", section])

        # Address range
        if start_addr:
            cmd.extend(["--start-address", start_addr])
        if end_addr:
            cmd.extend(["--stop-address", end_addr])

        cmd.append(str(binary_file))

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        return {
            "success": proc.returncode == 0,
            "tool": "objdump",
            "binary_path": str(binary_file),
            "disassembly": stdout.decode("utf-8", errors="replace"),
            "error": stderr.decode("utf-8", errors="replace") if stderr else None,
        }

    async def _disassemble_radare2(
        self,
        binary_file: Path,
        arch: str,
        section: Optional[str],
        start_addr: Optional[str],
        end_addr: Optional[str],
    ) -> Dict[str, Any]:
        """Disassemble using radare2."""
        r2_cmd = shutil.which("r2") or shutil.which("radare2")
        if not r2_cmd:
            return {"success": False, "error": "radare2 not found"}

        # Build r2 command
        # 'aaa' analyze-all; then seek and disassemble
        r2_script = "aaa; "

        if section:
            # seek to a section by name; r2 usually flags sections as 'section.<name>'
            # fallback to plain section name if flag doesn't exist
            r2_script += f"s section.{section}; "
        elif start_addr:
            r2_script += f"s {start_addr}; "

        if start_addr and end_addr:
            # pd takes a length, not an end address. Compute len if possible.
            try:
                sa = int(start_addr, 16) if isinstance(start_addr, str) else int(start_addr)
                ea = int(end_addr, 16) if isinstance(end_addr, str) else int(end_addr)
                length = max(0, ea - sa)
                if length > 0:
                    r2_script += f"pd {length} @ {start_addr}"
                else:
                    r2_script += f"pdf @ {start_addr}"
            except Exception:
                r2_script += f"pdf @ {start_addr}"
        elif start_addr:
            r2_script += f"pdf @ {start_addr}"
        else:
            # Default to function at entry if possible, else a small chunk
            r2_script += "s entry0; pdf"

        cmd = [r2_cmd, "-q", "-c", r2_script, str(binary_file)]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        return {
            "success": proc.returncode == 0,
            "tool": "radare2",
            "binary_path": str(binary_file),
            "disassembly": stdout.decode("utf-8", errors="replace"),
            "error": stderr.decode("utf-8", errors="replace") if stderr else None,
        }


class HexEditorTool(Tool):
    """Programmatic hex editing and analysis."""

    def __init__(self):
        super().__init__("hex_editor", "Hex dump, search, and edit binary files.")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to analyze"
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["dump", "search", "patch", "info"],
                        "description": "Operation to perform",
                        "default": "dump"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Starting offset for dump/patch operations",
                        "default": 0
                    },
                    "length": {
                        "type": "integer",
                        "description": "Number of bytes to dump/patch",
                        "default": 256
                    },
                    "search_pattern": {
                        "type": "string",
                        "description": "Hex pattern to search for (e.g. '48656c6c6f')"
                    },
                    "patch_data": {
                        "type": "string",
                        "description": "Hex data to patch at offset (e.g. 'deadbeef')"
                    }
                },
                "required": ["file_path"]
            }
        }

    async def execute(
        self,
        file_path: str,
        operation: str = "dump",
        offset: int = 0,
        length: int = 256,
        search_pattern: Optional[str] = None,
        patch_data: Optional[str] = None,
        **_,
    ) -> Dict[str, Any]:
        try:
            target_file = Path(file_path).resolve()
            if not target_file.exists():
                return {"success": False, "error": f"File not found: {file_path}"}

            if operation == "dump":
                return await self._hex_dump(target_file, offset, length)
            elif operation == "search":
                if not search_pattern:
                    return {"success": False, "error": "search_pattern required for search operation"}
                return await self._hex_search(target_file, search_pattern)
            elif operation == "patch":
                if not patch_data:
                    return {"success": False, "error": "patch_data required for patch operation"}
                return await self._hex_patch(target_file, offset, patch_data)
            elif operation == "info":
                return await self._file_info(target_file)
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}

        except Exception as e:
            logger.exception("HexEditorTool error")
            return {"success": False, "error": str(e)}

    async def _hex_dump(self, file_path: Path, offset: int, length: int) -> Dict[str, Any]:
        """Create hex dump of file section."""
        with open(file_path, 'rb') as f:
            f.seek(offset)
            data = f.read(length)

        hex_lines = []
        for i in range(0, len(data), 16):
            chunk = data[i:i + 16]
            addr = f"{offset + i:08x}"
            hex_part = ' '.join(f"{b:02x}" for b in chunk).ljust(47)
            ascii_part = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in chunk)
            hex_lines.append(f"{addr}  {hex_part}  |{ascii_part}|")

        return {
            "success": True,
            "operation": "hex_dump",
            "file_path": str(file_path),
            "offset": offset,
            "length": len(data),
            "hex_dump": '\n'.join(hex_lines),
        }

    async def _hex_search(self, file_path: Path, pattern: str) -> Dict[str, Any]:
        """Search for hex pattern in file."""
        try:
            pattern_bytes = bytes.fromhex(pattern.replace(' ', ''))
        except ValueError:
            return {"success": False, "error": "Invalid hex pattern"}

        matches = []
        with open(file_path, 'rb') as f:
            data = f.read()

        offset = 0
        while True:
            pos = data.find(pattern_bytes, offset)
            if pos == -1:
                break
            matches.append({
                "offset": pos,
                "hex_offset": f"0x{pos:08x}",
                "context": data[max(0, pos - 8):pos + len(pattern_bytes) + 8].hex()
            })
            offset = pos + 1

        return {
            "success": True,
            "operation": "hex_search",
            "file_path": str(file_path),
            "pattern": pattern,
            "matches": matches,
            "match_count": len(matches),
        }

    async def _hex_patch(self, file_path: Path, offset: int, patch_data: str) -> Dict[str, Any]:
        """Patch file with hex data at offset."""
        try:
            patch_bytes = bytes.fromhex(patch_data.replace(' ', ''))
        except ValueError:
            return {"success": False, "error": "Invalid hex patch data"}

        # Read original data for backup
        with open(file_path, 'rb') as f:
            f.seek(offset)
            original_data = f.read(len(patch_bytes))

        # Apply patch
        with open(file_path, 'r+b') as f:
            f.seek(offset)
            f.write(patch_bytes)

        return {
            "success": True,
            "operation": "hex_patch",
            "file_path": str(file_path),
            "offset": offset,
            "hex_offset": f"0x{offset:08x}",
            "original_data": original_data.hex(),
            "patch_data": patch_data,
            "bytes_patched": len(patch_bytes),
        }

    async def _file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get file information and basic analysis."""
        stat = file_path.stat()

        # Read file header
        with open(file_path, 'rb') as f:
            header = f.read(64)

        # Detect file type based on magic bytes
        file_type = "unknown"
        if header.startswith(b'\x7fELF'):
            file_type = "ELF executable"
        elif header.startswith(b'MZ'):
            file_type = "PE executable"
        elif header.startswith(b'\xfe\xed\xfa'):
            file_type = "Mach-O executable"

        return {
            "success": True,
            "operation": "file_info",
            "file_path": str(file_path),
            "file_size": stat.st_size,
            "file_type": file_type,
            "header_hex": header.hex(),
            "creation_time": stat.st_ctime,   # Note: ctime is change time on Unix
            "modification_time": stat.st_mtime,
        }


class PatternSearchTool(Tool):
    """Search for patterns, signatures, and strings in binaries."""

    def __init__(self):
        super().__init__("pattern_search", "Search for patterns, strings, and signatures in binary files.")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to search"
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["strings", "regex", "hex_pattern", "yara"],
                        "description": "Type of search to perform",
                        "default": "strings"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Pattern to search for (depends on search_type)"
                    },
                    "min_length": {
                        "type": "integer",
                        "description": "Minimum string length for strings search",
                        "default": 4
                    },
                    "encoding": {
                        "type": "string",
                        "enum": ["ascii", "unicode", "both"],
                        "description": "String encoding to search for",
                        "default": "both"
                    }
                },
                "required": ["file_path"]
            }
        }

    async def execute(
        self,
        file_path: str,
        search_type: str = "strings",
        pattern: Optional[str] = None,
        min_length: int = 4,
        encoding: str = "both",
        **_,
    ) -> Dict[str, Any]:
        try:
            target_file = Path(file_path).resolve()
            if not target_file.exists():
                return {"success": False, "error": f"File not found: {file_path}"}

            if search_type == "strings":
                return await self._search_strings(target_file, min_length, encoding)
            elif search_type == "regex":
                if not pattern:
                    return {"success": False, "error": "pattern required for regex search"}
                return await self._search_regex(target_file, pattern)
            elif search_type == "hex_pattern":
                if not pattern:
                    return {"success": False, "error": "pattern required for hex_pattern search"}
                return await self._search_hex_pattern(target_file, pattern)
            elif search_type == "yara":
                if not pattern:
                    return {"success": False, "error": "pattern required for yara search"}
                return await self._search_yara(target_file, pattern)
            else:
                return {"success": False, "error": f"Unknown search type: {search_type}"}

        except Exception as e:
            logger.exception("PatternSearchTool error")
            return {"success": False, "error": str(e)}

    async def _search_strings(self, file_path: Path, min_length: int, encoding: str) -> Dict[str, Any]:
        """Extract strings from binary file."""
        strings_found = []

        with open(file_path, 'rb') as f:
            data = f.read()

        # ASCII strings
        if encoding in ["ascii", "both"]:
            import re
            ascii_pattern = rb'[\x20-\x7E]{' + str(min_length).encode() + rb',}'
            for match in re.finditer(ascii_pattern, data):
                strings_found.append({
                    "offset": match.start(),
                    "hex_offset": f"0x{match.start():08x}",
                    "string": match.group().decode('ascii', errors='replace'),
                    "encoding": "ascii",
                    "length": len(match.group())
                })

        # Unicode strings (UTF-16LE heuristic)
        if encoding in ["unicode", "both"]:
            i = 0
            n = len(data)
            while i < n - 1:
                if data[i] != 0 and data[i + 1] == 0:
                    start = i
                    j = i
                    while j < n - 1 and data[j] != 0 and data[j + 1] == 0:
                        j += 2
                    if j - start >= min_length * 2:
                        try:
                            string = data[start:j].decode('utf-16le')
                            if len(string) >= min_length:
                                strings_found.append({
                                    "offset": start,
                                    "hex_offset": f"0x{start:08x}",
                                    "string": string,
                                    "encoding": "utf-16le",
                                    "length": len(string),
                                })
                        except UnicodeDecodeError:
                            pass
                    i = j
                else:
                    i += 2

        strings_found.sort(key=lambda x: x["offset"])

        return {
            "success": True,
            "search_type": "strings",
            "file_path": str(file_path),
            "strings_found": strings_found[:1000],  # Limit output
            "total_count": len(strings_found),
        }

    async def _search_regex(self, file_path: Path, pattern: str) -> Dict[str, Any]:
        """Search using regular expression."""
        import re

        try:
            regex = re.compile(pattern.encode())
        except re.error as e:
            return {"success": False, "error": f"Invalid regex pattern: {e}"}

        matches = []
        with open(file_path, 'rb') as f:
            data = f.read()

        for match in regex.finditer(data):
            matches.append({
                "offset": match.start(),
                "hex_offset": f"0x{match.start():08x}",
                "match": match.group().decode('utf-8', errors='replace'),
                "length": len(match.group()),
            })

        return {
            "success": True,
            "search_type": "regex",
            "file_path": str(file_path),
            "pattern": pattern,
            "matches": matches[:1000],  # Limit output
            "total_count": len(matches),
        }

    async def _search_hex_pattern(self, file_path: Path, pattern: str) -> Dict[str, Any]:
        """Search for hex byte pattern."""
        try:
            pattern_bytes = bytes.fromhex(pattern.replace(' ', ''))
        except ValueError:
            return {"success": False, "error": "Invalid hex pattern"}

        matches = []
        with open(file_path, 'rb') as f:
            data = f.read()

        offset = 0
        while True:
            pos = data.find(pattern_bytes, offset)
            if pos == -1:
                break
            matches.append({
                "offset": pos,
                "hex_offset": f"0x{pos:08x}",
                "context": data[max(0, pos - 16):pos + len(pattern_bytes) + 16].hex(),
            })
            offset = pos + 1

        return {
            "success": True,
            "search_type": "hex_pattern",
            "file_path": str(file_path),
            "pattern": pattern,
            "matches": matches,
            "total_count": len(matches),
        }

    async def _search_yara(self, file_path: Path, yara_rule: str) -> Dict[str, Any]:
        """Search using YARA rules (if available)."""
        try:
            import yara
        except ImportError:
            return {"success": False, "error": "YARA not installed. Install with: pip install yara-python"}

        try:
            rules = yara.compile(source=yara_rule)
            matches = rules.match(str(file_path))

            results = []
            for match in matches:
                match_data = {
                    "rule": match.rule,
                    "namespace": match.namespace,
                    "tags": match.tags,
                    "meta": match.meta,
                    "strings": [],
                }

                for string in match.strings:
                    match_data["strings"].append({
                        "identifier": string.identifier,
                        "instances": [{"offset": inst.offset, "length": inst.length}
                                      for inst in string.instances]
                    })

                results.append(match_data)

            return {
                "success": True,
                "search_type": "yara",
                "file_path": str(file_path),
                "rule": yara_rule,
                "matches": results,
                "total_count": len(results),
            }

        except yara.YaraSyntaxError as e:
            return {"success": False, "error": f"YARA syntax error: {e}"}
        except Exception as e:
            return {"success": False, "error": f"YARA error: {e}"}


class DebuggingTool(Tool):
    """GDB integration for dynamic analysis."""

    def __init__(self):
        super().__init__("debugger", "Debug binaries using GDB with scripting support.")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "binary_path": {
                        "type": "string",
                        "description": "Path to the binary to debug"
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["info", "disas", "script"],  # removed 'breakpoint', 'run' (not implemented)
                        "description": "Debug operation to perform",
                        "default": "info"
                    },
                    "address": {
                        "type": "string",
                        "description": "Address for breakpoint or disassembly"
                    },
                    "function": {
                        "type": "string",
                        "description": "Function name for breakpoint or disassembly"
                    },
                    "gdb_commands": {
                        "type": "string",
                        "description": "Custom GDB commands to execute"
                    },
                    "args": {
                        "type": "string",
                        "description": "Command line arguments for the binary"
                    }
                },
                "required": ["binary_path"]
            }
        }

    def is_available(self) -> bool:
        """Check if GDB is available."""
        return shutil.which("gdb") is not None

    async def execute(
        self,
        binary_path: str,
        operation: str = "info",
        address: Optional[str] = None,
        function: Optional[str] = None,
        gdb_commands: Optional[str] = None,
        args: Optional[str] = None,
        **_,
    ) -> Dict[str, Any]:
        try:
            if not self.is_available():
                return {"success": False, "error": "GDB not found"}

            binary_file = Path(binary_path).resolve()
            if not binary_file.exists():
                return {"success": False, "error": f"Binary file not found: {binary_path}"}

            if operation == "info":
                return await self._gdb_info(binary_file)
            elif operation == "disas":
                return await self._gdb_disassemble(binary_file, address, function)
            elif operation == "script":
                if not gdb_commands:
                    return {"success": False, "error": "gdb_commands required for script operation"}
                return await self._gdb_script(binary_file, gdb_commands, args)
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}

        except Exception as e:
            logger.exception("DebuggingTool error")
            return {"success": False, "error": str(e)}

    async def _gdb_info(self, binary_file: Path) -> Dict[str, Any]:
        """Get binary information using GDB."""
        gdb_script = f"""file {binary_file}
set pagination off
info functions
info variables
info sections
quit
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.gdb', delete=False) as f:
            f.write(gdb_script)
            script_path = f.name

        try:
            cmd = ["gdb", "-batch", "-x", script_path, str(binary_file)]
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            return {
                "success": proc.returncode == 0,
                "operation": "gdb_info",
                "binary_path": str(binary_file),
                "output": stdout.decode("utf-8", errors="replace"),
                "error": stderr.decode("utf-8", errors="replace") if stderr else None,
            }
        finally:
            Path(script_path).unlink(missing_ok=True)

    async def _gdb_disassemble(self, binary_file: Path, address: Optional[str], function: Optional[str]) -> Dict[str, Any]:
        """Disassemble using GDB."""
        if address:
            disas_cmd = f"disas {address}"
        elif function:
            disas_cmd = f"disas {function}"
        else:
            disas_cmd = "disas main"

        gdb_script = f"""file {binary_file}
set pagination off
{disas_cmd}
quit
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.gdb', delete=False) as f:
            f.write(gdb_script)
            script_path = f.name

        try:
            cmd = ["gdb", "-batch", "-x", script_path, str(binary_file)]
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            return {
                "success": proc.returncode == 0,
                "operation": "gdb_disas",
                "binary_path": str(binary_file),
                "target": address or function or "main",
                "disassembly": stdout.decode("utf-8", errors="replace"),
                "error": stderr.decode("utf-8", errors="replace") if stderr else None,
            }
        finally:
            Path(script_path).unlink(missing_ok=True)

    async def _gdb_script(self, binary_file: Path, commands: str, args: Optional[str]) -> Dict[str, Any]:
        """Execute custom GDB script."""
        # Safety check for dangerous commands
        dangerous_commands = ["shell", "!", "run", "continue", "attach"]
        if any(cmd in commands.lower() for cmd in dangerous_commands):
            blocked = _blocked(commands)
            if blocked:
                return {"success": False, "error": blocked}

        gdb_script = f"file {binary_file}\nset pagination off\n"
        if args:
            gdb_script += f"set args {args}\n"
        gdb_script += commands
        gdb_script += "\nquit\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.gdb', delete=False) as f:
            f.write(gdb_script)
            script_path = f.name

        try:
            cmd = ["gdb", "-batch", "-x", script_path, str(binary_file)]
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            except asyncio.TimeoutError:
                proc.kill()
                return {"success": False, "error": "GDB script execution timed out"}

            return {
                "success": proc.returncode == 0,
                "operation": "gdb_script",
                "binary_path": str(binary_file),
                "script": commands,
                "output": stdout.decode("utf-8", errors="replace"),
                "error": stderr.decode("utf-8", errors="replace") if stderr else None,
            }
        finally:
            Path(script_path).unlink(missing_ok=True)


class GhidraAnalysisTool(Tool):
    """Perform headless binary analysis using Ghidra."""

    def __init__(self):
        super().__init__("ghidra_analysis", "Perform headless binary analysis using Ghidra.")

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
                        "description": "Analysis timeout in seconds (default: 1800 for decompile/full, 300 for others)",
                        "default": 1800
                    },
                    "dll_directory": {
                        "type": "string",
                        "description": "Optional path to directory containing DLLs for symbol resolution"
                    }
                },
                "required": ["binary_path"]
            }
        }

    def is_available(self) -> bool:
        """Check if Ghidra is available in the system."""
        ghidra_paths = [
            "/opt/ghidra/support/analyzeHeadless",
            "/usr/local/ghidra/support/analyzeHeadless",
            shutil.which("analyzeHeadless"),
        ]
        for path in ghidra_paths:
            if path and Path(path).exists():
                return True

        ghidra_install = os.environ.get("GHIDRA_INSTALL_DIR")
        if ghidra_install:
            headless_path = Path(ghidra_install) / "support" / "analyzeHeadless"
            if headless_path.exists():
                return True

        return False

    def _find_ghidra_headless(self) -> Optional[str]:
        """Find the Ghidra headless analyzer executable."""
        common_paths = [
            "/opt/ghidra/support/analyzeHeadless",
            "/usr/local/ghidra/support/analyzeHeadless",
            "/Applications/ghidra/support/analyzeHeadless",  # macOS
        ]
        for path in common_paths:
            if Path(path).exists():
                return path

        which_result = shutil.which("analyzeHeadless")
        if which_result:
            return which_result

        ghidra_install = os.environ.get("GHIDRA_INSTALL_DIR")
        if ghidra_install:
            headless_path = Path(ghidra_install) / "support" / "analyzeHeadless"
            if headless_path.exists():
                return str(headless_path)

        return None

    def _find_ghidra_script(self, script_name: str) -> Optional[str]:
        """Find a Ghidra script in common locations."""
        script_paths = [
            f"/opt/ghidra/Ghidra/Features/Base/ghidra_scripts/{script_name}",
            f"/usr/local/ghidra/Ghidra/Features/Base/ghidra_scripts/{script_name}",
        ]

        ghidra_install = os.environ.get("GHIDRA_INSTALL_DIR")
        if ghidra_install:
            script_paths.append(f"{ghidra_install}/Ghidra/Features/Base/ghidra_scripts/{script_name}")

        for path in script_paths:
            if Path(path).exists():
                return path

        return None

    async def _create_json_export_script(self, temp_path: Path) -> Path:
        """Create a temporary JSON export script."""
        script_path = temp_path / "ExportJson.py"

        script_content = '''
#!/usr/bin/env python3
"""
Temporary JSON Export Script for Ghidra Analysis
"""

import json
from ghidra.app.script import GhidraScript

program = getCurrentProgram()
if not program:
    print("JSON_START")
    print(json.dumps({"error": True, "message": "No program loaded"}))
    print("JSON_END")
else:
    analysis_data = {
        "program_info": {
            "name": str(program.getName()),
            "language": str(program.getLanguage()),
            "image_base": hex(program.getImageBase().getOffset()),
            "min_address": hex(program.getMinAddress().getOffset()),
            "max_address": hex(program.getMaxAddress().getOffset())
        },
        "functions": [],
        "symbols": [],
        "entry_points": []
    }

    fm = program.getFunctionManager()
    for i, func in enumerate(fm.getFunctions(True)):
        if i > 100:
            break
        analysis_data["functions"].append({
            "name": str(func.getName()),
            "address": hex(func.getEntryPoint().getOffset()),
            "size": func.getBody().getNumAddresses()
        })

    st = program.getSymbolTable()
    sym_iter = st.getAllSymbols(True)
    i = 0
    for sym in sym_iter:
        if i > 100:
            break
        analysis_data["symbols"].append({
            "name": str(sym.getName()),
            "address": hex(sym.getAddress().getOffset()),
            "type": str(sym.getSymbolType()),
        })
        i += 1

    print("JSON_START")
    print(json.dumps(analysis_data, indent=2))
    print("JSON_END")
'''
        script_path.write_text(script_content)
        return script_path

    async def execute(
        self,
        binary_path: str,
        analysis_type: str = "basic",
        output_format: str = "text",
        script_path: Optional[str] = None,
        timeout: Optional[int] = None,
        dll_directory: Optional[str] = None,
        **_,
    ) -> Dict[str, Any]:
        # Set appropriate timeout based on analysis type
        if timeout is None:
            if analysis_type in ["decompile", "full"]:
                timeout = 3600  # 1 hour for decompilation
            else:
                timeout = 600   # 10 minutes for basic analysis
        try:
            ghidra_headless = self._find_ghidra_headless()
            if not ghidra_headless:
                return {
                    "success": False,
                    "error": "Ghidra headless analyzer not found. Install Ghidra or set GHIDRA_INSTALL_DIR.",
                }
            
            logger.info(f"Using Ghidra headless at: {ghidra_headless}")
            
            # Verify the executable is valid and can run
            if not Path(ghidra_headless).exists():
                return {
                    "success": False,
                    "error": f"Ghidra headless executable not found at: {ghidra_headless}",
                }
            
            # Quick test to see if Ghidra can run at all
            try:
                test_process = await asyncio.create_subprocess_exec(
                    ghidra_headless, "-help",
                    stdout=asyncio.subprocess.PIPE, 
                    stderr=asyncio.subprocess.PIPE
                )
                test_stdout, test_stderr = await asyncio.wait_for(test_process.communicate(), timeout=10)
                logger.info(f"Ghidra test run completed with exit code: {test_process.returncode}")
            except Exception as e:
                logger.warning(f"Ghidra test run failed: {e}")
                # Continue anyway - this was just a test

            binary_file = Path(binary_path).resolve()
            if not binary_file.exists():
                return {"success": False, "error": f"Binary file not found: {binary_path}"}
            if not binary_file.is_file():
                return {"success": False, "error": f"Path is not a file: {binary_path}"}

            with tempfile.TemporaryDirectory(prefix="ghidra_analysis_") as temp_dir:
                temp_path = Path(temp_dir)
                project_dir = temp_path / "project"
                project_name = "analysis_project"
                
                # Ensure project directory exists
                project_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created project directory: {project_dir}")

                # Build command with proper Ghidra headless format
                # Ghidra format: analyzeHeadless <project_location> <project_name> [OPTIONS] [IMPORT_OPTIONS] <imported_file>
                cmd = [ghidra_headless]
                
                # Project location and name are required positional args (must come first)
                cmd.extend([str(project_dir), project_name])
                
                # Global options after project args  
                # Re-enable DLL directory support now that basic command works
                if dll_directory:
                    # Normalize path separators (handle Windows-style paths on Linux)  
                    normalized_search_path = dll_directory.replace('\\', '/')
                    dll_search_path = Path(normalized_search_path).resolve()
                    if dll_search_path.exists() and dll_search_path.is_dir():
                        cmd.extend(["-librarySearchPaths", str(dll_search_path)])
                        logger.info(f"Added DLL search path: {dll_search_path}")
                    else:
                        logger.warning(f"DLL directory not found or not a directory: {dll_directory}")
                
                # Import and analysis options
                cmd.extend(["-import", str(binary_file)])
                
                # If DLL directory is available, import key DLLs for better symbol resolution
                if dll_directory:
                    # Normalize path separators (handle Windows-style paths on Linux)
                    normalized_path = dll_directory.replace('\\', '/')
                    dll_path = Path(normalized_path).resolve()  # Resolve to absolute path first
                    if dll_path.exists() and dll_path.is_dir():
                        # Import key system and game DLLs that are likely to be referenced
                        priority_dlls = ['kernel32.dll', 'user32.dll', 'advapi32.dll', 'gdi32.dll', 
                                       'SMACKW32.DLL', 'ALBRIEF.DLL', 'ALSPRITE.DLL', 'D3DRM.DLL']
                        for dll_name in priority_dlls:
                            dll_file = dll_path / dll_name
                            # Check case-insensitive
                            if not dll_file.exists():
                                # Try lowercase
                                dll_file = dll_path / dll_name.lower()
                            if dll_file.exists():
                                cmd.extend(["-import", str(dll_file)])
                                logger.info(f"Added DLL for symbol resolution: {dll_name}")
                    else:
                        logger.warning(f"DLL directory not found: {dll_path}")
                
                # Analysis toggle comes after import
                # NOTE: Analysis happens by default! Only add -noanalysis to disable it
                if analysis_type == "basic":
                    cmd.append("-noanalysis")
                # For all other analysis types (decompile, full, etc.), analysis happens automatically

                # output format
                if output_format == "json":
                    json_script_path = self._find_ghidra_script("ExportJson.py")
                    if json_script_path:
                        cmd.extend(["-postScript", json_script_path])
                    else:
                        json_script = await self._create_json_export_script(temp_path)
                        cmd.extend(["-postScript", str(json_script)])
                elif output_format == "xml":
                    xml_script_path = self._find_ghidra_script("ExportXml.py")
                    if xml_script_path:
                        cmd.extend(["-postScript", xml_script_path])
                    else:
                        logger.warning("ExportXml.py script not found; falling back to text output")

                # custom user script
                if script_path:
                    script_file = Path(script_path).resolve()
                    if script_file.exists():
                        cmd.extend(["-postScript", str(script_file)])

                # create built-in helper scripts and attach requested one
                await self._create_analysis_scripts(temp_path, analysis_type)
                if analysis_type in ["functions", "full"]:
                    cmd.extend(["-scriptPath", str(temp_path / "scripts"), "-postScript", "ListFunctions.py"])
                if analysis_type in ["strings", "full"]:
                    cmd.extend(["-scriptPath", str(temp_path / "scripts"), "-postScript", "ListStrings.py"])
                if analysis_type in ["imports", "full"]:
                    cmd.extend(["-scriptPath", str(temp_path / "scripts"), "-postScript", "ListImports.py"])
                if analysis_type in ["exports", "full"]:
                    cmd.extend(["-scriptPath", str(temp_path / "scripts"), "-postScript", "ListExports.py"])
                if analysis_type in ["decompile", "full"]:
                    cmd.extend(["-scriptPath", str(temp_path / "scripts"), "-postScript", "DecompileAll.py"])

                # Log the command being executed
                logger.info(f"Executing Ghidra analysis: {' '.join(cmd)}")
                logger.info(f"Analysis type: {analysis_type}, timeout: {timeout}s")
                if dll_directory:
                    logger.info(f"DLL directory specified: {dll_directory}")
                
                cmd_final = cmd[:]  # Simple copy
                
                # Try running from a different working directory to avoid path confusion
                # The issue might be that analyzeHeadless (shell script) is somehow picking up the cwd
                process = await asyncio.create_subprocess_exec(
                    *cmd_final,
                    stdout=asyncio.subprocess.PIPE, 
                    stderr=asyncio.subprocess.PIPE,
                    cwd="/"  # Run from root to eliminate any working directory path contamination
                )

                try:
                    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
                except asyncio.TimeoutError:
                    process.kill()
                    return {"success": False, "error": f"Analysis timed out after {timeout} seconds"}

                stdout_text = stdout.decode('utf-8', errors='replace')
                stderr_text = stderr.decode('utf-8', errors='replace')

                results = await self._parse_analysis_results(
                    stdout_text, stderr_text, analysis_type, temp_path, output_format
                )

                # Provide better error messaging
                if process.returncode != 0:
                    error_msg = f"Ghidra analysis failed (exit code {process.returncode})"
                    if stderr_text.strip():
                        error_msg += f": {stderr_text.strip()}"
                    elif "timeout" in stdout_text.lower():
                        error_msg += f": Analysis timed out (consider increasing timeout beyond {timeout}s for complex binaries)"
                    elif "OutOfMemoryError" in stdout_text or "OutOfMemoryError" in stderr_text:
                        error_msg += ": Out of memory - try reducing analysis scope or increasing JVM memory"
                    elif "Exception" in stderr_text and not stderr_text.strip():
                        error_msg += ": Internal Ghidra error occurred during analysis"
                    
                    return {
                        "success": False,
                        "error": error_msg,
                        "binary": str(binary_file),
                        "analysis_type": analysis_type,
                        "stdout": stdout_text[-1000:] if len(stdout_text) > 1000 else stdout_text,
                        "stderr": stderr_text[-500:] if len(stderr_text) > 500 else stderr_text,
                        "return_code": process.returncode,
                    }

                return {
                    "success": True,
                    "binary": str(binary_file),
                    "analysis_type": analysis_type,
                    "output_format": output_format,
                    "results": results,
                    "stdout": stdout_text[-1000:] if len(stdout_text) > 1000 else stdout_text,
                    "stderr": stderr_text[-500:] if len(stderr_text) > 500 else stderr_text,
                    "return_code": process.returncode,
                }

        except Exception as e:
            logger.exception("Ghidra analysis failed")
            return {"success": False, "error": f"Analysis failed: {str(e)}"}

    async def _create_analysis_scripts(self, temp_path: Path, analysis_type: str):
        """Create basic Ghidra analysis scripts."""
        scripts_dir = temp_path / "scripts"
        scripts_dir.mkdir(exist_ok=True)

        # Functions
        if analysis_type in ["functions", "full"] and not (scripts_dir / "ListFunctions.py").exists():
            (scripts_dir / "ListFunctions.py").write_text('''
# List all functions in the binary
from ghidra.program.model.listing import *
program = getCurrentProgram()
listing = program.getListing()
functions = listing.getFunctions(True)
print("=== FUNCTIONS ===")
for func in functions:
    try:
        print("Function: {} at {}".format(func.getName(), func.getEntryPoint()))
        print("  Size: {} bytes".format(func.getBody().getNumAddresses()))
        print("  Parameters: {}".format(func.getParameterCount()))
        print("")
    except:
        pass
''')

        # Strings
        if analysis_type in ["strings", "full"] and not (scripts_dir / "ListStrings.py").exists():
            (scripts_dir / "ListStrings.py").write_text('''
# List strings found in defined data
from ghidra.program.model.data import *
program = getCurrentProgram()
listing = program.getListing()
print("=== STRINGS ===")
data_iter = listing.getDefinedData(True)
for data in data_iter:
    try:
        if data.hasStringValue():
            print("String at {}: {}".format(data.getAddress(), data.getValue()))
    except:
        pass
''')

        # Imports
        if analysis_type in ["imports", "full"] and not (scripts_dir / "ListImports.py").exists():
            (scripts_dir / "ListImports.py").write_text('''
# List imported libraries and symbols
program = getCurrentProgram()
ext = program.getExternalManager()
print("=== IMPORTS ===")
for lib in ext.getExternalLibraries():
    try:
        print("LIB:", lib.getName())
    except:
        pass
''')

        # Exports
        if analysis_type in ["exports", "full"] and not (scripts_dir / "ListExports.py").exists():
            (scripts_dir / "ListExports.py").write_text('''
# List external/exported functions
from ghidra.program.model.symbol import SymbolType
program = getCurrentProgram()
st = program.getSymbolTable()
print("=== EXPORTS ===")
for s in st.getAllSymbols(True):
    try:
        if s.getSymbolType() == SymbolType.FUNCTION and s.isExternal():
            print("EXPORT:", s.getName(), s.getAddress())
    except:
        pass
''')

        # Decompile all (basic)
        if analysis_type in ["decompile", "full"] and not (scripts_dir / "DecompileAll.py").exists():
            (scripts_dir / "DecompileAll.py").write_text('''
# Enhanced decompilation with better error handling and symbol resolution
from ghidra.app.decompiler import DecompInterface, DecompileOptions
from ghidra.app.decompiler.flatapi import FlatDecompilerAPI
from ghidra.program.model.listing import Function
from ghidra.program.model.symbol import SourceType
from java.lang import Exception as JavaException

program = getCurrentProgram()
fm = program.getFunctionManager()

# Enhanced decompiler interface with better options
iface = DecompInterface()
options = DecompileOptions()
options.grabFromProgram(program)
# Enable more thorough analysis
options.setEliminateUnreachable(True)
options.setSimplifyDoublePrecision(True)
iface.setOptions(options)
iface.openProgram(program)

print("=== ENHANCED DECOMPILATION RESULTS ===")
print("Binary: {}".format(program.getName()))
print("Architecture: {}".format(program.getLanguage().getLanguageID()))
print("Base Address: {}".format(program.getImageBase()))
print("")

# Get all functions, including those discovered during analysis
functions = fm.getFunctions(True)
func_count = 0
decompiled_count = 0
error_count = 0

print("=== FUNCTION SUMMARY ===")
for f in functions:
    func_count += 1

# Now decompile functions with enhanced error handling
print("\\n=== DECOMPILED FUNCTIONS ===")
for f in functions:
    try:
        # Skip functions that are too small or likely data
        if f.getBody().getNumAddresses() < 10:
            continue
            
        print("\\n" + "="*60)
        print("FUNCTION: {}".format(f.getName()))
        print("ADDRESS: {}".format(f.getEntryPoint()))
        print("SIZE: {} bytes".format(f.getBody().getNumAddresses()))
        print("PARAMETERS: {}".format(f.getParameterCount()))
        
        # Get function signature info
        sig = f.getSignature()
        if sig:
            print("SIGNATURE: {}".format(sig.getPrototypeString()))
            
        print("-" * 40)
        
        # Decompile with extended timeout for complex functions
        timeout = 120 if f.getBody().getNumAddresses() > 1000 else 60
        res = iface.decompileFunction(f, timeout, monitor)
        
        if res and res.getDecompiledFunction():
            decompiled_func = res.getDecompiledFunction()
            c_code = decompiled_func.getC()
            if c_code and len(c_code.strip()) > 0:
                print(c_code)
                decompiled_count += 1
            else:
                print("// Decompilation produced empty result")
                error_count += 1
        else:
            error_msg = "Unknown error"
            if res:
                error_msg = res.getErrorMessage()
            print("// DECOMPILATION FAILED: {}".format(error_msg))
            error_count += 1
            
    except JavaException as je:
        print("// JAVA EXCEPTION: {}".format(str(je)))
        error_count += 1
    except Exception as e:
        print("// PYTHON EXCEPTION: {}".format(str(e)))
        error_count += 1

print("\\n" + "="*60)
print("=== DECOMPILATION SUMMARY ===")
print("Total Functions Found: {}".format(func_count))
print("Successfully Decompiled: {}".format(decompiled_count))
print("Decompilation Errors: {}".format(error_count))
if func_count > 0:
    success_rate = (decompiled_count * 100.0) / func_count
    print("Success Rate: {:.1f}%".format(success_rate))
    
    # Mark analysis as complete vs partial based on success rate
    if success_rate >= 80:
        print("ANALYSIS STATUS: COMPLETE")
    elif success_rate >= 50:
        print("ANALYSIS STATUS: MOSTLY_COMPLETE") 
    else:
        print("ANALYSIS STATUS: PARTIAL")
else:
    print("ANALYSIS STATUS: FAILED")

iface.dispose()
''')

    async def _parse_analysis_results(
        self,
        stdout: str,
        stderr: str,
        analysis_type: str,
        temp_path: Path,
        output_format: str,
    ) -> Dict[str, Any]:
        """Parse Ghidra analysis results."""
        results: Dict[str, Any] = {
            "analysis_type": analysis_type,
            "summary": {},
            "details": [],
        }

        # If JSON markers exist, parse them regardless of analysis_type
        if "JSON_START" in stdout and "JSON_END" in stdout:
            try:
                json_block = stdout.split("JSON_START", 1)[1].split("JSON_END", 1)[0]
                import json
                parsed = json.loads(json_block.strip())
                results["summary"]["status"] = "completed"
                results["details"].append({"type": "json", "data": parsed})
                return results
            except Exception as e:
                results["summary"]["json_parse_error"] = str(e)

        # Otherwise parse simple text sections
        lines = stdout.split('\n')
        functions_found: List[str] = []
        strings_found: List[str] = []
        imports_found: List[str] = []
        exports_found: List[str] = []

        current_section = None
        for line in lines:
            s = line.strip()
            if s == "=== FUNCTIONS ===":
                current_section = "functions"
                continue
            if s == "=== STRINGS ===":
                current_section = "strings"
                continue
            if s == "=== IMPORTS ===":
                current_section = "imports"
                continue
            if s == "=== EXPORTS ===":
                current_section = "exports"
                continue

            if current_section == "functions" and s.startswith("Function:"):
                functions_found.append(s)
            elif current_section == "strings" and s.startswith("String at"):
                strings_found.append(s)
            elif current_section == "imports" and s:
                imports_found.append(s)
            elif current_section == "exports" and s.startswith("EXPORT:"):
                exports_found.append(s)

        if functions_found:
            results["details"].append({
                "type": "functions",
                "count": len(functions_found),
                "items": functions_found[:50],
            })
        if strings_found:
            results["details"].append({
                "type": "strings",
                "count": len(strings_found),
                "items": strings_found[:50],
            })
        if imports_found:
            results["details"].append({
                "type": "imports",
                "count": len(imports_found),
                "items": imports_found[:50],
            })
        if exports_found:
            results["details"].append({
                "type": "exports",
                "count": len(exports_found),
                "items": exports_found[:50],
            })

        if "Analysis complete" in stdout:
            results["summary"]["status"] = "completed"
        elif "Error" in stderr or "Exception" in stderr:
            results["summary"]["status"] = "error"
        else:
            results["summary"]["status"] = "partial"

        return results

