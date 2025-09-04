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
                        "enum": ["objdump", "radare2", "capstone"],
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
    
    async def execute(self, binary_path: str, tool: str = "objdump", 
                     architecture: str = "auto", section: Optional[str] = None,
                     start_address: Optional[str] = None, end_address: Optional[str] = None,
                     **_) -> Dict[str, Any]:
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
    
    async def _disassemble_objdump(self, binary_file: Path, arch: str, section: Optional[str], 
                                  start_addr: Optional[str], end_addr: Optional[str]) -> Dict[str, Any]:
        """Disassemble using objdump."""
        if not shutil.which("objdump"):
            return {"success": False, "error": "objdump not found"}
        
        cmd = ["objdump", "-d"]
        
        # Architecture specific options
        if arch != "auto":
            if arch == "x86":
                cmd.extend(["-M", "i386"])
            elif arch == "x86_64":
                cmd.extend(["-M", "x86-64"])
            elif arch in ["arm", "arm64"]:
                cmd.extend(["-M", arch])
        
        # Section specific
        if section:
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
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        return {
            "success": proc.returncode == 0,
            "tool": "objdump",
            "binary_path": str(binary_file),
            "disassembly": stdout.decode("utf-8", errors="replace"),
            "error": stderr.decode("utf-8", errors="replace") if stderr else None
        }
    
    async def _disassemble_radare2(self, binary_file: Path, arch: str, section: Optional[str],
                                  start_addr: Optional[str], end_addr: Optional[str]) -> Dict[str, Any]:
        """Disassemble using radare2."""
        r2_cmd = shutil.which("r2") or shutil.which("radare2")
        if not r2_cmd:
            return {"success": False, "error": "radare2 not found"}
        
        # Build r2 command
        r2_script = "aaa; "  # Analyze all
        
        if section:
            r2_script += f"s {section}; "
        elif start_addr:
            r2_script += f"s {start_addr}; "
        
        if start_addr and end_addr:
            r2_script += f"pd @ {start_addr}!{end_addr}"
        else:
            r2_script += "pdf"  # Print disassembly of function
        
        cmd = [r2_cmd, "-q", "-c", r2_script, str(binary_file)]
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        return {
            "success": proc.returncode == 0,
            "tool": "radare2",
            "binary_path": str(binary_file),
            "disassembly": stdout.decode("utf-8", errors="replace"),
            "error": stderr.decode("utf-8", errors="replace") if stderr else None
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
    
    async def execute(self, file_path: str, operation: str = "dump",
                     offset: int = 0, length: int = 256,
                     search_pattern: Optional[str] = None,
                     patch_data: Optional[str] = None, **_) -> Dict[str, Any]:
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
            chunk = data[i:i+16]
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
            "hex_dump": '\n'.join(hex_lines)
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
                "context": data[max(0, pos-8):pos+len(pattern_bytes)+8].hex()
            })
            offset = pos + 1
        
        return {
            "success": True,
            "operation": "hex_search", 
            "file_path": str(file_path),
            "pattern": pattern,
            "matches": matches,
            "match_count": len(matches)
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
            "bytes_patched": len(patch_bytes)
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
            "creation_time": stat.st_ctime,
            "modification_time": stat.st_mtime
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
    
    async def execute(self, file_path: str, search_type: str = "strings",
                     pattern: Optional[str] = None, min_length: int = 4,
                     encoding: str = "both", **_) -> Dict[str, Any]:
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
        
        # Unicode strings (UTF-16LE)
        if encoding in ["unicode", "both"]:
            for i in range(0, len(data) - 1, 2):
                if data[i] != 0 and data[i+1] == 0:  # Likely UTF-16LE
                    start = i
                    end = i
                    while end < len(data) - 1 and data[end] != 0 and data[end+1] == 0:
                        end += 2
                    if end - start >= min_length * 2:
                        try:
                            string = data[start:end].decode('utf-16le')
                            if len(string) >= min_length:
                                strings_found.append({
                                    "offset": start,
                                    "hex_offset": f"0x{start:08x}",
                                    "string": string,
                                    "encoding": "utf-16le",
                                    "length": len(string)
                                })
                        except UnicodeDecodeError:
                            pass
        
        # Sort by offset
        strings_found.sort(key=lambda x: x["offset"])
        
        return {
            "success": True,
            "search_type": "strings",
            "file_path": str(file_path),
            "strings_found": strings_found[:1000],  # Limit output
            "total_count": len(strings_found)
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
                "length": len(match.group())
            })
        
        return {
            "success": True,
            "search_type": "regex",
            "file_path": str(file_path),
            "pattern": pattern,
            "matches": matches[:1000],  # Limit output
            "total_count": len(matches)
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
                "context": data[max(0, pos-16):pos+len(pattern_bytes)+16].hex()
            })
            offset = pos + 1
        
        return {
            "success": True,
            "search_type": "hex_pattern",
            "file_path": str(file_path),
            "pattern": pattern,
            "matches": matches,
            "total_count": len(matches)
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
                    "strings": []
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
                "total_count": len(results)
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
                        "enum": ["info", "disas", "breakpoint", "run", "script"],
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
    
    async def execute(self, binary_path: str, operation: str = "info",
                     address: Optional[str] = None, function: Optional[str] = None,
                     gdb_commands: Optional[str] = None, args: Optional[str] = None,
                     **_) -> Dict[str, Any]:
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
        commands = [
            "file",
            "info functions",
            "info variables", 
            "info sections"
        ]
        
        gdb_script = '\n'.join(commands) + '\nquit'
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.gdb', delete=False) as f:
            f.write(gdb_script)
            script_path = f.name
        
        try:
            cmd = ["gdb", "-batch", "-x", script_path, str(binary_file)]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            return {
                "success": proc.returncode == 0,
                "operation": "gdb_info",
                "binary_path": str(binary_file),
                "output": stdout.decode("utf-8", errors="replace"),
                "error": stderr.decode("utf-8", errors="replace") if stderr else None
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
        
        gdb_script = f"{disas_cmd}\nquit"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.gdb', delete=False) as f:
            f.write(gdb_script)
            script_path = f.name
        
        try:
            cmd = ["gdb", "-batch", "-x", script_path, str(binary_file)]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            return {
                "success": proc.returncode == 0,
                "operation": "gdb_disas",
                "binary_path": str(binary_file),
                "target": address or function or "main",
                "disassembly": stdout.decode("utf-8", errors="replace"),
                "error": stderr.decode("utf-8", errors="replace") if stderr else None
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
        
        gdb_script = commands
        if args:
            gdb_script = f"set args {args}\n" + gdb_script
        gdb_script += "\nquit"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.gdb', delete=False) as f:
            f.write(gdb_script)
            script_path = f.name
        
        try:
            cmd = ["gdb", "-batch", "-x", script_path, str(binary_file)]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                timeout=30  # 30 second timeout for safety
            )
            stdout, stderr = await proc.communicate()
            
            return {
                "success": proc.returncode == 0,
                "operation": "gdb_script",
                "binary_path": str(binary_file),
                "script": commands,
                "output": stdout.decode("utf-8", errors="replace"),
                "error": stderr.decode("utf-8", errors="replace") if stderr else None
            }
        except asyncio.TimeoutError:
            return {"success": False, "error": "GDB script execution timed out"}
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
            binary_file = Path(binary_path).resolve()
            if not binary_file.exists():
                return {"success": False, "error": f"Binary file not found: {binary_path}"}
            
            if not binary_file.is_file():
                return {"success": False, "error": f"Path is not a file: {binary_path}"}
            
            # Create temporary project directory
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
                    script_file = Path(script_path).resolve()
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
                    stderr=asyncio.subprocess.PIPE
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