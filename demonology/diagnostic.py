#!/usr/bin/env python3
"""
Demonology Diagnostic Tools

Helps troubleshoot reverse engineering setup and common issues.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import magic
import json

class FileValidator:
    """Utilities for validating files and directories."""
    
    @staticmethod
    def find_case_insensitive(directory: Path, filename: str) -> Optional[Path]:
        """Find file with case-insensitive matching."""
        if not directory.exists() or not directory.is_dir():
            return None
        
        # Try exact match first
        exact_path = directory / filename
        if exact_path.exists():
            return exact_path
        
        # Try case-insensitive search
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.name.lower() == filename.lower():
                return file_path
        
        return None
    
    @staticmethod
    def validate_dll_directory(dll_dir: str, expected_dlls: List[str]) -> Dict[str, Any]:
        """Validate DLL directory and suggest fixes for missing files."""
        results = {
            "valid": True,
            "found": [],
            "missing": [],
            "case_mismatches": [],
            "suggestions": []
        }
        
        dll_path = Path(dll_dir).resolve()
        if not dll_path.exists():
            results["valid"] = False
            results["suggestions"].append(f"Directory {dll_dir} does not exist")
            return results
        
        if not dll_path.is_dir():
            results["valid"] = False
            results["suggestions"].append(f"{dll_dir} is not a directory")
            return results
        
        for dll in expected_dlls:
            found_path = FileValidator.find_case_insensitive(dll_path, dll)
            if found_path:
                if found_path.name != dll:
                    results["case_mismatches"].append({
                        "expected": dll,
                        "found": found_path.name,
                        "suggestion": f"cd {dll_dir} && ln -sf {found_path.name} {dll}"
                    })
                results["found"].append(dll)
            else:
                results["missing"].append(dll)
        
        if results["case_mismatches"]:
            results["suggestions"].append("Fix case mismatches using the symlink commands shown above")
        
        if results["missing"]:
            results["suggestions"].append(f"Find and add missing DLL files: {', '.join(results['missing'])}")
        
        return results

def find_ghidra_headless() -> Optional[str]:
    """Find Ghidra headless analyzer."""
    potential_paths = [
        "/opt/ghidra/support/analyzeHeadless",
        "/usr/local/ghidra/support/analyzeHeadless",
        "/Applications/ghidra/support/analyzeHeadless"
    ]
    
    for path in potential_paths:
        if Path(path).exists():
            return path
    
    # Check environment variable
    ghidra_dir = Path.home() / ".ghidra" if "GHIDRA_INSTALL_DIR" not in sys.path else Path(sys.path["GHIDRA_INSTALL_DIR"])
    if ghidra_dir.exists():
        headless_path = ghidra_dir / "support" / "analyzeHeadless"
        if headless_path.exists():
            return str(headless_path)
    
    return None

def diagnose_binary(binary_path: str, dll_dir: Optional[str] = None, verbose: bool = False) -> Dict[str, Any]:
    """Comprehensive diagnostic of binary and environment for reverse engineering."""
    
    results = {
        "binary": {},
        "dll_directory": {},
        "tools": {},
        "overall_status": "unknown"
    }
    
    # Check binary
    binary = Path(binary_path).resolve()
    results["binary"]["path"] = str(binary)
    results["binary"]["exists"] = binary.exists()
    
    if binary.exists():
        try:
            stat = binary.stat()
            results["binary"]["size"] = stat.st_size
            results["binary"]["readable"] = True
            
            # Try to get file type
            try:
                file_type = magic.from_file(str(binary))
                results["binary"]["type"] = file_type
                results["binary"]["is_pe"] = "PE32" in file_type or "PE32+" in file_type
                results["binary"]["is_elf"] = "ELF" in file_type
                results["binary"]["architecture"] = "32-bit" if "PE32" in file_type and "PE32+" not in file_type else "64-bit" if "PE32+" in file_type or "64-bit" in file_type else "unknown"
            except Exception as e:
                results["binary"]["type"] = f"Unable to determine: {e}"
                results["binary"]["is_pe"] = False
                results["binary"]["is_elf"] = False
                
        except PermissionError:
            results["binary"]["readable"] = False
            results["binary"]["error"] = "Permission denied"
    else:
        results["binary"]["error"] = "File not found"
    
    # Check DLL directory if provided
    if dll_dir:
        # Common Windows DLLs that games often reference
        common_dlls = [
            "KERNEL32.dll", "USER32.dll", "ADVAPI32.dll", "GDI32.dll", "WINMM.dll",
            "DDRAW.dll", "DSOUND.dll", "D3DRM.dll", "COMCTL32.dll", "VERSION.dll"
        ]
        
        validation = FileValidator.validate_dll_directory(dll_dir, common_dlls)
        results["dll_directory"] = validation
    
    # Check tools
    ghidra_path = find_ghidra_headless()
    results["tools"]["ghidra"] = {
        "available": ghidra_path is not None,
        "path": ghidra_path
    }
    
    # Overall status
    issues = []
    if not results["binary"]["exists"]:
        issues.append("Binary file not found")
    elif not results["binary"].get("readable", False):
        issues.append("Binary file not readable")
    
    if dll_dir and not results["dll_directory"].get("valid", True):
        issues.append("DLL directory issues")
    
    if not results["tools"]["ghidra"]["available"]:
        issues.append("Ghidra not found")
    
    if not issues:
        results["overall_status"] = "ready"
    elif len(issues) == 1:
        results["overall_status"] = "minor_issues"
    else:
        results["overall_status"] = "major_issues"
    
    results["issues"] = issues
    
    return results

def print_diagnostic_report(results: Dict[str, Any], verbose: bool = False):
    """Print a formatted diagnostic report."""
    
    print("🔍 Demonology Reverse Engineering Diagnostic Report")
    print("=" * 60)
    
    # Binary status
    binary = results["binary"]
    print(f"\n📁 Binary Analysis: {binary['path']}")
    print(f"   Exists: {'✅' if binary['exists'] else '❌'}")
    
    if binary['exists']:
        print(f"   Size: {binary.get('size', 0):,} bytes")
        print(f"   Type: {binary.get('type', 'Unknown')}")
        if 'architecture' in binary:
            print(f"   Architecture: {binary['architecture']}")
        print(f"   Readable: {'✅' if binary.get('readable', False) else '❌'}")
    
    # DLL directory status
    if results["dll_directory"]:
        dll_info = results["dll_directory"]
        print(f"\n📚 DLL Directory Analysis")
        print(f"   Valid: {'✅' if dll_info['valid'] else '❌'}")
        
        if dll_info['found']:
            print(f"   Found DLLs: {len(dll_info['found'])}")
            if verbose:
                for dll in dll_info['found']:
                    print(f"      ✅ {dll}")
        
        if dll_info['missing']:
            print(f"   Missing DLLs: {len(dll_info['missing'])}")
            if verbose:
                for dll in dll_info['missing']:
                    print(f"      ❌ {dll}")
        
        if dll_info['case_mismatches']:
            print(f"   Case Mismatches: {len(dll_info['case_mismatches'])}")
            for mismatch in dll_info['case_mismatches']:
                print(f"      ⚠️  Expected: {mismatch['expected']}")
                print(f"         Found: {mismatch['found']}")
                if verbose:
                    print(f"         Fix: {mismatch['suggestion']}")
    
    # Tools status
    tools = results["tools"]
    print(f"\n🛠️  Tools Analysis")
    print(f"   Ghidra: {'✅' if tools['ghidra']['available'] else '❌'}")
    if tools['ghidra']['available']:
        print(f"      Path: {tools['ghidra']['path']}")
    
    # Overall status
    print(f"\n🎯 Overall Status: ", end="")
    status = results["overall_status"]
    if status == "ready":
        print("✅ READY - All systems go!")
    elif status == "minor_issues":
        print("⚠️  MINOR ISSUES - Should work but could be improved")
    else:
        print("❌ MAJOR ISSUES - Likely to fail")
    
    # Suggestions
    if results["dll_directory"] and results["dll_directory"].get("suggestions"):
        print(f"\n💡 Suggestions:")
        for i, suggestion in enumerate(results["dll_directory"]["suggestions"], 1):
            print(f"   {i}. {suggestion}")
    
    if results.get("issues"):
        print(f"\n⚠️  Issues to Address:")
        for i, issue in enumerate(results["issues"], 1):
            print(f"   {i}. {issue}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose demonology reverse engineering setup")
    parser.add_argument("binary", help="Path to binary file to analyze")
    parser.add_argument("--dll-dir", help="Path to DLL directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    results = diagnose_binary(args.binary, args.dll_dir, args.verbose)
    
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_diagnostic_report(results, args.verbose)