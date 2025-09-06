# Demonology CLI Improvements Based on REBEXE.EXE Debugging

## 1. Enhanced Input Handling

### Problem Solved
- Auto-continue setup consumed piped input meant for analysis commands
- Non-interactive mode wasn't properly detected

### Improvements to Implement

```python
# Add to cli.py - better input mode detection
def detect_input_mode(self) -> str:
    """Detect if we're in interactive, piped, or batch mode."""
    if not sys.stdin.isatty():
        if sys.stdin.peek(1):  # Has piped data
            return "piped"
        else:
            return "batch"
    return "interactive"

# Add --batch flag support
@click.option('--batch', is_flag=True, help='Run in batch mode, skip all interactive prompts')
def main(batch: bool, ...):
    if batch:
        config.ui.batch_mode = True
```

## 2. Progress Indicators for Long Operations

### Problem Solved
- Ghidra analysis appeared to hang with no progress feedback
- Users couldn't tell if analysis was working or stuck

### Improvements to Implement

```python
# Add to tools/reverse_eng.py
async def execute_with_progress(self, ...):
    async def show_progress():
        dots = 0
        while not analysis_complete:
            print(f"Analyzing {'.' * (dots % 4):<3}", end='\r')
            dots += 1
            await asyncio.sleep(1)
    
    progress_task = asyncio.create_task(show_progress())
    try:
        result = await self.execute(...)
    finally:
        analysis_complete = True
        progress_task.cancel()
```

## 3. Smart File/Directory Validation

### Problem Solved
- Case sensitivity issues with DLL references
- Silent failures when expected files don't exist

### Improvements to Implement

```python
# Add to tools/base.py
class FileValidator:
    @staticmethod
    def find_case_insensitive(directory: Path, filename: str) -> Optional[Path]:
        """Find file with case-insensitive matching."""
        if not directory.exists():
            return None
        
        # Try exact match first
        exact_path = directory / filename
        if exact_path.exists():
            return exact_path
        
        # Try case-insensitive search
        for file_path in directory.iterdir():
            if file_path.name.lower() == filename.lower():
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
        
        dll_path = Path(dll_dir)
        if not dll_path.exists():
            results["valid"] = False
            results["suggestions"].append(f"Directory {dll_dir} does not exist")
            return results
        
        for dll in expected_dlls:
            found_path = FileValidator.find_case_insensitive(dll_path, dll)
            if found_path:
                if found_path.name != dll:
                    results["case_mismatches"].append({
                        "expected": dll,
                        "found": found_path.name,
                        "suggestion": f"ln -sf {found_path.name} {dll}"
                    })
                results["found"].append(dll)
            else:
                results["missing"].append(dll)
        
        return results
```

## 4. Diagnostic Mode

### Problem Solved
- Hard to troubleshoot why analysis failed
- No visibility into tool parameter validation

### Improvements to Implement

```python
# Add diagnostic command
@click.command()
@click.argument('binary_path')
@click.option('--dll-dir', help='DLL directory to validate')
def diagnose(binary_path: str, dll_dir: str):
    """Diagnose potential issues with reverse engineering setup."""
    print("🔍 Demonology Diagnostic Mode")
    
    # Check binary
    binary = Path(binary_path)
    print(f"Binary: {binary}")
    print(f"  Exists: {'✅' if binary.exists() else '❌'}")
    if binary.exists():
        print(f"  Size: {binary.stat().st_size:,} bytes")
        print(f"  Type: {magic.from_file(str(binary))}")
    
    # Check DLL directory
    if dll_dir:
        expected_dlls = ['KERNEL32.dll', 'USER32.dll', 'ADVAPI32.dll', ...]
        validation = FileValidator.validate_dll_directory(dll_dir, expected_dlls)
        
        print(f"\nDLL Directory: {dll_dir}")
        print(f"  Valid: {'✅' if validation['valid'] else '❌'}")
        print(f"  Found: {len(validation['found'])}/{len(expected_dlls)} DLLs")
        
        if validation['case_mismatches']:
            print("  Case Mismatches:")
            for mismatch in validation['case_mismatches']:
                print(f"    {mismatch['expected']} → {mismatch['found']}")
                print(f"    Fix: {mismatch['suggestion']}")
    
    # Check Ghidra installation
    ghidra_path = find_ghidra_headless()
    print(f"\nGhidra: {'✅' if ghidra_path else '❌'}")
    if ghidra_path:
        print(f"  Path: {ghidra_path}")
```

## 5. Better Error Messages

### Problem Solved
- Generic error messages didn't help with troubleshooting
- No guidance on how to fix issues

### Improvements to Implement

```python
# Enhanced error reporting
class AnalysisError(Exception):
    def __init__(self, message: str, suggestions: List[str] = None):
        super().__init__(message)
        self.suggestions = suggestions or []
    
    def format_error(self) -> str:
        error_msg = f"❌ {str(self)}"
        if self.suggestions:
            error_msg += "\n\n💡 Suggestions:"
            for i, suggestion in enumerate(self.suggestions, 1):
                error_msg += f"\n  {i}. {suggestion}"
        return error_msg

# Usage in tools
if not dll_validation['valid']:
    suggestions = [
        f"Check if directory {dll_dir} exists",
        "Verify DLL files have correct case sensitivity",
        "Run 'demonology diagnose <binary>' for detailed analysis"
    ]
    raise AnalysisError(f"DLL directory validation failed", suggestions)
```

## 6. Timeout Handling with User Control

### Problem Solved
- Analysis timeouts were hard-coded
- No way to cancel or extend running operations

### Improvements to Implement

```python
# Interactive timeout handling
async def execute_with_interactive_timeout(self, timeout: int = 300):
    """Execute with user-controllable timeout."""
    
    def timeout_handler():
        print(f"\n⏰ Analysis has been running for {timeout} seconds.")
        print("Options:")
        print("  c - Continue for another 5 minutes")
        print("  q - Quit analysis")
        print("  w - Wait indefinitely")
        
        choice = input("Your choice [c/q/w]: ").lower()
        if choice == 'c':
            return 300  # 5 more minutes
        elif choice == 'w':
            return None  # No timeout
        else:
            return 0    # Quit
    
    while True:
        try:
            result = await asyncio.wait_for(self._execute(), timeout=timeout)
            return result
        except asyncio.TimeoutError:
            new_timeout = timeout_handler()
            if new_timeout == 0:
                return {"success": False, "error": "Analysis cancelled by user"}
            elif new_timeout is None:
                timeout = None
            else:
                timeout = new_timeout
```

## Implementation Priority

1. **High Priority**: Input handling and diagnostic mode
2. **Medium Priority**: Progress indicators and better error messages  
3. **Low Priority**: Interactive timeout handling

These improvements would make demonology much more robust and user-friendly for complex reverse engineering tasks.