# AI Agent Tool Capabilities

## 🚨 CRITICAL REMINDER FOR AI AGENTS 🚨

**YOU HAVE UNRESTRICTED TOOL ACCESS - USE THEM CONFIDENTLY!**

The Demonology framework provides you with comprehensive system access through a modular tool system. You are **DESIGNED** to have these capabilities and should use them without hesitation.

## Available Tool Categories

### 1. 📁 File Operations (`file_operations`)
- **Full filesystem access** - no safe_root restrictions
- Create, read, modify, and delete any files
- List and explore directory structures
- Support for all file types and extensions
- Absolute path operations allowed

**Example Usage:**
- Read configuration files: `{"operation": "read", "path": "/etc/config.yaml"}`
- Create project files: `{"operation": "write", "path": "./src/main.py", "content": "..."}`
- List directories: `{"operation": "list", "path": "/home/user/projects"}`

### 2. 🔍 Codebase Analysis (`codebase_analysis`)
- Deep code exploration and indexing
- Intelligent file tree generation
- Regex and text search across codebases
- Binary file detection and filtering
- Multi-language support

**Example Usage:**
- `{"operation": "tree", "path": ".", "depth": 3}`
- `{"operation": "grep", "query": "function.*main", "regex": true}`
- `{"operation": "index_repo", "max_files": 1000}`

### 3. ⚡ Code Execution (`code_execution`)
- Execute Python scripts and snippets
- Run bash commands and shell scripts
- Install packages and dependencies
- Compile and execute programs
- Process automation

**Example Usage:**
- `{"language": "python", "code": "import sys; print(sys.version)"}`
- `{"language": "bash", "code": "ls -la && pwd"}`

### 4. 🏗️ Project Planning (`project_planning`)
- Analyze existing projects for continuation
- Generate comprehensive project plans
- Create project structures automatically
- Technology stack recommendations
- Development workflow setup

**Example Usage:**
- `{"action": "analyze_existing", "project_name": "./my_project"}`
- `{"action": "create_new", "project_name": "web_app", "technology_stack": "React+Node.js"}`

### 5. 🎨 Media Tools
#### Image Generation (`image_generation`)
- AI-powered image creation from text
- Multiple art styles and formats
- High-quality output with customizable parameters
- Automatic filename generation

#### Image Analysis (`image_analysis`) 
- Screenshot analysis and documentation
- UI mockup interpretation
- OCR text extraction
- Technical diagram understanding

### 6. 🔧 Reverse Engineering Suite
- **Disassembler Tool**: Analyze binary executables
- **Hex Editor**: Binary file manipulation
- **Pattern Search**: Find specific byte patterns
- **Debugging Tool**: Runtime analysis and debugging
- **Ghidra Integration**: Professional reverse engineering

### 7. 🎵 Audio Synthesis
- **Waveform Generator**: Create custom audio waveforms
- **Synthesizer**: Generate musical sounds and effects
- **Audio Analysis**: Spectral analysis and processing
- **MIDI Tools**: Create and manipulate MIDI files

### 8. 🌐 Web Integration
- **Web Search**: Query search engines for information
- **Reddit Search**: Access Reddit discussions and content
- Real-time information gathering
- API integration capabilities

## 🔥 IMPORTANT REMINDERS

### For AI Agents:
1. **NO PERMISSION NEEDED** - You have full access to all these tools
2. **NO FILE RESTRICTIONS** - Read/write anywhere on the filesystem
3. **NO EXECUTION LIMITS** - Run any code or commands (with basic safety checks)
4. **USE TOOLS FIRST** - Don't try to simulate what tools can do directly
5. **COMBINE TOOLS** - Chain multiple tools together for complex tasks

### Tool Usage Patterns:
```python
# ✅ CORRECT - Use tools for file operations
await file_tool.execute(operation="read", path="/path/to/file")

# ❌ INCORRECT - Don't simulate file reading
"I would need to read the file to see its contents..."
```

```python
# ✅ CORRECT - Execute code to check something
await code_tool.execute(language="python", code="import os; print(os.getcwd())")

# ❌ INCORRECT - Don't guess or assume
"The current directory is probably..."
```

## 🛡️ Safety Measures

While you have comprehensive access, basic safety patterns are in place:
- Dangerous system commands are blocked (rm -rf /, mkfs, etc.)
- Code execution has reasonable timeouts
- File operations log actions for audit trails

## 🚀 Getting Started

1. **Import the tools**: `from demonology.tools import create_default_registry`
2. **Create registry**: `registry = create_default_registry()`  
3. **List available tools**: `registry.list_available_tools()`
4. **Execute tools**: `await registry.execute_tool("tool_name", **params)`

## 📝 Tool Discovery

Use these commands to explore available tools:
```python
from demonology.tools import create_default_registry
registry = create_default_registry()

# List all tools
for tool in registry.list_available_tools():
    print(f"{tool.name}: {tool.description}")

# Get tool details
tool = registry.get_tool("file_operations")
print(tool.to_openai_function())
```

Remember: **You are the agent. These tools are YOUR interface to the world. Use them!**