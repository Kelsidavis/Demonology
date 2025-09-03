# 🛠️ Demonology Tools Reference

This document provides detailed information about all available tools in Demonology CLI.

## Overview

Demonology includes a comprehensive suite of AI-powered tools that enable proactive project creation and information gathering. The AI can use these tools automatically when appropriate, making interactions more intelligent and productive.

## 🔍 Search & Information Tools

### Web Search Tool (`web_search`)

**Purpose**: Search the internet for current information using DuckDuckGo.

**When AI uses it**: Automatically when uncertain about facts or needing current information.

**Parameters**:
- `query` (required): Search query string
- `num_results` (optional): Number of results to return (default: 5)

**Example Usage**:
```
User: "What are the latest Python 3.12 features?"
AI: [Automatically searches web for Python 3.12 features]
```

**Configuration**: No setup required, works out of the box.

### Reddit Search Tool (`reddit_search`)

**Purpose**: Search Reddit discussions and community insights.

**When AI uses it**: For community opinions, user experiences, and discussion-based information.

**Parameters**:
- `query` (required): Search query string  
- `subreddit` (optional): Specific subreddit to search
- `sort` (optional): Sort method (relevance, hot, top, new, comments)
- `time_filter` (optional): Time range (all, day, week, month, year)
- `limit` (optional): Number of results (1-25, default: 5)

**Example Usage**:
```
User: "What do developers think about FastAPI vs Django?"
AI: [Searches Reddit for community discussions and comparisons]
```

**Configuration**:
- **Basic**: Works with public API (no setup required)
- **Enhanced**: Set environment variables for authenticated access:
  ```bash
  export REDDIT_CLIENT_ID="your_client_id"
  export REDDIT_CLIENT_SECRET="your_client_secret"
  export REDDIT_USER_AGENT="your_app_name"
  ```

## 🏗️ Project Creation Tools

### Project Planning Tool (`project_planning`)

**Purpose**: Generate complete project plans and automatically create project structures.

**When AI uses it**: When asked to "build", "create", or "make" a project.

**Parameters**:
- `project_name` (required): Name of the project
- `project_description` (required): What the project should do
- `technology_stack` (optional): Preferred technologies (Python, C++, React, etc.)
- `complexity` (optional): Project complexity (simple, medium, complex)
- `save_to_file` (optional): Save plan to markdown file (default: true)
- `execute_plan` (optional): Create actual project structure (default: true)

**Supported Project Types**:
- **Python**: Package structure with tests, requirements.txt, setup.py
- **C++**: Makefile-based projects with proper compilation setup
- **JavaScript/Node.js**: NPM projects with package.json and scripts
- **React**: Full React applications with components and build config
- **Generic**: Basic structure for any technology

**Example Usage**:
```
User: "Build me a Python web scraper project"
AI: [Creates complete Python project with structure, files, and documentation]
```

**Generated Structure Example** (Python):
```
project_name/
├── src/
│   ├── __init__.py
│   ├── main.py
│   └── modules/
├── tests/
│   └── test_main.py
├── requirements.txt
├── README.md
└── setup.py
```

## 📁 File Operations Tool (`file_operations`)

**Purpose**: Safe file system operations with security boundaries.

**When AI uses it**: For individual file operations not covered by project planning.

**Parameters**:
- `operation` (required): Type of operation
- `path` (optional): File/directory path
- `content` (optional): File content for write operations
- `recursive` (optional): Recursive deletion for directories

**Operations**:
- `create_directory`: Create directories
- `create_or_write_file`: Create or modify files
- `read`: Read file contents
- `list`: List directory contents
- `delete_file`: Delete files
- `delete_directory`: Delete directories

**Security Features**:
- All operations restricted to safe root directory
- Dangerous commands are blocked
- Automatic directory creation when needed

## ⚡ Code Execution Tool (`code_execution`)

**Purpose**: Execute code snippets in sandboxed environments.

**When AI uses it**: Only when explicitly requested by user.

**Parameters**:
- `language` (required): Programming language (python, bash)
- `code` (required): Code to execute
- `timeout` (optional): Execution timeout in seconds (default: 15)

**Supported Languages**:
- **Python**: Executed with `python3`
- **Bash**: Executed in shell with safety checks

**Security Features**:
- Sandboxed execution environments
- Timeout protection (max 10 minutes)
- Dangerous commands are blocked
- Output capture and error handling

## 🎨 Image Generation Tool (`image_generation`)

**Purpose**: Generate AI images for textures, icons, and visual assets using free APIs.

**When AI uses it**: When user requests image generation or visual asset creation.

**Parameters**:
- `prompt` (required): Description of the image to generate
- `style` (optional): Image style - realistic, artistic, anime, fantasy, pixel-art, concept-art (default: realistic)
- `size` (optional): Image dimensions - 512x512, 768x768, 1024x1024 (default: 512x512)
- `filename` (optional): Custom filename for saved image
- `save_image` (optional): Whether to save image to file (default: true)

**Supported APIs**:
- **Pollinations.ai**: Free, no API key required
- **Hugging Face FLUX.1**: High quality, requires HF token
- **Craiyon**: Backup option, basic quality

**Example Usage**:
```
User: "Generate a fantasy sword icon for my game"
AI: [Creates fantasy-styled sword icon and saves to file]
```

**Output**:
- Generated images saved to working directory
- Base64 encoded image data returned
- Automatic fallback between APIs for reliability

## 🔧 Tool Configuration

### Enabling/Disabling Tools

Edit your `config.yaml`:

```yaml
tools:
  enabled: true  # Set to false to disable all tools
  allowed_tools:
    - "file_operations"
    - "web_search"
    - "reddit_search" 
    - "project_planning"
    - "code_execution"
    - "image_generation"
```

### Working Directory

Set the safe root for file operations:

```yaml
tools:
  working_directory: "/path/to/safe/directory"  # null = current directory
```

## 🤖 AI Tool Usage Patterns

### Proactive Behavior

The AI automatically uses tools when:
- **Uncertain about information**: Triggers web search
- **Needing community insights**: Uses Reddit search  
- **Asked to create projects**: Uses project planning with execution
- **Managing individual files**: Uses file operations
- **Generating visual assets**: Uses image generation for textures, icons, and graphics

### Command Examples

**Information Gathering**:
```
"What are the latest trends in web development?"
→ AI uses web_search and reddit_search automatically
```

**Project Creation**:
```
"Create a Python REST API project"
→ AI uses project_planning to build complete structure
```

**File Management**:
```
"Add a configuration file to my project"
→ AI uses file_operations to create the file
```

**Image Generation**:
```
"Create a fantasy castle texture for my game"
→ AI uses image_generation to create and save the texture
```

## 🛡️ Security & Safety

### File Operations Security
- Operations restricted to configured safe directories
- No access to system files or sensitive locations
- Automatic validation of file paths and operations

### Code Execution Safety
- All execution happens in isolated environments
- Timeout protection prevents infinite loops
- Dangerous system commands are blocked
- Output is captured and sanitized

### Network Request Safety
- All web requests use appropriate timeouts
- Rate limiting respected for external APIs
- Fallback mechanisms for service unavailability

## 🔌 Extending Tools

### Creating Custom Tools

1. **Inherit from Tool base class**:
```python
class MyCustomTool(Tool):
    def __init__(self):
        super().__init__("my_tool", "Description of my tool")
```

2. **Implement required methods**:
```python
def to_openai_function(self) -> Dict[str, Any]:
    # Return OpenAI function specification
    
async def execute(self, **kwargs) -> Dict[str, Any]:
    # Implement tool functionality
```

3. **Register in ToolRegistry**:
```python
# In _register_default_tools method
self.register_tool(MyCustomTool())
```

4. **Add parameter handling** if needed in `execute_tool` method.

### Tool Development Guidelines

- **Security First**: Always validate inputs and restrict access
- **Error Handling**: Provide clear error messages and fallback behavior
- **Documentation**: Include comprehensive parameter descriptions
- **Testing**: Add tests for all functionality
- **Async Support**: Use async/await patterns for I/O operations

## 🐛 Troubleshooting

### Tool Not Available
```bash
# Check tool status
demonology
/tools
```

### Missing Dependencies
```bash
# For Reddit search enhanced features
pip install praw>=7.6.0

# For web requests  
pip install requests>=2.28.0
```

### Permission Errors
- Check working_directory configuration
- Ensure safe root directory exists and is writable
- Verify file permissions

### Network Issues
- Check internet connectivity for web/Reddit search
- Verify no firewall blocking outbound requests
- Test Reddit API credentials if using authenticated access

---

**Happy tool crafting with Demonology!** 🔮✨