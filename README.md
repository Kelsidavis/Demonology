# ⛧ Demonology CLI ⛧

> *"Conjure the power of local LLMs through mystical terminal incantations"*

A terminal-only CLI application that provides a **Claude Code-like interface** but connects to your local **llama.cpp backend** with an OpenAI-compatible API. Experience the arcane arts of AI interaction through beautifully themed terminal interfaces with streaming responses, conversation management, and tool integration.

## ✨ Features

### 🛡️ **Ultra-Stability System** ⭐ NEW!
- **Auto-Restart**: Automatic server restart with VRAM clearing
- **Connection Retry**: 5x retry attempts with exponential backoff  
- **Health Monitoring**: Real-time connection health checks
- **Error Recovery**: Smart recovery from 500 errors and timeouts
- **Multi-GPU Support**: Optimized for dual GPU setups (RTX 5080 + RTX 3060)

### 🧠 **Intelligent Context Management** ⭐ NEW!
- **Smart Trimming**: Preserves important messages at 85% context usage
- **Real-time Monitoring**: Color-coded warnings (Green/Yellow/Red)
- **Accurate Tracking**: 26768 token context matching your server
- **Context Commands**: `/context`, `/optimize`, `/trim smart`

### 🎨 **Mystical Theming System**
- **Amethyst**: Deep purple mysticism (default)
- **Infernal**: Red/amber flames of knowledge  
- **Stygian**: Teal depths of ancient wisdom

### 🌊 **Streaming Response Display**
- Real-time streaming text display with heartbeat monitoring
- Animated loading messages with mystical flavor
- Smooth, responsive terminal interface

### 🛠️ **Advanced Tool Integration**
- **File Operations**: Read, write, list files safely with automatic project scaffolding
- **Web Search**: Search the internet for current information using DuckDuckGo
- **Reddit Search**: Access community discussions and insights from Reddit
- **Project Planning**: Generate and automatically execute complete project plans
- **Code Execution**: Run Python, JavaScript, and Bash in sandboxed environments
- **Image Generation**: Create AI-generated textures, icons, and visual assets for games
- **Extensible Framework**: Easy to add custom tools

### 💬 **Advanced Conversation Management**
- Save/load conversation history
- Search through past conversations
- Export conversations in multiple formats (JSON, Markdown, Text)
- Conversation tagging and metadata

### ⚙️ **Flexible Configuration**
- YAML-based configuration files
- Runtime theme switching
- Model and endpoint customization
- Permissive mode toggle

## 🚀 Quick Start

**See `QUICKSTART.md` for the complete 2-minute setup guide!**

### ⚡ **Ultra-Fast Launch**

**Terminal 1: Start Server**
```bash
cd /home/k/Desktop/Demonology  
./llama-server-nosudo.sh
```

**Terminal 2: Launch Demonology**
```bash
demonology
```

**That's it!** 🎉 You now have ultra-stable sessions with automatic restart and intelligent context management.

### 📋 **Essential Commands**
- `/help` - Show all commands
- `/context` - Context usage with color warnings  
- `/optimize` - Smart context optimization
- `/restart` - Manual server restart + VRAM clear

### Installation (for Development)

```bash
# Clone the repository
git clone https://github.com/demonology-dev/demonology-cli.git
cd demonology-cli

# Install dependencies
pip install -r requirements.txt

# Install Demonology
pip install -e .
```

### Basic Usage

```bash
# Start Demonology with default settings
demonology

# Use different theme
demonology --theme infernal

# Connect to different model/endpoint
demonology --model llama-3-70b --base-url http://localhost:8080/v1

# Enable permissive mode
demonology --permissive
```

## 📋 Prerequisites

### Required
- Python 3.8 or higher
- A running **llama.cpp** server with OpenAI-compatible API
- Terminal with Unicode support for mystical symbols

### Optional
- **Node.js** (for JavaScript code execution)
- **Python 3** (for Python code execution)
- Text editor for configuration editing

## ⚙️ Configuration

Grimoire uses a YAML configuration file stored at:
- Linux/macOS: `~/.config/demonology/config.yaml`
- Windows: `%APPDATA%\\grimoire\\config.yaml`

### Default Configuration

```yaml
api:
  base_url: \"http://127.0.0.1:8080/v1\"
  model: \"Qwen-3-Coder-30B\"
  max_tokens: 1024
  temperature: 0.7
  top_p: 0.95
  timeout: 60.0

ui:
  theme: \"amethyst\"  # Options: amethyst, infernal, stygian
  permissive_mode: false
  auto_save_conversations: true
  max_history_length: 1000

tools:
  enabled: true
  allowed_tools:
    - \"file_operations\"
    - \"code_execution\"
    - \"web_search\"
    - \"reddit_search\"
    - \"project_planning\"
    - \"image_generation\"
  working_directory: null  # null means use current directory
```

### Runtime Configuration

You can modify settings during runtime:

```
# Change theme
/theme stygian

# Toggle permissive mode
/permissive

# Change model
/model llama-3-70b

# Edit configuration file
/config edit
```

## 🎮 Commands

### Chat Commands
- Simply type your message to chat with the AI
- Multi-line input supported
- Streaming responses with live updates

### System Commands
| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/quit`, `/exit`, `/q` | Exit Demonology |
| `/status` | Show current status and configuration |
| `/themes` | List available themes with previews |
| `/theme <name>` | Change theme (amethyst, infernal, stygian) |
| `/permissive` | Toggle permissive mode |
| `/model <name>` | Change or show current model |
| `/tools` | List available tools and their status |
| `/clear` | Clear conversation history |

### Conversation Management
| Command | Description |
|---------|-------------|
| `/save <filename>` | Save current conversation |
| `/load <filename>` | Load conversation from file |
| `/list` | List saved conversations |
| `/search <query>` | Search conversations |

### Configuration
| Command | Description |
|---------|-------------|
| `/config` | Show current configuration |
| `/config edit` | Open configuration file in editor |

## 🛠️ Enhanced Tools System

Demonology includes a comprehensive suite of AI-powered tools that enable proactive project creation and information gathering:

### 🔍 Search & Information Tools

#### Web Search
- **General web search**: Uses DuckDuckGo for current information
- **Automatic activation**: AI uses this when uncertain about facts
- **Structured results**: Returns titles, URLs, and snippets
- **No API keys required**: Works out of the box

#### Reddit Search
- **Community insights**: Search Reddit discussions and user experiences
- **Dual API support**: PRAW (authenticated) or public JSON API (fallback)
- **Subreddit filtering**: Search specific communities or all of Reddit
- **Rich results**: Scores, comments, authors, and full context

### 🏗️ Project Creation Tools

#### Project Planning & Execution
- **Complete project creation**: "Build me a Python web app" → Full project structure
- **Technology-specific scaffolding**: Python, C++, JavaScript, React, and more
- **Automatic execution**: Plans are automatically implemented, not just generated
- **Production-ready structure**: Includes Makefiles, package.json, requirements.txt, etc.

**Supported Project Types:**
- **Python**: Complete package structure with tests, requirements, and entry points
- **C++**: Makefile-based projects with proper compilation flags
- **JavaScript/Node.js**: NPM projects with scripts and testing setup  
- **React**: Full React applications with components and build configuration
- **Generic**: Flexible structure for any technology

#### File Operations
- **Safe file management**: Read, write, list files with security boundaries
- **Automatic directories**: Creates directory structures as needed
- **Extension filtering**: Optional file type restrictions for security
- **Project integration**: Works seamlessly with project planning tool

### ⚡ Code Execution
- **Multi-language support**: Python, JavaScript, and Bash
- **Sandboxed execution**: Isolated environments with timeout protection
- **Real-time output**: Streaming results with error handling
- **Development integration**: Execute build commands, run tests, compile projects

### 🎨 Image Generation
- **AI-powered visuals**: Generate textures, icons, and assets for video games
- **Multiple APIs**: Pollinations.ai, Hugging Face FLUX.1, and Craiyon fallback
- **Style options**: Realistic, artistic, anime, fantasy, pixel-art, concept-art
- **Automatic saving**: Generated images saved to working directory
- **No API keys required**: Works out of the box with free services

### 🤖 Proactive AI Behavior

The AI actively uses tools when appropriate:
- **Web search**: Automatically searches for current information when uncertain
- **Reddit search**: Finds community discussions for user experience insights
- **Project creation**: When asked to "build" or "create", automatically generates complete projects
- **File operations**: Creates necessary files and structures without explicit requests
- **Image generation**: Creates visual assets when asked for textures, icons, or graphics

### 🔧 Configuration & Setup

#### Optional Reddit API Setup
For enhanced Reddit search with higher rate limits:

```bash
# Set environment variables
export REDDIT_CLIENT_ID="your_client_id"
export REDDIT_CLIENT_SECRET="your_client_secret" 
export REDDIT_USER_AGENT="your_app_name"
```

Create Reddit API credentials at: https://www.reddit.com/prefs/apps

#### Tool Configuration
Enable/disable tools in your config file:

```yaml
tools:
  enabled: true
  allowed_tools:
    - "file_operations"      # File system operations
    - "web_search"           # General web search  
    - "reddit_search"        # Reddit community search
    - "project_planning"     # Complete project creation
    - "code_execution"       # Code running capabilities
    - "image_generation"     # AI image generation
```

### 🛡️ Security Features
- **Safe root boundaries**: All file operations restricted to configured directories
- **Command filtering**: Dangerous system commands are blocked
- **Timeout protection**: All code execution is time-limited
- **Sandboxed environments**: Tools run in isolated contexts

### 🔌 Extensibility
Create custom tools by:

1. Inheriting from the `Tool` base class
2. Implementing `execute()` and `to_openai_function()` methods  
3. Adding parameter handling in `ToolRegistry`
4. Registering in `_register_default_tools()`

## 🎨 Theme Gallery

### Amethyst Theme (Default)
- **Primary**: Deep purple (#9966CC)
- **Accent**: Lavender (#E6E6FA)  
- **Symbols**: 𓃶 🜏 𖤐 𐕣
- **Aesthetic**: Mystical purple magic

### Infernal Theme
- **Primary**: Deep red (#CC3333)
- **Accent**: Light orange (#FFCC99)
- **Symbols**: 👿 ⁶⁶⁶ 🜄 𖤍
- **Aesthetic**: Demonic flames and power

### Stygian Theme
- **Primary**: Dark cyan (#008B8B)
- **Accent**: Pale turquoise (#AFEEEE)
- **Symbols**: ⊕ ⊗ ⊙ ○
- **Aesthetic**: Deep ocean mysteries

## 🔮 Mystical Loading Messages

Demonology entertains you while waiting with rotating mystical messages:

- *\"Consulting the Book of Shadows… 𓃶\"*
- *\"Invoking minor daemon: patience… 🜏\"*
- *\"Conjuring purple flames of the TUI… ⁶⁶⁶\"*
- *\"Negotiating with the daemon for GPU cycles… 👿\"*
- And many more arcane proclamations...

## 🚨 Permissive Mode

When enabled, Demonology operates with reduced restrictions:

```
The seals weaken… PERMISSIVE MODE awakes.
Walls between realms: disabled.
Daemon unchained. May the logs forgive us.
```

**Use responsibly** - this mode provides more freedom but maintains ethical guidelines.

## 🔧 Development

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/demonology-dev/demonology-cli.git
cd demonology-cli

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .

# Run tests
pytest

# Format code
black demonology/
isort demonology/

# Type checking
mypy demonology/
```

### Project Structure

```
demonology/
├── demonology/                 # Main package
│   ├── __init__.py          # Package initialization
│   ├── cli.py               # Main CLI entry point
│   ├── client.py            # API client for llama.cpp
│   ├── config.py            # Configuration management
│   ├── themes.py            # Theming system
│   ├── ui.py                # Terminal UI components
│   ├── tools.py             # Tool integration framework
│   └── utils.py             # Utility functions
├── config.yaml              # Default configuration
├── pyproject.toml           # Project metadata
├── requirements.txt         # Core dependencies
├── requirements-dev.txt     # Development dependencies
└── README.md               # This file
```

## 🤝 Contributing

We welcome contributions to the Demonology project! Please see our contributing guidelines for:

- 🐛 Bug reports
- ✨ Feature requests  
- 🔧 Pull requests
- 📚 Documentation improvements

## 🙏 Acknowledgments

- **llama.cpp** team for the incredible local LLM inference
- **Rich** library for beautiful terminal interfaces
- **Claude** for inspiration in CLI design
- The open source AI community for pushing boundaries

## ⚡ Troubleshooting

### Connection Issues
```bash
# Test your llama.cpp server
curl http://127.0.0.1:8080/v1/models

# Check Demonology connection
demonology --debug
```

### Theme Problems
```bash
# Reset to default theme
/theme amethyst

# Check terminal Unicode support
echo \"𓃶 🜏 𖤐 𐕣\"
```

### Tool Execution Issues
```bash
# Check Python availability
python3 --version

# Check Node.js (for JavaScript tools)
node --version

# Verify file permissions
ls -la ~/.config/demonology/
```

---

*\"Through terminal and daemon, through code and incantation, may your queries be answered and your knowledge expanded.\"* ⛧⛤

**Happy conjuring with Demonology CLI!** 🔮✨


## 📜 License

Copyright (c) 2025 Kelsi Davis

Permission is granted to use, copy, modify, and distribute this software 
for personal, educational, or research purposes only. Commercial use, 
including selling, offering paid services, or integrating into 
commercial products, requires prior written permission from the author.
