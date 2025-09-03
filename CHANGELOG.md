# 📜 Demonology CLI Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Plugin system for custom tools
- Multiple model support per session

### Changed
- Performance optimizations planned
- UI/UX improvements

## [0.2.0] - 2024-09-03

### Added

#### 🔍 **Advanced Search & Information Tools**
- **Web Search Tool**
  - DuckDuckGo integration for current information
  - Automatic activation when AI is uncertain
  - Structured results with titles, URLs, and snippets
  - No API keys required - works out of the box
- **Reddit Search Tool**
  - Search Reddit discussions and community insights
  - Dual API support: PRAW (authenticated) + public JSON API (fallback)
  - Subreddit filtering and advanced search options
  - Rich results with scores, comments, authors, and context
  - Optional Reddit API credentials for enhanced features

#### 🏗️ **Complete Project Creation System**
- **Project Planning & Execution Tool**
  - AI can now build complete projects from simple requests
  - Technology-specific scaffolding (Python, C++, JavaScript, React)
  - Automatic plan execution - creates actual files and structure
  - Production-ready templates with build files, tests, and documentation
- **Enhanced File Operations**
  - Automatic project directory creation
  - Template-based file generation
  - Integration with project planning system

#### 🤖 **Proactive AI Behavior**
- AI now actively uses tools when appropriate
- Web search triggered automatically for uncertain information
- Reddit search used for community insights and user experiences
- Project creation happens end-to-end without manual steps
- Enhanced system prompts for better tool utilization

#### 🛠️ **Supported Project Types**
- **Python**: Complete package with tests, requirements, entry points
- **C++**: Makefile-based projects with compilation flags
- **JavaScript/Node.js**: NPM projects with scripts and testing
- **React**: Full React applications with components and build config
- **Generic**: Flexible structure for any technology

### Enhanced
- **Tool System Architecture**
  - Better parameter handling for all tools
  - Improved error messages and fallback behavior
  - Enhanced security with safe boundaries
- **Configuration System**
  - Updated default tool list to include new capabilities
  - Better documentation and examples

### Technical Details

#### **New Dependencies**
- `requests>=2.28.0` - HTTP requests for web and Reddit APIs
- `praw>=7.6.0` (optional) - Enhanced Reddit API access

#### **Security Features**
- All web requests use appropriate timeouts and error handling
- Reddit API credentials are optional with secure fallback
- File operations maintain safe root boundaries
- Project creation respects security constraints

## [0.1.0] - 2024-01-XX

### Added

#### 🎨 **Mystical Theming System**
- **Amethyst theme** - Deep purple mysticism (default)
- **Infernal theme** - Red/amber flames of knowledge  
- **Stygian theme** - Teal depths of ancient wisdom
- Runtime theme switching with `/theme` command
- Themed symbols and decorations for each theme
- Rich terminal UI with consistent color schemes

#### 🌊 **Streaming Response System**
- Real-time streaming text display from llama.cpp API
- Server-sent events (SSE) parsing
- Live updating conversation display
- Smooth terminal animations
- Mystical loading messages with rotating content

#### 🛠️ **Tool Integration Framework**
- **File Operations Tool**
  - Safe file reading with extension filtering
  - File writing with directory auto-creation
  - Directory listing and navigation
  - File deletion with safety checks
- **Code Execution Tool**
  - Python code execution in isolated environment
  - JavaScript execution via Node.js
  - Bash command execution with timeout protection
  - Sandboxed execution for security
- **Extensible tool system** for custom integrations
- Tool registry and management system

#### 💬 **Advanced Conversation Management**
- Conversation history persistence
- Save/load conversations with metadata
- Conversation search by content and tags
- Export conversations in multiple formats (JSON, Markdown, Text)
- Auto-save functionality
- Conversation metadata tracking

#### 🔧 **Robust Configuration System**
- YAML-based configuration files
- XDG Base Directory compliance
- Runtime configuration updates
- CLI argument overrides
- Configuration validation and error handling

#### 🎮 **Interactive CLI Interface**
- Click-based command line argument parsing
- Interactive command system (`/help`, `/status`, etc.)
- Graceful signal handling (Ctrl+C)
- Multi-line input support
- Command history and editing

#### 🔮 **Mystical Features**
- Rotating mystical loading messages
- Permissive mode with arcane warnings
- Themed error messages and notifications
- Unicode symbol support for mystical aesthetics
- Easter eggs and flavor text

#### 🔌 **API Integration**
- Full llama.cpp OpenAI-compatible API support
- Streaming and non-streaming response modes
- Connection testing and error handling
- Configurable model parameters
- Request/response validation

#### 🧪 **Development Features**
- Comprehensive test suite structure
- Type hints throughout codebase
- Async/await pattern implementation
- Logging and debugging support
- Code formatting and linting configuration

### Technical Details

#### **Dependencies**
- `httpx>=0.25.0` - Async HTTP client for API communication
- `rich>=13.0.0` - Rich terminal rendering and theming
- `click>=8.0.0` - Command line interface framework
- `pyyaml>=6.0.0` - Configuration file handling
- Python 3.8+ support

#### **Architecture**
- Modular design with clear separation of concerns
- Async/await throughout for responsive UI
- Plugin-style tool system
- Configuration-driven behavior
- Rich terminal UI with live updates

#### **Security Features**
- File operation safety checks
- Code execution sandboxing
- Timeout protection for all operations
- Input validation and sanitization
- Safe file extension filtering

### Installation
- PyPI package ready (`pip install demonology-cli`)
- Development installation support
- Docker container support
- Cross-platform compatibility (Linux, macOS, Windows)

### Documentation
- Comprehensive README with examples
- Detailed installation guide
- API documentation
- Configuration reference
- Troubleshooting guide

---

## Version Notes

### Version 0.1.0 - \"The First Incantation\"
This initial release establishes the core mystical framework of Demonology CLI. The foundation includes streaming chat interface, theming system, tool integration, and conversation management. Perfect for early adopters and developers looking to interact with local LLMs through a beautifully themed terminal interface.

**Breaking Changes**: None (initial release)

**Migration Guide**: None (initial release)

---

## Future Roadmap

### 0.2.0 - \"Enhanced Mysticism\"
- Advanced tool system with more integrations
- Conversation branching and merging
- Plugin system for custom tools
- Performance optimizations

### 0.3.0 - \"Realm Expansion\" 
- Multiple model support per session
- Advanced conversation search
- Custom theme creation

### 1.0.0 - "Demonology"
- Stable API
- Complete documentation
- Production-ready features

---

*\"May each version bring new powers to your terminal incantations...\"* ⛧
