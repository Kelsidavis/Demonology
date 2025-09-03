# 📦 Demonology Installation Guide

This guide will help you install and set up Demonology CLI on your system.

## 🔧 System Requirements

### Required
- **Python 3.8+** - Check with `python3 --version`
- **pip** - Python package installer
- **Terminal with Unicode support** - For mystical symbols
- **llama.cpp server** - Running with OpenAI-compatible API

### Optional
- **Node.js** - For JavaScript code execution tools
- **Git** - For development installation
- **Reddit API credentials** - For enhanced Reddit search features

## 🚀 Installation Methods

### Method 1: PyPI Installation (Recommended)

```bash
# Install from PyPI (when published)
pip install demonology-cli

# Verify installation
demonology --help
```

### Method 2: Development Installation

```bash
# Clone repository
git clone https://github.com/demonology-dev/demonology-cli.git
cd demonology-cli

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e \".[dev]\"
```

### Method 3: Local Installation from Source

```bash
# Download and extract source
# cd into demonology-cli directory

# Install dependencies
pip install -r requirements.txt

# Install package
pip install .
```

## 🐳 Docker Installation (Alternative)

```bash
# Build Docker image
docker build -t demonology-cli .

# Run container
docker run -it --rm demonology-cli

# With volume for configuration persistence
docker run -it --rm -v ~/.config/demonology:/root/.config/demonology demonology-cli
```

## ⚙️ Setting Up llama.cpp Server

Demonology requires a running llama.cpp server with OpenAI-compatible API:

### Option 1: llama.cpp Server

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build with server support
make server

# Download a model (example with Qwen)
./server -m path/to/your/model.gguf --port 8080 --host 0.0.0.0 --api-key your-key
```

### Option 2: Ollama (Alternative)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull qwen2.5-coder:7b

# Serve with OpenAI-compatible API
ollama serve --port 11434
```

### Option 3: Text Generation WebUI

```bash
# Clone repository
git clone https://github.com/oobabooga/text-generation-webui.git
cd text-generation-webui

# Install dependencies
pip install -r requirements.txt

# Run with OpenAI API extension
python server.py --extensions openai --listen-port 8080
```

## 📁 Configuration Setup

### Automatic Configuration
Demonology will create default configuration on first run:

```bash
# First run creates config directory and files
demonology

# Configuration will be created at:
# Linux/macOS: ~/.config/grimoire/config.yaml
# Windows: %APPDATA%\\grimoire\\config.yaml
```

### Manual Configuration
You can create the configuration manually:

```bash
# Create config directory
mkdir -p ~/.config/grimoire

# Copy default config
cp config.yaml ~/.config/grimoire/config.yaml

# Edit configuration
nano ~/.config/grimoire/config.yaml
```

### Configuration Options

```yaml
api:
  base_url: \"http://127.0.0.1:8080/v1\"  # Your llama.cpp server URL
  model: \"Qwen-3-Coder-30B\"              # Model name
  max_tokens: 1024
  temperature: 0.7
  top_p: 0.95
  timeout: 60.0

ui:
  theme: \"amethyst\"                       # amethyst, infernal, stygian
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
  working_directory: null
```

### Optional: Reddit API Setup

For enhanced Reddit search with higher rate limits and full API access:

```bash
# 1. Create Reddit app at https://www.reddit.com/prefs/apps
# 2. Set environment variables
export REDDIT_CLIENT_ID="your_client_id"
export REDDIT_CLIENT_SECRET="your_client_secret" 
export REDDIT_USER_AGENT="your_app_name:v1.0"

# 3. Install PRAW (optional, fallback to public API if not available)
pip install praw>=7.6.0
```

**Note:** Reddit search works without API credentials using the public JSON API, but authenticated access provides better rate limits and more features.

## 🧪 Testing Installation

### Quick Test
```bash
# Test basic installation
demonology --version

# Test with help
demonology --help

# Test configuration
demonology --config ~/.config/demonology/config.yaml --theme infernal --help
```

### Connection Test
```bash
# Start Demonology (it will test connection)
demonology

# You should see the banner and connection test
# If connection fails, check your llama.cpp server
```

### Manual API Test
```bash
# Test your API endpoint directly
curl -X POST http://127.0.0.1:8080/v1/chat/completions \\
  -H \"Content-Type: application/json\" \\
  -d '{
    \"model\": \"your-model\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}],
    \"max_tokens\": 50
  }'
```

## 🛠️ Troubleshooting Installation

### Python Version Issues

```bash
# Check Python version
python3 --version

# If using older Python, upgrade
# Ubuntu/Debian
sudo apt update && sudo apt install python3.10

# macOS with Homebrew
brew install python@3.10

# Windows: Download from python.org
```

### Permission Errors

```bash
# Use virtual environment (recommended)
python3 -m venv demonology-env
source demonology-env/bin/activate  # Linux/macOS
# demonology-env\Scripts\activate  # Windows
pip install demonology-cli

# Or use user installation
pip install --user grimoire-cli
```

### Missing Dependencies

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt install python3-dev python3-pip

# macOS
xcode-select --install
brew install python

# Update pip
pip install --upgrade pip setuptools wheel
```

### Unicode/Symbol Issues

```bash
# Test Unicode support
python3 -c \"print('𓃶 🜏 𖤐 𐕣')\"

# If symbols don't display correctly:
# - Use a terminal with good Unicode support (iTerm2, Windows Terminal, etc.)
# - Install appropriate fonts (Noto, Fira Code, etc.)
# - Set terminal encoding to UTF-8
```

### Configuration Directory Issues

```bash
# Check configuration directory permissions
ls -la ~/.config/demonology/

# Fix permissions if needed
chmod 755 ~/.config/demonology/
chmod 644 ~/.config/demonology/config.yaml

# Reset configuration
rm -rf ~/.config/demonology/
demonology  # This will recreate defaults
```

### Tool Execution Issues

```bash
# For JavaScript tools - install Node.js
# Ubuntu/Debian
sudo apt install nodejs npm

# macOS
brew install node

# Windows: Download from nodejs.org

# For Python tools - ensure Python 3 is available
which python3

# For bash tools - ensure bash is available (usually default on Unix systems)
which bash
```

## 🔄 Updating Demonology

### PyPI Installation
```bash
# Update to latest version
pip install --upgrade demonology-cli

# Check new version
demonology --version
```

### Development Installation
```bash
# Pull latest changes
cd demonology-cli
git pull origin main

# Reinstall
pip install -e .
```

## 🗑️ Uninstallation

```bash
# Uninstall package
pip uninstall demonology-cli

# Remove configuration (optional)
rm -rf ~/.config/demonology/

# Remove virtual environment if used
rm -rf grimoire-env/
```

## 🐳 Docker Installation Details

### Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY demonology/ ./demonology/
COPY pyproject.toml .
RUN pip install .

ENTRYPOINT ["demonology"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  demonology:
    build: .
    stdin_open: true
    tty: true
    volumes:
      - ~/.config/demonology:/root/.config/grimoire
    depends_on:
      - llama-server
  
  llama-server:
    image: ghcr.io/ggerganov/llama.cpp:server
    ports:
      - \"8080:8080\"
    volumes:
      - ./models:/models
    command: --model /models/your-model.gguf --port 8080 --host 0.0.0.0
```

---

## 📞 Getting Help

If you encounter issues during installation:

1. **Check the logs** - Run with `--debug` flag
2. **Verify requirements** - Ensure all prerequisites are met
3. **Test components individually** - API server, Python environment, etc.
4. **Check GitHub Issues** - Someone might have encountered the same problem
5. **Ask for help** - Create an issue with detailed error information

**Happy installing! May your setup be swift and your conjurations successful!** ⛧✨