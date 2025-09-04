# 📜 Demonology Scripts Reference

Complete reference for all server and utility scripts.

## 🚀 **Production Scripts** (Use These)

### `llama-server-nosudo.sh` ✅ **MAIN SERVER SCRIPT**
**Purpose**: Primary production server script with all stability features

**Features**:
- ✅ No sudo required
- ✅ Continuous batching (`--cont-batching`)
- ✅ Memory defragmentation (`--defrag-thold 0.1`)
- ✅ 8 parallel processing slots
- ✅ NUMA optimization (`--numa isolate`)
- ✅ f16 cache types for memory efficiency
- ✅ Dual GPU support (RTX 5080 + RTX 3060)
- ✅ 26768 context tokens
- ✅ Graceful shutdown handling

**Usage**:
```bash
./llama-server-nosudo.sh
```

### `restart-llama.sh` ✅ **AUTO-RESTART SCRIPT**
**Purpose**: Used by Demonology for automatic server restart

**Features**:
- ✅ Kills existing llama-server processes
- ✅ Clears GPU memory (nvidia-smi --gpu-reset)
- ✅ Starts llama-server-nosudo.sh in background
- ✅ Logs output to /tmp/llama-server.log

**Usage**: Called automatically by Demonology, or manually:
```bash
./restart-llama.sh
```

## 🔧 **Alternative Scripts**

### `auto-restart-server.sh` ⚙️ **MONITORING SCRIPT**
**Purpose**: Full monitoring solution with health checks

**Features**:
- 🔄 Monitors server health every 30 seconds
- 🔄 Rate-limited restarts (max 10/hour)
- 🔄 Launches Demonology automatically
- 🔄 Comprehensive logging and error handling

**Usage**:
```bash
./auto-restart-server.sh
```

### `llama-server-stable.sh` ⚠️ **LEGACY (REQUIRES SUDO)**
**Purpose**: Original enhanced server script

**Issues**:
- ❌ Requires sudo password
- ❌ Includes system-level optimizations that need admin access
- ❌ Uses unsupported --timeout-read/--timeout-write parameters

**Status**: Superseded by `llama-server-nosudo.sh`

## 📁 **File Locations**

```
/home/k/Desktop/Demonology/
├── llama-server-nosudo.sh     ← Main server script ✅
├── restart-llama.sh           ← Auto-restart helper ✅
├── auto-restart-server.sh     ← Full monitoring (alternative)
└── llama-server-stable.sh     ← Legacy (don't use)

Logs:
├── /tmp/llama-server.log      ← Server output log
```

## 🎯 **Which Script to Use When**

| Scenario | Use This Script | Why |
|----------|----------------|-----|
| **Normal usage** | `llama-server-nosudo.sh` | Main production script, no issues |
| **Auto-restart needed** | `restart-llama.sh` | Called automatically by Demonology |
| **Full monitoring** | `auto-restart-server.sh` | Alternative all-in-one approach |
| **Legacy/troubleshooting** | `llama-server-stable.sh` | Only if others don't work (requires sudo) |

## 🔧 **Script Parameters**

### Key Settings (in llama-server-nosudo.sh):
```bash
CONTEXT=26768          # Context window size
NGL=50                 # GPU layers
THREADS=20             # CPU threads  
BATCH_SIZE=170         # Batch size
PARALLEL=8             # Parallel slots
CACHE_TYPE_K="f16"     # Key cache type
CACHE_TYPE_V="f16"     # Value cache type
DEFRAG_THOLD=0.1       # Memory defrag threshold
```

### Model & Binary Paths:
```bash
LLAMA_BIN="/home/k/llama.cpp/build/bin/llama-server"
MODEL_PATH="/media/k/vbox1/models/Qwen3/Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf"
```

## 🚨 **Troubleshooting**

### Server Won't Start
1. Check if port 8080 is busy: `sudo netstat -tulpn | grep 8080`
2. Kill existing processes: `pkill -f llama-server`
3. Check model path exists: `ls -la /media/k/vbox1/models/Qwen3/`
4. Check binary exists: `ls -la /home/k/llama.cpp/build/bin/llama-server`

### Permission Issues
- ✅ All scripts are designed to run without sudo
- ✅ User-level optimizations only (ulimit, etc.)
- ✅ No system-level changes required

### Performance Issues  
- Check GPU memory: `nvidia-smi`
- Monitor logs: `tail -f /tmp/llama-server.log`
- Verify dual GPU usage in server startup output

## 🛡️ **Security Notes**

- ✅ **No sudo required** for any production scripts
- ✅ **Safe process management** using pkill with specific patterns
- ✅ **Isolated execution** via nohup and background processes
- ✅ **Proper cleanup** with signal handling and graceful shutdown