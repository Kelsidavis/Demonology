# 📚 Demonology Documentation Index

Complete guide to all documentation and getting started with Demonology.

## 🚀 **Start Here**

### 📖 **For New Users**
1. **[QUICKSTART.md](QUICKSTART.md)** ⭐ **START HERE!**
   - 2-minute setup guide
   - Essential commands
   - Troubleshooting basics

2. **[README.md](README.md)** 
   - Complete feature overview
   - Installation instructions
   - Basic usage examples

### 🛡️ **For Power Users**
3. **[STABILITY.md](STABILITY.md)**
   - Ultra-stability system details
   - Client-side improvements
   - Server-side optimizations
   - Technical specifications

4. **[SCRIPTS.md](SCRIPTS.md)**
   - Complete script reference
   - Which script to use when
   - Configuration parameters
   - Troubleshooting guide

## 🔧 **Reference Documentation**

### ⚙️ **Setup & Configuration**
- **[INSTALL.md](INSTALL.md)** - Detailed installation guide
- **[config.yaml](config.yaml)** - Configuration file with comments

### 🛠️ **Features & Tools**
- **[TOOLS.md](TOOLS.md)** - Available tools and their usage
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and updates

## 📋 **Quick Reference**

### **Essential Files**
| File | Purpose |
|------|---------|
| `QUICKSTART.md` | 2-minute setup guide ⚡ |
| `README.md` | Main project overview |
| `STABILITY.md` | Technical stability details |
| `SCRIPTS.md` | Script reference guide |

### **Key Scripts**  
| Script | Purpose |
|--------|---------|
| `llama-server-nosudo.sh` | Main server (use this) ✅ |
| `restart-llama.sh` | Auto-restart helper |
| `auto-restart-server.sh` | Full monitoring alternative |
| `demonology` | Main CLI application |

### **Essential Commands**
| Command | Description |
|---------|-------------|
| `demonology` | Launch the main application |
| `/help` | Show all available commands |
| `/context` | Show context usage with warnings |
| `/optimize` | Smart context optimization |
| `/restart` | Manual server restart + VRAM clear |
| `/quit` | Exit application |

## 🎯 **Documentation Structure**

```
Demonology Documentation
├── 🚀 Getting Started
│   ├── QUICKSTART.md      ← Start here (2 min setup)
│   ├── README.md          ← Project overview
│   └── INSTALL.md         ← Detailed installation
│
├── 🛡️ Stability & Performance  
│   ├── STABILITY.md       ← Ultra-stability system
│   └── SCRIPTS.md         ← Script reference
│
├── 🛠️ Features & Tools
│   ├── TOOLS.md           ← Available tools
│   └── config.yaml        ← Configuration
│
└── 📚 Reference
    ├── CHANGELOG.md       ← Version history
    └── DOCUMENTATION.md   ← This file
```

## 🆘 **Getting Help**

### **Quick Issues**
1. **Server won't start**: Check `QUICKSTART.md` troubleshooting
2. **Connection problems**: Use `/restart` command in Demonology
3. **Context issues**: Use `/context` to check usage, `/optimize` to fix
4. **Script confusion**: Check `SCRIPTS.md` for which one to use

### **Common Questions**
- **"Which script should I use?"** → `llama-server-nosudo.sh` (see SCRIPTS.md)
- **"How do I check if it's working?"** → See QUICKSTART.md setup verification
- **"Context is full, what do I do?"** → Use `/optimize` or `/trim smart`
- **"Server keeps disconnecting"** → The system auto-restarts (see STABILITY.md)

### **Deep Dive Topics**
- **Technical details**: Read STABILITY.md
- **Script configuration**: Read SCRIPTS.md  
- **Tool customization**: Read TOOLS.md
- **Advanced setup**: Read INSTALL.md

## 🔄 **Updates & Maintenance**

### **Keeping Documentation Updated**
- Documentation is automatically updated when new features are added
- Check CHANGELOG.md for recent changes
- Configuration examples are kept in sync with actual config files

### **Version Information**
- All scripts and documentation updated for current Demonology version
- Compatibility verified with llama.cpp server build 6183
- Optimized for Qwen3 30B model with dual GPU setup

---

**💡 Tip**: Bookmark this file as your documentation hub. Everything links back here!