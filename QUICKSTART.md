# 🚀 Demonology Quick Start Guide

Get up and running with ultra-stable llama server sessions in 2 minutes.

## ⚡ **Quick Launch** (Recommended)

### Terminal 1: Start Server
```bash
cd /home/k/Desktop/Demonology
./llama-server-nosudo.sh
```
Wait for: `server is listening on http://0.0.0.0:8080`

### Terminal 2: Launch Demonology  
```bash
demonology
```

**That's it!** 🎉 You now have:
- ✅ Ultra-stable server with auto-restart
- ✅ Intelligent context management
- ✅ 26768 token context (matches your server)
- ✅ Dual GPU optimization (RTX 5080 + RTX 3060)

---

## 🔄 **Alternative: Fully Automated**

**One command for everything:**
```bash
./auto-restart-server.sh
```
This will:
- Start the llama server
- Launch Demonology automatically  
- Monitor and restart on crashes
- Handle VRAM clearing

---

## 📋 **Essential Commands**

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/context` | Show context usage (with color warnings) |
| `/optimize` | Smart context optimization |
| `/restart` | Manual server restart + VRAM clear |
| `/trim smart` | Intelligent message trimming |
| `/quit` | Exit Demonology |

---

## 🔧 **Troubleshooting**

### Server Won't Start?
```bash
# Check if port 8080 is in use
sudo netstat -tulpn | grep 8080

# Kill existing servers
pkill -f llama-server

# Try again
./llama-server-nosudo.sh
```

### Demonology Connection Issues?
1. **Auto-restart**: It will try to restart server automatically
2. **Manual restart**: Use `/restart` command within Demonology
3. **Check logs**: `tail -f /tmp/llama-server.log`

### Context Full Warnings?
- **Auto-handled**: Smart trimming at 85% usage
- **Manual control**: `/optimize` or `/trim smart`
- **Monitor**: `/context` shows detailed stats

---

## 📊 **What You Get**

### **Server Optimizations**
- 🔄 Continuous batching for efficiency
- 💾 Memory defragmentation (0.1 threshold)  
- ⚡ 8 parallel processing slots
- 🧠 NUMA isolation for GPU performance
- 🎯 f16 cache types for memory efficiency

### **Client Features**
- 🔁 5x retry attempts with exponential backoff
- ⏱️ 12-minute streaming timeouts (720s)
- 🔗 HTTP keep-alive connections (5min expiry)
- 💓 Heartbeat monitoring (120s timeout)
- 🧠 Smart context management (26768 tokens)

### **Auto-Recovery**
- 🚨 Detects server crashes and 500 errors
- 🔄 Automatic server restart with VRAM clearing
- 📊 Real-time context monitoring with warnings
- 💾 Intelligent message preservation during trimming

---

## 🎯 **Pro Tips**

1. **Monitor context**: Use `/context` regularly to see usage
2. **Long sessions**: Let smart trimming handle context automatically
3. **GPU memory**: Server clears VRAM on restart automatically
4. **Performance**: Use the dual-terminal setup for best results
5. **Logs**: Check `/tmp/llama-server.log` for server issues

---

**Need help?** Check `STABILITY.md` for detailed technical information.