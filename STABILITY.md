# 🛡️ Demonology Ultra-Stability System

Complete stability and auto-restart solution for uninterrupted llama server sessions.

## 🔧 Client-Side Improvements (Demonology)

### Enhanced Connection Handling
- **Automatic Retry Logic**: Exponential backoff with jitter (up to 5 retries)
- **Extended Timeouts**: 4x longer read timeouts for streaming responses (720s total)
- **Connection Pooling**: HTTP keep-alive connections with 5-minute expiry
- **Heartbeat Monitoring**: Detects stalled connections (120s timeout)
- **HTTP/2 Fallback**: Uses HTTP/2 if available, falls back to HTTP/1.1

### Intelligent Context Management
- **Accurate Context Tracking**: 26768 tokens (matches your server config)
- **Smart Auto-Trimming**: Preserves recent + important messages at 85% usage
- **Real-time Monitoring**: Color-coded warnings (Green/Yellow/Red)
- **Context Commands**: `/context`, `/optimize`, `/trim smart`

### Automatic Server Restart
- **Connection Health Check**: Tests server on startup
- **Auto-Restart on Failure**: Detects 500 errors and connection issues
- **VRAM Clearing**: Kills processes + GPU reset if available
- **Manual Restart**: `/restart` command for immediate server restart

## 🚀 Server-Side Improvements (llama.cpp)

### Production Server Script: `llama-server-nosudo.sh` ✅

#### Memory & Performance Optimization
- **Continuous Batching**: `--cont-batching` for efficient memory usage
- **Memory Defragmentation**: `--defrag-thold 0.1` prevents fragmentation
- **Optimized Cache**: f16 cache types for better memory efficiency
- **Parallel Processing**: 8 parallel slots for better throughput
- **NUMA Optimization**: `--numa isolate` for better GPU utilization

#### Multi-GPU Support
- **Dual GPU**: Automatically uses RTX 5080 + RTX 3060
- **Layer Splitting**: Optimized layer distribution across GPUs
- **VRAM Optimization**: 50 layers offloaded to GPU

#### Process Management
- **No Sudo Required**: User-level optimizations only
- **Graceful Shutdown**: Proper signal handling and cleanup
- **Background Operation**: Runs via nohup for stability

### Automatic Restart System: `restart-llama.sh`
- **Process Cleanup**: `pkill -f llama-server`
- **VRAM Clearing**: GPU memory reset if nvidia-smi available
- **Background Launch**: Starts server via nohup with logging
- **Status Logging**: Output to `/tmp/llama-server.log`

## 🎯 Usage Instructions

### 🚀 **Recommended Setup** (Fully Automated)

**Step 1: Start the Server**
```bash
cd /home/k/Desktop/Demonology
./llama-server-nosudo.sh
```

**Step 2: Launch Demonology** (in separate terminal)
```bash
demonology
```

### 🔄 **Alternative: Full Auto-Restart Manager**
```bash
# Starts server + Demonology with continuous monitoring:
./auto-restart-server.sh
```
Features:
- Monitors server health every 30 seconds
- Restarts server automatically if it crashes  
- Rate limits restarts to prevent loops
- Launches Demonology client automatically

### ⚙️ **Configuration**
Your `config.yaml` now includes:
```yaml
api:
  timeout: 180.0      # 3 minutes
  max_retries: 5      # More attempts
  retry_delay: 3.0    # Longer delays
```

### 📋 **New Commands**
- **`/restart`**: Manually restart server and clear VRAM 
- **`/context`**: Show detailed context usage statistics
- **`/optimize`**: Intelligent context optimization 
- **`/trim smart`**: Smart context trimming

### ✅ **Expected Behavior**
- **Startup Health Check**: Automatically checks server on launch, restarts if needed
- **Automatic Recovery**: Client will retry on 500 errors and attempt server restart
- **Better Error Messages**: Shows attempt counts and retry info  
- **Longer Sessions**: Much more stable long-running conversations
- **Connection Reuse**: Faster subsequent requests via keep-alive
- **VRAM Management**: Automatically clears VRAM on server restarts

## 🚨 Troubleshooting

### If You Still Experience Issues:

1. **Check Server Logs**: The enhanced script provides detailed startup info
2. **Monitor System Resources**: Ensure adequate RAM/VRAM for your model
3. **Network Issues**: Check for firewall or network configuration problems
4. **Model Size**: Large models may need even longer timeouts

### Advanced Tweaking:
- Increase `timeout` in config.yaml for very slow responses
- Adjust `max_retries` and `retry_delay` for your network conditions
- Modify server script timeouts if needed

## 📊 Performance Impact

### Benefits:
- ✅ Much more stable long sessions
- ✅ Automatic error recovery
- ✅ Better connection reuse
- ✅ Reduced disconnection frequency

### Trade-offs:
- ⚠️ Slightly slower initial connection setup
- ⚠️ Higher memory usage on server
- ⚠️ More aggressive system optimization

These changes should dramatically improve your session stability while maintaining excellent performance for your Qwen3 30B model.