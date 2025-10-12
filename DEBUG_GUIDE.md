# Dynamo Frontend Debugging Guide

## Understanding the "No Responders" Error

The error you're seeing:
```
2025-10-12T23:28:47.349066Z ERROR http-request: dynamo_llm::http::service::openai: Internal server error: Failed to generate completions: no responders: no responders method=POST uri=/v1/chat/completions version=HTTP/1.1
```

This occurs when:
1. **NATS Request Plane**: No backend workers are registered/available to handle requests
2. **Service Discovery**: etcd doesn't have any registered backend instances
3. **Network Issues**: Backend workers can't connect to NATS or etcd

## VS Code Launch Configuration

Add these configurations to your `.vscode/launch.json` file (after the existing Rust configurations):

```json
{
    "name": "Debug Dynamo Frontend (NATS)",
    "type": "python",
    "request": "launch",
    "module": "dynamo.frontend",
    "args": ["--http-port=8000"],
    "env": {
        "DYN_REQUEST_PLANE": "nats",
        "DYN_HTTP_RPC_PORT": "8085",
        "DYN_HTTP_HOST": "0.0.0.0",
        "DYN_HTTP_PORT": "8000",
        "PYTHONPATH": "${workspaceFolder}/components/src:${workspaceFolder}/components/frontend/src",
        "RUST_LOG": "debug",
        "RUST_BACKTRACE": "1"
    },
    "console": "integratedTerminal",
    "cwd": "${workspaceFolder}",
    "justMyCode": false,
    "stopOnEntry": false,
    "python": "${workspaceFolder}/dynamo/bin/python"
},
{
    "name": "Debug Dynamo Frontend (HTTP Mode)",
    "type": "python",
    "request": "launch",
    "module": "dynamo.frontend",
    "args": ["--http-port=8000"],
    "env": {
        "DYN_REQUEST_PLANE": "http",
        "DYN_HTTP_RPC_HOST": "0.0.0.0",
        "DYN_HTTP_RPC_PORT": "8081",
        "DYN_HTTP_HOST": "0.0.0.0",
        "DYN_HTTP_PORT": "8000",
        "PYTHONPATH": "${workspaceFolder}/components/src:${workspaceFolder}/components/frontend/src",
        "RUST_LOG": "debug",
        "RUST_BACKTRACE": "1"
    },
    "console": "integratedTerminal",
    "cwd": "${workspaceFolder}",
    "justMyCode": false,
    "stopOnEntry": false,
    "python": "${workspaceFolder}/dynamo/bin/python"
},
{
    "name": "Debug with Verbose Logging",
    "type": "python",
    "request": "launch",
    "module": "dynamo.frontend",
    "args": ["--http-port=8000"],
    "env": {
        "DYN_REQUEST_PLANE": "nats",
        "DYN_HTTP_RPC_PORT": "8085",
        "DYN_HTTP_HOST": "0.0.0.0",
        "DYN_HTTP_PORT": "8000",
        "DYN_ETCD_HOST": "localhost",
        "DYN_ETCD_PORT": "2379",
        "DYN_NATS_HOST": "localhost",
        "DYN_NATS_PORT": "4222",
        "PYTHONPATH": "${workspaceFolder}/components/src:${workspaceFolder}/components/frontend/src",
        "RUST_LOG": "dynamo_runtime=debug,dynamo_llm=debug,nats=debug",
        "RUST_BACKTRACE": "full"
    },
    "console": "integratedTerminal",
    "cwd": "${workspaceFolder}",
    "justMyCode": false,
    "stopOnEntry": false,
    "python": "${workspaceFolder}/dynamo/bin/python"
}
```

## Debugging Steps

### 1. Check Prerequisites

Before debugging, ensure these services are running:

```bash
# Check if etcd is running
ps aux | grep etcd
# Or start etcd if not running
etcd

# Check if NATS is running (if using NATS mode)
ps aux | grep nats
# Or start NATS if not running
nats-server

# Check if any backend workers are running
ps aux | grep dynamo
```

### 2. Verify Service Discovery

Check if any backend instances are registered in etcd:

```bash
# Install etcdctl if not available
# sudo apt-get install etcd-client

# Check registered instances
etcdctl get --prefix v1/instances/

# Check for specific namespaces
etcdctl get --prefix v1/instances/namespace.

# Look for backend workers
etcdctl get --prefix v1/instances/ | grep -i backend
```

### 3. Debug with Different Request Planes

#### Option A: Use HTTP Mode (Recommended for debugging)

HTTP mode is easier to debug and doesn't require NATS:

```bash
# Terminal 1: Start a backend worker in HTTP mode
export DYN_REQUEST_PLANE=http
export DYN_HTTP_RPC_HOST=0.0.0.0
export DYN_HTTP_RPC_PORT=8081
python -m dynamo.backend.mocker  # or your preferred backend

# Terminal 2: Start frontend in HTTP mode
export DYN_REQUEST_PLANE=http
export DYN_HTTP_RPC_PORT=8085
python -m dynamo.frontend --http-port=8000
```

#### Option B: Debug NATS Mode

```bash
# Terminal 1: Start NATS server
nats-server

# Terminal 2: Start etcd
etcd

# Terminal 3: Start backend worker
export DYN_REQUEST_PLANE=nats
python -m dynamo.backend.mocker

# Terminal 4: Start frontend
export DYN_REQUEST_PLANE=nats
export DYN_HTTP_RPC_PORT=8085
python -m dynamo.frontend --http-port=8000
```

### 4. Test the Setup

Once both frontend and backend are running:

```bash
# Test with curl
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test-model",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'
```

### 5. Common Issues and Solutions

#### Issue: "No responders available"
**Cause**: No backend workers registered
**Solution**:
- Start a backend worker first
- Check etcd for registered instances: `etcdctl get --prefix v1/instances/`
- Ensure backend and frontend use the same request plane mode

#### Issue: "Connection refused" on port 8085
**Cause**: Port conflict or incorrect configuration
**Solution**:
- Check if port is in use: `lsof -i :8085`
- Use a different port: `export DYN_HTTP_RPC_PORT=8086`
- Ensure backend uses the same port

#### Issue: "etcd connection failed"
**Cause**: etcd not running or wrong address
**Solution**:
- Start etcd: `etcd`
- Check etcd status: `etcdctl endpoint health`
- Set correct etcd address: `export DYN_ETCD_HOST=localhost`

#### Issue: "NATS connection failed"
**Cause**: NATS server not running
**Solution**:
- Start NATS: `nats-server`
- Check NATS status: `nats-server --help`
- Set correct NATS address: `export DYN_NATS_HOST=localhost`

### 6. Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DYN_REQUEST_PLANE` | `nats` | Request plane mode: `nats` or `http` |
| `DYN_HTTP_RPC_HOST` | `0.0.0.0` | HTTP RPC bind address |
| `DYN_HTTP_RPC_PORT` | `8081` | HTTP RPC port (backend), `8085` (your config) |
| `DYN_HTTP_HOST` | `0.0.0.0` | Frontend HTTP bind address |
| `DYN_HTTP_PORT` | `8000` | Frontend HTTP port |
| `DYN_ETCD_HOST` | `localhost` | etcd server address |
| `DYN_ETCD_PORT` | `2379` | etcd server port |
| `DYN_NATS_HOST` | `localhost` | NATS server address |
| `DYN_NATS_PORT` | `4222` | NATS server port |
| `RUST_LOG` | `info` | Rust logging level |
| `RUST_BACKTRACE` | `0` | Rust backtrace level |

### 7. Debugging with Breakpoints

When using VS Code debugger:

1. **Set breakpoints** in key files:
   - `/home/ubuntu/dynamo/components/src/dynamo/frontend/main.py` - Frontend startup
   - Python binding files that interface with Rust runtime

2. **Key debugging points**:
   - Service discovery initialization
   - Router configuration
   - Request handling in the HTTP service

3. **Watch variables**:
   - `runtime` object
   - `engine` configuration
   - Environment variables

### 8. Log Analysis

Enable verbose logging to understand the flow:

```bash
export RUST_LOG="dynamo_runtime=debug,dynamo_llm=debug,nats=debug"
export RUST_BACKTRACE=full
```

Look for these log patterns:
- `"Registered instance"` - Backend registration
- `"No instances found"` - Service discovery failure
- `"NATS connection"` - NATS connectivity
- `"HTTP endpoint started"` - HTTP server startup

## Quick Fix: Switch to HTTP Mode

The fastest way to resolve the "no responders" issue is to switch to HTTP mode:

```bash
# Set environment variables
export DYN_REQUEST_PLANE=http
export DYN_HTTP_RPC_PORT=8085

# Start your command
python -m dynamo.frontend --http-port=8000
```

This bypasses NATS entirely and uses direct HTTP communication between frontend and backend.
