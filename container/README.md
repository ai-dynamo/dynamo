# Container Development Guide

## Overview

The NVIDIA Dynamo project uses containerized development and deployment to maintain consistent environments across different AI inference frameworks and deployment scenarios. This directory contains the tools for building and running Dynamo containers:

### Core Components

- **`build.sh`** - A Docker image builder that creates containers for different AI inference frameworks (vLLM, TensorRT-LLM, SGLang). It handles framework-specific dependencies, multi-stage builds, and development vs production configurations.

- **`run.sh`** - A container runtime manager that launches Docker containers with proper GPU access, volume mounts, and environment configurations. It supports different development workflows from root-based legacy setups to user-based development environments.

- **Multiple Dockerfiles** - Framework-specific Dockerfiles that define the container images:
  - `Dockerfile.vllm` - For vLLM inference backend
  - `Dockerfile.trtllm` - For TensorRT-LLM inference backend
  - `Dockerfile.sglang` - For SGLang inference backend
  - `Dockerfile` - Base/standalone configuration

### Why Containerization?

Each inference framework (vLLM, TensorRT-LLM, SGLang) has specific CUDA versions, Python dependencies, and system libraries. Containers provide consistent environments, framework isolation, and proper GPU configurations across development and production.

The scripts in this directory abstract away the complexity of Docker commands while providing fine-grained control over build and runtime configurations.

### Convenience Scripts vs Direct Docker Commands

The `build.sh` and `run.sh` scripts are convenience wrappers that simplify common Docker operations. They automatically handle:
- Framework-specific image selection and tagging
- GPU access configuration and runtime selection
- Volume mount setup for development workflows
- Environment variable management
- Build argument construction for multi-stage builds

**You can always use Docker commands directly** if you prefer more control or want to customize beyond what the scripts provide. The scripts use `--dry-run` flags to show you the exact Docker commands they would execute, making it easy to understand and modify the underlying operations.

## Development Targets Feature Matrix

These targets are specified with `build.sh --target <target>` and correspond to Docker multi-stage build targets defined in the Dockerfiles (e.g., `FROM somebase AS <target>`). Some commonly used targets include:

- `runtime` - For running pre-built containers without development tools (minimal size)
- `dev` - For development (inferencing/benchmarking/etc, runs as root user)
- `local-dev` - For development with local user permissions matching host UID/GID. This is useful when mounting host partitions (with local user permissions) to Docker partitions.

Additional targets are available in the Dockerfiles for specific build stages and use cases.

| Feature | **dev + `run.sh`** | **local-dev + `run.sh`** | **local-dev + Dev Container** |
|---------|-------------------|--------------------------|-------------------------------|
| **Default User** | root | ubuntu | ubuntu |
| **User Setup** | None | Matches UID/GID of `build.sh` user | Matches UID/GID of `build.sh` user |
| **Permissions** | root | ubuntu with sudo | ubuntu with sudo |
| **Home Directory** | `/root` | `/home/ubuntu` | `/home/ubuntu` |
| **Working Directory** | `/workspace` | `/workspace` | `/workspace` |
| **Rust Toolchain** | System install (`/usr/local/rustup`, `/usr/local/cargo`) | User install (`~/.rustup`, `~/.cargo`) | User install (`~/.rustup`, `~/.cargo`) |
| **Python Env** | root owned | User owned venv | User owned venv |
| **File Permissions** | root-level | user-level, safe | user-level, safe |
| **Compatibility** | Legacy workflows, workspace not writable on NFS | workspace writable on NFS | workspace writable on NFS |

## Environment Variables Across Build Stages

Understanding how environment variables change across different build stages is crucial for development and debugging. The Dynamo build system uses a multi-stage Docker build process where environment variables are set, inherited, and overridden at different stages.

### Build Stage Flow

```
Dockerfile → base → dev (dynamo-base image)
              ↓
Dockerfile.vllm → framework → runtime → dev (vllm dev image)
                                         ↓
Dockerfile.local_dev → local-dev (from vllm dev image)
```

### Environment Variables by Stage

| Variable | **base** | **base→dev** | **vllm→framework** | **vllm→runtime** | **vllm→dev** | **local-dev** |
|----------|----------|--------------|--------------------|-----------------|--------------|--------------------|
| **DYNAMO_HOME** | ❌ Not set | `/opt/dynamo` | ❌ Not set | `/opt/dynamo` | `/workspace` ✅ **OVERRIDE** | `/workspace` (inherited) |
| **WORKSPACE_DIR** | ❌ Not set | ❌ Not set | ❌ Not set | ❌ Not set | `/workspace` | `/workspace` (inherited) |
| **CARGO_TARGET_DIR** | ❌ Not set | `/opt/dynamo/target` | ❌ Not set | ❌ Not set | `/workspace/target` ✅ **OVERRIDE** | `/workspace/target` (inherited) |
| **VIRTUAL_ENV** | `/opt/dynamo/venv` | (inherited) | `/opt/dynamo/venv` | `/opt/dynamo/venv` | `/opt/dynamo/venv` ✅ **REDEFINE** | `/opt/dynamo/venv` (inherited) |
| **RUSTUP_HOME** | `/usr/local/rustup` | (inherited) | ❌ Not set | ❌ Not set | `/usr/local/rustup` | `/home/ubuntu/.rustup` ✅ **OVERRIDE** |
| **CARGO_HOME** | `/usr/local/cargo` | (inherited) | ❌ Not set | ❌ Not set | `/usr/local/cargo` | `/home/ubuntu/.cargo` ✅ **OVERRIDE** |
| **USERNAME** | ❌ Not set | ❌ Not set | ❌ Not set | ❌ Not set | ❌ Not set | `ubuntu` ✅ **NEW** |
| **HOME** | (system default) | (system default) | (system default) | (system default) | (system default) | `/home/ubuntu` ✅ **NEW** |
| **PATH** | (includes cargo) | (inherited) | (system default) | (includes venv, etcd, ucx) | `/usr/local/cargo/bin:$PATH` | `/home/ubuntu/.cargo/bin:$PATH` ✅ **OVERRIDE** |

### Key Insights

**1. DYNAMO_HOME Dual Purpose:**
- `base→dev` and `vllm→runtime`: `/opt/dynamo` - For **installed/packaged** Dynamo (CI, production)
- `vllm→dev` and `local-dev`: `/workspace` - For **development** with source code mounted from host

**2. Rust Toolchain Location:**
- `dev` target: System-wide at `/usr/local/rustup` and `/usr/local/cargo` (suitable for root)
- `local-dev` target: User-specific at `/home/ubuntu/.rustup` and `/home/ubuntu/.cargo` (proper UID/GID ownership)

**3. Build Artifacts Location:**
- `base→dev`: `/opt/dynamo/target` - Build artifacts with installed package
- `vllm→dev` onward: `/workspace/target` - Build artifacts in mounted workspace for persistence

**4. Variables That Stay Constant:**
- `VIRTUAL_ENV`: Always `/opt/dynamo/venv` (ownership changes in local-dev via rsync)
- `WORKSPACE_DIR`: Always `/workspace` once set in vllm→dev
- `DYNAMO_HOME`: Always `/workspace` once overridden in vllm→dev (for development)

**5. local-dev Specific Changes:**
From `Dockerfile.local_dev`, the Rust toolchain is moved to user home because:
- Workspace mount points may change, breaking toolchain paths
- User needs ownership of cargo binaries and registry for package installation
- Toolchain requires consistent system paths that don't depend on workspace location

The Python venv ownership is also updated via rsync in local-dev to match the user's UID/GID, ensuring package installation permissions work correctly.

## Usage Guidelines

- **Use dev + `run.sh`**: for command-line testing. Runs as root user
- **Use local-dev + `run.sh`**: for command-line development and Docker mounted partitions using your local user ID
- **Use local-dev + Dev Container**: VS Code/Cursor Dev Container Plugin, using your local user ID

## Example Commands

### 1. dev + `run.sh` (runs as root):
```bash
run.sh ...
```

### 2. local-dev + `run.sh` (runs as the local user):
```bash
run.sh --mount-workspace -it --image dynamo:latest-vllm-local-dev ...
```

### 3. local-dev + Dev Container Extension:
Use VS Code/Cursor Dev Container Extension with devcontainer.json configuration

## Build and Run Scripts Overview

### build.sh - Docker Image Builder

The `build.sh` script is responsible for building Docker images for different AI inference frameworks. It supports multiple frameworks and configurations:

**Purpose:**
- Builds Docker images for NVIDIA Dynamo with support for vLLM, TensorRT-LLM, SGLang, or standalone configurations
- Handles framework-specific dependencies and optimizations
- Manages build contexts, caching, and multi-stage builds
- Configures development vs production targets

**Key Features:**
- **Framework Support**: vLLM (default when --framework not specified), TensorRT-LLM, SGLang, or NONE
- **Multi-stage Builds**: Build process with base images
- **Development Targets**: Supports `dev` target and `local-dev` target
- **Build Caching**: Docker layer caching and sccache support
- **GPU Optimization**: CUDA, EFA, and NIXL support

**Common Usage Examples:**

```bash
# Build vLLM dev image called dynamo:latest-vllm (default). This runs as root and is fine to use for inferencing/benchmarking, etc.
./build.sh

# Build both development and local-dev images (integrated into build.sh). While the dev image runs as root, the local-dev image will run as the local user, which is useful when mounting partitions. It will also contain development tools.
./build.sh --framework vllm --target local-dev

# Build TensorRT-LLM development image called dynamo:latest-trtllm
./build.sh --framework trtllm

# Build with custom tag
./build.sh --framework sglang --tag my-custom-tag

# Dry run to see commands
./build.sh --dry-run

# Build with no cache
./build.sh --no-cache

# Build with build arguments
./build.sh --build-arg CUSTOM_ARG=value
```

### build.sh --dev-image - Local Development Image Builder

The `build.sh --dev-image` option takes a dev image and then builds a local-dev image, which contains proper local user permissions. It also includes extra developer utilities (debugging tools, text editors, system monitors, etc.).

**Common Usage Examples:**

```bash
# Build local-dev image from dev image dynamo:latest-vllm
./build.sh --dev-image dynamo:latest-vllm --framework vllm

# Build with custom tag from dev image dynamo:latest-vllm
./build.sh --dev-image dynamo:latest-vllm --framework vllm --tag my-local:dev

# Dry run to see what would be built
./build.sh --dev-image dynamo:latest-vllm --framework vllm --dry-run
```

### run.sh - Container Runtime Manager

The `run.sh` script launches Docker containers with the appropriate configuration for development and inference workloads.

**Purpose:**
- Runs pre-built Dynamo Docker images with proper GPU access
- Configures volume mounts, networking, and environment variables
- Supports different development workflows (root vs user-based)
- Manages container lifecycle and resource allocation

**Key Features:**
- **GPU Management**: Automatic GPU detection and allocation
- **Volume Mounting**: Workspace and HuggingFace cache mounting
- **User Management**: Root or user-based container execution
- **Network Configuration**: Configurable networking modes (host, bridge, none, container sharing)
- **Resource Limits**: Memory, file descriptors, and IPC configuration

**Common Usage Examples:**

```bash
# Basic container launch (inference/production, runs as root user)
./run.sh --image dynamo:latest-vllm -v $HOME/.cache:/home/ubuntu/.cache

# Mount workspace for development (use local-dev image for local host user permissions)
./run.sh --image dynamo:latest-vllm-local-dev --mount-workspace -it -v $HOME/.cache:/home/ubuntu/.cache

# Use specific image and framework for development
./run.sh --image v0.1.0.dev.08cc44965-vllm-local-dev --framework vllm --mount-workspace -it -v $HOME/.cache:/home/ubuntu/.cache

# Interactive development shell with workspace mounted
./run.sh --image dynamo:latest-vllm-local-dev --mount-workspace -v $HOME/.cache:/home/ubuntu/.cache -it -- bash

# Development with custom environment variables
./run.sh --image dynamo:latest-vllm-local-dev -e CUDA_VISIBLE_DEVICES=0,1 --mount-workspace -it -v $HOME/.cache:/home/ubuntu/.cache

# Dry run to see docker command
./run.sh --dry-run

# Development with custom volume mounts
./run.sh --image dynamo:latest-vllm-local-dev -v /host/path:/container/path --mount-workspace -it -v $HOME/.cache:/home/ubuntu/.cache
```

### Network Configuration Options

The `run.sh` script supports different networking modes via the `--network` flag (defaults to `host`):

#### Host Networking (Default)
```bash
# Same examples with local host user permissions
./run.sh --image dynamo:latest-vllm-local-dev --network host -v $HOME/.cache:/home/ubuntu/.cache
./run.sh --image dynamo:latest-vllm-local-dev -v $HOME/.cache:/home/ubuntu/.cache
```
**Use cases:**
- High-performance ML inference (default for GPU workloads)
- Services that need direct host port access
- Maximum network performance with minimal overhead
- Sharing services with the host machine (NATS, etcd, etc.)

**⚠️ Port Sharing Limitation:** Host networking shares all ports with the host machine, which means you can only run **one instance** of services like NATS (port 4222) or etcd (port 2379) across all containers and the host.

#### Bridge Networking (Isolated)
```bash
# CI/testing with isolated bridge networking and host cache sharing
./run.sh --image dynamo:latest-vllm --mount-workspace -it --network bridge -v $HOME/.cache:/home/ubuntu/.cache
```
**Use cases:**
- Secure isolation from host network
- CI/CD pipelines requiring complete isolation
- When you need absolute control of ports
- Exposing specific services to host while maintaining isolation

**Note:** For port sharing with the host, use the `--port` or `-p` option with format `host_port:container_port` (e.g., `--port 8000:8000` or `-p 9081:8081`) to expose specific container ports to the host.

#### No Networking ⚠️ **LIMITED FUNCTIONALITY**
```bash
# Complete network isolation - no external connectivity
./run.sh --image dynamo:latest-vllm --network none --mount-workspace -it -v $HOME/.cache:/home/ubuntu/.cache

# Same with local user permissions
./run.sh --image dynamo:latest-vllm-local-dev --network none --mount-workspace -it -v $HOME/.cache:/home/ubuntu/.cache
```
**⚠️ WARNING: `--network none` severely limits Dynamo functionality:**
- **No model downloads** - HuggingFace models cannot be downloaded
- **No API access** - Cannot reach external APIs or services
- **No distributed inference** - Multi-node setups won't work
- **No monitoring/logging** - External monitoring systems unreachable
- **Limited debugging** - Cannot access external debugging tools

**Very limited use cases:**
- Pre-downloaded models with purely local processing
- Air-gapped security environments (models must be pre-staged)

#### Container Network Sharing
Use `--network container:name` to share the network namespace with another container.

**Use cases:**
- Sidecar patterns (logging, monitoring, caching)
- Service mesh architectures
- Sharing network namespaces between related containers

See Docker documentation for `--network container:name` usage.

#### Custom Networks
Use custom Docker networks for multi-container applications. Create with `docker network create` and specify with `--network network-name`.

**Use cases:**
- Multi-container applications
- Service discovery by container name

See Docker documentation for custom network creation and management.

#### Network Mode Comparison

| Mode | Performance | Security | Use Case | Dynamo Compatibility | Port Sharing | Port Publishing |
|------|-------------|----------|----------|---------------------|---------------|-----------------|
| `host` | Highest | Lower | ML/GPU workloads, high-performance services | ✅ Full | ⚠️ **Shared with host** (one NATS/etcd only) | ❌ Not needed |
| `bridge` | Good | Higher | General web services, controlled port exposure | ✅ Full | ✅ Isolated ports | ✅ `-p host:container` |
| `none` | N/A | Highest | Air-gapped environments only | ⚠️ **Very Limited** | ✅ No network | ❌ No network |
| `container:name` | Good | Medium | Sidecar patterns, shared network stacks | ✅ Full | ⚠️ Shared with target container | ❌ Use target's ports |
| Custom networks | Good | Medium | Multi-container applications | ✅ Full | ✅ Isolated ports | ✅ `-p host:container` |

## Workflow Examples

### Development Workflow
```bash
# 1. Build local-dev image (creates both dynamo:latest-vllm and dynamo:latest-vllm-local-dev)
./build.sh --framework vllm --target local-dev

# 2. Run development container using the local-dev image
./run.sh --image dynamo:latest-vllm-local-dev --mount-workspace -v $HOME/.cache:/home/ubuntu/.cache -it

# 3. Inside container, run inference (requires both frontend and backend)
# Start frontend
python -m dynamo.frontend &

# Start backend (vLLM example)
python -m dynamo.vllm --model Qwen/Qwen3-0.6B --gpu-memory-utilization 0.20 &
```

### Production Workflow
```bash
# 1. Build production image
./build.sh --framework vllm --release-build

# 2. Run production container (runs as root)
./run.sh --image dynamo:latest-vllm --gpus all
```

### CI/CD Workflow
```bash
# 1. Build image for CI
./build.sh --framework vllm --no-cache

# 2. Run tests with network isolation for reproducible results
./run.sh --image dynamo:latest-vllm --mount-workspace -it --network bridge -v $HOME/.cache:/home/ubuntu/.cache -- python -m pytest tests/

# 3. Inside the container with bridge networking, start services
# Note: Services are only accessible from the same container - no port conflicts with host
nats-server -js &
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://0.0.0.0:2379 --data-dir /tmp/etcd &
python -m dynamo.frontend &

# 4. Start worker backend (choose one framework):
# vLLM
DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8081 python -m dynamo.vllm --model Qwen/Qwen3-0.6B --gpu-memory-utilization 0.20 --enforce-eager --no-enable-prefix-caching --max-num-seqs 64 &

# SGLang
DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8081 python -m dynamo.sglang --model Qwen/Qwen3-0.6B --mem-fraction-static 0.20 --max-running-requests 64 &

# TensorRT-LLM
DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8081 python -m dynamo.trtllm --model Qwen/Qwen3-0.6B --free-gpu-memory-fraction 0.20 --max-num-tokens 8192 --max-batch-size 64 &
```

**Framework-Specific GPU Memory Arguments:**
- **vLLM**: `--gpu-memory-utilization 0.20` (use 20% GPU memory), `--enforce-eager` (disable CUDA graphs), `--no-enable-prefix-caching` (save memory), `--max-num-seqs 64` (max concurrent sequences)
- **SGLang**: `--mem-fraction-static 0.20` (20% GPU memory for static allocation), `--max-running-requests 64` (max concurrent requests)
- **TensorRT-LLM**: `--free-gpu-memory-fraction 0.20` (reserve 20% GPU memory), `--max-num-tokens 8192` (max tokens in batch), `--max-batch-size 64` (max batch size)
