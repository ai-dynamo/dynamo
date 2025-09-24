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

```
Feature           │ 1. dev + `run.sh`     │ 2. local-dev + `run.sh`  │ 3. local-dev + Dev Container
──────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────
Default User      │ root                  │ ubuntu                   │ ubuntu
──────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────
User Setup        │ None                  │ Matches UID/GID of       │ Matches UID/GID of
                  │                       │ `build.sh` user          │ `build.sh` user
──────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────
Permissions       │ root                  │ ubuntu with sudo         │ ubuntu with sudo
──────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────
Home Directory    │ /root                 │ /home/ubuntu             │ /home/ubuntu
──────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────
Working Directory │ /workspace            │ /workspace               │ /home/ubuntu/dynamo
──────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────
Rust Toolchain    │ System install        │ User install (~/.rustup, │ User install (~/.rustup,
                  │ (/usr/local/rustup,   │  ~/.cargo)               │  ~/.cargo)
                  │  /usr/local/cargo)    │                          │
──────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────
Python Env        │ root owned            │ User owned venv          │ User owned venv
──────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────
File Permissions  │ root-level            │ user-level, safe         │ user-level, safe
──────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────
Compatibility     │ Legacy workflows,     │ workspace writable on NFS│workspace writable on NFS
                  │   workspace not       │                          │
                  │   writable on NFS     │                          │
──────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────
```

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
run.sh --mount-workspace --image dynamo:latest-vllm-local-dev ...
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
- **Network Configuration**: Host networking for service communication
- **Resource Limits**: Memory, file descriptors, and IPC configuration

**Common Usage Examples:**

```bash
# Basic container launch (inference/production)
./run.sh --image dynamo:latest-vllm

# Mount workspace for development (use local-dev image for local user permissions)
./run.sh --image dynamo:latest-vllm-local-dev --mount-workspace

# Use specific image and framework for development
./run.sh --image v0.1.0.dev.08cc44965-vllm-local-dev --framework vllm --mount-workspace

# Interactive development shell with workspace mounted
./run.sh --image dynamo:latest-vllm-local-dev --mount-workspace -it -- bash

# Development with custom environment variables
./run.sh --image dynamo:latest-vllm-local-dev -e CUDA_VISIBLE_DEVICES=0,1 --mount-workspace

# Production inference without GPU access
./run.sh --image dynamo:latest-vllm --gpus none

# Dry run to see docker command
./run.sh --dry-run

# Development with custom volume mounts
./run.sh --image dynamo:latest-vllm-local-dev -v /host/path:/container/path --mount-workspace
```

## Workflow Examples

### Development Workflow
```bash
# 1. Build local-dev image (creates both dynamo:latest-vllm and dynamo:latest-vllm-local-dev)
./build.sh --framework vllm --target local-dev

# 2. Run development container using the local-dev image
./run.sh --image dynamo:latest-vllm-local-dev --mount-workspace -it

# 3. Inside container, run inference (requires both frontend and backend)
# Start frontend
python -m dynamo.frontend &

# Start backend (vLLM example)
python -m dynamo.vllm --model Qwen/Qwen3-0.6B --gpu-memory-utilization 0.50 &
```

### Production Workflow
```bash
# 1. Build production image
./build.sh --framework vllm --release-build

# 2. Run production container
./run.sh --image dynamo:latest-vllm-local-dev --gpus all
```

### Testing Workflow
```bash
# 1. Build with no cache for clean build
./build.sh --framework vllm --no-cache

# 2. Test container functionality (--image defaults to dynamo:latest-vllm)
./run.sh --mount-workspace -it -- python -m pytest tests/
```
