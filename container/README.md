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
  - `Dockerfile.frontend` - For Kubernetes Gateway API Inference Extension integration with EPP
  - `Dockerfile.epp` - For building the Endpoint Picker (EPP) image

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

**Note**: In Dynamo, "targets" and "Docker stages" are synonymous. Each target corresponds to a stage in the multi-stage Docker build. Similarly, "frameworks" and "engines" are synonymous (vLLM, TensorRT-LLM, SGLang).

| Feature | **runtime + `run.sh`** | **local-dev (`run.sh` or Dev Container)** | **dev + `run.sh`** (legacy) |
|---------|----------------------|-------------------------------------------|--------------------------|
| **Usage** | Benchmarking inference and deployments, non-root | Development, compilation, testing locally | Legacy workflows, root user, use with caution |
| **User** | dynamo (UID 1000) | dynamo (UID=host user) with sudo | root (UID 0, use with caution) |
| **Home Directory** | `/home/dynamo` | `/home/dynamo` | `/root` |
| **Working Directory** | `/workspace` (in-container or mounted) | `/workspace` (must be mounted w/ `--mount-workspace`) | `/workspace` (must be mounted w/ `--mount-workspace`) |
| **Rust Toolchain** | None (uses pre-built wheels) | System install (`/usr/local/rustup`, `/usr/local/cargo`) | System install (`/usr/local/rustup`, `/usr/local/cargo`) |
| **Cargo Target** | None | `/workspace/target` | `/workspace/target` |
| **Python Env** | venv (`/opt/dynamo/venv`) for vllm/trtllm, system site-packages for sglang | venv (`/opt/dynamo/venv`) for vllm/trtllm, system site-packages for sglang | venv (`/opt/dynamo/venv`) for vllm/trtllm, system site-packages for sglang |

## Usage Guidelines

- **Use runtime target**: for benchmarking inference and deployments. Runs as non-root `dynamo` user (UID 1000, GID 0) for security
- **Use local-dev + `run.sh`**: for command-line development and Docker mounted partitions. Runs as `dynamo` user with UID matched to your local user, GID 0. Add `-it` flag for interactive sessions
- **Use local-dev + Dev Container**: VS Code/Cursor Dev Container Plugin, using `dynamo` user with UID matched to your local user, GID 0
- **Use dev + `run.sh`**: Root user, use with caution. Runs as root for backward compatibility with early workflows

## Example Commands

### 1. runtime target (runs as non-root dynamo user):
```bash
# Build runtime image
./build.sh --framework vllm --target runtime

# Run runtime container
./run.sh --image dynamo:latest-vllm-runtime -it
```

### 2. local-dev + `run.sh` (runs as dynamo user with matched host UID/GID):
```bash
run.sh --mount-workspace -it --image dynamo:latest-vllm-local-dev ...
```

### 3. local-dev + Dev Container Extension:
Use VS Code/Cursor Dev Container Extension with devcontainer.json configuration. The `dynamo` user UID is automatically matched to your local user.

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

# Build both development and local-dev images (integrated into build.sh). While the dev image runs as root, the local-dev image will run as dynamo user with UID/GID matched to your host user, which is useful when mounting partitions. It will also contain development tools.
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

### Building the Frontend Image

The frontend image is a specialized container that includes the Dynamo components (NATS, etcd, dynamo, NIXL, etc) along with the Endpoint Picker (EPP) for Kubernetes Gateway API Inference Extension integration. This image is primarily used for inference gateway deployments.

**Step 1: Build the Custom Dynamo EPP Image**

Follow the instructions in [`deploy/inference-gateway/README.md`](../deploy/inference-gateway/README.md) under "Build the custom EPP image" section. This process:
- Clones the Gateway API Inference Extension repository
- Applies Dynamo-specific patches for custom routing
- Builds the Dynamo router as a static library
- Creates a custom EPP image with integrated Dynamo routing capabilities

**Step 2: Build the Dynamo Base Image**

The base image contains the core Dynamo runtime components, NATS server, etcd, and Python dependencies:
```bash
# Build the base dev image (framework=none for frontend-only deployment)
# Note: --framework none defaults ENABLE_MEDIA_NIXL=false
./build.sh --framework none --target dev
```

**Step 3: Build the Frontend Image**

Now build the frontend image that combines the Dynamo base with the EPP:

```bash
# 2. Build the frontend image using the pre-built EPP
docker buildx build --load --platform linux/amd64 \
  --build-arg DYNAMO_BASE_IMAGE=dynamo:latest-none-dev \
  --build-arg EPP_IMAGE={EPP_IMAGE_TAG} \
  --build-arg PYTHON_VERSION=3.12 \
  -f container/Dockerfile.frontend \
  -t dynamo:latest-none-frontend \
  .
```
#### Frontend Image Contents

The frontend image includes:
- **EPP (Endpoint Picker)**: Handles request routing and load balancing for inference gateway
- **Dynamo Runtime**: Core platform components and routing logic
- **NIXL**: NVIDIA InfiniBand Library for high-performance network communication
- **Benchmarking Tools**: Performance testing utilities (aiperf, aiconfigurator, etc)
- **Python Environment**: Virtual environment with all required dependencies
- **NATS Server**: Message broker for Dynamo's distributed communication
- **etcd**: Distributed key-value store for configuration and coordination

#### Deployment

The frontend image is designed for Kubernetes deployment with the Gateway API Inference Extension. See [`deploy/inference-gateway/README.md`](../deploy/inference-gateway/README.md) for complete deployment instructions using Helm charts.

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
- **User Management**: Non-root `dynamo` user execution (UID 1000, GID 0), with optional `--user` flag to override
- **Network Configuration**: Configurable networking modes (host, bridge, none, container sharing)
- **Resource Limits**: Memory, file descriptors, and IPC configuration
- **Interactive Mode**: Use `-it` flag for interactive terminal sessions (required for shells, debugging, and interactive development)

**Common Usage Examples:**

```bash
# Basic container launch with dev image (runs as root by default, non-interactive)
./run.sh --image dynamo:latest-vllm -v $HOME/.cache:/root/.cache

# Interactive development with workspace mounted using dev image (runs as root)
./run.sh --image dynamo:latest-vllm --mount-workspace -it -v $HOME/.cache:/home/dynamo/.cache

# Interactive development with local-dev image (runs as dynamo user with matched host UID/GID)
./run.sh --image dynamo:latest-vllm-local-dev --mount-workspace -it -v $HOME/.cache:/home/dynamo/.cache

# Use specific image and framework for development
./run.sh --image v0.1.0.dev.08cc44965-vllm-local-dev --framework vllm --mount-workspace -it -v $HOME/.cache:/home/dynamo/.cache

# Interactive development shell with workspace mounted (local-dev)
./run.sh --image dynamo:latest-vllm-local-dev --mount-workspace -v $HOME/.cache:/home/dynamo/.cache -it -- bash

# Development with custom environment variables
./run.sh --image dynamo:latest-vllm-local-dev -e CUDA_VISIBLE_DEVICES=0,1 --mount-workspace -it -v $HOME/.cache:/home/dynamo/.cache

# Dry run to see docker command
./run.sh --dry-run

# Development with custom volume mounts
./run.sh --image dynamo:latest-vllm-local-dev -v /host/path:/container/path --mount-workspace -it -v $HOME/.cache:/home/dynamo/.cache

# Run runtime image as non-root dynamo user (for production)
./run.sh --image dynamo:latest-vllm-runtime -v $HOME/.cache:/home/dynamo/.cache

# Run dev image as specific user (override default root)
./run.sh --image dynamo:latest-vllm --user dynamo -v $HOME/.cache:/home/dynamo/.cache
```

### Network Configuration Options

The `run.sh` script supports different networking modes via the `--network` flag (defaults to `host`):

#### Host Networking (Default)
```bash
# Examples with dynamo user
./run.sh --image dynamo:latest-vllm-local-dev --network host -v $HOME/.cache:/home/dynamo/.cache
./run.sh --image dynamo:latest-vllm-local-dev -v $HOME/.cache:/home/dynamo/.cache
```
**Use cases:**
- High-performance ML inference (default for GPU workloads)
- Services that need direct host port access
- Maximum network performance with minimal overhead
- Sharing services with the host machine (NATS, etcd, etc.)

**⚠️ Port Sharing Limitation:** Host networking shares all ports with the host machine, which means you can only run **one instance** of services like NATS (port 4222) or etcd (port 2379) across all containers and the host.

#### Bridge Networking (Isolated)
```bash
# CI/testing with isolated bridge networking and host cache sharing (no -it for automated CI)
./run.sh --image dynamo:latest-vllm --mount-workspace --network bridge -v $HOME/.cache:/home/dynamo/.cache
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
./run.sh --image dynamo:latest-vllm --network none --mount-workspace -it -v $HOME/.cache:/home/dynamo/.cache

# Same with local-dev image (dynamo user with matched host UID/GID)
./run.sh --image dynamo:latest-vllm-local-dev --network none --mount-workspace -it -v $HOME/.cache:/home/dynamo/.cache
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
./run.sh --image dynamo:latest-vllm-local-dev --mount-workspace -v $HOME/.cache:/home/dynamo/.cache -it

# 3. Inside container, run inference (requires both frontend and backend)
# Start frontend
python -m dynamo.frontend &

# Start backend (vLLM example)
python -m dynamo.vllm --model Qwen/Qwen3-0.6B --gpu-memory-utilization 0.20 &
```

### Production Workflow
```bash
# 1. Build production runtime image (runs as non-root dynamo user)
./build.sh --framework vllm --target runtime

# 2. Run production container as non-root dynamo user
./run.sh --image dynamo:latest-vllm-runtime --gpus all -v $HOME/.cache:/home/dynamo/.cache
```

### Testing Workflow
```bash
# 1. Build dev image
./build.sh --framework vllm --no-cache

# 2. Run tests with network isolation for reproducible results (no -it needed for CI)
./run.sh --image dynamo:latest-vllm --mount-workspace --network bridge -v $HOME/.cache:/home/dynamo/.cache -- python -m pytest tests/

# 3. Inside the container with bridge networking, start services
# Note: Services are only accessible from the same container - no port conflicts with host
nats-server -js &
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://0.0.0.0:2379 --data-dir /tmp/etcd &
python -m dynamo.frontend &

# 4. Start worker backend (choose one framework):
# vLLM
DYN_SYSTEM_PORT=8081 python -m dynamo.vllm --model Qwen/Qwen3-0.6B --gpu-memory-utilization 0.20 --enforce-eager --no-enable-prefix-caching --max-num-seqs 64 &

# SGLang
DYN_SYSTEM_PORT=8081 python -m dynamo.sglang --model Qwen/Qwen3-0.6B --mem-fraction-static 0.20 --max-running-requests 64 &

# TensorRT-LLM
DYN_SYSTEM_PORT=8081 python -m dynamo.trtllm --model Qwen/Qwen3-0.6B --free-gpu-memory-fraction 0.20 --max-num-tokens 8192 --max-batch-size 64 &
```

**Framework-Specific GPU Memory Arguments:**
- **vLLM**: `--gpu-memory-utilization 0.20` (use 20% GPU memory), `--enforce-eager` (disable CUDA graphs), `--no-enable-prefix-caching` (save memory), `--max-num-seqs 64` (max concurrent sequences)
- **SGLang**: `--mem-fraction-static 0.20` (20% GPU memory for static allocation), `--max-running-requests 64` (max concurrent requests)
- **TensorRT-LLM**: `--free-gpu-memory-fraction 0.20` (reserve 20% GPU memory), `--max-num-tokens 8192` (max tokens in batch), `--max-batch-size 64` (max batch size)
