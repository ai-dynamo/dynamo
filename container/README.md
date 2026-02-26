# Container Development Guide

## Overview

The NVIDIA Dynamo project uses containerized development and deployment to maintain consistent environments across different AI inference frameworks and deployment scenarios. This directory contains the tools for building and running Dynamo containers:

### Rendering Requirements:
- Python
- Python Packages:
  - pyyaml
  - jinja2

### Core Components

- **`render.py`** - A render script used to generate Dockerfiles for AI inference frameworks (vLLM, TensorRT-LLM, SGLang) and the frontend image. The generated Dockerfile includes the needed multi-stage steps for development vs production configurations.

- **`run.sh`** - A container runtime manager that launches Docker containers with proper GPU access, volume mounts, and environment configurations. It supports different development workflows from root-based legacy setups to user-based development environments.

### Stage Summary for Frameworks

<details>
<summary>Show Stage Summary Table</summary>
Dockerfile General Structure

Below is a summary of the general file structure for the framework Dockerfile stages. Some exceptions exist.

| Stage/Filepath | Target |
| --- | --- |
| **STAGE dynamo_base** | **FROM ${BASE_IMAGE}** |
| /bin/uv, /bin/uvx | COPY from ghcr.io/astral-sh/uv:latest (→ framework, runtime) |
|  /usr/bin/nats-server | Downloaded from GitHub (→ runtime) |
|  /usr/local/bin/etcd/ | Downloaded from GitHub (→ runtime) |
|  /usr/local/rustup/ | Installed via rustup-init (→ wheel_builder, dev) |
|  /usr/local/cargo/ | Installed via rustup-init (→ wheel_builder, dev) |
|  /usr/local/cuda/ | Inherited from BASE_IMAGE (→ wheel_builder, runtime) |
| **STAGE: wheel_builder** | **FROM quay.io/pypa/manylinux_2_28_${ARCH_ALT}** |
|  /usr/local/ucx/ | Built from source (→ runtime)
|  /opt/nvidia/nvda_nixl/ | Built from source (→ runtime)
|  /opt/nvidia/nvda_nixl/lib64/ | Built from source (→ runtime)
|  /opt/dynamo/target/ | Cargo build output (→ runtime)
|  /opt/dynamo/dist/*.whl | Built wheels (→ runtime)
|  /opt/dynamo/dist/nixl/ | Built nixl wheels (→ runtime)
| **STAGE: framework** | **FROM ${BASE_IMAGE}** |
|  /opt/dynamo/venv/ | Created with uv venv (→ runtime)
|  /${FRAMEWORK_INSTALL} | Built framework (→ runtime)
| **STAGE: runtime** | **FROM ${RUNTIME_IMAGE}** |
|  /usr/local/cuda/{bin,include,nvvm}/ | COPY from dynamo_base |
|  /usr/bin/nats-server | COPY from dynamo_base |
|  /usr/local/bin/etcd/ | COPY from dynamo_base |
|  /usr/local/ucx/ | COPY from wheel_builder |
|  /opt/nvidia/nvda_nixl/ | COPY from wheel_builder |
|  /opt/dynamo/wheelhouse/ | COPY from wheel_builder |
|  /opt/dynamo/venv/ | COPY from framework |
|  /opt/vllm/ | COPY from framework |
|  /workspace/{tests,examples,deploy}/ |COPY from build context |
| **STAGE: dev** | **FROM runtime (via dev/Dockerfile.dev)** |
|  /usr/bin/, /usr/lib/, etc. | COPY from dynamo_tools (dev utilities, git, sudo, etc.) |
|  /usr/local/rustup/ | COPY from dynamo_tools |
|  /usr/local/cargo/ | COPY from dynamo_tools |
|  /usr/local/bin/maturin | COPY from dynamo_tools |
|  /opt/dynamo/venv/ | For SGLang: created with --system-site-packages, includes uv and maturin |
|  /workspace/ | Full source code copied from build context with editable install |
|  **💡 Recommendation** | **Use --mount-workspace with run.sh** for live editing (bind mount overrides baked-in code) |
|  PATH | Includes /opt/dynamo/venv/bin:/usr/local/cargo/bin |
|  umask 002 | Login shell sources /etc/profile.d/00-umask.sh for group-writable files |
| **STAGE: local-dev** | **FROM dev (via dev/Dockerfile.dev)** |
|  /home/dynamo/.rustup/ | COPY from /usr/local/rustup (user-writable) |
|  USER | dynamo (UID/GID remapped to match host user) |
|  **💡 Recommendation** | **Use --mount-workspace with run.sh** for live editing (bind mount overrides baked-in code) |
|  RUSTUP_HOME | /home/dynamo/.rustup |
|  CARGO_HOME | /home/dynamo/.cargo |
</details>

### Why Containerization?

Each inference framework (vLLM, TensorRT-LLM, SGLang) has specific CUDA versions, Python dependencies, and system libraries. Containers provide consistent environments, framework isolation, and proper GPU configurations across development and production.

The scripts in this directory abstract away the complexity of Docker commands while providing fine-grained control over build and runtime configurations.

### Convenience Scripts vs Direct Docker Commands

The `run.sh` script and rendering scripts are convenience that simplify common Docker operations. They automatically handle:
- GPU access configuration and runtime selection
- Volume mount setup for development workflows
- Environment variable management
- Build argument construction for multi-stage builds

**You can always use Docker commands directly** if you prefer more control or want to customize beyond what the scripts provide. The `run.sh` uses a `--dry-run` flag to show you the exact commands they would execute, making it easy to understand and modify the underlying operations.

## Development Targets Feature Matrix

**Note**: In Dynamo, "targets" and "Docker stages" are synonymous. Each target corresponds to a stage in the multi-stage Docker build. Similarly, "frameworks" and "engines" are synonymous (vLLM, TensorRT-LLM, SGLang).

| Feature | **runtime + `run.sh`** | **local-dev (`run.sh` or Dev Container)** | **dev + `run.sh`** (legacy) |
|---------|----------------------|-------------------------------------------|--------------------------|
| **Usage** | Benchmarking inference and deployments, non-root | Development, compilation, testing locally | Legacy workflows, root user, use with caution |
| **User** | dynamo (UID 1000) | dynamo (UID=host user) with sudo | root (UID 0, use with caution) |
| **Home Directory** | `/home/dynamo` | `/home/dynamo` | `/root` |
| **Working Directory** | `/workspace` (in-container or mounted) | `/workspace` (baked-in, optionally mounted w/ `--mount-workspace`) | `/workspace` (baked-in, optionally mounted w/ `--mount-workspace`) |
| **Rust Toolchain** | None (uses pre-built wheels) | System install (`/usr/local/rustup`, `/usr/local/cargo`) | System install (`/usr/local/rustup`, `/usr/local/cargo`) |
| **Cargo Target** | None | `/workspace/target` | `/workspace/target` |
| **Python Env** | venv (`/opt/dynamo/venv`) for vllm/trtllm, system site-packages for sglang | venv (`/opt/dynamo/venv`) for all frameworks (with --system-site-packages for sglang) | venv (`/opt/dynamo/venv`) for all frameworks (with --system-site-packages for sglang) |

**Note (SGLang)**: SGLang runtime uses system site-packages, but the `dev` and `local-dev` images create `/opt/dynamo/venv` with `--system-site-packages` for build tooling like `maturin` and `uv`.

## Usage Guidelines

- **Use runtime target**: for benchmarking inference and deployments. Runs as non-root `dynamo` user (UID 1000, GID 0) for security
- **Use local-dev + `run.sh`**: for command-line development and Docker mounted partitions. Runs as `dynamo` user with UID matched to your local user, GID 0. Add `-it` flag for interactive sessions
- **Use local-dev + Dev Container**: VS Code/Cursor Dev Container Plugin, using `dynamo` user with UID matched to your local user, GID 0
- **Use dev + `run.sh`**: Root user, use with caution. Runs as root for backward compatibility with early workflows

## Example Commands

### 1. runtime target (runs as non-root dynamo user):
```bash
# Build runtime image
python container/render.py --framework vllm --target runtime --output-short-filename
docker build -t dynamo:latest-vllm-runtime -f rendered.Dockerfile .

# Run runtime container
container/run.sh --image dynamo:latest-vllm-runtime -it
```

### 2. local-dev + `run.sh` (runs as dynamo user with matched host UID/GID):
```bash
run.sh --mount-workspace -it --image dynamo:latest-vllm-local-dev ...
```

### 3. local-dev + Dev Container Extension:
Use VS Code/Cursor Dev Container Extension with devcontainer.json configuration. The `dynamo` user UID is automatically matched to your local user.

## Build and Run Scripts Overview

### render.py - Docker Image Generator

The `render.py` script is responsible for generating Dockerfiles for different AI inference frameworks. It supports multiple frameworks and configurations:

**Purpose:**
- Generates Dockerfiles for NVIDIA Dynamo with support for vLLM, TensorRT-LLM, SGLang, or standalone configurations
- Handles framework-specific dependencies and optimizations
- Manages build contexts, caching, and multi-stage builds
- Configures development vs production targets

**Key Features:**
- **Framework Support**: vLLM (default when --framework not specified), TensorRT-LLM, SGLang, or NONE (standalone Dynamo)
- **Multi-stage Builds**: Build process with base images
- **Development Targets**: Supports `dev`, `runtime`, and `local-dev` targets via `render.py`.
- **Build Caching**: Docker layer caching and sccache support
- **GPU Optimization**: CUDA, EFA, and NIXL support

#### BuildKit cache mounts in Dockerfiles

The framework Dockerfiles use BuildKit cache mounts (`RUN --mount=type=cache,...`) to reduce repeated downloads across builds. These caches are stored in Docker/BuildKit’s cache storage on the host (not in your host `~/.cache`), and are shared across builds that use the same builder.

Common cache mount targets:
- `--mount=type=cache,target=/root/.cache/uv`: `uv` download cache (wheels/sdists, git checkouts used by `uv`, etc.)
- `--mount=type=cache,target=/var/cache/apt,sharing=locked`: apt download cache (`sharing=locked` avoids apt/dpkg races with concurrent builds)
- `--mount=type=cache,target=/var/cache/{yum,dnf},sharing=locked`: yum/dnf metadata cache (`sharing=locked` avoids corruption with concurrent builds)
- `--mount=type=cache,target=/root/.cargo/{registry,git}`: Cargo crate/git download caches (Cargo has its own locking; no `sharing=locked` needed)

To inspect cache usage:
```bash
docker buildx du
docker info --format 'DockerRootDir: {{.DockerRootDir}}'
```

##### Inspecting BuildKit cache on the host (quick checklist)

1. Quick summary:
```bash
docker buildx du | tail -5
```

2. Find Docker root:
```bash
docker info | grep "Docker Root Dir"
# Output example: Docker Root Dir: /var/lib/docker
```

3. Check executor storage size:
```bash
DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}')"
sudo du -sh "${DOCKER_ROOT}/buildkit/executor" 2>/dev/null || true
```

4. Find specific caches (example: uv cache under BuildKit executor rootfs):
```bash
DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}')"
sudo sh -c 'find '"${DOCKER_ROOT}"'/buildkit/executor/*/rootfs/root/.cache/uv -type d 2>/dev/null | while read -r dir; do
  parent=$(dirname "$(dirname "$(dirname "$dir")")")
  du -sh "$parent/root/.cache/uv" 2>/dev/null
done'
```

5. List all large cache directories:
```bash
DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}')"
sudo sh -c 'du -sh '"${DOCKER_ROOT}"'/buildkit/executor/* 2>/dev/null | sort -h | tail -10'
```

Cleanup commands:
```bash
# Safe: clean only reclaimable cache
docker buildx prune

# Aggressive: clean everything
docker buildx prune --all

# Time-based: remove cache older than 3 days
docker buildx prune --filter until=72h
```

Current cache types (as mounted in various Dockerfiles):
1. `/root/.cache/uv` and `/home/dynamo/.cache/uv` - Python packages (uv; match the current `USER`)
2. `/root/.cargo/registry` - Rust crates
3. `/root/.cargo/git` - Rust git deps
4. `/var/cache/yum`, `/var/cache/dnf` - AlmaLinux packages
5. `/var/cache/apt` - Ubuntu packages

Note: `uv` commands set `UV_CACHE_DIR` per `RUN` so `uv` always uses the same path as the cache mount (instead of relying on `$HOME`).

> **💡 Tip**: The `dev` and `local-dev` images have source code baked in, but **using `--mount-workspace` with `run.sh` is recommended for development** to bind mount your local workspace for live editing.

**Common Usage Examples:**

```bash
# Build vLLM dev image called dynamo:latest-vllm (default). This runs as root and is for development.
python container/render.py --framework=vllm --target=dev --output-short-filename
docker build -t dynamo:latest-vllm-dev -f container/rendered.Dockerfile .

# Build a local-dev image. The local-dev image will run as `dynamo` with UID/GID matched to your host user,
# which is useful when mounting partitions for development.
python container/render.py --framework=vllm --target=local-dev --output-short-filename
docker build --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -f container/rendered.Dockerfile -t dynamo:latest-vllm-local-dev .

# Build TensorRT-LLM development image called dynamo:latest-trtllm
python container/render.py --framework=trtllm --target=runtime --output-short-filename --cuda-version=13.1
docker build -t dynamo:latest-trtllm-runtime -f rendered.Dockerfile .
```

### Building the Frontend Image

The frontend image is a specialized container that includes the Dynamo components (Dynamo, NIXL, etc) along with the Endpoint Picker (EPP) for Kubernetes Gateway API Inference Extension integration. This image is primarily used for inference gateway deployments.

**Build EPP Image**
```bash
sudo apt-get update && sudo apt-get install -y git build-essential protobuf-compiler libclang-dev
curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable
. "$HOME/.cargo/env"
cargo install cbindgen

pushd deploy/inference-gateway/epp
make all
popd

EPP_GIT_TAG=$(git describe --tags --dirty --always 2>/dev/null || echo "dev")
EPP_IMAGE="dynamo/dynamo-epp:${EPP_GIT_TAG}"
```

**Build Frontend Image**
```bash
# Build the frontend image (automatically builds EPP image as a dependency)
python container/render.py --framework=dynamo --target=frontend --output-short-filename
docker build -t dynamo:frontend --build-arg EPP_IMAGE=${EPP_IMAGE} -f rendered.Dockerfile .
```

The build process automatically:
1. Builds the Dynamo static library for EPP KV-aware routing
2. Builds the custom EPP Docker image using `make all` from `deploy/inference-gateway/epp/Makefile`
3. Builds the frontend image with the EPP binary and Dynamo runtime components

For more details, see [`deploy/inference-gateway/README.md`](../deploy/inference-gateway/README.md).

#### Frontend Image Contents

The frontend image includes:
- **EPP (Endpoint Picker)**: Handles request routing and load balancing for inference gateway
- **Dynamo Runtime**: Core platform components and routing logic
- **NIXL**: NVIDIA InfiniBand Library for high-performance network communication
- **Benchmarking Tools**: Performance testing utilities (aiperf, aiconfigurator, etc)
- **Python Environment**: Virtual environment with all required dependencies

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
container/run.sh --image dynamo:latest-vllm -v $HOME/.cache:/root/.cache

# Interactive development with workspace mounted using dev image (runs as root)
container/run.sh --image dynamo:latest-vllm --mount-workspace -it -v $HOME/.cache:/home/dynamo/.cache

# Interactive development with local-dev image (runs as dynamo user with matched host UID/GID)
container/run.sh --image dynamo:latest-vllm-local-dev --mount-workspace -it -v $HOME/.cache:/home/dynamo/.cache

# Use specific image and framework for development
container/run.sh --image v0.1.0.dev.08cc44965-vllm-local-dev --framework vllm --mount-workspace -it -v $HOME/.cache:/home/dynamo/.cache

# Interactive development shell with workspace mounted (local-dev)
container/run.sh --image dynamo:latest-vllm-local-dev --mount-workspace -v $HOME/.cache:/home/dynamo/.cache -it -- bash

# Development with custom environment variables
container/run.sh --image dynamo:latest-vllm-local-dev -e CUDA_VISIBLE_DEVICES=0,1 --mount-workspace -it -v $HOME/.cache:/home/dynamo/.cache

# Dry run to see docker command
container/run.sh --dry-run

# Development with custom volume mounts
container/run.sh --image dynamo:latest-vllm-local-dev -v /host/path:/container/path --mount-workspace -it -v $HOME/.cache:/home/dynamo/.cache

# Run runtime image as non-root dynamo user (for production)
container/run.sh --image dynamo:latest-vllm-runtime -v $HOME/.cache:/home/dynamo/.cache

# Run dev image as specific user (override default root)
container/run.sh --image dynamo:latest-vllm --user dynamo -v $HOME/.cache:/home/dynamo/.cache
```

### Network Configuration Options

The `run.sh` script supports different networking modes via the `--network` flag (defaults to `host`):

#### Host Networking (Default)
```bash
# Examples with dynamo user
container/run.sh --image dynamo:latest-vllm-local-dev --network host -v $HOME/.cache:/home/dynamo/.cache
container/run.sh --image dynamo:latest-vllm-local-dev -v $HOME/.cache:/home/dynamo/.cache
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
container/run.sh --image dynamo:latest-vllm --mount-workspace --network bridge -v $HOME/.cache:/home/dynamo/.cache
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
container/run.sh --image dynamo:latest-vllm --network none --mount-workspace -it -v $HOME/.cache:/home/dynamo/.cache

# Same with local-dev image (dynamo user with matched host UID/GID)
container/run.sh --image dynamo:latest-vllm-local-dev --network none --mount-workspace -it -v $HOME/.cache:/home/dynamo/.cache
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
# 1. Build local-dev image (builds runtime, then dev as intermediate, then local-dev as final image)
python container/render.py --framework=vllm --target=local-dev --output-short-filename
docker build --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -f container/rendered.Dockerfile -t dynamo:latest-vllm-local-dev .

# 2. Run development container using the local-dev image
# RECOMMENDED: --mount-workspace for live editing in dev and local-dev images
container/run.sh --image dynamo:latest-vllm-local-dev --mount-workspace -v $HOME/.cache:/home/dynamo/.cache -it

# 3. Inside container, run inference (requires both frontend and backend)
# Start frontend
python -m dynamo.frontend &

# Start backend (vLLM example)
python -m dynamo.vllm --model Qwen/Qwen3-0.6B --gpu-memory-utilization 0.20 &
```

### Production Workflow
```bash
# 1. Build production runtime image (runs as non-root dynamo user)
python container/render.py --framework=vllm --target=runtime --output-short-filename
docker build -t dynamo:latest-vllm-runtime -f rendered.Dockerfile .

# 2. Run production container as non-root dynamo user
container/run.sh --image dynamo:latest-vllm-runtime --gpus all -v $HOME/.cache:/home/dynamo/.cache
```

### Testing Workflow
```bash
# 1. Build dev image
python container/render.py --framework=vllm --target=dev --output-short-filename
docker build -t dynamo:latest-vllm-dev -f rendered.Dockerfile .

# 2. Run tests with network isolation for reproducible results (no -it needed for CI)
container/run.sh --image dynamo:latest-vllm --mount-workspace --network bridge -v $HOME/.cache:/home/dynamo/.cache -- python -m pytest tests/

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
