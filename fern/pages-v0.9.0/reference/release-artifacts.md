---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# Dynamo Release Artifacts

This document provides a comprehensive inventory of all Dynamo release artifacts including container images, Python wheels, Helm charts, and Rust crates.

> **See also:** [Support Matrix](support-matrix.md) for hardware and platform compatibility | [Feature Matrix](feature-matrix.md) for backend feature support

Release history in this document begins at v0.6.0.

## Current Release: Dynamo v0.9.0

- **GitHub Release:** [v0.9.0](https://github.com/ai-dynamo/dynamo/releases/tag/v0.9.0)
- **Docs:** [v0.9.0](https://docs.dynamo.nvidia.com/dynamo)
- **NGC Collection:** [ai-dynamo](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo)

### Container Images

| Image:Tag | Description | Backend | CUDA | Arch | NGC | Notes |
|-----------|-------------|---------|------|------|-----|-------|
| `vllm-runtime:0.9.0` | Runtime container for vLLM backend | vLLM `v0.14.1` | `v12.9` | AMD64/ARM64 | [link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/vllm-runtime?version=0.9.0) | |
| `vllm-runtime:0.9.0-cuda13` | Runtime container for vLLM backend (CUDA 13) | vLLM `v0.14.1` | `v13.0` | AMD64/ARM64* | [link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/vllm-runtime?version=0.9.0-cuda13) | Experimental |
| `sglang-runtime:0.9.0` | Runtime container for SGLang backend | SGLang `v0.5.8` | `v12.9` | AMD64/ARM64 | [link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/sglang-runtime?version=0.9.0) | |
| `sglang-runtime:0.9.0-cuda13` | Runtime container for SGLang backend (CUDA 13) | SGLang `v0.5.8` | `v13.0` | AMD64/ARM64* | [link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/sglang-runtime?version=0.9.0-cuda13) | Experimental |
| `tensorrtllm-runtime:0.9.0` | Runtime container for TensorRT-LLM backend | TRT-LLM `v1.3.0rc1` | `v13.0` | AMD64/ARM64 | [link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/tensorrtllm-runtime?version=0.9.0) | |
| `dynamo-frontend:0.9.0` | API gateway with Endpoint Prediction Protocol (EPP) | — | — | AMD64/ARM64 | [link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/dynamo-frontend?version=0.9.0) | |
| `kubernetes-operator:0.9.0` | Kubernetes operator for Dynamo deployments | — | — | AMD64/ARM64 | [link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/kubernetes-operator?version=0.9.0) | |

\* Multimodal inference on CUDA 13 images: works on AMD64 for all backends; works on ARM64 only for TensorRT-LLM (`vllm-runtime:*-cuda13` and `sglang-runtime:*-cuda13` do not support multimodality on ARM64).

### Python Wheels

We recommend using the TensorRT-LLM NGC container instead of the `ai-dynamo[trtllm]` wheel. See the [NGC container collection](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo) for supported images.

| Package | Description | Python | Platform | PyPI |
|---------|-------------|--------|----------|------|
| `ai-dynamo==0.9.0` | Main package with backend integrations (vLLM, SGLang, TRT-LLM) | `3.10`–`3.12` | Linux (glibc `v2.28+`) | [link](https://pypi.org/project/ai-dynamo/0.9.0/) |
| `ai-dynamo-runtime==0.9.0` | Core Python bindings for Dynamo runtime | `3.10`–`3.12` | Linux (glibc `v2.28+`) | [link](https://pypi.org/project/ai-dynamo-runtime/0.9.0/) |
| `kvbm==0.9.0` | KV Block Manager for disaggregated KV cache | `3.12` | Linux (glibc `v2.28+`) | [link](https://pypi.org/project/kvbm/0.9.0/) |

### Helm Charts

| Chart | Description | NGC |
|-------|-------------|-----|
| `dynamo-crds-0.9.0` | Custom Resource Definitions for Dynamo Kubernetes resources | [link](https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-crds-0.9.0.tgz) |
| `dynamo-platform-0.9.0` | Platform services (etcd, NATS) for Dynamo cluster | [link](https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-0.9.0.tgz) |

> **Note:** The `dynamo-graph` Helm chart is deprecated as of v0.9.0. Use the Kubernetes operator for deployment graph management.

### Rust Crates

| Crate | Description | MSRV (Rust) | crates.io |
|-------|-------------|-------------|-----------|
| `dynamo-runtime@0.9.0` | Core distributed runtime library | `v1.82` | [link](https://crates.io/crates/dynamo-runtime/0.9.0) |
| `dynamo-llm@0.9.0` | LLM inference engine | `v1.82` | [link](https://crates.io/crates/dynamo-llm/0.9.0) |
| `dynamo-async-openai@0.9.0` | Async OpenAI-compatible API client | `v1.82` | [link](https://crates.io/crates/dynamo-async-openai/0.9.0) |
| `dynamo-parsers@0.9.0` | Protocol parsers (SSE, JSON streaming) | `v1.82` | [link](https://crates.io/crates/dynamo-parsers/0.9.0) |
| `dynamo-memory@0.9.0` | Memory management utilities | `v1.82` | [link](https://crates.io/crates/dynamo-memory/0.9.0) |
| `dynamo-config@0.9.0` | Configuration management | `v1.82` | [link](https://crates.io/crates/dynamo-config/0.9.0) |
| `dynamo-tokens@0.9.0` | Tokenizer bindings for LLM inference | `v1.82` | [link](https://crates.io/crates/dynamo-tokens/0.9.0) |

## Quick Install Commands

### Container Images (NGC)

> For detailed run instructions, see the [Container README](https://github.com/ai-dynamo/dynamo/tree/main/container/README.md) or backend-specific guides: [vLLM](../backends/vllm/README.md) | [SGLang](../backends/sglang/README.md) | [TensorRT-LLM](../backends/trtllm/README.md)

```bash
# Runtime containers
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.9.0
docker pull nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.9.0
docker pull nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.9.0

# CUDA 13 variants (experimental)
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.9.0-cuda13
docker pull nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.9.0-cuda13

# Infrastructure containers
docker pull nvcr.io/nvidia/ai-dynamo/dynamo-frontend:0.9.0
docker pull nvcr.io/nvidia/ai-dynamo/kubernetes-operator:0.9.0
```

### Python Wheels (PyPI)

> For detailed installation instructions, see the [Local Quick Start](https://github.com/ai-dynamo/dynamo#local-quick-start) in the README.

```bash
# Install Dynamo with a specific backend (Recommended)
uv pip install "ai-dynamo[vllm]==0.9.0"
uv pip install "ai-dynamo[sglang]==0.9.0"
# TensorRT-LLM requires the NVIDIA PyPI index and pip
pip install --pre --extra-index-url https://pypi.nvidia.com "ai-dynamo[trtllm]==0.9.0"

# Install Dynamo core only
uv pip install ai-dynamo==0.9.0

# Install standalone KVBM (Python 3.12 only)
uv pip install kvbm==0.9.0
```

### Helm Charts (NGC)

> For Kubernetes deployment instructions, see the [Kubernetes Installation Guide](../kubernetes/installation-guide.md).

```bash
helm install dynamo-crds oci://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-crds --version 0.9.0
helm install dynamo-platform oci://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform --version 0.9.0
```

### Rust Crates (crates.io)

> For API documentation, see each crate on [docs.rs](https://docs.rs/). To build Dynamo from source, see [Building from Source](https://github.com/ai-dynamo/dynamo#building-from-source).

```bash
cargo add dynamo-runtime@0.9.0
cargo add dynamo-llm@0.9.0
cargo add dynamo-async-openai@0.9.0
cargo add dynamo-parsers@0.9.0
cargo add dynamo-memory@0.9.0
cargo add dynamo-config@0.9.0
cargo add dynamo-tokens@0.9.0
```

## CUDA and Driver Requirements

For detailed CUDA toolkit versions and minimum driver requirements for each container image, see the [Support Matrix](support-matrix.md#cuda-and-driver-requirements).

## Known Issues

For a complete list of known issues, refer to the release notes for each version:
- [v0.9.0 Release Notes](https://github.com/ai-dynamo/dynamo/releases/tag/v0.9.0)
- [v0.8.1 Release Notes](https://github.com/ai-dynamo/dynamo/releases/tag/v0.8.1)

### Known Artifact Issues

| Version | Artifact | Issue | Status |
|---------|----------|-------|--------|
| v0.8.1 | `vllm-runtime:0.8.1-cuda13` | Container fails to launch. | Known issue |
| v0.8.1 | `sglang-runtime:0.8.1-cuda13`, `vllm-runtime:0.8.1-cuda13` | Multimodality not expected to work on ARM64. Works on AMD64. | Known limitation |

---

## Release History

- **v0.9.0**: Updated vLLM to `v0.14.1`, SGLang to `v0.5.8`, TRT-LLM to `v1.3.0rc1`, NIXL to `v0.9.0`. New `dynamo-tokens` Rust crate. Deprecated `dynamo-graph` Helm chart.
- **v0.8.1.post1/.post2/.post3 Patches**: Experimental patch releases updating TRT-LLM only (PyPI wheels and TRT-LLM container). No other artifacts changed.

### GitHub Releases

| Version | Release Date | GitHub | Docs |
|---------|--------------|--------|------|
| `v0.9.0` | Feb 11, 2026 | [Release](https://github.com/ai-dynamo/dynamo/releases/tag/v0.9.0) | [Docs](https://docs.dynamo.nvidia.com/dynamo) |
| `v0.8.1` | Jan 23, 2026 | [Release](https://github.com/ai-dynamo/dynamo/releases/tag/v0.8.1) | [Docs](https://docs.nvidia.com/dynamo/archive/0.8.1/index.html) |
| `v0.8.0` | Jan 15, 2026 | [Release](https://github.com/ai-dynamo/dynamo/releases/tag/v0.8.0) | [Docs](https://docs.nvidia.com/dynamo/archive/0.8.0/index.html) |
| `v0.7.1` | Dec 15, 2025 | [Release](https://github.com/ai-dynamo/dynamo/releases/tag/v0.7.1) | [Docs](https://docs.nvidia.com/dynamo/archive/0.7.1/index.html) |
| `v0.7.0` | Nov 26, 2025 | [Release](https://github.com/ai-dynamo/dynamo/releases/tag/v0.7.0) | [Docs](https://docs.nvidia.com/dynamo/archive/0.7.0/index.html) |
| `v0.6.1` | Nov 6, 2025 | [Release](https://github.com/ai-dynamo/dynamo/releases/tag/v0.6.1) | [Docs](https://docs.nvidia.com/dynamo/archive/0.6.1/index.html) |
| `v0.6.0` | Oct 28, 2025 | [Release](https://github.com/ai-dynamo/dynamo/releases/tag/v0.6.0) | [Docs](https://docs.nvidia.com/dynamo/archive/0.6.0/index.html) |

### Container Images

> **NGC Collection:** [ai-dynamo](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo)
>
> To access a specific version, append `?version=TAG` to the container URL:
> `https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/{container}?version={tag}`

#### vllm-runtime

| Image:Tag | vLLM | Arch | CUDA | Notes |
|-----------|------|------|------|-------|
| `vllm-runtime:0.9.0` | `v0.14.1` | AMD64/ARM64 | `v12.9` | |
| `vllm-runtime:0.9.0-cuda13` | `v0.14.1` | AMD64/ARM64* | `v13.0` | Experimental |
| `vllm-runtime:0.8.1` | `v0.12.0` | AMD64/ARM64 | `v12.9` | |
| `vllm-runtime:0.8.0` | `v0.12.0` | AMD64/ARM64 | `v12.9` | |
| `vllm-runtime:0.8.0-cuda13` | `v0.12.0` | AMD64/ARM64 | `v13.0` | Experimental |
| `vllm-runtime:0.7.0.post2` | `v0.11.2` | AMD64/ARM64 | `v12.8` | Patch |
| `vllm-runtime:0.7.1` | `v0.11.0` | AMD64/ARM64 | `v12.8` | |
| `vllm-runtime:0.7.0.post1` | `v0.11.0` | AMD64/ARM64 | `v12.8` | Patch |
| `vllm-runtime:0.7.0` | `v0.11.0` | AMD64/ARM64 | `v12.8` | |
| `vllm-runtime:0.6.1.post1` | `v0.11.0` | AMD64/ARM64 | `v12.8` | Patch |
| `vllm-runtime:0.6.1` | `v0.11.0` | AMD64/ARM64 | `v12.8` | |
| `vllm-runtime:0.6.0` | `v0.11.0` | AMD64 | `v12.8` | |

#### sglang-runtime

| Image:Tag | SGLang | Arch | CUDA | Notes |
|-----------|--------|------|------|-------|
| `sglang-runtime:0.9.0` | `v0.5.8` | AMD64/ARM64 | `v12.9` | |
| `sglang-runtime:0.9.0-cuda13` | `v0.5.8` | AMD64/ARM64* | `v13.0` | Experimental |
| `sglang-runtime:0.8.1` | `v0.5.6.post2` | AMD64/ARM64 | `v12.9` | |
| `sglang-runtime:0.8.1-cuda13` | `v0.5.6.post2` | AMD64/ARM64 | `v13.0` | Experimental |
| `sglang-runtime:0.8.0` | `v0.5.6.post2` | AMD64/ARM64 | `v12.9` | |
| `sglang-runtime:0.8.0-cuda13` | `v0.5.6.post2` | AMD64/ARM64 | `v13.0` | Experimental |
| `sglang-runtime:0.7.1` | `v0.5.4.post3` | AMD64/ARM64 | `v12.9` | |
| `sglang-runtime:0.7.0.post1` | `v0.5.4.post3` | AMD64/ARM64 | `v12.9` | Patch |
| `sglang-runtime:0.7.0` | `v0.5.4.post3` | AMD64/ARM64 | `v12.9` | |
| `sglang-runtime:0.6.1.post1` | `v0.5.3.post2` | AMD64/ARM64 | `v12.9` | Patch |
| `sglang-runtime:0.6.1` | `v0.5.3.post2` | AMD64/ARM64 | `v12.9` | |
| `sglang-runtime:0.6.0` | `v0.5.3.post2` | AMD64 | `v12.8` | |

#### tensorrtllm-runtime

| Image:Tag | TRT-LLM | Arch | CUDA | Notes |
|-----------|---------|------|------|-------|
| `tensorrtllm-runtime:0.9.0` | `v1.3.0rc1` | AMD64/ARM64 | `v13.0` | |
| `tensorrtllm-runtime:0.8.1.post3` | `v1.2.0rc6.post3` | AMD64/ARM64 | `v13.0` | Patch |
| `tensorrtllm-runtime:0.8.1.post1` | `v1.2.0rc6.post2` | AMD64/ARM64 | `v13.0` | Patch |
| `tensorrtllm-runtime:0.8.1` | `v1.2.0rc6.post1` | AMD64/ARM64 | `v13.0` | |
| `tensorrtllm-runtime:0.8.0` | `v1.2.0rc6.post1` | AMD64/ARM64 | `v13.0` | |
| `tensorrtllm-runtime:0.7.0.post2` | `v1.2.0rc2` | AMD64/ARM64 | `v13.0` | Patch |
| `tensorrtllm-runtime:0.7.1` | `v1.2.0rc3` | AMD64/ARM64 | `v13.0` | |
| `tensorrtllm-runtime:0.7.0.post1` | `v1.2.0rc3` | AMD64/ARM64 | `v13.0` | Patch |
| `tensorrtllm-runtime:0.7.0` | `v1.2.0rc2` | AMD64/ARM64 | `v13.0` | |
| `tensorrtllm-runtime:0.6.1-cuda13` | `v1.2.0rc1` | AMD64/ARM64 | `v13.0` | Experimental |
| `tensorrtllm-runtime:0.6.1.post1` | `v1.1.0rc5` | AMD64/ARM64 | `v12.9` | Patch |
| `tensorrtllm-runtime:0.6.1` | `v1.1.0rc5` | AMD64/ARM64 | `v12.9` | |
| `tensorrtllm-runtime:0.6.0` | `v1.1.0rc5` | AMD64/ARM64 | `v12.9` | |

#### dynamo-frontend

| Image:Tag | Arch | Notes |
|-----------|------|-------|
| `dynamo-frontend:0.9.0` | AMD64/ARM64 | |
| `dynamo-frontend:0.8.1` | AMD64/ARM64 | |
| `dynamo-frontend:0.8.0` | AMD64/ARM64 | Initial |

#### kubernetes-operator

| Image:Tag | Arch | Notes |
|-----------|------|-------|
| `kubernetes-operator:0.9.0` | AMD64/ARM64 | |
| `kubernetes-operator:0.8.1` | AMD64/ARM64 | |
| `kubernetes-operator:0.8.0` | AMD64/ARM64 | |
| `kubernetes-operator:0.7.1` | AMD64/ARM64 | |
| `kubernetes-operator:0.7.0.post1` | AMD64/ARM64 | Patch |
| `kubernetes-operator:0.7.0` | AMD64/ARM64 | |
| `kubernetes-operator:0.6.1` | AMD64/ARM64 | |
| `kubernetes-operator:0.6.0` | AMD64/ARM64 | |

### Python Wheels

> **PyPI:** [ai-dynamo](https://pypi.org/project/ai-dynamo/) | [ai-dynamo-runtime](https://pypi.org/project/ai-dynamo-runtime/) | [kvbm](https://pypi.org/project/kvbm/)
>
> To access a specific version: `https://pypi.org/project/{package}/{version}/`

#### ai-dynamo (wheel)

| Package | Python | Platform | Notes |
|---------|--------|----------|-------|
| `ai-dynamo==0.9.0` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `ai-dynamo==0.8.1.post3` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | TRT-LLM `v1.2.0rc6.post3` |
| `ai-dynamo==0.8.1.post1` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | TRT-LLM `v1.2.0rc6.post2` |
| `ai-dynamo==0.8.1` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `ai-dynamo==0.8.0` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `ai-dynamo==0.7.1` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `ai-dynamo==0.7.0` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `ai-dynamo==0.6.1` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `ai-dynamo==0.6.0` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |

#### ai-dynamo-runtime (wheel)

| Package | Python | Platform | Notes |
|---------|--------|----------|-------|
| `ai-dynamo-runtime==0.9.0` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `ai-dynamo-runtime==0.8.1.post3` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | TRT-LLM `v1.2.0rc6.post3` |
| `ai-dynamo-runtime==0.8.1.post1` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | TRT-LLM `v1.2.0rc6.post2` |
| `ai-dynamo-runtime==0.8.1` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `ai-dynamo-runtime==0.8.0` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `ai-dynamo-runtime==0.7.1` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `ai-dynamo-runtime==0.7.0` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `ai-dynamo-runtime==0.6.1` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `ai-dynamo-runtime==0.6.0` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |

#### kvbm (wheel)

| Package | Python | Platform | Notes |
|---------|--------|----------|-------|
| `kvbm==0.9.0` | `3.12` | Linux (glibc `v2.28+`) | |
| `kvbm==0.8.1` | `3.12` | Linux (glibc `v2.28+`) | |
| `kvbm==0.8.0` | `3.12` | Linux (glibc `v2.28+`) | |
| `kvbm==0.7.1` | `3.12` | Linux (glibc `v2.28+`) | |
| `kvbm==0.7.0` | `3.12` | Linux (glibc `v2.28+`) | Initial |

### Helm Charts

> **NGC Helm Registry:** [ai-dynamo](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo)
>
> Direct download: `https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/{chart}-{version}.tgz`

#### dynamo-crds (Helm chart)

| Chart | Notes |
|-------|-------|
| `dynamo-crds-0.9.0` | |
| `dynamo-crds-0.8.1` | |
| `dynamo-crds-0.8.0` | |
| `dynamo-crds-0.7.1` | |
| `dynamo-crds-0.7.0` | |
| `dynamo-crds-0.6.1` | |
| `dynamo-crds-0.6.0` | |

#### dynamo-platform (Helm chart)

| Chart | Notes |
|-------|-------|
| `dynamo-platform-0.9.0` | |
| `dynamo-platform-0.8.1` | |
| `dynamo-platform-0.8.0` | |
| `dynamo-platform-0.7.1` | |
| `dynamo-platform-0.7.0` | |
| `dynamo-platform-0.6.1` | |
| `dynamo-platform-0.6.0` | |

#### dynamo-graph (Helm chart) -- Deprecated

> **Note:** The `dynamo-graph` Helm chart is deprecated as of v0.9.0.

| Chart | Notes |
|-------|-------|
| `dynamo-graph-0.8.1` | Last release |
| `dynamo-graph-0.8.0` | |
| `dynamo-graph-0.7.1` | |
| `dynamo-graph-0.7.0` | |
| `dynamo-graph-0.6.1` | |
| `dynamo-graph-0.6.0` | |

### Rust Crates

> **crates.io:** [dynamo-runtime](https://crates.io/crates/dynamo-runtime) | [dynamo-llm](https://crates.io/crates/dynamo-llm) | [dynamo-async-openai](https://crates.io/crates/dynamo-async-openai) | [dynamo-parsers](https://crates.io/crates/dynamo-parsers) | [dynamo-memory](https://crates.io/crates/dynamo-memory) | [dynamo-config](https://crates.io/crates/dynamo-config) | [dynamo-tokens](https://crates.io/crates/dynamo-tokens)
>
> To access a specific version: `https://crates.io/crates/{crate}/{version}`

#### dynamo-runtime (crate)

| Crate | MSRV (Rust) | Notes |
|-------|-------------|-------|
| `dynamo-runtime@0.9.0` | `v1.82` | |
| `dynamo-runtime@0.8.1` | `v1.82` | |
| `dynamo-runtime@0.8.0` | `v1.82` | |
| `dynamo-runtime@0.7.1` | `v1.82` | |
| `dynamo-runtime@0.7.0` | `v1.82` | |
| `dynamo-runtime@0.6.1` | `v1.82` | |
| `dynamo-runtime@0.6.0` | `v1.82` | |

#### dynamo-llm (crate)

| Crate | MSRV (Rust) | Notes |
|-------|-------------|-------|
| `dynamo-llm@0.9.0` | `v1.82` | |
| `dynamo-llm@0.8.1` | `v1.82` | |
| `dynamo-llm@0.8.0` | `v1.82` | |
| `dynamo-llm@0.7.1` | `v1.82` | |
| `dynamo-llm@0.7.0` | `v1.82` | |
| `dynamo-llm@0.6.1` | `v1.82` | |
| `dynamo-llm@0.6.0` | `v1.82` | |

#### dynamo-async-openai (crate)

| Crate | MSRV (Rust) | Notes |
|-------|-------------|-------|
| `dynamo-async-openai@0.9.0` | `v1.82` | |
| `dynamo-async-openai@0.8.1` | `v1.82` | |
| `dynamo-async-openai@0.8.0` | `v1.82` | |
| `dynamo-async-openai@0.7.1` | `v1.82` | |
| `dynamo-async-openai@0.7.0` | `v1.82` | |
| `dynamo-async-openai@0.6.1` | `v1.82` | |
| `dynamo-async-openai@0.6.0` | `v1.82` | |

#### dynamo-parsers (crate)

| Crate | MSRV (Rust) | Notes |
|-------|-------------|-------|
| `dynamo-parsers@0.9.0` | `v1.82` | |
| `dynamo-parsers@0.8.1` | `v1.82` | |
| `dynamo-parsers@0.8.0` | `v1.82` | |
| `dynamo-parsers@0.7.1` | `v1.82` | |
| `dynamo-parsers@0.7.0` | `v1.82` | |
| `dynamo-parsers@0.6.1` | `v1.82` | |
| `dynamo-parsers@0.6.0` | `v1.82` | |

#### dynamo-memory (crate)

| Crate | MSRV (Rust) | Notes |
|-------|-------------|-------|
| `dynamo-memory@0.9.0` | `v1.82` | |
| `dynamo-memory@0.8.1` | `v1.82` | |
| `dynamo-memory@0.8.0` | `v1.82` | Initial |

#### dynamo-config (crate)

| Crate | MSRV (Rust) | Notes |
|-------|-------------|-------|
| `dynamo-config@0.9.0` | `v1.82` | |
| `dynamo-config@0.8.1` | `v1.82` | |
| `dynamo-config@0.8.0` | `v1.82` | Initial |

#### dynamo-tokens (crate)

| Crate | MSRV (Rust) | Notes |
|-------|-------------|-------|
| `dynamo-tokens@0.9.0` | `v1.82` | Initial |
