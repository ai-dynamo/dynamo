---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Release Artifacts
---

This document provides a comprehensive inventory of all Dynamo release artifacts including container images, Python wheels, Helm charts, and Rust crates.

> **See also:** [Support Matrix](support-matrix.md) for hardware and platform compatibility | [Feature Matrix](feature-matrix.md) for backend feature support

Release history in this document begins at v0.6.0.

## Current Release: Dynamo v1.0.1

- **GitHub Release:** [v1.0.1](https://github.com/ai-dynamo/dynamo/releases/tag/v1.0.1)
- **Docs:** [v1.0.1](https://docs.dynamo.nvidia.com/dynamo)
- **NGC Collection:** [ai-dynamo](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo)

> **Experimental:** [v1.1.0-dev.1](#pre-release-artifacts) is available as an experimental preview. See [Pre-Release Artifacts](#pre-release-artifacts) for container images, wheels, and Helm charts.

### Container Images

| Image:Tag | Description | Backend | CUDA | Arch | NGC | Notes |
|-----------|-------------|---------|------|------|-----|-------|
| `vllm-runtime:1.0.1` | Runtime container for vLLM backend | vLLM `v0.16.0` | `v12.9` | AMD64/ARM64 | [link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/vllm-runtime?version=1.0.1) | |
| `vllm-runtime:1.0.1-cuda13` | Runtime container for vLLM backend (CUDA 13) | vLLM `v0.16.0` | `v13.0` | AMD64/ARM64* | [link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/vllm-runtime?version=1.0.1-cuda13) | |
| `vllm-runtime:1.0.1-efa-amd64` | Runtime container for vLLM with AWS EFA | vLLM `v0.16.0` | `v12.9` | AMD64 | [link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/vllm-runtime?version=1.0.1-efa-amd64) | Experimental |
| `sglang-runtime:1.0.1` | Runtime container for SGLang backend | SGLang `v0.5.9` | `v12.9` | AMD64/ARM64 | [link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/sglang-runtime?version=1.0.1) | |
| `sglang-runtime:1.0.1-cuda13` | Runtime container for SGLang backend (CUDA 13) | SGLang `v0.5.9` | `v13.0` | AMD64/ARM64* | [link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/sglang-runtime?version=1.0.1-cuda13) | |
| `tensorrtllm-runtime:1.0.1` | Runtime container for TensorRT-LLM backend | TRT-LLM `v1.3.0rc5.post1` | `v13.1` | AMD64/ARM64 | [link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/tensorrtllm-runtime?version=1.0.1) | |
| `tensorrtllm-runtime:1.0.1-efa-amd64` | Runtime container for TensorRT-LLM with AWS EFA | TRT-LLM `v1.3.0rc5.post1` | `v13.1` | AMD64 | [link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/tensorrtllm-runtime?version=1.0.1-efa-amd64) | Experimental |
| `dynamo-frontend:1.0.1` | API gateway with Endpoint Prediction Protocol (EPP) | — | — | AMD64/ARM64 | [link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/dynamo-frontend?version=1.0.1) | |
| `kubernetes-operator:1.0.1` | Kubernetes operator for Dynamo deployments | — | — | AMD64/ARM64 | [link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/kubernetes-operator?version=1.0.1) | |
| `snapshot-agent:1.0.1` | Snapshot agent for fast GPU worker recovery via CRIU | — | — | AMD64/ARM64 | [link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/snapshot-agent?version=1.0.1) | Preview |

\* Multimodal inference on CUDA 13 images: works on AMD64 for all backends; works on ARM64 only for TensorRT-LLM (`vllm-runtime:*-cuda13` and `sglang-runtime:*-cuda13` do not support multimodality on ARM64).

### Python Wheels

We recommend using the TensorRT-LLM NGC container instead of the `ai-dynamo[trtllm]` wheel. See the [NGC container collection](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo) for supported images.

| Package | Description | Python | Platform | PyPI |
|---------|-------------|--------|----------|------|
| `ai-dynamo==1.0.1` | Main package with backend integrations (vLLM, SGLang, TRT-LLM) | `3.10`–`3.12` | Linux (glibc `v2.28+`) | [link](https://pypi.org/project/ai-dynamo/1.0.1/) |
| `ai-dynamo-runtime==1.0.1` | Core Python bindings for Dynamo runtime | `3.10`–`3.12` | Linux (glibc `v2.28+`) | [link](https://pypi.org/project/ai-dynamo-runtime/1.0.1/) |
| `kvbm==1.0.1` | KV Block Manager for disaggregated KV cache | `3.12` | Linux (glibc `v2.28+`) | [link](https://pypi.org/project/kvbm/1.0.1/) |

### Helm Charts

| Chart | Description | NGC |
|-------|-------------|-----|
| `dynamo-platform-1.0.1` | Platform services (etcd, NATS) and Dynamo Operator for Dynamo cluster | [link](https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-1.0.1.tgz) |
| `snapshot-1.0.1` | Snapshot DaemonSet for fast GPU worker recovery | [link](https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/snapshot-1.0.1.tgz) |

> [!NOTE]
> The `dynamo-crds` Helm chart is deprecated as of v1.0.0; CRDs are now managed by the Dynamo Operator. The `dynamo-graph` Helm chart is deprecated as of v0.9.0.

### Rust Crates

| Crate | Description | MSRV (Rust) | crates.io |
|-------|-------------|-------------|-----------|
| `dynamo-runtime@1.0.1` | Core distributed runtime library | `v1.82` | [link](https://crates.io/crates/dynamo-runtime/1.0.1) |
| `dynamo-llm@1.0.1` | LLM inference engine | `v1.82` | [link](https://crates.io/crates/dynamo-llm/1.0.1) |
| `dynamo-protocols@1.0.1` | Async OpenAI-compatible API client | `v1.82` | [link](https://crates.io/crates/dynamo-protocols/1.0.1) |
| `dynamo-parsers@1.0.1` | Protocol parsers (SSE, JSON streaming) | `v1.82` | [link](https://crates.io/crates/dynamo-parsers/1.0.1) |
| `dynamo-memory@1.0.1` | Memory management utilities | `v1.82` | [link](https://crates.io/crates/dynamo-memory/1.0.1) |
| `dynamo-config@1.0.1` | Configuration management | `v1.82` | [link](https://crates.io/crates/dynamo-config/1.0.1) |
| `dynamo-tokens@1.0.1` | Tokenizer bindings for LLM inference | `v1.82` | [link](https://crates.io/crates/dynamo-tokens/1.0.1) |
| `dynamo-mocker@1.0.1` | Inference engine simulator for benchmarking | `v1.82` | [link](https://crates.io/crates/dynamo-mocker/1.0.1) |
| `dynamo-kv-router@1.0.1` | KV-aware request routing library | `v1.82` | [link](https://crates.io/crates/dynamo-kv-router/1.0.1) |

## Quick Install Commands

### Container Images (NGC)

> [!TIP]
> For detailed run instructions, see the backend-specific guides: [vLLM](../backends/vllm/README.md) | [SGLang](../backends/sglang/README.md) | [TensorRT-LLM](../backends/trtllm/README.md)

```bash
# Runtime containers
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1
docker pull nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.0.1
docker pull nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.0.1

# CUDA 13 variants
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1-cuda13
docker pull nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.0.1-cuda13

# EFA variants (AWS, AMD64 only, experimental)
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1-efa-amd64
docker pull nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.0.1-efa-amd64

# Infrastructure containers
docker pull nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.0.1
docker pull nvcr.io/nvidia/ai-dynamo/kubernetes-operator:1.0.1
docker pull nvcr.io/nvidia/ai-dynamo/snapshot-agent:1.0.1
```

### Python Wheels (PyPI)

> [!TIP]
> For detailed installation instructions, see the [Local Quick Start](https://github.com/ai-dynamo/dynamo#local-quick-start) in the README.

```bash
# Install Dynamo with a specific backend (Recommended)
uv pip install "ai-dynamo[vllm]==1.0.1"
uv pip install "ai-dynamo[sglang]==1.0.1"
# TensorRT-LLM requires the NVIDIA PyPI index and pip
pip install --pre --extra-index-url https://pypi.nvidia.com "ai-dynamo[trtllm]==1.0.1"

# Install Dynamo core only
uv pip install ai-dynamo==1.0.1

# Install standalone KVBM (Python 3.12 only)
uv pip install kvbm==1.0.1
```

### Helm Charts (NGC)

> [!TIP]
> For Kubernetes deployment instructions, see the [Kubernetes Installation Guide](../kubernetes/installation-guide.md).

```bash
helm install dynamo-platform oci://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform --version 1.0.1
helm install snapshot oci://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/snapshot --version 1.0.1
```

### Rust Crates (crates.io)

> [!TIP]
> For API documentation, see each crate on [docs.rs](https://docs.rs/). To build Dynamo from source, see [Building from Source](https://github.com/ai-dynamo/dynamo#building-from-source).

```bash
cargo add dynamo-runtime@1.0.1
cargo add dynamo-llm@1.0.1
cargo add dynamo-protocols@1.0.1
cargo add dynamo-parsers@1.0.1
cargo add dynamo-memory@1.0.1
cargo add dynamo-config@1.0.1
cargo add dynamo-tokens@1.0.1
cargo add dynamo-mocker@1.0.1
cargo add dynamo-kv-router@1.0.1
```

**CUDA and Driver Requirements:** For detailed CUDA toolkit versions and minimum driver requirements for each container image, see the [Support Matrix](support-matrix.md#cuda-and-driver-requirements).

## Known Issues

For a complete list of known issues, refer to the release notes for each version:
- [v1.0.1 Release Notes](https://github.com/ai-dynamo/dynamo/releases/tag/v1.0.1)
- [v1.0.0 Release Notes](https://github.com/ai-dynamo/dynamo/releases/tag/v1.0.0)
- [v0.9.0 Release Notes](https://github.com/ai-dynamo/dynamo/releases/tag/v0.9.0)
- [v0.8.1 Release Notes](https://github.com/ai-dynamo/dynamo/releases/tag/v0.8.1)

### Known Artifact Issues

| Version | Artifact | Issue | Status |
|---------|----------|-------|--------|
| v0.9.0 | `dynamo-platform-0.9.0` | Helm chart sets operator image to `0.7.1` instead of `0.9.0`. | Fixed in v0.9.0.post1 |
| v0.8.1 | `vllm-runtime:0.8.1-cuda13` | Container fails to launch. | Known issue |
| v0.8.1 | `sglang-runtime:0.8.1-cuda13`, `vllm-runtime:0.8.1-cuda13` | Multimodality not expected to work on ARM64. Works on AMD64. | Known limitation |
| v0.8.0 | `sglang-runtime:0.8.0-cuda13` | CuDNN installation issue caused PyTorch `v2.9.1` compatibility problems with `nn.Conv3d`, resulting in performance degradation and excessive memory usage in multimodal workloads. | Fixed in v0.8.1 ([#5461](https://github.com/ai-dynamo/dynamo/pull/5461)) |

---

## Release History

- **v1.1.0-dev.1** *(experimental)*: Preview release. SGLang `v0.5.9`, TRT-LLM `v1.3.0rc5.post1`, vLLM `v0.17.1`, NIXL `v0.10.1`. Not recommended for production use.
- **v1.0.1**: Patch release. Same backend versions as v1.0.0: SGLang `v0.5.9`, TRT-LLM `v1.3.0rc5.post1`, vLLM `v0.16.0`, NIXL `v0.10.1`.
- **v1.0.0**: First major release. SGLang `v0.5.9`, TRT-LLM `v1.3.0rc5.post1` (CUDA 13.1), vLLM `v0.16.0`, NIXL `v0.10.1`. New `snapshot-agent` container and `snapshot` Helm chart (Preview). New EFA container variants for vLLM and TRT-LLM (Experimental, AMD64 only). New `dynamo-mocker` and `dynamo-kv-router` Rust crates. Deprecated `dynamo-crds` Helm chart (CRDs now managed by the Operator). `v1alpha1` CRDs deprecated.
- **v0.9.1**: Updated TRT-LLM to `v1.3.0rc3`. All other backend versions unchanged from v0.9.0.
- **v0.9.0.post1**: Fixed `dynamo-platform` Helm chart operator image tag (Helm chart only, NGC)
- **v0.9.0**: Updated vLLM to `v0.14.1`, SGLang to `v0.5.8`, TRT-LLM to `v1.3.0rc1`, NIXL to `v0.9.0`. New `dynamo-tokens` Rust crate. Deprecated `dynamo-graph` Helm chart.
- **v0.8.1.post1/.post2/.post3 Patches**: Experimental patch releases updating TRT-LLM only (PyPI wheels and TRT-LLM container). No other artifacts changed.
- **Standalone Frontend Container**: `dynamo-frontend` added in v0.8.0
- **EFA Runtimes**: Experimental AWS EFA variants for vLLM and TRT-LLM (AMD64 only) in v1.0.0
- **CUDA 13 Runtimes**: Experimental CUDA 13 runtime for SGLang and vLLM in v0.8.0
- **New Rust Crates**: `dynamo-memory` and `dynamo-config` added in v0.8.0

### GitHub Releases

| Version | Release Date | GitHub | Docs | Notes |
|---------|--------------|--------|------|-------|
| `v1.1.0-dev.1` | Mar 17, 2026 | [Branch](https://github.com/ai-dynamo/dynamo/tree/release/1.1.0-dev.1) | — | Experimental |
| `v1.0.1` | Mar 16, 2026 | [Release](https://github.com/ai-dynamo/dynamo/releases/tag/v1.0.1) | [Docs](https://docs.dynamo.nvidia.com/dynamo) | |
| `v1.0.0` | Mar 12, 2026 | [Release](https://github.com/ai-dynamo/dynamo/releases/tag/v1.0.0) | [Docs](https://docs.dynamo.nvidia.com/dynamo) | |
| `v0.9.1` | Mar 4, 2026 | [Release](https://github.com/ai-dynamo/dynamo/releases/tag/v0.9.1) | [Docs](https://docs.dynamo.nvidia.com/dynamo) |
| `v0.9.0` | Feb 11, 2026 | [Release](https://github.com/ai-dynamo/dynamo/releases/tag/v0.9.0) | Archived docs unavailable |
| `v0.8.1` | Jan 23, 2026 | [Release](https://github.com/ai-dynamo/dynamo/releases/tag/v0.8.1) | Archived docs unavailable |
| `v0.8.0` | Jan 15, 2026 | [Release](https://github.com/ai-dynamo/dynamo/releases/tag/v0.8.0) | Archived docs unavailable |
| `v0.7.1` | Dec 15, 2025 | [Release](https://github.com/ai-dynamo/dynamo/releases/tag/v0.7.1) | Archived docs unavailable |
| `v0.7.0` | Nov 26, 2025 | [Release](https://github.com/ai-dynamo/dynamo/releases/tag/v0.7.0) | Archived docs unavailable |
| `v0.6.1` | Nov 6, 2025 | [Release](https://github.com/ai-dynamo/dynamo/releases/tag/v0.6.1) | — |
| `v0.6.0` | Oct 28, 2025 | [Release](https://github.com/ai-dynamo/dynamo/releases/tag/v0.6.0) | — |

### Container Images

> **NGC Collection:** [ai-dynamo](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo)
>
> To access a specific version, append `?version=TAG` to the container URL:
> `https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/{container}?version={tag}`

#### vllm-runtime

| Image:Tag | vLLM | Arch | CUDA | Notes |
|-----------|------|------|------|-------|
| `vllm-runtime:1.0.1` | `v0.16.0` | AMD64/ARM64 | `v12.9` | |
| `vllm-runtime:1.0.1-cuda13` | `v0.16.0` | AMD64/ARM64* | `v13.0` | |
| `vllm-runtime:1.0.1-efa-amd64` | `v0.16.0` | AMD64 | `v12.9` | Experimental |
| `vllm-runtime:1.0.0` | `v0.16.0` | AMD64/ARM64 | `v12.9` | |
| `vllm-runtime:1.0.0-cuda13` | `v0.16.0` | AMD64/ARM64* | `v13.0` | |
| `vllm-runtime:1.0.0-efa-amd64` | `v0.16.0` | AMD64 | `v12.9` | Experimental |
| `vllm-runtime:0.9.1` | `v0.14.1` | AMD64/ARM64 | `v12.9` | |
| `vllm-runtime:0.9.1-cuda13` | `v0.14.1` | AMD64/ARM64* | `v13.0` | Experimental |
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
| `sglang-runtime:1.0.1` | `v0.5.9` | AMD64/ARM64 | `v12.9` | |
| `sglang-runtime:1.0.1-cuda13` | `v0.5.9` | AMD64/ARM64* | `v13.0` | |
| `sglang-runtime:1.0.0` | `v0.5.9` | AMD64/ARM64 | `v12.9` | |
| `sglang-runtime:1.0.0-cuda13` | `v0.5.9` | AMD64/ARM64* | `v13.0` | |
| `sglang-runtime:0.9.1` | `v0.5.8` | AMD64/ARM64 | `v12.9` | |
| `sglang-runtime:0.9.1-cuda13` | `v0.5.8` | AMD64/ARM64* | `v13.0` | Experimental |
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
| `tensorrtllm-runtime:1.0.1` | `v1.3.0rc5.post1` | AMD64/ARM64 | `v13.1` | |
| `tensorrtllm-runtime:1.0.1-efa-amd64` | `v1.3.0rc5.post1` | AMD64 | `v13.1` | Experimental |
| `tensorrtllm-runtime:1.0.0` | `v1.3.0rc5.post1` | AMD64/ARM64 | `v13.1` | |
| `tensorrtllm-runtime:1.0.0-efa-amd64` | `v1.3.0rc5.post1` | AMD64 | `v13.1` | Experimental |
| `tensorrtllm-runtime:0.9.1` | `v1.3.0rc3` | AMD64/ARM64 | `v13.0` | |
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
| `dynamo-frontend:1.0.1` | AMD64/ARM64 | |
| `dynamo-frontend:1.0.0` | AMD64/ARM64 | |
| `dynamo-frontend:0.9.1` | AMD64/ARM64 | |
| `dynamo-frontend:0.9.0` | AMD64/ARM64 | |
| `dynamo-frontend:0.8.1` | AMD64/ARM64 | |
| `dynamo-frontend:0.8.0` | AMD64/ARM64 | Initial |

#### kubernetes-operator

| Image:Tag | Arch | Notes |
|-----------|------|-------|
| `kubernetes-operator:1.0.1` | AMD64/ARM64 | |
| `kubernetes-operator:1.0.0` | AMD64/ARM64 | |
| `kubernetes-operator:0.9.1` | AMD64/ARM64 | |
| `kubernetes-operator:0.9.0` | AMD64/ARM64 | |
| `kubernetes-operator:0.8.1` | AMD64/ARM64 | |
| `kubernetes-operator:0.8.0` | AMD64/ARM64 | |
| `kubernetes-operator:0.7.1` | AMD64/ARM64 | |
| `kubernetes-operator:0.7.0.post1` | AMD64/ARM64 | Patch |
| `kubernetes-operator:0.7.0` | AMD64/ARM64 | |
| `kubernetes-operator:0.6.1` | AMD64/ARM64 | |
| `kubernetes-operator:0.6.0` | AMD64/ARM64 | |

#### snapshot-agent

| Image:Tag | Arch | Notes |
|-----------|------|-------|
| `snapshot-agent:1.0.1` | AMD64/ARM64 | Preview |
| `snapshot-agent:1.0.0` | AMD64/ARM64 | Preview |

### Python Wheels

> **PyPI:** [ai-dynamo](https://pypi.org/project/ai-dynamo/) | [ai-dynamo-runtime](https://pypi.org/project/ai-dynamo-runtime/) | [kvbm](https://pypi.org/project/kvbm/)
>
> To access a specific version: `https://pypi.org/project/{package}/{version}/`

#### ai-dynamo (wheel)

| Package | Python | Platform | Notes |
|---------|--------|----------|-------|
| `ai-dynamo==1.0.1` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `ai-dynamo==1.0.0` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `ai-dynamo==0.9.1` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
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
| `ai-dynamo-runtime==1.0.1` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `ai-dynamo-runtime==1.0.0` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `ai-dynamo-runtime==0.9.1` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
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
| `kvbm==1.0.1` | `3.12` | Linux (glibc `v2.28+`) | |
| `kvbm==1.0.0` | `3.12` | Linux (glibc `v2.28+`) | |
| `kvbm==0.9.1` | `3.12` | Linux (glibc `v2.28+`) | |
| `kvbm==0.9.0` | `3.12` | Linux (glibc `v2.28+`) | |
| `kvbm==0.8.1` | `3.12` | Linux (glibc `v2.28+`) | |
| `kvbm==0.8.0` | `3.12` | Linux (glibc `v2.28+`) | |
| `kvbm==0.7.1` | `3.12` | Linux (glibc `v2.28+`) | |
| `kvbm==0.7.0` | `3.12` | Linux (glibc `v2.28+`) | Initial |

### Helm Charts

> **NGC Helm Registry:** [ai-dynamo](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo)
>
> Direct download: `https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/{chart}-{version}.tgz`

#### dynamo-crds (Helm chart) -- Deprecated

> [!NOTE]
> The `dynamo-crds` Helm chart is deprecated as of v1.0.0. CRDs are now managed by the Dynamo Operator.

| Chart | Notes |
|-------|-------|
| `dynamo-crds-0.9.1` | Last release |
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
| `dynamo-platform-1.0.1` | |
| `dynamo-platform-1.0.0` | |
| `dynamo-platform-0.9.1` | |
| `dynamo-platform-0.9.0-post1` | Helm fix: operator image tag |
| `dynamo-platform-0.9.0` | |
| `dynamo-platform-0.8.1` | |
| `dynamo-platform-0.8.0` | |
| `dynamo-platform-0.7.1` | |
| `dynamo-platform-0.7.0` | |
| `dynamo-platform-0.6.1` | |
| `dynamo-platform-0.6.0` | |

#### snapshot (Helm chart)

| Chart | Notes |
|-------|-------|
| `snapshot-1.0.1` | Preview |
| `snapshot-1.0.0` | Preview |

#### dynamo-graph (Helm chart) -- Deprecated

> [!NOTE]
> The `dynamo-graph` Helm chart is deprecated as of v0.9.0.

| Chart | Notes |
|-------|-------|
| `dynamo-graph-0.8.1` | Last release |
| `dynamo-graph-0.8.0` | |
| `dynamo-graph-0.7.1` | |
| `dynamo-graph-0.7.0` | |
| `dynamo-graph-0.6.1` | |
| `dynamo-graph-0.6.0` | |

### Rust Crates

> **crates.io:** [dynamo-runtime](https://crates.io/crates/dynamo-runtime) | [dynamo-llm](https://crates.io/crates/dynamo-llm) | [dynamo-protocols](https://crates.io/crates/dynamo-protocols) | [dynamo-parsers](https://crates.io/crates/dynamo-parsers) | [dynamo-memory](https://crates.io/crates/dynamo-memory) | [dynamo-config](https://crates.io/crates/dynamo-config) | [dynamo-tokens](https://crates.io/crates/dynamo-tokens)
>
> To access a specific version: `https://crates.io/crates/{crate}/{version}`

#### dynamo-runtime (crate)

| Crate | MSRV (Rust) | Notes |
|-------|-------------|-------|
| `dynamo-runtime@1.0.1` | `v1.82` | |
| `dynamo-runtime@1.0.0` | `v1.82` | |
| `dynamo-runtime@0.9.1` | `v1.82` | |
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
| `dynamo-llm@1.0.1` | `v1.82` | |
| `dynamo-llm@1.0.0` | `v1.82` | |
| `dynamo-llm@0.9.1` | `v1.82` | |
| `dynamo-llm@0.9.0` | `v1.82` | |
| `dynamo-llm@0.8.1` | `v1.82` | |
| `dynamo-llm@0.8.0` | `v1.82` | |
| `dynamo-llm@0.7.1` | `v1.82` | |
| `dynamo-llm@0.7.0` | `v1.82` | |
| `dynamo-llm@0.6.1` | `v1.82` | |
| `dynamo-llm@0.6.0` | `v1.82` | |

#### dynamo-protocols (crate)

| Crate | MSRV (Rust) | Notes |
|-------|-------------|-------|
| `dynamo-protocols@1.0.1` | `v1.82` | |
| `dynamo-protocols@1.0.0` | `v1.82` | |
| `dynamo-protocols@0.9.1` | `v1.82` | |
| `dynamo-protocols@0.9.0` | `v1.82` | |
| `dynamo-protocols@0.8.1` | `v1.82` | |
| `dynamo-protocols@0.8.0` | `v1.82` | |
| `dynamo-protocols@0.7.1` | `v1.82` | |
| `dynamo-protocols@0.7.0` | `v1.82` | |
| `dynamo-protocols@0.6.1` | `v1.82` | |
| `dynamo-protocols@0.6.0` | `v1.82` | |

#### dynamo-parsers (crate)

| Crate | MSRV (Rust) | Notes |
|-------|-------------|-------|
| `dynamo-parsers@1.0.1` | `v1.82` | |
| `dynamo-parsers@1.0.0` | `v1.82` | |
| `dynamo-parsers@0.9.1` | `v1.82` | |
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
| `dynamo-memory@1.0.1` | `v1.82` | |
| `dynamo-memory@1.0.0` | `v1.82` | |
| `dynamo-memory@0.9.1` | `v1.82` | |
| `dynamo-memory@0.9.0` | `v1.82` | |
| `dynamo-memory@0.8.1` | `v1.82` | |
| `dynamo-memory@0.8.0` | `v1.82` | Initial |

#### dynamo-config (crate)

| Crate | MSRV (Rust) | Notes |
|-------|-------------|-------|
| `dynamo-config@1.0.1` | `v1.82` | |
| `dynamo-config@1.0.0` | `v1.82` | |
| `dynamo-config@0.9.1` | `v1.82` | |
| `dynamo-config@0.9.0` | `v1.82` | |
| `dynamo-config@0.8.1` | `v1.82` | |
| `dynamo-config@0.8.0` | `v1.82` | Initial |

#### dynamo-tokens (crate)

| Crate | MSRV (Rust) | Notes |
|-------|-------------|-------|
| `dynamo-tokens@1.0.1` | `v1.82` | |
| `dynamo-tokens@1.0.0` | `v1.82` | |
| `dynamo-tokens@0.9.1` | `v1.82` | |
| `dynamo-tokens@0.9.0` | `v1.82` | Initial |

#### dynamo-mocker (crate)

| Crate | MSRV (Rust) | Notes |
|-------|-------------|-------|
| `dynamo-mocker@1.0.1` | `v1.82` | |
| `dynamo-mocker@1.0.0` | `v1.82` | Initial |

#### dynamo-kv-router (crate)

| Crate | MSRV (Rust) | Notes |
|-------|-------------|-------|
| `dynamo-kv-router@1.0.1` | `v1.82` | |
| `dynamo-kv-router@1.0.0` | `v1.82` | Initial |

---

## Pre-Release Artifacts

> [!WARNING]
> **Pre-Release artifacts do not go through QA validation.** Pre-release versions are experimental previews intended for early testing and feedback. They may contain bugs, breaking changes, or incomplete features. Use stable releases for production workloads.

### v1.1.0-dev.1

- **Branch:** [release/1.1.0-dev.1](https://github.com/ai-dynamo/dynamo/tree/release/1.1.0-dev.1)
- **GitHub Tag:** [v1.1.0-dev.1](https://github.com/ai-dynamo/dynamo/releases/tag/v1.1.0-dev.1)
- **Backends:** SGLang `v0.5.9` | TensorRT-LLM `v1.3.0rc5.post1` | vLLM `v0.17.1` | NIXL `v0.10.1`

#### Container Images

| Image:Tag | Backend | CUDA | Arch |
|-----------|---------|------|------|
| `vllm-runtime:1.1.0-dev.1` | vLLM `v0.17.1` | `v12.9` | AMD64/ARM64 |
| `vllm-runtime:1.1.0-dev.1-cuda13` | vLLM `v0.17.1` | `v13.0` | AMD64/ARM64* |
| `vllm-runtime:1.1.0-dev.1-efa-amd64` | vLLM `v0.17.1` | `v12.9` | AMD64 |
| `sglang-runtime:1.1.0-dev.1` | SGLang `v0.5.9` | `v12.9` | AMD64/ARM64 |
| `sglang-runtime:1.1.0-dev.1-cuda13` | SGLang `v0.5.9` | `v13.0` | AMD64/ARM64* |
| `tensorrtllm-runtime:1.1.0-dev.1` | TRT-LLM `v1.3.0rc5.post1` | `v13.1` | AMD64/ARM64 |
| `tensorrtllm-runtime:1.1.0-dev.1-efa-amd64` | TRT-LLM `v1.3.0rc5.post1` | `v13.1` | AMD64 |
| `dynamo-frontend:1.1.0-dev.1` | — | — | AMD64/ARM64 |
| `kubernetes-operator:1.1.0-dev.1` | — | — | AMD64/ARM64 |
| `snapshot-agent:1.1.0-dev.1` | — | — | AMD64/ARM64 |

#### Python Wheels

Available from [pypi.nvidia.com](https://pypi.nvidia.com/) (pre-release index):

```bash
uv pip install --pre --extra-index-url https://pypi.nvidia.com/ ai-dynamo==1.1.0.dev1
uv pip install --pre --extra-index-url https://pypi.nvidia.com/ ai-dynamo-runtime==1.1.0.dev1
uv pip install --pre --extra-index-url https://pypi.nvidia.com/ kvbm==1.1.0.dev1
```

#### Helm Charts

| Chart | NGC |
|-------|-----|
| `dynamo-platform-1.1.0-dev.1` | [link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/helm-charts/dynamo-platform?version=1.1.0-dev.1) |
| `snapshot-1.1.0-dev.1` | [link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/helm-charts/snapshot?version=1.1.0-dev.1) |

#### Rust Crates

Not shipped for pre-release versions.
