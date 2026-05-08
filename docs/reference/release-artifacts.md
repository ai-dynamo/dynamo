---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Release Artifacts
subtitle: Container images, Python wheels, Helm charts, Rust crates, and release history
---

This document provides a comprehensive inventory of all Dynamo release artifacts including container images, Python wheels, Helm charts, and Rust crates.

> **See also:** [Support Matrix](support-matrix.md) for hardware and platform compatibility | [Feature Matrix](feature-matrix.md) for backend feature support

Release history in this document begins at v0.6.0.

## Current Release: Dynamo v1.1.1

- **GitHub Release:** [v1.1.1](https://github.com/ai-dynamo/dynamo/releases/tag/v1.1.1)
- **Docs:** [v1.1.1](https://docs.dynamo.nvidia.com/dynamo)
- **NGC Collection:** [ai-dynamo](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo)

> **Experimental:** [v1.2.0-deepseek-v4-dev.2](#v120-deepseek-v4-dev2) *(DeepSeek-V4-Flash / V4-Pro on Blackwell, vLLM + SGLang containers only)* is available as an experimental preview. Tagged **Pre-Releases** and experimental builds are listed under [Pre-Release Artifacts](#pre-release-artifacts).

### Container Images

| Image:Tag | Description | Backend | CUDA | Arch | NGC | Notes |
|-----------|-------------|---------|------|------|-----|-------|
| `vllm-runtime:1.1.1` | Runtime container for vLLM backend | vLLM `v0.19.0` | `v12.9` | AMD64/ARM64 | [NGC: vllm-runtime 1.1.1](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/vllm-runtime?version=1.1.1) | |
| `vllm-runtime:1.1.1-cuda13` | Runtime container for vLLM backend (CUDA 13) | vLLM `v0.19.0` | `v13.0` | AMD64/ARM64 | [NGC: vllm-runtime 1.1.1-cuda13](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/vllm-runtime?version=1.1.1-cuda13) | |
| `vllm-runtime:1.1.1-efa-amd64` | Runtime container for vLLM with AWS EFA | vLLM `v0.19.0` | `v12.9` | AMD64 | [NGC: vllm-runtime 1.1.1-efa-amd64](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/vllm-runtime?version=1.1.1-efa-amd64) | Experimental |
| `sglang-runtime:1.1.1` | Runtime container for SGLang backend | SGLang `v0.5.10.post1` | `v12.9` | AMD64/ARM64 | [NGC: sglang-runtime 1.1.1](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/sglang-runtime?version=1.1.1) | |
| `sglang-runtime:1.1.1-cuda13` | Runtime container for SGLang backend (CUDA 13) | SGLang `v0.5.10.post1` | `v13.0` | AMD64/ARM64 | [NGC: sglang-runtime 1.1.1-cuda13](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/sglang-runtime?version=1.1.1-cuda13) | |
| `tensorrtllm-runtime:1.1.1` | Runtime container for TensorRT-LLM backend | TRT-LLM `v1.3.0rc11` | `v13.1` | AMD64/ARM64 | [NGC: tensorrtllm-runtime 1.1.1](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/tensorrtllm-runtime?version=1.1.1) | |
| `tensorrtllm-runtime:1.1.1-efa-amd64` | Runtime container for TensorRT-LLM with AWS EFA | TRT-LLM `v1.3.0rc11` | `v13.1` | AMD64 | [NGC: tensorrtllm-runtime 1.1.1-efa-amd64](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/tensorrtllm-runtime?version=1.1.1-efa-amd64) | Experimental |
| `dynamo-frontend:1.1.1` | API gateway with Endpoint Prediction Protocol (EPP) | — | — | AMD64/ARM64 | [NGC: dynamo-frontend 1.1.1](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/dynamo-frontend?version=1.1.1) | |
| `dynamo-planner:1.1.1` | Standalone Planner image used by Profiler jobs and Planner pods | — | — | AMD64/ARM64 | [NGC: dynamo-planner 1.1.1](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/dynamo-planner?version=1.1.1) | |
| `kubernetes-operator:1.1.1` | Kubernetes operator for Dynamo deployments | — | — | AMD64/ARM64 | [NGC: kubernetes-operator 1.1.1](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/kubernetes-operator?version=1.1.1) | |
| `snapshot-agent:1.1.1` | Snapshot agent for fast GPU worker recovery via CRIU | — | — | AMD64/ARM64 | [NGC: snapshot-agent 1.1.1](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/snapshot-agent?version=1.1.1) | Preview |

### Python Wheels

We recommend using the TensorRT-LLM NGC container instead of the `ai-dynamo[trtllm]` wheel. See the [NGC container collection](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo) for supported images.

| Package | Description | Python | Platform | PyPI |
|---------|-------------|--------|----------|------|
| `ai-dynamo==1.1.1` | Main package with backend integrations (vLLM, SGLang, TRT-LLM) | `3.10`–`3.12` | Linux (glibc `v2.28+`) | [PyPI: ai-dynamo 1.1.1](https://pypi.org/project/ai-dynamo/1.1.1/) |
| `ai-dynamo-runtime==1.1.1` | Core Python bindings for Dynamo runtime | `3.10`–`3.12` | Linux (glibc `v2.28+`) | [PyPI: ai-dynamo-runtime 1.1.1](https://pypi.org/project/ai-dynamo-runtime/1.1.1/) |
| `kvbm==1.1.1` | KV Block Manager for disaggregated KV cache | `3.10`–`3.12` | Linux (glibc `v2.28+`) | [PyPI: kvbm 1.1.1](https://pypi.org/project/kvbm/1.1.1/) |

### Helm Charts

| Chart | Description | NGC |
|-------|-------------|-----|
| `dynamo-platform-1.1.1` | Platform services (etcd, NATS) and Dynamo Operator for Dynamo cluster | [NGC Helm: dynamo-platform-1.1.1](https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-1.1.1.tgz) |
| `snapshot-1.1.1` | Snapshot DaemonSet for fast GPU worker recovery | [NGC Helm: snapshot-1.1.1](https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/snapshot-1.1.1.tgz) |

> [!NOTE]
> The `dynamo-crds` Helm chart is deprecated as of v1.0.0; CRDs are now managed by the Dynamo Operator. The `dynamo-graph` Helm chart is deprecated as of v0.9.0.

### Rust Crates

| Crate | Description | MSRV (Rust) | crates.io |
|-------|-------------|-------------|-----------|
| `dynamo-runtime@1.1.1` | Core distributed runtime library | `v1.82` | [crates.io: dynamo-runtime 1.1.1](https://crates.io/crates/dynamo-runtime/1.1.1) |
| `dynamo-llm@1.1.1` | LLM inference engine | `v1.82` | [crates.io: dynamo-llm 1.1.1](https://crates.io/crates/dynamo-llm/1.1.1) |
| `dynamo-protocols@1.1.1` | Async OpenAI-compatible API client | `v1.82` | [crates.io: dynamo-protocols 1.1.1](https://crates.io/crates/dynamo-protocols/1.1.1) |
| `dynamo-async-openai@1.0.2` | Deprecated legacy OpenAI client; use **`dynamo-protocols`** | `v1.82` | [crates.io: dynamo-async-openai 1.0.2](https://crates.io/crates/dynamo-async-openai/1.0.2) |
| `dynamo-parsers@1.1.1` | Protocol parsers (SSE, JSON streaming) | `v1.82` | [crates.io: dynamo-parsers 1.1.1](https://crates.io/crates/dynamo-parsers/1.1.1) |
| `dynamo-memory@1.1.1` | Memory management utilities | `v1.82` | [crates.io: dynamo-memory 1.1.1](https://crates.io/crates/dynamo-memory/1.1.1) |
| `dynamo-config@1.1.1` | Configuration management | `v1.82` | [crates.io: dynamo-config 1.1.1](https://crates.io/crates/dynamo-config/1.1.1) |
| `dynamo-tokens@1.1.1` | Tokenizer bindings for LLM inference | `v1.82` | [crates.io: dynamo-tokens 1.1.1](https://crates.io/crates/dynamo-tokens/1.1.1) |
| `dynamo-mocker@1.1.1` | Inference engine simulator for benchmarking | `v1.82` | [crates.io: dynamo-mocker 1.1.1](https://crates.io/crates/dynamo-mocker/1.1.1) |
| `dynamo-kv-router@1.1.1` | KV-aware request routing library | `v1.82` | [crates.io: dynamo-kv-router 1.1.1](https://crates.io/crates/dynamo-kv-router/1.1.1) |

## Quick Install Commands

### Container Images (NGC)

> [!TIP]
> For detailed run instructions, see the backend-specific guides: [vLLM](../backends/vllm/README.md) | [SGLang](../backends/sglang/README.md) | [TensorRT-LLM](../backends/trtllm/README.md)

```bash
# Runtime containers
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1
docker pull nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.1
docker pull nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.1.1

# CUDA 13 variants
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1-cuda13
docker pull nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.1-cuda13

# EFA variants (AWS, AMD64 only, experimental)
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1-efa-amd64
docker pull nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.1.1-efa-amd64

# Infrastructure containers
docker pull nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.1.1
docker pull nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.1.1
docker pull nvcr.io/nvidia/ai-dynamo/kubernetes-operator:1.1.1
docker pull nvcr.io/nvidia/ai-dynamo/snapshot-agent:1.1.1
```

### Python Wheels (PyPI)

> [!TIP]
> For detailed installation instructions, see the [Local Quick Start](https://github.com/ai-dynamo/dynamo#local-quick-start) in the README.

```bash
# Install Dynamo with a specific backend (Recommended)
uv pip install "ai-dynamo[vllm]==1.1.1"
uv pip install --prerelease=allow "ai-dynamo[sglang]==1.1.1"
# TensorRT-LLM requires the NVIDIA PyPI index and pip
pip install --pre --extra-index-url https://pypi.nvidia.com "ai-dynamo[trtllm]==1.1.1"

# Install Dynamo core only
uv pip install ai-dynamo==1.1.1

# Install standalone KVBM
uv pip install kvbm==1.1.1
```

### Helm Charts (NGC)

> [!TIP]
> For Kubernetes deployment instructions, see the [Kubernetes Installation Guide](../kubernetes/installation-guide.md).

```bash
helm install dynamo-platform oci://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform --version 1.1.1
helm install snapshot oci://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/snapshot --version 1.1.1
```

### Rust Crates (crates.io)

> [!TIP]
> For API documentation, see each crate on [docs.rs](https://docs.rs/). To build Dynamo from source, see [Building from Source](https://github.com/ai-dynamo/dynamo#building-from-source).

```bash
cargo add dynamo-runtime@1.1.1
cargo add dynamo-llm@1.1.1
cargo add dynamo-protocols@1.1.1
# Deprecated legacy crate name — pin only if a dependency requires it; new code should use dynamo-protocols:
# cargo add dynamo-async-openai@1.0.2
cargo add dynamo-parsers@1.1.1
cargo add dynamo-memory@1.1.1
cargo add dynamo-config@1.1.1
cargo add dynamo-tokens@1.1.1
cargo add dynamo-mocker@1.1.1
cargo add dynamo-kv-router@1.1.1
```

**CUDA and Driver Requirements:** For detailed CUDA toolkit versions and minimum driver requirements for each container image, see the [Support Matrix](support-matrix.md#cuda-and-driver-requirements).

## Known Issues

For a complete list of known issues, refer to the release notes for each version:
- [v1.1.1 Release Notes](https://github.com/ai-dynamo/dynamo/releases/tag/v1.1.1)
- [v1.1.0 Release Notes](https://github.com/ai-dynamo/dynamo/releases/tag/v1.1.0)
- [v1.0.2 Release Notes](https://github.com/ai-dynamo/dynamo/releases/tag/v1.0.2)
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

## Release Artifact History

Each bullet is a **delta** to what ships on NGC / Helm / PyPI / crates.io: net-new crates, removed Helm charts, or image lines that **split** or **appear** on the registry. See the inventory tables above for full matrices.

Stable releases first (newest first). **Pre-Release Git Tags** (`v*-dev.*`, experimental tracks) are summarized below; per-tag images and wheels are spelled out in [Pre-Release Artifacts](#pre-release-artifacts).

For backend version pins, see the version-pins table above and the [GitHub Releases](#github-releases) table below.

**Stable Releases**

- **v1.1.1**: Patch release. Same backend versions as v1.1.0: SGLang `v0.5.10.post1` (NIXL `v1.0.1`), TRT-LLM `v1.3.0rc11` (NIXL `v0.10.1`), vLLM `v0.19.0` (NIXL `v0.10.1`).
- **v1.1.0**: **Images:** Split Planner into its own `dynamo-planner` image on NGC for Profiler jobs and Planner pods; worker and runtime images no longer bundle Planner (**artifact boundary change**, not a new engine capability). **Crates:** First **`1.y.z`** publication on crates.io for **`dynamo-protocols`** (multi-protocol types; **`dynamo-async-openai`** remains deprecated with final release **`1.0.2`**).
- **v1.0.2 / v1.0.1**: No artifact additions or removals versus v1.0.0.
- **v1.0.0**: **Images:** `snapshot-agent`, EFA variants for vLLM and TRT-LLM (AMD64 only). **Crates:** First publish of `dynamo-mocker`, `dynamo-kv-router`. **Helm:** Added `snapshot` (preview); dropped deprecated `dynamo-crds` from the publish stream (CRDs owned by the Operator).
- **v0.9.1**: No artifact additions or removals versus v0.9.0.
- **v0.9.0**: **Crates:** First publish of `dynamo-tokens`. **Helm:** Dropped deprecated `dynamo-graph` from the publish stream.
- **v0.8.0**: **Images:** `dynamo-frontend`, CUDA 13 variants for vLLM and SGLang. **Crates:** First publish of `dynamo-memory`, `dynamo-config`.

**Dynamo Nightlies**

- **New as of v1.1.0\*:** **`ai-dynamo`** and **`ai-dynamo-runtime`** — nightly builds from **`main`** publish wheels tagged **`*.devYYYYMMDD`**. Install with **`pip`** or **`uv`** using **`--pre`** and the same NVIDIA extra-index pattern as [Pre-Release Artifacts](#pre-release-artifacts).

\* **`*.devYYYYMMDD`** versioning for nightly **`main`** wheels began **Apr 24, 2026**.

**Pre-Release and Experimental Git Tags**

- **v1.2.0-deepseek-v4-dev.2**: **Images:** `vllm-runtime:*-deepseek-v4-cuda13-dev.2`, `sglang-runtime:*-deepseek-v4-cuda12-dev.2`, `sglang-runtime:*-deepseek-v4-cuda13-dev.2`. **Helm / PyPI:** Not published for this tag (see [Pre-Release Artifacts](#v120-deepseek-v4-dev2)).
- **v1.1.0-dev.3**: **Images:** `tensorrtllm-runtime:1.1.0-dev.3`. **Wheels:** `ai-dynamo`, `ai-dynamo-runtime` on [pypi.nvidia.com](https://pypi.nvidia.com/) (see [below](#v110-dev3)).
- **v1.1.0-dev.2**: **Images:** `sglang-runtime:1.1.0-dev.2`, `tensorrtllm-runtime:1.1.0-dev.2`. **Wheels:** `ai-dynamo`, `ai-dynamo-runtime` on [pypi.nvidia.com](https://pypi.nvidia.com/) (see [below](#v110-dev2)).
- **v1.1.0-dev.1**: **Images:** vLLM, SGLang, TRT-LLM runtime matrix (CUDA 12 / 13 and EFA variants as listed), `dynamo-frontend`, `kubernetes-operator`, `snapshot-agent`. **Wheels:** `ai-dynamo`, `ai-dynamo-runtime` on [pypi.nvidia.com](https://pypi.nvidia.com/). **Helm:** `dynamo-platform`, `snapshot` at `1.1.0-dev.1` (see [below](#v110-dev1)).

**Helm-Only Patches**

- **v0.9.0.post1**: Republished `dynamo-platform` Helm chart only (operator image tag correction).

**Backend-Only Patch Trains**

- **v0.8.1.post1 / .post2 / .post3**: Republished TRT-LLM runtime image and PyPI wheels only.

### crates.io Rust Packages

These crates use repository `https://github.com/ai-dynamo/dynamo.git`. The table lists each crate’s **first non-placeholder** publication on crates.io (excluding reservation uploads named `0.0.0-prerelease.0`). Dates are from the crates.io registry index.

| Crate | First Published Version | Date (crates.io) |
|-------|-------------------------|------------------|
| `dynamo-runtime` | `0.1.0` | 2025-03-18 |
| `dynamo-llm` | `0.2.0` | 2025-05-01 |
| `dynamo-async-openai` | `0.4.1` | 2025-08-27 |
| `dynamo-parsers` | `0.5.0` | 2025-09-18 |
| `dynamo-memory` | `0.8.0` | 2026-01-15 |
| `dynamo-config` | `0.8.0` | 2026-01-15 |
| `dynamo-tokens` | `0.9.0` | 2026-02-12 |
| `dynamo-mocker` | `1.0.0` | 2026-03-13 |
| `dynamo-kv-router` | `1.0.0` | 2026-03-13 |
| `dynamo-protocols` | `1.1.0` | 2026-05-04 |

**`dynamo-async-openai`** is **deprecated**; **`1.0.2`** is its final crates.io release. Use **`dynamo-protocols`** for new dependencies ([crate](https://crates.io/crates/dynamo-protocols)).

**`dynamo-tokenizers`** exists only in the Dynamo workspace and is **not** published on crates.io.

### GitHub Releases

| Version | Release Date | GitHub | Docs | Notes |
|---------|--------------|--------|------|-------|
| `v1.2.0-deepseek-v4-dev.2` | May 1, 2026 | [Tag](https://github.com/ai-dynamo/dynamo/releases/tag/v1.2.0-deepseek-v4-dev.2) | — | Experimental (DeepSeek-V4-Flash / V4-Pro Blackwell preview; vLLM + SGLang containers only) |
| `v1.1.1` | May 5, 2026 | [Release](https://github.com/ai-dynamo/dynamo/releases/tag/v1.1.1) | [Docs](https://docs.dynamo.nvidia.com/dynamo) | |
| `v1.1.0` | May 1, 2026 | [Release](https://github.com/ai-dynamo/dynamo/releases/tag/v1.1.0) | [Docs](https://docs.dynamo.nvidia.com/dynamo) | |
| `v1.1.0-dev.3` | Apr 18, 2026 | [Tag](https://github.com/ai-dynamo/dynamo/releases/tag/v1.1.0-dev.3) | — | Pre-Release (TRT-LLM Runtime Image + Wheels; see Pre-Release Artifacts) |
| `v1.1.0-dev.2` | Apr 9, 2026 | [Tag](https://github.com/ai-dynamo/dynamo/releases/tag/v1.1.0-dev.2) | — | Pre-Release (SGLang + TRT-LLM Runtime Images + Wheels; see Pre-Release Artifacts) |
| `v1.1.0-dev.1` | Mar 17, 2026 | [Tag](https://github.com/ai-dynamo/dynamo/releases/tag/v1.1.0-dev.1) | — | Experimental |
| `v1.0.2` | Apr 22, 2026 | [Release](https://github.com/ai-dynamo/dynamo/releases/tag/v1.0.2) | [Docs](https://docs.dynamo.nvidia.com/dynamo) | |
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
| `vllm-runtime:1.1.1` | `v0.19.0` | AMD64/ARM64 | `v12.9` | |
| `vllm-runtime:1.1.1-cuda13` | `v0.19.0` | AMD64/ARM64 | `v13.0` | |
| `vllm-runtime:1.1.1-efa-amd64` | `v0.19.0` | AMD64 | `v12.9` | Experimental |
| `vllm-runtime:1.1.0` | `v0.19.0` | AMD64/ARM64 | `v12.9` | |
| `vllm-runtime:1.1.0-cuda13` | `v0.19.0` | AMD64/ARM64 | `v13.0` | |
| `vllm-runtime:1.1.0-efa-amd64` | `v0.19.0` | AMD64 | `v12.9` | Experimental |
| `vllm-runtime:1.0.2` | `v0.16.0` | AMD64/ARM64 | `v12.9` | |
| `vllm-runtime:1.0.2-cuda13` | `v0.16.0` | AMD64/ARM64 | `v13.0` | |
| `vllm-runtime:1.0.2-efa-amd64` | `v0.16.0` | AMD64 | `v12.9` | Experimental |
| `vllm-runtime:1.0.1` | `v0.16.0` | AMD64/ARM64 | `v12.9` | |
| `vllm-runtime:1.0.1-cuda13` | `v0.16.0` | AMD64/ARM64 | `v13.0` | |
| `vllm-runtime:1.0.1-efa-amd64` | `v0.16.0` | AMD64 | `v12.9` | Experimental |
| `vllm-runtime:1.0.0` | `v0.16.0` | AMD64/ARM64 | `v12.9` | |
| `vllm-runtime:1.0.0-cuda13` | `v0.16.0` | AMD64/ARM64 | `v13.0` | |
| `vllm-runtime:1.0.0-efa-amd64` | `v0.16.0` | AMD64 | `v12.9` | Experimental |
| `vllm-runtime:0.9.1` | `v0.14.1` | AMD64/ARM64 | `v12.9` | |
| `vllm-runtime:0.9.1-cuda13` | `v0.14.1` | AMD64/ARM64 | `v13.0` | Experimental |
| `vllm-runtime:0.9.0` | `v0.14.1` | AMD64/ARM64 | `v12.9` | |
| `vllm-runtime:0.9.0-cuda13` | `v0.14.1` | AMD64/ARM64 | `v13.0` | Experimental |
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
| `sglang-runtime:1.1.1` | `v0.5.10.post1` | AMD64/ARM64 | `v12.9` | |
| `sglang-runtime:1.1.1-cuda13` | `v0.5.10.post1` | AMD64/ARM64 | `v13.0` | |
| `sglang-runtime:1.1.0` | `v0.5.10.post1` | AMD64/ARM64 | `v12.9` | |
| `sglang-runtime:1.1.0-cuda13` | `v0.5.10.post1` | AMD64/ARM64 | `v13.0` | |
| `sglang-runtime:1.0.2` | `v0.5.9` | AMD64/ARM64 | `v12.9` | |
| `sglang-runtime:1.0.2-cuda13` | `v0.5.9` | AMD64/ARM64 | `v13.0` | |
| `sglang-runtime:1.0.1` | `v0.5.9` | AMD64/ARM64 | `v12.9` | |
| `sglang-runtime:1.0.1-cuda13` | `v0.5.9` | AMD64/ARM64 | `v13.0` | |
| `sglang-runtime:1.0.0` | `v0.5.9` | AMD64/ARM64 | `v12.9` | |
| `sglang-runtime:1.0.0-cuda13` | `v0.5.9` | AMD64/ARM64 | `v13.0` | |
| `sglang-runtime:0.9.1` | `v0.5.8` | AMD64/ARM64 | `v12.9` | |
| `sglang-runtime:0.9.1-cuda13` | `v0.5.8` | AMD64/ARM64 | `v13.0` | Experimental |
| `sglang-runtime:0.9.0` | `v0.5.8` | AMD64/ARM64 | `v12.9` | |
| `sglang-runtime:0.9.0-cuda13` | `v0.5.8` | AMD64/ARM64 | `v13.0` | Experimental |
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
| `tensorrtllm-runtime:1.1.1` | `v1.3.0rc11` | AMD64/ARM64 | `v13.1` | |
| `tensorrtllm-runtime:1.1.1-efa-amd64` | `v1.3.0rc11` | AMD64 | `v13.1` | Experimental |
| `tensorrtllm-runtime:1.1.0` | `v1.3.0rc11` | AMD64/ARM64 | `v13.1` | |
| `tensorrtllm-runtime:1.1.0-efa-amd64` | `v1.3.0rc11` | AMD64 | `v13.1` | Experimental |
| `tensorrtllm-runtime:1.0.2` | `v1.3.0rc5.post1` | AMD64/ARM64 | `v13.1` | |
| `tensorrtllm-runtime:1.0.2-efa-amd64` | `v1.3.0rc5.post1` | AMD64 | `v13.1` | Experimental |
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
| `dynamo-frontend:1.1.1` | AMD64/ARM64 | |
| `dynamo-frontend:1.1.0` | AMD64/ARM64 | |
| `dynamo-frontend:1.0.2` | AMD64/ARM64 | |
| `dynamo-frontend:1.0.1` | AMD64/ARM64 | |
| `dynamo-frontend:1.0.0` | AMD64/ARM64 | |
| `dynamo-frontend:0.9.1` | AMD64/ARM64 | |
| `dynamo-frontend:0.9.0` | AMD64/ARM64 | |
| `dynamo-frontend:0.8.1` | AMD64/ARM64 | |
| `dynamo-frontend:0.8.0` | AMD64/ARM64 | Initial |

#### kubernetes-operator

| Image:Tag | Arch | Notes |
|-----------|------|-------|
| `kubernetes-operator:1.1.1` | AMD64/ARM64 | |
| `kubernetes-operator:1.1.0` | AMD64/ARM64 | |
| `kubernetes-operator:1.0.2` | AMD64/ARM64 | |
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

#### dynamo-planner

| Image:Tag | Arch | Notes |
|-----------|------|-------|
| `dynamo-planner:1.1.1` | AMD64/ARM64 | |
| `dynamo-planner:1.1.0` | AMD64/ARM64 | New |

#### snapshot-agent

| Image:Tag | Arch | Notes |
|-----------|------|-------|
| `snapshot-agent:1.1.1` | AMD64/ARM64 | Preview |
| `snapshot-agent:1.1.0` | AMD64/ARM64 | Preview |
| `snapshot-agent:1.0.2` | AMD64/ARM64 | Preview |
| `snapshot-agent:1.0.1` | AMD64/ARM64 | Preview |
| `snapshot-agent:1.0.0` | AMD64/ARM64 | Preview |

### Python Wheels

> **PyPI:** [ai-dynamo](https://pypi.org/project/ai-dynamo/) | [ai-dynamo-runtime](https://pypi.org/project/ai-dynamo-runtime/) | [kvbm](https://pypi.org/project/kvbm/)
>
> To access a specific version: `https://pypi.org/project/{package}/{version}/`

#### ai-dynamo (wheel)

| Package | Python | Platform | Notes |
|---------|--------|----------|-------|
| `ai-dynamo==1.1.1` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `ai-dynamo==1.1.0` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `ai-dynamo==1.0.2` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
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
| `ai-dynamo-runtime==1.1.1` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `ai-dynamo-runtime==1.1.0` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `ai-dynamo-runtime==1.0.2` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
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
| `kvbm==1.1.1` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `kvbm==1.1.0` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `kvbm==1.0.2` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `kvbm==1.0.1` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `kvbm==1.0.0` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `kvbm==0.9.1` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `kvbm==0.9.0` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `kvbm==0.8.1` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `kvbm==0.8.0` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `kvbm==0.7.1` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | |
| `kvbm==0.7.0` | `3.10`–`3.12` | Linux (glibc `v2.28+`) | Initial |

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
| `dynamo-platform-1.1.1` | |
| `dynamo-platform-1.1.0` | |
| `dynamo-platform-1.0.2` | |
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
| `snapshot-1.1.1` | Preview |
| `snapshot-1.1.0` | Preview |
| `snapshot-1.0.2` | Preview |
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

> **crates.io:** [dynamo-runtime](https://crates.io/crates/dynamo-runtime) | [dynamo-llm](https://crates.io/crates/dynamo-llm) | [dynamo-protocols](https://crates.io/crates/dynamo-protocols) | [dynamo-async-openai](https://crates.io/crates/dynamo-async-openai) *(deprecated)* | [dynamo-parsers](https://crates.io/crates/dynamo-parsers) | [dynamo-memory](https://crates.io/crates/dynamo-memory) | [dynamo-config](https://crates.io/crates/dynamo-config) | [dynamo-tokens](https://crates.io/crates/dynamo-tokens)
>
> To access a specific version: `https://crates.io/crates/{crate}/{version}`

#### dynamo-runtime (crate)

| Crate | MSRV (Rust) | Notes |
|-------|-------------|-------|
| `dynamo-runtime@1.1.1` | `v1.82` | |
| `dynamo-runtime@1.1.0` | `v1.82` | |
| `dynamo-runtime@1.0.2` | `v1.82` | |
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
| `dynamo-llm@1.1.1` | `v1.82` | |
| `dynamo-llm@1.1.0` | `v1.82` | |
| `dynamo-llm@1.0.2` | `v1.82` | |
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

On crates.io, **`dynamo-protocols`** lists **`1.1.0`** as its first installable release (placeholder reservation **`0.0.0-prerelease.0`** omitted here like other **`0.0.0-prerelease.*`** uploads). Earlier semver lines for the OpenAI-compatible client shipped under **`dynamo-async-openai`** — see **`#### dynamo-async-openai (crate)`** below.

| Crate | MSRV (Rust) | Notes |
|-------|-------------|-------|
| `dynamo-protocols@1.1.1` | `v1.82` | |
| `dynamo-protocols@1.1.0` | `v1.82` | |

#### dynamo-async-openai (crate)

**Deprecated.** Prefer **`dynamo-protocols`**. This crate remains published on crates.io for manifests pinned to the old package name.

| Crate | MSRV (Rust) | Notes |
|-------|-------------|-------|
| `dynamo-async-openai@1.0.2` | `v1.82` | Final crates.io release |
| `dynamo-async-openai@1.0.1` | `v1.82` | |
| `dynamo-async-openai@1.0.0` | `v1.82` | |
| `dynamo-async-openai@0.9.1` | `v1.82` | |
| `dynamo-async-openai@0.9.0` | `v1.82` | |
| `dynamo-async-openai@0.8.1` | `v1.82` | |
| `dynamo-async-openai@0.8.0` | `v1.82` | |
| `dynamo-async-openai@0.7.1` | `v1.82` | |
| `dynamo-async-openai@0.7.0` | `v1.82` | |
| `dynamo-async-openai@0.7.0-post1` | `v1.82` | |
| `dynamo-async-openai@0.6.1` | `v1.82` | |
| `dynamo-async-openai@0.6.0` | `v1.82` | |
| `dynamo-async-openai@0.5.1` | `v1.82` | |
| `dynamo-async-openai@0.5.0` | `v1.82` | |
| `dynamo-async-openai@0.4.1` | `v1.82` | |

#### dynamo-parsers (crate)

| Crate | MSRV (Rust) | Notes |
|-------|-------------|-------|
| `dynamo-parsers@1.1.1` | `v1.82` | |
| `dynamo-parsers@1.1.0` | `v1.82` | |
| `dynamo-parsers@1.0.2` | `v1.82` | |
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
| `dynamo-memory@1.1.1` | `v1.82` | |
| `dynamo-memory@1.1.0` | `v1.82` | |
| `dynamo-memory@1.0.2` | `v1.82` | |
| `dynamo-memory@1.0.1` | `v1.82` | |
| `dynamo-memory@1.0.0` | `v1.82` | |
| `dynamo-memory@0.9.1` | `v1.82` | |
| `dynamo-memory@0.9.0` | `v1.82` | |
| `dynamo-memory@0.8.1` | `v1.82` | |
| `dynamo-memory@0.8.0` | `v1.82` | Initial |

#### dynamo-config (crate)

| Crate | MSRV (Rust) | Notes |
|-------|-------------|-------|
| `dynamo-config@1.1.1` | `v1.82` | |
| `dynamo-config@1.1.0` | `v1.82` | |
| `dynamo-config@1.0.2` | `v1.82` | |
| `dynamo-config@1.0.1` | `v1.82` | |
| `dynamo-config@1.0.0` | `v1.82` | |
| `dynamo-config@0.9.1` | `v1.82` | |
| `dynamo-config@0.9.0` | `v1.82` | |
| `dynamo-config@0.8.1` | `v1.82` | |
| `dynamo-config@0.8.0` | `v1.82` | Initial |

#### dynamo-tokens (crate)

| Crate | MSRV (Rust) | Notes |
|-------|-------------|-------|
| `dynamo-tokens@1.1.1` | `v1.82` | |
| `dynamo-tokens@1.1.0` | `v1.82` | |
| `dynamo-tokens@1.0.2` | `v1.82` | |
| `dynamo-tokens@1.0.1` | `v1.82` | |
| `dynamo-tokens@1.0.0` | `v1.82` | |
| `dynamo-tokens@0.9.1` | `v1.82` | |
| `dynamo-tokens@0.9.0` | `v1.82` | Initial |

#### dynamo-mocker (crate)

| Crate | MSRV (Rust) | Notes |
|-------|-------------|-------|
| `dynamo-mocker@1.1.1` | `v1.82` | |
| `dynamo-mocker@1.1.0` | `v1.82` | |
| `dynamo-mocker@1.0.2` | `v1.82` | |
| `dynamo-mocker@1.0.1` | `v1.82` | |
| `dynamo-mocker@1.0.0` | `v1.82` | Initial |

#### dynamo-kv-router (crate)

| Crate | MSRV (Rust) | Notes |
|-------|-------------|-------|
| `dynamo-kv-router@1.1.1` | `v1.82` | |
| `dynamo-kv-router@1.1.0` | `v1.82` | |
| `dynamo-kv-router@1.0.2` | `v1.82` | |
| `dynamo-kv-router@1.0.1` | `v1.82` | |
| `dynamo-kv-router@1.0.0` | `v1.82` | Initial |

---

## Pre-Release Artifacts

> [!WARNING]
> **Pre-Release artifacts do not go through QA validation.** Pre-release versions are experimental previews intended for early testing and feedback. They may contain bugs, breaking changes, or incomplete features. Use stable releases for production workloads.

**Pre-Release Python Wheels** are published on the NVIDIA package index at [pypi.nvidia.com](https://pypi.nvidia.com/), not on the public [PyPI](https://pypi.org/) index. Like stable wheels, they are **Linux (manylinux) builds** for the Python versions in the [Support Matrix](support-matrix.md); `pip`/`uv` on macOS or Windows will not find matching wheels. Install on a supported Linux host or inside a Linux container.

Install by adding that URL as an extra index and allowing pre-releases (PEP 440 dev versions):

```bash
# uv (recommended in other Dynamo docs)
uv pip install --pre --extra-index-url https://pypi.nvidia.com/ ai-dynamo==1.1.0.dev2

# pip
pip install --pre --extra-index-url https://pypi.nvidia.com ai-dynamo==1.1.0.dev2
```

A GitHub or container tag `v1.1.0-dev.N` maps to a wheel version `1.1.0.devN` (for example `v1.1.0-dev.2` → `==1.1.0.dev2`). Optional extras such as `ai-dynamo[vllm]` use the same flags; pin the version you want from the sections below.

### v1.2.0-deepseek-v4-dev.2

- **Branch:** [release/1.2.0-deepseek-v4-dev.2](https://github.com/ai-dynamo/dynamo/tree/release/1.2.0-deepseek-v4-dev.2)
- **GitHub Tag:** [v1.2.0-deepseek-v4-dev.2](https://github.com/ai-dynamo/dynamo/releases/tag/v1.2.0-deepseek-v4-dev.2)
- **Backends:** vLLM `v0.20.0` (native DeepSeek-V4 support) | SGLang upstream `lmsysorg/sglang:deepseek-v4-blackwell` preview | NIXL `v0.10.1`
- **Coverage:** DeepSeek-V4-Flash and V4-Pro only. vLLM and SGLang containers are published for Blackwell. TensorRT-LLM container, other component containers, Helm charts, and wheels are not published for this tag. Snapshot dev build for early-access V4 model support; not QA-gated.

#### Container Images

| Image:Tag | Backend | CUDA | Arch |
|-----------|---------|------|------|
| `vllm-runtime:1.2.0-deepseek-v4-cuda13-dev.2` | vLLM `v0.20.0` | `v13.0` | AMD64/ARM64 |
| `sglang-runtime:1.2.0-deepseek-v4-cuda12-dev.2` | SGLang upstream DSv4 preview | `v12.9` | AMD64 |
| `sglang-runtime:1.2.0-deepseek-v4-cuda13-dev.2` | SGLang upstream DSv4 preview | `v13.0` | ARM64 |

#### Python Wheels

Not published for this dev release. Use the `v1.1.0` wheels or `v1.1.0-dev.3` from [pypi.nvidia.com](https://pypi.nvidia.com/).

#### Helm Charts

Not published for this dev release. Use `v1.1.0` charts for platform install.

#### Rust Crates

Not shipped for pre-release versions.

### v1.1.0-dev.3

- **Branch:** [release/1.1.0-dev.3](https://github.com/ai-dynamo/dynamo/tree/release/1.1.0-dev.3)
- **GitHub Tag:** [v1.1.0-dev.3](https://github.com/ai-dynamo/dynamo/releases/tag/v1.1.0-dev.3)
- **Backends (branch ToT):** SGLang `v0.5.10.post1` | TensorRT-LLM `v1.3.0rc11` | vLLM `v0.19.0` | NIXL `v0.10.1`
- **Coverage:** TensorRT-LLM runtime container plus **`ai-dynamo`** and **`ai-dynamo-runtime`** wheels on [pypi.nvidia.com](https://pypi.nvidia.com/). SGLang and vLLM containers, component containers (`dynamo-frontend`, `dynamo-planner`, `kubernetes-operator`, `snapshot-agent`), **`kvbm`** wheel, and Helm charts are not published for this tag.

#### Container Images

| Image:Tag | Backend | CUDA | Arch |
|-----------|---------|------|------|
| `tensorrtllm-runtime:1.1.0-dev.3` | TRT-LLM `v1.3.0rc11` | `v13.1` | AMD64/ARM64 |

#### Python Wheels

Available from [pypi.nvidia.com](https://pypi.nvidia.com/) (pre-release index):

```bash
uv pip install --pre --extra-index-url https://pypi.nvidia.com/ ai-dynamo==1.1.0.dev3
uv pip install --pre --extra-index-url https://pypi.nvidia.com/ ai-dynamo-runtime==1.1.0.dev3
```

`kvbm==1.1.0.dev3` is not yet published.

#### Helm Charts

Not published for this dev release. Use the latest stable (`v1.1.0`) for platform install.

#### Rust Crates

Not shipped for pre-release versions.

### v1.1.0-dev.2

- **Branch:** [release/1.1.0-dev.2](https://github.com/ai-dynamo/dynamo/tree/release/1.1.0-dev.2)
- **GitHub Tag:** [v1.1.0-dev.2](https://github.com/ai-dynamo/dynamo/releases/tag/v1.1.0-dev.2)
- **Backends (branch ToT):** SGLang `v0.5.9` | TensorRT-LLM `v1.3.0rc9` | vLLM `v0.19.0` | NIXL `v0.10.1`
- **Coverage:** SGLang and TensorRT-LLM runtime containers plus **`ai-dynamo`** and **`ai-dynamo-runtime`** wheels on [pypi.nvidia.com](https://pypi.nvidia.com/). vLLM runtime container, component containers (`dynamo-frontend`, `dynamo-planner`, `kubernetes-operator`, `snapshot-agent`), **`kvbm`** wheel, and Helm charts are not published for this tag.

#### Container Images

| Image:Tag | Backend | CUDA | Arch |
|-----------|---------|------|------|
| `sglang-runtime:1.1.0-dev.2` | SGLang `v0.5.9` | `v12.9` | AMD64/ARM64 |
| `tensorrtllm-runtime:1.1.0-dev.2` | TRT-LLM `v1.3.0rc9` | `v13.1` | AMD64/ARM64 |

#### Python Wheels

Available from [pypi.nvidia.com](https://pypi.nvidia.com/) (pre-release index):

```bash
uv pip install --pre --extra-index-url https://pypi.nvidia.com/ ai-dynamo==1.1.0.dev2
uv pip install --pre --extra-index-url https://pypi.nvidia.com/ ai-dynamo-runtime==1.1.0.dev2
```

#### Helm Charts

Not published for this dev release. Use the latest stable (`v1.1.0`) for platform install.

#### Rust Crates

Not shipped for pre-release versions.

### v1.1.0-dev.1

- **Branch:** [release/1.1.0-dev.1](https://github.com/ai-dynamo/dynamo/tree/release/1.1.0-dev.1)
- **GitHub Tag:** [v1.1.0-dev.1](https://github.com/ai-dynamo/dynamo/releases/tag/v1.1.0-dev.1)
- **Backends:** SGLang `v0.5.9` | TensorRT-LLM `v1.3.0rc5.post1` | vLLM `v0.17.1` | NIXL `v0.10.1`

#### Container Images

| Image:Tag | Backend | CUDA | Arch |
|-----------|---------|------|------|
| `vllm-runtime:1.1.0-dev.1` | vLLM `v0.17.1` | `v12.9` | AMD64/ARM64 |
| `vllm-runtime:1.1.0-dev.1-cuda13` | vLLM `v0.17.1` | `v13.0` | AMD64/ARM64 |
| `vllm-runtime:1.1.0-dev.1-efa-amd64` | vLLM `v0.17.1` | `v12.9` | AMD64 |
| `sglang-runtime:1.1.0-dev.1` | SGLang `v0.5.9` | `v12.9` | AMD64/ARM64 |
| `sglang-runtime:1.1.0-dev.1-cuda13` | SGLang `v0.5.9` | `v13.0` | AMD64/ARM64 |
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
```

#### Helm Charts

| Chart | NGC |
|-------|-----|
| `dynamo-platform-1.1.0-dev.1` | [NGC Helm: dynamo-platform 1.1.0-dev.1](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/helm-charts/dynamo-platform?version=1.1.0-dev.1) |
| `snapshot-1.1.0-dev.1` | [NGC Helm: snapshot 1.1.0-dev.1](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/helm-charts/snapshot?version=1.1.0-dev.1) |

#### Rust Crates

Not shipped for pre-release versions.
