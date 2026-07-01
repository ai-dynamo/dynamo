---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Release Artifacts
subtitle: Container images, Python wheels, Helm charts, and Rust crates for the current and pre-release versions
---

This document provides an inventory of Dynamo release artifacts including container images, Python wheels, Helm charts, and Rust crates.

> **See also:** [Compatibility](../getting-started/compatibility.md) for hardware, platform, and backend feature support.

This page lists artifacts for the current stable release and the active pre-release and experimental previews.

## Current Release: Dynamo v1.2.1

- **GitHub Release:** [v1.2.1](https://github.com/ai-dynamo/dynamo/releases/tag/v1.2.1)
- **Docs:** [v1.2.1](https://docs.nvidia.com/dynamo)
- **NGC Collection:** [ai-dynamo](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo)

> **Experimental:** [v1.2.0-deepseek-v4-dev.3](#v120-deepseek-v4-dev3) *(DeepSeek-V4-Flash / V4-Pro on Blackwell, vLLM + SGLang containers only)* is available as an experimental preview. Tagged **Pre-Releases** and experimental builds are listed under [Pre-Release Artifacts](#pre-release-artifacts).

### Container Images

| Image:Tag | Description | Backend | CUDA | Arch | NGC | Notes |
|-----------|-------------|---------|------|------|-----|-------|
| `vllm-runtime:1.2.1` | Runtime container for vLLM backend | vLLM `v0.20.1` | `v12.9` | AMD64/ARM64 | [NGC: vllm-runtime 1.2.1](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/vllm-runtime?version=1.2.1) | |
| `vllm-runtime:1.2.1-cuda13` | Runtime container for vLLM backend (CUDA 13) | vLLM `v0.20.1` | `v13.0` | AMD64/ARM64 | [NGC: vllm-runtime 1.2.1-cuda13](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/vllm-runtime?version=1.2.1-cuda13) | |
| `vllm-runtime:1.2.1-efa-amd64` | Runtime container for vLLM with AWS EFA | vLLM `v0.20.1` | `v12.9` | AMD64 | [NGC: vllm-runtime 1.2.1-efa-amd64](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/vllm-runtime?version=1.2.1-efa-amd64) | Experimental |
| `sglang-runtime:1.2.1` | Runtime container for SGLang backend | SGLang `v0.5.11` | `v12.9` | AMD64/ARM64 | [NGC: sglang-runtime 1.2.1](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/sglang-runtime?version=1.2.1) | |
| `sglang-runtime:1.2.1-cuda13` | Runtime container for SGLang backend (CUDA 13) | SGLang `v0.5.11` | `v13.0` | AMD64/ARM64 | [NGC: sglang-runtime 1.2.1-cuda13](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/sglang-runtime?version=1.2.1-cuda13) | |
| `tensorrtllm-runtime:1.2.1` | Runtime container for TensorRT-LLM backend | TRT-LLM `v1.3.0rc14` | `v13.1` | AMD64/ARM64 | [NGC: tensorrtllm-runtime 1.2.1](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/tensorrtllm-runtime?version=1.2.1) | |
| `tensorrtllm-runtime:1.2.1-efa-amd64` | Runtime container for TensorRT-LLM with AWS EFA | TRT-LLM `v1.3.0rc14` | `v13.1` | AMD64 | [NGC: tensorrtllm-runtime 1.2.1-efa-amd64](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/tensorrtllm-runtime?version=1.2.1-efa-amd64) | Experimental |
| `dynamo-frontend:1.2.1` | API gateway with Endpoint Prediction Protocol (EPP) | — | — | AMD64/ARM64 | [NGC: dynamo-frontend 1.2.1](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/dynamo-frontend?version=1.2.1) | |
| `dynamo-planner:1.2.1` | Standalone Planner image used by Profiler jobs and Planner pods | — | — | AMD64/ARM64 | [NGC: dynamo-planner 1.2.1](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/dynamo-planner?version=1.2.1) | |
| `kubernetes-operator:1.2.1` | Kubernetes operator for Dynamo deployments | — | — | AMD64/ARM64 | [NGC: kubernetes-operator 1.2.1](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/kubernetes-operator?version=1.2.1) | |
| `snapshot-agent:1.2.1` | Snapshot agent for fast GPU worker recovery via CRIU | — | — | AMD64/ARM64 | [NGC: snapshot-agent 1.2.1](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/snapshot-agent?version=1.2.1) | Preview |

### Python Wheels

We recommend using the TensorRT-LLM NGC container instead of the `ai-dynamo[trtllm]` wheel. See the [NGC container collection](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo) for supported images.

| Package | Description | Python | Platform | PyPI |
|---------|-------------|--------|----------|------|
| `ai-dynamo==1.2.1` | Main package with backend integrations (vLLM, SGLang, TRT-LLM) | `3.10`–`3.12` | Linux (glibc `v2.28+`) | [PyPI: ai-dynamo 1.2.1](https://pypi.org/project/ai-dynamo/1.2.1/) |
| `ai-dynamo-runtime==1.2.1` | Core Python bindings for Dynamo runtime | `3.10`–`3.12` | Linux (glibc `v2.28+`) | [PyPI: ai-dynamo-runtime 1.2.1](https://pypi.org/project/ai-dynamo-runtime/1.2.1/) |
| `kvbm==1.2.1` | KV Block Manager for disaggregated KV cache | `3.10`–`3.12` | Linux (glibc `v2.28+`) | [PyPI: kvbm 1.2.1](https://pypi.org/project/kvbm/1.2.1/) |

### Helm Charts

| Chart | Description | NGC |
|-------|-------------|-----|
| `dynamo-platform-1.2.1` | Platform services (etcd, NATS) and Dynamo Operator for Dynamo cluster | [NGC Helm: dynamo-platform-1.2.1](https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-1.2.1.tgz) |
| `snapshot-1.2.1` | Snapshot DaemonSet for fast GPU worker recovery | [NGC Helm: snapshot-1.2.1](https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/snapshot-1.2.1.tgz) |

> [!NOTE]
> The `dynamo-crds` Helm chart is deprecated as of v1.0.0; CRDs are now managed by the Dynamo Operator. The `dynamo-graph` Helm chart is deprecated as of v0.9.0.

### Rust Crates

| Crate | Description | MSRV (Rust) | crates.io |
|-------|-------------|-------------|-----------|
| `dynamo-runtime@1.2.1` | Core distributed runtime library | `v1.82` | [crates.io: dynamo-runtime 1.2.1](https://crates.io/crates/dynamo-runtime/1.2.1) |
| `dynamo-llm@1.2.1` | LLM inference engine | `v1.82` | [crates.io: dynamo-llm 1.2.1](https://crates.io/crates/dynamo-llm/1.2.1) |
| `dynamo-protocols@1.2.1` | Async OpenAI-compatible API client | `v1.82` | [crates.io: dynamo-protocols 1.2.1](https://crates.io/crates/dynamo-protocols/1.2.1) |
| `dynamo-async-openai@1.0.2` | Deprecated legacy OpenAI client; use **`dynamo-protocols`** | `v1.82` | [crates.io: dynamo-async-openai 1.0.2](https://crates.io/crates/dynamo-async-openai/1.0.2) |
| `dynamo-parsers@1.2.1` | Protocol parsers (SSE, JSON streaming) | `v1.82` | [crates.io: dynamo-parsers 1.2.1](https://crates.io/crates/dynamo-parsers/1.2.1) |
| `dynamo-memory@1.2.1` | Memory management utilities | `v1.82` | [crates.io: dynamo-memory 1.2.1](https://crates.io/crates/dynamo-memory/1.2.1) |
| `dynamo-config@1.2.1` | Configuration management | `v1.82` | [crates.io: dynamo-config 1.2.1](https://crates.io/crates/dynamo-config/1.2.1) |
| `dynamo-tokens@1.2.1` | Tokenizer bindings for LLM inference | `v1.82` | [crates.io: dynamo-tokens 1.2.1](https://crates.io/crates/dynamo-tokens/1.2.1) |
| `dynamo-tokenizers@1.2.1` | Tokenizer library for LLM inference | `v1.82` | [crates.io: dynamo-tokenizers 1.2.1](https://crates.io/crates/dynamo-tokenizers/1.2.1) |
| `dynamo-mocker@1.2.1` | Inference engine simulator for benchmarking | `v1.82` | [crates.io: dynamo-mocker 1.2.1](https://crates.io/crates/dynamo-mocker/1.2.1) |
| `dynamo-kv-router@1.2.1` | KV-aware request routing library | `v1.82` | [crates.io: dynamo-kv-router 1.2.1](https://crates.io/crates/dynamo-kv-router/1.2.1) |
| `kvbm-logical@1.2.1` | Logical layer for the KV Block Manager | `v1.82` | [crates.io: kvbm-logical 1.2.1](https://crates.io/crates/kvbm-logical/1.2.1) |

## Quick Install Commands

### Container Images (NGC)

> [!TIP]
> For detailed run instructions, see the backend-specific guides: [vLLM](../backends/vllm/README.md) | [SGLang](../backends/sglang/README.md) | [TensorRT-LLM](../backends/trtllm/README.md)

```bash
# Runtime containers
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1
docker pull nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.1
docker pull nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.2.1

# CUDA 13 variants
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1-cuda13
docker pull nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.1-cuda13

# EFA variants (AWS, AMD64 only, experimental)
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1-efa-amd64
docker pull nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.2.1-efa-amd64

# Infrastructure containers
docker pull nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.2.1
docker pull nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.2.1
docker pull nvcr.io/nvidia/ai-dynamo/kubernetes-operator:1.2.1
docker pull nvcr.io/nvidia/ai-dynamo/snapshot-agent:1.2.1
```

### Python Wheels (PyPI)

> [!TIP]
> For detailed installation instructions, see the [Quickstart](https://docs.nvidia.com/dynamo/getting-started/quickstart) in the docs.

```bash
# Install Dynamo with a specific backend (Recommended)
uv pip install "ai-dynamo[vllm]==1.2.1"
uv pip install --prerelease=allow "ai-dynamo[sglang]==1.2.1"
# TensorRT-LLM requires the NVIDIA PyPI index and pip
pip install --pre --extra-index-url https://pypi.nvidia.com "ai-dynamo[trtllm]==1.2.1"

# Install Dynamo core only
uv pip install ai-dynamo==1.2.1

# Install standalone KVBM
uv pip install kvbm==1.2.1
```

### Helm Charts (NGC)

> [!TIP]
> For Kubernetes deployment instructions, see the [Kubernetes Installation Guide](../kubernetes/installation-guide.md).

```bash
helm install dynamo-platform oci://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform --version 1.2.1
helm install snapshot oci://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/snapshot --version 1.2.1
```

### Rust Crates (crates.io)

> [!TIP]
> For API documentation, see each crate on [docs.rs](https://docs.rs/). To build Dynamo from source, see [Building from Source](https://github.com/ai-dynamo/dynamo#building-from-source).

```bash
cargo add dynamo-runtime@1.2.1
cargo add dynamo-llm@1.2.1
cargo add dynamo-protocols@1.2.1
# Deprecated legacy crate name — pin only if a dependency requires it; new code should use dynamo-protocols:
# cargo add dynamo-async-openai@1.0.2
cargo add dynamo-parsers@1.2.1
cargo add dynamo-memory@1.2.1
cargo add dynamo-config@1.2.1
cargo add dynamo-tokens@1.2.1
cargo add dynamo-tokenizers@1.2.1
cargo add dynamo-mocker@1.2.1
cargo add dynamo-kv-router@1.2.1
cargo add kvbm-logical@1.2.1
```

**CUDA and Driver Requirements:** For detailed CUDA toolkit versions and minimum driver requirements for each container image, see [Compatibility](../getting-started/compatibility.md#cuda--driver-requirements).

## Known Issues

For known issues, refer to the release notes:

- [v1.2.1 Release Notes](https://github.com/ai-dynamo/dynamo/releases/tag/v1.2.1)

---

## Release Artifact History

Each bullet is a **delta** to what ships on NGC / Helm / PyPI / crates.io: net-new crates, removed Helm charts, or image lines that **split** or **appear** on the registry. See the inventory tables above for full current matrices, and the [GitHub Releases](#github-releases) table below for release links.

**Stable Releases**

- **v1.2.1**: Patch release. Same backend versions as v1.2.0: SGLang `v0.5.11` (NIXL `v1.0.1`), TRT-LLM `v1.3.0rc14` (NIXL `v0.10.1`), vLLM `v0.20.1` (NIXL `v0.10.1`).

**Dynamo Nightlies**

- **`ai-dynamo`** and **`ai-dynamo-runtime`** nightly builds from **`main`** publish wheels tagged **`*.devYYYYMMDD`**. Install with **`pip`** or **`uv`** using **`--pre`** and the same NVIDIA extra-index pattern as [Pre-Release Artifacts](#pre-release-artifacts). **`*.devYYYYMMDD`** versioning for nightly **`main`** wheels began **Apr 24, 2026**.

**Pre-Release and Experimental Git Tags**

- **v1.3.0-dev.1**: **Images:** full runtime matrix -- `vllm-runtime` (cuda12/cuda13/efa), `tensorrtllm-runtime` (cuda13/efa), `sglang-runtime` (cuda12/cuda13/efa), plus `dynamo-frontend`, `dynamo-planner`, `kubernetes-operator`, `snapshot-agent`. **Wheels:** `ai-dynamo`, `ai-dynamo-runtime`, `kvbm` on [pypi.nvidia.com](https://pypi.nvidia.com/). **Crates:** on [crates.io](https://crates.io/) at `1.3.0-dev.1`. **Helm:** `dynamo-platform`, `snapshot` at `1.3.0-dev.1` (see [below](#v130-dev1)).
- **v1.2.0-deepseek-v4-dev.3**: **Images:** `vllm-runtime:*-deepseek-v4-cuda13-dev.3`, `sglang-runtime:*-deepseek-v4-cuda12-dev.3`, `sglang-runtime:*-deepseek-v4-cuda13-dev.3`. **Helm / PyPI:** Not published for this tag (see [Pre-Release Artifacts](#v120-deepseek-v4-dev3)).

### GitHub Releases

| Version | Release Date | GitHub | Docs | Notes |
|---------|--------------|--------|------|-------|
| `v1.2.1` | TBD | [Release](https://github.com/ai-dynamo/dynamo/releases/tag/v1.2.1) | [Docs](https://docs.nvidia.com/dynamo) | |
| `v1.2.0-deepseek-v4-dev.3` | May 9, 2026 | [Tag](https://github.com/ai-dynamo/dynamo/releases/tag/v1.2.0-deepseek-v4-dev.3) | — | Experimental (DeepSeek-V4-Flash / V4-Pro Blackwell preview; vLLM + SGLang containers only) |

---

## Pre-Release Artifacts

<Warning>
**Pre-Release artifacts do not go through QA validation.** Pre-release versions are experimental previews intended for early testing and feedback. They may contain bugs, breaking changes, or incomplete features. Use stable releases for production workloads.
</Warning>

**Pre-Release Python Wheels** are published on the NVIDIA package index at [pypi.nvidia.com](https://pypi.nvidia.com/), not on the public [PyPI](https://pypi.org/) index. Like stable wheels, they are **Linux (manylinux) builds** for the Python versions in [Compatibility](../getting-started/compatibility.md); `pip`/`uv` on macOS or Windows will not find matching wheels. Install on a supported Linux host or inside a Linux container.

Install by adding that URL as an extra index and allowing pre-releases (PEP 440 dev versions):

```bash
# uv (recommended in other Dynamo docs)
uv pip install --pre --extra-index-url https://pypi.nvidia.com/ ai-dynamo==1.3.0.dev1

# pip
pip install --pre --extra-index-url https://pypi.nvidia.com ai-dynamo==1.3.0.dev1
```

A GitHub or container tag `v1.3.0-dev.N` maps to a wheel version `1.3.0.devN` (for example `v1.3.0-dev.1` → `==1.3.0.dev1`). Optional extras such as `ai-dynamo[vllm]` use the same flags; pin the version you want from the sections below.

### v1.3.0-dev.1

- **Branch:** [release/1.3.0-dev.1](https://github.com/ai-dynamo/dynamo/tree/release/1.3.0-dev.1)
- **GitHub Tag:** `v1.3.0-dev.1` *(tag publication pending)*
- **Backends:** SGLang `0.5.12.post1` | TensorRT-LLM `1.3.0rc17` | vLLM `0.22.0` | NIXL `1.1.0` (vLLM); `1.0.1` (SGLang); `0.10.1` (TRT-LLM)
- **Coverage:** Full-platform preview of v1.3.0 -- all runtime containers (vLLM and SGLang on CUDA 12 + 13 + EFA, TensorRT-LLM on CUDA 13 + EFA) and component containers, plus `ai-dynamo` / `ai-dynamo-runtime` / `kvbm` wheels, Rust crates, and the `dynamo-platform` and `snapshot` Helm charts. Cut from `main` after the TensorRT-LLM `v1.3.0rc17` upgrade; experimental snapshot, not QA-gated.

#### Container Images

| Image:Tag | Backend | CUDA | Arch |
|-----------|---------|------|------|
| `vllm-runtime:1.3.0-dev.1-cuda13` | vLLM `v0.22.0` | `v13.0` | AMD64/ARM64 |
| `vllm-runtime:1.3.0-dev.1-cuda12` | vLLM `v0.22.0` | `v12.9` | AMD64/ARM64 |
| `vllm-runtime:1.3.0-dev.1-efa` | vLLM `v0.22.0` (AWS EFA) | `v13.0` | AMD64/ARM64 |
| `tensorrtllm-runtime:1.3.0-dev.1-cuda13` | TensorRT-LLM `v1.3.0rc17` | `v13.1` | AMD64/ARM64 |
| `tensorrtllm-runtime:1.3.0-dev.1-efa` | TensorRT-LLM `v1.3.0rc17` (AWS EFA) | `v13.1` | AMD64/ARM64 |
| `sglang-runtime:1.3.0-dev.1-cuda13` | SGLang `v0.5.12.post1` | `v13.0` | AMD64/ARM64 |
| `sglang-runtime:1.3.0-dev.1-cuda12` | SGLang `v0.5.12.post1` | `v12.9` | AMD64/ARM64 |
| `sglang-runtime:1.3.0-dev.1-efa` | SGLang `v0.5.12.post1` (AWS EFA) | `v13.0` | AMD64/ARM64 |
| `dynamo-frontend:1.3.0-dev.1` | -- | -- | AMD64/ARM64 |
| `dynamo-planner:1.3.0-dev.1` | -- | -- | AMD64/ARM64 |
| `kubernetes-operator:1.3.0-dev.1` | -- | -- | AMD64/ARM64 |
| `snapshot-agent:1.3.0-dev.1` | -- | -- | AMD64 |

#### Python Wheels

`ai-dynamo`, `ai-dynamo-runtime`, and `kvbm` at `1.3.0.dev1` on [pypi.nvidia.com](https://pypi.nvidia.com/) (prerelease index, not public PyPI):

```bash
pip install --pre --extra-index-url https://pypi.nvidia.com ai-dynamo==1.3.0.dev1
```

#### Helm Charts

`dynamo-platform` and `snapshot` at `1.3.0-dev.1`.

#### Rust Crates

Published to [crates.io](https://crates.io/) at `1.3.0-dev.1` (`dynamo-runtime`, `dynamo-llm`, and the dependent workspace crates).

### v1.2.0-deepseek-v4-dev.3

- **Branch:** [release/1.2.0-deepseek-v4-dev.3](https://github.com/ai-dynamo/dynamo/tree/release/1.2.0-deepseek-v4-dev.3)
- **GitHub Tag:** [v1.2.0-deepseek-v4-dev.3](https://github.com/ai-dynamo/dynamo/releases/tag/v1.2.0-deepseek-v4-dev.3)
- **Backends:** vLLM `v0.20.1` (DSv4 stabilization patch over `v0.20.0` native DSv4 support) | SGLang upstream `lmsysorg/sglang:deepseek-v4-blackwell` preview (refreshed for dev.3) | NIXL `v0.10.1`
- **Coverage:** Partial -- DeepSeek-V4-Flash and V4-Pro only. vLLM and SGLang containers are published for Blackwell (B200 plus GB200); no TensorRT-LLM container, no other component containers, no Helm charts, no wheels. Snapshot dev build for early-access V4 model support; not QA-gated.

#### Container Images

| Image:Tag | Backend | CUDA | Arch |
|-----------|---------|------|------|
| `vllm-runtime:1.2.0-deepseek-v4-cuda13-dev.3` | vLLM `v0.20.1` | `v13.0` | AMD64/ARM64 |
| `sglang-runtime:1.2.0-deepseek-v4-cuda12-dev.3` | SGLang upstream DSv4 preview | `v12.9` | AMD64 |
| `sglang-runtime:1.2.0-deepseek-v4-cuda13-dev.3` | SGLang upstream DSv4 preview | `v13.0` | ARM64 |

#### Python Wheels

Not published for this dev release. Use the `v1.2.1` stable wheels.

#### Helm Charts

Not published for this dev release. Use the `v1.2.1` charts for platform install.

#### Rust Crates

Not shipped for pre-release versions.
