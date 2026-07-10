---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Nightly Release Info
subtitle: Nightly container images, Python wheels, install patterns, and current backend versions
---

Dynamo publishes nightly builds from `main` every day. Nightlies let you try the latest features and backend upgrades before they land in a stable release. This page covers what nightly publishes, how to install it, and which backend versions the current and past nightlies ship.

> [!WARNING]
> **Nightly builds are experimental and are not QA-validated.** They are built from the tip of `main` and may contain bugs, breaking changes, or incomplete features. Use [stable releases](release-artifacts.md) for production workloads.

## What Gets Published

Every night (around 08:00 UTC) the [Nightly CI pipeline](https://github.com/ai-dynamo/dynamo/blob/main/.github/workflows/nightly-ci.yml) builds `main` at a single commit and publishes:

- **Container images (CUDA 13):** `vllm-runtime-nightly`, `sglang-runtime-nightly`, and `tensorrtllm-runtime-nightly` to NGC.
- **Python wheels:** `ai-dynamo`, `ai-dynamo-runtime`, and `kvbm` to the NVIDIA prerelease index at [pypi.nvidia.com](https://pypi.nvidia.com/).

Nightly deliberately does **not** publish the EFA image variants, `dynamo-frontend`, `kubernetes-operator`, `dynamo-planner`, `snapshot-agent`, Helm charts, or Rust crates. For those, use a [stable or pre-release build](release-artifacts.md).

## Installing Nightly Containers

Nightly images live in their own `-nightly` NGC repositories so they cannot be pulled accidentally in place of a stable image. Each nightly is published with:

- a floating `:latest` tag — always the most recent nightly;
- an immutable `:YYYYMMDD-<shortsha>` tag — a specific night's build;
- a `-cuda13` alias — pins CUDA 13 explicitly.

```bash
# Always the latest nightly
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:latest
docker pull nvcr.io/nvidia/ai-dynamo/sglang-runtime-nightly:latest
docker pull nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime-nightly:latest

# Pin a specific nightly
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:20260710-abc1234
```

## Installing Nightly Wheels

Nightly wheels are published to the NVIDIA prerelease index at [pypi.nvidia.com](https://pypi.nvidia.com/), not the public PyPI. They are Linux (manylinux) builds for the Python versions in the [Support Matrix](support-matrix.md); install on a supported Linux host or inside a Linux container. Nightly versions follow PEP 440 dev versioning, `X.Y.Z.devYYYYMMDD`.

```bash
# Latest nightly (uv — recommended)
uv pip install --pre --extra-index-url https://pypi.nvidia.com/ ai-dynamo

# Latest nightly (pip)
pip install --pre --extra-index-url https://pypi.nvidia.com/ ai-dynamo

# Pin a specific nightly
pip install --pre --extra-index-url https://pypi.nvidia.com/ ai-dynamo==1.3.0.dev20260710
```

Backend extras such as `ai-dynamo[vllm]` use the same flags. For TensorRT-LLM, use the nightly container rather than a PyPI extra.

## Backend Versions

Nightlies track `main`, so backend versions move as `main` advances. The table below is generated from the build's source of truth, [`container/context.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/container/context.yaml):

{/* BEGIN:backend-versions (auto-generated from container/context.yaml — do not edit by hand) */}

| Backend | Version | `context.yaml` runtime image tag |
|---------|---------|----------------------------------|
| vLLM | `v0.24.0` | `v0.24.0-ubuntu2404` |
| SGLang | `v0.5.14` | `v0.5.14-cu130-runtime` |
| TensorRT-LLM | `1.3.0rc20` | `1.3.0rc20` |

{/* END:backend-versions */}

To confirm the exact versions a specific nightly shipped, read them from the pulled image:

```bash
docker run --rm nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:latest pip show vllm
```

## Backend Version History

Each nightly ships whatever backend versions `main` pinned that day. To run a specific backend version, pick a nightly dated within its window below. This history is reconstructed automatically from the git history of `container/context.yaml`.

{/* BEGIN:backend-history (auto-generated from git history of container/context.yaml — do not edit by hand) */}

### vLLM

| Version | In nightlies |
|---------|--------------|
| `v0.24.0` | 2026-06-30 → present |
| `v0.23.0` | 2026-06-16 → 2026-06-30 |
| `v0.22.1` | 2026-06-05 → 2026-06-16 |
| `v0.22.0` | 2026-06-02 → 2026-06-05 |
| `v0.21.0` | 2026-05-19 → 2026-06-02 |
| `v0.20.1` | 2026-05-15 → 2026-05-19 |

### SGLang

| Version | In nightlies |
|---------|--------------|
| `v0.5.14` | 2026-06-29 → present |
| `v0.5.13.post1` | 2026-06-24 → 2026-06-29 |
| `v0.5.12.post1` | 2026-05-25 → 2026-06-24 |
| `v0.5.11` | 2026-05-11 → 2026-05-25 |
| `v0.5.10.post1` | ≤2026-04-24 → 2026-05-11 |

### TensorRT-LLM

| Version | In nightlies |
|---------|--------------|
| `1.3.0rc20` | 2026-07-08 → present |
| `1.3.0rc19` | 2026-06-29 → 2026-07-08 |
| `1.3.0rc18` | 2026-06-11 → 2026-06-29 |
| `1.3.0rc17` | 2026-06-03 → 2026-06-11 |
| `1.3.0rc16` | 2026-05-28 → 2026-06-03 |
| `1.3.0rc14` | 2026-05-21 → 2026-05-28 |

{/* END:backend-history */}

To install a version from the table above:

- **Wheels** — pin any date within the range: `ai-dynamo==X.Y.Z.devYYYYMMDD` (see [Installing Nightly Wheels](#installing-nightly-wheels)).
- **Containers** — pull `…-runtime-nightly:YYYYMMDD-<sha>` for a date in the range. Find the `<sha>` for a given date from the [NGC catalog](https://catalog.ngc.nvidia.com/) tags — it is `main` HEAD at that night's build, so it cannot be derived from `container/context.yaml`.

## See Also

- [Release Artifacts](release-artifacts.md) — stable and pre-release artifact inventory
- [Support Matrix](support-matrix.md) — hardware, platform, CUDA, and driver support
- [Feature Matrix](feature-matrix.md) — backend feature support
