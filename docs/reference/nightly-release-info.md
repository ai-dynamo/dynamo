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

Nightlies track `main`, so the backend versions they ship change as `main` advances. To find which nightly — or stable — build ships a given backend version, and get the exact pull/install command, see the **[Quickstart](../getting-started/quickstart.mdx#get-dynamo)**.

To confirm the exact versions a specific nightly shipped, read them from the pulled image:

```bash
docker run --rm nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:latest pip show vllm
```

## See Also

- [Release Artifacts](release-artifacts.md) — stable and pre-release artifact inventory
- [Support Matrix](support-matrix.md) — hardware, platform, CUDA, and driver support
- [Feature Matrix](feature-matrix.md) — backend feature support
