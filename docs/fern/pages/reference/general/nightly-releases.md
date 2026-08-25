---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Nightly Releases
subtitle: Nightly container images, Python wheels, install patterns, and current backend versions.
---

import { ReferenceStyles } from "@/components/ReferenceStyles";
import { NightlyBuilds } from "@/components/NightlyBuilds";

<ReferenceStyles />

Dynamo publishes nightly builds from `main`. Nightlies let you try the latest features and backend upgrades before they land in a stable release. This page covers what nightly publishes, how to install it, and which backend versions the current and recent nightlies ship.

<Warning>
**Nightly builds are experimental and are not QA-validated.** They are built from the tip of `main` and may contain bugs, breaking changes, or incomplete features. Use [stable releases](release-artifacts.mdx) for production workloads.
</Warning>

## Recent Nightlies

<NightlyBuilds />

## What Gets Published

Every night, the [Nightly CI pipeline](https://github.com/ai-dynamo/dynamo/blob/main/.github/workflows/nightly-ci.yml) builds `main` and publishes:

- **Runtime images (CUDA 13):** `vllm-runtime-nightly`, `sglang-runtime-nightly`, and `tensorrtllm-runtime-nightly` to NGC.
- **Component images:** `dynamo-frontend-nightly`, `dynamo-planner-nightly`, `kubernetes-operator-nightly`, and `snapshot-agent-nightly` to NGC.
- **Python wheels:** `ai-dynamo`, `ai-dynamo-runtime`, and `kvbm` to the NVIDIA prerelease index at [pypi.nvidia.com](https://pypi.nvidia.com/).

Each nightly image repository is separate from its stable counterpart: the nightly planner is `dynamo-planner-nightly`, not a `nightly` tag on `dynamo-planner`.

Nightly deliberately does **not** publish EFA image variants, Helm charts, or Rust crates. For those, use a [stable or prerelease build](release-artifacts.mdx).

## Installing Nightly Containers

Nightly images live in their own `-nightly` NGC repositories so they cannot be pulled accidentally in place of a stable image. Runtime containers use a floating `:latest` tag for the most recent nightly build.

```bash
# Always the latest nightly
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:latest
docker pull nvcr.io/nvidia/ai-dynamo/sglang-runtime-nightly:latest
docker pull nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime-nightly:latest
```

Every nightly also gets an immutable `:YYYYMMDD-<shortsha>` tag naming the commit it was built from. Use those to pin a specific night; the build selector in [Install Dynamo](../../cli/installation/install-dynamo.mdx) emits them for each backend version.

## Installing Nightly Wheels

Nightly wheels are published to the NVIDIA prerelease index at [pypi.nvidia.com](https://pypi.nvidia.com/), not the public PyPI. They are Linux manylinux builds for the Python versions in [Compatibility](compatibility.mdx); install on a supported Linux host or inside a Linux container. Nightly versions follow PEP 440 dev versioning, `X.Y.Z.devYYYYMMDD`.

```bash
# Latest nightly (uv)
uv pip install --pre --extra-index-url https://pypi.nvidia.com/ ai-dynamo

# Latest nightly (pip)
pip install --pre --extra-index-url https://pypi.nvidia.com/ ai-dynamo

# Pin a specific nightly wheel
uv pip install --pre --extra-index-url https://pypi.nvidia.com/ "ai-dynamo[vllm]==1.5.0.dev20260825"
```

Backend extras such as `ai-dynamo[vllm]` and `ai-dynamo[sglang]` use the same flags. For TensorRT-LLM, use the nightly container rather than a PyPI extra.

## Backend Versions

Nightlies track `main`, so the backend versions they ship change as `main` advances. The build selector in [Install Dynamo](../../cli/installation/install-dynamo.mdx) lists the last three versions of each backend and the newest nightly that shipped each one, with the exact install command. For Kubernetes image variables, use the selector in the [Kubernetes Quickstart](../../kubernetes/getting-started/quickstart.mdx#install-dynamo).

To confirm the exact versions a specific nightly shipped, read them from the pulled image:

```bash
docker run --rm nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:latest pip show vllm
```

## See Also

- [Release Artifacts](release-artifacts.mdx) — stable and pre-release artifact inventory
- [Compatibility](compatibility.mdx) — hardware, platform, CUDA, and driver support
- [Model Early Access Builds](model-early-access-builds.mdx) — model-specific pre-release container builds
