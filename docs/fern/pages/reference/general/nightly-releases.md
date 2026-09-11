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

- **Runtime container images (CUDA 13):** `vllm-runtime-nightly`, `sglang-runtime-nightly`, and `tensorrtllm-runtime-nightly` to NGC, each with an Elastic Fabric Adapter (EFA) variant under a `-efa` tag suffix.
- **Component container images:** `kubernetes-operator-nightly`, `dynamo-planner-nightly`, and `dynamo-frontend-nightly` to NGC.
- **Python wheels:** `ai-dynamo`, `ai-dynamo-runtime`, and `kvbm` to the NVIDIA prerelease index at [pypi.nvidia.com](https://pypi.nvidia.com/).
- **Helm chart:** [`dynamo-platform-nightly`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/helm-charts/dynamo-platform-nightly) to the NGC Helm registry, kept separate from the stable `dynamo-platform` chart.

The runtime images and the wheels gate the release: if any of them fails to build, that night publishes nothing. The component images, the EFA variants, and the Helm chart stage fail-soft, so a flake in one of them skips that artifact for the night without holding back the rest.

Nightly does not publish Rust crates — for those, use a [stable or pre-release build](release-artifacts.mdx).

## Installing Nightly Containers

Nightly images live in their own `-nightly` NGC repositories so they cannot be pulled accidentally in place of a stable image. Every nightly build pushes an immutable `YYYYMMDD-<shortsha>` tag and moves the `latest` tag to that build.

```bash
# Most recent nightly
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:latest
docker pull nvcr.io/nvidia/ai-dynamo/sglang-runtime-nightly:latest
docker pull nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime-nightly:latest

# Pin one nightly build
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:20260830-4c7e981

# EFA variant, floating or pinned
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:latest-efa
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:20260830-4c7e981-efa
```

Pin the dated tag for anything you need to reproduce later: `latest` moves every night, so the image behind it changes underneath you. The runtime repositories also carry a `nightly` tag, but it stopped tracking `main` in July 2026 — use `latest` or a dated tag instead. The component repositories publish the same dated tags and a `latest` float, without the `nightly` alias or the EFA variants.

## Installing Nightly Wheels

Nightly wheels are published to the NVIDIA prerelease index at [pypi.nvidia.com](https://pypi.nvidia.com/), not the public PyPI. They are Linux manylinux builds for the Python versions in [Compatibility](compatibility.mdx); install on a supported Linux host or inside a Linux container. Nightly versions follow PEP 440 dev versioning, `X.Y.Z.devYYYYMMDD`.

```bash
# Latest nightly (uv)
uv pip install --pre --extra-index-url https://pypi.nvidia.com/ ai-dynamo

# Latest nightly (pip)
pip install --pre --extra-index-url https://pypi.nvidia.com/ ai-dynamo

# Pin a specific nightly wheel
uv pip install --pre --extra-index-url https://pypi.nvidia.com/ "ai-dynamo[vllm]==1.5.0.dev20260831"
```

Backend extras such as `ai-dynamo[vllm]` and `ai-dynamo[sglang]` use the same flags. For TensorRT-LLM, use the nightly container rather than a PyPI extra.

## Backend Versions

Nightlies track `main`, so the backend versions they ship change as `main` advances. To find which nightly or stable build ships a given backend version, and get the exact pull or install command, use the build selector in the [Kubernetes Quickstart](../../kubernetes/getting-started/quickstart.mdx#install-dynamo).

To confirm the exact versions a specific nightly shipped, read them from the pulled image:

```bash
docker run --rm nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:latest pip show vllm
```

## See Also

- [Release Artifacts](release-artifacts.mdx) — stable and pre-release artifact inventory
- [Compatibility](compatibility.mdx) — hardware, platform, CUDA, and driver support
- [Model Early Access Builds](model-early-access-builds.mdx) — model-specific pre-release container builds
