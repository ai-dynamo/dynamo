<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Nemotron-3-Ultra Recipes

This directory contains Dynamo recipes for `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`.
Existing Day-0 profiles remain available with explicit `256k` identifiers. The Dynamo 1.4.0
Refresh profiles cover B200, GB200, and H200 at 256K and 1M context lengths with aggregated and
disaggregated serving.

Use the [Nemotron-3-Ultra Fern recipe](../../docs/fern/pages/recipes/model-recipes/nemotron-3-ultra.mdx)
to select a target, prepare the model cache, deploy a manifest, run a smoke test, and review known
limitations.

Repository assets are organized as follows:

- `model-cache/`: persistent volume, download, and validation manifests
- `vllm/`: Day-0 profiles and 12 Refresh deployment manifests
- `perf/`: shared AIPerf assets for Day-0 and Refresh profiles

The deployment manifests are the source of truth for runtime images, worker topology, scheduling,
and transport configuration.
