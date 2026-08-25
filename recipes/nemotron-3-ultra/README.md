<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Nemotron-3-Ultra Recipes

This directory contains the optimized Dynamo 1.4.0 recipes for
`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`. The 12 profiles cover B200, GB200, and H200 at
the model-native 256K context length and the opt-in 1M context length, with aggregated and
disaggregated serving.

Use the [Nemotron-3-Ultra Fern recipe](../../docs/fern/pages/recipes/model-recipes/nemotron-3-ultra.mdx)
to select a target, prepare the model cache, deploy a manifest, run a smoke test, and review known
limitations.

Repository assets are organized as follows:

- `model-cache/`: persistent volume, download, and validation manifests
- `vllm/`: 12 optimized deployment manifests
- `perf/`: shared AIPerf assets

The deployment manifests are the source of truth for runtime images, worker topology, scheduling,
and transport configuration.
