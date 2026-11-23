<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo Enterprise Certified Recipes

These recipes deploy a model with NVIDIA Dynamo using only
[Dynamo Enterprise Support](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/enterprise/overview.mdx)
images, pinned to a single Dynamo release. Each one is a copy of an open-source recipe in
[`recipes/`](https://github.com/ai-dynamo/dynamo/tree/main/recipes), adjusted to run on the
`-enterprise` images.

Commercial support covers the enterprise artifacts, not the recipe or configurations derived from
it. See
[Scope Boundaries](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/enterprise/scope-boundaries.mdx).

## Available Recipes

| Model | Framework | GPU | Topology | Dynamo release | Manifest |
|-------|-----------|-----|----------|----------------|----------|
| [Nemotron 3.5 Super VL](nemotron-3.5-super-vl-120b-a12b/README.md) | vLLM | 4x B200 | Aggregated, 4 workers | 1.7.0 | [`agg-b200-swebenchpro`](nemotron-3.5-super-vl-120b-a12b/vllm/agg-b200-swebenchpro/deploy.yaml) |
| [Nemotron 3.5 Super VL](nemotron-3.5-super-vl-120b-a12b/README.md) | vLLM | 5x B200 | Disaggregated, 3 prefill and 2 decode | 1.7.0 | [`disagg-b200-swebenchpro`](nemotron-3.5-super-vl-120b-a12b/vllm/disagg-b200-swebenchpro/deploy.yaml) |

## Prerequisites

Install the Dynamo platform from the enterprise Helm chart with the enterprise operator image, as
described in
[Install](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/enterprise/install.mdx).
Use the same release version as the recipe.
