<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Motif-3 NVFP4

Experimental NVIDIA B200 recipes using vLLM, TP2, expert parallelism, and MTP2.
See the [Fern recipe documentation](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/recipes/model-recipes/motif-3.mdx)
for prerequisites, deployment, configuration, and benchmarking.

| Target | GPUs | Manifest |
|---|---:|---|
| Aggregated | 2 | [Generic](vllm/agg-b200-chat/base/deploy.yaml) · [Nscale Kustomization](vllm/agg-b200-chat/kustomize) |
| 3P1D, KV-aware routing | 8 | [Generic](vllm/disagg-b200-chat/deploy-generic-3p1d-kv.yaml) · [Nscale](vllm/disagg-b200-chat/deploy-nscale-3p1d-kv.yaml) |
