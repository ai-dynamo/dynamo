<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# K-EXAONE 2.0

Serving recipes for LG AI Research's **K-EXAONE 2.0 750B-A37B** in NVFP4 on NVIDIA B200,
with vLLM via Dynamo. Aggregated (4 GPU) and disaggregated 1P1D (8 GPU) targets.

See the [Dynamo recipe documentation](https://docs.nvidia.com/dynamo/recipes/k-exaone-2)
for prerequisites, deployment, smoke tests, benchmarking, expected performance and
limitations.

| Target | GPUs | Manifest |
| --- | --- | --- |
| Aggregated, chat | 4x B200 | [`vllm/agg-b200-chat/deploy-generic.yaml`](vllm/agg-b200-chat/deploy-generic.yaml) |
| Disaggregated 1P1D, chat | 8x B200 | [`vllm/disagg-b200-chat/deploy-ib.yaml`](vllm/disagg-b200-chat/deploy-ib.yaml) |

Model cache and download job: [`model-cache/`](model-cache). Benchmark job: [`perf/`](perf).
