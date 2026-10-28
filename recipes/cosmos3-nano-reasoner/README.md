<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cosmos3 Nano Reasoner

This experimental recipe serves Cosmos3 Nano Reasoner with the released
`nvcr.io/nim/nvidia/cosmos3:2.1.0` NIM image, pinned by digest, and NVIDIA
Dynamo. It compares round-robin routing with per-video session affinity.

Use the [Cosmos3 Nano Reasoner Fern recipe](../../docs/fern/pages/recipes/model-recipes/cosmos3-nano-reasoner.mdx)
for prerequisites, deployment instructions, the smoke test, benchmark commands,
and measured performance. See [`perf/README.md`](perf/README.md) for benchmark
details and the full results table.

| Configuration | GPUs | Precision | Routing |
| --- | --- | --- | --- |
| `vllm/agg-h200-round-robin` | 8x H200 | FP8 | Round robin, no affinity |
| `vllm/agg-h200-session-affinity` | 8x H200 | FP8 | Round robin with 300-second session affinity |

| Path | Purpose |
| --- | --- |
| `model-cache/` | Shared model-cache PVC and model-download Job |
| `vllm/agg-h200-*/deploy.yaml` | `DynamoGraphDeployment` for each routing profile |
| `perf/profiles/` | Routing-specific benchmark settings |
| `perf/smoke.yaml` | One-video end-to-end smoke test |
| `perf/perf.yaml` | AIPerf concurrency sweep |
