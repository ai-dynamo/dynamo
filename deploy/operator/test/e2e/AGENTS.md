<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AGENTS.md

## Labels

| Label | Meaning |
| --- | --- |
| `gpu_0` | Runs without schedulable Kubernetes `nvidia.com/gpu` capacity. |
| `gpu_1`, `gpu_2`, `gpu_4`, `gpu_8` | Requires at least that many schedulable Kubernetes GPUs. |
| `h100` | Requires H100; combine with `gpu_N`. |
| `validation` | Webhook, CRD metadata, API discovery, or conversion checks only. |
| `rapid` | DGDR rapid profiling strategy. |
| `thorough` | DGDR thorough profiling strategy; usually combine with non-zero `gpu_N`. |

Do not use generic `gpu` or `real-gpu` labels.
Do not add `mocker` unless selecting explicit mocker/no-mocker variants.
Place labels at `Describe` or `Context` only when all contained specs match.
