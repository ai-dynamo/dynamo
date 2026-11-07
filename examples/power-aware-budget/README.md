<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Power-Aware Budget Scaling

This directory holds a contract fragment that shows how to author per-GPU power caps and a
deployment-wide power budget so the Planner keeps its projected power draw within the budget when
it scales replicas. The manifests are not a fully deployable DGD.

For the ownership model, setup steps, projection formula, and limitations, see
[Power-Aware Budget Scaling](../../docs/fern/pages/developer-guide/knowledge-base/modular-components/planner/planner-examples.md#power-aware-budget-scaling).

## Contents

| Path | Description |
|---|---|
| `dgd.yaml` | Power-relevant DGD fields: worker `dynamo.nvidia.com/gpu-power-limit` annotations and GPU limits |
| `planner_config.json` | Planner config that enables power awareness with `total_gpu_power_limit` |
