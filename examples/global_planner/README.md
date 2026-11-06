<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Global Planner Examples

This directory holds `DynamoGraphDeployment` manifests that demonstrate GlobalPlanner, the
centralized scaling layer that enforces shared scaling policy across multiple DGDs. The manifests
use `envsubst` variables such as `K8S_NAMESPACE`, `DYNAMO_IMAGE`, and `STORAGE_CLASS_NAME`.

For deployment patterns, prerequisites, configuration, and validation, see the
[Global Planner Guide](../../docs/fern/pages/developer-guide/knowledge-base/modular-components/planner/global-planner-guide.md)
and the [Global Planner example](../../docs/fern/pages/recipes/examples/global-planner.mdx).

## Contents

| Path | Pattern | Backend |
|---|---|---|
| `global-planner-gpu-budget.yaml` | Multiple models with a shared GPU budget: two model DGDs and one control DGD with `max_total_gpus` | vLLM |
| `global-planner-vllm-test.yaml` | One endpoint, multiple pools: Frontend, GlobalRouter, and GlobalPlanner with two prefill pools (TP1, TP2) and one decode pool | vLLM |
| `global-planner-mocker-test.yaml` | Same topology with Mocker workers; GlobalPlanner in `no_operation` mode | Mocker |
| `global-planner-vllm-test-xpu-dra.yaml` | Intel XPU/DRA variant with two TP1 prefill pools and one TP1 decode pool | vLLM (Intel XPU) |
