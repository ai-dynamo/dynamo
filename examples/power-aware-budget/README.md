<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DGD-owned power caps and budget-aware scaling

This example shows how to author per-GPU power caps and a deployment-wide power
budget so the Planner keeps its **projected** power draw within a configured
rack/DGD budget when it scales replicas.

> [!NOTE]
> These manifests are a **contract fragment**, not a fully deployable DGD: they
> show only the power-relevant fields (annotations, GPU limits, Planner config).
> A runnable deployment also needs container images, model args, env/secrets,
> and a Frontend — see [examples/backends/vllm/deploy/](../backends/vllm/deploy/)
> for a complete DGD to merge these fields into.

> [!IMPORTANT]
> The budget is a **projected** ceiling over the *requested* per-GPU caps, not a
> proven hardware limit. The Power Agent applies each requested cap but may
> clamp it up to the GPU's hardware minimum, and a cap write can fail; neither
> is fed back to the Planner today. So the Planner prevents an admitted
> scale-up from pushing `projected_watts` over `total_gpu_power_limit` for the
> requested caps — it does **not** automatically remediate a deployment that is
> already over budget (it holds scale-ups and allows scale-downs), and it does
> not guarantee the *effective* hardware draw stays under the budget.
> Effective-cap and enforcement-health feedback are deferred to the
> dynamic-control design.

## Ownership model

| Value | Source of truth | Consumer |
| --- | --- | --- |
| Per-GPU cap (watts, static) | Worker component `podTemplate` annotation `dynamo.nvidia.com/gpu-power-limit` | Operator → Pods; Power Agent (applies NVML/DCGM cap); Planner (reads) |
| Total DGD power budget (watts) | PlannerConfig `total_gpu_power_limit` | Planner |
| Replica targets | Planner (after the budget clamp) | Existing scaling adapter |
| Applied GPU hardware cap | Power Agent | NVML / DCGM |

The Planner **reads** the caps from the DGD; it does **not** patch Pods. New
replicas created by a scale-up are born with the annotation because the operator
renders it into the Deployment / LeaderWorkerSet template.

## Author the caps

1. Put the per-GPU cap on each worker component's `podTemplate.metadata.annotations`
   (see [dgd.yaml](dgd.yaml)). Prefill and decode may differ.
2. Set the total budget on the Planner's mounted config (see
   [planner_config.json](planner_config.json)):

   ```json
   {
     "enable_power_awareness": true,
     "total_gpu_power_limit": 5200
   }
   ```

   `enable_power_awareness` requires `environment: "kubernetes"` and a
   `total_gpu_power_limit > 0`. Per-GPU caps are **not** set here — they live on
   the worker `podTemplate` annotations only.

## Projection

```text
projected_watts =
    prefill_replicas × prefill_gpus_per_replica × prefill_cap_watts
  + decode_replicas  × decode_gpus_per_replica  × decode_cap_watts
```

For the values in this example (2 prefill replicas × 2 GPU × 350 W + 2 decode
replicas × 4 GPU × 300 W) the projection is `1400 + 2400 = 3800 W` against a
`5200 W` budget. A scale-up that would exceed `5200 W` is clamped to fit, and
the `dynamo_planner_power_budget_utilization` gauge reports the ratio.

## Changing a cap

> [!IMPORTANT]
> Editing a worker `podTemplate` cap rolls that worker (the template hash
> changes). After the rollout completes, **restart the Planner** so it re-reads
> the settled annotation at startup. Online cap retargeting without a restart is
> deferred to a dedicated dynamic-control design.
