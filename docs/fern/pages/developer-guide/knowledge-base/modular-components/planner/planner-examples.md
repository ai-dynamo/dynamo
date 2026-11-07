---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Planner Examples
subtitle: Examples for custom load predictors and the VirtualConnector for non-Kubernetes scaling environments.
---

Planner-specific examples for advanced configuration and non-Kubernetes
integrations. For DGDR manifests, see
[DGDR Templates](../../../../recipes/examples/dgdr.mdx). For the full configuration
reference, see the [Planner Guide](planner-guide.md).

## Direct Deployments Without Profiling ConfigMaps

The disaggregated [vLLM](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/disagg_planner.yaml),
[SGLang](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/sglang/deploy/disagg_planner.yaml),
and [TensorRT-LLM](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/trtllm/deploy/disagg_planner.yaml)
examples pass the Planner configuration as inline JSON through `--config`, with
`pre_deployment_sweeping_mode: none`. They do not create or mount a Planner
configuration or profiling-data ConfigMap.

The Planner bootstraps from worker self-benchmark results when available. With a
configured `ais_perf_model`, supported deployments can use native AISimulate
estimates before live FPM regression is ready. These examples leave that setting
unset, so without benchmark data the model learns from live Forward Pass Metrics
(FPM), and SLA scaling can report `model_not_ready` during warmup. Update the inline
configuration to tune the Planner; no placeholder profiling arrays are required. See
[Tune the Planner](../../../../kubernetes/auto-deployment/dynamo-planner.mdx)
for the deployment prerequisites and workflow.

## Compare Autoscaling Policies with Planner Gym

**Experimental.** [Planner Gym](https://github.com/ai-dynamo/dynamo/tree/main/gyms/planner-gym)
compares Dynamo Planner, a KEDA/HPA policy port, reactive scaling, and fixed
capacity on the same CPU replay substrate. Its
[getting started guide](https://github.com/ai-dynamo/dynamo/blob/main/gyms/planner-gym/docs/getting-started.md)
runs a two-policy comparison using the public Mooncake trace bundled with
Dynamo. No GPU or model weights are needed.

Inspect SLO-qualified goodput per average GPU together with latency,
good-request rate, GPU-hours, and scaling activity. The online backend uses
AIPerf to compare existing endpoints; deployment cost telemetry is not collected.

To replay a trace through one Planner configuration and inspect each scaling
or retune decision, see
[Simulate Planner Decisions with AISimulate](aisimulate-planner-replay.mdx).

## Custom Load Predictors

Each YAML block in this section is a standalone `PlannerConfig`. Save the block
as `planner.yaml` and pass it to
`python -m dynamo.planner --config planner.yaml`. To use the same fields in a
DGDR, nest them under `spec.features.planner`.

### Warm-starting with Trace Data

Pre-load predictors with historical request patterns before live traffic:

```yaml
optimization_target: sla
load_predictor: arima
load_predictor_warmup_trace: /data/trace.jsonl
load_predictor_log1p: true
```

The parser accepts per-request Mooncake JSONL records:

```json
{"timestamp": 0, "input_length": 4096, "output_length": 512}
```

It also accepts `dynamo.request.trace.v1` `request_end` records. The Planner
groups requests into adjustment intervals and computes request count, average
input sequence length (ISL), and average output sequence length (OSL).

### Kalman Filter Tuning

For workloads with rapid changes, tune the Kalman filter:

```yaml
optimization_target: sla
load_predictor: kalman
kalman_q_level: 2.0       # Higher = more responsive to level changes
kalman_q_trend: 0.5       # Higher = trend changes faster
kalman_r: 5.0             # Lower = trusts new measurements more
kalman_min_points: 3      # Fewer points before forecasting starts
load_predictor_log1p: true
```

### Prophet for Seasonal Workloads

For workloads with daily/weekly patterns:

```yaml
optimization_target: sla
load_predictor: prophet
prophet_window_size: 100   # Larger window for seasonal detection
load_predictor_log1p: true
```

## Power-Aware Budget Scaling

Keep the Planner's projected GPU power draw within a configured rack/DGD budget.
Per-GPU caps are DGD-owned: authored on each worker component's `podTemplate`
annotation (`dynamo.nvidia.com/gpu-power-limit`), applied to Pods by the
operator, and enforced by the Power Agent. The Planner only reads them and
combines them with `total_gpu_power_limit` (in its config) to project a budget
and clamp scale-up — it never patches Pods.

The [`power-aware-budget/`](https://github.com/ai-dynamo/dynamo/tree/main/examples/power-aware-budget)
example holds a [`dgd.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/power-aware-budget/dgd.yaml)
and a [`planner_config.json`](https://github.com/ai-dynamo/dynamo/blob/main/examples/power-aware-budget/planner_config.json)
that show the contract.

> [!NOTE]
> These manifests are a **contract fragment**, not a fully deployable DGD: they
> show only the power-relevant fields (annotations, GPU limits, Planner config).
> A runnable deployment also needs container images, model args, env/secrets,
> and a Frontend. Merge these fields into a complete DGD, such as one from
> [`examples/backends/vllm/deploy/`](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm/deploy).

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

### Ownership Model

| Value | Source of truth | Consumer |
| --- | --- | --- |
| Per-GPU cap (watts, static) | Worker component `podTemplate` annotation `dynamo.nvidia.com/gpu-power-limit` | Operator → Pods; Power Agent (applies NVML/DCGM cap); Planner (reads) |
| Total DGD power budget (watts) | PlannerConfig `total_gpu_power_limit` | Planner |
| Replica targets | Planner (after the budget clamp) | Existing scaling adapter |
| Applied GPU hardware cap | Power Agent | NVML / DCGM |

The Planner **reads** the caps from the DGD; it does **not** patch Pods. New
replicas created by a scale-up are born with the annotation because the operator
renders it into the Deployment / LeaderWorkerSet template.

The Planner caches each component's power tuple at startup: the cap, the
effective main-container `nvidia.com/gpu` count (limit first, request fallback),
and `multinode.nodeCount` (default `1`). DGD admission rejects updates to that
tuple for an annotated component. Delete and recreate the DGD to change it.
Changing `total_gpu_power_limit` also requires restarting the Planner process.

### Author the Caps

1. Put the per-GPU cap on each worker component's `podTemplate.metadata.annotations`.
   Prefill and decode may differ.
2. Set the total budget on the Planner's mounted config:

   ```json
   {
     "enable_power_awareness": true,
     "total_gpu_power_limit": 5200
   }
   ```

   `enable_power_awareness` requires `environment: "kubernetes"`, a
   `total_gpu_power_limit > 0`, and `mode` set to `disagg`, `prefill`, or
   `decode` (`agg` is not currently supported). Per-GPU caps are **not** set
   here — they live on the worker `podTemplate` annotations only.

3. Enable the Planner's `pods/list` RBAC permission in the Helm installation.
   Power-annotation settlement reads Pod annotations at startup to verify that
   power caps have propagated before caching them, which requires this
   permission. Set it when installing or upgrading the platform chart:

   ```bash
   helm dependency build deploy/helm/charts/platform
   helm upgrade --install dynamo deploy/helm/charts/platform \
     --set dynamo-operator.planner.powerAwareness.enabled=true
   ```

   Without this flag the Planner's ServiceAccount lacks `pods/list` and the
   startup settlement check fails with a Kubernetes RBAC error.

### Projection

```text
projected_watts =
    prefill_replicas × prefill_gpus_per_replica × prefill_cap_watts
  + decode_replicas  × decode_gpus_per_replica  × decode_cap_watts
```

For the values in the example (2 prefill replicas × 2 GPU × 350 W + 2 decode
replicas × 4 GPU × 300 W) the projection is `1400 + 2400 = 3800 W` against a
`5200 W` budget. A scale-up that would exceed `5200 W` is clamped to fit, and
the `dynamo_planner_power_budget_utilization` gauge reports the ratio.

### Static-Input Limitations

> [!IMPORTANT]
> You cannot patch a power annotation, effective GPU count, or node count in
> place on a power-annotated component. Admission rejects the update; delete and
> recreate the DGD, then start the Planner against the replacement. Restart the
> Planner after changing `total_gpu_power_limit`.

Power awareness currently assumes a static, uniform GPU count and cap per
component. Mixed GPU generations within one component, dynamic cap retargeting,
and GPU allocation through DRA resource claims are not supported by this
projection.

## Virtual Connector

For non-Kubernetes environments, use the VirtualConnector to communicate scaling
decisions:

```python
from dynamo._core import DistributedRuntime, VirtualConnectorClient

# Initialize client
client = VirtualConnectorClient(distributed_runtime, namespace)

# Main loop: watch for planner decisions and execute them
while True:
    # Block until the planner makes a new scaling decision
    await client.wait()

    # Read the decision
    decision = await client.get()
    print(f"Scale to: prefill={decision.num_prefill_workers}, "
          f"decode={decision.num_decode_workers}, "
          f"id={decision.decision_id}")

    # Execute scaling in your environment
    scale_prefill_workers(decision.num_prefill_workers)
    scale_decode_workers(decision.num_decode_workers)

    # Report completion
    await client.complete(decision)
```

See the
[VirtualConnector integration test](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/planner/tests/integration/test_virtual_connector.py)
for a complete example.

## Related Documentation

- [Planner Guide](planner-guide.md) -- Planner configuration reference
- [DGDR Templates](../../../../recipes/examples/dgdr.mdx) -- DGDR YAML examples
- [Profiler Guide](../profiler/profiler-guide.md) -- Profiling workflow
