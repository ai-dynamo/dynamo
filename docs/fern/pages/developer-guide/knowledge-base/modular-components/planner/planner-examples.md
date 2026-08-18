---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Planner Examples
subtitle: Examples for custom load predictors and the VirtualConnector for non-Kubernetes scaling environments.
---

Planner-specific examples for advanced configuration and non-Kubernetes
integrations. For DGDR manifests, see
[DGDR Templates](../../../../recipes/kubernetes-templates/dgdr.mdx). For the full configuration
reference, see the [Planner Guide](planner-guide.md).

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

The mounted PlannerConfig enables it:

```json
{
  "enable_power_awareness": true,
  "total_gpu_power_limit": 5200
}
```

`enable_power_awareness` requires `environment: "kubernetes"` and
`mode` set to `disagg`, `prefill`, or `decode` (`agg` is not supported).
The Planner caches each annotated component's cap, effective main-container GPU
count, and node count at startup. DGD admission rejects changes to those fields;
delete and recreate the DGD to change them. Restart the Planner after changing
`total_gpu_power_limit`.

You must also enable `pods/list` RBAC for the Planner's ServiceAccount at
install time. The Planner reads Pod annotations during startup to verify that
power caps have propagated before caching them. Without the permission the
startup settlement check fails. Pass this flag when installing or upgrading the
platform chart:

```bash
helm dependency build deploy/helm/charts/platform
helm upgrade --install dynamo deploy/helm/charts/platform \
  --set dynamo-operator.planner.powerAwareness.enabled=true
```

See the `power-aware-budget/` directory in
[Dynamo examples](https://github.com/ai-dynamo/dynamo/tree/main/examples) for
the full annotation + config contract and its limitations (the budget is a
projected ceiling over requested caps, not a proven hardware limit). Mixed GPU
generations, dynamic cap retargeting, and DRA-backed GPU allocation are not
supported.

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
- [DGDR Templates](../../../../recipes/kubernetes-templates/dgdr.mdx) -- DGDR YAML examples
- [Profiler Guide](../profiler/profiler-guide.md) -- Profiling workflow
