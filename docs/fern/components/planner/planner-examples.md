---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Planner Examples
---

Planner-specific examples for advanced configuration and non-Kubernetes
integrations. For DGDR manifests, see
[DGDR Examples](../../kubernetes/dgdr-examples.md). For the full configuration
reference, see the [Planner Guide](planner-guide.md).

## Custom Load Predictors

### Warm-starting with Trace Data

Pre-load predictors with historical request patterns before live traffic:

```yaml
# In planner arguments
args:
  - --load-predictor arima
  - --load-predictor-warmup-trace /data/trace.jsonl
  - --load-predictor-log1p
```

The trace file should be in mooncake-style JSONL format with request-count, ISL,
and OSL samples.

### Kalman Filter Tuning

For workloads with rapid changes, tune the Kalman filter:

```yaml
args:
  - --load-predictor kalman
  - --kalman-q-level 2.0      # Higher = more responsive to level changes
  - --kalman-q-trend 0.5      # Higher = trend changes faster
  - --kalman-r 5.0            # Lower = trusts new measurements more
  - --kalman-min-points 3     # Fewer points before forecasting starts
  - --load-predictor-log1p    # Often helps with request-rate series
```

### Prophet for Seasonal Workloads

For workloads with daily/weekly patterns:

```yaml
args:
  - --load-predictor prophet
  - --prophet-window-size 100   # Larger window for seasonal detection
  - --load-predictor-log1p
```

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

See `components/planner/test/test_virtual_connector.py` for a full working
example.

## Related Documentation

- [Planner Guide](planner-guide.md) -- Planner configuration reference
- [DGDR Examples](../../kubernetes/dgdr-examples.md) -- DGDR YAML examples
- [Profiler Guide](../profiler/profiler-guide.md) -- Profiling workflow
