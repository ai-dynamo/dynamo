<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cache Affinity Within a Load Budget

This policy separates a hard cache preference from its load-safety bound. It first finds the least modeled active-load cost, keeps workers within `max_load_cost_delta_blocks` of that floor, and selects the remaining worker with the most effective prefix overlap. Effective overlap is the same tier-weighted value used by Dynamo's cache-loss funnel, including device and configured lower-tier cache credit.

The active-load cost uses Dynamo's configured prefill-load scale and active-request weight:

```text
load_cost = prefill_load_scale * active_prefill_blocks
          + decode_cost_blocks
          + decode_active_request_weight * active_requests
```

This is lexicographic selection, not another cache coefficient. A sufficiently large budget makes cache overlap dominate load; a finite budget prevents a cache-hot worker from winning when its modeled backlog is too far above the load floor.

```yaml
worker_selection:
  aggregated: cache-affinity-budget
  instances:
    - name: cache-affinity-budget
      type: cache-affinity-budget
      parameters:
        max_load_cost_delta_blocks: 64
```

The unit is modeled KV blocks. Set the value from a matched workload experiment; it is not a latency duration. A value of `0` restricts selection to minimum-load workers and uses cache overlap only as a tie-breaker.

## Test

```bash
cargo test -p dynamo-custom-policy-example-cache-affinity-budget
```
