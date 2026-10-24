<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# adaptive-imbalance

This policy keeps the default cost shape but moves the three cost weights from a background loop driven by pool-level load imbalance. The picker temperature stays at its baseline (or the per-request override); this version of the loop never changes it. It reproduces the default policy's device-cache and load terms; under even load it reproduces the default policy's picks whenever the omitted terms (see Known gaps) are inactive. Under a hot prefix it lowers the cache credit and raises the load weights so new requests move off the overloaded worker.

```
cost[w] = prefill_load_scale(t) × max(0, prefill_backlog_blocks[w] + request_blocks − overlap_score_credit(t) × cached[w])
        + decode_blocks[w]
        + decode_active_request_weight(t) × active_requests[w]

imbalance(t) = stddev / mean of active_requests across workers, clamped to 0..1, smoothed
split(t)     = sigmoid(imbalance(t); sigmoid_midpoint, sigmoid_slope)   when engaged, else 0
overlap_score_credit(t)         = baseline × (1 − split(t))
prefill_load_scale(t)           = baseline × (1 + gain × split(t))
decode_active_request_weight(t) = baseline × (1 + gain × split(t))
```

`request_blocks` and `prefill_backlog_blocks[w]` are token counts divided by the block size, as the default policy computes them, and the whole prefill term is zero for requests that do not track prefill tokens. The loop scales each load weight from its baseline, so with a baseline `decode_active_request_weight` of 0 (the router default) that weight stays 0 at any split.

## Parameters

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `overlap_score_credit` | router configuration | finite, >= 0 | Baseline cache credit |
| `prefill_load_scale` | router configuration | finite, >= 0 | Baseline prefill load scale |
| `decode_active_request_weight` | router configuration | finite, >= 0 | Baseline active-request weight |
| `router_temperature` | router configuration | finite, >= 0 | Baseline picker temperature |
| `interval_ms` | 250 | > 0 | Control loop period |
| `smoothing` | 0.3 | 0..1 | Weight of the previous imbalance value |
| `sigmoid_midpoint` | 0.5 | 0..1 | Imbalance at which the split is 0.5 |
| `sigmoid_slope` | 10 | > 0 | Steepness of the sigmoid |
| `gain` | 4.0 | >= 0 | How far the load weights rise at full split |
| `dwell_cycles` | 4 | integer >= 0 | Minimum cycles before the engaged state can change; 0 disables the dwell |
| `stale_after_ms` | 5000 | >= `interval_ms` | Age after which the last load sample counts as stale and the pool as idle; the scorer refreshes the sample only when it scores a request, so keep this above the longest request gap the loop should ride through |
| `enter_threshold` | 0.6 | 0..1 | Smoothed imbalance that engages the loop |
| `exit_threshold` | 0.4 | 0..1, below `enter_threshold` | Smoothed imbalance that disengages the loop |

With the defaults, the weights stay at baseline while the pool is even, so picks match the default policy's whenever the omitted terms are inactive; the loop only changes weights after the smoothed imbalance passes `enter_threshold`.

## Known gaps

The cost omits these default-policy terms, so picks can differ from the default policy's when they are active:

- Host and disk cache hits (`host_cache_hit_weight`, `disk_cache_hit_weight`) and shared-cache hits.
- `overlap_score_credit_decay`, which softens the cache credit for workers with a prefill backlog.
- Per-request `overlap_score_credit` and `prefill_load_scale` overrides, which the plugin context does not expose; only the temperature override is honored.

Other gaps:

- Requests constrained to a subset of workers (allowed worker ids, routing constraints) can produce a load sample that covers only that subset. The scorer skips samples with fewer than two observed workers and pinned requests, but not larger subsets.
- Decode pools: decode requests routed after a disaggregated prefill do not track prefill tokens, so this policy drops their prefill term, as the default policy does. With normal disaggregation the default policy also forces `overlap_score_credit` to 0 for them, and its cost is then the same as this policy's. With conditional disaggregation the default policy keeps a positive credit and uses a decode-specific branch that subtracts cached blocks from the decode blocks; this policy does not reproduce that branch, so it ignores the cache for decode.

## Run the Standalone EPP

```bash
DYN_EPP_MODE=standalone \
DYN_ROUTER_POLICY_CONFIG=examples/router/custom-policy-example/adaptive-imbalance/worker-selection.yaml \
DYN_ROUTER_WORKER_SELECTION_POLICY=adaptive \
RUST_LOG=info \
  cargo run --release -p dynamo-custom-policy-example-epp
```

## Request-path cost

This policy is an exception to the rule of keeping shared state out of `score` and `pick`: the scorer and the picker each take one lock-free `arc-swap` guard read of the current weights per request (the picker skips it when the request overrides the temperature). For requests with at least two observed workers and no pinned worker, the scorer also calls `Instant::now()` once and stores one new load sample (one `Arc` allocation and store) at most once per `interval_ms`. Both reuse their scratch buffers, so the request path does not allocate otherwise.

## Observability

Engaging and disengaging each log one line at info level (`adaptive-imbalance: engaged` / `adaptive-imbalance: disengaged`); every other weight change logs one `adaptive-imbalance: weights updated` line at debug level. All three carry the same fields: the imbalance, split, the three weights, the temperature and the age of the load sample. If no tokio runtime is present when the policy is created, a warning is logged and the weights stay at baseline.

## Tests

```bash
cargo test -p dynamo-custom-policy-example-adaptive-imbalance
```
