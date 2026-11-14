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

imbalance(t) = sqrt(max(0, variance − mean)) / mean of active_requests across workers, clamped to 0..1, smoothed
split(t)     = sigmoid(imbalance(t); sigmoid_midpoint, sigmoid_slope)   when engaged, else 0
               the loop engages above enter_threshold, or on a full worker (see slot_capacity)
               at least sigmoid(enter_threshold; ...) while a worker is full (see slot_capacity)
overlap_score_credit(t)         = baseline × (1 − split(t))
prefill_load_scale(t)           = baseline × (1 + gain × split(t))
decode_active_request_weight(t) = baseline × (1 + gain × split(t)) + split(t) × spread_active_request_weight
```

Requests placed at random give each worker a count whose variance is about its mean, so `imbalance` subtracts the mean from the variance: a sparse pool (under one request per worker) that is uneven only by chance scores 0, while a spike piled on a few workers still scores high. Under heavy load the measure converges to the plain coefficient of variation.

`request_blocks` and `prefill_backlog_blocks[w]` are token counts divided by the block size, as the default policy computes them, and the whole prefill term is zero for requests that do not track prefill tokens. The loop scales each load weight from its baseline. Scaling alone would keep a baseline `decode_active_request_weight` of 0 (the router default) at 0, and that is the only term that counts in-flight requests (sequence slots) rather than KV blocks, so the loop also adds `split(t) × spread_active_request_weight` to it. At split 0 the term is 0 and the weights equal the baseline.

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
| `spread_active_request_weight` | 64 | finite, >= 0 | Active-request weight added at full split, in KV blocks per in-flight request; 0 restores pure scaling |
| `dwell_cycles` | 4 | integer >= 0 | Minimum cycles before the engaged state can change; 0 disables the dwell |
| `stale_after_ms` | 5000 | >= `interval_ms` | Age after which the loop stops reading the last load sample and holds its state; the scorer refreshes the sample only when it scores a request, and a gap between requests says nothing about the pool's load, so the weights drain only when a new sample shows an even or idle pool |
| `enter_threshold` | 0.6 | 0..1 | Smoothed imbalance that engages the loop |
| `exit_threshold` | 0.4 | 0..1, below `enter_threshold` | Smoothed imbalance that disengages the loop |
| `slot_capacity` | absent (hold off) | integer > 0 | Sequence slots per worker (the engine's `max_num_seqs`); the router does not expose it to the policy |
| `saturation_hold` | 0.85 | above 0, up to 1 | Fraction of `slot_capacity` at which the busiest worker counts as full |

With the defaults, the weights stay at baseline while the pool is even, so picks match the default policy's whenever the omitted terms are inactive; the loop only changes weights after the smoothed imbalance passes `enter_threshold`.

The imbalance measures how even the load is, not whether workers have free slots. When the loop spreads a burst evenly over workers that are all full, the imbalance falls below `exit_threshold`, and returning to the affinity weights sends the next wave to a worker without a free slot. A wave that lands while the loop is off goes with the affinity weights to the cached workers, which fill up before the imbalance reaches `enter_threshold`. With `slot_capacity` set, a worker with at least `saturation_hold × slot_capacity` requests in flight engages the loop, an engaged loop does not disengage while any worker is that full, and the split stays at or above the split at `enter_threshold` meanwhile. Engaging on a full worker still waits for `dwell_cycles`. The engaged and disengaged log lines carry `saturated`, which says whether a worker was full in that tick's sample.

## Known gaps

The cost omits these default-policy terms, so picks can differ from the default policy's when they are active:

- Host and disk cache hits (`host_cache_hit_weight`, `disk_cache_hit_weight`) and shared-cache hits.
- `overlap_score_credit_decay`, which softens the cache credit for workers with a prefill backlog.
- Per-request `overlap_score_credit` and `prefill_load_scale` overrides, which the plugin context does not expose; only the temperature override is honored.

Other gaps:

- Requests constrained to a subset of workers (allowed worker ids, routing constraints) can produce a load sample that covers only that subset. The scorer skips samples with fewer than two observed workers, pinned requests and previews, but not larger subsets.
- Decode pools: decode requests routed after a disaggregated prefill do not track prefill tokens, so this policy drops their prefill term, as the default policy does. With normal disaggregation the default policy also forces `overlap_score_credit` to 0 for them, and its cost is then the same as this policy's. With conditional disaggregation the default policy keeps a positive credit and uses a decode-specific branch that subtracts cached blocks from the decode blocks; this policy does not reproduce that branch, so it ignores the cache for decode.

## Run the Standalone EPP

```bash
DYN_EPP_MODE=standalone \
DYN_ROUTER_POLICY_CONFIG=examples/router/custom-policy-example/adaptive-imbalance/worker-selection.yaml \
RUST_LOG=info \
  cargo run --release -p dynamo-custom-policy-example-epp
```

`worker-selection.yaml` selects the policy for the aggregated role with its parameters. Do not also set `DYN_ROUTER_WORKER_SELECTION_POLICY=adaptive-imbalance`: that override replaces the YAML selection and runs the policy with default parameters, so edits to the file would have no effect.

## Request-path cost

This policy is an exception to the rule of keeping shared state out of `score` and `pick`: the scorer takes one lock-free `arc-swap` guard read of the current weights per request. For requests with at least two observed workers, no pinned worker and no preview, the scorer also reads the clock once; at most once per `interval_ms` one of them wins a compare-and-swap and stores a new load sample (one `Vec` and one `Arc` allocation). The picker holds a short lock on its PRNG and a reused cost buffer while it picks, as the default picker does with its softmax scratch. The request path does not allocate otherwise.

## Observability

Engaging and disengaging each log one line at info level (`adaptive-imbalance: engaged` / `adaptive-imbalance: disengaged`); every other weight change logs one `adaptive-imbalance: weights updated` line at debug level. All three carry the same fields: the imbalance, split, the three weights and the age of the load sample. If no tokio runtime is present when the policy is created, a warning is logged and the weights stay at baseline.

## Tests

```bash
cargo test -p dynamo-custom-policy-example-adaptive-imbalance
```
