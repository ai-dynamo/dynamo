<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LoRA routing-churn optimization findings

These CPU-only experiments measure changes to frontend routing-target replica sets. They do not
measure actual GPU adapter loads, unloads, or eviction latency.

## Method

The ignored tests in `optimization_experiments.rs` use the production `LoraController`, fixed
seeds, capacity-only worker registrations, and a fresh per-tick load snapshot. Run them with:

```bash
cargo test -p dynamo-llm --test lora_simulation \
  optimization_experiments -- --ignored --nocapture
```

The experiments also served as a correctness check for comparison mode. They found that a
configured scale-down cooldown of zero still deferred the first scale-down. The controller now
treats zero as an explicit off switch; all results below include that fix.

## Hypothesis 1: MCF incentive saturation

For a feasible prior placement, moving from HRW rank `r_old` to `r_new` is cheaper only when the
rank improvement exceeds the combined load/keep incentive:

```text
alpha_pref * (r_old - r_new) > gamma_load + beta_keep
```

With eight workers and `alpha_pref=1`, the maximum rank advantage is seven. The hypothesis was
that any combined incentive greater than seven would have the same placement and churn as the
production default for this topology.

The workload used 32 distinct adapters, 8 workers, 4 resident slots per worker, 100 ticks,
Zipf-1 popularity, Poisson arrivals with mean total load 40, and seed 42. Thus `L=32=N*K`.

| Label | `gamma_load` | `beta_keep` | Total churn | Adds | Removes | Mean targets | Peak targets | Overflow sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0 | 0 | 2,592 | 1,312 | 1,280 | 31.960 | 32 | 0 |
| below bound | 3 | 2 | 2,468 | 1,250 | 1,218 | 31.960 | 32 | 0 |
| above bound | 8 | 0 | 2,468 | 1,250 | 1,218 | 31.960 | 32 | 0 |
| default | 1,000 | 250 | 2,468 | 1,250 | 1,218 | 31.960 | 32 | 0 |

Result: the hypothesis held. The above-bound and default runs were byte-for-byte equivalent in
their collected metrics. A combined incentive of five also reached the same result in this
workload because no observed beneficial rank move exceeded five; that is not a topology-wide
guarantee. Any positive tested incentive reduced churn by 4.8% relative to no incentive.

Implication: increasing the default weights further will not reduce churn on this topology.
Lowering them may preserve the same behavior for small worker pools, but a production default
must cover larger pools and heterogeneous churn weights. A topology-normalized configuration,
or at least a warning when `gamma_load + beta_keep <= alpha_pref * (N-1)`, is more defensible than
blindly tuning larger constants.

## Hypothesis 2: cooldown trades churn for route pressure

Four always-active adapters alternated hot/cold load every tick on four workers with two slots
each. The hypothesis was that cooldown would suppress repeated scale-down/scale-up churn but
retain more targets and possibly create pressure or solver overflow.

| Algorithm | Cooldown | Total churn | Adds | Removes | Mean targets | Peak targets | Ticks above `N*K` | Overflow sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HRW | 0 | 241 | 124 | 117 | 7.000 | 7 | 0 | 0 |
| HRW | 1, 3, or 5 | 10 | 10 | 0 | 9.925 | 10 | 39 | 0 |
| MCF | 0 | 241 | 124 | 117 | 7.000 | 7 | 0 | 0 |
| MCF | 1, 3, or 5 | 8 | 8 | 0 | 7.975 | 8 | 0 | 40 |

Result: the hypothesis held, with an important failure mode. Any positive cooldown reduced HRW
churn by 95.9% and MCF churn by 96.7%, but it did so by never applying the alternating
scale-down. HRW retained up to 10 route targets against 8 resident slots for 39 of 40 ticks. MCF
kept its published target count at the capacity boundary but reported one overflowed unit on each
tick.

Cooldown values 1, 3, and 5 were identical because each scale-up clears hysteresis state; the next
scale-down is therefore a new first scale-down and is deferred again. This workload exposes a
binary behavior: any positive cooldown stabilizes routes, while zero follows demand immediately.

Implication: cooldown is a powerful churn control, but churn alone is the wrong optimization
objective. It must be evaluated jointly with route-target pressure, MCF overflow, time to adapt
after a sustained drop, and eventually observed mock residency churn.

## Recommended next experiments

1. **Adaptive scale-down guard.** Defer scale-down only while projected targets stay within a
   configurable pressure budget and MCF has no overflow. Compare against fixed cooldown on burst,
   diurnal, and MMPP loads.
2. **Volatility-aware cooldown.** Give oscillating adapters a longer cooldown and stable adapters
   a short or zero cooldown. Measure churn, pressure, and convergence delay after a permanent load
   drop.
3. **HRW processing order.** Compare the current deterministic name order with descending desired
   replicas and descending load. The hypothesis is that demand-first placement reduces partial
   allocations under capacity pressure, but it may add churn when load ranks cross.
4. **MCF candidate width.** Sweep `candidate_m` with solver runtime instrumentation. Correctness
   should remain stable because the solver expands or retries densely, while a smaller candidate
   graph may reduce solve time until retries dominate.
5. **Topology scale.** Repeat the incentive sweep at worker counts 16, 64, 256, and 1,024. Check
   the predicted saturation threshold `alpha_pref * (N-1)` and record solve latency as well as
   churn.
6. **Closed-loop mock residency.** After mocker can publish synthetic load/eviction model cards,
   compare routing-target churn with actual mock residency churn and adapter cold-load latency.

No production default should change from these two seed-42 experiments alone. They establish
testable bounds and identify the metrics required for a safe optimization.
