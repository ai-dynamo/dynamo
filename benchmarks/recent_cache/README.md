<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Hot-block pressure routing: offline experiment

**TTL retention is promising; a broadly superior adaptive credit is not established.**
This experiment found a sustained replication/eviction regime in which keeping
recent routing predictions substantially improves reuse and latency. Most of that
benefit also appears with the default credit and TTL prediction alone. On the
original full-hour Mooncake trace, static credit 64 beats every tested hybrid.

The primary metric is completed output tokens divided by simulated replay duration.
The secondary metric is p90 request end-to-end latency, plus p90 complete trajectory
latency for Weka. These are CPU simulations with AIC engine timing, not GPU
measurements. Finite replay throughput includes arrivals, tool gaps, and queue drain;
it does not by itself establish maximum sustainable server throughput.

## Controller and comparison

The controller uses:

```text
pressure = sum(hot input block copies across workers and DP ranks)
           / sum(worker and DP-rank KV block capacities)

credit = boosted_credit if pressure > 0.5 else 1
```

A hot block is a complete input-prefix block eagerly booked or refreshed on
successful router admission. The same prefix on three workers counts three times.
Expiration uses **virtual replay time**, never host elapsed time. Eviction and
generated output do not update this index; worker removal removes its contribution.
The TTL predictor supplements event-reported overlap with the longest contiguous
hot prefix, taking the larger overlap. It changes placement and router load hints,
but actual engine cache reuse is measured independently at first admission.

TTL-only predictions are **not guaranteed to be a subset of resident blocks**.
They may include evicted or not-yet-materialized KV; pressure can exceed 1. A short
TTL limits this error but does not prove the proposed subset invariant. The threshold
stays at 0.5 throughout the campaign. TTL, arrival rate, working set, and worker count
are calibrated to exercise both sides of the threshold. No scale-invariance sweep
is part of this experiment.

| Arm | TTL prediction | Overlap credit / policy |
| --- | --- | --- |
| Stock default | Off | Existing cache-aware cost, credit 1 |
| Prediction only | On | Same cost, credit 1 |
| Static | On | Credit 4, 16, or 64 throughout; 32 supplemental |
| Hybrid | On | Credit 1 below threshold; 4, 16, or 64 above; 32 supplemental |
| SMG-style default | Off | Existing local `dynamo-two-tier-cost-fn` |

An adaptive win must beat **all static credits 1/4/16/64 and SMG** on the same
workload. Prediction-only credit 1 is essential: comparing only with stock would
confound retained prediction with adaptive switching. Stock already has normal
cache-aware routing and eager active-request tracking; it is not cache-blind.

For these device-cache-only configurations, the default cost is approximately
`max(active_prefill + request_blocks - credit * overlap, 0) + decode_load`.
Consequently even credit 64 is not strict sticky-session routing: the zero clamp
can erase overlap distinctions and decode load still matters. Some small repeated
working sets actually replicate more at credit 64.

The SMG arm runs Dynamo's existing builtin implementation, a port of the historical
[SGLang experimental `cache_aware_zmq` policy](https://github.com/sgl-project/sglang/blob/ecd97de1fcefd83f08ca3eaa8bcf75e3e8233805/experimental/sgl-router/src/policies/cache_aware_zmq.rs).
Its defaults prefer least requests when `max_load - min_load > 32` and
`max_load > 1.1 * min_load`; otherwise a hit ratio greater than 0.5 uses
maximum device overlap, with least requests as fallback/tiebreaker. This is not a
claim about the full current SMG product. The local port uses router in-flight load
and ceiling request blocks, rather than upstream engine queue snapshots and floor
full-block counts.

## Full-hour public Mooncake result

All 23,608 original requests; eight TP1 workers; arrival speedup 1.2; virtual TTL
25 seconds. Every arm completed. Each row is one run; differences are exploratory.

| Policy | Output tokens/s | p90 request latency, s | Actual input reuse |
| --- | ---: | ---: | ---: |
| Stock default | 1,214.50 | 462.83 | 29.83% |
| Prediction only, credit 1 | 1,227.61 | 446.17 | 30.06% |
| Static 4 | 1,223.49 | 432.71 | 30.50% |
| Static 16 | 1,244.12 | 388.72 | 32.41% |
| Static 64 | **1,276.22** | **308.53** | **35.47%** |
| Hybrid 4 | 1,222.66 | 447.75 | 30.34% |
| Hybrid 16 | 1,227.40 | 436.80 | 30.51% |
| Hybrid 64 | 1,231.08 | 415.28 | 31.17% |
| SMG-style default | 1,185.24 | 432.26 | 32.04% |

Static 64 improves throughput **5.08%** over stock; the best hybrid improves it
**1.36%**, but trails static 64 by **3.54%**. Hybrid 64 boosts on 72.44% of decisions
and crosses the threshold upward/downward 246/245 times, so this negative result
does not come from an inactive switch. Event-visible replication is only about
1.02–1.03 here: much of the churn is distinct working-set pressure.

## Sustained replication and eviction

The constructed shared128 workload repeats the first 128 public Mooncake requests
as four overlapping copies, every 60 seconds for 20 cycles: 10,240 requests.
Prefixes remain shared across copies and cycles. Four TP1 workers provide 15,616
aggregate KV blocks. The approximate distinct complete-input footprint is 12,328
blocks (79% of aggregate capacity), before generated outputs, partial blocks,
in-flight allocation, and replication. Virtual TTL is 15 seconds.

For the first comparison run, follow the last five arrival cohorts (900–1,200
virtual seconds) through completion. Cache observations use that common time window.

| Policy | Late actual reuse | Late p90 request, s | Median duplicate fraction of resident copies | Cache-visible capacity removals |
| --- | ---: | ---: | ---: | ---: |
| Stock default | 79.50% | 84.68 | 27.68% | 80,314 |
| Prediction only | 94.33% | 33.95 | 12.37% | 27,126 |
| Hybrid 16 | 94.35% | 35.03 | 11.69% | 27,078 |
| Static 16 | 92.68% | 35.24 | 12.97% | 33,049 |
| Static 64 | 87.10% | 44.13 | 13.75% | 52,996 |
| SMG-style default | 75.30% | 256.24 | 30.37% | 82,436 |

Default's duplicate fraction stays at 27.55% in the first 180 seconds and 27.68%
in the late window, with roughly 98.7% occupancy. Late arrivals still suffer 136
preemptions across 110 requests. Prediction-only and hybrid have none in that
cohort. This supports sustained replication consuming capacity and accompanying
eviction/reuse loss, rather than merely a cold-start effect.

Across that full run, output throughput is 1,701.16 / 1,755.50 / 1,766.64 tokens/s
for stock / prediction-only / hybrid 16, and p90 request latency is
92.31 / 36.82 / 36.16 seconds. Prediction alone explains 83% of the throughput gain
and 98.8% of the latency reduction. Hybrid exceeds the best static in this first
run by only 0.18%, near the offered-load ceiling. Hybrid's late-cohort p90 is
slightly worse than prediction-only.

Repeating every arm three times gives the following medians. The shared140
variant uses 140 source requests and otherwise the same configuration (11,200
requests; approximate distinct complete-input footprint 84% of capacity).

| Policy | Shared128 tokens/s | Shared128 p90, s | Shared140 tokens/s | Shared140 p90, s |
| --- | ---: | ---: | ---: | ---: |
| Stock default | 1,701.16 | 92.31 | 1,420.00 | 326.24 |
| Prediction only, credit 1 | 1,756.73 | 35.84 | 1,839.48 | 40.51 |
| Static 4 | 1,761.62 | **33.98** | **1,848.43** | **35.88** |
| Static 16 | 1,756.05 | 36.75 | 1,837.98 | 40.08 |
| Static 64 | 1,742.80 | 42.53 | 1,831.69 | 45.65 |
| Hybrid 4 | 1,755.30 | 35.88 | 1,848.13 | 38.99 |
| Hybrid 16 | **1,763.74** | 36.16 | 1,843.38 | 39.41 |
| Hybrid 64 | 1,754.49 | 39.53 | 1,834.86 | 41.59 |
| SMG-style default | 1,477.40 | 226.10 | 1,369.64 | 408.61 |

Shared128's best hybrid leads the best static median by **0.12%**, with overlapping
throughput ranges: hybrid 16 is 1,760.49–1,766.64, static 4 is 1,751.01–1,768.23.
Its median p90 is 6.43% worse. Shared140's initial hybrid lead reverses: static 4
wins the median, and hybrid 4 has 8.67% worse p90. These results do not establish
a material adaptive advantage over tuned static routing.

## Public Weka/AgentX subset

Four private copies of seven complete plays; 1,772 requests; two TP2 workers;
28 agentic lanes; timing speedup 5; virtual TTL 120 seconds. The graph and tool
gaps are preserved and scaled. Qwen3-32B/vLLM/H200 timing is a proxy for these
recorded Claude plays, not a measurement of Claude serving. Each arm below is
one run.

| Policy | Output tokens/s | p90 request latency, s | p90 trajectory latency, s |
| --- | ---: | ---: | ---: |
| Stock default | 102.32 | 563.37 | 25,139.54 |
| Prediction only, credit 1 | **105.38** | 550.60 | 23,845.07 |
| Static 4 | 103.34 | 557.26 | 24,632.97 |
| Static 16 | 102.76 | **532.74** | 24,847.23 |
| Static 64 | 105.03 | 556.81 | 23,956.06 |
| Hybrid 4 | 104.10 | 552.60 | 24,567.25 |
| Hybrid 16 | 104.50 | 552.26 | 24,117.42 |
| Hybrid 64 | 105.02 | 566.19 | 24,360.02 |
| SMG-style default | 103.48 | 560.16 | 23,923.85 |

Prediction-only wins throughput here. Hybrid 64 exercises its boost on 76.41% of
decisions, with 108/107 upward/downward crossings, yet trails prediction-only
on throughput and both latency metrics.

At TTL 60, hybrid 4 produces 103.78 tokens/s versus static 16's 103.76. That
tiny ordering is not a robust win: static 4 varies from roughly 102.5 to 107.31
across repeats. The native seven-play, source-timing reference at TTL 300 is
sparse and gap-limited, with approximately 81 tokens/s for stock, hybrid, and
static 4. Neither result supports a general adaptive advantage.

## Broader screen and interpretation

Calibration and comparison cover original Mooncake prefixes and the full hour,
the locally available correlated tool-agent transformation, repeated shared/private
Mooncake working sets, and a public Weka/AgentX subset. Weka uses seven complete
supported plays (443 requests), plus four private copies (1,772 requests), retaining
dependencies and tool gaps. This is a selected subset, not the complete AgentX
benchmark. The tool-agent transformation is not an independent source corpus.

Virtual TTLs span 5–300 seconds; the useful setting depends strongly on the
arrival rate and working set. At 300 seconds, the full-hour Mooncake calibration
stays above the threshold on 99.38% of decisions with no downward crossing. At
25 seconds with higher offered load, the full comparison crosses repeatedly.
The repeated workloads run for twenty cycles so retention, replication, and
eviction can persist well beyond startup. Small repeated working sets give
default routing 97–98% reuse and do not produce the desired default pathology;
larger sets occupying roughly 79–84% of aggregate input capacity do.

The higher-rate shared128 case (45-second period) yields 1,482.60 tokens/s for
stock, 2,313.98 for prediction-only, 2,313.30 for the best hybrid (16), and
2,301.36 for the best static 4/16/64 (16). The apparent hybrid win disappears
when prediction-only credit 1 is included. The private128 control at a 60-second
period yields 593.44 for stock, 609.84 for the best hybrid (64), and 627.34 for
static 64. Private copies change both cross-copy sharing and total unique working
set; they are not a control that holds unique footprint constant.

These results support retaining a short-lived predictor as a useful separate
idea. They do not establish that global hot-block pressure reliably selects the
best overlap credit. Similar pressure can reflect distinct working-set demand
or redundant copies, while increasing credit is not monotonic in actual
colocation because of the existing cost clamp. Selecting a dose after inspecting
the same workloads is exploratory; a tiny median advantage is not a universal
speedup or held-out validation.

## Reproduction and measurement scope

See [REPRODUCING.md](REPRODUCING.md) for pinned public downloads, workload generation,
all nine policy arms, parallel execution, and sanitized CSV export. No raw traces,
per-request reports, host logs, or private workload material are included here.

The timing configuration is vLLM 0.24.0 / H200 SXM / Qwen3-32B, with 64-token
engine blocks. TP1 uses 3,904 KV blocks per worker; TP2 uses 11,742. The Python
package is AISimulate 0.12.0.dev1; native core is 0.12.0-dev.1. The baseline Dynamo
commit is `a996762e7aaa09c7dfc66faefb12924048ddb63c`. Comparison bindings use
`aic-forward-pass`; the expanded SMG/diagnostic campaigns also use `replay-builtin`.
Earlier baselines predate that adapter. None use the identity/seed changes of
`replay-bench`. Normal selector ties can vary; replicate ranges are not confidence
intervals, and identical repeats are not independent statistical evidence.

The independent resident ledger observes native device `Stored`, `Removed`, and
`Cleared` events. It counts published full input/output sequence-block copies,
global unique blocks, duplicates, and occupancy. It excludes partial, private,
uncomputed, and same-rank duplicate physical pages. For the native vLLM cache used
here, `vllm_block_pool.rs` emits `Removed` when capacity eviction removes the last
cache-visible same-rank copy; completion merely makes blocks eligible for LRU.
These counts are therefore **cache-visible capacity removals**, not all physical
page evictions. Samples occur at replay event boundaries; sampled medians/means
are not exact time integrals. No CPU/SSD spill tier is simulated.

Actual cache reuse comes from engine first-admission records. A routing prediction
above actual reuse may be stale, eagerly booked, or affected by block granularity;
it is not itself proof of eviction. Native preemption logs provide counts, while
their host timestamps are deliberately excluded from virtual-time analysis.
Request readmissions do not directly measure executed recomputation.

Implementation is opt-in and limited to aggregated offline replay. Set
`DYN_REPLAY_RECENT_CACHE` to a JSON object containing `ttl_secs`, `threshold`,
`boost_credit`, `mode` (`observe`, `fixed`, `adaptive`), `predict`, and optional
`trace_decisions`. Omitting the variable leaves the experiment disabled. The
`replay-builtin` feature enables only the existing two-tier policy adapter for
this comparison. AISimulate source and live routing policy math are unchanged.

Validation includes 41 targeted offline-router tests, seven existing builtin
policy tests, a rebuilt Python extension, and virtual-time smoke replays checking
TTL refresh/expiry, threshold crossings, policy activation, and resident accounting.
The smoke spans 360 virtual seconds in roughly two host seconds. Broad campaigns
run only after these checks; timeouts and invalid-input runs are retained as failed
evidence and excluded from performance comparisons.

The campaign contains **252 successful replays**, including calibration, smoke,
and repeated comparisons, plus **11 failed attempts** (five host timeouts and six
input/build-configuration failures). Successful rows pass matching expected/total/completed
request counts; agentic rows also have complete trajectories. Resident-ledger
runs have zero unknown removals and no observed copy count above configured capacity.
Twenty-one additional focused harness checks cover incomplete-result rejection and
unsupported dependency-trace rejection. Formatting, pre-commit hooks, and strict
CODEOWNERS coverage pass. Generated CSVs, figures, per-request reports, traces, and
logs remain external; the committed tables and reproducers are the tracking record.
