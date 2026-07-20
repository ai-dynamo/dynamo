---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Planner Goodput-per-GPU Sweep
subtitle: Experimental Spica search over Planner policies for MiniMax-M2.5 on B200 GPUs
---

> [!WARNING]
> **Experimental.** These replay results characterize one software snapshot and workload. Do not
> treat them as production capacity guidance or a performance commitment. Spica's search behavior
> and output may change without a standard deprecation period.

Enable the planner (SLA mode) on the same deployment as the
[router experiment](router-e2e-latency-sweep.md), then use Spica's smart sweep to **find the
planner configuration that maximizes goodput-per-GPU** on the Mooncake toolagent trace, and
compare against the static deployment and the default planner.

## Setup

| | |
|---|---|
| model / hardware | `MiniMaxAI/MiniMax-M2.5`, `b200_sxm`, **TP4 = TEP** (`moe_ep=4`), vLLM 0.19.0 |
| deployment | agg, **planner-managed replicas** (1–8 workers; `gpu_budget=32` cap) |
| router | fixed `kv_router` (`overlap_score_credit=1.0`, `prefill_load_scale=4.0`, `router_temperature=0.0`) |
| workload | Mooncake toolagent trace (23,608 reqs), **open-loop** (arrival timestamps, native rate ~6.7 req/s) |
| SLA | `ttft_ms=2000`, `itl_ms=50` (per-request goodput SLA) |
| goal | maximize `goodput_per_gpu` = goodput / **avg_gpu** (time-averaged provisioned GPUs) |
| Dynamo source revision | `c7378f8e` |

### Why open-loop (not closed-loop)

A planner experiment must be **open-loop**. In closed-loop (fixed N in flight) the throughput
is *concurrency-bound* — fewer workers serve the same N concurrent requests **slower**, so
scaling down lowers throughput (hence goodput) roughly as fast as it lowers GPU count, and
goodput-per-GPU does **not** improve (the optimum degenerates to "stay at max workers"). In
open-loop the throughput is *arrival-bound* (fixed total tokens over the trace's ~59 min), so
adding workers does **not** raise throughput — it only lowers latency. The planner can then
scale **down** in lulls without losing goodput, and goodput-per-GPU improves. That is the
regime where the planner has value.

## Baselines (open-loop, full trace)

| config | goodput (tok/s) | avg_gpu | **goodput/gpu** | mean TTFT | mean TPOT |
|---|---|---|---|---|---|
| static 8 workers (32 GPU) | 1203 | 32.0 | **37.6** | 217 | 12.5 |
| planner — default (`hybrid_180_5`) | 1183 | 13.5 | **87.7** | 458 | 36.0 |

The default planner already gives **2.3×** the static goodput-per-GPU — same goodput, but it
scales the fleet to an average of ~13.5 GPUs instead of pinning 32.

## Sweep config

`run_smart_search`, `goal.target = goodput_per_gpu`. Deployment + router + workload + SLA
fixed; only the planner knobs swept (full trace, open-loop):

```yaml
search_space:
  model_name: MiniMaxAI/MiniMax-M2.5
  hardware_sku: b200_sxm
  backend: [vllm]
  deployment_mode: [agg]
  gpu_budget: 32
  context_length: 131072
  parallel_configs:
    - {tp: 4, moe_ep: 4, replicas: 8}           # fixed; planner scales replicas 1..8
  agg_max_num_batched_tokens: [16384]
  agg_max_num_seqs: [512]
  router_mode: [kv_router]                       # fixed tuned router
  overlap_score_credit: [1.0]
  prefill_load_scale: [4.0]
  router_temperature: [0.0]
  # swept: the planner
  planner_scaling_policy: [throughput_180_5, throughput_600_5, load_180_5,
                           load_180_10, hybrid_180_5, hybrid_600_5]
  planner_load_sensitivity: [aggressive, default, conservative]
  planner_fpm_sampling: [small, default, large, fine]
workload:
  trace_path: <toolagent_trace.jsonl>            # open-loop: no replay_concurrency
  trace_format: mooncake
goal:
  target: goodput_per_gpu
  sla: {ttft_ms: 2000, itl_ms: 50}
sweep:
  max_rounds: 6
  candidates_per_round: 8
  parallel_evals: 8
```

## Result — goodput-per-GPU by planner-policy family

The dominant factor is the **scaling-policy family** (48 trials → 19 distinct configs):

| policy family | goodput/gpu | avg_gpu | goodput | mean TTFT | mean TPOT |
|---|---|---|---|---|---|
| **`load_*` (reactive)** | **~117–121** | ~8 | ~950–1000 | ~1100–1250 | ~73–84 |
| `hybrid_*` | ~82–86 | ~14 | ~1180 | ~440–460 | ~34–37 |
| `throughput_*` (predictive) | ~59 | ~20 | ~1190 | ~320–370 | ~23–28 |

**Best:** `load_180_10` → **goodput/gpu ≈ 121** (avg_gpu 8.3, goodput 999).

**Static 37.6 → default planner 87.7 → optimized planner ≈ 121** — the tuned planner is
**3.2× static** and **+38% over the default planner**.

## All searched configurations (full-trace sweep)

Every distinct planner config the Vizier sweep evaluated (48 trials → 19 distinct), best
goodput-per-GPU first. `load_scaling_down_sensitivity` 70/80/90 = aggressive/default/conservative;
`max_num_fpm_samples` is the FPM budget (small=32, default=64, large/fine=128) and **only affects
the predictive throughput-scaling policies** — for the `load_*` policies it is inert (so its
spread within the `load_180_10` family is noise).

| goodput/gpu | avg_gpu | goodput | ttft_ms | tpot_ms | planner_scaling_policy | load_scaling_down_sensitivity | max_num_fpm_samples |
|---|---|---|---|---|---|---|---|
| 121.0 | 8.26 | 999 | 1121 | 77.8 | load_180_10 | 90 | 128 |
| 119.1 | 7.76 | 924 | 1249 | 83.8 | load_180_10 | 90 | 32 |
| 117.0 | 8.09 | 946 | 1207 | 81.6 | load_180_10 | 70 | 32 |
| 116.7 | 8.43 | 984 | 1097 | 77.1 | load_180_10 | 80 | 64 |
| 116.6 | 8.03 | 936 | 1248 | 83.5 | load_180_10 | 90 | 64 |
| 116.4 | 8.30 | 966 | 1181 | 80.1 | load_180_10 | 70 | 64 |
| 115.5 | 8.77 | 1013 | 1032 | 72.8 | load_180_10 | 70 | 128 |
| 115.4 | 8.83 | 1019 | 1041 | 73.3 | load_180_10 | 80 | 128 |
| 115.3 | 8.22 | 948 | 1193 | 80.4 | load_180_5 | 90 | 32 |
| 115.3 | 8.10 | 934 | 1245 | 83.1 | load_180_10 | 80 | 32 |
| 113.7 | 8.57 | 974 | 1163 | 79.0 | load_180_5 | 90 | 128 |
| 111.0 | 8.87 | 984 | 1117 | 76.8 | load_180_5 | 80 | 64 |
| 109.6 | 8.89 | 975 | 1113 | 76.1 | load_180_5 | 80 | 32 |
| 108.4 | 9.42 | 1021 | 1014 | 71.3 | load_180_5 | 70 | 128 |
| 85.9 | 13.75 | 1181 | 458 | 36.5 | hybrid_180_5 | 70 | 64 |
| 82.0 | 14.10 | 1155 | 514 | 37.3 | hybrid_600_5 | 90 | 128 |
| 81.8 | 14.51 | 1188 | 433 | 33.5 | hybrid_180_5 | 80 | 128 |
| 59.5 | 20.17 | 1199 | 319 | 23.1 | throughput_180_5 | 90 | 32 |
| 59.1 | 20.11 | 1189 | 370 | 27.5 | throughput_600_5 | 90 | 128 |

## Takeaways

- **Reactive load scaling ≫ hybrid ≫ predictive throughput scaling** for goodput-per-GPU.
  Predictive throughput scaling *over-provisions* (it pre-scales to avoid SLA misses → avg_gpu
  ~20 → worst goodput/gpu); reactive load scaling tracks the actual load and scales down hard
  (avg_gpu ~8).
- The optimizer drives latency **to the SLA edge** to minimize GPUs: the best `load_*` config
  runs TPOT ~78 ms (near the 50 ms ITL bound) and TTFT ~1100 ms (under the 2000 ms bound),
  letting goodput dip to ~1000 while avg_gpu drops to ~8 → goodput/gpu peaks. The conservative
  predictive policies keep latency low but waste GPUs.
- **Caveat:** `load_*` policies don't use `planner_fpm_sampling` (it only affects predictive
  throughput scaling), so the `sens`/`fpm` spread *within* the `load_180_10` family (117–121) is
  mostly mocker noise (~5%). The robust conclusion is the **policy family** (`load_180_10`), not
  a precise `sens`/`fpm` setting.

## Reproduce

```bash
python -m dynamo.profiler.spica --config path/to/planner-sweep.yaml
```

Notes: the planner path needs the `aic-forward-pass` binding; a per-throughput-interval
load-predictor sub-sweep runs first (forecast-loss winner pinned per interval). Static and
default-planner baselines are run via `dynamo.replay.run_trace_replay` (static: no
`planner_config`; planner: `planner_config` + the SLA). Same deployment as
[Router End-to-End Latency Sweep](router-e2e-latency-sweep.md).
