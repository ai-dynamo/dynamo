---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Router End-to-End Latency Sweep
subtitle: Experimental Spica search over KV router settings for MiniMax-M2.5 on B200 GPUs
---

> [!WARNING]
> **Experimental.** These replay results characterize one software snapshot and workload. Do not
> treat them as production capacity guidance or a performance commitment. Spica's search behavior
> and output may change without a standard deprecation period.

Reproduce the DynoSim router experiment, then use Spica's smart sweep to **find the
KV-router configuration that minimizes mean end-to-end latency** on the Mooncake toolagent
trace, and validate the winner against the default-router baseline on the full trace.

## Setup

| | |
|---|---|
| model | `MiniMaxAI/MiniMax-M2.5` (MLA + MoE, FP8) |
| hardware | `b200_sxm`, **TP4 = TEP** (`aic_tp_size=4, aic_moe_tp_size=1, aic_moe_ep_size=4`) |
| backend | vLLM 0.19.0, **8 aggregated workers** (32 GPUs) |
| engine | `max_num_batched_tokens=16384`, `max_num_seqs=512`, `block_size=64` |
| workload | Mooncake FAST'25 **toolagent trace** (23,608 reqs, `traces/mooncake_fast25/toolagent_trace.jsonl`) |
| replay | offline, **closed-loop** `replay_concurrency=32`, `trace_block_size=512` |
| goal | minimize `mean_e2e_latency_ms` (no SLA) |
| Dynamo source revision | `c7378f8e` |

The deployment is **fixed** (pinned `parallel_configs`, fixed engine batching, fixed
concurrency) so any change in e2e latency is attributable to the **router** alone.

## Sweep config

Searched with `run_smart_search` (Vizier GP-bandit), `goal.target = e2e_latency`. Search
runs on a 2k-request subset (fast, representative — same long-context mix); the winner is
then validated on the full trace.

```yaml
search_space:
  model_name: MiniMaxAI/MiniMax-M2.5
  hardware_sku: b200_sxm
  backend: [vllm]
  deployment_mode: [agg]
  gpu_budget: 32
  context_length: 131072                       # covers the trace's 126k-token max
  parallel_configs:
    - {tp: 4, moe_ep: 4, replicas: 8}           # fixed: TEP, 8 workers x TP4
  agg_max_num_batched_tokens: [16384]
  agg_max_num_seqs: [512]
  planner_scaling_policy: [disabled]            # static — no autoscaling
  # swept: the router
  router_mode: [kv_router, round_robin]
  overlap_score_credit: [0.0, 0.5, 1.0]         # kv-router hard-caps this at 1.0
  prefill_load_scale: [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
  router_temperature: [0.0, 0.2, 0.5, 1.0]
workload:
  trace_path: <toolagent_trace.jsonl>           # 2k subset for the search; full trace to validate
  trace_format: mooncake
  replay_concurrency: 32
goal:
  target: e2e_latency
sweep:
  max_rounds: 6
  candidates_per_round: 10
  parallel_evals: 10
```

## Result — full trace, c=32 (the validation)

| config | mean e2e (ms) | mean TTFT (ms) | mean TPOT (ms) | throughput (tok/s) | prefix reuse |
|---|---|---|---|---|---|
| `round_robin` | 2317* | 263* | 11.0* | 2483* | — |
| **kv_router — default** | 2034 | 142 | 16.5 | 2861 | 0.494 |
| **kv_router — optimized** | **1961** | **122** | **12.1** | **2965** | **0.554** |

`*` round_robin numbers are from the 2k subset (worst on every latency metric; not re-run full).

**Optimized vs default kv_router (full trace):** e2e **−3.6%**, TTFT **−14%**, TPOT **−27%**,
throughput **+3.6%**, prefix reuse **+12%** (0.494 → 0.554).

**Winning config:** `kv_router`, **`overlap_score_credit=1.0`**, **`prefill_load_scale=32`**,
**`router_temperature=0.0`**.

### Why `prefill_load_scale` matters most (and where it saturates)

`prefill_load_scale` was best at the original ceiling (4.0), so the search space was extended
to 32. Higher values keep helping but **saturate around 16–32** (2k subset, overlap=1.0, temp=0):

| prefill_load_scale | 4 | 8 | 16 | 32 |
|---|---|---|---|---|
| mean e2e (ms) | 2180 | 2155 | 2141 | 2135 |
| Δ vs previous | — | −25 | −14 | −6 |

Going beyond 32 (64/128) is expected to give negligible further gain.

## All searched configurations (2k-subset search)

Every distinct config the Vizier sweep evaluated (2k subset, c=32), best mean-e2e first.
`n` = how many of the 48 trials landed on it (Vizier concentrates samples on the optimum).
`round_robin` ignores the kv-router weights (shown as —).

| mean_e2e_ms | ttft_ms | tpot_ms | tput_tok/s | n | router_mode | overlap_score_credit | prefill_load_scale | router_temperature |
|---|---|---|---|---|---|---|---|---|
| 2135 | 162 | 13.2 | 2707 | 13 | kv_router | 1.0 | 32.0 | 0.0 |
| 2141 | 163 | 14.8 | 2726 | 15 | kv_router | 1.0 | 16.0 | 0.0 |
| 2147 | 162 | 13.5 | 2698 | 4 | kv_router | 0.5 | 32.0 | 0.0 |
| 2155 | 166 | 15.2 | 2695 | 3 | kv_router | 1.0 | 8.0 | 0.0 |
| 2162 | 166 | 14.1 | 2680 | 2 | kv_router | 0.5 | 16.0 | 0.0 |
| 2180 | 170 | 17.5 | 2660 | 1 | kv_router | 1.0 | 4.0 | 0.0 |
| 2183 | 169 | 17.2 | 2657 | 1 | kv_router | 0.5 | 8.0 | 0.0 |
| 2186 | 181 | 13.1 | 2661 | 1 | kv_router | 1.0 | 16.0 | 0.2 |
| 2199 | 180 | 13.5 | 2646 | 1 | kv_router | 1.0 | 32.0 | 0.2 |
| 2203 | 183 | 13.3 | 2624 | 1 | kv_router | 0.5 | 8.0 | 0.2 |
| 2215 | 184 | 14.4 | 2596 | 1 | kv_router | 1.0 | 8.0 | 0.2 |
| 2236 | 197 | 18.4 | 2595 | 1 | kv_router | 1.0 | 0.25 | 0.0 |
| 2239 | 184 | 16.0 | 2587 | 3 | kv_router | 0.0 | 8.0 | 0.0 |
| 2242 | 198 | 14.2 | 2589 | 1 | kv_router | 0.0 | 4.0 | 0.2 |
| 2255 | 218 | 16.2 | 2580 | 1 | kv_router | 1.0 | 0.0 | 0.2 |
| 2260 | 186 | 17.8 | 2572 | 1 | kv_router | 0.0 | 4.0 | 0.0 |
| 2265 | 219 | 13.3 | 2566 | 1 | kv_router | 0.5 | 16.0 | 0.5 |
| 2268 | 211 | 13.1 | 2555 | 1 | kv_router | 1.0 | 8.0 | 0.5 |
| 2292 | 231 | 13.2 | 2531 | 1 | kv_router | 0.0 | 4.0 | 0.5 |
| 2306 | 220 | 13.8 | 2504 | 1 | kv_router | 0.0 | 8.0 | 0.5 |
| 2317 | 263 | 11.0 | 2483 | 5 | round_robin | — | — | — |
| 2352 | 255 | 12.9 | 2471 | 1 | kv_router | 0.5 | 32.0 | 1.0 |

## Takeaways

- **kv_router ≫ round_robin** for e2e/TTFT (cache-affine placement → higher prefix reuse →
  less prefill recompute → lower TTFT), matching the DynoSim blog direction.
- **`prefill_load_scale` is the dominant router knob** here: pushing it up (to ~16–32) spreads
  prefill load off hot workers, cutting both TTFT and (notably) TPOT. It saturates ~16–32.
- **`overlap_score_credit=1.0`** (its hard max) and **`router_temperature=0.0`** (fully
  deterministic, cache-affine routing) are best.
- The smart sweep cut mean e2e **−3.6%** and TPOT **−27%** over the default kv_router with no
  change to the deployment — purely router tuning.

## Reproduce

```bash
python -m dynamo.profiler.spica --config path/to/router-sweep.yaml
```

Run the winning router configuration against the full trace with
`dynamo.replay.run_trace_replay`. The AI Configurator performance model needs the
`aic-forward-pass` binding. To suppress router logs, set `RUST_LOG=error`.
`prefill_load_scale` values up to 32 require the extended `SEARCH_CHOICES` in `config.py`.
