<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# H200 Counterpart: Kimi-K2.5 DynoSim Data

This is a companion data note for the main
[DynoSim blog draft](./README.md). It follows the same section structure where
we have H200 data, but keeps the H200 results separate from the B200 MiniMax and
Planner figures in the main post.

## Shared H200 Setup

Unless noted otherwise, these runs use the full 23,608-request Mooncake FAST25
`toolagent_trace.jsonl` trace with `trace_format=mooncake`,
`trace_block_size=512`, `moonshotai/Kimi-K2.5`, vLLM 0.19.0 timing through AIC,
and `h200_sxm`. Engine settings are `block_size=512`,
`num_gpu_blocks=16384`, and `max_num_batched_tokens=16384`. MoE fields are
plumbed as `moe_tp_size=tp_size`, `moe_ep_size=1`, and
`attention_dp_size=1`.

Data and plotting scripts live in
[scripts/h200_kimi_counterpart](./scripts/h200_kimi_counterpart/README.md).

## 1. Architecture: Same Twin, Different Engine Profile

The H200 runs exercise the same DynoSim structure as the main draft: workload
trace, scheduler simulation, AIC-backed pass timing, Router decisions, and
offline replay metrics. The difference is the model/system profile: Kimi-K2.5
on H200 instead of the B200 MiniMax router and KVBM setup.

## 2. Simulating The Dynamo Digital Twin

### 2.1 Single Engine Simulation

There is no H200 hardware-fidelity counterpart in this data set. The main post's
single-engine fidelity section compares hardware, mocker, and AIC on B200. For
H200, we only have simulation/AIC-backed replay data, not live H200 silicon
measurements for the same workload.

### 2.2 Multi Engine Simulation: Router

The H200 router sweep uses eight aggregated workers at TP=4, so the total budget
is 32 GPUs. It compares round robin, KV Router, and KV Router with AIC prefill
load modeling over replay concurrencies 32, 64, 128, 256, and 512.

![H200 Kimi router counterpart](./images/h200_kv_router_exp.png)

KV Router raises average prefix reuse from `0.413` to `0.492` and cuts TTFT by
46-58% versus round robin across the sweep. At c=512, round robin reaches
`34.64 TPS/GPU` with `6832.97 ms` TTFT; KV Router reaches `39.01 TPS/GPU` with
`2976.70 ms` TTFT. Adding router-side AIC prefill-load modeling nudges c=512
throughput to `39.42 TPS/GPU` and prefix reuse to `0.502`, but TTFT is slightly
higher than KV Router alone in this aggregate-worker sweep.

### 2.2 Multi Engine Simulation: KVBM Diagnostic

We also ran the baseline-vs-G2 KVBM setup on H200 Kimi, but it should be treated
as a diagnostic rather than a result. With `num_gpu_blocks=16384`, the workload
does not create enough G1 pressure for the G2 host-memory tier to matter, so the
baseline and G2 lines overlap.

![H200 Kimi KVBM diagnostic](./images/h200_kvbm_g2_diagnostic.png)

This confirms that the current H200 Kimi setup is not a useful KVBM benefit
figure. Yongming's KVBM-specific run should remain the source for that section.

### 2.2 Multi Engine Simulation: Planner

No new Kimi/H200 Planner sweep was run here. The main blog already uses H200 for
the Planner experiments, but with Qwen3-32B at TP=2 rather than Kimi-K2.5.

## 3. Optimization And Discovery With DynoSim

The H200 optimizer run uses the same replay-optimization idea as section 3.1 of
the main post: block-coordinate search over TP shape, worker split, and router
setting. This run used a 16-GPU budget, `arrival_speedup_ratio=0.25`, objective
`throughput`, and SLA constraints of mean TTFT <= 4000 ms, mean TPOT <= 75 ms,
and mean E2E <= 20000 ms.

![H200 Kimi optimizer candidates](./images/h200_optimizer_candidates.png)

No row is feasible under the strict 4 s TTFT SLA. The best near-miss is only
58.07 ms over the TTFT threshold:

| Field | Value |
|---|---:|
| `prefill_tp / decode_tp` | `2 / 1` |
| `prefill_workers / decode_workers` | `5 / 6` |
| `prefill_load_scale` | `0.5` |
| Output throughput | `303.13 tok/s` |
| Prefix reuse | `0.5383` |
| Mean TTFT | `4058.07 ms` |
| Mean TPOT | `58.57 ms` |
| Mean E2E | `13319.99 ms` |

The optimizer artifact still names this field `overlap_score_weight`, but this
code path maps that backward-compatible value to `prefill_load_scale`.

## 3.3 Simulation In The Production Routing Loop

The H200 aggregate-worker router sweep includes an AIC prefill-load line, but it
is not the same as the B200 fixed-layout result in the main post. In this H200
sweep, AIC prefill-load modeling mostly improves throughput and prefix reuse
slightly while increasing TTFT by a few percent relative to KV Router alone.
That makes it useful as a counterpart data point, not as a replacement for the
main post's router-side AIC TTFT reduction example.

## 4. How To Use This Data

Use this README as an H200 appendix or parallel working note. The clean claims
from this data are:

- H200 Kimi router simulation reproduces the same qualitative Router story:
  cache-aware routing improves prefix reuse, throughput per GPU, and TTFT versus
  round robin.
- The current H200 Kimi KVBM setup is not a valid G2 benefit plot because G1 is
  not pressured.
- The H200 optimizer run found a strong near-feasible layout under a strict
  4 s TTFT SLA, but not a feasible one.

The source artifacts are committed separately so the main draft can choose
whether to reference them, replace the B200 figures later, or keep them as an
appendix.
