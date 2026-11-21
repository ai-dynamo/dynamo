<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# cloudAI-MPC

cloudAI-MPC is a model predictive controller for offline Gym matches. Select
`type: mpc` in a Match Config; [match.mpc.example.yaml](../configs/match.mpc.example.yaml)
runs it beside Planner and the CPU baselines. The implementation is
`MPCAutoscaler` in `adapters/mpc.py`; controller tests live in
`tests/test_mpc.py`. It needs no trained checkpoint.

```bash
python scripts/run_match_config.py configs/match.mpc.example.yaml --validate-only --print-matrix
python scripts/run_match_config.py configs/match.mpc.example.yaml
```

> [!IMPORTANT]
> cloudAI-MPC reads measured outcomes from replay telemetry and needs the
> `completion_feedback` block described under [Runtime requirement](#runtime-requirement).
> The AISimulate release pinned by Dynamo does not emit that block yet.

## Decision model

The capacity model combines prefill time, decode step time and the KV-cache
concurrency limit. For aggregate workers, the sustainable request rate is:

```text
c_sat = min(0.9 * worker_kv_tokens / attention_dp / (isl + osl), rank_max_num_seqs)
worker_mu = attention_dp / (prefill_seconds + (osl - 1) * decode_step_seconds(c_sat) / c_sat)
```

Prefill produces a request's first token, so it takes `osl - 1` decode steps.

Capacity is evaluated at saturation. Offered load also enters the latency
estimate, while a bounded correction learns scheduler effects from observed
completions during saturated intervals. Arrival-rate forecasts use coarse
bins, an exponentially weighted level, change-point detection and significant
linear trends. Token lengths and cache-hit rates use smoothed observations.

A pending replica costs GPUs immediately but contributes service capacity only
after its cold-start delay. A replica still starting after that delay is late:
the plan expects it at the next control interval and does not count it in the
current one. While no replica of a pool is ready, the plan gives that pool no
serving capacity. Candidate plans cover the cold-start delay plus `horizon`
control intervals. The controller maximizes an approximation of good requests
per GPU-second using a Dinkelbach objective: predicted good requests minus
`rho * GPU-seconds`, with `rho` tracking measured recent efficiency. Both TTFT
and ITL enter the predicted good-request fraction.

Replay supplies exact cumulative terminal, completed and joint-SLO success
counts plus integrated GPU-seconds. The controller computes `rho` from the
trailing `rho_window_s` completion window once outcomes exist, including failed
requests and GPUs occupied by starting/draining workers. Until then it uses its
capacity model. This online window is not the final arrival-cohort score:
completion delays can cross decision intervals.

A deterministic reservoir retains at most 64 paired TTFT/mean-ITL observations
per telemetry interval. MPC keeps at most 256 latency residual pairs from the
last 60 seconds, comparing outcomes with the previous interval's prediction.
It learns only when the predicted and observed fleet agree and the sample has
no starting or draining workers. Candidate joint success uses these paired
residuals, blending with the model's smooth prior until 32 samples arrive.
One-token outputs obey the scoring profile's `short_output_itl` rule.
The correction remains an approximation: completed requests can have started
under an earlier fleet, and observed residuals need not transfer perfectly to
an untried fleet.

Scale-up uses the mean forecast. Scale-down requires no pending replicas and a
smaller fleet that remains below 75% utilization, half the TTFT limit and the
calibrated ITL limit under
the 80th-percentile forecast at every plan step. That check must hold on
`ceil(cold_start_s / poll_interval_s)` consecutive decisions (at least one)
before the fleet shrinks. A queue-spike
response accounts for capacity already starting. The initial fleet stays in
place until a positive-duration traffic observation is available; an empty
startup snapshot is not evidence of zero demand.

In disaggregated mode the controller evaluates separate prefill and decode
counts within the run's GPU budget. The plan keeps one backlog per stage:
prefill capacity (`prefill_count * prefill_rate`) drains arrivals and the
prefill backlog, and decode capacity (`decode_count * decode_rate`) drains
what prefill hands over and the work already waiting at decode. Work waiting
at decode does not pass through prefill again, and sustained
throughput is `min(prefill_count * prefill_rate,
decode_count * decode_rate)`. The model includes KV-transfer cost, per-pool
pending replicas and the GPU cost of each pool. Attention-DP throughput and
load use each engine's configured parallelism. Shared transfer bandwidth is
not multiplied by the rank count.

The feasible set enforces `prefill_count * prefill_gpu_cost + decode_count *
decode_gpu_cost <= backend.gpu_budget` together with the pool bounds.
The same set governs normal optimization, backlog recovery and reallocation.
If no candidate clears the backlog within the horizon, recovery selects the
feasible pair with the smallest remaining queue. A pool exchange may bypass
the ordinary scale-down hold when neither pool has pending orders and the
coupled plan improves without worsening queue or latency beyond the current
plan/SLO bounds.

## Configuration

| Setting | Purpose |
|---|---|
| `forward_model` | `ais` predicts with AISimulate for the configured engines' model, system, backend and parallelism; `roofline` is an analytical approximation |
| `slo_ttft_ms`, `slo_itl_ms` | Controller latency targets; they must equal every SLO profile in the Match Config |
| `poll_interval_s`, `horizon` | Decision cadence (a multiple of the 5-second telemetry interval) and planning horizon beyond cold start |
| `min_replicas`, `max_replicas` | Candidate bounds; applied to each pool in disaggregated mode |
| `cold_start_s` | Replica startup delay, filled from the decode or aggregate engine's `cold_start_delay_s` if omitted; a set value must equal the engine's delay |
| `scale_up_penalty`, `scale_down_penalty` | Optional switching costs |
| `rho_min`, `rho_window_s` | Efficiency weight floor (default `1e-6`) and measured trailing window; also the startup model's smoothing window |
| `reactive_queue_threshold` | Threshold for the queue-spike floor |
| `gpus_per_worker` | Aggregate/decode worker cost, filled from the engine if omitted; a set value must equal the engine's GPU count |
| `prefill_gpus_per_worker`, `kv_transfer_gbps`, `kv_bytes_per_token` | Disaggregated prefill cost and transfer parameters, filled from the engine where available; a set prefill cost must equal the engine's GPU count |

Match Config defaults are a 5-second cadence, horizon 5, bounds 1–8, and
controller targets 300 ms TTFT / 50 ms ITL. Switching penalties default to
1.5 for scale-up and 10 for scale-down through Match Config; set both to zero
when the scale-down rule should supply all hysteresis. These are generic
defaults. Set the latency targets and replica limits explicitly when adding
MPC to a case.

Match Config validates what the controller cannot handle at run time. The
controller plans against one TTFT/ITL pair, so `slo_ttft_ms` and `slo_itl_ms`
must equal every SLO profile, and a profile with an end-to-end target is
rejected. The starting fleet must fit `backend.gpu_budget`. In disaggregated
mode `max_replicas` may be larger than both pools could reach together,
because the controller chooses only among the pairs that fit the budget; an
aggregated `max_replicas` the budget cannot hold is rejected. Worker GPU costs
must match the deployed engines, so the controller scores the GPU-seconds
replay charges. The controller plans with one cold-start delay, so a
disaggregated match whose prefill and decode engines have different
`cold_start_delay_s` is rejected, and so is a `cold_start_s` that differs from
the engine's delay or an engine delay set to `null`.
`poll_interval_s` must be a multiple of the telemetry interval, and
`backend.replay.telemetry_sample_interval_s` must be 5; both are checked by
`--validate-only`.

`forward_model: ais` uses the AISimulate identity of the configured engines
and does not follow a custom `extra_args.timing_model`. Where AISimulate has
no estimate for an operating point, the controller logs a warning and uses the
analytical roofline for that point.

In disaggregated mode, separate AIS sessions predict prefill and decode using
each pool's deployed role and engine parallelism. Aggregate workers reuse one
session for both phases, so prefill and decode pools with different tensor
parallelism keep their own predictions.

The Match runner supplies the feasible `(prefill, decode)` fleets from the
resolved backend: every whole-worker pair inside the replica bounds whose GPU
cost fits `backend.gpu_budget`. The controller chooses only among them.

## Runtime requirement

The controller requires five-second native telemetry
(`backend.replay.telemetry_sample_interval_s: 5`) in which every sample carries
`completion_feedback` with `schema_version == 1`: cumulative terminal,
completed and SLO-satisfying request counts, integrated GPU-seconds, and up to
64 paired TTFT/ITL samples per interval. A replay runtime without that block
stops at the first sample with an explicit error; MPC never substitutes a
guess for a missing measurement. AISimulate emits the block from its
2026-11-19 nightly (`0.15.0-dev.202611190000000110`) onward, so the Dynamo
build in use must pin that release or a later one. The Match runner passes the scorer's
short-output policy to the controller and records the `paired-slo-feedback-v2`
policy identity in each result.
