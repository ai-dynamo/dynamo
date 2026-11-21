# CloudAI autoscalers for the Autoscaling Arena

Two NVIDIA CloudAI planners join the Arena roster (Planner, KEDA port,
reactive, static, Jev) as first-class Match Config autoscaler types:

| Leaderboard entry | Class | Module | Approach |
|---|---|---|---|
| `cloudai-mpc` | `CloudAIMPCV3Autoscaler` (type `cloudai_mpc_v3`) | `adapters/cloudai_mpc_v3.py` | Model Predictive Control: a saturated-capacity plant model over AIS (AISimulate) prefill/decode predictions, a cold-start order book, a model-based scale-down rule, a goodput-per-GPU (Dinkelbach) objective with ITL, and a binned arrival forecaster; independent prefill/decode pools in disagg. Design and measured effect of each step: [cloudai_mpc_v3.md](cloudai_mpc_v3.md). |
| `cloudai-rl` (agg) | `CloudAIRLAutoscaleLSTM` (type `cloudai_rl_lstm`) | `adapters/cloudai_rl_lstm.py` | Offline-RL (CQL) policy over 27 current-interval replay-telemetry features, the last ten observations through a 32-unit LSTM, nine relative deltas with exact bounded-target masks, Q-only. Checkpoint `checkpoints/cql_autoscaler_best_v3_lstm.pt`. |
| `cloudai-rl` (disagg) | `CloudAIRLDisaggAutoscaler` (type `cloudai_rl_disagg`) | `adapters/cloudai_rl_disagg.py` | Offline-RL (CQL) two-pool policy over 37 current-interval telemetry features plus its last requested targets, ten-step LSTM, one delta per pool from an additive two-head critic (prefill 1-16, decode 1-8), infeasible deltas masked, Q-only. Checkpoint `checkpoints/cql_autoscaler_best_disagg_v3_lstm.pt`. |

Shared inference code lives in `adapters/cql_common.py` (residual Q head,
LSTM encoder, two-head critic, bounded history window, replay-telemetry adapter
base). Every adapter follows the Arena's rival-adapter contract: a no-op AIS
regression bootstrap (`supports_ais_bootstrap = False`) and a replica baseline
taken from the lifecycle-aware fleet (active + starting workers), so a hold
tick never cancels a scale-up still in flight.

## Dependencies

```bash
pip install -e '.[cloudai]'        # numpy (MPC) and torch (CPU inference for RL)
# the two 17 MB CQL exports ship in checkpoints/ (plain Git)
```

Both adapters run inside Dynamo's replay runtime through the Arena's
simulation runner (`runners/sims.py`); set it up as described in
[getting-started.md](getting-started.md) and [replay-integration.md](replay-integration.md).
The RL planner additionally needs a Dynamo build whose
`run_mocker_trace_replay()` exposes `telemetry_callback`; the runner refuses to
start it on an older build.

## CloudAI MPC (`cloudai_mpc_v3`)

Every 5 s the controller forecasts arrivals (15 s bins, EWMA level with the
cold start as time constant, Poisson reset, significant trends only) and
scores candidate fleets over a short horizon with a plant model whose service
rate per worker is `mu = 1 / (T_p + osl * T_d(c_sat) / c_sat)`, evaluated at the
worker's saturated concurrency (`c_sat` from the KV budget and `max_num_seqs`,
`T_p`/`T_d` from the AIS session's `predict_prefill`/`predict_decode`). Capacity ordered at
`t` lands at `t + cold_start_s`; scale-downs cancel the newest pending orders
first and are admitted only when the reduced fleet keeps utilisation below 0.75
and TTFT under half the SLO for a full cold start under the 80th-percentile
forecast. The objective is good requests minus `rho x GPU-seconds`, with `rho`
tracking the trailing goodput per GPU-second (Dinkelbach), so one saturated
replica beats two idle ones. In disagg, prefill and decode pools are sized
independently over the `(n_p, n_d)` grid with the series throughput
`min(n_p mu_p, n_d mu_d)` and a KV-transfer term in `mu_p`.

Keys: `forward_model` (`ais` or `roofline`), `poll_interval_s`, `horizon`,
`slo_ttft_ms`, `slo_itl_ms`, `min_replicas`, `max_replicas`, `cold_start_s`,
`rho_min`, `rho_window_s`, `scale_up_penalty`, `scale_down_penalty`,
`reactive_queue_threshold`, and in disagg `prefill_gpus_per_worker`,
`kv_transfer_gbps`, `kv_bytes_per_token` (filled from the prefill engine when
omitted). The remaining `cloudai_mpc` keys (`violation_weight`, `gpu_cost_per_s`,
`headroom_decay_ticks`, `max_rps_per_worker`, `smoothing_alpha`,
`reactive_rps_jump_ratio`, `kv_pressure_threshold`, `gpus_per_worker`) are
accepted for compatibility with the published configs. `forward_model: ais`
expands into the canonical AIS identity (`model`, `system`, `backend`,
`backend_version`, `tp`, MoE sizes, `worker_type`) of the prefill (or aggregate)
engine, the same identity Match Config renders into that engine's
`ais_perf_config`.

## CloudAI RL planner (`cloudai_rl_lstm`, `cloudai_rl_disagg`)

The RL planner is served from `cql_trainer` exports (Conservative Q-Learning on
offline Arena experience; the training code is not part of this repository).
Both topologies observe Dynamo's `dynamo.replay.telemetry.v1` samples
directly: the replay emits one sample every five simulated seconds *before* the
scaling callback at the same instant, and the adapter encodes it with a
verbatim port of the trainer's feature code.

- **agg** (`CloudAIRLAutoscaleLSTM`): 27 current-interval features (demand and
  completed ISL/OSL, waiting and running requests, KV usage, active / starting /
  draining fleet, startup ETAs, trends, SLO-relative TTFT and ITL, validity
  flags, counts, shape age, interval and observation-validity flags), ten-step
  history with left zero padding and a zero hidden state per window. Nine
  relative deltas `[-8,-4,-2,-1,0,1,2,4,8]` on a 1-8 fleet with exact
  bounded-target masks (one smallest-magnitude delta per distinct executable
  target), chosen Q-only. Keys: `checkpoint_path` (required), `poll_interval_s`
  (must be 5), `slo_ttft_ms` / `slo_itl_ms` (2000 / 50), `min_replicas` /
  `max_replicas` (1 / 8), `cold_start_s` (aggregate engine runtime),
  `hidden_dim`, `num_blocks`.
- **disagg** (`CloudAIRLDisaggAutoscaler`): the same idea per pool (37 features
  including the targets the policy last requested), one delta in
  `[-4,-2,-1,0,1,2,4]` per pool relative to the last requested target
  (prefill 1-16, decode 1-8), infeasible deltas masked rather than clamped,
  argmax per head of the additive critic `Q(s, aP, aD) = qP(s, aP) + qD(s, aD)`.
  Keys: `checkpoint_path` (required), `poll_interval_s` (5), `slo_ttft_ms` /
  `slo_itl_ms` (2000 / 50), `min_prefill` / `max_prefill` (1 / 16),
  `min_decode` / `max_decode` (1 / 8), `cold_start_s` (engine runtime),
  `smoothing_alpha` (0.4), `prefill_gpus_per_worker` / `decode_gpus_per_worker`
  (engine GPU counts), `hidden_dim`, `num_blocks`.

Both adapters refuse a checkpoint whose training settings (SLO, bounds, cold
start, replica normalization, the decode engine's KV capacity, GPUs per worker)
differ from the serving configuration, require
`backend.replay.telemetry_sample_interval_s: 5`, raise when a decision has no
current telemetry sample or the runtime fleet differs from the requested
targets, and write an `rl-decisions.jsonl` artifact per run (encoded state, Q
values, history length, current and target counts). Replaying a run's
`telemetry.jsonl` through the encoder must reproduce those states exactly; this
offline/serving parity was checked against the trainer's own code on every
published run (2026-10-02).

## Leaderboards and tables

- `configs/match.sim.agg.all-datasets.yaml` and
  `configs/match.sim.disagg.all-datasets.yaml`: the seven-planner sweeps
  (Planner, Jev, KEDA, reactive, static, `cloudai-mpc`, `cloudai-rl`) over the
  eight registry workloads and the six Astra GPT-OSS Golden Set traces, 3
  repetitions, SLA TTFT 2000 ms / ITL 50 ms, ranked by `goodput_per_gpu`.
- `scripts/rank_tables.py results.json`: mean-rank and pairwise-win-rate tables
  on `goodput_per_gpu` for all workloads and for the Golden Set.
- `scripts/evaluation_table.py --results agg=... --results disagg=... --out
  evaluation.md`: the final summary tables (every column as `all (golden)`,
  goodput per GPU-second) plus an `.xlsx` workbook with per-workload goodput,
  rank matrices and the long-form data (needs the `xlsx` extra).
- `configs/match.sim.cloudai.example.yaml`: a registry-only agg example of the
  roster that validates in CI.

The published tables live in [`results/`](../results/README.md) with the
exact commands that regenerate them; raw results JSON, HTML reports and
per-run artifacts land under the gitignored `runs/`.
