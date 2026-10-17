# CloudAI MPC v3 — what changed from v1 and why

v3 replaced v1 as the leaderboard entry `cloudai-mpc` on 2026-09-30 and is the
only MPC adapter in the tree (Match Config type `cloudai_mpc_v3`); the v1
adapter it was rebuilt from was removed on 2026-10-02 once the comparison was
complete.

| | Path |
|---|---|
| v3 adapter | `src/autoscaling_arena/adapters/cloudai_mpc_v3.py` |
| v3 tests | `tests/test_cloudai_mpc_v3.py` (`ARENA_AIS_TESTS=1` also validates the capacity model against the AIS database) |
| Registration | `adapters/__init__.py`, `match_config.py` (`_CLOUDAI_MPC_TYPES`, key set, auto-filled keys), `match_runner.py` (factory) |
| Leaderboard entries | `configs/match.sim.agg.all-datasets.yaml`, `configs/match.sim.disagg.all-datasets.yaml` (entry `cloudai-mpc`) |

## Diagnosis that motivated the rewrite

Published v1 ranked 4.25 of 6 on goodput per GPU-hour in agg (4.00 in
disagg). The per-workload timelines showed one failure signature: on a
constant 5.4 rps trace v1 cycled the fleet 1 → 8 → 1 every ~60 s, paying for
more GPUs than the static baseline while delivering a quarter of its goodput.
The forecaster was not the cause. v1's plant model (AIS prefill time of one
batch, `queue / n * avg_service_time`, fixed `max_rps_per_worker = 20`)
declared every fleet feasible, so heuristic scale-down paths dropped to one
replica whenever the backlog cleared; a 30 s cold start against a 25 s
horizon let the reactive floor add one replica per tick during the wait; and
GPU cost 100 per replica-step against a weak violation term meant the
optimizer only bought a replica once predicted TTFT exceeded 4.8 s. ITL was
never modelled and disagg applied one count to both pools.

## Component-by-component diff

| Component | v1 | v3 |
|---|---|---|
| Plant / capacity model | `predict_ttft(isl, batch)` for one prefill batch; queue delay `queue / n * avg_service_time`; `max_rps_per_worker` constant | `CapacityModel`: per-worker rate `mu = 1 / (T_p + osl * T_d(c_sat) / c_sat)` from AIS `predict_prefill` and `predict_decode`, evaluated at the saturated concurrency `c_sat = min(0.9 * max_kv_tokens / (isl + osl), max_num_seqs)`; online multiplicative bias from completions during saturated ticks. Queue balance `q' = max(0, q + (lambda - n mu) dt)`, `TTFT = q / (n mu) + T_p` |
| Actuation lag | none; starting workers counted as capacity; horizon 5 x 5 s | order book of pending replicas with ready time `t + cold_start_s`; plan horizon `ceil(cold_start / dt) + horizon` = 11 steps; each candidate evaluated on its active-fleet schedule; scale-down cancels pending orders first |
| Reactive floor | `ceil(queue / threshold)`, `current + 1` per tick, RPS-jump floor, oscillation hold | queue-spike floor = smallest target whose plan clears the backlog; never below the provisioned fleet while orders are pending; RPS-jump and oscillation floors removed |
| Decision paths | short-circuit, eager scale-down (defaulted to `min_replicas` when no smaller fleet passed), scheduled scale-down, adaptive `2**headroom` GPU cost | deleted; every decision goes through the candidate cost |
| Scale-down rule | penalties with headroom decay | admitted only if no orders pending, the reduced fleet keeps utilisation <= 0.75 and TTFT < 0.5 SLO at every plan step under the 80th-percentile forecast, and that held for one cold start of ticks |
| Objective | `violation_weight * ((TTFT - SLO)/SLO)^2 + gpu_cost * n` plus ad-hoc terms; ITL absent | Dinkelbach surrogate of the ratio metric: `-(sum_k served_k * s(TTFT_k) * s(ITL_k) - rho * provisioned_gpus * dt)`, `rho` = trailing goodput per GPU-second (EMA `rho_window_s`, floor `rho_min`); ITL = `T_p + T_d(c_sat)` under saturation, processor-sharing form otherwise; switching penalties default 0 |
| Forecaster | Holt on 5 s bins with residual-std snap | `_ArrivalForecaster`: 15 s bins, EWMA level with time constant = cold start, reset at 2.5 Poisson sigma, least-squares trend only when > 2 SE; mean and 80th percentile per step; ISL/OSL/KV-hit through 60 s EWMAs |
| Disagg | one count for prefill and decode | `(n_p, n_d)` over the 8 x 8 grid: `mu_p = 1 / (T_p + T_xfer)`, `mu_d = c_sat / (osl T_d(c_sat))`, series throughput `min(n_p mu_p, n_d mu_d)`, per-pool order book, floor and scale-down timer, GPU cost `n_p * prefill_gpus + n_d * decode_gpus` |

## Config keys

Accepted: the `cloudai_mpc` key set (see `cloudai_autoscalers.md`) plus `cold_start_s` (default 30),
`rho_min` (0.1), `rho_window_s` (60), and for disagg
`prefill_gpus_per_worker`, `kv_transfer_gbps`, `kv_bytes_per_token`, which
the loader fills from the prefill engine when omitted. Accepted but ignored
(kept for drop-in compatibility): `max_rps_per_worker`, `violation_weight`,
`gpu_cost_per_s`, `headroom_decay_ticks`, `smoothing_alpha`,
`reactive_rps_jump_ratio`, `kv_pressure_threshold`. The published entries set
`scale_up_penalty: 0`, `scale_down_penalty: 0`, `reactive_queue_threshold: 6`,
`cold_start_s: 30`, `max_replicas: 8`.

## Measured effect of each step

Each point was added on top of the previous one and A/B-tested on all 14
workloads, one repetition (seed 0), against the published v1 rows of the same
seed. Mean goodput per GPU-hour, agg:

| Step | All 14 | Golden 6 | Mean rank of 6 |
|---|---|---|---|
| Published v1 | 1.132 | 0.660 | 4.25 |
| 1 capacity model | 1.137 | 0.603 | 4.21 |
| 2 order book | 1.127 | 0.604 | 4.29 |
| 3 paths removed + scale-down rule | 1.294 | 0.750 | 3.00 |
| 4 ratio objective with ITL | 1.334 | 0.814 | 2.79 |
| 5 binned forecaster | 1.357 | 0.925 | 2.71 |

Disagg after step 6: 0.729 → 0.929 (rank 4.00 → 3.14).

Published leaderboard with 3 repetitions: agg 1.400 (v1 1.138), mean rank
2.61, first on both agg tables; disagg 0.951 (v1 0.727), mean rank 3.18,
third behind reactive and planner.

## Known gaps

* `rho` starts at its floor, so GPUs look cheap in the first minute; flat and
  composition-shift over-buy early in agg.
* The 15 s binning lags moderate steps: staircase and flash_crowd in agg are
  below step 4's values.
* In disagg v3 sometimes runs a third worker on the low-load Golden Set
  traces where 1P1D scores best on the ratio.
