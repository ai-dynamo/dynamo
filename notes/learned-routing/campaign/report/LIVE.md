<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

# Live GPU validation of the relative result (A13.2, A20)

This is §15 of `report/REPORT.md`, written by `runs/live/finalists/scripts/live_results.py` from
`facts/live_results.json`.

**Validation only.** These runs check whether the simulated *relative* result (the policy ranking
and the sign of M1-v2 − ramjet) holds on real GPUs. They never change a finalist, the headline,
selection or any number in §1–§14. The analysis, validity rule and wording rules were
pre-registered in `facts/live_plan.json` before the first finalist run; every number below is in
`facts/live_results.json`.

### 15.1 What ran

- **Deployment.** Qwen3-32B on vLLM 0.24.0, N = 4 workers × TP2 on one 8×H100 SXM node per job,
  aggregated, behind the Dynamo frontend whose KV router runs each frozen policy YAML (byte-identical
  to the YAML the test pass replayed). AIPerf 0.13.0 sends the cell's requests with the replay's
  arrival schedule, exact ISL, forced OSL, prefix structure and session headers; the harness's own
  A2 scorer scores them with the AIS E0, as in simulation.
- **Cells.** 6 frozen N = 4 test cells, chosen at registration for family and load-mode coverage:
  Mooncake open L2, Mooncake open L3, Mooncake closed L3, FAST25 conv. open L2, FAST25 synth. open L2, Sessions open L2.
  No AgentX (no live load generator); FAST25 and sessions in open loop only. One CRN replicate
  (k = 0, policy seed 1).
- **Policies.** default@defaults twice (to measure live noise), tuned ramjet (the val-selected best
  baseline), M1-v2 (the headline learned arm), M1, M0 and round_robin, in a pre-registered random
  order per cell. One run per (cell, policy).
- **Sim counterpart.** The frozen test pass's record of the same (cell, policy, k = 0); nothing was
  re-simulated.
- **Completeness.** 42 of 42 cell runs and 4 of 4 idle-calibration runs are valid;
  all 6 cells are complete, so there is **no shortfall**. 2 cell runs failed
  the validity rule at their first attempt: the load generator exited non-zero in its shutdown phase
  after every request had completed (a slow shared-filesystem export tripped its heartbeat watchdog).
  Each was re-run once under the registered rule and the rerun is used; the first attempts
  (default_b on FAST25 conv. open L2: 0.886; round_robin on Mooncake open L2: 3.085 good req/s) are descriptive only (reruns: 0.893; 3.087).
- **Comparisons across node allocations** (the registered analysis flags them): vs default, round_robin on Mooncake open L2; vs ramjet, round_robin on Mooncake open L2, M0 on Sessions open L2, round_robin on Sessions open L2; noise pairs, FAST25 conv. open L2, Sessions open L2. Every other comparison is within one job.

### 15.2 Goodput, live vs sim

Windowed goodput (good requests/s), live / sim at k = 0. Live default@defaults is the mean of its two
runs.

| Policy | Mooncake open L2 | Mooncake open L3 | Mooncake closed L3 | FAST25 conv. open L2 | FAST25 synth. open L2 | Sessions open L2 |
|---|---|---|---|---|---|---|
| default@defaults | 3.547 / 3.544 | 3.582 / 3.733 | 3.948 / 4.034 | 0.896 / 1.249 | 1.894 / 1.962 | 8.249 / 7.691 |
| ramjet (tuned) | 3.631 / 3.678 | 3.902 / 4.135 | 4.575 / 4.617 | 1.068 / 1.321 | 2.086 / 2.134 | 8.357 / 8.385 |
| M1-v2 | 3.921 / 3.939 | 4.720 / 4.671 | 5.225 / 5.318 | 2.072 / 2.115 | 2.159 / 2.157 | 8.359 / 8.381 |
| M1 | 3.902 / 3.860 | 4.573 / 4.677 | 5.155 / 5.148 | 1.947 / 2.137 | 2.178 / 2.235 | 8.356 / 8.380 |
| M0 | 3.636 / 3.649 | 3.859 / 3.961 | 4.497 / 4.641 | 0.995 / 1.303 | 1.917 / 2.037 | 8.365 / 8.367 |
| round_robin | 3.087 / 3.187 | 2.442 / 2.774 | 3.037 / 3.326 | 0.891 / 1.095 | 1.541 / 1.663 | 3.831 / 1.757 |

Live/sim ratios span 0.72–2.18. On the three Mooncake cells, every policy but
round_robin is at 0.94–1.01 of simulation. The ratio differs between policies in the same cell, so the departures
also move the relative gaps, in both directions: 18 of the 30 live Δ vs default@defaults (9 higher live than in simulation, 9 lower) and 13 of the 30 live Δ vs ramjet
differ from simulation by more than twice the sd of the live delta (a descriptive rule, not registered;
§15.3 lists every difference). Cell by cell:

- **FAST25 conversation.** Live goodput is 0.72–0.81 of simulation for default@defaults, ramjet, M0 and round_robin,
  but 0.91–0.98 for M1 and M1-v2. Every gain over default is larger live than in simulation (M1-v2 +0.833 vs +0.526,
  ramjet +0.171 vs +0.056), which widens the learned arms' lead over ramjet (M1-v2
  +0.663 vs +0.470). It also all but erases round_robin's deficit vs default (-0.010 live vs
  -0.132 in simulation, within live noise), because default@defaults loses the most live. vLLM
  preempted 18–24 requests per run under default@defaults, ramjet and M0 and 12 under round_robin, against 1–6
  under M1 and M1-v2, and the non-learned policies' TTFT p50 rose more over simulation (1.43–1.76× against 1.08–1.13×).
- **Sessions.** Live default@defaults runs at 1.07× its simulated goodput, while ramjet, M1-v2, M1 and M0 run at 0.997–1.000×.
  Live, 97.8–98.4% of default's in-window requests are good, against 91.6% in simulation; the four tuned policies
  serve 99.9–100% in both, at ceiling. So every tuned policy's gain over default, +0.084 to +0.086
  in simulation, is only +0.011 to +0.016 live, inside twice the sd of a live delta (0.026): on this cell the simulated gain over
  default is **not reproduced beyond live noise**. Live round_robin reaches 2.18× its simulated goodput; it is still last by
  far, but its deficit vs default shrinks from -1.099 (the ±ln 3 clip) in simulation to -0.770 live.
- **Mooncake and FAST25 synthetic.** round_robin's deficit vs default is larger live on all 4 cells, by 0.033–0.086.
  The other live − sim differences beyond twice their sd are, for Δ vs default, M1-v2 on Mooncake open L3 +0.052, M1 on Mooncake closed L3 +0.023, M1-v2 on FAST25 synth. open L2 +0.036, M0 on FAST25 synth. open L2 -0.026;
  for Δ vs ramjet, M1-v2 on Mooncake open L3 +0.068, M1 on Mooncake open L3 +0.035, M0 on Mooncake open L3 +0.032, round_robin on Mooncake open L3 -0.070, round_robin on Mooncake closed L3 -0.082, M0 on FAST25 synth. open L2 -0.038, round_robin on FAST25 synth. open L2 -0.053.

A plausible reading, not tested here: the live engine's timing differs from AIS (live TTFT is higher and
decode faster, §15.6), so the same cell runs at a different effective load or SLO point on hardware.
FAST25 conversation would then run closer to prefill saturation live, where the policies that leave more
prefill work queued lose more goodput. On sessions, live mean-ITL p50 is 0.84–0.85 of simulation for every policy, which
loosens the AIS-E0 SLA for the policies that were not already at ceiling (default@defaults and round_robin).
Absolute goodput is therefore not validated in general, and neither is the size of the gap between two
policies. What the live runs test is the ranking (§15.4) and the sign of M1-v2 − ramjet (§15.5).

### 15.3 Paired deltas and live noise

Clipped log-ratio s(a, b) = clip(ln((a + 0.001)/(b + 0.001)), ±ln 3), as in the headline. Entries are
**live (sim)**. Δ vs default uses the default runs of the same job.

| Δ vs default@defaults | Mooncake open L2 | Mooncake open L3 | Mooncake closed L3 | FAST25 conv. open L2 | FAST25 synth. open L2 | Sessions open L2 | Mean live [range] | Mean sim |
|---|---|---|---|---|---|---|---|---|
| ramjet (tuned) | +0.023 (+0.037) | +0.086 (+0.102) | +0.147 (+0.135) | +0.171 (+0.056) | +0.096 (+0.084) | +0.016 (+0.086) | +0.090 [+0.016, +0.171] | +0.084 |
| M1-v2 | +0.100 (+0.106) | +0.276 (+0.224) | +0.280 (+0.276) | +0.833 (+0.526) | +0.131 (+0.095) | +0.016 (+0.086) | +0.273 [+0.016, +0.833] | +0.219 |
| M1 | +0.095 (+0.085) | +0.244 (+0.225) | +0.267 (+0.244) | +0.771 (+0.537) | +0.140 (+0.131) | +0.016 (+0.086) | +0.256 [+0.016, +0.771] | +0.218 |
| M0 | +0.025 (+0.029) | +0.075 (+0.059) | +0.130 (+0.140) | +0.100 (+0.043) | +0.012 (+0.038) | +0.011 (+0.084) | +0.059 [+0.011, +0.130] | +0.066 |
| round_robin | -0.139 (-0.106) | -0.383 (-0.297) | -0.262 (-0.193) | -0.010 (-0.132) | -0.206 (-0.165) | -0.770 (-1.099) | -0.295 [-0.770, -0.010] | -0.332 |

| Δ vs ramjet | Mooncake open L2 | Mooncake open L3 | Mooncake closed L3 | FAST25 conv. open L2 | FAST25 synth. open L2 | Sessions open L2 | Mean live [range] | Mean sim |
|---|---|---|---|---|---|---|---|---|
| M1-v2 | +0.077 (+0.069) | +0.190 (+0.122) | +0.133 (+0.141) | +0.663 (+0.470) | +0.035 (+0.011) | +0.0002 (-0.0004) | +0.183 [+0.0002, +0.663] | +0.135 |
| M1 | +0.072 (+0.048) | +0.159 (+0.123) | +0.119 (+0.109) | +0.600 (+0.481) | +0.043 (+0.046) | -0.0002 (-0.001) | +0.166 [-0.0002, +0.600] | +0.134 |
| M0 | +0.001 (-0.008) | -0.011 (-0.043) | -0.017 (+0.005) | -0.070 (-0.014) | -0.084 (-0.046) | +0.001 (-0.002) | -0.030 [-0.084, +0.001] | -0.018 |
| default@defaults | -0.023 (-0.037) | -0.086 (-0.102) | -0.147 (-0.135) | -0.171 (-0.056) | -0.096 (-0.084) | -0.016 (-0.086) | -0.090 [-0.171, -0.016] | -0.084 |
| round_robin | -0.162 (-0.143) | -0.469 (-0.399) | -0.410 (-0.328) | -0.181 (-0.188) | -0.303 (-0.249) | -0.780 (-1.099) | -0.384 [-0.780, -0.162] | -0.401 |

**Live noise.** The repeated default@defaults run gives r = s(default_a, default_b) per cell: Mooncake open L2 -0.010, Mooncake open L3 -0.016, Mooncake closed L3 +0.023, FAST25 conv. open L2 +0.008, FAST25 synth. open L2 -0.004, Sessions open L2 -0.006.
So σ_live = √(mean r²/2) = **0.0090** per run (6 cells), and the sd of a single live delta
is about 0.011 against the mean of two same-job default runs, 0.013 against a single one (on FAST25 conv. open L2 and Sessions open L2,
where only one default ran in the same job as the policy), and 0.013 against ramjet.
24 of the 30 live Δ vs default exceed twice their sd (0.022 or 0.026); the rest are round_robin on FAST25 conv. open L2, M0 on FAST25 synth. open L2, ramjet (tuned) on Sessions open L2, M1-v2 on Sessions open L2, M1 on Sessions open L2, M0 on Sessions open L2.
With 6 cells and one run per policy, no significance test was registered.

**Live − sim.** Each entry is the live delta minus the simulated delta from the two tables above; * marks a
difference larger than twice the sd of the live delta. This rule is descriptive and was not registered: the
simulated counterpart replays the same requests, so live noise is the only sampling term, and σ_live itself
rests on 6 default pairs.

| Live − sim | Mooncake open L2 | Mooncake open L3 | Mooncake closed L3 | FAST25 conv. open L2 | FAST25 synth. open L2 | Sessions open L2 |
|---|---|---|---|---|---|---|
| ramjet (tuned) vs default@defaults | -0.014 | -0.017 | +0.012 | +0.114\* | +0.012 | -0.070\* |
| M1-v2 vs default@defaults | -0.005 | +0.052\* | +0.004 | +0.307\* | +0.036\* | -0.070\* |
| M1 vs default@defaults | +0.010 | +0.019 | +0.023\* | +0.234\* | +0.009 | -0.070\* |
| M0 vs default@defaults | -0.004 | +0.015 | -0.010 | +0.058\* | -0.026\* | -0.073\* |
| round_robin vs default@defaults | -0.033\* | -0.086\* | -0.069\* | +0.121\* | -0.041\* | +0.329\* |
| M1-v2 vs ramjet | +0.008 | +0.068\* | -0.008 | +0.193\* | +0.024 | +0.001 |
| M1 vs ramjet | +0.024 | +0.035\* | +0.011 | +0.120\* | -0.003 | +0.0004 |
| M0 vs ramjet | +0.009 | +0.032\* | -0.022 | -0.056\* | -0.038\* | +0.003 |
| default@defaults vs ramjet | +0.014 | +0.017 | -0.012 | -0.114\* | -0.012 | +0.070\* |
| round_robin vs ramjet | -0.019 | -0.070\* | -0.082\* | +0.007 | -0.053\* | +0.319\* |

### 15.4 Ranking agreement, sim vs live (Kendall τ_b)

| | Mooncake open L2 | Mooncake open L3 | Mooncake closed L3 | FAST25 conv. open L2 | FAST25 synth. open L2 | Sessions open L2 |
|---|---|---|---|---|---|---|
| τ_b, sim k = 0 (registered) | 0.87 | 0.87 | 0.87 | 0.87 | 1.00 | 0.87 |
| τ_b, sim k0–2 mean | 0.87 | 1.00 | 0.87 | 0.87 | 1.00 | 0.73 |
| τ_b, sim lag 10 ms | 0.87 | 1.00 | 1.00 | 1.00 | 0.87 | 0.97 |
| τ_b, sim lag 50 ms | 0.87 | 1.00 | 1.00 | 0.87 | 0.87 | 0.87 |
| Discordant pairs (k = 0) | ramjet (tuned)/M0 (within noise) | M1-v2/M1 | ramjet (tuned)/M0 (within noise) | M1-v2/M1 | none | ramjet (tuned)/M1-v2 (within noise) |

- **Pooled (registered primary):** τ_b between the per-policy means over cells is **1.00**
  (exact one-sided permutation p = 0.0014 over the 720 orderings, descriptive).
  Over all 30 (cell, policy) points it is 0.74; the mean per-cell τ_b is 0.89.
- **Sensitivity (registered):** pooled τ_b with sim k0–2 mean 1.00 (points 0.84, mean per-cell 0.89); lag 10 ms 1.00 (points 0.78, mean per-cell 0.95); lag 50 ms 1.00 (points 0.75, mean per-cell 0.91).
- **Mean Δ vs default, live (sim):** M1-v2 +0.273 (+0.219), M1 +0.256 (+0.218), ramjet (tuned) +0.090 (+0.084), M0 +0.059 (+0.066), default@defaults +0.000 (+0.000), round_robin -0.295 (-0.332).
- **Verdict (registered rule τ_b ≥ 0.6):** the live ranking **agrees** with simulation.
- **Where cells disagree:** all 5 per-cell discordances are pairs that simulation separates by at most 0.011.
  3 are within live noise (|live difference| ≤ 2√2·σ_live). The other 2 put M1-v2 ahead of M1 live on Mooncake open L3 (live +0.032, sim -0.001); M1-v2 ahead of M1 live on FAST25 conv. open L2 (live +0.062, sim -0.011).

### 15.5 M1-v2 − ramjet, sign agreement

| Cell | Sim (k = 0) | Live | Same sign | \|live\| > 2√2·σ_live |
|---|---|---|---|---|
| Mooncake open L2 | +0.069 | +0.077 | yes | yes |
| Mooncake open L3 | +0.122 | +0.190 | yes | yes |
| Mooncake closed L3 | +0.141 | +0.133 | yes | yes |
| FAST25 conv. open L2 | +0.470 | +0.663 | yes | yes |
| FAST25 synth. open L2 | +0.011 | +0.035 | yes | yes |
| Sessions open L2 (not informative: \|sim\| ≤ 0.01) | -0.0004 | +0.0002 | no | no |

Over the 5 informative cells the live delta is positive in **5/5**, agrees in sign with simulation in
5/5 (5/5 beyond 2√2·σ_live = 0.0256), and averages **+0.219** live against +0.162 in
simulation. By the registered rule (positive in ≥ 2/3 of informative cells and a positive mean),
**M1-v2 > ramjet holds live**. On the sessions cell, where the four tuned policies tie at ceiling, both are ties (-0.0004 sim, +0.0002 live).

### 15.6 Registered secondaries

- **Token-weighted goodput (A19).** Pooled τ_b 0.60, exactly at the registered 0.6 threshold (mean per-cell 0.58).
  M1-v2 − ramjet is positive live in 3/5 informative cells, mean -0.035, so M1-v2 > ramjet
  **does not hold** on this metric live. By the same rule it does not hold in simulation on these cells
  (2/5 positive, mean -0.044; signs agree in 4/5), in line with §1 limit 4 and §10.2: the gain is a
  request-count property.
- **Live-idle E0.** Re-scoring every run against an E0 fitted on the live idle runs (72 requests from 4 jobs)
  instead of the AIS E0 gives pooled τ_b 1.00 and M1-v2 − ramjet positive in 5/5 cells, mean +0.241.
  So the ranking and sign results do not hinge on the simulator's E0. The per-node-fit sensitivity has only 2 complete cells
  (only some jobs ran an idle calibration); there τ_b is 0.87 and the sign holds in 2/2.
- **Engine-level descriptives.** Median live/sim ratio over the 6 cells for TTFT and per-request mean ITL;
  prefix-cache hit rate (live: vLLM counters; sim: replay reuse fraction) over the cells where the vLLM
  counter is comparable (see below); maximum per-worker request share over the 6 cells.

| Policy | TTFT p50 | TTFT p90 | Mean-ITL p50 | Mean-ITL p90 | Prefix hit live / sim (cells) | Max worker share live / sim |
|---|---|---|---|---|---|---|
| default@defaults | 1.39 | 1.12 | 0.99 | 1.02 | 0.363 / 0.359 (3) | 0.28 / 0.29 |
| ramjet (tuned) | 1.37 | 1.09 | 0.97 | 1.02 | 0.448 / 0.447 (2) | 0.33 / 0.34 |
| M1-v2 | 1.14 | 1.09 | 0.90 | 0.99 | 0.439 / 0.443 (2) | 0.82 / 0.81 |
| M1 | 1.22 | 1.10 | 0.93 | 1.01 | 0.442 / 0.441 (2) | 0.81 / 0.78 |
| M0 | 1.35 | 1.09 | 0.99 | 1.05 | 0.423 / 0.425 (2) | 0.27 / 0.28 |
| round_robin | 1.32 | 1.22 | 0.97 | 1.10 | – (0) | 0.25 / 0.25 |

  Live TTFT is higher than AIS (median ratios 1.14–1.39 at p50, 1.09–1.22 at p90) and the median
  request decodes faster (0.90–0.99; p90 0.99–1.10), the same offsetting pattern as the smoke (§11).
  Where it is comparable, the live prefix-cache hit rate is within 0.009 of replay's reuse fraction. In 28 of the
  42 runs it is not: vLLM records a prefix-cache query each time its scheduler retries a waiting request
  (`get_computed_blocks`), so a worker that queues requests reports more queries than prompt tokens and a
  deflated hit rate. Those runs are left out of the column rather than corrected.
  The LR-13 worker-share cap (0.5 at N = 4) is exceeded live by M1-v2 on Mooncake open L3, M1-v2 on Mooncake closed L3, M1 on Mooncake closed L3, M1 on FAST25 conv. open L2, and in simulation by M1-v2 on Mooncake open L3, M1-v2 on Mooncake closed L3, M1-v2 on FAST25 conv. open L2, M1 on Mooncake closed L3, M1 on FAST25 conv. open L2: the concentration flagged in §10.1 also happens on hardware.
  Load-generator lateness on the 35 open-loop runs is at most 2.6 ms at p99 (15.2 ms max).
  18 requests in 16 runs failed in AIPerf because a 1–3 token output carried no text; the
  scorer counts them not good, as registered.

### 15.7 What this can and cannot show

**It shows** that on 6 N = 4 test cells, on real H100s with the same requests, the live ranking of these six
policies agrees with the simulator's (pooled τ_b 1.00, the same order of cell-mean gains over default; per-cell τ_b ≥ 0.87), and that M1-v2 leads ramjet live
in every cell where simulation predicts a lead (simulated +0.011 or more), each time by more than the registered noise bar (2√2·σ_live).

**It cannot show:**

- **The headline's magnitude or significance.** One run per (cell, policy) at one replicate, 6 cells, no
  registered test. The live cells are not a random sample: their simulated M1-v2 − ramjet averages
  +0.135, against the N = 4 test stratum's segment mean of +0.055 (§3.5), and FAST25
  conversation alone contributes 58% of that sum (the other five cells average +0.068). Live deltas exceed
  simulated ones in 4 of 5 informative cells; with single runs that is not evidence that the effect is larger on hardware.
- **The size of the gaps between policies.** 18 of the 30 live Δ vs default@defaults (9 higher live, 9 lower)
  and 13 of the 30 live Δ vs ramjet differ from simulation by more than twice the sd of the live delta (§15.2, §15.3).
  On sessions, every tuned policy's simulated gain over default (+0.084 to +0.086) is +0.011 to +0.016 live, within
  live noise, so the live runs do not support the size of the simulated sessions gains over default (§3.3).
  On FAST25 conversation every gain over default is larger live, and round_robin's simulated deficit vs default (-0.132)
  is -0.010 live.
- **Close pairs.** M1-v2 − M1 is within live noise (2√2·σ_live) in 4 of 6 cells and M0 − ramjet in 4 of 6,
  so the live order inside those pairs is mostly unresolved.
- **Noise for every policy.** σ_live comes from 6 default@defaults repeats (2 across node allocations)
  and is assumed to hold for the other policies.
- **Scope.** One model, one engine version, one GPU type, N = 4 only; no AgentX; one closed-loop cell;
  FAST25 and sessions in open loop only; the four tuned policies tie at ceiling on the sessions cell, so it
  cannot inform M1-v2 − ramjet. Nothing here tests worker-count extrapolation, the AIS league, router-state lag
  on hardware, or faithful LMetric (§9), which was not in the live matrix.
- **Absolute goodput.** Live/sim ratios range from 0.72 to 2.18 and differ between policies in a cell;
  only the ranking and the sign of M1-v2 − ramjet are validated.
- **A token-weighted gain.** M1-v2 > ramjet does not hold live on A19 goodput, as in simulation.
