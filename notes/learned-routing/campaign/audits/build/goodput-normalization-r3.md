# Audit: build checkpoint, lens "goodput-normalization", round 3 (dynamic)

- Auditor: independent dynamic auditor, 2026-10-02 (about 17:38–17:55 PDT).
- Audited state: WT `rupei/learned-routing` at `<commit-14>` (build fixer r2 head), bindings build_id
  `6955b0ee…` (unchanged), `HARNESS_VERSION` `lrh-4`, E0 method `ais-chunked-estimator-v2`, cells
  `lr-cells-v3`.
- Evidence directory: `CR/runs/audits/build-goodput-normalization-r3/` (`scripts/`, `out/`, `logs/`,
  `cells/`, `raw/`). Key outputs: `out/goodput_compare_*.json`, `out/e0_isolated.json`,
  `out/train_recompute.json`, `out/key_probe.json`, `out/env_probe.json`, `out/audit_summary.json`,
  `logs/report_recompute.log`, `logs/rows_vs_cache.log`.
- **Verdict: PASS.** 0 blocker, 0 major, 8 minor (1 new, 6 carried from r2 and re-verified, 1 known
  gap already assigned to calibration).
  - The r2 major (E2E knife-edge at S × scale = 1) is fixed. My own scorer, fed only by my own
    direct replays and my own isolated-replay E0, matches the harness exactly on 132/132 records,
    every window field and all 6 rescore scales, including an AgentX cell at S = 1.0 where an
    exact compare would differ.
  - The normalization reference is default@defaults on the same cell content and replicate in
    lr-train (12/12 objectives exact) and lr-report (norm_mean exact).
  - New: the replay workers inherit `DYN_ROUTER_*` environment variables, which can silently
    replace the policy under an unchanged cache key (latent; none is set today).

## What I verified independently

### 1. Independent A2 goodput from raw `per_request` (7 cells, 132 records)

- **Audit cells** (`scripts/make_cells.py`, `scripts/make_edge.py`). Five PROVISIONAL `gn3-` copies of
  train candidates (no val or test cell), all different from the r1/r2/fixer cells, plus two
  `gn3edge-` cells on a 46-row synthetic trace. Loads are B2's `load_provisional`; SLAs are set so
  both A2 clauses bind.

  | Cell | Mode | Load | SLA (I ms, S) | Measure |
  |---|---|---|---|---|
  | mooncake-w4-osl2.0-n4-open-L2 | open | speedup 0.346663 | 28, 2.5 | trace warm-up/window ÷ speedup |
  | sessions-s3-root2-n4-open-L2 | open, causal turns | speedup 0.464584 | 17, 1.5 | same |
  | mooncake-w4-base-n4-closed-L2 | closed C=16 | 16 | 26, 2.0 | identity warm-up, `full_occupancy` |
  | sessions-s2-think2.0-n4-closed-L2 | closed C=16, multi-turn | 16 | 16, 1.3 | identity warm-up, `full_occupancy` |
  | agentx-A2-think0.5-n4-lanes-L2 | lanes 4, 11 plays | 4 | 20, **1.0** | `warmup_ms` 300 s, `full_occupancy` |
  | gn3edge-open / gn3edge-closed | open 1×, closed C=4, N=2 | – | 30, 2.0 | 10 s warm-up, 30 s window / identity |

- **Harness runs.**
  - `lr-eval`: 6 policies (default@defaults, round_robin, lmetric@defaults, learned-choice θ0
    one_draw, sticky-session hard, and default with a `router_config` sidecar
    `{overlap_score_credit 0.5, prefill_load_scale 1.5}`) × replicates k = 5, 6: 60 records, 0 errors
    (`out/harness_results.jsonl`).
  - Edge cells: default and round_robin, k = 0: 4 records (`out/edge_results.jsonl`).
  - A tiny `lr-train` (below): 68 distinct records.
- **Direct replays** (`scripts/replay_direct.py`). All 132 re-run through `dynamo.replay.run_trace_replay`
  without importing `learned_routing`. From each record I used only locators:
  - the replicate trace, found by SHA-256 among `CR/runs/replicates` and re-hashed;
  - the replay YAML, whose text is checked (one instance; seed = k + 1 for seeded types, none
    otherwise);
  - the `router_config` sidecar, read from the canonical policy file.

  Everything else comes from the cell JSON and `engine.json`. Each replay held one `CR/slots` slot
  (own flock code).
- **Rows.** Direct rows projected to the harness's compact fields equal the cached rows the harness
  scored as multisets, and the canonical row SHA matches, on **132/132** (`logs/rows_vs_cache.log`).
- **E0** (`scripts/e0_isolated.py`). Every distinct completed (ISL, OSL), 15,186 pairs, replayed
  alone on one worker, 4,000 s apart, with fresh hash ids at block size 32 (r2 used 64).
  Completion, isolation, zero reuse and realized lengths are asserted.
- **Scoring** (`scripts/goodput_indep.py`), implemented from the A2 text, with windows derived from
  the cells' slice facts rather than from `measure`:
  - **open loop:** the trace-time window `[t_first + W0, t_first + W0 + W]` from `measure_trace`,
    mapped to replay time; the affine residual on first turns is 0 ms;
  - **closed loop:** warm-up sessions by first trace timestamp, through `request_<line>` IDs (the
    line ↔ ID map is checked on `input_length`, 0 mismatches); start = first non-warm-up arrival;
    end = first session departure strictly after the last session admission;
  - **lanes:** start = first arrival + 300 s; end = min over `agentic.lane_id` of each lane's last
    terminal.
- **Result: 132/132 exact** on every field: `window_good`, `window_requests`, `good_frac_window`,
  `goodput_rps_window`, window start and end, `good`, `goodput_rps`, `good_frac`, and `rescore` at
  all 6 scales (`out/goodput_compare_{harness_results,edge_results,train_tiny_full}.json`). The
  native ITL-only relative error is 0 on all records.
- **The r2 F1 fix holds.** On the AgentX cell at S = 1.0, an exact compare against my isolated E0
  scores fewer requests good than the tolerant one: 164 vs 162 (default k5), 151 vs 143 (RR k5),
  81 vs 76 (default k6). The harness equals the tolerant value on 12/12. `slowdown_atom_frac` is
  0.058–0.076 for default-like policies and 0.109–0.117 for RR on this N=4 cell.
- **The shared E0 v2 table** has 31,297 entries, all now covered by r2's or my isolated replays.
  Relative error ranges from −1.97e-8 to +1.65e-8; 0 entries exceed 1e-7
  (`logs/e0_table_check.log`). My isolated E0 agrees with r2's on 1,122 shared pairs to within
  1.9e-8.

### 2. Edge cases (`gn3edge-`, raw rows inspected directly)

| Row | Replay output | Harness treatment | Verdict |
|---|---|---|---|
| OSL 0 (t = 10 s) | `completed`, `ttft`/`e2e` null | not completed, so a forced in-window miss; summary says 44 completed, harness 43 | F2 (carried) |
| OSL 1 | completed, e2e = ttft | ITL skipped; E0 = prefill only; good | correct |
| ISL 140,000, and ISL 131,072 OSL 1 | `rejected` (`prompt_tokens >= max_model_len`) | arrival basis: in-window misses; completion basis: invisible | policy-independent; F5 (carried) |
| ISL 131,000, OSL 500 | completed with `output_length` 72 (context cap) | scored as completed with E0(131000, 72) and ITL over 71 steps | consistent and policy-independent; 0 such rows in the 40 Mooncake-format candidate traces (145,104 rows) |
| arrival exactly at the window end (t = 40 s) | – | counted (window is inclusive at both ends) | benign: open-loop flat arrivals are policy-independent |
| warm-up rows / post-window rows | – | excluded / excluded; in-window late finishers counted | correct (A2.3) |

- Open-loop windows: 0 rows within 1 ms of a boundary on Mooncake. On sessions-s3 one first turn
  lies within 1e-6 ms of a boundary for every policy. Its membership is identical across policies,
  so it cannot bias ratios.
- Warm-up guard (F3) is still absent: all 55 open-loop candidates have only `{basis: arrival}`
  (each carries a positive `measure_trace.warmup_ms` hint), and all 22 lanes candidates have no
  `warmup_ms`.

### 3. Normalization reference

- **lr-train** (`out/train_tiny/`).
  - Run: `spaces/default_cost_fn.yaml`, `clipped_log_ratio`, seed 23, popsize 4, 2 generations,
    3 train cells (open Mooncake, closed sessions, AgentX lanes) and 1 validation cell (open
    sessions), with `--val-every 1`.
  - Recompute (`scripts/train_recompute.py`): every one of the **12 objectives** (8 generation and
    4 validation), recomputed from my own scorer on my own direct replays, equals the harness
    value (max |Δ| 0).
  - Reference: `38a79285…` = `dynamo-default-cost-fn`, `parameters {}`, `router_config {}`. Its
    replay YAML has seed k + 1 on all pairs.
  - Pairing: candidate and reference share `cell_sha`, `build_id`, replicate `trace_sha256`,
    protocol and harness version on **68/68**.
- **lr-report.** On my 60 lr-eval records it picks `default@defaults` (`38a79285…`) and pairs on
  `(cell_id, repeat, cell_sha, build_id, protocol, harness_version)`. `norm_mean` equals my own
  paired ratios exactly for all 6 policies (`logs/report_recompute.log`).
- **Sidecar spec.** The `router_config` spec gets its own `policy_sha`, and the replay applies its
  knobs: rows differ from default@defaults, and my direct replay with the same knobs reproduces
  them.

### 4. What closed-loop and open-loop goodput mean (measured on gn3 cells)

| Cell | Count scored | Window length | So the rate ratio is |
|---|---|---|---|
| Mooncake open (flat) | 2,480 for every policy | fixed | pure attainment: RR 1.22–1.23×, lmetric 1.28–1.35× default |
| Sessions open (causal turns) | 2,726–2,728 (policy-dependent by ≤ 2) | fixed | essentially attainment |
| Mooncake closed C=16 | 2,464 for every policy (M − warm-up − C) | 737–848 s | throughput × attainment: RR attainment +7–8% vs default, rate −0.5% to −2.7% |
| Sessions closed C=16, think ×2 | 2,629–2,631 | 3,019–3,041 s (think-bound) | essentially attainment |
| AgentX lanes 4 | 217–218 (k5), 126–131 (k6) | replicate-dependent | throughput × attainment on a short full-occupancy window |

- Warm-up spill into closed-loop windows is policy-independent: 16 warm-up completions on Mooncake
  (0.65%) and 49–51 on sessions (1.83–1.90%) are excluded by identity for every policy
  (`logs/closed_edge_effects.log`).
- As in r2, open and closed ratios measure different quantities, so gates and tables should be
  stratified by `load_mode` (LR-09). lr-report already groups by it.

### 5. Cache-key coverage (`scripts/key_probe.py`, `out/key_probe.json`, `out/env_probe.json`)

- **Keyed:** cell content (load, SLA, `measure`, N, block size, `engine_overrides`, `replay_options`,
  transform), trace content SHA, `mock_engine_args` and model, policy parameters plus the
  `router_config` sidecar, replicate k and protocol, `HARNESS_VERSION`, and bindings build_id
  (`.so` SHA + aisimulate version; aisimulate is a wheel install, so its data is pinned by version).
- **Not keyed:**
  - the caller's environment (F1, new);
  - `seeded` (F4a);
  - E0 identity: top-level `ais_perf_config`, E0 method and code (F4b);
  - scoring and replicate code (F4c).

  Every one of these depends on a manual bump or on hygiene.

## Findings

### F1 (minor, NEW): replay workers inherit `DYN_ROUTER_*` environment variables, which change results under an unchanged cache key

- **Code.** `pool.py:201` builds the worker environment as `dict(os.environ)`. kv-router reads several
  variables at replay time, none of which is in the cache key or the record:
  - `DYN_ROUTER_WORKER_SELECTION_POLICY`. It overrides the YAML role selection
    (`lib/kv-router/src/scheduling/config.rs:1259-1275`: "overrides the role-specific YAML
    selections").
  - `DYN_ROUTER_ACTIVE_REQUEST_EXPIRY_SECS` (`sequences/multi_worker.rs:47`).
  - `DYN_ROUTER_OVERLAP_REFRESH_AFTER_SECS` (`scheduling/overlap_refresh.rs:127`).
  - `DYN_ROUTER_POSITIONAL_SEARCH_MODE` (`indexer/positional.rs:69`).
- **Measured.** I ran `lr-eval` against a private root (its own cache, with the shared slots) and
  `DYN_ROUTER_WORKER_SELECTION_POLICY=default` set, on gn3 Mooncake open, k = 5:

  | Policy | Cache key | `goodput_rps_window` with the env var | Clean |
  |---|---|---|---|
  | lmetric@defaults | identical (`6ab12948…`) | 1.1478 | 1.4781 |
  | learned-choice θ0 | identical (`21af2df5…`) | 1.1305 | 1.1478 |

  Neither run raised an error. Run against the shared root, either result would have been cached
  as that policy's entry and served to every later stage. It would also poison
  `default@defaults` references, so normalization would be affected too.
- **Exposure.** Latent: no `DYN_*` variable is set in this shell, and no campaign file sets one.
  The variable is an ordinary live-router knob, though, and remote bundles run under whatever
  environment the node provides.
- **Fix.** In `EvalPool`, refuse to start, or strip the variables, when any `DYN_ROUTER_*`
  (or, more broadly, `DYN_*` other than `DYN_LOG`) variable is present. Record the decision in
  the worker's ready message, and apply the same in the bundle runner.

### F2 (minor, carried from r2 F2, re-verified under lrh-4): zero-output requests are never scored

- AISim reports `requested_output_length 0` as `completed` with null `ttft_ms` and
  `e2e_latency_ms`. On the edge cell the summary says 44 completed; the harness counts 43.
- On arrival basis such a request is a forced miss for every policy; on completion basis it is
  dropped.
- It is policy-independent. Within the campaign it is 1 AgentX row (play 0021, val pool). A3
  recycling multiplies it in val copies.
- Fix: score it as e2e := terminal − arrival against S·E0(ISL, 0), or exclude it explicitly and
  record that.

### F3 (minor, carried r1 F3 / r2 F3): no plan-time warm-up guard for open loop and lanes

All 55 open-loop candidates have `measure = {basis: arrival}` with a positive
`measure_trace.warmup_ms` hint, and all 22 lanes candidates lack `warmup_ms`. If calibration fills
loads and SLAs but not `measure`, the warm-up is scored silently. Fix: `validate_measure`
raises when an arrival-basis cell has a positive `measure_trace.warmup_ms` but no
`measure.warmup_ms`, or when a lanes cell has no `warmup_ms`.

### F4 (minor, carried r2 F4, re-verified): remaining stale-cache risks outside the key

- **a. `seeded`.** `{"type": "dynamo-default-cost-fn", "seeded": false}` still has **the same
  `policy_sha` as default@defaults**, and its replay YAML has no seed, so an unseeded,
  nondeterministic run can fill the reference's cache entry.
- **b. E0 identity.**
  - Changing only the top-level `ais_perf_config` (the E0 source) changes the E0 table path but
    neither `cell_sha` nor the result key.
  - The E0 method is not in the key.
  - Today the table is correct (item 1, max |rel| 2e-8), and the top-level and
    `mock_engine_args` AIS configs are equal.
- **c. Code.** Scoring code, E0 code, the replicate materializer and the editable Python replay
  wrapper are covered only by manual `HARNESS_VERSION` and protocol bumps.
- **d.** The worker memoizes `engine.json` by path.

Fix as in r2: put `seeded` in `identity()` when it differs from the type default; put the E0
method and the E0 file or engine SHA, plus `rep.trace_sha256`, in the key.

### F5 (minor, carried): completion basis hides rejections

- Rejected rows (`terminal_time = dispatch`) never count as completions, so the closed-loop
  edge cell shows 2 in-window rejections in neither the numerator nor the denominator.
- This is policy-independent (static admission limit) and consistent with A2's "good completions
  per second". Record it.

### F6 (minor, carried r2 F6, magnitude updated): absolute `eps = 1e-3` in lr-train

- On the N=4 AgentX lanes cell, default's `goodput_rps_window` is 0.070–0.103 rps. There, eps
  shrinks log-ratios by up to **1.81%**, against 0.04–0.18% on Mooncake and sessions
  (`logs/eps_effect.log`).
- lr-report's ratios use no eps, so the training objective and the reported headline differ
  slightly.
- Fix: use a relative eps, or record the choice.

### F7 (minor, carried): the A1 noise rule has no caller

`noise.differs_beyond_noise` is still called only from tests. The pilot glue must build its
inputs from content-keyed records.

### F8 (minor, known; assigned to calibration by the A3 sidecar): A3 lowered AgentX cells cannot be planned

- `Cell.load_kwargs` raises "agentic_lanes needs trace_format weka", and `resolve_replicate`
  raises "unsupported trace_format for crn-order-v1: agentic_mooncake" (`out/key_probe.json`).
- `facts/agentx_lowered.json` (`generator_usage.harness_gap`) assigns this to calibration.
- **Lens requirements for that extension:**
  1. If replicate k becomes "the same GenSpec with seed k" (new play draws and arrivals, not a
     tie permutation), it needs its **own protocol id**, and the generated file's SHA must enter
     the key. Generator changes happen (A3 fix r0 changed the base transform handling).
  2. On open-loop agentic arrival basis, a descendant's own arrival is policy-dependent, so the
     scored set moves with the policy. This is harmless for the rate objective in steady state,
     but `good_frac_window` becomes gameable. Score by copy-root arrival `a_i` (as A3 parity
     audit m2 suggested), or record the choice.
  3. Lanes over recycled copies need `play_id` on every row for occupancy. Native Weka rows have
     it; lowered rows are untested.

## Closed since r2

r2 F1 (major, E2E knife-edge at S × scale = 1) is fixed and confirmed:

- the v2 E0 matches 31,297 isolated replays to within 2e-8;
- the 1e-6 tolerance makes the AgentX S = 1 atom good;
- the harness equals my tolerant independent value on 12/12 AgentX records and on all 132 records
  overall.

## Lessons (LESSONS.md)

- **Applied:**
  - LR-01: windowed attainment, re-derived from slice facts on open, closed and lanes cells;
  - LR-03: CRN pairing, same replicate trace and seed k + 1, verified on 68/68 lr-train pairs;
  - LR-08: the rescore sweep verified at 6 scales on 132 records, and the S × scale = 1 atom
    re-tested;
  - LR-09: the per-load-mode meaning of the ratio (item 4);
  - LR-11: content-keyed pairing in lr-report and lr-train.
- **Rejected:** LR-04 (Amendment A2.5) and the TTFT part of LR-08 (A2.2).
- **Out of lens:** LR-02, LR-05–07, LR-10, LR-12–15.

## Process notes

- **Replays,** all holding `CR/slots` slots (at most 12 at once):
  - 60 + 4 through lr-eval;
  - about 68 fresh in lr-train;
  - 2 env-probe replays through lr-eval with a private root and cache, using the shared slots;
  - 132 direct replays;
  - 7 isolated-E0 batches (15,186 single-request replays).
- **Writes:**
  - none to WT or the main checkout; nothing committed; no fix applied;
  - 132 audit-only entries (`gn3-`/`gn3edge-` cell IDs, about 14.7 MB) added to `CR/runs/cache`;
  - 15,186 verified pairs merged into the shared E0 v2 table.

  Scratch is listed in `CLEANUP.md`.
