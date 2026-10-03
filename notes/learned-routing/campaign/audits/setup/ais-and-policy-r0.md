# Audit: setup checkpoint, lens "ais-and-policy", round 0

- **Auditor role:** independent dynamic auditor. I ran my own replays and wrote my own check scripts;
  I did not take the setup summaries on trust.
- **Date:** 2026-10-02.
- **Worktree:** `rupei/learned-routing-public` at `<commit-01>`. Bindings `_core.abi3.so` were built at
  13:21:03. The seed patch's Rust sources date from 13:17:52; only the test file changed after the
  build, so the `.so` contains the patch.
- **Python:** `WT/.venv/bin/python`, with `DYN_LOG=warn` and `PYTHONDONTWRITEBYTECODE=1`.
- **Evidence:**
  - every number below comes from `CR/runs/audits/setup-ais-and-policy-r0/out/*.json`;
  - the scripts that produced them are in `CR/runs/audits/setup-ais-and-policy-r0/scripts/`;
  - the policy YAMLs used are in `CR/runs/audits/setup-ais-and-policy-r0/policies/`.
- **CPU:** the harness slot pool (`CR/slots/`) does not exist yet. All audit replays ran one at a
  time: one Python process, or a parent process waiting on a single child. Each was a smoke-scale
  replay of 1 to 2,000 requests, about 1 s each. Recorded in `facts/DEVIATIONS.md`.

**Verdict: PASS.** I found no blocker or major issue:

- AIS timing is in effect with exactly the engine.json identity, on every worker and in every
  replay mode.
- Policies selected through `router_policy_config` change routing, and so do their parameters.
- The Python entry point supports every mode the campaign needs, with the claimed seeding.

There are six minor findings, plus one observation, F7, which is not a defect.

## Check 1. AIS timing is active with exactly the engine.json identity

Evidence: `out/ais_identity.json`, `out/per_worker_ais.json`, `out/estimation_mode_pin.json`,
`out/entry_point.json`.

### Identity sensitivity

Each run is a single-request replay with ISL 8192 and OSL 64, N=1, concurrency 1. I changed one AIS
identity field at a time. Every variant's replay TTFT equals the Python AIS estimator
(`dynamo._internal.ais.create_session`) for the same identity, to about 1e-9 ms.

| Variant | Replay TTFT (ms) | Replay ITL (ms) | Estimator prefill (ms) |
|---|---|---|---|
| engine.json (Qwen3-32B, h100_sxm, vLLM 0.24.0, TP2) | 475.975 | 15.337 | 475.975 |
| tp=4 | 273.628 | 10.107 | 273.628 |
| tp=1 | 864.916 | 26.361 | 864.916 |
| system=h200_sxm | 473.798 | 12.583 | 473.798 |
| model=Qwen/Qwen3-8B | 127.487 | 5.295 | 127.487 |
| **no ais_perf_config (polynomial default)** | **169.137** | **7.193** | — |

What this shows:

- Every identity field changes timing. The engine.json identity is the one in effect.
- The estimator reports `readiness=ready` and `selected_estimation_mode=op_level`.
- Unknown identities fail loudly; replay never falls back to polynomial timing. A bogus system,
  bogus version, bogus model, the sglang backend, or vLLM 0.25.0 for this model each raise
  `ValueError: AIS estimator is not ready ...`. `fallback_policy` is `deny`.

### Every worker is AIS-timed

`out/per_worker_ais.json`: Poisson 0.5 rps, N=4, round robin, ISL 1024.

- Under AIS, the per-worker minimum TTFT is 55.4868 ms and the minimum ITL is 15.1108 ms on all
  four workers.
- Under polynomial timing the same values are 32.49 ms and 5.93 ms.

### Loaded cells: AIS vs polynomial timing

`out/entry_point.json`:

| Cell | AIS goodput (rps) | AIS mean TTFT (ms) | Polynomial goodput (rps) | Polynomial mean TTFT (ms) |
|---|---|---|---|---|
| Mooncake 2000-row slice, N=4, speedup 4/11, round robin | 1.1273 | 2661 | 1.8275 | 482 |
| same cell, seeded default | 1.0280 | 1385 | 1.7029 | 367 |
| Mooncake closed loop, concurrency 48, seeded default | 2.0963 | 741 | 5.6062 | 231 |
| AgentX weka, 2 lanes, N=2 | 0.0432 | 1299 | 0.0573 | 235 |

### Capacity and model length

- The pinned `num_gpu_blocks` 18,863 equals AIS's `estimate_canonical_num_gpu_blocks` at both
  `max_num_seqs` 1024 and 256.
- The `from_json` round-trip keeps block_size 16, max_model_len 131072 and the full identity.

### Estimation mode

`ais_perf_config.estimation_mode` is `auto` in engine.json. Pinning `op_level` reproduces ISL
1k, 8k and 64k replays exactly (`out/estimation_mode_pin.json`). See F5.

## Check 2. Catalog policies via router_policy_config change routing (policies setup did not test)

Evidence: `out/policy_audit.json`, `out/yaml_and_knobs.json`, `out/optbase_modeled.json`,
`out/thunderagent_det.json`.

**Cells:** setup's `mooncake_first2000.jsonl`, SLA TTFT 2000 ms and ITL 50 ms, at two loads:

- N=4, `arrival_speedup_ratio` 4/11;
- N=8, `arrival_speedup_ratio` 8/11.

Every policy ran twice in the same process.

| Policy | N=4 goodput (rps) | N=4 mean TTFT (ms) | N=8 goodput (rps) | N=8 mean TTFT (ms) |
|---|---|---|---|---|
| round_robin | 1.1273 | 2661 | 2.8903 | 1974 |
| dynamo-default-cost-fn seed 1 | 1.0280 | 1385 | 2.8687 | 945 |
| dynamo-default-cost-fn seed 2 | 1.0519 | 1373 | 2.8180 | 945 |
| chwbl (defaults) | 1.0151 | 2320 | 2.9414 | 1471 |
| chwbl load_factor 1.0 | 1.1428 | 2048 | 3.0905 | 1306 |
| chwbl load_factor 4.0 | 0.0000 | 358185 | 0.4216 | 123446 |
| chwbl prefix_tokens 4096 | 1.1472 | 2117 | 3.0210 | 1362 |
| llm-d-precise-prefix (defaults) | 1.2233 | 1966 | 3.2950 | 1162 |
| precise, prefix weight 0 | 1.1597 | 1899 | 3.0450 | 1237 |
| precise, prefix weight only | 1.1107 | 2606 | 2.6522 | 1981 |
| llm-d-optimized-baseline (throughput) | 1.1451 | 1260 | 3.1880 | 802 |
| ramjet | 1.2411 | 1347 | 3.3923 | 796 |
| dualmap | 1.1828 | 2963 | 3.1340 | 974 |

### Results

- **Repeats.** In-process repeats are identical (canonical per-request multiset) for every policy
  in the table. They are also identical for thunderagent selection (three runs each on Mooncake N=4
  and on AgentX).
- **Reproduction.** My independent runner reproduces setup's N=8 seeded-default numbers exactly
  (2.8687 and 2.8180).
- **Parameters take effect, with policy-specific signatures:**
  - chwbl with `load_factor` 4.0 is pure consistent hashing. With only 3 distinct first blocks in
    the slice, it sends all 2,000 requests to one worker at N=4 and to 3 workers at N=8.
  - chwbl prefix affinity is real, so prefix hashes reach prefix-keyed policies. The modal-worker
    share per first-block group is 0.35/0.53/0.38 at N=4 and 0.20/0.70/0.61 at N=8. Round robin
    gives 0.26 and 0.13.
  - llm-d-precise-prefix with prefix weight 0 loses the affinity: N=8 modal shares are
    0.14/0.14/0.15, against 0.17/0.36/0.69 with the defaults.
- **Goodput gaps exceed seed noise.** Gaps between policies (for example precise +19% and ramjet
  +21% over seed-1 default at N=4) are far above the default's tie-seed spread (2.3% at N=4 and
  1.8% at N=8).
- **Malformed YAML fails loudly in most cases.** These fail before replay:
  - an unknown top-level key, or a typo such as `worker_selectoin`;
  - an empty file;
  - an instance without `type`;
  - `aggregated` naming a missing instance;
  - an invalid parameter value (chwbl `load_factor` 0.5);
  - a negative or float `seed`;
  - a nonexistent path.

  **Exception:** a YAML with no `aggregated:` key silently runs the unseeded built-in default
  (F2).
- **Sidecar `KvRouterConfig` knobs work with a YAML policy.** With YAML `dynamo-default-cost-fn`
  plus `seed`, each of these changes routing: `router_temperature`, `router_queue_threshold`,
  `overlap_score_credit`, `prefill_load_scale` and `decode_active_request_weight`. A YAML
  `router_temperature: 0.5` reproduces the sidecar `router_temperature=0.5` run bit-for-bit.
  - Seeded softmax at temperature 0.5 is repeatable, and seed 2 differs.
  - chwbl ignores `router_temperature`; its run stays identical.

## Check 3. The chosen entry point supports what the campaign needs

The entry point is `dynamo.replay.run_trace_replay` / `run_synthetic_trace_replay`. Evidence:
`out/entry_point.json` and `out/lanes_and_sessions.json`.

| Need | Result |
|---|---|
| Mooncake open loop (`arrival_speedup_ratio`) | OK: 2000/2000 completed; per_request rows present |
| Mooncake closed loop (`replay_concurrency`) | OK: 2000/2000 completed. Each request gets a synthetic session id `request_<n>`, turn 0. In open loop the session id is None. |
| Weka/AgentX, `agentic_lanes` | OK: 85/85 completed; session ids for all 85 rows (10 sessions). Lanes are honoured: lanes=1 gives goodput 0.0229, lanes=2 or 4 give 0.0432. With only 2 plays, lanes ≥ 2 are identical. |
| Weka/AgentX, timestamps | OK. Speedup 1.0 gives the same result as lanes=2. Speedup 4.0 gives goodput 0.0682 and duration 968 s, against 1,503 s at speedup 1.0. |
| Synthetic sessions (`request_rate`, `turns_per_session`) | OK: 360/360 completed, 120 sessions. `arrival_seed` 42 vs 43 changes the workload. Closed-loop synthetic works. |
| per_request output | OK in every mode: TTFT, ITL, worker index, `routing_history` and `terminal_status`. My recomputed goodput equals the native field exactly (relative difference 0) in all 42 runs I checked. |
| router_policy_config | OK in every mode (seeded default, chwbl and precise under Mooncake open/closed, weka lanes/timestamps and synthetic) |
| AIS timing | Active in every mode; AIS results differ from polynomial for Mooncake open, Mooncake closed and weka lanes (table above) |
| Seeding | Policy `seed` makes runs repeatable and distinct seeds differ. A fresh process equals an in-process run. Running chwbl and lmetric first in the same process does not change a later seeded-default result (Mooncake, weka and synthetic). |
| Cost | In-process 2000-request replays: wall time median 0.94 s, range 0.66–1.30 s over 26 runs, matching setup's "about 1.0 s" |

Two related facts:

- Passing the top-level `ais_perf_config` without `router_prefill_load_model='ais'` raises an
  error, both with a default `KvRouterConfig` and with no router config
  (`out/top_level_ais_guard.txt`). An accidental router-AIS coupling cannot pass silently.
- With `router_prefill_load_model='ais'` and an `AisPerfConfig`, routing changes for
  llm-d-optimized-baseline in every `ttft_source` mode, including `throughput` (1.1451 → 1.1701).
  The modeled prefill-load model changes the host's load accounting for every policy, not only
  `ttft_source`. This supports engine.json's "keep `router_prefill_load_model` none".

## Findings

### F1 (minor). The max_model_len rejection rule in facts/setup.json is wrong: replay truncates, it doesn't reject

- **Claim.** setup.json `long_context.notes` and key facts say "Requests with ISL+OSL above 131072
  are rejected".
- **What replay actually does** (`out/mml_boundary.json`):
  - It rejects only when ISL ≥ 131072. With ISL 131072 the request is rejected even at OSL 1.
  - With ISL < 131072, the output is silently truncated to 131072 − ISL tokens and
    `terminal_status` is `completed`.

  | ISL | OSL requested | Output tokens | Status |
  |---|---|---|---|
  | 131009 | 64 | 63 | completed |
  | 131071 | 64 | 1 | completed |
  | 131000 | 500 | 72 | completed |
  | 130000 | 5000 | 1072 | completed |

- **Why setup missed it.** Setup only probed 131000+64 and 140000.
- **Impact today.**
  - None for the shared traces: Mooncake max ISL+OSL is 125,878 and toolagent's is 126,527; 0 of
    each trace's 23,608 rows exceed 131072.
  - None for the 2 AgentX plays: max 124,597, 0 truncated.
- **Where it can bite.**
  - AgentX play selection, and the ISL-stretch transforms used for out-of-range test cells.
  - A check based on `terminal_status == rejected` would miss truncations, and truncated requests
    count as completed with fewer decode tokens.
- **Fix.**
  - Correct the setup.json note.
  - The trace stage keeps the pre-filter (drop or cap requests with ISL+OSL > 131072, as the
    next-stage notes already say). This also matches the vLLM API, which rejects
    prompt + max_tokens > max_model_len.
  - `lr-eval` flags any row where `output_length < requested_output_length`.

### F2 (minor). A policy YAML without `worker_selection.aggregated` silently runs the unseeded built-in default

- **Symptom.** `worker_selection: {instances: [{name: candidate, type: chwbl}]}` with no
  `aggregated:` key replays without error.
- **What actually runs.** The default cost function, unseeded. Two identical runs differ: goodput
  1.0384 vs 1.0281, and in a rerun 1.0281 vs 1.0269; the chwbl reference gives 1.0151
  (`out/optbase_modeled.json`, `out/yaml_and_knobs.json`).
- **Contrast.** Every other malformed YAML I tried fails loudly.
- **Why it matters.** If a harness or YAML-writer bug drops the key, a "policy" evaluation would be
  the default plus tie noise, with no error.
- **Fix (harness).**
  - Before replay, assert that `worker_selection.aggregated` names an instance with a `type`.
  - Optionally, a one-cell canary checks that a non-default policy's canonical per-request hash
    differs from the default's.
  - The upstream option is to reject `worker_selection` without a pool binding. Record it in
    `UPSTREAM_FOLLOWUPS.md`.

### F3 (minor). Setup's "policy is in effect" evidence metric doesn't separate policy from tie-seed noise

- **Setup's evidence.** Setup cites raw differing worker assignments: 761/1000 for default vs
  lmetric and 767/1000 for default vs two-tier.
- **Why it can't discriminate.** The same default policy with seed 1 vs seed 2 differs on
  1650/2000 requests at N=4 and 1764/2000 at N=8 (`out/policy_audit.json`). Tie-break relabeling
  and chaotic divergence dominate the metric.
- **Conclusion still true.** Policies are in effect, as shown by goodput gaps far above the 2% seed
  spread and by the parameter-specific signatures in Check 2.
- **Fix.** Later stages and audits (for example the degenerate-policy audit) must not use raw
  assignment diffs as evidence of a policy difference. Use either of:
  - goodput or TTFT deltas against a tie-seed band (LR-02);
  - permutation-invariant statistics, such as co-location of prefix groups or the per-worker load
    distribution.

### F4 (minor, hypothesis-labeled fidelity caveat). AIS times heterogeneous prefill passes from batch means, and prefill attention beyond 16K is extrapolated

- **Mean-based timing.**
  - aisimulate-core's vLLM scheduler times a multi-request prefill pass as
    `predict_prefill(B, mean_new, mean_prefix)` (`aisimulate-core-0.13.0-dev.202609300000000061`,
    `src/engine/scheduler/vllm/core.rs`, `predict_prefill_duration`). The AIS forward-pass metrics
    only carry sums.
  - With the engine.json identity, a 7,936-token chunk at a 100K prefix takes 1,744 ms alone. With
    one 256-token prompt added, the mean-based estimate is 1,150 ms (−34%): adding work makes the
    pass faster (`out/mixed_batch_mean.json`).
  - In replay, vLLM's token budget limits mixing to roughly the final chunk of a long prompt. For
    the final 2,048-token chunk at a 98K prefix plus 12 short prompts, the mean-based estimate is
    541 ms. A physically consistent estimate is ≥ ~770 ms [hypothesis].
  - A long request alone has TTFT 13,298 ms; with 40 short arrivals it is 13,409 ms
    (`out/mixed_batch_replay.json`).
- **Extrapolation.** The AIS context-attention table for h100_sxm/vLLM 0.24.0 covers `isl` ≤ 16,384
  with `step` (prefix) = 0 only. Prefill timing for longer prompts or cached prefixes is therefore
  extrapolated. The 120K-token TTFT of 17.8 s is still plausible by FLOP count.
- **Impact** [hypothesis]. A small, systematic optimism when short prompts are co-located with
  long-context prefills. Black-box search could exploit it, which matters most for AgentX at 128K.
- **Fix.**
  - Add this to the report's fidelity caveats (LR-14).
  - In the Stage-4 sim-artifact audit, check whether the learned policy co-locates short prompts
    with long-context prefills more than baselines do (LR-13).
  - Record it as an AISim upstream follow-up: per-request prefill descriptors for attention cost.

### F5 (minor). engine.json leaves `estimation_mode: auto`

- **Today.** `auto` selects `op_level`, and pinning `op_level` is bit-identical
  (`out/estimation_mode_pin.json`).
- **Risk.** In remote-shard mode, a different aisimulate build or venv could resolve `auto`
  differently and change timing silently.
- **Fix.**
  - Pin `estimation_mode: op_level` in `engine.json.mock_engine_args.ais_perf_config`. This needs
    no re-run: the result is proved identical.
  - Have the remote bundle assert `aisimulate==0.13.0.dev202609300000000061`.
- **Not done by me.** engine.json is the single source of truth owned by setup and calibration.

### F6 (minor; observation for calibration). Under AIS timing the default cost fn loses to round robin on goodput at moderate load too, through ITL misses

- **Not only in the collapsed smoke cell.** At both audit cells, round robin beats the seeded
  default:
  - N=4: 1.1273 vs 1.0280 / 1.0519;
  - N=8: 2.8903 vs 2.8687 / 2.8180.

  This holds even though the default halves mean TTFT (1385 vs 2661 ms at N=4).
- **The misses are ITL-dominated** (`out/sla_breakdown.json`). At N=4, ITL-only misses are 627 for
  the default vs 317 for round robin. The p90 of per-request mean ITL is 474 ms vs 190 ms.
- **Mechanism** [hypothesis]. Per-request mean ITL on short outputs is dominated by single
  co-scheduled 8K prefill-chunk stalls of about 0.45–0.6 s each.
- **Not a setup defect.** But a weak default inflates "ratio to default" (LR-01, LR-12).
- **For calibration.**
  - Report gains against the best baseline. In these cells ramjet and llm-d-precise-prefix beat
    the seed-1 default by +19% and +21% at N=4, and by +15% and +18% at N=8.
  - Check how sensitive the ITL SLA is to very short outputs before freezing T and I.

### F7 (minor, informational). llm-d-optimized-baseline `ttft_source` is inert in these cells

- **What I saw.** `throughput`, `modeled` and `auto` give bit-identical routing at N=4, both
  without and with the router AIS model (`out/optbase_modeled.json`).
- **Why.** `ttft_source` only matters when the 18 s `max_ttft_penalty_ms` stickiness gate binds.
- **For the baseline stage.**
  - The "modeled (AIS)" variant differs from "throughput" only through the router AIS load model
    (see Check 3).
  - Label it as such, or also tune `max_ttft_penalty_ms`.
  - The default `peak_prefill_tokens_per_second` of 15,928 is within about 14% of AIS's
    uncontended short-context rate for this deployment: 18,455 tok/s at 1k and 17,211 tok/s at
    8k. It falls to about 9.0k tok/s for an 8K chunk at a 32K prefix (LR-04).

## Setup claims I confirmed independently

- **AIS in effect.** The estimator equals replay at 1k and 8k.
- **Op-level mode.** The estimator selects `op_level`.
- **Capacity.** 18,863 blocks.
- **Determinism.** Seeded default, round robin and every catalog port are deterministic. Seeds
  differ by about 2% goodput.
- **Session ids.** Present for weka, synthetic sessions and Mooncake closed loop.
- **Entry point.** The Python API is the right choice.
- **Cost.** About 1 s per 2,000-request in-process replay.

## Lessons from literature/LESSONS.md (it appeared during this audit)

**Applied:**

- **LR-02** (seed noise, not repeats). I measured the tie-seed spread, used it to judge "differs",
  and it underlies F3.
- **LR-14** (fidelity, information parity). It shaped F4 (AIS mixed-pass and long-context
  extrapolation) and the router-AIS load-accounting observation.
- **LR-13** (sim exploitation). F4 names a concrete exploitable artifact for the Stage-4 audit.
- **LR-01 and LR-12** (weak, collapsed default; regime). They frame F6.
- **LR-04** (deployment constants). I checked the llm-d `peak_prefill_tokens_per_second` against
  AIS (F7).

**Rejected as out of scope for a setup audit:**

- LR-03: the CRN draw-consumption design is for the build and training stages.
- LR-05 to LR-11 and LR-15: they concern features, splits, statistics and the model ladder.
