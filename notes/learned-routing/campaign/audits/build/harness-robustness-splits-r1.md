# Audit: build checkpoint, lens "harness-robustness-splits", round 1 (dynamic)

- **Auditor:** independent dynamic auditor (adversarial), 2026-10-02 16:15–16:40 PDT.
- **Audited state:** WT `rupei/learned-routing-public` at `<commit-10>` (clean apart from an untracked
  `benchmarks/learned_routing/tools/agentx_lower/` that belongs to the A3 sidecar), bindings build_id
  `6955b0ee…`, cells lr-cells-v2 (SPLIT_MANIFEST `0f8ea452…`), pycma 4.5.0. pytest: 114 passed
  (`out/pytest.txt`).
- **Evidence root:** `CR/runs/audit-build/harness-robustness-splits-r1/` (`scripts/`, `out/`, `cells/`).
  Bundle scratch: session scratchpad `…/scratchpad/aud1-bundle/`.
- **Audit cells:** PROVISIONAL copies of 12 current train/val candidate cells (`cells/*.jsonl`, made by
  `scripts/make_cells.py`): the per-cell `load_provisional` values from
  `runs/build-fix-r0/workloads/cells_validation.json`, SLA I = 40 ms and S = 3, open-loop windows
  from `measure_trace / speedup`, closed-loop cells with their own identity warm-up. The set covers
  Mooncake open and closed (base, islu1.5, islp1.5, root2), synthetic sessions open and closed, and
  AgentX lanes. I replayed no test cell. No number here is a calibrated campaign result.
- **Discipline:** every replay ran through `lr-eval`, `lr-train` or the bundle's `lr-eval`, all holding
  `CR/slots` with `--num-slots 20`. Processes were stopped only by exact PID, after checking the
  cmdline and parent of each. Nothing was deleted except my own temporary tar file. No WT file was
  changed.

## Verdict: PASS (0 blocker, 0 major, 11 minor)

Both round-0 majors are fixed:
- **S1 (overlapping windows):** no source row is shared between splits.
- **B1 (incomplete bundle):** the bundle runs in a clean environment from an empty venv and matches
  local runs bit for bit.

The harness also held up under every robustness test:

| Test | Result |
|---|---|
| Slot cap, 3 concurrent processes plus another agent | held |
| Timeouts, crashes, `--max-wall-seconds` chunking | correct |
| lr-train after 3 × `kill -9` of exact PIDs | resumes exactly, including continuation |
| CRN pairing | intact |
| CMA-ES bounds, pins and transforms | correct |
| Derived traces, manifests, hold-out axes | match |
| New closed-loop identity warm-up | correct on 119 real records |

Everything remaining is minor: a window rule that is still open, latent harness gaps, and some
operational notes.

## What I verified (no finding)

**V1. Slot pool: 3 concurrent `lr-eval`, plus a sidecar agent** (`out/sp/`, `scripts/slot_monitor.py`,
`scripts/sp_analyze.py`)
- **Load.** 3 processes, each with `--slots 20`, ran 108 + 108 + 72 = 288 replays over 12 cells,
  5 policies and k = 0..3, with 0 errors and 38 s wall. At the same time the A3 sidecar
  (`agentx_lowered parity`, PID 309779) held a slot in 618 samples.
- **Monitor.** A passive /proc/locks monitor (it never calls flock) took 741 samples at 50 ms.
  - At most 20 slots were held at once, and no slot ever had two holders.
  - Slot-time shares were p1 3,860, p2 4,104 and p3 3,026 samples, against task counts of
    108, 108 and 72.
- **Busy-worker check.** A worker counts as busy if it used ≥ 2 CPU ticks in a sampling interval.
  10 samples showed more busy workers than slots held by that parent; none is a breach of the cap:
  - 2 were mid-run, each off by one, explained by a slot hand-off within one sampling interval;
  - 8 came after the parent's last slot release: worker interpreter teardown at pool close,
    ≤ 2 workers, ≤ 1 s in total.

**V2. Timeouts and failures** (`out/to/`, `out/crash/`)
- **Timeouts.** `--timeout-s 0.3` and `--timeout-s 1.6` gave 24/24 timeout records each, exit 2,
  0 cache entries, 0 per-request files and 0 leftover workers. A rerun produced 24/24 fresh results,
  all OK, whose per-request SHAs equal the slot-pool run of the same content.
- **`--max-wall-seconds 1` chunking.** Four calls went 6 done / 18 skipped (exit 3), then 6 / 12
  (exit 3), then 6 / 6 (exit 3), then exit 0, reusing cached results each time.
- **Crash.** SIGKILL of one busy worker (exact PID, a verified child) gave 1 `worker_crashed`
  record, not cached; the other 47 were OK.

**V3. lr-train kill -9 and resume, real replays** (`out/train/`, `scripts/kill_when.py`, `scripts/compare_train.py`)
- **Run B.** Space `learned_choice_m1.yaml`, seed 11, popsize 4, budget 20, 2 train cells and
  1 validation cell, `clipped_log_ratio`. I SIGKILLed it at exact PIDs three times: mid generation 0,
  mid validation, and mid generation 3. Then I resumed it.
- **Run C.** The same arguments, uninterrupted.
- **Comparison** (`compare_B_C.json`):
  - `history.jsonl` identical, with 0 duplicate lines;
  - `best.json` identical, and `best_policy.yaml` identical apart from the run name;
  - CMA mean, sigma and C bit-identical;
  - fevals 20 = 20 and tasks 118 = 118.
- **Continuation.** Extending both runs to budget 28 also matched exactly (`compare_Bx_Cx.json`).
- **Global numpy RNG state.** It differs at the end (`np_rng_equal: false`). This is benign, as the
  matching continuations show. A resumed run samples from the RandomState that was pickled inside
  the CMA object, not from the global generator.

**V4. Common random numbers** (`out/sp/crn.json`)
- Across 3 processes and 5 policies, each of the 48 (cell, k) pairs used exactly one replicate trace,
  and each cell had 4 distinct traces.
- Seeded types got seed k+1; round_robin and lmetric got none.
- 60 cache keys were computed concurrently in two or three processes. All copies are identical on
  the per-request SHA and 9 metrics.

**V5. Closed-loop identity warm-up (fixer r0), independent recomputation** (`scripts/identity_checks.py`,
`out/sp/identity_checks.json`). On 119 closed-loop records, which span k = 0..3, five policies,
Mooncake (base, islu1.5, islp1.5) and sessions:
- **Row identity.** Replay's `request_<n>` is line n of the replicate trace actually replayed. ISL and
  requested OSL match for every row.
- **Exclusion and window.** My own warm-up set equals `warmup_excluded_rows` everywhere, and window
  start, end, `window_requests` and `window_good` all match. I recomputed A2 good from the E0 cache.
- **Dispatch order.** No warm-up first turn is dispatched after the window starts.
- **Open loop.** The 51 open-loop records also match on window counts and good counts.

**V6. Remote bundle, unpacked and run with a local empty venv** (`out/bundle/`)
- **Bundle.** 12 cells × 2 specs × 2 replicates = 48 tasks, including 7 closed-loop identity-warm-up
  cells, 2 AgentX lanes cells and session traces. `site_check` reports 291 modules, 0 from outside
  the site or the stdlib.
- **Clean-room run.** I copied the bundle with tar and ran it as
  `env -i HOME=… PATH=/usr/bin:/bin PYTHON=<fresh uv venv, python3.12, no packaging/yaml> strace -f bash run.sh --slots 8 --slots-dir CR/slots --num-slots 20`.
- **Results.** 48/48 OK, and all 8 worker execs carry `-S`.
- **Exactness.** 48/48 equal the local runs of the same content on the per-request SHA and 13 fields.
- **File access.** Outside the bundle and venv, the only successful accesses were the `CR/slots`
  flocks and directory probes of the `.so` RPATH (WT `target/release/build/*`). There were
  **0 successful file opens** under `<repo>`; those directories hold no `.so`.
- **Ingest.** 48 ingested, 0 rejected. A local rerun then gave 48/48 cache hits with per-request
  files present.

**V7. CMA-ES bounds and transforms** (`scripts/cma_checks.py`, `out/cma/checks.json`). On all 3 shipped
spaces:
- x0 and pycma's initial `xfavorite` decode exactly to `init`;
- 120 decoded samples over 15 generations show 0 bound violations, 0 pinned-element violations
  (θ0 = −1 never searched, LR-05) and 0 exact-bound hits;
- log endpoints map exactly (0.1, 1.0, 10.0).

**V8. Manifests and derived traces** (`scripts/manifest_checks.py`, `scripts/derived_checks_r0impl.py`)
- **SHA-256s.** All 96 `traces/MANIFEST.json` SHA-256s and row counts match. So do the
  SPLIT_MANIFEST file SHAs, its traces-manifest SHA, all 56 derived-trace SHAs (file, name and meta,
  plus source SHAs) and all 104 cells' `trace_sha256` and `transform == meta.spec`.
- **Re-derivation.** I re-ran the r0 auditor's re-derivation, which does not import the workloads
  code, on the current 56 derived traces: 0 problems.

**V9. Split hygiene** (`scripts/split_hygiene.py`, `out/workloads/split_hygiene.json`; my own matcher)
- **Mooncake slices.** All 54 windowed cells' rows map back to source rows inside their own segment
  slice. The one unmatched row is an OSL clamped at the 131,072 cap after ISL stretch: 130,682 + 390.
- **Mooncake split disjointness.** Source rows shared between splits:
  - train ∩ val = 0;
  - train ∩ test = 0;
  - val ∩ test = 0.

  Rows used: 11,558 train, 3,724 val and 8,257 test. No two segments' slices overlap.
- **Mooncake content repeats.** 11 of 4,952 test-scored Mooncake rows have byte-identical content to
  a row in a train trace. That is natural repetition.
- **Synthetic sessions.** Train seeds 0–3, validation 4–5 and test 6–9 share 0 session ids and
  0 identical requests.
- **AgentX.** 33, 21 and 28 plays in train, val and test. Plays are disjoint by name and by content
  fingerprint, the corpus has no duplicate play, and every cell's plays belong to its own split.
- **Hold-out axes.**
  - Train N = {4, 8}, val N = {4, 6, 8}, test N = {2, 4, 6, 8, 16, 32}.
  - No test segment is in train or val.
  - Every test cell at N ∈ {4, 8} has a hold-out axis.
  - Extrapolation values lie outside TRAIN_RANGES, and train and val lie inside them.
- **Validation segments.** There are 3 per load mode.

## Findings (all minor)

**m1. Session closed-loop window still contains the session drain** (`scripts/closed_drain.py`,
`scripts/closed_drain_ratio.py`, `out/sp/closed_drain*.json`)
- **What happens.** The identity rule ends the window at the last dispatch. For flat Mooncake
  closed-loop cells that is steady state: 0% of the window comes after the last new request. For
  session traces, the last dispatch is the last turn of the last session. The window therefore
  includes the period after the last new session starts, when active sessions decay from C:
  - 23.0–23.5% of the window on `sessions-s0-n8-closed` (about 302 s of 1,305 s);
  - 9.5–9.6% on `sessions-s2-think2.0-n4-closed`.
- **Effect.** Against a window that ends at the last new-session dispatch, policy/default ratios
  shift as follows (provisional load, 3–4 replicates):

  | Policy | Steady-state window | Shipped window |
  |---|---|---|
  | lmetric | +0.65% | +0.44% |
  | sticky-hard | +0.51% | +0.32% |
  | round_robin | −1.61% | −1.71% |

  So the shipped window mixes drain speed into closed-loop goodput (LR-01). This affects 10
  session closed-loop cells: 2 train, 2 val and 6 test.
- **Fix (calibration owns the rule).** End session closed-loop windows at the dispatch of the last
  new session's first turn, and record the rule.

**m2. AgentX native-lanes cells have no steady state at all; their window rule is still unset**
(`out/sp/lanes_window.json`)
- **What happens.** Plays are dealt statically to lanes. Measured at play level, in more than half
  of the window fewer plays are active than there are lanes:
  - A2-n4 (11 plays, 4 lanes): 62.6–79.1% of the window under-loaded, and 16.6–45.1% of it after
    the last play starts;
  - V1-n4 (7 plays): 63.1–97.8% under-loaded, and 40.2–66.0% after the last play starts.
- **Effect.** Lanes goodput is dominated by the longest lane's tail. This was known (A3, and the
  integration note), but it means native-lanes cells barely measure routing.
- **Action.** If `facts/agentx_lowered.json` is not ok and calibration falls back to native Weka, it
  must restrict the lanes window, for example to the period before the last play starts, or
  disclose that the AgentX cells are drain-dominated.

**m3. lr-train passes Python lists to `es.tell`, so pycma treats every candidate as injected**
(unchanged from r0 M1; new quantification in `scripts/cma_tell_ab*.py`, `out/cma/tell_ab*.json`)
- **Which spaces.** Only spaces whose samples cross a bound are affected. In the shipped spaces that
  means `default_cost_fn.yaml`, where 3 of 5 init values sit on a bound: after one generation its
  mean moves by 0.336 internal units. The learned-choice spaces give identical tells.
- **Effect.** With lr-train's own Trainer and the fake objective on the default space (30 seeds), the
  outcome depends on where the optimum lies:

  | Optimum | Budget (evals) | Gap, lists (shipped) | Gap, arrays | Wilcoxon p | Better path |
  |---|---|---|---|---|---|
  | 3 of 5 on bounds | 96 | 0.193 | 0.094 | 0.028 | arrays |
  | 3 of 5 on bounds | 240 | 0.0070 | 0.0026 | 0.007 | arrays |
  | interior | 96 | no significant difference | | 0.53 | — |
  | interior | 240 | 0.011 | 0.035 | 0.007 | lists |

  So baseline tuning behaves differently from the learned spaces, in a problem-dependent direction.
- **Fix:** `es.tell([np.asarray(z) for z in pending], fitness)`, before any tuning stage, and
  re-validate. Otherwise disclose it.

**m4. lr-train's resume identity still omits cell content and `--val-every`** (r0 M3, reconfirmed;
`out/train/Cm`)
- **Repro.** I copied a finished run and resumed it with every cell's `load.value` multiplied by
  1.5 or 2 and `sla.itl_ms` set to 30, under the same cell ids.
- **Result.** It continued silently. `selected_by_val` now compares validation objectives from two
  cell definitions: 0.104 under the old cells, 0.193 under the new.
- **Why it matters.** The contract's "recalibrate once" path changes exactly these fields.
- **Fix:** put each train and val cell's `content_sha` and `val_every` in `identity`.

**m5. Cache hits from lr-train entries carry no per-request rows** (`out/train/lr_eval_after_train.jsonl`)
- **What happens.** lr-train evaluates with `keep_per_request=False`. A later `lr-eval` on the same
  key, which keeps per-request rows by default, returns the cache hit with no `per_request_path`
  (4/4 in my check).
- **Effect.** Training-stage and validation records, including the reference default and validation
  candidates, cannot be re-scored or recomputed per request by later audits or the gate without
  `--refresh`.
- **Fix:** keep per-request rows in lr-train, or have lr-eval recompute a hit that lacks them.

**m6. Workers orphaned by a parent SIGKILL finish their replay outside a slot** (r0 M4; the code has no
PDEATHSIG)
- At each of my 3 lr-train kills, 10 worker children were alive. They were gone 3 s later, so they
  finished their in-flight replay unslotted (r0 measured ≤ 2.34 s).
- **Fix:** `prctl(PR_SET_PDEATHSIG)`, or a parent-liveness check in the worker.

**m7. The slot cap depends on every caller passing `--num-slots 20`**
- `SlotPool` uses `range(num_slots)`, and the bundle's `run.sh` defaults `--num-slots` to `nproc`
  (24 here). A bundle run on the campaign host with only `--slots-dir CR/slots`, as the fixer's next-stage
  note suggests, would use 24 slot files, 4 above the cap.
- **Fix:** store the pool size in the slots directory and refuse a mismatch, or document
  `--num-slots 20` alongside `--slots-dir`.

**m8. FAST25 conversation window boundary picks up one Mooncake *train* burst** (`scripts/split_hygiene.py`)
- **What happens.** Conversation timestamps are about floor(Mooncake ts × S). The scaled boundary
  40 min × S = 2,357,999.33 ms therefore takes in the conversation copy of Mooncake's first w4 burst
  (12 rows at 2,400,001 ms, train), as the last scored burst of the test cells `conv-w3-*`. It also
  drops w3's first burst into the gap.
- **Size.** 12 of 4,265 conversation test rows (0.28%); negligible.
- **Fix:** define conversation slices on t/S with the Mooncake [t0, t1) rule.

**m9. Unchanged r0 minors, rechecked**
- **M2.** History lines can still be duplicated by a kill in the save window. The code still writes
  history before the checkpoint. My kills did not hit that window: 0 duplicates.
- **M6.** Scoring code is still not in the cache key. HARNESS_VERSION is still `lrh-1` after
  fixer r0 changed `goodput.py`. That change only affects cells with `warmup_trace_ms`, which are new
  contents, and the rebuild changed the build_id, so no stale entry exists today.
- **M7.** ISL cap drops on test extrapolation cells: `w3-islu2.5` drops 27 rows with base ISL of at
  least 53,167, and `w5-islp2.5` drops 26 rows with base ISL of at least 56,411. Disclose these in
  REPORT.
- **M8.** FAST25 synthetic replicates are near-degenerate: 14 and 8 of about 2,000 units are in
  ties. Its noise must come from segments, of which there are only 2.
- **M10.**
  - The timeout message still prints "within 0 s" and "within 2 s" for sub-second timeouts.
  - The default-cost space still starts τ on its bound (LR-15).

**m10. A per-replay timeout also covers the first job's `import dynamo`**
- This explains why `--timeout-s 1.6` timed out 24/24, even though replays take 1.1–2.4 s.
- Harmless at the default of at least 120 s. Only explicit short timeouts are affected.

**m11. Bundle portability notes** [the node facts are a hypothesis; not verified on a node]
- The MANIFEST records a glibc floor of 2.39 but not the CPU architecture.
- `run.sh` has no preflight check for either.
- The cluster runbook notes that CPU-cluster nodes can differ in architecture.
- A mismatch fails loudly, since every replay becomes an error record that ingest rejects, but only
  after a node allocation is spent.
- **Fix:** add `platform.machine()` to the MANIFEST and a 1-line import preflight to `run.sh`.

## Literature lessons

- **Applied:**
  - LR-01: the drain inside closed-loop and lanes windows (m1, m2);
  - LR-02 and LR-11: segment disjointness and the 3 validation segments per mode (V9), plus
    degenerate replicates (m9);
  - LR-03: CRN across processes and the CRN assertion in lr-train (V3, V4);
  - LR-05: θ0 pinned and never searched (V7);
  - LR-10: validation selection survives kill and resume (V3), and resume identity (m4);
  - LR-15: τ on the bound in the M0 space (m9).
- **Rejected:** LR-04 and LR-08, which A2 supersedes.
- **Out of this lens:** LR-06, 07, 09, 12, 13 and 14.
