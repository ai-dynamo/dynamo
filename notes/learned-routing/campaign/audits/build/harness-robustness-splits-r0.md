# Audit: build checkpoint, lens "harness-robustness-splits", round 0 (dynamic)

- **Auditor:** independent dynamic auditor (adversarial), 2026-10-02 15:18–15:48 PDT.
- **Audited state:** WT `rupei/learned-routing` at `<commit-07>` (clean), bindings `.so` sha
  `ecdd2202…`, build_id `4b4525a9…`. Cells: `CR/cells/{train,val,test}.candidates.jsonl`
  (SPLIT_MANIFEST sha `791d6787…`).
- **Evidence root:** `CR/runs/audit-build/harness-robustness-splits-r0/` (scripts in `scripts/`,
  outputs in `out/`). The bundle scratch is in the session scratchpad,
  `<session-scratch>`.
- **Audit cells:** the integration's 11 provisional cells (`runs/integrate/smoke/cells_provisional.jsonl`
  and `cells_parity_extra.jsonl`), relabeled `aud-<test>-…` so every replay was fresh
  (`out/../cells/*.jsonl`, `scripts/make_cells.py`). Their loads and SLAs are the integration's
  PROVISIONAL values. No metric below is a calibrated campaign result.
- **Discipline:** every replay ran through `lr-eval`, `lr-train` or the bundle's `lr-eval`, all
  holding `CR/slots`. There is one exception, which was deliberate and is recorded in
  DEVIATIONS: after the parent SIGKILL test (H4), 20 orphaned workers finished their current replay
  outside a slot, for at most 2.34 s. Processes were stopped only by exact PID, after their cmdline
  and parent were verified. Nothing was deleted. No WT file was changed.

## Verdict: FAIL (0 blocker, 2 major, 10 minor)

The harness is robust where it counts:

- the slot cap held with 3 lr-eval processes plus other agents;
- timeouts and crashes are recorded and never cached;
- lr-train resumes exactly after `kill -9`;
- CRN pairing is intact;
- every manifest SHA matches;
- every derived trace matches its spec.

Two things fail, though:

1. **Major S1.** The Mooncake time windows overlap across splits. Half of test window w3's scored
   requests are replayed inside train cells, and train and val scored requests are replayed as the
   test windows' warm-up.
2. **Major B1.** The remote bundle is not self-contained. It lacks `packaging`, so in a clean
   Python every replay fails. B3's "hermetic" pass relied on the host's
   `/usr/lib/python3/dist-packages`, because worker subprocesses run without `-S`.

## Findings

### S1 (major): Mooncake windows overlap across splits, so test windows are not held out

- **Claim.** Each Mooncake slice is `[8k, 12+8k)` minutes, with measurement `[4+8k, 12+8k)`. Every
  window's 4-minute warm-up is therefore the last 4 minutes of the previous window's measurement.
  The split order is train, val, train, test, train, val, test, so every adjacent pair straddles
  two splits.
- **Evidence** (`out/workloads/split_hygiene.json`, `scripts/split_hygiene.py`; source rows of
  `traces/mooncake/mooncake_trace.jsonl`):
  - **Test w3 scored rows.** 1,634 of the 3,245 rows in test w3's measurement window (50.4%;
    t ∈ [1920 s, 2160 s)) are the warm-up rows of every train w4 cell.
  - **Test w3 warm-up.** The w3 warm-up (1,587 rows, t ∈ [1440 s, 1680 s)) is exactly train w2's
    scored rows.
  - **Test w3 trace overall.** 3,221 of the 4,832 rows in the w3 test trace (66.7%) also appear in
    train traces.
  - **Test w6 warm-up.** The w6 warm-up (1,676 rows) is val w5's scored rows.
  - **Train and val overlap too.** w0→w1, w1→w2 and w4→w5 overlap in the same way, so train cells
    replay val-scored rows and the reverse.
  - **Mooncake test totals.** Of 6,591 test-scored rows, 1,634 (24.8%) are replayed in train
    traces. Of 9,854 test-trace rows, 3,221 also appear in train traces.
  - **Closed-loop test cells.** Seven closed-loop Mooncake test cells (e.g.
    `mooncake-w3-base-n8-closed-L2`, `mooncake-w6-base-n32-closed-L2`) replay the whole slice in
    file order. The "steady-state window" rule is still unset (`measure.basis = completion`), so
    their scored set may include the train- or val-scored warm-up rows outright.
  - **FAST25 conversation.** The conversation test cells are aligned to w3/w6 and share the
    Mooncake skeleton, so they inherit the same overlap.
  - **Disclosure.** DEVIATIONS describes these as "7 independent windows", and SPLIT_MANIFEST
    describes w3 and w6 as "held-out Mooncake windows".
  - **What stays clean.** Within each split, windows are disjoint (train w0, w2, w4; val w1, w5;
    test w3, w6). Only 17 test-scored rows have byte-identical content to a train-scored row
    elsewhere in time, which is natural repetition.
- **Why it matters.**
  - The time-window hold-out and the cross-split independence (LR-11) are overstated.
  - Training replays half of w3's scored requests, unscored, as the initial state of the w4 cells.
  - Model selection on val replays w5 rows that seed test w6.
  - For a 7–23-parameter model the direct overfitting channel is probably small (hypothesis), but
    the claim would mislead, and the closed-loop channel can score train requests directly.
- **Fix** (cheap now, before the test freeze):
  - Make cross-split windows non-overlapping, either:
    - with contiguous same-split blocks separated by a discarded 4-minute buffer; or
    - with 12-minute slices that contain their own warm-up (5 windows), accepting fewer segments.
  - Define the closed-loop window rule so it excludes the warm-up prefix rows by identity.
  - Re-run `learned_routing.workloads.cells` and re-check with `scripts/split_hygiene.py`.

### B1 (major): the remote bundle is incomplete, and its "hermetic" test passed only through the host's dist-packages

- **Claim.** `lr-eval --bundle-out` does not ship `packaging`. That package is imported by
  `aisimulate_core/sdk/common.py:13` on every `MockEngineArgs.from_json` call, through
  `aisimulate_core.sdk.__getattr__`.
  - `run.sh` runs the parent with `python -S`.
  - `pool.py:89` starts each worker as `[python, "-m", "learned_routing.worker"]` **without `-S`**,
    so workers import the interpreter's site-packages.
  - B3's test (`facts/build_harness.json` smoke.bundle.hermetic_run) used the system `python3.12`,
    whose workers silently picked up `/usr/lib/python3/dist-packages/packaging`.
  - With an interpreter that has no `packaging`, every replay fails. That includes the
    uv-managed CPython that `run.sh` falls back to when a node has no `python3.12`.
- **Evidence** (`out/bundle/`; scratchpad `aud-bundle/`):
  - **Setup.** I built a bundle (11 cells × 4 specs × 2 replicates = 88 tasks) and transferred it
    with `tar` to `remote/`. I ran it as
    `env -i HOME=<scratch> PATH=/usr/bin:/bin PYTHON=<fresh uv venv python3.12> bash run.sh --slots 4 --slots-dir CR/slots --num-slots 20`.
  - **Result:** 88 of 88 errors, all `ModuleNotFoundError: No module named 'packaging'`
    (`run.log`, `remote/results.jsonl`).
  - **B3's own bundle fails the same import.** `bundle_test/b1/site` fails the same
    `MockEngineArgs.from_json` call under `python3.12 -S`.
  - **With `packaging` added, the bundle is complete and exact.** I copied `packaging` into a second
    unpacked copy (`remote_fixed/site/packaging`) and ran it under `strace -f -e trace=%file`.
    - 88 of 88 replays succeeded.
    - All 88 matched the local runs of the same cell content (`out/sp/p1.jsonl`, `p2.jsonl`) on
      `per_request_canonical_sha256` and 12 metric fields (`out/bundle/remote_vs_local.txt`).
    - `lr-eval --ingest` took 88, rejected 0, and a local rerun gave 88 of 88 cache hits.
    - Apart from the slots dir I passed in, nothing touched `CR`, `WT/.venv` or `~/.cargo`.
  - **Leftover build paths in the `.so`.** The `.so` RPATH points into
    `WT/lib/bindings/python/target/release/build/{zmq-sys,esaxx-rs,nixl-sys}`. The loader probes it
    (ENOENT) and falls back to the system libraries. NEEDED lists only libstdc++, libgcc_s, libm
    and libc.
- **Fix:**
  - Add `packaging` and its dist-info to `bundle.build_site`.
  - Start workers with the parent's isolation flags: `-S` when `sys.flags.no_site`, or always
    `-S` with an explicit `sys.path`.
  - Re-test with a clean uv venv, as above.
  - Correct `facts/build_harness.json` smoke.bundle.

### Minor

**M1. lr-train passes `tell()` Python lists, so pycma's genotype archive is bypassed.**
- **What happens.** `train.py` stores `pending` as float lists and calls `es.tell(pending, …)`.
  `es.sent_solutions.get(list)` misses: `np.asarray(list)` hits, a list does not. pycma therefore
  treats every candidate as an unknown or injected solution. It uses the inverse BoundTransform of
  the phenotype, plus `repair_genotype`, instead of the sampled genotype.
- **Size.** After one generation on `default_cost_fn.yaml`, the mean differs by 0.36 in internal
  units from `tell(arrays)`, and sigma is 0.229 vs 0.189 (`scripts/cma_checks.py`,
  `out/cma/checks.json`).
- **No measured degradation.** Over 4 synthetic problems × 10 seeds, final quality was comparable.
  The list form was worse in 6/10, 3/10 and 5/10 seeds (`out/cma/tell_lists.json`).
- **Resume exactness still holds,** because both the uninterrupted and the resumed paths use lists.
- **Fix:** `es.tell([np.asarray(z) for z in pending], fitness)`, which I checked makes the lookup
  hit.

**M2. A kill between a history append and the next checkpoint duplicates history lines.**
- **Evidence.** I simulated a SIGKILL right after each of the 9 history lines of a fake-objective
  run, then resumed. 9 of 9 cases left 1–2 duplicate `gen`/`val` lines, while the deduplicated
  history, `best.json` and the CMA state equalled the reference (`out/train_fake2/report.json`).
- **Fix:** consumers of `history.jsonl` should deduplicate on (kind, generation, which), or the
  history should be written after the checkpoint.

**M3. lr-train's resume identity does not cover cell content or `--val-every`.**
- A copy of a finished run resumed silently after every cell's `load.value` and `sla.itl_ms` had
  been changed under the same `cell_id`s (`out/train_fake2/ref_resume_modcells`). After the
  contract's "recalibrate once" path, a resumed pilot run would mix CMA state across cell
  definitions.
- **Fix:** add each cell's `content_sha` (train and val) and `val_every` to `identity`.

**M4. Orphaned workers outlive a SIGKILLed parent.**
- After I SIGKILLed an lr-eval parent (exact PID), its 20 workers, now with ppid 1, finished their
  current replay outside any slot, for at most 2.34 s (`out/kill/orphans.json`).
- The rerun reused the 6 records cached before the kill and computed the rest (exit 0).
- **Fix:** `prctl(PR_SET_PDEATHSIG, SIGKILL)` in the worker, or a parent-liveness check.

**M5. There is no per-key lock, so concurrent processes duplicate work.**
- p1 and p3 ran the same 66 tasks at the same time, and both ran all 66 fresh. The results were
  identical (66 of 66 on the per-request sha and metrics), so only CPU was lost (`out/sp/analysis.txt`).

**M6. Scoring code is not part of the cache key.**
- The key's only harness identity is the manually bumped `HARNESS_VERSION = "lrh-1"`.
- If a fixer changes `goodput.py`, `e0.py` or `worker.py` without bumping it, for example after
  the parallel goodput audit, stale cached metrics are served silently.
- **Fix:** hash those modules into the key, or add a test that fails when they change without a
  bump.

**M7. ISL-stretched cells drop their longest requests at the context cap.**
- The 7 ISL-multiplier traces lost 126 rows: 6–36 per trace, 0.12–0.75%. This is recorded in each
  trace's `transform_stats.dropped_isl_at_cap`, and my independent count matches.
- The test extrapolation cells `w3-islu2.5` and `w6-islp2.5` drop 36 and 33 rows with base ISL of
  at least 54.6K and 56.4K. ×2.5 therefore excludes exactly the longest prompts.
- Disclose this in REPORT.

**M8. FAST25 synthetic replicates are near-degenerate.**
- Only 0.58–0.80% of units are in ties, but `replicate_degenerate` is False
  (`out/crn/crn_checks.json`). This confirms B2's note.
- Take this family's noise from segments, not replicates. The "zero spread" guard will not trigger.

**M9. Bundle operational notes.**
- `run.sh` defaults to a bundle-local slot pool. Running a bundle on the campaign host therefore
  bypasses `CR/slots` unless `--slots-dir` is passed.
- The manifest's `glibc_floor` (2.39) omits the libstdc++ floor (GLIBCXX_3.4.30).
- The bundle keeps a `src/` build copy.
- [hypothesis] If one bundle directory on a shared filesystem serves several nodes:
  - the bundle-local slot pool would be shared across them;
  - `runs/tmp/jobs/<pid>-<thread>.json` names could collide across hosts.
- `--ingest` appends every row of the bundle's `results.jsonl` on each call; lr-report
  deduplicates by `cache_key`.

**M10. Smaller items.**
- **Timeout message.** It formats `timeout_s` with `:.0f`, so sub-second test timeouts read
  "within 0 s".
- **Integer coordinates.** Bounded `int` coordinates give the two endpoints half the mass of
  interior values (1–4: 0.167/0.333/0.333/0.167). No shipped space uses `int`.
- **`default_cost_fn.yaml`.** It initializes 3 of 5 coordinates on a bound. BoundTransform never
  samples them exactly. In generation 0:
  - `router_temperature` has median 0.085 and maximum 0.57, so every generation-0 candidate routes
    stochastically;
  - `overlap_score_credit_decay` has median 0.40.

  The mean can still converge to a boundary optimum (3.8e-7 after 60 generations). For the M0
  baseline, consider pinning τ = 0 per LR-15, as the learned-choice spaces do.

## What I verified (no finding)

**H1. Slot pool under 3 concurrent lr-eval processes** (`out/sp/`):
- **Setup.** 198 replays, each process with `--slots 20`. Two other auditors were using the pool
  at the same time (six foreign PIDs held slots).
- **Monitor.** A /proc/locks monitor took 1,847 samples at 50 ms. It never calls flock itself, so
  it does not perturb the pool.
- **Cap.** At most 20 slots were held and at most 20 workers were busy (≥ 50% CPU). No sample
  exceeded 20, and no slot ever had two holders.
- **Fairness.** Slot-time shares were 415, 411 and 410 samples.
- **Correctness.** 0 errors, and cache entries are consistent.

**H2. Timeouts** (`out/to/`):
- With `--timeout-s 0.4`: 22 of 22 timeout error records, exit 2, none cached, no leftover workers.
- A rerun without the timeout: 22 of 22 OK, identical to the concurrent run of the same content
  (`out/sp/p1_vs_to_identity.txt`).
- `--max-wall-seconds 1`: exit 3, 6 done and 60 skipped. The next call reused those 6 from cache,
  ran the other 60, and exited 0.

**H3. Worker crash** (`out/crash/`): SIGKILL to one busy worker (exact PID, a verified child of my
lr-eval) produced 1 `worker_crashed` record, not cached; the other 43 were OK.

**H4. lr-eval parent SIGKILL** (`out/kill/`): the 6 records written before the kill were cached and
reused exactly on the rerun, and the remaining 60 were recomputed. Orphans: see M4.

**H5. lr-train kill -9 and resume, real replays** (`out/train/`):
- **Interrupted run B.** Seed 8, `default_cost_fn.yaml`, popsize 6, budget 24, 2 train cells and
  1 val cell, `clipped_log_ratio`. It was SIGKILLed three times through a guarded exact-PID killer
  (`scripts/kill_when.py`):
  - mid generation 1, after 40 result lines;
  - during validation;
  - mid generation 4.

  It was then resumed to completion.
- **Comparison with an uninterrupted run C** (same arguments, `out/train/compare_B_C.txt`):
  - `history.jsonl` identical, with no duplicates;
  - `best.json` identical, and `best_policy.yaml` identical modulo the run name;
  - CMA mean, sigma and C bit-identical;
  - fevals 24 and 24, tasks 130 and 130.
- **Continuation.** Extending both copies to budget 36 also matched exactly
  (`compare_Bx_Cx.txt`).
- **Changed space.** A changed space was correctly refused ("resume mismatch: ['space_sha']").

**H6. Common random numbers** (`out/crn/crn_checks.json`):
- **Shared workloads.** Across 3 processes and 4 policies, each of the 33 (cell, k) pairs used
  exactly one replicate trace. Each cell had 3 distinct traces over k = 0..2.
- **Seeds.** Seeded types got seed k+1; round-robin and lmetric got none.
- **lr-train.** It asserts (cell, k, policy_sha) coverage per generation, and the run-C records
  agree.
- **Units in ties per family:**

  | Family | Units in ties |
  |---|---|
  | Mooncake and FAST25 conversation | 100% |
  | AgentX | full play-order permutation |
  | Synthetic sessions | 60–67% |
  | FAST25 synthetic | < 1% (see M8) |

**H7. CMA-ES bounds and transforms** (`out/cma/checks.json`):
- pycma 4.5.0 with BoundTransform.
- For all 3 shipped spaces, the generation-0 favourite decodes exactly to `init`, with the pinned θ0
  = −1 not searched.
- 320 decoded samples per space all lay within bounds.
- Log endpoints map exactly (0.1, 1.0, 10.0).
- Mixed `None` and bounded coordinates are accepted.
- A boundary optimum is reachable.

**H8. Remote bundle with `packaging` added:** exact, self-contained, ingest round trip OK (see B1).

**H9. Manifests** (`out/workloads/manifest_checks.json`):
- all 96 `traces/MANIFEST.json` SHA-256s and row counts;
- the SPLIT_MANIFEST file SHAs and its traces-manifest SHA;
- all 57 derived-trace SHAs and their meta SHAs;
- all 104 cells' `trace_sha256`, `transform` == `meta.spec`, and source SHAs.

All match.

**H10. Derived traces against their specs** (`scripts/derived_checks.py`, `out/workloads/derived_checks.json`):
- **Method.** An independent re-derivation that does not import `learned_routing.workloads`. All
  57 of 57 derived traces pass.
- **What it checks:**
  - window slicing and rebase;
  - dense renumbering in first-appearance order;
  - `osl_mult` with cap clamps;
  - `think_mult` on delays;
  - `isl_unique_mult`: unique suffix in {⌊nm⌋, ⌊nm⌋+1} and the shared prefix kept 1:1;
  - `isl_prefix_mult`: consistent expansion across rows sharing a prefix;
  - `prefix_root_mult`: only the root-segment blocks are duplicated, at most k copies, and families
    stay intact;
  - context-cap drops and clamps;
  - AgentX:
    - plays match the spec subset and order;
    - inputs and hash ids unchanged;
    - outputs and `api_time` scaled;
    - total busy time preserved;
    - idle gaps at most 300 s × think_mult (maximum 1200 s on the T1 think4.0 cell).
- **Realized ratios:**

  | Multiplier | Realized |
  |---|---|
  | islu 1.5 | 1.496 |
  | islu 2.5 | 2.497 |
  | islp 1.5 | 1.547 |
  | islp 2.5 | 2.553 |

**H11. Split hygiene outside Mooncake** (`out/workloads/split_hygiene.json`):
- **AgentX.** 33/21/28 plays in train/val/test, disjoint by file name, play id and request-sequence
  content. Every cell's plays belong to its own split.
- **Synthetic sessions.** 0 identical requests and 0 shared session ids between train seeds 0–3 and
  test seeds 6–9. Block-id ranges overlap across seeds, but that is meaningless because each replay
  starts with empty state.
- **Hold-out axes.** No test cell has a worker count in train N, an extrapolation value inside the
  train ranges, or a train segment.
- **pytest.** 107 of 107 pass on the audited tree (`out/pytest.txt`).

## Literature lessons

- **Applied:**
  - LR-03: CRN pairing per generation and per process (H6), and the lr-train CRN assertion (H5).
  - LR-01 and LR-11: windowed warm-ups and segment independence. This produced S1: the LR-01
    warm-up is implemented by overlapping adjacent windows, which breaks LR-11's cross-split
    independence.
  - LR-02: replicate noise and degeneracy (M8).
  - LR-05: pinned θ0 is excluded from the search (H7).
  - LR-10: validation-selected `best.json` and `best_policy.yaml` survive resume (H5).
  - LR-15: advisory in M10 (pin τ for the M0 space).
- **Not applied:** LR-04 and LR-08 are superseded by Amendment A2. LR-06, 07, 09 (beyond the AgentX
  lanes check), 12, 13 and 14 are outside this lens.
