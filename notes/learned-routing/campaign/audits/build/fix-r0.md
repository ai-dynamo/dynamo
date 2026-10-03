# Fixer response: build checkpoint, round 0

- **Role:** fixer, checkpoint "build", round 0. 2026-10-02 15:50–16:15 PDT.
- **Input:** the four major findings relayed from `leakage-features-r0.md` (F1),
  `goodput-normalization-r0.md` (F1) and `harness-robustness-splits-r0.md` (S1, B1).
- **Verdict:** all four are **valid**. All four are **fixed at the root**, with code committed on
  WT and the affected tests and smoke runs re-run. No finding is rebutted.
- **WT:** `rupei/learned-routing-public`, three signed-off commits on `<commit-07>`:
  - `<commit-08>` fix(router-plugins): key learned-choice hash_home on the prompt prefix only
  - `<commit-09>` fix(learned-routing): pair report ratios on cell content and verify bundles
  - `<commit-10>` fix(learned-routing): use disjoint workload windows and identity warm-up
  No co-author trailer, nothing pushed.
- **Bindings rebuilt** (release, `ais-forward-pass`, 2 min incremental) for the Rust change:
  `.so` sha `ecdd2202…` → `1d67c131…`, build_id `4b4525a9…` → **`6955b0ee…`**. Cache entries
  under the old build are unreachable; none were calibrated results.
- **Evidence root:** `runs/build-fix-r0/`.
- Every replay ran through `lr-eval` and the `CR/slots` pool, except the per-cell replay check
  (`runs/build_workloads/scripts/validate_cells.py`, B2's script, one slot held, sequential).

## leakage-features F1 (major): v2 `hash_home` keyed on replay-synthesized session IDs — VALID, FIXED (option b)

**Confirmed.** aisimulate-core `replay/loadgen/trace.rs:446` gives every flat trace row without a
`session_id` the ID `request_<line>`, and the closed-loop driver passes it to `place()`
(`replay/agg.rs:667-673`), so the router sees a per-request session ID in closed loop and none in
open loop. `features.rs` keyed `hash_home` on that ID whenever present.

**Fix (option b, strict form).** `hash_home_key` now uses only the prefix hash of the prompt's
first 256 tokens (whole blocks, at least one), exactly as `chwbl` keys it. Without prefix hashes
no worker is home. The session-ID branch and FNV-1a helper are gone.

- I did not keep a session-ID fallback for requests without prefix hashes. A fallback would
  reintroduce an environment-dependent key, and replay always supplies prefix hashes.
- Session continuity remains `session_affinity`'s job.
- Option (a) alone would have left a broken feature definition in v2.
- Option (c) changes `session_context` for every policy and needs a vendored aisimulate-core. It
  would also remove the per-request identities that the closed-loop warm-up rule below relies
  on. Recorded instead as upstream follow-up #6.

**Documentation.** FEATURES.md (#11 definition plus a paragraph on why) records the change. It
also flags the LR-13 concentration property the auditor noted: Mooncake has 4 first-block keys,
and two of them hold about 85% of rows.

**Unit test.** `v2_features_follow_their_definitions` now asserts three things:
- the home set is identical with session `s` and with `request_17`;
- it matches the sessionless request;
- with `token_seq = None`, no worker is home.

**Checks.**
- `cargo test -p dynamo-custom-policy-builtin`: 127 passed, 2 ignored (pre-existing).
- `clippy -D warnings` and `fmt` are clean.

**Dynamic re-check.** I re-ran the auditor's θ₁₁ = 1 spec on the same two cells, with the same
derived trace `a1be1df5…`, at build `6955b0ee` (`runs/build-fix-r0/hash_home/`):

| Cell | Workers per first-block key (keys 0 / 46 / 74 / 24294) |
|---|---|
| open loop | 2 / 2 / 2 / 2 |
| closed loop, before the fix (audit) | 8 / 8 / 8 / 2 |
| **closed loop, after the fix** | **2 / 2 / 2 / 2** |

The per-key home sets are identical in open and closed loop (`out/home_sets.json`,
`open_equals_closed: true`). The closed-loop rows still carry 4,249/4,249 session IDs, so the
feature no longer reads them.

**No v1 or baseline regression** (`runs/build-fix-r0/postbuild/parity.json`). 11 provisional
`int-` cells × 2 replicates × 9 policies = 154 replays, 0 errors.
- `learned-choice@theta0-reservoir` equals `default@defaults` request for request on 22/22.
- `default@defaults` per-request rows are byte-identical to the pre-rebuild integration smoke on
  14/14 (cell, k).

## goodput-normalization F1 (major, latent): lr-report paired by (cell_id, repeat) only — VALID, FIXED

**Confirmed.** I reproduced it with the auditor's script.

**Fix** (`report.py`, `report_cli.py`, `train_cli.py`):

- **Pairing key.** `paired_ratios` pairs on (cell_id, repeat, cell_sha, build_id,
  replicate_protocol, harness_version), which is `CONTENT_KEYS`. A record without a same-content
  reference is dropped and counted, never paired across contents.
- **Mixed inputs.** `Report` and `lr-report` raise `MixedContentError` (CLI exit 2) when one
  cell_id carries more than one content, unless the caller makes an explicit selection:
  - `--build-id PREFIX` keeps one build;
  - `--mixed-contents split` reports each content as its own cell, `<cell_id>@<digest>`, which
    suits calibration sweeps and Stage 5 timing perturbations.

  Raw means and the bootstrap therefore can no longer pool two contents under one cell either.
- **lr-train.** `SLIM_FIELDS` now carry `cell_sha`, `build_id`, `replicate_protocol`,
  `harness_version` and `trace_sha256`.

**Evidence** (`runs/build-fix-r0/report_pairing/`):

- **Auditor demo** (`runs/audits/build-goodput-normalization-r0/scripts/report_pairing.py`, re-run
  on the fixed code):

  | | Before | After |
  |---|---|---|
  | candidate ratio | [0.55] | **[1.1]** |
  | reference self-ratios | [0.5, 1.0] | **[1.0, 1.0]** |

- **Real trigger.** The fix met a real case immediately: the integration smoke (build `4b4525a9`)
  plus the post-rebuild smoke (build `6955b0ee`) hold the same `int-` cell_ids under two builds.
  - `lr-report` now refuses that input (exit 2), naming 7 cell_ids
    (`mixed_builds.stderr`).
  - With `--build-id 6955b0ee` it reports the 154 records cleanly.
- **Tests.** New tests `test_paired_ratios_match_cell_content_not_just_cell_id` and
  `test_report_rejects_or_splits_a_cell_id_with_several_contents`.

## harness-robustness-splits B1 (major): remote bundle missing `packaging`; workers ignored `-S` — VALID, FIXED

**Fix** (`bundle.py`, `pool.py`):

1. **Ship `packaging`.** `packaging` and its dist-info are now in the site (`SITE_PACKAGES`,
   `DIST_INFOS`).
2. **Inherit isolation flags.** Workers start with the parent's isolation flags
   (`pool.isolation_flags()`: `-S`, `-s`, `-E`, `-P`, or `-I`). A bundle parent run as
   `python -S` therefore gets `-S` workers.
3. **Check the site at build time.** `build_bundle` runs `check_site` before writing the manifest.
   - It runs `python -S -s -P` from `site/`, with `PYTHONPATH=site` and a minimal environment.
   - It imports what a worker imports and builds the bundled engine's `MockEngineArgs` and a
     `KvRouterConfig`.
   - It fails the build on any import error, or if any loaded module comes from outside `site/`
     or the stdlib.
   - The result is recorded as `MANIFEST.site_check`.

**Evidence** (`runs/build-fix-r0/bundle/`):

- **Bundle and site check.** The bundle `b2` has 11 `int-` cells × 4 specs × 2 replicates = 88
  tasks, build `6955b0ee`. `site_check`: 291 modules, 0 outside site/stdlib.
- **Negative control.** The same site with `packaging` renamed away fails the check with
  `ModuleNotFoundError: No module named 'packaging'` (`site_check_without_packaging.json`). I then
  renamed it back.
- **Clean-room run** (the audit's failing condition). I copied the bundle with tar to `remote/`
  and ran it under:

  ```
  env -i HOME=… PATH=/usr/bin:/bin PYTHON=<fresh empty uv venv python3.12> strace -f bash remote/run.sh --slots 4 --slots-dir CR/slots --num-slots 20
  ```

  Results (`remote_checks.json`):
  - **88/88 OK, 0 errors** (audit: 88/88 ModuleNotFoundError).
  - All 4 worker execs carry `-S`.
  - 88/88 equal the local runs of the same cache keys on `per_request_canonical_sha256` and 12
    metric fields.
- **Ingest.** `lr-eval --ingest`: 0 rejected, 88 already cached (keys identical to local).
- **File access.** Outside the bundle, the strace shows only:
  - `CR/slots` flocks;
  - parent-directory stats;
  - the leftover RPATH probes into `WT/lib/bindings/python/target/release/build/*`, all ENOENT.
    This is the auditor's minor note, unchanged.
- **Tests.** New tests `test_workers_inherit_the_parents_isolation_flags` and
  `test_site_check_sees_only_the_site` (an empty site must fail, so the check cannot be satisfied
  by the venv or dist-packages).
- **Facts.** `facts/build_harness.json` smoke.bundle is corrected (field `correction_fix_r0`).

## harness-robustness-splits S1 (major): Mooncake windows overlapped across splits — VALID, FIXED

**Fix** (`workloads/cells.py`, `goodput.py`, `worker.py`; `CELLS_VERSION` lr-cells-v2):

**1. Mooncake windows.** There are now six disjoint, self-contained 10-minute slices `[10k, 10k+10)`
minutes, each a 4-minute warm-up plus a 6-minute measurement window. The 60-minute trace allows
6; the auditor's 12-minute variant allows 5.

| Split | Windows |
|---|---|
| train | w0, w2, w4 |
| val | w1 |
| test | w3, w5 |

- No source row is in two slices, so no split replays another split's rows, even as warm-up.
- I chose 6 × 10 minutes over 5 × 12 to keep 3 train windows and 2 test windows.
- **Cost:**
  - shorter measurement windows: about 2,000–2,500 scored rows each, against about 3,200 before;
  - one validation Mooncake window instead of two.

  The two old val-w5 open cells moved to w1. The val-w5 closed L1 cell became
  `sessions-s5-base-n4-closed-L1`, so that validation keeps 3 independent segments per load mode
  (open and closed: mooncake:w1, sessions:s4, sessions:s5; lanes: V1–V3). The existing test
  `test_validation_has_three_segments_per_load_mode` enforces this.
- Old w6 test cells are now w5, and FAST25 conversation is aligned to w3/w5.

**2. FAST25 synthetic.** Its two test windows also overlapped (x0 `[0,10)`, x1 `[7,17)` minutes),
which the audit did not flag because both are test. They are now disjoint: x0 `[0, 8.5)`, x1
`[8.5, 17)` minutes, each with a 3-minute warm-up.

**3. Guard.** `validate()` rejects any plan whose windowed segments of one family share trace
time (`overlapping_slices()`). The new test `test_windowed_segments_share_no_trace_rows` also
shows that the old 8-minute stride fails it.

**4. Closed-loop identity warm-up.** Every closed-loop cell (27: Mooncake, FAST25 and synthetic
sessions) now carries `measure.warmup_trace_ms`. The rule is load-independent, so the generator
fixes it.
- **Exclusion.** Every request of a session whose first arrival in the replayed trace is less
  than `warmup_trace_ms` after the first arrival is excluded by identity.
  - The identity is the per-request `session_id`. That is replay's `request_<line>` for flat rows,
    recomputed by `goodput.warmup_ids_from_trace` from the exact replicate trace replayed.
- **Window.** It opens at the first dispatch of a measured request and ends at the last
  dispatch.
- **Errors.** Missing session IDs, a non-completion basis, or a combination with `warmup_ms`
  raise an error.
- **Open loop is unchanged.** It still uses the arrival-time window that calibration converts from
  `measure_trace`.

**Regeneration.** Old candidates and manifest are preserved in `cells/superseded/build-r0/`
(SPLIT_MANIFEST `791d6787…`).
- New files:

  | File | SHA-256 |
  |---|---|
  | `cells/SPLIT_MANIFEST.json` | `0f8ea452…` |
  | train | `4ca440c7…` |
  | val | `059ce901…` |
  | test | `f8fd8d39…` |

  The cell counts are unchanged at 34 / 14 / 56.
- 24 new derived traces were materialized. The 25 superseded ones stay in `traces/derived`; see
  CLEANUP.

**Re-checks:**

- **Split hygiene** (`runs/build-fix-r0/workloads/split_hygiene.py`; it extends the auditor's
  script to every windowed family, every segment pair, the derived traces themselves and the
  identity rule). Output: `out/split_hygiene.json`.
  - **0** segment pairs share a source row, in Mooncake, conversation and FAST25 synthetic.
  - Every cross-split count is **0**:
    - test-scored rows in train traces: 0 (audit: 1,634 / 3,245 of w3);
    - train or val scored rows in test traces: 0;
    - test and train trace rows shared: 0 (audit: 3,221).
  - All 54 windowed cells' derived traces lie inside their own slice (source time from
    `window_rebase_ms`).
  - Hold-out axes have no problems.
  - AgentX and synthetic sessions are still disjoint.
  - 7 of 4,962 Mooncake test-scored rows have byte-identical content scored in train elsewhere in
    time. This is natural repetition; the audit counted 17 under the old windows.
- **Per-cell replay check.** All 104 candidate cells have a replay check at the current trace SHA
  (`runs/build-fix-r0/workloads/cells_validation.json`). 55 are new replays: 54 windowed cells plus
  the new sessions cell, at B2's provisional loads, at build `6955b0ee`.
  - 0 errors, and every request completed.
  - Each of the 17 closed-loop flat cells carries one session ID per row, which the identity rule
    needs.
  - `SPLIT_MANIFEST.replay_check` records all 104, with no stale or missing cells.
- **Identity rule end to end** (`runs/build-fix-r0/identity_smoke/`). 6 provisional `fix-` copies
  of train/val cells (none from test), × 4 policies × 2 replicates = 48 replays, 0 errors.
  - On all 32 closed-loop records, the excluded rows are exactly the trace's warm-up sessions, and
    the distinct excluded sessions equal `sessions_in_warmup`.
  - The first measured dispatch comes after every warm-up first-turn dispatch.
  - θ0-reservoir equals default 12/12.
  - Example (mooncake-w0 n8 closed, default k=0): 1,395 warm-up rows are excluded, and the window
    starts at 242.1 s. Without the rule 3,390 completions would be scored; with it, 1,995.
- **pytest.** 114 passed: 107 before, plus 7 new.

**Not done; calibration's call.** The pilot's doubled-warm-up sensitivity check (LR-01) can no
longer extend into the preceding slice inside a split cell. Run it as a diagnostic copy that
prepends the previous 4 minutes. That is fine for a sensitivity check, but never as a split cell.

## Literature lessons

- **Applied:**
  - **LR-11** (independent segments): disjoint slices; segment-level split guard.
  - **LR-02** (segments as the noise unit): 3 val segments per load mode kept.
  - **LR-01** (windowed scoring with warm-up): closed-loop steady-state window by identity;
    Decima's caution kept, since late finishers in open loop still count.
  - **LR-14** (information parity: no replay-only state in features): hash_home.
  - **LR-07** (hash_home keyed by "prefix root").
  - **LR-13** (concentration risk of a 4-key prefix home, documented for training).
- **Rejected:** LR-01's roughly 10-minute warm-up. A 4-minute warm-up is kept so that six disjoint
  windows fit in 60 minutes.
- **Out of scope here:** LR-03–06, 08–10, 12, 15.
