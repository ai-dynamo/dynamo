# Build checkpoint, fixer round 2

- Fixer: build fixer r2, 2026-10-02 (about 17:24–17:50 PDT).
- Input: `audits/build/goodput-normalization-r2.md`, finding F1 (major). It was the only finding
  assigned to this round. Minors F2–F7 of that audit are still open (see the end).
- WT `rupei/learned-routing-public`: base `<commit-13>` (another stage's AgentX transform commit, which
  landed while this round ran), head **`<commit-14>`** (+1 signed commit). Bindings were not rebuilt:
  build_id `6955b0ee…` is unchanged and the fix is Python only.
- `HARNESS_VERSION` `lrh-3` → **`lrh-4`**. E0 method `ais-chunked-estimator-v1` →
  **`ais-chunked-estimator-v2`**.
- Evidence: `runs/build-fix-r2/` (`scripts/`, `out/`, `logs/`, `smoke/`, `train_tiny/`,
  `idle_check/`).

## F1 (major): E2E knife-edge at S × scale = 1. VALID, FIXED

### Root cause, confirmed in code and by measurement

Both defects are real, and **both** had to be fixed (see the tolerance-only and E0-only rows
below).

1. **E0 decode context was off by one.**
   - Replay's vLLM scheduler (aisimulate-core `src/engine/scheduler/vllm/core.rs`, around lines 2820
     and 3000) times a decode step with `context_length = sequence.len()`. For the step that
     produces output token j + 2, that is ISL + j + 1.
   - The mocker turns this into `callback.predict_decode(batch, context_length, 2)`
     (`WT/lib/mocker/src/common/perf_model.rs:297`). That is one step at `context_length + 1`
     (`WT/lib/bindings/python/rust/llm/ais_callback.rs:105-117`: "Mocker uses osl=2, i.e. one
     generation step at isl+1").
   - So replay runs step j at KV context **ISL + j + 2**. `e0.py` used ISL + j + 1.
2. **The comparison had no tolerance.** `a2_good` tested `e2e > scale·S·E0`, so float noise of
   about 1e-8 decided the atom at e2e/E0 = 1.

### Fix (commit `<commit-14>`)

- **`e0.py`:** decode context is now `ISL + j + 2`. `METHOD` is `ais-chunked-estimator-v2`, so the
  stored table moves to a new file (`CR/runs/cache/e0/01277503f73c1abe-ais-chunked-estimator-v2.json`).
  The 63,065-entry v1 table can no longer be served. The docstring records the replay convention
  and its source lines.
- **`goodput.py`:**
  - `a2_good` fails only if `e2e > scale·S·E0·(1 + E2E_REL_TOL)`, with `E2E_REL_TOL = 1e-6`. The
    ITL clause stays exact, so the native ITL-only cross-check (A2.4) is unchanged.
  - New record fields:
    - `slowdown_atom_frac`: the share of completed requests with |e2e − E0| ≤ 1e-6·E0;
    - `slowdown_p95`.
- **`__init__.py`:** `HARNESS_VERSION = "lrh-4"`. The docstring now says to bump it when the E0
  method changes too, because the result key carries no E0 identity.
- **Tests:**
  - `test_uncontended_request_is_good_at_unit_slowdown_scale`: noise from −2.4e-9 to +9e-7 is good
    at S = 1 and at S = 2 × scale 0.5; +2e-6 and +1e-3 are bad.
  - `test_slowdown_atom_counts_requests_at_their_uncontended_latency`.
  - The E0 unit test now expects ISL + j + 2.
  - Replay test `test_idle_requests_are_good_at_unit_slowdown`:
    - six requests, each alone on an idle worker with fresh hash ids, through the Evaluator and
      `CR/slots`, under round-robin and default;
    - the (ISL, OSL) pairs cover OSL 1, OSL 2, chunked prefill at 8,192/8,193, and unaligned
      blocks;
    - it asserts e2e = E0 to 1e-7 per row, good 6/6, `slowdown_atom_frac` 1.0, and that rescore
      scale 0.75 scores 0/6.
- **Validation:** `pytest tests learned_routing/workloads/tests`: 142 passed. `pre-commit` is clean
  on the six files, and the DCO hook passed.

### Evidence

All numbers below trace to files under `runs/build-fix-r2/out/`.

**1. E0 against the auditor's 17,233 isolated single-request replays**
(`out/e0_v2_vs_isolated.json`). Relative error is (E0 − e2e_isolated)/e2e_isolated.

| E0 | min | median | max | \|err\| > 1e-7 | isolated requests good at S = 1, exact | good at S = 1, tol 1e-6 |
|---|---|---|---|---|---|---|
| v1 (ISL + j + 1) | −1.64e-5 | −1.60e-6 | +6.3e-9 | 16,479 | 24 | 3,487 |
| v2 (ISL + j + 2) | −1.97e-8 | −1.5e-9 | +1.65e-8 | **0** | 3,741 | **17,233** |

- The v1 row reproduces the auditor's range.
- A tolerance alone (v1 with 1e-6) still fails 13,746 of 17,233.
- v2 alone with an exact compare is a coin flip (3,741 of 17,233).
- The shared v2 table that the smoke filled (17,233 entries) equals the isolated replays to within
  the same ±2e-8.

**2. The new test's workload, re-scored** (`out/idle_regression.json`, `idle_check/`). Six idle
requests with no reuse, under RR and under default:

| Scoring | Good |
|---|---|
| lrh-3 semantics (v1, exact) | **1/6** |
| v1 with tolerance | 3/6 |
| v2, exact | 5/6. The OSL-1 request is +1.1e-8 over. |
| lrh-4 | **6/6** |

So the test fails on the old code.

**3. The auditor's six gn2 cells re-run under lrh-4.**
- **Setup:** `fx2-` copies of the six cells, 5 policies × k = 3, 4. That is 60 replays, 0 errors
  (`smoke/results.jsonl`, `logs/smoke_lr_eval.log`).
- **Rows unchanged:** per-request canonical SHA equal to the auditor's lrh-3 records on 60/60. The
  fix is scoring only.
- **Independent rescoring** (`scripts/smoke_analysis.py`, `out/smoke_analysis.json`). My own
  ITL + E2E scorer uses the auditor's isolated-replay E0 × (1 + 1e-6). Identity warm-up is
  re-derived from the replicate trace. It equals the harness's `window_good` and all 6 rescore
  scales on **60/60** records.
- **What changed:** only the 10 AgentX records (S = 1.0), and only at scale 1.0. Default k3 went
  from 226 to 274 of 278 window completions, and RR k3 from 146 to 275 of 304. The 50 non-AgentX
  records are unchanged at every scale, including the osl0.5 cell at S = 2 × scale 0.5 = 1, where
  the atom is 0.
- **AgentX `goodput_rps_window`** equals the auditor's `fixed_tol1e-6` value **bit-exactly on
  10/10** (`out/knife_edge_match.json`). Ratios against default:

  | Policy | k3, lrh-3 | k3, lrh-4 | k4, lrh-3 | k4, lrh-4 |
  |---|---|---|---|---|
  | round_robin | 0.5672 | **0.8812** | 0.5615 | **0.9420** |
  | lmetric@defaults | 0.9860 | 0.9576 | 1.0012 | 0.9815 |
  | sticky-hard | 1.0048 | 0.9967 | 1.0103 | 0.9874 |
  | learned-choice θ0 | 0.9959 | 0.9967 | 1.0078 | 1.0036 |

  These are the audit's "any tolerant compare" column. The sticky-hard sign flip at k3 and the
  lmetric one at k4 are gone.
- **Atom fractions** (`slowdown_atom_frac`, over all completed rows):

  | Cells | Policy | Atom |
  |---|---|---|
  | AgentX lanes | default | 0.112 / 0.118 |
  | AgentX lanes | lmetric | 0.105 / 0.107 |
  | AgentX lanes | sticky-hard | 0.107 / 0.110 |
  | AgentX lanes | learned-choice θ0 | 0.114 / 0.121 |
  | AgentX lanes | round_robin | 0.386 / 0.418 |
  | Mooncake | all | ≤ 0.0003 |
  | sessions | all | 0.0019–0.0043 |

**4. lr-train under lrh-4** (`train_tiny/`, `out/train_recompute.json`).
- **Run:** the auditor's tiny run re-run with the same arguments, seed 11, `clipped_log_ratio`, and
  3 train cells (`fx2-` copies), with `--keep-per-request`. Validation lines were not produced,
  because I used the default `--val-every 5`, not the auditor's setting.
- **Same candidates:** the gen-0 candidates are identical (same z).
- **Recompute:** my recompute with isolated E0 × (1 + 1e-6) equals all 8 harness objectives
  exactly (max |Δ| 0).
- **Pairing:** the reference is default@defaults with seed k + 1, and candidate and reference share
  `cell_sha`, `build_id`, `trace_sha256`, protocol and lrh-4 on 48/48 pairs.
- **Gen-0 objectives:**

  | Harness | Gen-0 objectives |
  |---|---|
  | lrh-3 | −0.0147, −0.0663, −0.1254, −0.0085 |
  | lrh-4 | −0.0128, +0.0196, −0.0151, +0.0582 |

  The largest change is 0.110. Ranks 2 and 3 swap, as the audit predicted.

### Facts

- **`facts/build_harness.json`:** new key `corrections_fix_r2`. The B3 claim "E2E to < 2e-6
  relative" came from 5 pairs. Over 17,233 pairs, v1 was low by up to 1.64e-5. The `e0` interface
  line is superseded by v2.
- **`facts/build_fix_r2.json`:** this round's record.
- **Earlier facts checked:** none of them used S × scale = 1 on AgentX.
  - The r1 fixer's fx1 AgentX cell used S = 0.8. Over SCALES the product is 0.4–2.4, never 1.
  - `facts/agentx_lowered.json` and `facts/integration.json` used S = 3.

  No recorded ratio needs correcting.

## Notes for calibration

- **Recording.** Record `e0_method: ais-chunked-estimator-v2` and `E2E_REL_TOL = 1e-6` in
  `facts/calibration.json`. Also record `slowdown_atom_frac` per family (A2 band rule), as the
  audit asked.
- **What S = 1 means.** S = 1 (and S = 2 at scale 0.5, S = 4/3 at scale 0.75) is now well-defined:
  "no slower than idle without reuse". Even so, at S × scale = 1 the E2E clause passes only
  requests that were uncontended or had cache hits. On AgentX that set is policy-dependent by
  construction: it is 39–42% of RR's rows. Calibration should know this before it picks S at the
  e2e/E0 atom.
- **Cache.** lrh-3 cache entries are unreachable, and none of them were calibrated.

## Lessons (LESSONS.md)

- **Applied:**
  - LR-08: the rescore sweep at S × scale = 1 is now well-defined, and the atom is reported;
  - LR-03: CRN pairing re-verified on 48 lr-train pairs;
  - LR-01: windows unchanged and re-derived;
  - LR-11: content-keyed pairing checked.
- **Rejected:** the TTFT part of LR-08 (A2.2).
- **Out of scope for this finding:** the rest.

## Still open (minors of goodput-normalization r2, not assigned to this round)

| Finding | Issue |
|---|---|
| F2 | Zero-output requests are never scored |
| F3 | No warm-up guard for open loop or lanes |
| F4 | Cache-key gaps. `seeded` is missing from `policy_sha`. There is no E0 identity in the result key: this round relies on the manual `HARNESS_VERSION` bump. Code and replicate bytes are not keyed. |
| F5 | Completion basis hides rejections |
| F6 | `eps` is absolute |
| F7 | The noise rule has no caller |

## Process

- **Replays,** all through `lr-eval`, `lr-train` or the Evaluator holding `CR/slots`:
  - 60 smoke;
  - 57 fresh in the tiny lr-train;
  - 2 idle-regression;
  - the 5 replay-marked pytest tests, including the new one (2 full-suite runs).
- **Not replays:** E0 v1/v2 tables for the 17,233 pairs are pure AIS-estimator calls, in a private
  scratch dir (`runs/build-fix-r2/e0_scratch`).
- **Concurrency:** another stage committed `<commit-13>` to WT while this round ran. Only this
  round's six files were committed, by explicit pathspec.
- **Housekeeping:** nothing deleted, nothing pushed. Scratch is listed in `CLEANUP.md`.
