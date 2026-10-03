# Fix round 0: AgentX lowering sidecar (A3)

- **Fixer:** AgentX lowering sidecar, fix round 0, 2026-10-02.
- **Input:** `audits/agentx-lowering/parity-fidelity-r0.md`, which had 1 major finding (F1) and 2 minor
  ones (m1, m2).
- **Result:** F1 is VALID and FIXED at the root. Nothing is rebutted.
- **Commit:** WT `<commit-13>`, signed off, no co-author trailer. It touches only
  `workloads/agentx_lowered.py` and the new `workloads/tests/test_workloads_agentx_lowered_transforms.py`.
- **Status:** `facts/agentx_lowered.json` stays `"ok"` and now has a `transforms` section.
- **Evidence:** every path below is under `CR/runs/agentx_lowered/fix_r0/`, unless it starts with `traces/`.

## F1 (major): VALID, FIXED

The finding was correct. 5 of the 22 AgentX cells carry `think_mult` or `osl_mult`, and the module could
only build `base_cap300`. Confirmed from `cells/*.candidates.jsonl`:

| Cell | Transform |
|---|---|
| `agentx-A2-think0.5-n4-lanes-L2` | `think_mult` 0.5 |
| `agentx-A3-think2.0-n8-lanes-L1` | `think_mult` 2.0 |
| `agentx-A3-osl1.5-n4-lanes-L2` | `osl_mult` 1.5 |
| `agentx-T1-think4.0-n4-lanes-L2` | `think_mult` 4.0 |
| `agentx-T2-osl2.5-n8-lanes-L2` | `osl_mult` 2.5 |

### What changed

**`PlayTransform`** (think_cap_s, think_mult, osl_mult) names the per-play Weka transform.
- `from_cell(cell["transform"])` maps a cell to its transform. It rejects `window`, `isl_*`,
  `prefix_root_mult` and a `max_model_len` other than 131072.
- `cell_transforms(root)` lists the 6 transforms that B2's cells use.

**Transforms threaded through the pipeline:**
- `capped_play_bytes`, `build_base`, `check_against_cells` and `run_parity` take the transform. The B2
  byte check now selects cells whose transform equals this one.
- `GenSpec` gains `think_mult` and `osl_mult`, and `generate` reads `base_<tag>`.
- `generate` raises on a missing base (`FileNotFoundError` naming the `build-base` command), and on a
  MANIFEST built for another transform.

**Directories.** Each transform has its own `weka_<tag>` and `base_<tag>`: `cap300`, `cap300_think0.5`,
`cap300_think2`, `cap300_think4`, `cap300_osl1.5` and `cap300_osl2.5`.

**Recording the transform.**
- The trace meta gains `transform` (think_cap_s, think_mult, osl_mult, max_model_len, tag), and
  `spec.think_mult` / `spec.osl_mult`.
- The trace digest key (`GenSpec.key_dict`) includes the non-identity multipliers.
- The base MANIFEST records them too.

**Identity is kept stable.** Identity multipliers are omitted from the key and from the MANIFEST, so
untransformed outputs keep their bytes.

**CLI.**
- `--think-mult` and `--osl-mult` on `generate`, `build-base` and `parity`.
- `build-base --all-cell-transforms`.

### Evidence

**1. Graph digests per transform.** `build-base --all-cell-transforms` (`build_base.log`, 41 s):
- For each of the 6 transforms, native `load_weka_agentic_graph` and the base file compile to the same
  digest in 82 of 82 plays.
- B2 byte check: every transformed play that appears in a B2 cell trace with the same transform is
  byte-identical to its line there.

| Transform | B2 plays byte-identical |
|---|---|
| cap300 | 82/82 |
| think0.5 | 11/11 |
| think2 | 11/11 |
| osl1.5 | 11/11 |
| think4 | 7/7 |
| osl2.5 | 7/7 |

**2. Relation to cap300** (`transform_relation.json`, from `scripts/transform_relation.py`). This script
reads files only and does not import the module. Per transform, over 82 plays and 3,132 rows:
- Request, session and play ids, hash ids, input lengths, models and dependency targets, relations and
  triggers are identical to cap300 (0 mismatching rows).
- **think_mult m:** every root offset, `delay_ms` and `recorded_api_time_ms` equals m × cap300's.
  - 0 of 10,594 values mismatch, and the maximum relative error is 0.
  - The play span ratio is exactly 0.5, 2 and 4.
- **osl_mult m:** `output_length` follows `min(max(1, round(out × m)), 131072 − in)` in 3,132 of 3,132
  rows.
  - The one authored zero output stays 0.
  - The context clamp is hit in 2 rows (osl 1.5) and 9 rows (osl 2.5).

**3. Replay parity per transform** (`parity_cap300_<m>/parity.json`). The `parity` command was re-run
for each of the 5 transforms with the full default setup:
- 6 plays;
- the single, corpus and 12-copy recycle tiers;
- N = 2 and 4;
- seeded default and round robin;
- timestamp mode, plus lanes 2/3 (corpus) and lanes 3/5 (recycle).

Results:
- **84/84 `per_request` identical** for each transform (420/420 in total), on 37 fields. Every case is
  non-vacuous: rows equal completed requests on both sides.
- **Copy-order variant:** 0/12 identical, as designed. Its mean e2e range grows with output length, from
  −5.7% to +18.0% at osl 1.5 and from −15.6% to +13.5% at osl 2.5. This is recorded as a caveat.

**4. Identity unchanged.**
- `base_cap300` and `weka_cap300` were rebuilt with the new code: 165/165 files byte-identical, and the
  MANIFEST sha is still `9868b9f6…` (`cap300_shas_{before,after}.txt`).
- Identity `GenSpec`s produce byte-identical traces under `<commit-12>` and `<commit-13>`: 3/3 cases
  (`identity_bytes_check.json`).

**5. Paired validation** (`validate/`, 11 runs). Each run is closed mode with the cell's split and N, at
the provisional lanes for the cell's level, 8 copies per lane and seed 0. Each identity twin draws the
same plays in the same order. Every request and trajectory completed. Good fractions below are default
at I40 S3.

| Transform | Run | Result |
|---|---|---|
| think 2 | N8, 4 lanes/worker | busy 0.70, good 0.993 (light) |
| think 4 | N4, 5 lanes/worker | busy 0.50, good 0.990 (light) |
| think 0.5 | N4, 5 lanes/worker | not stationary (drift 1.49 > 0.41), good 0.575 |
| think 0.5 | N4, 4 lanes/worker | not stationary (drift 1.69 > 0.50), good 0.846 |
| think 0.5 | N4, 3 lanes/worker | stationary, good 0.972 |
| osl 1.5 | N4, 5 lanes/worker | stationary, good 0.825 (identity 0.842) |
| osl 2.5 | N8, 5 lanes/worker | stationary, good 0.862 (identity 0.885) |

So transform cells need their own load calibration. This is recorded in `facts.suggested_loads.transform_cells`.

**6. Tests.**
- The new file has 7 tests:
  - `from_cell` mapping and rejection;
  - identity key stability;
  - every B2 transform has a verified base;
  - think_mult dilates exactly and leaves draws, labels, hash ranges and arrivals unchanged, in open and
    closed mode;
  - osl_mult scaling with the clamp;
  - an unbuilt transform fails loudly.
- Workloads tests: 45 pass. The whole `benchmarks/learned_routing` suite: 142 pass.
- pre-commit (isort, black, flake8, ruff, codespell) is clean.

### The confound in the evidence is resolved

**Neither of the auditor's failure scenarios applies any more.**
- A lowered `agentx-T1-think4.0` now replays think-4.0 traffic.
- All 22 AgentX cells can use the lowered generator, so the family no longer mixes generators.

**Calibration integrates as follows.**
- Call `GenSpec(..., **fields of PlayTransform.from_cell(cell["transform"]))`.
- Calibrate loads per transform cell.

## m1 (minor): VALID, wording corrected

**What changed.**
- `facts.parity_evidence.replay.wording_correction_fix_r0` now states that the single tier ran in
  timestamp mode only, and that the builder never replayed an unmodified generator file. It credits the
  auditor's 64/64, 24/24 and 36/36 closure of that gap.
- The module docstring's Evidence section is corrected the same way.

**Not changed.** The STATE.md line of 16:47 is append-only, so it stays as written; this round's STATE
line points here.

## m2 (minor, outside the lens): not changed, left to calibration

Window membership for open-loop agentic requests is a scoring rule owned by calibration and the harness
(`learned_routing.goodput`), not by the lowering. Every open trace's meta already carries the
policy-independent per-copy arrival `copies[].start_ms`, which is what the auditor suggests using.

## Scratch (listed in the facts)

- `traces/agentx_lowered/{base,weka}_cap300_{think0.5,think2,think4,osl1.5,osl2.5}`: 5 × 80 MiB. Calibration
  needs these.
- 11 validation traces in `traces/agentx_lowered/gen`: 573 MiB.
- `runs/agentx_lowered/fix_r0`: 283 MiB.
