# Build checkpoint fixer, round 1

- Fixer: build fixer r1, 2026-10-02 (about 16:44–17:05 PDT).
- Input: one major finding, `goodput-normalization-r1.md` F1. The other two lenses passed in round 1.
- Worktree: WT `rupei/learned-routing`, head `<commit-12>` (one signed commit on the A3 sidecar's
  `<commit-11>`). Bindings were not rebuilt; `build_id` is still `6955b0ee…`.
- Evidence: `CR/runs/build-fix-r1/` (`smoke/`, `a3_smoke/`, `candidates_check/`, `cells/`).
- **Verdict on F1: VALID, FIXED at the root.** Nothing was rebutted.

## F1 (major): completion-basis windows scored the end-of-run drain. VALID, FIXED

### Why it is valid

I reproduced the defect and its cause.

- **Root cause.** With completion basis, `goodput.measurement_window` ended at the last arrival of
  any row unless `window_ms` was set.
- **Why that window is mostly drain on lanes cells.** aisimulate-core deals plays to lanes once:
  `driver.rs:1352`, `lane_index = play_index % lane_count`. `release_lane` (`driver.rs:718`)
  activates a lane's next play at the release instant and does nothing once the lane's list is
  empty.
- **What the old window contained.** On the auditor's own A2 lanes cell (same policies, same
  replicates, `runs/build-fix-r1/smoke/analysis.json`):
  - 62.6–67.0% of the old window had fewer than 4 occupied lanes, recomputed independently from
    per-play spans;
  - 485 completions were scored, against 241–345 inside full occupancy.
- **The note conflict.** `facts/build_fix_r0.json` `next_stage_notes[1]` did say calibration sets
  only the open-loop windows.
- **Dilution: the auditor's range is slightly overstated.** I measured how much the old window
  shrank deviations from 1:

  | Policy, replicate | Shrinkage |
  |---|---|
  | round_robin k0 | 10.6% |
  | round_robin k1 | 22.8% |
  | lmetric k1 | 40.6% |
  | lmetric k0 | sign flip: 1.0003 under the old window, 0.9858 under full occupancy |

  The audit's "23–41%" range leaves out round_robin k0. The finding stands either way.

### The fix

Commit `<commit-12>`, `fix(learned-routing): end completion windows at full occupancy`, changes
`goodput.py`, `evaluate.py`, `worker.py`, `__init__.py`, `workloads/cells.py` and the tests.

**1. New required key `measure.end` on completion basis.**
- `full_occupancy` ends the window at the last instant at which `load.value` slots were occupied.
  - A slot holder is an agentic play (`play_id`), or else a closed-loop session (`session_id`).
  - Each holder spans its first arrival to its last terminal.
  - This is the first departure after the last admission for a work-conserving closed loop, and
    the first lane's exhaustion for lanes. It applies unchanged to A3's lowered `agentic_mooncake`
    lanes files, whose per-copy play IDs are unique.
- `fixed` ends the window `window_ms` after its start. It requires `window_ms > 0`, which also
  closes the "`window_ms: 0` read as unset" hole for this basis.
- A completion cell with no `end`, `end: last_dispatch`, or `full_occupancy` combined with
  `window_ms` raises `ValueError`. `lr-eval` checks the rule at plan time, so a bad cell becomes a
  `plan_error` record without running a replay.

**2. Guards that keep the slot model honest.**
- Occupancy above `load.value` is an error. Keying lanes by session instead of by play would
  trip it.
- Occupancy that never reaches the cap is an error, for example lanes ≥ plays.
- An empty window (warm-up past the end of full occupancy) is an error. Completion basis no
  longer falls back silently to the makespan.

**3. Half-open completion windows `[start, end)`.**
- The full-occupancy end is itself a completion: the one that empties a slot for good. Counting it
  added exactly one completion to every run. On A2, `sticky-hard` k1 moved from 1.0074 to 1.0111.
- With the half-open window, the harness reproduces the auditor's independent full-occupancy
  ratios exactly: 6/6 at 4 decimals (`smoke/boundary_variant.json`).
- The ISL buckets' `good_frac_window` now uses the same completion set. Before, it counted every
  non-warm-up row on completion basis, drain included.

**4. Diagnostics.**
- Every completion record carries `occupancy_cap`, `occupancy_peak`, `occupancy_units`,
  `full_occupancy_end_ms` and `window_below_cap_ms`/`_frac`. The last pair shows any drain a
  `fixed` window would include.
- Compact per-request rows keep `play_id` and `lane_id`.
- `HARNESS_VERSION` is now `lrh-3`. `lrh-2` existed only in this fixer's intermediate smoke, which
  used an inclusive end. Every `lrh-1` and `lrh-2` cache entry is unreachable (CLEANUP).

**5. Cells.**
- The generator emits `end: full_occupancy` on every completion-basis cell: 27 closed-loop cells
  and 22 lanes cells (`workloads/cells.py` `measure_rule`).
- Version is now `lr-cells-v3`. `SPLIT_MANIFEST` is `c42dacea…`; the candidate files are:

  | File | SHA-256 |
  |---|---|
  | train | `c7eb4cb7…` |
  | val | `e0eaf74c…` |
  | test | `4ba73c06…` |

- Field-by-field diff against the preserved v2 copies (`cells/superseded/build-r1/`,
  `runs/build-fix-r1/cells/diff_cells.json`). The only changes are:
  - `measure.end` on 49/49 completion cells;
  - the `measure_trace.note` of the 22 AgentX cells, which is a hint, not cell content.

  Every derived trace, ID, split and load slot is unchanged, and 0 new traces were materialized.
- The A3 sidecar's play pools in `SPLIT_MANIFEST` are unchanged.

**6. The fixer r0 note is corrected.** `facts/build_fix_r0.json` gains
`next_stage_notes_correction_fix_r1`; the original text is kept. `facts/build_fix_r1.json` holds
the current notes.

### Verification

**Tests.** `pytest` passes 132: 114 from round 0, 11 from the sidecar and 7 new. Pre-commit
(isort, black, flake8, ruff, codespell, DCO) passes.

- **Unit tests** (`tests/test_goodput.py`):
  - the end rule is required, and each bad combination raises;
  - lanes end at the first exhaustion, with contiguous play handoff and a concurrent subagent
    session;
  - session keys alone trip the over-cap guard;
  - closed loop ends at the first departure after the last admission, with a session holding its
    slot through think time;
  - the never-full and empty-window errors fire;
  - zero-length units never occupy a slot;
  - the identity warm-up composes with `full_occupancy`;
  - `compact_row` keeps `play_id` and `lane_id`.
- **Real-replay tests** (`tests/test_replay_e2e.py`):
  - closed loop at C = 4: the window end equals the first departure after the last admission, the
    peak equals the cap, and the cell without an end rule becomes a `plan_error`;
  - B2's A2 lanes trace at 4 lanes: the window end equals the minimum over lanes of each lane's
    last terminal, using the driver's own `lane_id`. It is exact.
- **Generator test.** Every planned measure rule passes `validate_measure`, and completion rules
  end at full occupancy.

**Auditor-cell smoke** (`runs/build-fix-r1/smoke/`; the auditor's 6 gn1 cells copied as fx1-, the
same 4 specs, k = 0 and 1; 48 replays, 0 errors).
- **Rows and pre-existing fields.** Per-request rows are identical to the auditor's lrh-1 rows
  except for the two new fields, 48/48. All pre-existing aggregates (`good`, `good_frac`,
  `goodput_rps`, TTFT and ITL) are equal 48/48. Open-loop windows and goodput are unchanged
  24/24.
- **Window end** (24/24 completion records):
  - lanes: equal to the first lane exhaustion from `lane_id`;
  - closed loop: equal to the first session departure after the last session admission.

  Time below the cap inside the window is 0 by the harness and 0 by an independent recomputation.
- **AgentX A2 ratios against default, old → new:**

  | Policy | k0 | k1 |
  |---|---|---|
  | round_robin | 0.8471 → 0.8289 | 0.8483 → 0.8033 |
  | lmetric | 1.0003 → 0.9858 | 0.9412 → 0.9009 |
  | sticky-hard | 0.9983 → 0.9901 | 0.9995 → 1.0074 |

  Every new value equals the auditor's full-occupancy value.
- **Sessions s2-think2.0 (F2).** The old window spent 9.3–9.4% below the cap; the new one spends
  0%. The window shrinks from 3,337–3,365 s to 3,027–3,051 s, and ratios move by at most 0.0015.
- **Mooncake closed.** The scored count is still 2,464 = M − C. The window is longer by
  0.1–1.2 s, and ratios move by at most 0.0008.
- **Reproducibility.** A `--refresh` re-run with the final code reproduced every field 48/48.

**A3 lowered lanes** (`runs/build-fix-r1/a3_smoke/`). The input is the sidecar's closed train file
`18d91a50…` (64 copies, 2,497 rows) at N = 4 with 8 lanes. I ran 2 direct replays through one
slot and scored them with the harness code, using a provisional SLA (I = 40 ms, S = 3) and the
sidecar's provisional warm-up.
- The window end equals the first lane exhaustion, with peak 8 = lanes and 64 units, for both
  default-seed1 and round_robin.
- Ending at the last dispatch instead would add 51–54% below-cap time.
- It would also move round_robin/default from 0.9067 to 0.9755, so A3 does not remove the drain.
- The identity warm-up rule raises on the lowered file ("needs a trace with timestamps"), as
  designed. Lanes cells use `warmup_ms`.

**All 49 completion candidates** (`runs/build-fix-r1/candidates_check/`). Each cell ran once with
default k = 0 at B2's provisional loads and a provisional SLA: 0 errors; peak = cap on 49/49;
below-cap 0 on 49/49.

| Cells | Full-occupancy window as a share of the old window | Scored completions | Window |
|---|---|---|---|
| Mooncake/FAST25 closed | 1.000–1.001 | — | — |
| Synthetic sessions closed | 0.52–0.97 | — | — |
| Native-Weka lanes | 0.05–0.64 | 68–364 | 243–7,323 s |

Native lanes with 7–11 plays over 2–16 lanes are barely measurable. A3 says these cells move to
the lowered traces anyway, now that `facts/agentx_lowered.json` reports `status: ok`.

### What calibration must do now (replaces fixer r0's note)

**Completion-basis cells (closed loop and lanes)**
- These already carry `end: full_occupancy`. Calibration does not set their window end.
- Lanes cells still need a replay-time `measure.warmup_ms`. The sidecar's suggestion is the p90
  over lanes of each lane's first-copy end.
- Closed-loop cells keep `warmup_trace_ms`.
- Choose `load.value` and copies so that the full-occupancy window holds enough completions. The
  sidecar suggests ≥ 8 copies per lane.
- The A3 lowered AgentX cells must carry `{basis: completion, end: full_occupancy, warmup_ms}`.

**Open-loop cells.** Calibration still sets `warmup_ms` and `window_ms`.

**Harness integration still missing for lowered AgentX.** Calibration must add:
- `agentic_lanes` for `agentic_mooncake` in `cells.Cell.load_kwargs`;
- a CRN replicate for `agentic_mooncake`.

**Interpreting completion-basis ratios.** They are good completions per second at full
occupancy, i.e. attainment × throughput. Stratify them by load mode (LR-09).

## Lessons (LESSONS.md)

**Applied:**
- **LR-01.** I made its hypothesis, "a fixed window in which all lanes are active", the harness
  rule, and verified it on native and lowered lanes and on closed loop. Windowed attainment is
  unchanged for open loop.
- **LR-09.** Per-mode meaning: completion basis measures throughput × attainment at full
  occupancy. Lanes are the primary AgentX mode.
- **LR-03.** CRN pairing is unchanged: the same replicate trace and seed k + 1 on every compared
  record.
- **LR-11.** Content-keyed comparisons: I compared old and new records only on the same (cell,
  policy, k, trace).

**Rejected:**
- **LR-01 Action 2**, which keeps makespan goodput for closed-loop and lane cells. Makespan
  includes the drain this finding removes, and A2.3 asks for a steady-state window.

**Out of scope for this finding:** LR-02, LR-04–08, LR-10 and LR-12–15.

## Not addressed here (minor, still open from the audit)

| Finding | Issue |
|---|---|
| F3 | open-loop warm-up guard |
| F4 | E0 off-by-one |
| F5 | cache-key gaps: `seeded`, E0 build identity, scoring code |
| F6 | completion basis hides rejections; still policy-independent today |
| F7 | absolute eps |
| F8 | the noise rule has no content-keyed caller |

F3–F8 were minor and not assigned to this round. F2 (sessions drain) is fixed by the same end rule.
