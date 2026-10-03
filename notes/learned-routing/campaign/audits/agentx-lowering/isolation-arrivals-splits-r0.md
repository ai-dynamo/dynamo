# Audit: AgentX lowering sidecar (A3), lens "isolation-arrivals-splits", round 0

- Auditor: independent adversarial auditor, round 0, 2026-10-02 (dynamic: own scanners and own replays)
- Audited: WT `<commit-11>` (`learned_routing.workloads.agentx_lowered`, `tools/agentx_lower`), base data
  `CR/traces/agentx_lowered/{weka_cap300,base_cap300}` (current MANIFEST sha `9868b9f6…`), the 30 builder traces in
  `CR/traces/agentx_lowered/gen/`, and `facts/agentx_lowered.json`.
- Method: I did not import the module under test for verification. The only use of the module was its CLI, as a black box, to
  generate fresh traces. All checks are my own code under
  `CR/runs/audit_agentx_lowering/isolation_r0/scripts/`:
  - `scan_isolation.py` identifies each copy's source play from its native namespace and compares every row with the base row.
  - `arrivals_check.py` re-derives the Poisson arrivals from the documented formula with its own blake2b draw.
  - `replay_iso.py` calls `dynamo.replay` directly and holds one `CR/slots` slot per run.
  - `steady_closed.py` and `open_release_check.py` analyze replay output.
- Replays: 19 replays, run sequentially, never more than one slot held.
  - 14 isolation runs at N=1 and N=2.
  - 2 native-Weka anchor runs.
  - 1 open-mode N=8 run.
  - 2 closed-lane N=32 runs (151 s and 181 s wall).
- Not modified: the module, cells, and every existing file. No deletions.

## Verdict: PASS (0 blocker, 0 major, 5 minor)

Recycled copies are isolated, split pools are pure, open-mode arrivals are a correctly seeded Poisson process at the
requested total rate, non-root timing is preserved exactly, and closed lanes recycle into a stationary window at N = 32.
The minor findings concern calibration and integration hygiene, not the lowering's correctness.

## Verified OK (own evidence)

| Check | Result | Evidence |
|---|---|---|
| **Pools are B2's.** Pools from `SPLIT_MANIFEST.agentx_play_subsets` equal the union of `transform.plays` over the AgentX cells per split, and are pairwise disjoint. | train 33 / val 21 / test 28, equal: yes, disjoint: yes | `scan_gen_all.json` `base` |
| **Base files are correctly labeled.** Checked against the source: play id = file name = `weka_cap300` id = raw `plays/` id, and the multiset of input lengths equals the capped and raw plays. | 82/82 | `base_label_check.json` |
| **Static isolation scan**, 43 files. Covers all 30 builder traces plus 13 fresh CLI traces: val/test/train in open and closed mode at seeds 1-5, 11-13, 456 and 717, 2-copy files, and N=32 traces. | 638,382 rows, 17,308 copies, 59,527 sessions; 0 errors on every counter | `scan_gen_all.json`, `scan_fresh.json`, `scan_steady_files.json` |
| ↳ **IDs.** request_id unique per file; every play/session/request/dependency id carries its copy's `lrx:NNNNNN:` prefix; 1 play_id per copy; no session shared across copies; no dangling or cross-copy dependencies; ordinals 0..C-1 contiguous, 1 per copy. | 0 violations | same |
| ↳ **hash_id isolation.** Each copy's hash ids form a contiguous range, and the ranges are pairwise disjoint (sorted-interval check, exact). The base↔copy hash map is a bijection per copy (intra-play sharing kept), and per-copy distinct count = base distinct count. | 0 overlaps, 0 non-contiguous, 0 bijection or count mismatches | same |
| ↳ **Content.** Every non-relabeled field equals the base row: lengths, model, `recorded_api_time_ms`, dependency relation/trigger/`delay_ms`. | 0 mismatches | same |
| ↳ **Split purity.** Every copy's source play, identified from content rather than meta, belongs to the file's split. Each split's outputs use all of its own pool and nothing else (train 33/33, val 21/21, test 28/28 plays seen). Meta copies, plays, ordinals, hash ranges and starts all agree with the content. | 0 foreign copies, 0 meta mismatches | same |
| **Cross-copy KV reuse, measured in replay.** Two copies of the SAME play (0098 at seed 456 with 20 sessions and subagents; 0128 at seed 717 with 94% intra-play block reuse), drawn by the CLI, in open mode with copy 1 isolated in time (a₁ = 326,720 s and 8,525 s). | Copy 1's `reused_input_tokens` equals the single play request-for-request, 39/39 and 24/24, at N=1 and at N=2 (seeded default); root reuse 0 | `iso_reuse_analysis.txt` |
| ↳ **Positive control** (power): my own file with copy 1 given copy 0's hash ids. | +220,768 and +44,496 extra reused tokens; root reuse 240 and 1,648. The measure detects sharing, and replay interns hash ids globally across plays, so relabeling is necessary. | same |
| ↳ **Concurrent copies** (closed file, both copies at t=0). | Per-copy reuse ≤ single (0098 at N=1 −48,384 from contention; otherwise identical); root reuse 0 | same |
| ↳ **Native anchor.** Native Weka replay vs base file replay at N=1. | Every per_request field identical (0098, 0128); recycled copy 1 matches native reuse 39/39 and 24/24, with ttft/e2e within 1.6e-6 ms | `native_anchor.json` |
| **Open-mode arrivals: formula, seed, rate.** Own re-derivation `a_i = a_{i-1} − ln(1−u_i)/rate`, rounded to ns. | Exact match with the root `not_before_ms` in 5 files (3 fresh seeds, 2 builder files, up to 1,024 copies). Same spec regenerated: identical sha256 (`93ed45…` twice). Builder files regenerate with identical rows; only the header digest differs (m5). | `arrivals_check.json` |
| ↳ **Distribution.** | 200 seeds × 1,000 copies: per-seed KS p-values uniform (p = 0.68, 4.5% below 0.05); mean gap 1.0037 (sd of means 0.0316, Poisson expects 0.0316); variance 1.002; lag-1 autocorrelation −0.001; window-count dispersion 0.998. Fresh 999,000 gaps: KS p = 0.49, mean z = 0.32. Arrival draws are independent of play draws (Spearman max \|ρ\| 0.074 over 50 seeds). | `arrivals_check.json`, `arrivals_pooled_extra.json` |
| ↳ **Rate is the TOTAL.** All 30 builder validation runs use total rate = per-worker rate × N (open) and lanes = per-worker lanes × N with copies ≥ 8 × lanes (closed); split and mode match the meta. | 30/30 | inline check |
| **Non-root timing unchanged.** For every row in open files, `nb − a_i` equals `base nb − base root nb`; closed files keep `nb` byte-equal. Every base play has exactly 1 root with nb = 0 and no row below it. | max \|err\| 1.9e-9 ms, 0 rows clamped | scan files |
| ↳ **In replay, open mode** (N=8, val, seed 12, 140 copies). | All 140 roots arrive exactly at a_i. All 5,826 descendants match `max(a_i + offset, max over deps of trigger time + delay_ms)` exactly (max err 7e-12 ms); the recorded floor binds for 251 of them. All completed. | `open_release_n8.json` |
| ↳ **Code path.** | The generator's open rebase is the replay's own lane activation, `start + max(nb − min root nb, 0)` (`aisimulate-core driver.rs:454-481`) | source |
| **Closed lanes recycle, steady at N=32, L1.** 4 lanes per worker = 128 lanes, 1,024 train copies, seed 11, seeded default. | 37,646/37,646 completed; 8 copies per lane; all 896 lane handoffs have gap exactly 0 ms (next copy starts at the previous copy's last terminal; `release_lane` fires when every node of the play is terminal, `driver.rs:681-686`); exactly 128 plays in flight until the first lane runs out at 9,011 s | `steady_n32_closed.json` |
| ↳ **Stationarity, L1.** Window [3,900, 9,000] s (17 × 300 s bins). Thirds and linear trend: completions/s 2.36/2.33/2.30 (t −0.56); activations/s flat (t 0.09); reuse 0.512/0.505/0.506; mean ITL 28.0/29.1/28.1 ms (t 0.06); new prefill tok/s flat (t 0.39); in-flight requests 53/66/61 (t 1.34); mean e2e 22.3/28.0/25.5 s (t 1.37). | No significant trend | same |
| ↳ **L3.** 6 lanes per worker = 192 lanes, 1,536 copies, seed 13. | 56,718/56,718 completed; 1,344 handoffs at gap 0; 192 plays in flight until 10,755 s. Window [4,500, 10,500] s: completions 2.99/2.86/2.95 (t 0.20), ITL 47.4/46.3/43.9 ms (t −1.23), in-flight requests 110/103/102 (t −1.48). No significant trend (\|t\| ≤ 1.66). | `steady_n32_closed_l192.json` |
| **Multi-file collisions are impossible.** Every file's hash ids start at 1, but replay accepts exactly one `agentic_mooncake` file (`validate_trace_files`, `trace.rs:283-295`). | n/a | source |

## Findings

### m1 (minor): every builder open-mode number comes from one arrival realization whose in-window rate is up to 13% off nominal

- **One realization.** All 30 builder traces use seed 0. The arrival draws do not depend on split or rate, so every open
  run shares one normalized arrival sequence. That sequence is an atypical draw:
  - the 512-copy prefix rejects Exp(1) at 5% (KS 0.0673 > 0.0601), and so does the 640-copy prefix (0.0628 > 0.0537);
  - mean normalized gap is 1.09.
- **Realized vs nominal.** Measured in each builder run's own scoring window, the realized play rate is 0.870–1.056 ×
  nominal:
  - N16 at "0.002": 0.892 × (0.00178 per worker);
  - N32 at "0.002": 0.937 ×;
  - N8 at "0.002": 1.002 ×;
  - N8 at "0.003": 0.870 ×.
- **What this confounds:**
  - the per-N comparison reported under LR-12. N16 at 0.962 vs N8 at 0.950 at nominal 0.002 matches N16's 11% lower
    realized load;
  - N16 vs N32 at 0.0025 (0.854 vs 0.733; realized 0.00228 vs 0.00234 per worker);
  - the exact location of the "cliff".
- **The generator is not at fault.** Its Poisson process is correct (see the table).
- **Fix (calibration):** set open AgentX loads from ≥ 3 seeds (the suggested replicate k = seed), and record the realized
  in-window play rate for every run. Do not compare N, or place the open-L3 cutoff, from seed 0 alone.
- **Lessons:** LR-02 and LR-12.

### m2 (minor): open-mode arrival skeletons are shared across splits and N at the same seed

- `unit_draw(seed, "agentx-lowered-arrival", i)` has no split in its label, while play draws do (`agentx-lowered-draw|split`).
- With replicate k = seed k in every split, the test cells' normalized arrival processes are identical to the train
  cells'. Plays stay disjoint, so the play-pool held-out axis is intact.
- Cells at different N with the same seed also share the arrival sequence, and share play draws whenever the split matches.
- **Impact.** On the arrival side, test replicates are not independent draws from the train ones. The effect on a
  ~10-parameter policy is likely small, but that is a hypothesis.
- **Fix:** give each split disjoint seed ranges (e.g. test seeds 1000 + k), or add the split to the arrival label. Under
  LR-11/A4, count seeds, not copies, as the unit of independence.

### m3 (minor): integration hazards, since files do not guard their own mode

1. **Closed file without lanes.** Closed-mode rows keep t = 0-based `not_before_ms`. Replayed without `agentic_lanes`,
   every copy starts at t = 0, which is exactly the A3 failure.
2. **Open file with lanes.** Replayed with `agentic_lanes`, an open file's Poisson starts are discarded, because the lane
   rebase restarts each play at its lane activation.
3. **Rate units.** The CLI's `--play-rate-per-s` is the TOTAL rate. A cell that passes the per-worker rate would run at 1/N
   load.

Today `cells.py:227-232` raises for `agentic_lanes` with non-Weka formats, so nothing runs silently wrong yet. When
calibration extends cells to `agentic_mooncake`, it should assert:
- `meta.spec.mode == closed` ⇔ `load.mode == agentic_lanes`;
- open ⇔ no lanes;
- `play_rate_per_s == load.value × num_workers`.

The builder's 30 runs did all of this correctly (30/30).

### m4 (minor, cost): closed lanes spend about 43% of replay rows in the drain

- At N=32 L1, 43.4% of rows arrive after the window ends (first lane out at 9,011 s; makespan 24,550 s), and only 31.6%
  arrive inside the window.
- Replay wall time is roughly linear in rows.
- **Hypothesis to test in calibration:** `run_trace_replay(max_sim_time_ms=…)` set just past the window end could save
  about 40% of AgentX closed-cell cost without changing window-scored requests. This needs a check that the truncation
  leaves in-window records byte-identical.
- **Lesson:** LR-01.

### m5 (nit): trace bytes depend on the base MANIFEST file, not on the rows

- The header `source.digest` hashes `MANIFEST.json`, and that file embeds the helper binary sha.
- When the helper was rebuilt after `cargo fmt` (manifest `0a0323046f` → `9868b9f6`), the builder's 30 validation traces
  could no longer be regenerated byte-identically. Their bodies are identical: I checked the body sha for 2 files.
- Any future helper rebuild changes every trace sha, and so every cache key, without changing a row.
- **Fix:** digest the set of per-play `base_sha256` values instead.

## Scratch (auditor-created, not deleted)

`CR/runs/audit_agentx_lowering/isolation_r0/`, 1.4 GiB in total:
- `gen/`: 1.3 GiB of fresh traces, regenerable with the CLI;
- `replays_*`: 15 MiB of per_request gz;
- `*.log`: 40 MiB of replay INFO logs.

## LESSONS

- Applied:
  - LR-01: drain cost (m4);
  - LR-02: seeds as the noise unit (m1);
  - LR-09: causal release verified exactly;
  - LR-11: independence unit (m2);
  - LR-12: per-N calibration (m1).
- Others are out of this lens.
