# Audit: AgentX lowering sidecar (A3), lens "parity-fidelity", round 0

- **Auditor:** independent adversarial auditor, round 0, 2026-10-02. Dynamic audit with my own tools and my own replays.
- **Audited:**
  - WT `<commit-11>`: `learned_routing.workloads.agentx_lowered` and `tools/agentx_lower`.
  - `CR/traces/agentx_lowered/{weka_cap300,base_cap300}`, with MANIFEST sha `9868b9f6…`.
  - `facts/agentx_lowered.json` and `runs/agentx_lowered/parity/parity.json`.
- **Method:** I did not import the module under test to verify it. I used its CLI only as a black box, to produce fresh
  generator output. Everything else is my own code, in `CR/runs/audit_agentx_lowering/parity_fidelity_r0/scripts/`:
  - `dumper/` is my own Rust CLI. It links `aisimulate-core =0.13.0-dev.202609300000000061`, and its lock resolves the
    same `serde_json 1.0.150` and `blake3 1.8.7` as the bindings. It serializes the full compiled `AgenticNode` lists
    (ids, lengths, hash ids, `not_before_ms`, dependencies) of:
    - a native Weka source, through `load_weka_agentic_graph`;
    - an agentic file, through `AgenticTrace::from_agentic_mooncake`;
    - the importer's rows, through `WekaImporter::collect_rows`.
  - `semantics.py` recomputes every edge and delay from the raw capped Weka play, following `weka.rs`.
  - `gen_check.py` diffs generator output against the base rows.
  - `parity_run.py`, `recycle_run.py` and `open_shift.py` call `dynamo.replay.run_trace_replay` directly. Each replay
    holds one `CR/slots` slot, and the replays run one at a time.
- **Replays:** 249 calls, all sequential. Replay rejected one of them by design (kv_router at N=1).
- **Not modified:** the module, the cells, and every existing file. Nothing was deleted.

## Verdict: FAIL (0 blocker, 1 major, 2 minor)

The lowering itself passes every parity test I ran:
- The base rows are AISim's own Weka lowering, node for node, for all 82 plays.
- Every edge and delay agrees with `weka.rs` semantics recomputed from the raw plays.
- Unmodified generator output replays exactly like native Weka, per request and in every field, in 156 of 156 cases.

The FAIL comes from one coverage gap that the builder did not disclose (F1). The generator cannot express the AgentX
cells that carry `think_mult` or `osl_mult` transforms. Amendment A3 nonetheless switches every AgentX cell to the
lowered traces once the status is "ok".

## Plays chosen for my replays

None of these plays is in the builder's parity set (0098, 0073, 0390, 0013, 0021, 0315). Together they cover all three
splits.

| Play | Split | Rows | Structure (edges in base rows) |
|---|---|---|---|
| 0007 | test | 25 | 1 subagent lowered to 14 streams: 14 joins, 50 replay barriers, 1 completion spawn |
| 0010 | train | 73 | 1 subagent (35 nested), 2 dispatch spawns, 70 barriers, 1 join |
| 0293 | val | 76 | 4 subagents: 4 joins, 2 dispatch and 2 completion spawns, 14 barriers |
| 0325 | test (extra) | 111 | 3 subagents (85 nested): 3 joins, 2 completion and 1 dispatch spawns |

- **Zero-output request:** only play 0021 has one among the 82, and 0021 is in the builder's set. It still appears
  4 times in my recycled val test (row 8 of the table below).
- **Background subagents:** no play has an `async_launched` subagent, as the builder disclosed.

## Verified OK (own evidence)

All evidence paths are relative to `CR/runs/audit_agentx_lowering/parity_fidelity_r0/`.

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | **Base rows are AISim's lowering.** For every play I compared two things. (a) The base file's header and rows against `WekaImporter::open(capped play).collect_rows()`. (b) The compiled node list of native `load_weka_agentic_graph` against `from_agentic_mooncake(base)`, on every field. | 82/82 rows identical; 82/82 node lists identical (3,132 nodes); my digests reproduce the MANIFEST's `graph_digest_weka` and `graph_digest_agentic` 82/82 | `out/node_diff.json` |
| 2 | **Corpus-wide timestamp basis.** `weka.rs` resolves the nested-timestamp basis for a whole corpus: one relative witness flips every play. Per-play lowering therefore matches corpus lowering only if no play is relative. | 64 `not_applicable`, 18 `absolute`, 0 `relative`. B2's idle cap rejects relative plays anyway. | `out/node_diff.json` |
| 3 | **Same plays as B2's native cells.** First, a byte comparison of each capped play with its line in B2's cell traces (own code). Second, my 4 plays compiled inside B2's multi-play cell traces versus the base rows. | 82/82 byte-identical. Inside the cell corpora, all non-hash fields are identical once the namespace is stripped (`<cell file>#<line>` vs file name), and the hash map is bijective. | `out/b2_cell_check.json` |
| 4 | **Edges and delays against `weka.rs` semantics,** recomputed from the raw plays for all 82 plays. Details are in the list below this table. | 4,330/4,330 edges, 0 errors in 82 plays. Power: 8/8 seeded mutations detected (delay +1 µs; dropped sequence, barrier or join edge; `not_before` +1 ms; spawn trigger flip; hash swap; api +0.5 ms). | `out/semantics.json`, `scripts/semantics.py` |
| 5 | **Generator output against base rows.** 10 fresh CLI files (60 copies each; train, val and test; open and closed; seeds 5–8) plus 3 builder validation files (`7b0087…` closed 384 copies, `9d89a9…` open 512, `f402f8…` test open 512). Each copy, label stripped, is checked against its base play. The checked properties are listed below this table. | 0 errors in all 13 files (up to 19,424 rows and 1.24M distinct hash ids) | `out/gen_check.json`, `out/gen_check_existing.json` |
| 6 | **Single-play replay parity, my plays.** Native capped Weka play compared with three lowered versions: the base file; one copy cut from a fresh **closed** generator file; and one copy cut from a fresh **open** file. Both cut copies keep the generator header and labels; only the ordinal is reset to 0. Run at N 2/4, seeded default (`dynamo-default-cost-fn` seed 1) and round robin, timestamp mode and `agentic_lanes=1`. | 96/96 `per_request` identical in all 37 fields, with only the `lrx:NNNNNN:` prefix mapped back | `out/parity/parity_cases.json` |
| 7 | **Two plays from one fresh file** (test, seed 5: copies 44 = 0325 and 48 = 0007, generator labels unchanged). Compared with a native two-file corpus and with the base files concatenated. Lanes none, 1 and 2; N 2/4; both policies. | 24/24 identical | `out/parity/parity_cases.json` |
| 8 | **Untouched generator prefixes against native recycling.** The first 12 copies of three fresh closed files, byte-for-byte as generated (labels, ordinals 0–11, hash ranges, header). The native reference is a corpus with one Weka file per copy. File names are chosen (blake3 search in my dumper) so that both native namespace order and traversal order equal copy order. Prefixes: val 605 rows, 7 distinct plays, with 0021 ×4 and 0293 ×2; train 538 rows, 11 plays, including 0010; test 406 rows, 11 plays. Timestamp mode and lanes 3/5; N 2/4; both policies. | 36/36 identical | `out/recycle_*/cases.json` |
| 9 | **Open-mode shift semantics.** At N=1 with round robin: copy 0 at a=0, then the target copy at a_i, after copy 0 has finished. The target is compared with the native play shifted by a_i. | **0007** (a = 2,693,282 ms) and **0325** (a = 2,522,492 ms): every field equal; shifted times within 3.3e-9 and 7.2e-8 ms; durations within 3e-11 relative. **0010 and 0293** overlapped copy 0, so they are inconclusive by construction and not counted. | `out/open_shift/open_shift.json` |
| 10 | **Builder's parity claims.** Re-read from `parity.json`, plus my own checks of the per-play and pool claims. Details are in the list below this table. | All claims hold, with the single exception recorded as m1 | `parity.json`, `cells/SPLIT_MANIFEST.json` |
| 11 | **Base data integrity.** | Base file sha = MANIFEST = the builder's pre-fmt record, 82/82. The MANIFEST sha changed from `0a0323…` to `9868b9…`, which alters only the header digest of earlier generated files. | inline |
| 12 | **Module tests.** | 11/11 pass | inline |

**Row 4: what `semantics.py` checks.** The session partition (chain detection) is taken from the rows; everything else
is recomputed.
- **Sequence edges:** consecutive in (t, source_order, id) order, with delay = t − (t_prev + api).
- **Subagent spawns:** the child root is spawned from the last parent-stream request before the marker. The trigger is
  dispatch exactly when t_child < parent end, with the delay measured from the parent's start; otherwise the trigger is
  completion, with the delay measured from the parent's end. Worker spawns and root-anchor spawns are checked against
  the same formula.
- **Joins:** the target is the first owner-stream request after the marker with t + 1e-6 ≥ child end. Child end comes
  from `duration_ms`, or else from the nested ends. The join has one edge from each child stream's last request.
- **Replay barriers:** the pruned cross-stream frontier per scope, with delay 0.
- **Per-row fields:** `not_before = round_ns(t − root_time)`, `recorded_api_time_ms`, lengths and model. Each play has
  exactly one root.
- **Hash identity:** two blocks share a hash exactly when they have the same source hash within the same play. Missing
  hashes and partial tails are private to their request.

**Row 5: what `gen_check.py` checks for every copy.**
- Ids, deps (relation, trigger, delay), lengths, `recorded_api_time_ms` and model all equal the base play's.
- The hash map is bijective within a copy. Hash ids are disjoint across copies.
- Label = ordinal = copy index, and the labels are contiguous.
- Closed mode: `not_before_ms` is unchanged. Open mode: `not_before_ms` = a_i + base nb, with a_0 = 0 and the a_i
  nondecreasing.
- Each copy has exactly one root, comes from the file's own split pool, and agrees with the meta. The file compiles.

**Row 10: the builder's claims, re-checked.**
- 84/84 exact cases are identical and non-vacuous: on both sides, rows = completed = candidate rows in every case.
  37 fields are compared.
- The copy-order variant is identical in 0/12 cases. Its mean-e2e range of −7.44% to +4.71% matches the facts.
- Per-play claims:
  - 0098 has 1 root-level session and 19 subagent sessions;
  - 0073 has 6 subagent sessions;
  - 0021 is the only play with a zero-output request;
  - no play has an `async_launched` subagent.
- The pools are 33/21/28 plays, equal SPLIT_MANIFEST name for name, and are disjoint.

**Conclusions on parity:**
- The lowered rows preserve every Weka dependency edge, trigger, delay, tool gap and length exactly.
- Recycling only relabels ids and remaps hash ids bijectively per copy into disjoint ranges.
- Open mode is an exact time shift of native timestamp semantics.
- The only free choice is cross-copy node order, which breaks ties. When it matches the native order, unmodified
  generator output is per-request identical to a native corpus that holds one file per copy.

## Findings

### F1 (major): the lowered generator cannot express B2's AgentX transform cells, and the gap is not disclosed

**The cells.** 5 of the 22 AgentX candidate cells carry an `osl_mult` or `think_mult` transform:

| Split | Cell | Transform | N | Holdout axis |
|---|---|---|---|---|
| train | `agentx-A2-think0.5-n4-lanes-L2` | think_mult 0.5 | 4 | none |
| train | `agentx-A3-think2.0-n8-lanes-L1` | think_mult 2.0 | 8 | none |
| train | `agentx-A3-osl1.5-n4-lanes-L2` | osl_mult 1.5 | 4 | none |
| test | `agentx-T1-think4.0-n4-lanes-L2` | think_mult 4.0 | 4 | `transform_extrapolation` |
| test | `agentx-T2-osl2.5-n8-lanes-L2` | osl_mult 2.5 | 8 | `transform_extrapolation` |

**What the module supports.** The module lowers only the base transform:
- `capped_play_bytes` builds `TransformSpec(think_cap_s=…, max_model_len=…)`.
- `build-base` and `GenSpec` take only `think_cap_s`.
- The CLI has no `--think-mult` or `--osl-mult` option.

The builder knew about these cells: `check_against_cells` explicitly skips cells whose `osl_mult` or `think_mult` is not
1.0. Yet neither the facts nor the returned issues mention that these cells cannot be lowered.

**Failure scenario.** A3 says that once `facts/agentx_lowered.json` reports `"status": "ok"`, AgentX cells for EVERY N
use the lowered traces. Calibration then has two options, and both are bad:
- **Generate the lowered trace from the cell's split and load.** `agentx-T1-think4.0` would then replay think 1.0
  traffic under a think-4.0 label. The test-set `transform_extrapolation` hold-out silently becomes an untransformed
  cell.
- **Keep native Weka for these 5 cells.** They are N ≤ 8, so this is possible. But one family would then mix two
  workload constructions: recycled steady-state lowered cells, and native lanes that run once and are dominated by
  the drain. The transform hold-out would be confounded with the change of generator.

Replay knobs cannot stand in for the transforms:
- `arrival_speedup_ratio` divides every `not_before_ms` and `delay_ms`. In open mode it would also compress the
  Poisson play arrivals.
- Nothing can emulate `osl_mult`.

**Fix.**
- Thread the cell's Weka transform (`think_mult`, `osl_mult`, alongside `think_cap_s`) through `capped_play_bytes`,
  `build-base` and `GenSpec`. Key each base directory by the full transform, e.g. `base_cap300_think4_osl1`.
- Re-run the 82-play graph-digest check for each transform.
- Add the transform to the generated trace's spec and meta.
- Or, at minimum, record in the facts that transformed AgentX cells stay native Weka at N ≤ 8 (with a DEVIATIONS
  entry). That second path still leaves the confound described above.

### m1 (minor): the parity summary overstates what was covered, and the builder never replayed unmodified generator output

**Wording.** The summary says each tier "was run at N=2 and 4 … in timestamp and lanes mode". In fact the single tier
(48 of the 84 cases) ran in timestamp mode only (`agentic_lanes: null` in all 48). `facts.parity_evidence.replay.tiers`
states this correctly.

**Coverage.** None of the 84 exact cases replays an unmodified generator file:
- Every relabeled file was written with the base play's header (native source digest).
- The recycled corpus was relabeled with native-ordered label ranks.
- The only case that used the generator's own labels, the copy-order variant, differs by design.

**This audit closes the gap.** Unmodified generator output (header, labels, ordinals, hash ranges) is identical to
native:
- 64/64 single copies cut from fresh open and closed files, including `agentic_lanes=1`;
- 24/24 two-play cases with the generator's own labels;
- 36/36 untouched 12-copy prefixes against a native corpus in copy order.

**Fix.**
- Correct the summary wording.
- Optionally add a `parity` tier: the first K copies of a real generated file against a native corpus whose file names
  sort, and hash to namespaces, in copy order (see `scripts/recycle_run.py` and the `ns-search` command of my dumper).

### m2 (minor, outside this lens): for agentic open loop, "arrival basis" window membership depends on the policy

**Observation.** `band_table` (`agentx_lowered.py:1337`) and the harness's arrival basis (`learned_routing.goodput`)
place a request in the window by its own `arrival_time_ms`. For non-root agentic requests, that is the replay's ready
time: `max(floor, dependency completion + delay)`. Every non-root request depends on earlier completions, so the
window's membership and denominator change with the policy under test.

**Effect (hypothesis, not measured).** A slower policy pushes late turns of plays that straddle the window edge out of
the window, rather than scoring them as not good.

**Suggestion for calibration.** For lowered open cells, define membership by the copy's arrival (the root `a_i`,
recorded in the meta), which is independent of the policy, and score all requests of plays that arrive inside the
window. The suggested loads are labeled "suggestion only", so this affects calibration design, not the parity result.

## Concurrence and notes

- **Arrival skeletons shared across splits.** I concur with isolation-arrivals-splits r0 m2. With equal seed and rate,
  train and test open files have identical arrival times (first four: 0, 58,864.190507, 113,870.6008 and
  162,520.565193 ms in both `9d89a9…` train and `f402f8…` test), because the arrival label carries no split. Play
  draws do carry the split.
- **Copy order.** It is a genuine free choice and I do not count it as a defect. Native corpora order plays by
  blake3(name); the generator orders them by copy index. Both are arbitrary, and every policy sees the same file.
  Replicates under "seed = k" re-draw plays and arrivals, which is a different and probably larger noise source than
  crn-order-v1's permutation of simultaneous arrivals. Calibration should measure noise for lowered cells with that
  scheme (A1: 8 replicates per family and N).
- **Untested importer branch.** No selected play exercises `async_launched` (background) subagents, so the
  `SubagentMode::Background` branch (spawn without a join) remains untested by data. The builder disclosed this.
- **Scratch.**
  - My evidence copy is 26 MiB, under `CR/runs/audit_agentx_lowering/parity_fidelity_r0/`.
  - Fresh generator files (154 MiB) and the dumper build (313 MiB) stayed in the session scratchpad and are not in CR.
    The CLI commands that regenerate the files are in `gen_meta/COMMANDS.txt`; their sha256 values are in the metas.
  - I did not edit CLEANUP.md. It is not in my write set.

## LESSONS

- **Applied:**
  - LR-01: windowed measurement. m2 concerns how the window membership is defined.
  - LR-02: noise comes from seeds and segments, not repeats. This is the note on seed = k replicates.
  - LR-09: causal release of agentic turns. Verified through dependency semantics (row 4) and the open-shift test
    (row 9).
  - LR-11: copies are not independent units. This is the note on replicates.
- **Out of lens:** all other lessons.
