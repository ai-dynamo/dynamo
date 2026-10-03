# Audit: AgentX lowering sidecar (A3), lens "parity-fidelity", round 1

- **Auditor:** independent adversarial auditor, round 1, 2026-10-02. Dynamic audit with my own code and my own replays.
- **Audited:**
  - WT `<commit-13>`, which contains `learned_routing.workloads.agentx_lowered` (fix r0: the `think_mult`/`osl_mult`
    transforms) and `tools/agentx_lower`. Its 18 module tests pass at this commit.
  - `CR/traces/agentx_lowered/{weka,base}_cap300{,_think0.5,_think2,_think4,_osl1.5,_osl2.5}` and their six MANIFESTs.
  - `facts/agentx_lowered.json`, `runs/agentx_lowered/parity/parity.json` and `runs/agentx_lowered/fix_r0/parity_<tag>/parity.json`.
- **Method:**
  - I never imported the module under test. Its CLI was used only as a black box, to produce fresh generator files.
  - Every check is my own code, in `CR/runs/audit_agentx_lowering/parity_fidelity_r1/scripts/`. Only replays import
    the harness, and only for `learned_routing.slots.SlotPool`.
  - The round-0 auditor's `semantics.py` took the session partition from the rows. In this round, `my_lower.py` is a
    from-scratch Python re-implementation of **all** of `weka.rs::lower_trace`. It covers preamble split, chain
    detection, seam splicing, worker streams, owner and join selection, cross-stream frontiers, anchor spawns, the
    timestamp basis, `seconds_to_milliseconds` with Rust half-away-from-zero rounding, and `normalized_hashes`. It
    was written from the `aisimulate-core =0.13.0-dev.202609300000000061` source.
- **Replays:** 661, all through `run_trace_replay`. Each process ran its replays sequentially while holding one
  `CR/slots` slot, and at most 2 of my processes ran at once.
- **Not modified:** the module, the cells, and every existing file. Nothing was deleted.

## Verdict: PASS (0 blocker, 0 major, 2 minor)

Every parity check I ran holds exactly:
- The lowered rows are AISim's Weka lowering, field for field, for all 82 plays × 6 transforms.
- Unmodified generator output replays per-request identically to native Weka, in all 37 `per_request` fields, from a
  single copy up to a whole 192-copy file at N=8 with 32 lanes. This holds for the identity transform and for both
  test hold-out transforms (think ×4, osl ×2.5).
- The builder's parity claims reproduce from its evidence files.

## Plays chosen for my single-copy replays

None of these plays is in the builder's parity set (0098, 0073, 0390, 0013, 0021, 0315), nor in round 0's set
(0007, 0010, 0293, 0325). The table counts edges from my census of the base rows.

| Play | Split | Rows | Structure |
|---|---|---|---|
| 0032 | test | 40 | 2 subagents: 2 joins, 2 dispatch spawns and 1 completion spawn, 1 worker stream, 6 replay barriers |
| 0170 | val | 61 | 3 subagents: 3 joins, 2 dispatch spawns and 1 completion spawn, 17 barriers |
| 0081 | train | 76 | 1 subagent: 1 completion spawn, 1 join, 3 detected worker streams, 42 barriers |

- **Zero-output request:** only play 0021 has one among the 82 plays, and it is the builder's. It is still covered
  by my semantic and hash checks under all six transforms (rows 1–2 of the table below). The builder replayed it for
  all six transforms.
- **Background subagents:** no play has an `async_launched` subagent.
- **Recycled tiers:** the multi-copy tiers (rows 7–8 of the table below) add 17 more plays that neither earlier
  party replayed. The 192-copy files contain all 28 test plays.

## Verified OK (own evidence)

All paths are relative to `CR/runs/audit_agentx_lowering/parity_fidelity_r1/`.

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | **Edges and delays against weka.rs semantics.** My full re-lowering of `weka_<tag>/<play>.json` was compared with `base_<tag>/<play>.jsonl` for every play and every transform. Compared: request-id set and order; play, session and ordinal; model and lengths; `not_before_ms` and `recorded_api_time_ms` as exact floats; dependency lists as ordered (target, relation, trigger, `delay_ms`) tuples; hash-identity bijection. | **82/82 plays exact for each of 6 transforms.** Each transform has 3,132 rows and 4,330 edges. The timestamp basis matches the MANIFEST (64 not_applicable, 18 absolute, 0 relative). Power: 10/10 planted corruptions detected (sequence delay +1 ns, dropped barrier, dropped join, spawn trigger flip, `not_before` +1 ms, output +1, api ×1.0001, session swap, hash merge, hash split). | `out/semantics.json`, `scripts/my_lower.py`, `scripts/semantics_check.py` |
| 2 | **Exact u64 hash ids and namespaces.** blake3 recomputation of `namespace(rel)` and of `unique_hash` with the nonce rule, from my lowering's block identities. | **3,245,191/3,245,191 hash values exact for each transform**, and 82/82 namespaces. Every value needed nonce 0. | `out/exact_hash.json` |
| 3 | **Transformed Weka plays.** (a) Byte comparison with the play lines in B2's cell traces. (b) My own think dilation and osl rule applied to `weka_cap300`. | (a) cap300: 82/82 distinct plays (150 occurrences in 17 cells). think0.5, think2 and osl1.5: 11/11 each. think4 and osl2.5: 7/7 each. (b) 82/82 exact for each of the 5 transforms. | `out/weka_transform.json` |
| 4 | **MANIFEST integrity.** Recorded raw, weka and base sha256 against the files on disk, and each weka/agentic digest pair. | 0 mismatches in 6 × 82 plays. MANIFEST shas: cap300 `9868b9f6…`, think0.5 `da6d56a8…`, think2 `9a45b84d…`, think4 `3561c1ec…`, osl1.5 `92c6314c…`, osl2.5 `c7a7a70a…`. | `out/manifest_integrity.json` |
| 5 | **Fresh generator files.** 24 files, black-box CLI, 120 copies, seed 31: {train, val, test} × {open, closed} × {identity, think4, osl2.5, think0.5}. Each copy was checked against the base rows of the transform the file claims. | 0 errors. Checked: ids and prefixes, deps, lengths, api; closed nb = base nb; open nb = a_i + base offset; per-copy hash bijection; row sets; ordinals; meta transform tag and base dir; draws from the own split only. | `gen/cuts/cut_report.json`, `gen/COMMANDS.txt` |
| 6 | **Single-copy replay parity, my 3 plays × 4 transforms** (identity, think4, osl2.5, think0.5). Native `weka_<tag>` play compared with the base file and with one copy cut from a fresh **closed** generator file (generator header, label and hash range kept; ordinal reset to 0). N 2/4, seeded default and round robin, timestamp mode and `agentic_lanes=1`. | **base 96/96 and closed copy 96/96 `per_request` identical** in all 37 fields, 5,664 rows each, with only the `lrx:` label stripped | `out/replay_parity.json` |
| 6b | **Open-mode single copies,** same 96 configurations. Replay's `normalize_starts` subtracts the minimum `not_before_ms`, so a lone copy at a_i (about 4.1e6 ms) is rebased to 0. | 96/96 equal to native within 9.3e-10 ms, which is the floating-point residue of (a_i + o) − a_i. 48/96 are bit-exact. ttft, e2e, itl and ttst are exact in 96/96, and non-float fields (workers, reuse, histories) match in 96/96. | `out/open_copy_tolerance.json` |
| 7 | **Untouched 12-copy generator prefixes, transform cells.** The first 12 copies of fresh closed files were replayed byte-for-byte as generated: test think4, test osl2.5 and train think0.5, with 8–9 distinct plays and repeats. The native reference is a corpus with one Weka file per copy; names were found by blake3 search so that traversal order and namespace order both equal copy order. N 2/4, both policies, timestamp mode and lanes 3/5. | **36/36 identical** (323, 323 and 346 rows) | `out/recycle_transform_parity.json`, `recycle/` |
| 8 | **Scale: a whole untouched 192-copy file** (test, seed 47, all 28 test plays, 7,680 rows) against a 192-file native corpus. N=8, 32 lanes (4 per worker, 6 copies per lane), both policies, identity, think4 and osl2.5. | **6/6 identical, 46,080 rows**, all fields | `out/scale_parity.json` |
| 9 | **Builder's replay claims.** `parity.json` and the five `fix_r0/parity_<tag>/parity.json` files. | 84/84 per transform (504/504 in total). Each is non-vacuous (rows = completed on both sides, `differing_rows` = 0 including uuid). Tiers: single 48 (timestamp mode only), corpus 24 (none/2/3), recycle 12 (none/3/5). The native corpus and recycle files in each transform's parity directory are byte-identical to **that** transform's `weka_<tag>` plays (6/6 and 12/12, 0 from other tags). Each transform's reference differs from cap300 in 84/84 paired cases, so the transform parity is not a re-run of cap300. Copy-order spreads match the facts: cap300 −7.44…+4.71%, osl1.5 −5.68…+18.04%, osl2.5 −15.61…+13.52%. | `out/claims_check.json` |
| 10 | **Helper vs bindings parse path.** The helper and the bindings lock the same `aisimulate-core`, `serde_json 1.0.150`, `blake3 1.8.7` and `ryu 1.0.23`. Neither enables `float_roundtrip` or `arbitrary_precision`. The only feature difference is the bindings' `preserve_order`, which is irrelevant to typed Weka structs. | The helper's digest pair cannot hide a float-parse difference from native replay. Rows 1, 2 and 6 confirm this directly. | `cargo tree -e features -i serde_json` (inline) |
| 11 | **Base selection, black box.** | `--think-mult 3` and `--think-mult 4 --osl-mult 2.5` fail loudly (FileNotFoundError naming the missing `base_<tag>`). `--think-mult 4` and `4.0` give byte-identical files. An explicit identity (`--think-mult 1 --osl-mult 1`) is byte-identical to the plain spec. | inline |

**Conclusions on parity:**
- The A3 statement holds for every AgentX transform the cells use. Dependency edges, triggers, `delay_ms`, tool gaps
  and lengths are exactly AISim's Weka lowering.
- Recycling only relabels ids, bijectively remaps hash ids into disjoint per-copy ranges, and (in open mode) offsets
  `not_before_ms` by a_i.
- With node order matched to the native order, unmodified generator output is per-request identical to native Weka.
  That includes a whole 192-copy lanes file for the test hold-out transforms.

## Findings

### m1 (minor): `generate` trusts the base files without checking them against the MANIFEST

**Where.** `generate` (`agentx_lowered.py:651-728`) checks only two things from `MANIFEST.json`:
`all_graph_digests_equal`, and the transform it was built for. `load_base` (`:489`) then reads
`base_<tag>/<stem>.jsonl` without comparing it to `plays[name].base_sha256`. The trace meta records only the
manifest's sha.

**Failure scenario.** `build_base` rewrites `weka_*` and `base_*` files inside its per-play loop and writes
`MANIFEST.json` only at the end. A crashed or partial re-run (or any later edit to a base file) leaves new rows under
an old MANIFEST that still says `all_graph_digests_equal: true`. Generated traces would then silently carry a
`base_manifest_sha256` that does not describe their rows. The graph-digest proof would no longer cover them.

**Today.** 0/492 files mismatch (row 4), so no current trace is affected.

**Fix.** Verify the sha256 of each drawn base file against its MANIFEST entry in `generate` (cache per file), or
record per-play base shas in the meta.

### m2 (minor, latent, calibration-facing): replay rebases every agentic file to its minimum `not_before_ms`

**Mechanism.** `normalize_starts` is applied to every agentic trace (bindings `replay.rs:1325`, aisimulate
`python.rs:1159`). Meta `copies[].start_ms` therefore equals replay time only when the file contains copy 0
(a_0 = 0, root nb = 0). That is true of every whole generated file.

**How it can break.** It stops being true if calibration slices an open file, for example by dropping leading
copies. All times then shift by the first remaining a_i. That would quietly break the policy-independent window
membership by `copies[].start_ms` that round-0 m2 suggested and fix-r0 endorsed.

**Evidence.** My lone open copies, with a_i ≈ 4.07e6 ms, replayed starting at 0 (row 6b). My first, shifted
comparison of them (48/48 "different" in `out/replay_parity.json`, key `open_copy_shifted`) was a test-design error
on my side, superseded by row 6b. It is not evidence against the lowering.

**Fix.** Add one sentence to the module docstring and the facts: membership by `start_ms` assumes the whole file;
otherwise subtract the file's minimum `not_before_ms`.

## Notes

- **Copy order** remains a free, disclosed choice. My exact matches all used native order equal to copy order.
- **Not re-verified here:** open-mode Poisson statistics and isolation. Those were audited by isolation-arrivals-splits r0.
- **LESSONS.**
  - Applied: LR-09 (causal release of agentic turns; rows 1, 6 and 8) and LR-11 (copies are not independent; nothing
    in this lens changes that).
  - Out of lens: all other lessons.

## Scratch (not in CLEANUP.md; my write set excludes it)

- **CR evidence:** `CR/runs/audit_agentx_lowering/parity_fidelity_r1/`, 49 MiB. Most of it is the INFO-level
  `logs/replay_parity.log`, the recycle corpora and the cut copies.
- **Stray empty directory:** `CR/runs/audit_agentx_lowered/parity_fidelity_r1/scripts/`, created by a path typo. Its
  one file was moved into the evidence directory. The directory was left in place rather than deleted.
- **Session scratchpad** (`<session-scratch>`):
  - `pf_r1_gen` 766 MiB, `pf_r1_scale` 172 MiB, `pf_r1_scale_native` 108 MiB, `pf_r1_neg` 3.6 MiB;
  - `pylib/` holds the blake3 1.0.10 wheel, installed with `uv pip install --target` for the audit only.

  These files regenerate from `gen/COMMANDS.txt`. Their sha256 values are in the metas and in `out/*.json`.
