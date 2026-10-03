# Audit: AgentX lowering sidecar (A3), lens "isolation-arrivals-splits", round 1

- Auditor: independent adversarial auditor, round 1, 2026-10-02 (dynamic: own scanners, own replays).
- Audited: WT `<commit-13>` (`learned_routing.workloads.agentx_lowered` after fix r0, which added the
  `think_mult`/`osl_mult` play transforms), the six base directories `CR/traces/agentx_lowered/base_cap300{,_think0.5,_think2,_think4,_osl1.5,_osl2.5}`,
  all 41 builder traces in `CR/traces/agentx_lowered/gen/` (30 sidecar + 11 fix r0), and `facts/agentx_lowered.json`.
- Method: the module under test was never imported. Its CLI was used only as a black box to generate 23 fresh traces.
  Every check is my own code under `CR/runs/audit_agentx_lowering/isolation_r1/scripts/`:

| Script | Purpose |
|---|---|
| `base_check.py` | pools and base-file labeling against the raw plays |
| `scan_gen.py` | static isolation, content, split, draw and arrival scan |
| `mutate.py` | planted defects, to measure the scanner's power |
| `run_replay.py` | replays through `dynamo.replay` |
| `causal_bound.py` | causal intra-copy bound on reuse and router overlap |
| `iso_compare.py` | each copy against its single-play replay |
| `release_check.py` | open-mode release rule |
| `arrivals_stats.py` | Poisson arrival statistics |
| `closed_steady.py` | lane recycling and steady state |
| `prefix_identity.py` | prefix stability of recycling |

- Replays: 50, each holding one `CR/slots` slot through `learned_routing.slots.SlotPool`, at most 2 at a time.
  - 45 small runs at N=1/2: 4 of them are positive controls.
  - 1 open run at N=8.
  - 4 closed-lane runs at N=32, 121-325 s wall each.
  - N=1 runs use `round_robin`, because replay rejects `kv_router` with one worker. All other runs use seeded
    `dynamo-default-cost-fn` (seed 1).
- I did not modify the module, the cells or any existing file, and I deleted nothing.
- Evidence: unless stated otherwise, every file named below is under `CR/runs/audit_agentx_lowering/isolation_r1/out/`.

## Verdict: PASS (0 blocker, 0 major, 4 minor, 1 info)

- **Isolation.** Recycled copies are isolated statically, and in replay at every scale tested, up to 2,560 concurrent-lane
  copies at N=32. This covers all six play transforms.
- **Split pools.** The pools are B2's, and every output is pure.
- **Open mode.** Root arrivals are a correctly seeded Poisson process at the requested total rate. Non-root timing is
  preserved exactly in the files and in replay.
- **Closed mode.** Lanes recycle exactly, and they reach a stationary window at N=32 once the window is long enough.
- **Minors.** The minors concern window sizing and same-seed coupling at integration time, not lowering bugs.

## Verified OK (own evidence)

### Pools and base files

**B2 pools** (`base_check.json`):
- `SPLIT_MANIFEST.agentx_play_subsets` gives train 33, val 21 and test 28 plays.
- Each pool equals the union of `transform.plays` over that split's AgentX cells.
- The subset sizes equal `facts/build_workloads.json`.
- The pools are pairwise disjoint.

**Base files**, for all 6 transforms × 82 plays = 492 files, with 0 errors:
- Each file is the lowering of the raw play it is named after:
  - `play_id` suffix = raw `id`;
  - rows = raw leaf requests;
  - input-length multiset = the raw one;
  - one namespace per file;
  - root `not_before_ms` = 0.
- Ids, hash ids, inputs and edges are identical to `cap300`.
- The MANIFEST split labels and multipliers match the directory.

**No hash id shared between plays** (`base_cross_play_hash.json`): each base directory has 182,261 distinct u64 hash ids,
and 0 of them occur in two plays.

### Static scan of generated files

`scan_builder.json`, `scan_fresh.json` and `scan_long.json` cover 62 files:
- 41 builder files and 21 of my own;
- 968,619 rows, 25,966 copies, 87,658 sessions and 55.3 M hash ids;
- train 38 / val 8 / test 16, open 31 / closed 31, all 6 transforms.

Result: **0 errors on every counter.**
- **IDs.**
  - Request ids are unique.
  - Every request, play, session and dependency id has its copy's `lrx:NNNNNN:` prefix and a single native namespace.
  - There is one play and one ordinal per copy, and ordinals are contiguous 0..C-1.
  - No session spans copies, and there are no dangling or cross-copy dependencies.
  - Rows are sorted.
- **Hash ids.**
  - The base→copy map is a bijection in every copy.
  - Each copy's hash set is contiguous, and the 25,966 intervals are pairwise disjoint.
  - The largest id is 6,042,287.
- **Content.** Every row equals its base row under the declared transform: lengths, model, api time, dependency
  relation, trigger, delay and target, native session and play.
  - The transform identified from content alone matches the declared one for every copy:

    | Transform | Copies |
    |---|---|
    | cap300 | 20,462 |
    | think0.5 | 2,888 |
    | think2 | 262 |
    | think4 | 1,464 |
    | osl1.5 | 164 |
    | osl2.5 | 726 |

- **Split purity.** Each copy's play is identified from content, not meta, and is always in the file's own split. No test
  play appears in any train or val output.
- **Draws.** Copy i's play equals `pool[index_draw(seed, "agentx-lowered-draw|"+split, i, |pool|)]` under my own blake2b
  implementation.
- **Timing.**
  - Closed mode: `not_before_ms` equals the base value exactly.
  - Open mode: `nb − a_i` equals `max(base nb − base root nb, 0)` (≤ 1e-6 ms).
  - Root `nb` equals my own Poisson re-derivation exactly (difference 0.0) in 31 of 31 open files.
- **Meta.** Copies, plays, hash ranges and starts all agree with the content.

**Power of the scanner** (`scan_mutations.json`): it flagged 12 of 12 planted defects:
- hash range overlap;
- a single cross-copy hash (caught by overlap plus map inconsistency);
- a train play inside a test file;
- a session moved across copies;
- a dependency crossing copies;
- a duplicate request;
- a non-root offset off by 1 ms;
- a root arrival off by 5 ms;
- a changed output length;
- a changed closed-mode `nb`;
- a copy with the wrong transform's timing;
- unsorted rows.

### Replay: copy 1 against its source play alone (`iso_compare.json`, `iso_compare_concurrent_vs_n1.json`)

The 2-copy same-play files come from the CLI, with seeds found by my own re-derivation of the draws. They cover three
plays in three transforms:
- 0013, test, identity transform: 119 rows, 4 subagent sessions;
- 0270, val, `osl_mult` 1.5: 112 rows;
- 0390, test, `think_mult` 4: 42 rows, 5 subagent sessions.

Each was replayed against the base file of that play alone, at N=1 and N=2.

| Form | Result |
|---|---|
| Open, copy 1 isolated in time (a₁ = 48,038 s / 89,020 s / 72,787 s) | Copy 1's `reused_input_tokens` equals the single play request for request: 119/119 and 112/112 at N=1 and N=2, and 42/42 at N=1. Its timeline equals the single play shifted by a₁ (≤ 2.1e-6 ms). Root reuse is 0. |
| Closed with `agentic_lanes=1`: copy 1 starts at copy 0's last terminal, while copy 0's blocks are still cached | Same as the open form; timeline ≤ 4.5e-8 ms |
| Closed without lanes: both copies concurrent from t = 0 | At N=2, each copy equals the single play at N=1 in every field (0.0 difference; 238/238, 224/224, 84/84). At N=1, contention lowers reuse in 138 of 238 requests (0013) and 18 of 84 (0390), never raises it, and root reuse stays 0. |
| 0390 at N=2, open and lanes1 | 41 of 42 equal; 1 request has *lower* reuse (see I1) |
| **Positive control**: my own file in which copy 1 reuses copy 0's hash ids | Root reuse rises to 50,544 tokens (0013) and 12,160 (0390). Copy 1 is above the single play in 47 and 117 requests (0013 at N=1/2) and in 3 and 14 (0390 at N=1/2). Sharing is detectable. |

### Replay: causal intra-copy reuse bound at scale (`causal_bound_small.json`, `causal_bound_n32_val.json`, `causal_bound_long.json`)

**The bound.** Replay synthesizes prompt tokens as the interned hash id repeated over each 64-token block, and hashes
16-token engine blocks with prefix chaining. Interning is per file and injective (`aisimulate-core` `driver.rs:1196-1272`,
`trace.rs:339-358`). So a request's reuse can never exceed

    floor(min(64·k*, ISL) / 16) · 16

where k* is the longest prefix the request shares with an *earlier-dispatched request of its own copy*. The same bound,
taken at dispatch, applies to the router's `reported_overlap_tokens` and `best_available_overlap_blocks × 16`.

**Result.** 0 engine violations, 0 router violations and 0 root reuse in every non-control replay:

| Replay | Requests | Copies |
|---|---|---|
| N=32 closed, test, 160 lanes | 46,484 | 1,280 |
| N=32 closed, test, 160 lanes | 94,688 | 2,560 |
| N=32 closed, val `think_mult` 0.5, 96 lanes | 32,717 | 768 |
| N=32 closed, val `think_mult` 0.5, 96 lanes | 65,174 | 1,536 |
| N=8 open, val `think_mult` 0.5 | 8,672 | 200 |
| All 2-copy and timing files | | |

- The bound is tight: reuse equals the bound for 16,463 of the 46,484 requests.
- The positive controls violate it: 47/116 engine and 118 router violations for 0013, 3/13 engine and 14 router for 0390.

**Session ids.** In `per_request`, every `session_id`, `play_id` and `conversation_id` carries its copy's prefix, and no
session id spans copies (0 violations over 168,534 records). The router's `SessionContext` comes from this id
(`agg.rs:650-690`), so sticky and learned-choice session state cannot cross copies.

### Non-root timing in replay (`iso_compare.json`, `release_*.json`)

**N=1 timing files.** These are open, rate 2e-6 per second, 6 copies each:
- val with `think_mult` 2;
- test with the identity transform;
- train with `osl_mult` 2.5.

Results:
- **Time-isolated copies** (16 of 18): every request equals the single-play replay shifted by a_i, in arrival, admission,
  first and last token, ttft, e2e, itl and reuse (630 requests, ≤ 2.1e-5 ms).
- **Overlapping copies** (2 of 18, in the osl 2.5 file): Poisson drew a 2,642 s gap, shorter than copy 4's duration.
  These two differ only in timing, from contention; reuse is equal in 61 of 61 requests.

**N=8 open run** (val, `think_mult` 0.5, 200 copies, 8,672 rows, all completed):
- 200 of 200 roots arrive exactly at my own Poisson draw (error 0.0).
- 8,472 of 8,472 descendants arrive at exactly max(a_i + base offset, max over dependencies of trigger time + delay_ms),
  where trigger time is dispatch or terminal (maximum error 0.0).
- The authored floor binds in 68 of those.

**N=1 osl 2.5 run:** 6/6 roots and 175/175 descendants exact.

### Open-mode arrivals (`arrivals_stats.json`)

**Distribution** (500 fresh seeds, 1,000,000 normalized gaps):

| Statistic | Value |
|---|---|
| Mean | 1.0013 |
| Variance | 1.0044 |
| KS against Exp(1) | p = 0.42 |
| Lag-1 autocorrelation | 0.0015 |
| Unit-window count variance/mean | 1.004 |
| Per-seed KS over 200 seeds | 5.5% below 0.05; uniformity p = 0.58 |

**Independence:** arrival draws are independent of play draws (Spearman max |ρ| = 0.067 over 50 seeds × 1,000).

**Rate:**
- `--play-rate-per-s` is the total rate.
- Doubling the rate halves every arrival to within 1 ns.
- Realized whole-file rates are 1.010–1.092 × nominal over 8 fresh files of 200–1,000 copies, which is sampling
  variation.

### Closed lanes at N=32 (`steady_*.json`, `prefix_identity_*.json`)

**Mechanics.** These hold in all 4 runs:
- every request completed;
- each copy runs in lane `ordinal % L`, 0 lane errors;
- each lane holds 8 or 16 copies in ordinal order, and every lane's first copy starts at t = 0;
- **every handoff gap is exactly 0 ms** (1,120, 2,400, 672 and 1,440 handoffs);
- in-flight plays equal L at every span boundary before the first lane runs out (714, 1,671, 221 and 915 checkpoints).

**Recycling is prefix-stable.** Doubling the copies per lane leaves the history byte-identical in every field until the
shorter file's first lane runs out:
- test: 28,196 of 28,196 requests;
- val `think_mult` 0.5: 11,243 of 11,243 requests.

So lengthening a cell never changes its earlier history.

**Stationarity.** Window = [p90 of first-copy ends, first lane out], in 300 s bins. Values are by thirds, with the OLS
trend t-statistic in parentheses.

| Run | Window | Completions/s | Mean ITL (ms) | Mean e2e (s) | In-flight requests | New prefill (ktok/s) | Reuse |
|---|---|---|---|---|---|---|---|
| test, 5/worker (160 lanes), 1,280 copies | 3,620–9,320 s (19 bins) | 2.92/2.81/3.01 (0.93) | 38.5/39.4/42.7 (**2.70**) | 29.1/30.6/30.1 (0.73) | 84.8/86.3/88.6 (0.76) | 110.8/111.9/119.7 (**3.18**) | 0.424/0.415/0.412 (−1.28) |
| same, 2,560 copies | 3,620–21,020 s (58 bins) | 2.92/2.97/2.99 (1.64) | 40.4/42.7/40.5 (0.72) | 30.0/31.8/30.0 (−0.35) | 86.7/96.2/88.0 (0.37) | 114.4/119.0/118.2 (2.53) | 0.417/0.405/0.421 (0.00) |
| val `think_mult` 0.5, 3/worker (96 lanes), 768 copies | 2,834–3,734 s (**3 bins**) | not assessable | | | | | |
| same, 1,536 copies | 2,834–15,434 s (42 bins) | 2.60/2.62/2.51 (−1.32) | 26.3/26.5/26.8 (0.46) | 20.8/21.0/21.5 (0.81) | 55.1/52.8/55.9 (0.01) | 65.2/65.7/64.6 (−0.58) | 0.606/0.611/0.607 (0.09) |

**Reading.** In the 8-copy test window, ITL and new-prefill trends look significant. With twice the window, ITL returns to
40.5 ms, and new prefill levels off (119.0 → 118.2). The 8-copy trend was play-mix noise in a window only about three
sojourns long, not drift. With the longer windows, both runs are stationary (|t| ≤ 2.53, ≤ 1.32 for val).

Bins overlap play sojourns of about 1,900 s and are autocorrelated, so a small |t| is necessary but not sufficient.

## Findings

### m1 (minor, new): "copies ≥ 8 × lanes" can leave a degenerate closed window at large N

**The guidance.** `facts.suggested_loads.closed_mode.copies` recommends at least 8 × lanes.

**The problem.** The window [W, T_out] runs from:
- W, the p90 over lanes of the first copy's end, set by the long plays;
- to T_out, the minimum over lanes of the sum of k sojourns, set by the luckiest lane.

So it shrinks with heavier-tailed pools and with more lanes:
- **val, `think_mult` 0.5, N=32, 3 lanes/worker, 768 copies:** T_out = 3,957 s and W = 2,834 s, a 1,123 s window (3 bins,
  about 0.7 of a mean sojourn). 65.5% of all requests are dispatched after T_out.
- **Same with 16 copies/lane:** a 12,705 s window, with 37% dispatched after T_out.
- **test, N=32, 8 copies/lane:** a 5,700 s window whose apparent ITL trend (t 2.70) disappears at 16 copies.

In the builder's own runs the window was 2,460 s on val at N=16.

**Fix (calibration):**
- Size the copies per cell so that T_out − W ≥ a target, e.g. ≥ 5 mean sojourns, and check it after the fact per run.
  Prefix stability makes extra copies cost-only.
- Consider `max_sim_time_ms` just past T_out to cut the drain (r0 m4).
- Window length shrinks with N, so check it at N=32, not N=8.

**Lessons:** LR-01, LR-12.

### m2 (minor, extends r0 m2, still open): same-seed coupling across splits, transforms, rates and cells

**Arrivals.** At the same seed, the arrival skeleton is identical across train, val and test, and across transforms
(think4 matches identity). It scales exactly with the rate (`arrivals_stats.json`).

**Play draws.** Draws are identical across all cells of a split. B2's per-cell subsets (A1–A3, V1–V3, T1–T4) therefore
collapse to one pool per split.

A3 mandates per-split pools, so the generator is compliant. But with replicate k = seed k:
- **All AgentX test cells replay the same copy sequence.** That is 8 base cells and 2 transform cells, differing only in
  N, lanes and transform.
- **The `agentx_plays` holdout loses its distinct subsets.** Its 4 cells were T1–T4 and no longer differ in plays. The
  test plays remain disjoint from train.

Under LR-11 and A4, the independent unit for AgentX is then the seed, not the cell.

**Fix:**
- Derive the seed from (cell_id, k), or add the split and cell to the draw and arrival labels.
- Record this in DEVIATIONS.
- Count seeds as segments in the headline test.

### m3 (minor, r0 m3, still open, now demonstrated): trace files do not guard their own mode

Replayed without `agentic_lanes`, a closed file starts every copy at t = 0 (`iso_*_closed_concurrent_*`: copies overlap
from 0), which is exactly the A3 failure.

Today `cells.py` rejects `agentic_lanes` for non-Weka formats, so nothing runs silently wrong yet. Calibration must
assert:
- `meta.spec.mode == "closed"` ⇔ `load.mode == "agentic_lanes"`;
- open ⇔ no lanes;
- `play_rate_per_s == load.value × N`.

### m4 (nit, r0 m5, unchanged at <commit-13>)

The header `source.digest` still hashes the base `MANIFEST.json` (`generate`, `spec_key`). The MANIFEST embeds the helper
binary's sha, so a rebuild of the helper changes every trace sha and cache key while every row stays the same. A digest
over the per-play `base_sha256` values would avoid this.

### I1 (info): the policy RNG stream is shared across copies

At N=2, copy 1 of 0390 diverged from the single play at `outer:6:inner:0`. It took worker 0 instead of 1 on an exact
tie (selected overlap = best = 1,068 blocks in both runs), because copy 0's routing had consumed draws from the seeded
default's tie-break RNG. One request has lower reuse, and timing differs by up to 4.2 s.

This is not a cache leak. It does mean a copy's routing depends on earlier copies through the RNG. This is the
crn-order-v1 class of variation, and it is shared by every policy under common random numbers.

### Carried from r0, not re-opened

**r0 m1.** All builder open-loop numbers come from one seed-0 realization. Fix r0 did not touch this. Calibration should
use ≥ 3 seeds and report the realized in-window rate.

## Scratch (auditor-created, not deleted)

`CR/runs/audit_agentx_lowering/isolation_r1/` holds 3.7 GiB:

| Path | Size | Contents |
|---|---|---|
| `gen/` | 2.8 GiB | fresh traces, regenerable with the CLI |
| `gen_mut/` | 916 MiB | planted-defect copies |
| `replays/` | 38 MiB | `per_request` gz |
| `out/`, `scripts/`, logs | < 1 MiB | |

Two other items:
- A path typo of mine created `CR/runs/audit_agentx_lowered/isolation_r1/{gen,scripts}`. Its contents were moved into
  the directory above, and the now-empty directories remain.
- `scripts/probe_write.txt` (6 bytes) is a leftover probe file.

## LESSONS

- Applied:
  - LR-01: window sizing, drain (m1);
  - LR-02: seeds as the noise unit;
  - LR-09: causal release verified exactly;
  - LR-11: independence unit (m2);
  - LR-12: window shrinks with N (m1).
- The other lessons are outside this lens.
