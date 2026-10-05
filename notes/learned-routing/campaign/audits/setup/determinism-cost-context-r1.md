# Audit: setup checkpoint, lens "determinism-cost-context", round 1

- Auditor: independent dynamic auditor (adversarial), 2026-10-02, round 1. Round 0 and the setup fixer's
  round-0 fix were read and treated as claims to re-verify, not as evidence.
- Verdict: **FAIL**, because of one **major** finding (F1): the shared "toolagent" trace is a relabeled copy of the
  shared Mooncake trace, so the two are not independent workload families. Every setup claim in this lens
  reproduced. One stale fact (F2) was a one-field correction, and I fixed it.
- Evidence root: `CR/runs/audit-setup/determinism-cost-context-r1/` (`A/` below), with `CR` =
  `<campaign-root>` and `WT` = `<worktree>`.
  - Scripts are in `A/scripts/`: `audit_runner.py`, `long_context_r1.py`, `session_replicates.py`,
    `replicate_check.py`, `weka_degenerate.py`, `e0_cost.py`, `trace_identity.py`, `summarize_r1.py`.
    Job specs are in `A/jobs/`, outputs in `A/out/`, and every number below is collected in
    `A/audit_summary.json`.
  - My runner reuses setup's `smoke_lib.run` and canonical compact hash, so hashes compare across sessions.
    It adds a stricter hash over **every** per_request field except `uuid`, and a per-request goodput
    recomputation.
- Method: about 55 short replays (0.4–1.7 s each), strictly one at a time, on an idle 24-core host (load 0.08).
  The harness slot pool `CR/slots/` still does not exist, so they ran outside it; this is recorded in
  `facts/DEVIATIONS.md`.

## Claims re-verified

| Claim (setup) | My evidence | Result |
|---|---|---|
| WT on `rupei/learned-routing`, base `2be8d9c43a` = `origin/rupei/router-policy-aic-ttft` | `git merge-base HEAD origin/...` = `2be8d9c43ac1b275fa6e5743caee9b01c0522463`, the same SHA that `git ls-remote origin` reports. HEAD is now `<commit-02>` (fixer) on top of `<commit-01>` (setup). Both commits are signed off, with no Co-Authored-By. Upstream is unset and the tree is clean. | PASS |
| Bindings unchanged since setup | `_core.abi3.so` SHA-256 `2679f4dd…` equals `facts/setup.json`, built at 13:21:03. The fixer commit touches only `benchmarks/learned_routing/` (Python). | PASS |
| Main checkout untouched | `git status` shows exactly the 3 pre-existing `M` files (mtimes 2026-10-01 21:49–22:13) plus `?? notes/`. HEAD is `8cd79d84ee` on `rupei/recovery-snapshot-cache`. The only file modified today outside `.git/.claude/target` is `notes/learned-routing/PLAN.md`, the top-level session's file (A2 edit at 14:11). | PASS |
| Determinism, pair 1: `dynamo-two-tier-cost-fn`, N=8, 2000-row Mooncake | Two fresh processes give identical full and compact hashes. The compact hash `bc73ec6b…` equals **setup's recorded hash** (cross-session). Goodput is 3.2145089936290185 in all three, and recomputed goodput = native. | PASS |
| Determinism, pair 2: synthetic sessions + lmetric (60 sessions × 3 turns) | Two fresh processes give identical hashes. Compact `93df9733…` equals setup's, at goodput 3.125169344380584. | PASS |
| Seed reproduces across sessions | Seeded default seed 2 at N=8 gives compact `67c0e9c8…`, equal to setup's, at 2.8180453202645364. | PASS |
| New: seeded default at `router_temperature` 0.5 (CMA-ES will tune it) | Two fresh runs are identical (2.947404738381404). Seed 2 gives 2.966905317278522, so the softmax draws are seeded and temperature takes effect. | PASS |
| New: `router_assume_kv_reuse=false` switches to unseeded `fastrand` tracking hashes (`scheduling/config.rs:1623`) | Two fresh runs are still identical (3.0802940115734807); the hash values only act as identities. | PASS |
| Unseeded default nondeterministic | Reconfirmed in passing: setup's cost specs use the no-YAML default, and identical reruns gave closed_n32 13.148 / 13.404 (setup 13.045) and open_n16 6.371 / 6.398 (setup 6.346). That is about 2% tie noise at N=32 closed. | PASS (confirmed) |
| No wall-clock leaks into replay | Replay builds its slot tracker with `new_without_expiry` (`mocker/src/replay/router_shared.rs:143-155`). Decay uses virtual time (`kv_router/mod.rs:585-588`). | PASS (static) |
| Cost: fresh 2000-row replay (2 data points round 0 did not use) | closed_n32: 1.511 / 1.567 s vs setup 1.497 s (×1.01 / ×1.05). open_n16: 1.588 / 1.636 s vs 1.638 s (×0.97 / ×1.00). `/usr/bin/time`: 99–100% CPU, 251–261 MiB maxrss. **Within 2×: yes, within 5%.** | PASS |
| Cost: in-process repeat ~0.95–1.0 s | The second replay in one process took 0.94 s (lmetric, `A/out/replicate_replay_k5.json`). | PASS |
| 128K: single 120K request, TTFT 17.8 s | Rerun: **17,815.540859 ms**. It equals my independent AIS chunked-prefill sum (15 chunks of ≤ 8192 at their prefixes) to 2e-7 ms. A length setup never ran, 100,000 tokens, gives 13,226.168107 ms, equal to its chunk sum. | PASS (exact) |
| 128K TTFT physically plausible | Chunk-by-chunk FLOPs from aisimulate's `Qwen--Qwen3-32B_config.json`: 64 layers, 64 query heads, 8 KV heads, head dim 128, hidden 5120, FFN 25600. That is 31.2B linear parameters; causal attention is 4·L·n_q·d·new·(prefix + new/2). Against 2 × 989 TFLOP/s dense BF16, the whole 120K request runs at **64.1% MFU**. Every chunk from prefix 0 to 114,688 sits at **61.8–64.6%**, so the estimator's prefix-linear attention term has constant efficiency out to 128K, which is plausible. ITL at 120K is 21.2 ms against about 14.2 ms of bandwidth-bound weights and KV reads (about 67% of HBM bandwidth), also plausible. | PASS ([hypothesis]-level physics check) |
| New: long prefix reuse at 120K | Request 2 extends request 1's 119,808-token prefix by 192 tokens and arrives after request 1 finishes. It reports `reused_input_tokens` 119,808 and TTFT 56.06 ms, equal to the estimator for 192 new tokens on a 119,808 prefix (56.0628). A same-length request with no shared prefix pays 17,815.54 ms. | PASS |
| `session_context` reaches policies | **Static, my own trace.** aisimulate-core `components/admission.rs:258,309` sets `session_id = emit_session_metadata.then_some(...)` for driver sources, and replay-context ids for request sources (`:233-237`). `agg.rs:667-673` passes it to `assign_request`, then `place()` (`:341`). WT `kv_router/mod.rs:454-473` passes it to `build_pending_request`, then `PendingRequest.session_id` (`:927`), then `SchedulingRequest.session_context` (`:321-324`), then `WorkerSelectionContext::session_context()` (`context.rs:69`). The session-less `on_request_arrival` is `#[cfg(test)]` only (`mod.rs:593-601`). **Dynamic, new path:** on a tied multi-turn trace (24 sessions × 3 turns, first arrivals tied in groups of 6), every crn-order-v1 replicate k=0..3 and the identity order give 72/72 session ids, turns in order, and an unchanged (session, ISL, OSL) multiset (`A/out/session_replicates`). | PASS (see F4 for scope) |
| `expected_output_tokens` = true OSL | Static: `mod.rs:459-463` gives `max_output_tokens = effective_max_output_tokens()`, and `:920-923` sets `Some(max_output_tokens)`. It feeds `SequenceRequest`, but `single.rs:260` discards it, and replay never calls `add_output_block`, so no load signal derives from it. The live default `router_track_output_blocks` is `false` (`config.rs:999`), so `decode_cost_blocks` has information parity. Dynamic: `requested_output_length == output_length` on 2000/2000 (two-tier N=8) and 180/180 (synthetic). The trace's (ISL, OSL) multiset equals the replay's (ISL, requested OSL). | PASS: the leak is real; learned features must never read it |
| Setup fixer's CRN replicates (`crn-order-v1`) | Materialized k=0..7 under `PYTHONHASHSEED` 0 and 12345: byte-identical to each other and to the fixer's files. Replaying the fixer's k=5 (default seed 6, lmetric) reproduces its recorded per-request SHA-256 and goodput exactly. Recomputed from raw `runs/setup-fix-r0/out/*.jsonl`, every per-cell CV and paired-ratio sd in `facts/noise.json` matches, including pooled 0.03565 / 0.0219 / 0.03071. 11/11 unit tests pass. Materializing the full 23,608-row trace takes about 60 ms per replicate (1,180 tie groups, about 22.4k rows moved). | PASS |
| Within-tie order is file order (the CRN mechanism) | aisimulate-core `admission.rs:205-230` pops due requests in queue order and breaks at the first not yet due. The session driver also honors it: on my tied session trace, first-turn worker assignments change for every k. | PASS |

## Findings

### F1 (MAJOR): the shared "toolagent" trace is a relabeled copy of the shared Mooncake trace; they are not independent families

- **Claim being audited.**
  - PLAN's workload table lists Toolagent (flat) as its own family.
  - Test axis 3 holds out a family.
  - LESSONS LR-02 (action 2) and LR-11 count "distinct Mooncake and toolagent windows" as independent
    workload segments.
  - Setup inspected both files ("these fit, max 125,878 and 126,527") without checking that they are
    independent.
- **Evidence** (`A/out/trace_identity/trace_identity.json`; script `A/scripts/trace_identity.py`). Row-aligned,
  `<traces>/toolagent_trace.jsonl` (SHA `48a2db1a…`) against `mooncake_trace.jsonl` (`b434f181…`):
  - `output_length` is identical on **23,608 / 23,608** rows.
  - `input_length` is identical on 9,260 rows. The median |ΔISL| is 2 tokens (0.05%), p99 is 80, max 805.
    The block count is identical on 23,349 rows.
  - The same 1,180 arrival groups, with timestamps scaled by about 0.9825 (3,600,000 → 3,536,999 ms).
  - A consistent hash-id bijection maps Mooncake onto toolagent with **0 conflicts over 409,356 aligned
    block positions**, so the prefix-sharing graph is the same.
  - The per-row prefix-reuse fraction correlates at **0.9998** (means 0.62979 vs 0.62975).
- **Provenance.**
  - The upstream file is the same: Mooncake's `FAST25-release/traces/toolagent_trace.jsonl` has the
    identical SHA-256 `48a2db1a…`. The shared-corpus doc (`<trace-corpus-index>:20`)
    nevertheless labels it an "Applied Compute-derived workload".
  - `FAST25-release/traces/conversation_trace.jsonl` (`b8cbb061…`, 12,031 rows) is no clean substitute
    (`A/out/fast25/conversation_vs_mooncake.txt`):
    - 10,938 of its rows (91%) match a toolagent row exactly on (timestamp, ISL, OSL);
    - it uses toolagent's 0–3,536,999 ms grid of 1,180 arrival groups;
    - its hash ids differ (6 of 10,938 matched rows identical) and its reuse is 0.384 vs 0.630.
  - So it is a different prefix structure on a mostly shared arrival and length skeleton.
- **Why it matters.** Without a fix the campaign would:
  1. call Mooncake → toolagent a "held-out family" when it is the same workload, inflating the
     generalization claim;
  2. count a Mooncake window and the matching toolagent window as two independent segments, halving the
     effective sample behind the noise rule's 2 × SE clause, the pilot gate and the headline test. That is
     pseudo-replication (LR-11);
  3. let a Mooncake window in train and the same toolagent window in validation or test leak across the
     split.

  None of this has happened yet: `CR/cells/` and `CR/traces/` are empty and nothing is frozen. That is why
  this is major, not a blocker.
- **Fix.**
  1. Record the identity in facts.
  2. Drop toolagent as a family, or treat it only as a perturbed copy of Mooncake: same segment ids as
     the Mooncake window it copies, never held out against Mooncake, never an independent segment.
  3. If a second flat family is wanted, the build stage's trace step should run
     `A/scripts/trace_identity.py`-style checks (OSL alignment, hash bijection, reuse correlation) on every
     candidate before use. If FAST25 `conversation_trace` is used, label it as sharing Mooncake/toolagent's
     arrival and length skeleton, so its windows are correlated with theirs and are not independent
     segments.
  4. The top-level agent should correct the label in `<trace-corpus-index>` (an instruction
     gap in user-space skills; not changed by me).

### F2 (MINOR, FIXED): `facts/setup.json` misstated the `max_model_len` rule (round-0 F2, still open)

- Re-confirmed (`A/out/mml/mml.json`):
  - ISL 130,900 + OSL 500 → `completed`, `output_length` 172, `requested_output_length` 500, TTFT 20,590.92 ms.
  - ISL 131,072 + OSL 1 → `rejected`, per-request `ttft_ms` null.
- **FIXED.** `long_context.notes` in `facts/setup.json` now states the real rule:
  - only ISL ≥ 131072 is rejected;
  - ISL+OSL > 131072 is silently truncated, so trace preparation must drop or flag it.

  The edit is one field and the one-line diff was verified; it cites both auditors' evidence files.
- For truncated rows `expected_output_tokens` (= requested) ≠ realized output, which is one more reason the
  harness must recompute everything from realized per_request fields.

### F3 (MINOR): AgentX `trace_timestamps` mode starts every play at t=0 and equals `agentic_lanes ≥ plays`; Weka replicates can be inert while reported non-degenerate

- **Code.** aisimulate-core `loadgen/weka.rs:999` sets `not_before_ms = t − root_time`, so each play is
  rebased to its own root and all plays start together at t=0. Lanes take plays round-robin
  (`driver.rs:1344-1352`, `play_index % lane_count`).
- **Dynamic** (`A/out/weka_degenerate.json`). On the 2-play sample at N=2 with lmetric:
  - timestamp mode and lanes=2 give the **same outcome hash and goodput** (0.04314204137748361) on all four
    replicates, including k3, whose play order is swapped;
  - lanes=1 gives 2 distinct outcomes;
  - both plays' first requests arrive at 0.0 ms in timestamp mode.
- **Impact.**
  - PLAN's "AgentX `trace_timestamps` with speedup" is not an open-loop arrival process. It is a
    synchronized start of every selected play: for 82 plays, 82 sessions at t=0, then a decaying,
    non-stationary load. It is not a load mode distinct from lanes.
  - For crn-order-v1, `permute_weka_plays` always reports `units_in_ties = plays`, so `degenerate=False`,
    even when only the t=0 tie order changes. On small play subsets the replicates can be inert for
    deterministic policies.
  - `noise.differs_beyond_noise` flags degeneracy only when pooled sd == 0, and the default's seed k+1 keeps
    it above 0, so a partial false floor (tie-seed noise only) can return.
- **Fix.**
  - Calibration uses lanes < plays as the AgentX load mode (LR-09), or labels timestamp cells as
    "synchronized start" and adds a seeded play-start stagger transform, which also gives a real replicate
    perturbation.
  - `lr-eval` adds a realized-degeneracy check: if every replicate of a deterministic policy is
    outcome-identical on a cell, mark the cell degenerate and use segments.

### F4 (MINOR): replay's `session_context` carries only `session_id`

- WT `kv_router/mod.rs:321-324` builds `SessionContext::new(session_id, None, None, None)`, so
  `parent_session_id`, `session_final` and `input_trigger` are always None in replay
  (`scheduling/types.rs:320-345`).
- Setup's "session_context reaches the policies" is true for the id only. In replay:
  - AgentX subagent sessions (`weka:<play>:session:subagent:…`) carry no parent link to the root's worker;
  - sticky-session cannot evict on a final marker and relies on its LRU;
  - single-turn Mooncake has no context at all (as setup said).
- **Fix.** Record the scope in facts. `learned-choice` feature 6 and `sticky-session` must use only
  `session_id()`. Any parent-aware variant needs a replay patch (logged as a follow-up) or must not be
  claimed.

### F5 (MINOR, build guidance from LR-03): no stable request id is available to policies

- `WorkerSelectionContext` (`context.rs:23-115`) exposes no request id. The internal one is the request
  UUID (`mod.rs:270-272`), which is random v4 for Mooncake replay (UPSTREAM_FOLLOWUPS #2).
- LR-03 action 2's alternative, counter-style draws from hash(seed, request id), is therefore unavailable.
  It would be nondeterministic if anyone exposed the UUID.
- **Fix.** `learned-choice` and `sticky-session` use one per-instance seeded `fastrand::Rng`, consume a
  fixed number of draws per decision (LR-03), and never key RNG, or iteration-order-dependent state, on
  UUID-derived values. Policy state keyed by `session_id` is deterministic: Weka ids are content-derived,
  and Mooncake/synthetic ids come from the trace or spec.

### F6 (MINOR, Amendment A2's E0): the estimator shortcut and its cost

- A single-call `predict_prefill(1, ISL, 0)` underestimates replay's chunked prefill by **1.5% at 120K and
  1.4% at 131K**. It is exact at ≤ 8K (round-0 `long_context.json`).
- Replay's TTFT equals the **chunked** estimator sum exactly (above).
- Cost (`A/out/e0_cost.json`):
  - a single-request replay takes about 0.39 s, and E2E = TTFT + (OSL−1)·ITL exactly;
  - an estimator prefill call takes 0.88 ms and a decode call 2.9 µs;
  - Mooncake has **17,185 distinct (ISL, OSL) pairs**.
  - So E0 by single-request replay is about 1.9 CPU-hours per Mooncake-sized family (about 6 min on 20
    slots), and by chunked estimator sums about a minute, before caching.
- **Fix.** Calibration computes E0 by single-request replay, or by the chunked estimator sum after
  checking it against replay on a sample of long requests. It must never use a single unchunked call.
  Record the method in `facts/calibration.json`, as A2 requires.

## Round-0 items still open (not re-raised)

- r0 F4: `runs/setup/cost/inprocess/result.json` still has no generating script. My in-process point
  (0.94 s) corroborates the numbers.
- r0 F5: cost was projected from 2000-row cells. With A1's replicate floors (≥ 2 per CMA-ES evaluation,
  ≥ 3 for validation and test) and full-trace replays of 7.6–8.8 s (r0), the pilot must project cost per
  family × K. Whether CPU cluster is needed is still open.
- r0 F6: the AgentX lanes evidence was vacuous. It is now explained by F3: lanes ≥ plays is the same as
  timestamp mode.

## LESSONS

- **LR-02, applied.** I re-verified the CRN replicate protocol that replaces repeat-sd: byte-stable,
  replays reproduce, and noise.json recomputes. I found its Weka blind spot (F3).
- **LR-03, applied.**
  - Seeded tie-breaks and softmax are deterministic at temperature 0.5.
  - Its request-id counter option is unavailable (F5).
- **LR-09, applied.** AgentX timestamp mode releases turns causally within a play and starts all plays
  together, the same as lanes ≥ plays (F3).
- **LR-11, applied.** Segment independence is the basis of F1.
- **LR-14, applied (information parity, decode load).** Replay tracks no output blocks, and the live
  default doesn't either, so `decode_cost_blocks` has parity.
- **Out of lens:** LR-01, LR-04–08, LR-10, LR-12, LR-13 and LR-15.
- **Amendment A2:** applied to E0 (F6).

## Process notes

- **Replays.** About 55 replays ran strictly sequentially outside the non-existent slot pool. The host was
  otherwise idle.
- **Downloads.** Two public traces were downloaded from GitHub (Mooncake `FAST25-release/traces`,
  7.2 MB) into `A/out/fast25/` as evidence for F1.
- **Writes.**
  - The only edit outside `A/` is the one `facts/setup.json` field (F2).
  - I appended `facts/DEVIATIONS.md`, `CLEANUP.md` and `facts/STATE.md`.
  - No WT or main-checkout changes. pytest ran with `-p no:cacheprovider`.
- **Scratch.** `A/` is about 61 MB, listed in `CLEANUP.md`.
