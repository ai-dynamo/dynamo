# Audit: build checkpoint, lens "leakage-features", round 0

- **Auditor role:** independent static auditor (adversarial), with small dynamic checks.
- **Date:** 2026-10-02.
- **Audited state:** WT `rupei/learned-routing` at `<commit-07>` (clean), bindings build_id
  `4b4525a9…` (`.so` sha `ecdd2202…`).
- **Evidence directory:** `runs/audits/build-leakage-features-r0/` (`scripts/`, `out/`, `traces/`,
  `policies/`).
- **Main checkout:** read only. **WT:** not modified (`git status` clean).

## Verdict: FAIL (0 blocker, 1 major, 2 minor)

The core of the lens passes with strong evidence:

- The contract v1 features match their definitions.
- No v1 feature reads the output length, AIS signals, future rows or engine-only state.
- The session map is bounded, keyed correctly, and causal.
- θ0 parity is real.
- Malformed coefficients fail loudly at startup.

The single major finding is confined to the **optional v2** feature `hash_home`, which reads a
replay-only synthetic session ID on closed-loop Mooncake cells (simulator-only state). It does not
affect v1, any baseline, or any result produced so far, but it would mislead any v2 model that
frees θ₁₁. It should be fixed, or θ₁₁ pinned to 0, before v2 training.

## What I checked and how

### 1. Information set of the policies (static)

I listed every plugin-API accessor called by `learned_choice/{mod,features,parameters}.rs`,
`choice.rs`, `session_map.rs`, `sticky_session.rs` and the composed default scorer
(`default/scorer.rs`):

- **Request:** `prompt_tokens`, `request_blocks`, `block_size`, `prefix_hashes` (v2 only),
  `session_context().session_id()`, `pinned_worker`, `tracks_prefill_tokens`.
- **Worker:** `worker()`, `cost()` (the composed default scorer only),
  `cache.accounting_cache_estimate`, `load.{active_prefill_tokens, decode_cost_blocks,
  active_requests, is_available}`, `worker_capacity().total_kv_blocks`.
- **Scorer only:** `has_tier_matches`, `device/host/disk_overlap_blocks`, `shared_hits` and
  `preferred_taint_multiplier`.

**Never called:**

- `expected_output_tokens`;
- `modeled_prefill_backlog_ms`, and `PREFILL_TIME` is never declared;
- `affinity_target`, `policy_class`, priorities.

The picker declares `CACHE | LOAD`. The scorer declares `LOAD | PREFERRED_TAINT | CACHE`.

**Host side.** I traced replay request construction
(`lib/mocker/src/replay/offline/extensions/kv_router/mod.rs`):

- `build_pending_request` (:857) does set `expected_output_tokens = Some(true OSL)` (:920).
- The only host consumer of that value is `RequestState.expected_output_tokens`, which is stored
  and then discarded (`sequences/single.rs:260`, `let _ = …`).
- Replay never calls `add_output_block`, so there is no output-block decay (no match in
  `lib/mocker`).
- `decode_cost_blocks` = `potential_decode_blocks()` = active prompt blocks plus this request's new
  blocks (`prompt_registry.rs:302-326`). It has no output term.

**AIS.**

- `active_prefill_tokens` decays only when a router `prefill_load_estimator` exists
  (`prefill_tracker.rs:79-85`).
- That estimator needs both `router_prefill_load_model='ais'` and a top-level `ais_perf_config`
  (`bindings/python/rust/llm/replay.rs:2072-2112`).
- The harness never passes a top-level `ais_perf_config`. Its `common` kwargs in `worker.py` have
  none, and the `REPLAY_OPTIONS` whitelist excludes it.
- Dynamic check: a `router_config` with `router_prefill_load_model="ais"` fails with
  `router_prefill_load_model='ais' requires ais_perf_config` (`out/ais_router_knob.json`).
- So the estimator is absent, `projects_prefill_time` is false (mod.rs:565), and no AIS-derived
  value reaches any feature. AIS times only the engine, which reaches the router legitimately
  through completions and KV events.

### 2. Dynamic future-information and OSL test (`scripts/future_invariance.py`, `out/future_invariance.json`)

**Setup.**

- Trace: the first 2,000 Mooncake rows, N=4, open loop at ×1.
- T* = the 57th arrival burst; the cutoff is 171,121 ms of trace time, with 965 decisions at or
  before T*, 19 of them in the T* burst.
- Variants:
  - **B:** OSL ×4 for every row at or after T*, so the current and future output lengths differ;
  - **C:** rows after T* deleted;
  - **D (positive control):** OSL ×4 from 26 bursts earlier.
- Policies:
  - learned-choice v1 with all features nonzero plus a named-source context;
  - v1 pooled context with τ = 0.3;
  - v2 with all 23 features nonzero;
  - seeded default;
  - sticky-bounded.

| Policy | B: decisions ≤ T* changed | C: decisions ≤ T* changed | D control: decisions changed (of which in the T* burst) |
|---|---|---|---|
| lc_v1_rich_ctx | 0 / 965 | 0 / 965 | 210 (19) |
| lc_v1_pooled_softmax | 0 / 965 | 0 / 965 | 214 (19) |
| lc_v2_rich | 0 / 965 | 0 / 965 | 282 (13) |
| default_seed1 | 0 / 965 | 0 / 965 | 286 (9) |
| sticky_bounded | 0 / 965 | 0 / 965 | 286 (9) |

The comparator is sensitive: D changes the state seen at T*. Neither the requests' own or future
output lengths nor future rows influence any decision.

(My first attempt ran at ×3 load. There, D also showed no change: the cell was so overloaded,
mean e2e of about 157 s, that no perturbed request completed before the last arrival. I reran at
×1. Both runs used the same script; only the final version is kept.)

**SLA inputs are evaluation-only** (`out/sla_invariance.json`). On the same 2,000-row trace,
passing `sla_itl_ms=1`, `sla_ttft_ms=1` or `sla_e2e_ms=1` leaves all 2,000 decisions unchanged,
for both v1 and v2. A repeated identical run is also identical. Per-request `uuid`s are fresh per
run, and no policy reads them.

### 3. v1 features computed as CONTRACT.md defines them

**Code.** `features.rs:244-271` matches the contract table:

- **#0:** `candidate.cost() / max(request_blocks, 1)`, with the default scorer composed at
  `KvRouterConfig::default()` weights (mod.rs:176).
- **#1:** effective overlap / B.
- **#2:** (prompt − cached, saturating) / 8192.
- **#3:** active prefill / 8192.
- **#4:** decode_cost_blocks / `worker_capacity().total_kv_blocks`, falling back to 65,536 (the
  fallback is recorded in `facts/build_rust.json`).
- **#5:** active requests / 32.
- **#6:** previous-worker indicator.
- **#7:** isl_k × #3.

The Python `KvRouterConfig` defaults equal the Rust `Default` for every score weight (credit 1,
decay 0, prefill scale 1, decode request weight 0, host 0.75, disk 0.25), so host-side defaults
cannot drift feature 0 from `default@defaults`. I re-derived the hand values in
`v1_features_follow_their_definitions` (for example worker 0: (32+96+64)/16 − 4 + 100 = 108,
/10 = 10.8).

**In-replay cross-check of features 2–5 against the independent default scorer**
(`scripts/decomposed*`, `out/decomposed_check.json`, `out/decomposed16k_check.json`).

In replay, cached tokens = 16 × overlap. With that, the default cost at weights (s=1, c=1, w)
equals 512·x2 + 512·x3 + C·x4 + 32w·x5 plus a request constant, where C is the capacity. So
learned-choice with θ = −(0, 0, 512, 512, C, 32w, 0, 0), θ₀ = 0 and `tie_break: reservoir` must
reproduce `dynamo-default-cost-fn` (with `decode_active_request_weight = w`) request for request.
Results over 11 cells × 2 replicates:

| Capacity | w = 0, identical runs | w = 1, identical runs | Control: default w=1 vs w=0 identical |
|---|---|---|---|
| 18,863 (engine) | 20/22 | 16/22 | 4/22 |
| 16,384 (power of two, `engine_overrides`) | **22/22** | **22/22** | 4/22 |

The divergences at 18,863 come from rounding: x4 = d/18863 × 18863 is inexact and breaks exact
cost ties differently. The divergences start late, at requests 580–4049. With an exact capacity
every term is a multiple of 1/16, and all of them vanish.

This proves three things:

- Features 2–5 are computed in replay exactly as specified, from the same router state the
  default uses.
- Replay advertises `num_gpu_blocks` as the capacity, so the 65,536 fallback is unused. If the
  fallback had been used, the 16,384 identity would fail.
- Feature 4's denominator is the engine's capacity.

Features 1 and 7 are covered by the hand-computed unit test only.

### 4. θ0 parity evidence is real

- I re-ran `cargo test -p dynamo-custom-policy-builtin`: 127 passed (101 + 5 + 1 + 2 + 14 + 4)
  with 2 ignored, including:
  - `theta0_picks_the_default_argmin`;
  - `reservoir_tie_break_reproduces_the_seeded_default`;
  - `feature_zero_ignores_the_hosts_score_weights`;
  - `context_term_matches_the_bilinear_formula`;
  - `duplicating_the_worker_set_keeps_features_sources_and_choice`;
  - `parameter_validation_is_strict`.
- The parity test is not vacuous:
  - it compares against the argmin of independently recorded default costs;
  - it also asserts that the registry default picks that row;
  - it requires at least 1,500 tie-free tables out of 2,000, across 4 host input shapes, for v1
    and v2.
- I re-derived the integration replay parity from the stored per-request rows, not the summary:
  learned-choice@θ0 with `reservoir` equals `default@defaults` on **88/88** (cell, replicate) runs,
  with identical worker and e2e per request.
- With `one_draw`, the same cells differ on 190–443 assignments per run, from tie relabeling. The
  comparison can therefore detect a difference.

### 5. Session-affinity map: bounded, keyed correctly, causal

**Code** (`session_map.rs`):

- `HashMap<String, (worker, stamp)>` plus a `BTreeMap` recency index, with LRU eviction while
  `len > capacity`.
- `max_sessions: 0` is rejected.
- The map is bound after every pick, including pins (mod.rs:163-165, sticky_session.rs:193-195).

**Dynamic** (`scripts/session_check.py`, `out/session_check.json`, via `lr-eval` and the slot
pool): synthetic sessions s0 (N4 open, N8 closed) and AgentX A2/A3 lanes.

| Policy | Later turns kept on the previous worker |
|---|---|
| θ = −e₀ + 1e6·e₆ | 2928/2928, 2928/2928, 453/453, 376/376 |
| θ = −e₀ − 1e6·e₆ (anti-affinity) | **0** in all four cells, so feature 6 marks exactly the previous worker |
| anti-affinity with `max_sessions: 2` | 1224, 1196, 94 and 123 stay, because evicted sessions are forgotten |
| sticky-bounded 1.25 | 433 and 547 moves on sessions, 8 and 2 on AgentX |

- No later turn arrives before its previous turn completes (0 overlaps), so the "previous
  request" is causal.
- Session IDs are namespaced per play or stream (for example
  `weka:655deb65e4fb1832:session:root`, `syn0-s000444`), so there are no cross-play key
  collisions.
- Each policy instance owns its map, and it persists across a replay. If it were rebuilt per
  request, anti-affinity could not reach 0.

**Sticky semantics match the contract:**

- no session, or a new session: the default cost function with the host weights;
- `hard`: the bound worker if it is a candidate, otherwise the default, then rebind;
- `bounded`: stay iff `active_requests ≤ load_factor × candidate mean`;
- `load_factor` defaults to 1.25, and passing it with `hard` is rejected.

### 6. Parameter validation through the real bindings path (`scripts/validation_e2e.py`, `out/validation_e2e.json`)

**Rejected at policy resolution, each with a message naming the key:**

- `theta` with 7 or 9 entries;
- `.inf` or `.nan` in theta;
- a string in theta;
- an unknown key;
- `p` without `q`;
- `q` of the wrong length;
- a negative temperature;
- v2 with an 8-entry theta;
- sticky `hard` with `load_factor`;
- sticky `load_factor` 0.5;
- sticky without `mode`.

The valid control ran 5/5 requests. A finite theta that overflows (1e308) fails at the first
decision with `utility is not finite`. That is loud, though not at startup. The harness also
refuses a spec without `worker_selection.aggregated`, which closes the setup F2 path where the
unseeded default runs silently.

## Findings

### F1 (major, v2 only): `hash_home` reads a replay-only synthetic session ID on closed-loop Mooncake

**Symptom.**

- `features.rs:349-358` keys `hash_home` on FNV-1a(session_id) whenever `session_context()`
  exists, and on the first 256 prompt tokens otherwise.
- Closed-loop Mooncake replay attaches a **synthetic per-request session ID** (`request_<n>`).
  On `int-mooncake-w0-base-n8-closed-L2` all 4,249 rows carry one, all distinct, with turn_index 0
  (`out/hash_home_keys.json`). A live router would not receive these: a closed-loop client sends
  no session header.
- Open-loop cells of the same trace carry none.

**Measured effect** (`out/hash_home_check.json`). With θ₁₁ = 1 and everything else 0, on the same
derived trace:

- **Open loop:** each first-block prefix key maps to exactly **2** workers.
- **Closed loop:** each prefix key spreads over all **8** workers.

The feature is a prefix-affinity home in open loop and a per-request random two-choice filter in
closed loop.

Separately, Mooncake has only 4 distinct first-block keys (10,938 / 9,203 / 3,449 / 18 of 23,608
rows). So the prefix-keyed home concentrates about 85% of traffic onto the homes of 2 keys
(LR-13 concentration risk). This is a design property, recorded for the training stage.

**Impact.**

- v1, sticky-session and every baseline are unaffected:
  - in v1, single-use IDs leave `session_affinity` = 0 and `is_first_turn` = 1, the same as
    sessionless traffic;
  - sticky-hard and sticky-bounded equal θ0 `one_draw` row for row on the closed-loop cell, 3/3
    replicates;
  - only `thunderagent`'s classifier, which is skipped, also reads session IDs.
- No existing result is affected.
- But LR-07 recommends v2 and `hash_home` starts. Any v2 model that frees θ₁₁ would be scored on
  closed-loop Mooncake cells with a mechanism that does not exist live, and results would differ
  by load mode on the same trace.

**Fix options**, for the fixer to choose and record:

- (a) Pin θ₁₁ = 0, and keep `hash_home` out of pooled `q` and sources, in every v2 space. Add a
  harness guard or space note. Zero code.
- (b) Key `hash_home` by prefix when `prefix_hashes()` is present, and fall back to the session ID
  only without one. Update FEATURES.md; v2 is not frozen.
- (c) Stop replay from synthesizing single-turn session IDs for closed-loop flat traces. This
  changes `session_context` for all policies and should be recorded in UPSTREAM_FOLLOWUPS.

### F2 (minor): the example pooled space leaves the one-hot pooled entry `q[6]` free (LR-06)

- `spaces/learned_choice_m2_pooled.yaml` frees every `q` element.
- The mean over candidates of `session_affinity` is 1/N when the previous worker is a candidate
  and 0 otherwise. A free `q[6]` therefore makes the context coefficient scale with 1/N, which
  breaks the N=2/16/32 extrapolation.
- FEATURES.md ("Pooling one-hot features") says to keep `q[6]` at 0.
- The `lowrank` kind supports pinning: flat index rank·dim + 6, which is `fixed: {14: 0.0}` at
  rank 1.
- The file is labelled a placeholder, and training owns the final space. This is a note, not a
  leak.

### F3 (minor, documentation): CONTRACT's feature-1 rationale is stale

- CONTRACT says replay "leaves device-tier counts at 0".
- The current replay sets `tier_overlap_blocks.device` equal to the indexer overlap
  (kv_router/mod.rs:296-305).
- So device = accounting estimate in replay. FEATURES.md states this correctly.
- No behavioral effect. I did not edit CONTRACT (an orchestrator file).

## Information-parity notes (LR-14), not findings

- KV events are applied to the router index synchronously in replay
  (`SyncReplayIndexer::apply_event`), while live events lag. Features 0–2 are therefore fresher in
  replay. This is the Stage 5 staleness-perturbation item.
- Active prefill tokens are released on the simulated prefill-completion event
  (`on_prefill_completed`, mod.rs:703-710). Live releases them on the first output
  (`lib/llm/src/kv_router/prefill_router/admission.rs:61`). The event semantics match; only the
  network lag differs. Neither path decays tokens over time without an AIS estimator.
- Active requests and decode blocks are released on completion (`free`) in both settings.
- [hypothesis] The mocker may emit KV "stored" events at a different point in the step than vLLM.
  Not checked; it applies equally to every cache-aware policy.

## Lessons (LESSONS.md)

- **Applied:**
  - LR-14: information-parity lens (sections 1, 2 and the notes);
  - LR-06: N-stability of pooled one-hot features (F2) and the duplication test re-run;
  - LR-07: `hash_home` and its key (F1);
  - LR-03: `one_draw` and `reservoir` draw discipline checked through parity;
  - LR-05: θ₀ pinned at −1 in both example spaces, and the pooled-form gauge noted;
  - LR-13: `hash_home` concentration on 4 Mooncake keys flagged.
- **Rejected or out of lens:** LR-01, 02, 04, 08–12 and 15. LR-04 is also superseded by A2.5.

## Process

- Every replay held a `CR/slots` slot, either through `lr-eval` or `learned_routing.slots.SlotPool`
  in my scripts, in one process at a time:
  - 80 direct short replays, 0.3–2 s each: future-invariance 44 (including the first ×3 attempt
    and a 2-replay debug run), validation 15, AIS knob 1, SLA invariance 18 (two runs);
  - 168 fresh `lr-eval` replays (sessions 12, decomposition 66 + 88, `hash_home` 2; another 26
    were cache hits), written to the shared cache under `audit-*` policy names and
    `audit-cap16k-*` cell IDs.
- `cargo test` reused WT/target and needed no rebuild.
- Nothing was deleted and nothing was pushed. WT and the main checkout are unchanged.
- New scratch: `runs/audits/build-leakage-features-r0` (2.5 MB). The cache grew to 181 MB in
  total, shared by all stages. Both are listed in CLEANUP.md.
