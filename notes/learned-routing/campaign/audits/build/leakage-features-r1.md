# Audit: build checkpoint, lens "leakage-features", round 1

- **Auditor role:** independent static auditor (adversarial), with my own unit-level probe and
  small replay checks. I did not rely on the stage summaries or on r0's scripts; every number below
  comes from my own runs.
- **Date:** 2026-10-02.
- **Audited state:** WT `rupei/learned-routing` at `<commit-10>`; `lib/` clean. Installed bindings
  `_core.abi3.so` sha `1d67c131…`, build_id `6955b0ee…`.
- **Evidence directory:** `runs/audits/build-leakage-features-r1/`, containing:
  - `probe/`: a scratch Rust crate outside WT that links WT's `dynamo-custom-policy-builtin` and
    `dynamo-kv-router` by path;
  - `scripts/`, `out/`, `logs/`, `traces/` and `policies/`.
- **WT and main checkout:** read only. I edited nothing in WT; `cargo test` reused WT/target and
  compiled nothing.

## Verdict: PASS (0 blocker, 0 major, 4 minor)

Every check in the lens passes:

- The v1 features match CONTRACT.md, and the v2 features match FEATURES.md.
- No excluded input reaches any decision.
- The session map is bounded, exact, and isolated per replay.
- Sticky semantics match the contract text.
- θ0 parity is real at the current build.
- Malformed coefficients fail at startup.

Round 0's major finding (v2 `hash_home` keyed on replay-synthetic session IDs) is fixed. My probe
reproduces the prefix-only key, and a mutant keyed on the first block is detected. The four minor
findings are a documentation and guard gap plus three carry-overs. None affects any result.

## 1. Bindings and source are the audited code

- The `.so` mtime is 15:58:47. The last edit to a builtin source file was `features.rs` at
  15:56:06 and `tests.rs` at 15:56:16, so the binary is newer than the source.
- `git status` shows nothing under `lib/`. Commit `<commit-08>` holds that content.
- The `.so` sha `1d67c131…` matches `facts/build_fix_r0.json`. The lr-eval runs below report
  build_id `6955b0ee…`.
- `lib/kv-router`, `lib/mocker` and `lib/bindings` are unchanged since `<commit-07>`
  (`git diff --stat <commit-07> <commit-10> -- lib/` touches only the three builtin learned_choice
  files).

## 2. Information set (static)

**What the policies read.**

- **learned-choice** (`learned_choice/{mod,features,parameters}.rs`, `choice.rs`,
  `session_map.rs`) reads:
  - request: `prompt_tokens`, `request_blocks` (= ceil(isl/bs), `scheduling/types.rs:556`),
    `block_size`, `prefix_hashes` (v2 `hash_home` only), `session_context().session_id()`,
    `pinned_worker`, `worker_capacity()`;
  - per worker: `cost()` (the composed default scorer), `accounting_cache_estimate`,
    `active_prefill_tokens`, `decode_cost_blocks`, `active_requests`.
- **sticky-session** reads the same session and pin fields, plus `cost()` and `active_requests`.
- Neither policy calls `expected_output_tokens`, `modeled_prefill_backlog_ms`, `affinity_target`,
  `policy_class` or the priorities.
- In the builtin crate, no policy reads `expected_output_tokens` at all. Only llm-d
  optimized-baseline declares `PREFILL_TIME`.

**Host side, re-read independently.** In the replay request
(`lib/mocker/src/replay/offline/extensions/kv_router/mod.rs`):

- `build_pending_request` sets `expected_output_tokens = Some(max_output_tokens)` (:920), which
  is the true OSL.
- Its only other consumer is `RequestState`, where it is discarded (`sequences/single.rs:260`).
- `PlacementPolicy` (aisimulate-core `replay/core/mod.rs:96`) has no output-progress hook, so
  replay never calls `add_output_block`.
- `decode_cost_blocks` = `active_decode_blocks + additional_active_blocks`
  (`prompt_registry.rs:41,302-326`). It has no output term.
- Active prefill tokens decay only through a router `prefill_load_estimator`. That estimator
  needs a top-level `ais_perf_config` (`bindings/python/rust/llm/replay.rs:2076-2113`), which the
  harness never passes.

**Re-checked at this build** (`scripts/ais_knob.py`, `out/ais_knob.json`). A learned-choice spec
whose `router_config` sets `router_prefill_load_model: ais` fails with
`router_prefill_load_model='ais' requires ais_perf_config`. The job carries no top-level
`ais_perf_config`.

**Wall-clock expiry.** Replay builds its slot tracker with `new_without_expiry`
(`router_shared.rs:143-155`), so wall-clock request expiry cannot inject CPU-time-dependent load
changes.

## 3. Independent unit-level probe through the public API (`probe/tests/probe.rs`, `out/probe_*.json`)

**How it works.**

- I re-implemented feature sets v1 (CONTRACT table) and v2 (FEATURES.md table), the default logit
  at `KvRouterConfig::default()` weights, and both context forms from the specification text, not
  from `features.rs`.
- Policies are resolved from YAML through `default_registry()` + `register()` +
  `registry.resolve()`, the path the bindings use.
- Decisions are fed replay-shaped `SchedulingRequest`s: device overlap = effective overlap, and
  cached tokens = 16 × overlap.

**Randomized over:**

- N from 2 to 9, with non-sequential worker IDs;
- ISL from 1 to 120K, including partial blocks and prompts shorter than one block;
- per-worker overlap, active prefill, decode and new blocks, and active requests;
- capacity advertised, odd, or absent (testing the 65,536 fallback);
- missing load projections (1/12 of workers);
- `track_prefill_tokens` off (1/8 of trials);
- non-default host weights (half the trials), to check that feature 0 stays at default weights;
- sessions: none, new, remembered, and remembered-but-ineligible (set up by a pinned turn);
- eligibility filters, and `one_draw` vs `reservoir`.

| Check | Trials or decisions | Result |
|---|---|---|
| Dense random θ, v1 and v2, × no context / pooled {p,q} / named sources: the pick must attain max u of my model | 18,000 (2,965–2,990 decisive per group of 3,000) | **0 mismatches** (`probe_dense.json`) |
| θ = ±e_k for every v1 and v2 feature | 62 probes × 300 = 18,600 | **0 mismatches** (`probe_single.json`) |
| Power: the same dense or single trials against a deliberately mis-defined model (31 mutants) | 3,000 per mutant | 30 of 31 detected; see below (`probe_power.json`) |
| Excluded inputs perturbed on a twin instance with the same seed (below), for lc v1, lc v2, sticky-hard, sticky-bounded and the default | 5 × 3,200 decisions | **0 differ**; the positive control (a visible load change) moved 272–744 picks per policy (`probe_excluded_inputs.json`) |

**Excluded inputs perturbed:** `expected_output_tokens`, per-worker `modeled_prefill_backlog_ms`,
priorities, `policy_class`, a per-request config override (credit, prefill scale, temperature),
and, for v1, the prefix hashes.

**Power detail.**

- **Detected mutants:**
  - every v1 and v2 feature scaled ×1.5, or a binary flipped;
  - x0 computed at host weights (65 hits);
  - capacity fallback 18,863 (148);
  - `is_first_turn` treating a remembered-but-ineligible worker as first turn (197);
  - x2 ignoring the cache (173);
  - leave-one-out below-fraction (25);
  - `hash_home` keyed on the first block (78).
- **Undetected:**
  - x7 with ceil(P/16)·16 in place of P, a difference under 16 tokens (0 hits);
  - x1 over P/16 in place of ceil(P/16) (2 hits).

  Both are numerically negligible. For both I confirmed by reading the code that the contract form
  is the one implemented (`features.rs:244-271`).

## 4. Replay checks through the harness (`scripts/replay_checks.py` + `closed_own_osl.py`, `out/replay_checks.json`, `out/closed_own_osl.json`)

Every replay held one `CR/slots` slot, one at a time, using `Evaluator.plan` +
`worker.run_replay`.

### Isolation (E)

Setup: anti-affinity learned-choice (θ = −e₀ − 1e6·e₆), run in one process in the order X, Y, X′,
Y′:

- X = Mooncake closed loop, N=8, with 4,249 replay-synthetic IDs, all distinct;
- Y = synthetic sessions s0, open loop, N=4.

Results:

- X equals X′, and Y equals Y′, on the canonical per-request sha.
- Y kept **0 of 2,928** later turns on the previous worker.
- So no session map or RNG state survives from one replay to the next.
- A persisting map would have changed X′, where every `request_<n>` would be remembered.

### Future and OSL invariance (G)

This is my design, different from r0's. It uses the Mooncake w5 trace (5,008 rows), N=8,
learned-choice v1 with dense θ and a sources context, v1 pooled with τ = 0.5 (softmax), v2 dense,
and sticky-bounded 1.1.

| Mode | Perturbation | Decisions checked | Changed |
|---|---|---|---|
| Open loop ×0.957, T* at 40% of bursts | B1 OSL := 1 from T*; B2 OSL ×5 from T*; C rows after T* deleted | 2,049 at or before T* | **0, 0, 0** for all 4 policies |
| Open loop | D control: OSL ×5 from 25% | same | 554–618 |
| Closed loop, concurrency 32 | B1, B2, C from trace index 1500 | requests 1–1500 by synthetic ID | **0** for all 4 policies |
| Closed loop | D control from index 900 | same | 440–512 |
| Closed loop, own OSL | request_1501 (its own OSL 238→1 or 238→1190) and the 1,501 requests dispatched at or before it | 1,501 | **0**; same worker for request_1501 |

### KV visibility (LR-14) (`scripts/kv_visibility.py`, `out/kv_visibility.json`)

Setup: N=2. R1 has a 65,536-token prompt at t=0. R2 has the identical prompt at t=d. The policy
is θ = +e₁ (maximize overlap) with reservoir ties, seeds 1–8.

| d (ms) | R2 follows R1 |
|---|---|
| 1, 100, 300, 450 | 4/8 (chance level) |
| 700, 1,500, 4,000, 9,000 | 8/8 |

The router sees R1's blocks only after its first prefill pass completes, matching the "pass-completion
boundary" note in `replay/offline/extensions/kv_events/mod.rs:56-58`. There is no pre-computation (future) cache
visibility. Replay is still fresher than live by the network lag (LR-14, Stage 5).

### Feature identities in replay (`out/feature_identity.json`)

Ran through lr-eval: 48 replays, 0 errors. Cells: Mooncake w0 open N8, Mooncake w1 closed N8,
sessions s0 open N4, AgentX A3 lanes N8, each with 2 CRN replicates, all with reservoir ties.

| Identity | What it shows | Result |
|---|---|---|
| θ=+e₁ ≡ θ=−e₂ | x1 and x2 are the same overlap, with opposite sign | identical on **8/8** runs |
| θ=−e₃ ≡ θ=−e₇ | x7 is a positive per-request multiple of x3 | identical on **8/8** runs |
| θ=−(e₅+e₇) ≡ θ=−e₅ with `context {sources:[isl_k], p:[−e₃]}` | x7 = isl_k·x3, using the `isl_k` source | identical on **8/8** runs |
| Controls (F1 vs F3, F3 vs F5) | the comparison is sensitive | differ on 405–4,399 requests per run |

Identical means the same worker and the same e2e latency for every request, over 25,420 requests
per pair. This adds dynamic coverage of features 1 and 7, which r0 covered by hand values only.
Features 2–5 were tied to the default's cost in replay by r0's decomposition.

## 5. θ0 parity is real

- `cargo test -p dynamo-custom-policy-builtin`: **127 passed** (101+5+1+2+14+4), 2 ignored, both
  pre-existing (`logs/cargo_test_builtin.txt`). This includes:
  - `theta0_picks_the_default_argmin`;
  - `reservoir_tie_break_reproduces_the_seeded_default`;
  - `feature_zero_ignores_the_hosts_score_weights`;
  - `context_term_matches_the_bilinear_formula`;
  - `v2_features_follow_their_definitions`, with the prefix-only `hash_home`.
- The unit test's reference is the default scorer itself, which is the default cost function's
  own code; it also checks the registry default. It requires at least 1,500 tie-free tables.
- **I re-derived the fixer's post-rebuild parity from the stored per-request rows**, not from the
  summary sha (`scripts/rederive_parity.py`, `out/parity_rederived_postbuild.json`):
  - learned-choice@θ0-reservoir equals default@defaults on **22/22** (cell, k) runs at build
    `6955b0ee`, over 80,358 requests, same worker and e2e;
  - both YAMLs carry the same seed k+1;
  - with `one_draw`, θ0 matches on 0/22, with 359–4,414 worker differences per run, from tie
    relabeling. The comparison is therefore not vacuous.

## 6. Session map and sticky semantics

**Code** (`session_map.rs`):

- HashMap plus a BTreeMap recency index, with LRU eviction while `len > capacity`;
- `max_sessions: 0` is rejected;
- `bind` runs after every pick, including pins, and is keyed by the exact `session_id` string.

**Probe** (`probe_session_map.json`):

- 3,039 affinity checks under caps 1, 2 and 5, with keys `"a"`, `"A"`, `"a "` and `""`, matched an
  independent LRU model with 0 failures.
- Anti-affinity with keys `"k"`, `"K"` and `"k "` had 0 violations in 2,000 decisions.

**Campaign workloads never reach the bound.**

- The largest lowered-AgentX trace (A3 sidecar) has 5,380 sessions, and the full Mooncake trace
  has 23,608 rows. Both are below 65,536.
- Lowered AgentX session IDs are copy-prefixed (`lrx:<copy>:weka:…`). 0 of 50,141 session IDs
  are shared across copies or plays, so affinity cannot cross copies.

**Sticky semantics** (`probe_sticky.json`). An independent model of the CONTRACT text was checked
on 20,000 decisions across 5 configurations: hard and bounded with lf 1.0, 1.25 and 2.0, caps 3,
4 and 65,536, host weights credit 1.7, scale 0.6 and request weight 2.5, eligibility filters, and
pins. **0 mismatches.** Every path was exercised:

| Path | Decisions per configuration |
|---|---|
| stay | 1,099–3,253 |
| leave or ineligible, then rebind | 156–1,947 |
| first turn or no session | 367–2,553 |
| pin | 184–208 |

Bounded mode stays iff active_requests ≤ lf × the candidate mean, as contracted.

**Closed-loop synthetic IDs** (`request_<n>`) reach both policies. They are single-use, so v1 sees
`session_affinity` = 0 and `is_first_turn` = 1 exactly as on sessionless traffic. Since the round-0
fix, `hash_home` no longer reads them.

## 7. Parameter validation (`probe_validation.json`)

All of the following fail at **resolution** through the real registry path, each with a message
naming the problem:

- **52 malformed learned-choice specs:**
  - wrong theta, p or q lengths for v1 and v2;
  - NaN or ±inf anywhere in theta, p or q;
  - non-numeric elements;
  - a non-list theta;
  - a missing or unknown `feature_set`;
  - unknown or miscased keys;
  - a negative, tiny-negative, NaN, inf or string temperature;
  - a negative, fractional or overflowing seed;
  - a bad `tie_break`;
  - `max_sessions` of 0, −1 or 1.5;
  - p without q or sources, and q without p;
  - a rank mismatch;
  - unknown, repeated or mis-counted sources;
  - both q and sources;
  - an unknown context key or a context given as a list;
  - a duplicate YAML key.
- **15 malformed sticky-session specs:** a missing, unknown, miscased or null mode; `load_factor`
  with hard; `load_factor` < 1, NaN, inf or a string; `max_sessions` 0; a negative seed; a bad
  `tie_break`; an unknown key; a duplicate key.

All 11 + 4 valid edge cases resolve, including `-0.0` temperature, empty contexts and
`seed: 2^64−1`. A finite but overflowing θ (1e308) fails loudly at the first decision with
`learned-choice utility is not finite`; it never routes on garbage.

## Findings

### F1 (minor): host structural knobs silently change learned-choice feature meanings, and the build notes say otherwise

- **Claim in the build notes.** `facts/build_rust.json` (next_stage_notes) says the
  `router_config` sidecar knobs "do not change learned-choice's feature 0". FEATURES.md says the
  same for "host score-weight knobs".
- **What is true.** That holds for weights. It does not hold for structural knobs that the harness
  accepts for any policy (`policy.py:243-249` allows every `KvRouterConfig` knob):
  - `router_track_prefill_tokens=false` switches feature 0 to the decode-subtraction formula
    (`default/scorer.rs:148-150,212-219`). The probe exercised this branch.
  - The same setting makes features 3 and 7 identically 0, because the slot tracker stops
    booking prefill.
  - `router_assume_kv_reuse=false` makes the tracking hashes random. That changes
    `decode_cost_blocks` (feature 4) and the v2 `hash_home` key.
- **Impact.** No current spec or space does this, so there is no impact today. A training-stage
  ablation that tuned these knobs jointly with θ would, however, re-define the features under it.
- **Fix, either of:**
  - the harness rejects `router_config` keys for learned-choice outside an allowlist (for
    example `router_queue_threshold` and `router_queue_policy`);
  - FEATURES.md lists `router_track_prefill_tokens`, `router_assume_kv_reuse` and
    `router_track_active_blocks` as feature-defining host settings that must stay at defaults.

### F2 (minor, carry-over of r0 F2): the pooled example space still frees `q[6]`

- `spaces/learned_choice_m2_pooled.yaml` still frees every `q` element.
- The candidate mean of `session_affinity` is 1/N, which breaks the N-extrapolation (LR-06).
- **Fix:** pin flat index rank·8+6, i.e. `fixed: {6: 0.0}` at rank 1; or use the named-sources
  form.

### F3 (minor, carry-over of r0 F3): CONTRACT's feature-1 rationale is stale

- CONTRACT.md:127 says replay "leaves device-tier counts at 0".
- Replay sets device = overlap (`kv_router/mod.rs:296-305`, comment at :298). FEATURES.md is correct.
- CONTRACT is an orchestrator file, so I did not edit it.

### F4 (minor, porting and information parity): live session affinity overrides the learned θ₆

- Both policies set `with_exclusive_affinity(true)` (`learned_choice/mod.rs:183`,
  `sticky_session.rs:210`).
- In a live router with `SessionAffinityMode` enabled, the host therefore pins every later turn to
  `affinity_target`. Pinned decisions compute no features.
- So whatever session behavior the model learned in replay is replaced by hard stickiness live.
  Replay never sets `affinity_target`.
- **Fix:** add a note to the REPORT porting notes. Either ship learned-choice with host session
  affinity off, or treat sticky as the live default. No campaign result is affected.

## Information-parity notes (LR-14), not findings

- KV-event visibility in replay is at pass completion: no future visibility, though fresher than
  live by the network lag (§4).
- Active prefill is released at simulated prefill completion; live releases it at the first
  token.
- Request expiry is disabled in replay; live expires requests after 300 s of wall time.
- I did not measure KV-event lag effects. That remains Stage 5's perturbation item.

## Lessons (LESSONS.md)

- **Applied:**
  - LR-14: information-parity lens, with KV visibility timing measured, the AIS knob re-checked,
    and expiry checked;
  - LR-03: `one_draw` and reservoir through parity and twin-instance invariance;
  - LR-05 and LR-06: the pooled `q[6]` gauge and one-hot issue (F2), and is_first_turn
    semantics checked by mutant 103;
  - LR-07: v2 features, including the prefix-only `hash_home`, verified by the probe;
  - LR-13: the `hash_home` concentration remains a training-stage audit item.
- **Rejected or out of lens:** LR-01, 02, 04 and 08–12, which concern goodput, noise, baselines
  and splits; LR-04 is also superseded by A2.5. LR-15 is a training-stage item.

## Process

- **Replays:** every replay held a `CR/slots` slot. I ran:
  - 112 direct in-process replays, one at a time: E 4, G 40, own-OSL 3, KV visibility 64
    (tiny, 2 requests), AIS knob 1 (failed at startup);
  - 48 lr-eval replays, with `--slots 12`, cached under `lfr1-*` policy names.
- **Probe crate:** built with a separate target dir and `--offline` (25 s), so WT/target was not
  touched.
- **Housekeeping:** nothing deleted, nothing pushed, WT unchanged.
- **Scratch:** listed in CLEANUP.md: `probe/target` 491M and `logs/` 81M (INFO replay log).
