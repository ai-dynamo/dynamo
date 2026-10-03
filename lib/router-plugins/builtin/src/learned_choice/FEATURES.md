<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `learned-choice` features and model

This file specifies the `learned-choice` worker-selection policy: its parameters, utility,
decision rule and every feature. `features.rs` implements it; keep the two in step. Feature set
`v1` is frozen once calibration starts. New features go into a new set.

## Parameters

```yaml
worker_selection:
  aggregated: candidate
  instances:
    - name: candidate
      type: learned-choice
      parameters:
        feature_set: v1          # required: v1 (8 features) or v2 (23 features)
        theta: [..d floats..]    # required: one coefficient per feature
        context:                 # optional; absent, {} or empty p = no context term (plain MNL)
          p: [[..d..], ...]      # rank r = len(p)
          q: [[..d..], ...]      # pooled form: len(q) = r
          # or, instead of q:
          # sources: [isl_k, mean_kv_load_frac, ...]   # named form: len(sources) = r
        temperature: 0.0         # optional, default 0; finite and >= 0
        seed: 1                  # optional u64; absent = fresh entropy per policy instance
        tie_break: one_draw      # optional: one_draw (default) or reservoir
        max_sessions: 65536      # optional, >= 1: bound of the session map
```

Unknown keys at any level, a wrong vector length, a non-finite coefficient, a negative or
non-finite temperature, both `q` and `sources`, `p` without either, an unknown or repeated
source, and `max_sessions: 0` all fail startup with a message naming the problem.

## Utility and decision

For candidate `i` in the eligible set `S`, with features `x_i` (in the fixed order below):

```text
u_i = θ·x_i + Σ_k c_k (p_k·x_i)
c_k = q_k·x̄_S          pooled form (x̄_S = candidate mean of x)
c_k = z_{S, sources[k]} named form (see "Context sources")
```

The implementation folds the context into effective weights `w = θ + Σ_k c_k p_k` and scores
`u_i = w·x_i`, so a decision costs O(N·d + r·d).

- `temperature: 0`: the highest utility wins. Ties are broken with the seeded RNG.
- `temperature > 0`: sample from `softmax(u / temperature)`. This temperature is **not**
  range-normalized, unlike `dynamo-default-cost-fn`'s `router_temperature` (which divides costs by
  their range first), so the two are different parameters (LR-15). At a fixed temperature the
  probability mass on non-best workers grows with N.
- A host pin always wins, and only eligible workers are candidates. Like the default, the policy
  asks the host to treat an eligible session-affinity target as exclusive.
- A non-finite utility fails the selection rather than routing on garbage.

### Determinism

- Rows are visited in canonical worker order (worker ID, then DP rank), never the host's row
  order, and set aggregates (means) are summed in that order. A seeded policy therefore decides
  identically across processes.
- Each policy instance (one per routing partition and worker role) owns one RNG, seeded from
  `seed`. Without `seed` it draws fresh entropy, so production router replicas do not break ties
  in lockstep.
- `tie_break: one_draw` (default, LR-03) consumes exactly one 64-bit draw per decision, whatever
  the path: a tie, a sample, a unique best, or a pin. Two policies with the same seed share every
  later draw even after one early decision differs, which keeps common random numbers aligned
  across CMA-ES candidates. Ties map the draw to the tied rows by multiply-shift; samples map it to
  [0, 1) with 53 bits.
- `tie_break: reservoir` reproduces the seeded `dynamo-default-cost-fn` picker draw for draw: no
  draw on a pin, one `usize` draw per tied candidate during the scan, one `f64` draw per sample. It
  exists for replay parity checks.

### Parity anchor

`θ = −e₀` (−1 on feature 0, 0 elsewhere), no context, `temperature: 0` takes the default cost
function's argmin on any table without a tied minimum (unit test `theta0_picks_the_default_argmin`,
2,000 random tables over the four host input shapes). With `tie_break: reservoir` and the same
`seed`, it also breaks ties exactly as the seeded `dynamo-default-cost-fn` does
(`reservoir_tie_break_reproduces_the_seeded_default`), so a replay of both policies should match
request for request. The rare exception is two costs that differ by less than the rounding of the
division by `request_blocks`.

## Feature set v1

`T = 8192` tokens (the engine's default chunked-prefill batch). `B = max(request_blocks, 1)`,
where `request_blocks = ceil(prompt_tokens / block_size)` comes from the host. Cache values come
from the host's accounting estimate (`accounting_cache_estimate()`: effective overlap blocks and
estimated cached tokens), because replay's primary index leaves lower-tier counts at zero. In
replay the effective overlap equals the device overlap and cached tokens are whole blocks.

| # | Name | Definition |
|---|---|---|
| 0 | `default_logit_scaled` | the default cost function's cost for this row at **default weights** (`KvRouterConfig::default()`: overlap credit 1, decay 0, prefill load scale 1, decode request weight 0, host 0.75, disk 0.25, no shared cache), divided by `B`. It is computed by the default scorer itself, composed into this policy, so the CACHE, LOAD and PREFERRED_TAINT inputs (tier matches vs estimate, unavailable load, untracked prefill, decode subtraction, taint multiplier) are handled exactly as the default handles them. Host score-weight knobs do not change it; the role structure (decode pools, conditional disaggregation) comes from the host config. |
| 1 | `overlap_frac` | effective overlap blocks / `B`. Not clamped; it stays ≤ 1 whenever the estimate does not exceed the prompt. |
| 2 | `new_prefill_tokens_k` | (prompt_tokens − estimated cached tokens, saturating at 0) / T |
| 3 | `active_prefill_tokens_k` | the worker's active prefill tokens / T (0 when the host has no load projection) |
| 4 | `kv_load_frac` | decode cost blocks (active decode blocks plus this request's new blocks) / the worker's `total_kv_blocks` from `worker_capacity()`, or / 65,536 when the worker advertises none. Replay advertises the engine's `num_gpu_blocks` (18,863 in the campaign engine), so the fallback is not used there. |
| 5 | `active_requests_s` | active requests / 32 |
| 6 | `session_affinity` | 1 if this worker served the previous request of this request's session, else 0. The policy keeps a bounded LRU map (`max_sessions`) from `session_context().session_id()` to the last worker it chose for the session, including pinned choices. 0 for every worker when the request has no session or the session is new or evicted. |
| 7 | `isl_x_prefill_load` | (prompt_tokens / T) × feature 3 |

## Feature set v2

v2 keeps v1's indices 0–7 unchanged and appends:

| # | Name | Definition | Source |
|---|---|---|---|
| 8 | `log_ptok` | ln(max(active_prefill_tokens + new_prefill_tokens, 1) / 1024) | LR-07 (LMetric's P-token with queued prefill) |
| 9 | `log_new_prefill` | ln(max(new_prefill_tokens, 1) / 1024) | the branch's `lmetric` port factor |
| 10 | `log_bs` | ln(1 + active_requests) | LR-07 (LMetric's batch size) |
| 11 | `hash_home` | 1 if the worker is among the two highest rendezvous weights (seed 0, `signals::rendezvous`) for the request's key, else 0; 1 for every worker when there are ≤ 2 candidates; 0 for every worker without a key. The key is the prefix hash of the prompt's first 256 tokens in whole blocks (at least one block), exactly as `chwbl` keys it; a request without prefix hashes has no key. The session ID is never the key (see below). | LR-07 (Lodestar, DualMap) |
| 12–14 | `kv_load_ratio`, `active_prefill_ratio`, `active_requests_ratio` | x / (ε + mean_S x) for features 4, 3, 5, with ε = 0.01, 512/T and 1/32 (about one unit of each load) | LR-06 |
| 15–17 | `kv_load_below_frac`, `active_prefill_below_frac`, `active_requests_below_frac` | #{j ∈ S : x_j < x_i} / N for features 4, 3, 5 (strictly below; ties do not count) | LR-06 (duplication-invariant, unlike the leave-one-out rank) |
| 18–20 | `kv_load_excess`, `active_prefill_excess`, `active_requests_excess` | max(x − mean_S x, 0) for features 4, 3, 5 | LR-06 |
| 21 | `new_prefill_x_requests` | feature 2 × feature 5 | lesson plan deltas (Jain24, Preble) |
| 22 | `prefill_attn` | n (L − n/2) / T², with n = new prefill tokens and L = prompt tokens: the attention work of prefilling the last n of L tokens | lesson plan deltas (SMetric) |

Representability:

- `θ = −(e_log_ptok + e_log_bs)` takes the argmin of (active_prefill + new_prefill) × (active_requests + 1),
  LMetric's product with queued prefill (ln is monotone and ln(a·b) = ln a + ln b).
- `θ = −(e_log_new_prefill + e_log_bs)` takes the argmin of the branch's `lmetric` port score
  without its hot-spot filter, when cached tokens equal device overlap blocks × block size, as in
  replay.

`hash_home` keys on the prompt prefix only. An earlier draft keyed it on the session ID whenever
the request had one. Offline replay, however, gives every request of a closed-loop flat trace
(Mooncake, FAST25) a synthetic single-use session ID (`request_<line>`, aisimulate-core
`loadgen/trace.rs`), while the same trace in open loop carries none and a live closed-loop client
sends none. The session key therefore turned the feature into a prefix home in open loop and a
per-request random pair in closed loop on the same trace (build audit, leakage-features F1).
Session continuity is `session_affinity`'s job. On Mooncake the key takes only 4 distinct values,
with about 85% of rows on two of them, so a large positive `θ₁₁` concentrates load (LR-13); the
training stage audits worker shares.

Not implemented from the lesson's v2 list: `affined_ctx_frac` (needs a clock the plugin API does
not expose; wall time would break replay determinism), a session-history EWMA of output length
(the policy never observes completions), z-scores and x − min_S (LR-06: no-ops or broken at N = 2).

## Context sources

The named form conditions coefficients on request-level, N-stable quantities (LR-05, LR-06). It
is Tomlinson & Benson's LCL with one free column `p_k` per source; it is linear in `p`, so it has
no gauge freedom.

| Name | Value |
|---|---|
| `mean_kv_load_frac` | mean over candidates of feature 4 |
| `mean_active_prefill_tokens_k` | mean over candidates of feature 3 |
| `mean_active_requests_s` | mean over candidates of feature 5 |
| `max_overlap_frac` | max over candidates of feature 1 |
| `isl_k` | prompt_tokens / T |
| `is_first_turn` | 1 when feature 6 is 0 for every worker because the session is new, evicted, or absent; 0 when the policy remembers the session's last worker (even if that worker is not a candidate) |

Every source is unchanged when the candidate set is duplicated (N → 2N identical copies), as are
all features except `hash_home` (unit test
`duplicating_the_worker_set_keeps_features_sources_and_choice`).

## Guidance for search spaces

These follow LESSONS.md and are advice to the training stage, not enforced by the policy.

- **Scale (LR-05).** At `temperature: 0`, scaling `θ` and `p` together never changes a decision.
  Pin one coefficient whose sign is known, for example `θ₀ = −1` near the default.
- **Pooled-form gauge (LR-05).** `p_k → c·p_k`, `q_k → q_k/c` leaves the pooled form unchanged.
  Prefer the named form, or fix `q` to unit vectors.
- **Pooling one-hot features (LR-06).** The candidate mean of `session_affinity` is 1/N or 0, and
  of `hash_home` is 2/N: in the pooled form keep `q[6]` and (v2) `q[11]` at 0.
- **ISL (LR-06).** v1 feature 7 equals the (`isl_k`, `active_prefill_tokens_k`) entry of a named
  context term. When `isl_k` is a source, freeze `θ₇ = 0`.
- **Constant sources (LR-05).** On traffic without sessions, `is_first_turn` is always 1, so its
  `p` row duplicates `θ`.

## Excluded inputs

The policy never reads `expected_output_tokens` (offline replay fills it with the true output
length), the modeled prefill backlog (`PREFILL_TIME`, an AIS-derived signal), anything else from a
performance model, or the true output length. It declares `CACHE | LOAD`; the composed default
scorer declares `LOAD | PREFERRED_TAINT | CACHE`.

## Information parity with the live router (LR-14)

- Replay applies KV events to the router's index synchronously
  (`lib/mocker/src/replay/offline/extensions/kv_router/mod.rs`, `SyncReplayIndexer`), while a live
  router sees them late, so features 0–2 and the v2 cache features are fresher in replay than live.
- A live router's accounting estimate blends weighted host and disk tiers into the effective
  overlap; replay has only a device tier, so feature 1 there is the device overlap.
- Active prefill tokens, decode blocks and active requests come from the router's own request
  tracking in both settings. That their update timing matches is a hypothesis for the
  information-parity audit to check.
- `worker_capacity()` is configured capacity, not free blocks, in both settings.
