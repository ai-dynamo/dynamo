# Fix record: setup checkpoint, round 0

- Fixer: setup fixer, round 0, 2026-10-02.
- Input: one major finding, F1 from `audits/setup/determinism-cost-context-r0.md`. No blocker was
  reported. The minor findings (F2–F6 of that audit and F1–F7 of `ais-and-policy-r0.md`) were not
  assigned to this round.
- In this file `CR` = `<campaign-root>`, `WT` =
  `<worktree>`, `R` = `CR/runs/setup-fix-r0`.

## F1 (major): repeat-based noise floors are false

**Verdict: valid. Fixed at the root (code, facts, contract amendment).**

### Independent confirmation

I did not reuse the auditor's ±1 ms jitter. I measured the spread with the fix's own perturbation:
a seeded permutation of sessions that share a first-arrival timestamp, so arrival times stay
exactly the same. Each cell had 8 replicates, and every replicate was shared by all policies
(`R/out/*.jsonl`, summarized in `R/crn_noise_summary.json`).

| Cell (2000-row Mooncake slice, SLA 2000/50 ms) | Goodput CV over 8 replicates: default (seed k+1) / lmetric / rr / two-tier | Paired-ratio sd vs default: lmetric / rr / two-tier |
|---|---|---|
| N=8 open, speedup 0.727273 | 1.83% / 1.36% / 1.27% / 2.48% | 0.0243 / 0.0168 / 0.0309 |
| N=4 open, speedup 0.363636 | 1.76% / 3.85% / 1.17% / 2.70% | 0.0508 / 0.0218 / 0.0409 |
| N=8 closed, concurrency 32 | 1.26% / 1.69% / 2.25% / 1.48% | 0.0253 / 0.0261 / 0.0141 |

- Every replicate is itself deterministic. Rerunning replicate 0 reproduced the full per-request
  hash for every policy and cell, so identical re-runs still show zero spread.
- In original file order, N=8 open reproduces setup and the auditor exactly: default seed 1 gives
  2.868672871547336, lmetric 3.153759121853308 and rr 2.890342652363542.
- The spread is as large as, or larger than, the auditor's (CV 1.4–1.8%, paired sd 0.0285), and
  it reaches 3.85% CV at N=4. The pooled paired-ratio sd over the three cells is 0.0356 for
  lmetric, 0.0219 for rr and 0.0307 for two-tier. Per-cell differences of 5–10% are therefore
  within 3 sd, so F1's consequences hold.
- The recorded file order is not a typical draw.
  - rr/default is 1.0076 in original order at N=8 open, below all 8 replicates (1.0195–1.0655).
  - At N=8 closed it is 0.8320, against a replicate mean of 0.8591.
  - So replicate 0 is permuted too. The mechanism is not established [hypothesis].
- Open loop: the spread is in the good count. Goodput CV ≈ good_frac CV, and makespan CV is
  0.2–1.1%.
- Closed loop: the spread is mostly in makespan. good_frac CV is 0.45–0.86% and duration CV is
  0.7–2.0%, which matters for LR-01's denominator choice.

### What changed

1. **Code: WT commit `<commit-02>`.** Subject: `feat(learned-routing): add CRN workload replicates and
   a paired noise rule`. It is signed off and has no Co-Authored-By trailer.
   - `benchmarks/learned_routing/learned_routing/replicates.py` implements protocol `crn-order-v1`.
     Replicate k is a permutation keyed by (source-trace SHA-256, k) and shared by every policy,
     plus policy seed k + 1. Every k is permuted, including 0. By format:
     - **Mooncake:** sessions that share a first arrival are permuted; turn order and arrival
       times are kept.
     - **Weka:** the play order of a play-per-line JSONL is permuted. Agentic lanes take plays in
       source order.
     - **Synthetic sessions:** `synthetic_arrival_seed` supplies the replicate's `arrival_seed`.
   - `PermutationStats.degenerate` flags traces that have nothing to permute.
     `materialize_replicate` writes idempotently and refuses an output directory inside the source.
   - `benchmarks/learned_routing/learned_routing/noise.py` provides:
     - `paired_ratios`, which asserts that both sides cover the same (cell, k) set (CRN);
     - `pooled_sd`, which requires at least 2 replicates;
     - `differs_beyond_noise`, which applies 3 × pooled paired sd per cell, then 2 × SE over
       independent segments (at least 3). A zero pooled sd counts as degenerate and never passes.
   - `tests/test_replicates.py` has 11 tests, all passing. They cover:
     - the multiset and arrival-time invariants;
     - session turn order;
     - k-dependence and exact repeatability;
     - Weka loader order and idempotence;
     - the false floor: identical re-runs cannot certify a difference;
     - noise hiding a small gain but not a large one;
     - segment counting;
     - the CRN key-set assertion.
   - The package is installed editable into PY. Pre-commit passes.
2. **Weka equivalence check (`R/out/weka_identity_check.json`).** The setup AgentX directory and
   its identity-order play-per-line JSONL replay identically on every per-request field except the
   label ids (`uuid`, `session_id`, `play_id`, `request_id`, `agentic`). This holds for seeded
   default and for lmetric (N=2, lanes=1). Each play keeps its own id namespace.
3. **Facts.**
   - `facts/noise.json` is new. It holds the protocol, the cache key, the added `lr-eval` record
     fields, the rules, the minimum replicate counts, the measured values above, and a calibration
     to-do: re-measure per family × N with K = 8.
   - `facts/setup.json` gains `determinism.noise_floor_correction`, which points at `noise.json`.
     I re-serialized the file; no existing values changed.
4. **Contract.**
   - `CONTRACT.md` "Noise, determinism, gates" gains Amendment A1, which binds over the original
     noise-floor and beyond-noise bullets.
   - It also fixes a latent bug in the contract's cache key: `sha256(policy YAML) + cell_id +
     harness_version` would return replicate 0 for every k of an unseeded policy, which silently
     restores the zero-spread floor.
   - Rationale is in `facts/DEVIATIONS.md`, with three entries: the refinement, the package skeleton
     created ahead of build, and replays run outside the not-yet-existing slot pool.

### How each part of the auditor's fix maps

| Auditor's fix | Status |
|---|---|
| `repeat=k` is a CRN workload perturbation keyed by (cell, k), shared by every policy, with policy seed k | Done, with two deliberate refinements. (a) The key is (source-trace SHA-256, k), not (cell_id, k), so cells sharing a trace at other N and load levels share replicate workloads, and the cache works per trace. (b) Policy seed is k + 1, so replicate 0 keeps setup's seed 1. Within-group permutation was chosen over ±1 ms jitter because it leaves every arrival time unchanged. |
| Derive the beyond-noise thresholds and gate condition 3 from paired replicates plus segments, never from identical re-runs | Done in `noise.differs_beyond_noise` and Amendment A1. |
| At least 3 replicates for validation and test, at least 2 in CMA-ES | Recorded as minimums in the contract amendment and `facts/noise.json`. The build stage's `lr-eval`/`lr-train` must enforce them. |
| Record the measured per-cell sd and recalibrate per family and N | Recorded for 3 Mooncake cells and the 2-play AgentX sample (`facts/noise.json`). Per-family × N recalibration is the calibration stage's to-do. |

### Remaining obligations for later stages (not papered over)

- **Build (harness).** `lr-eval` must:
  - call `materialize_replicate` per (cell, k);
  - inject `seed = k + 1`;
  - key the cache by protocol and k;
  - emit `repeat`, `replicate_protocol`, `replicate_seed`, `policy_seed`, `trace_sha256` and
    `replicate_degenerate`.
- **Build (traces).** AgentX traces must be materialized so that a play-per-line JSONL can be
  produced; `load_weka_plays` already accepts a directory.
- **Calibration.** Re-measure the paired sd per family × N × load mode with K = 8. Define the
  independent segments. The 2-play AgentX sample allows only 2 orders, so AgentX noise must come
  from play-subset segments.
- **Pilot.** The cost projection must include K: 3× for validation and test, at least 2× inside
  CMA-ES.

## LESSONS

- **LR-02, applied and refined.** Noise comes from replicates and segments, never repeats. Tie
  seeds alone are not enough, because deterministic policies have no tie RNG. The replicate
  perturbs the workload for every policy.
- **LR-03, applied.** CRN: one replicate set shared by policy and reference, and a key-set
  assertion in `paired_ratios`.
- **LR-11, applied.** Segments are the unit for the SE clause; cells that share a segment count
  once.
- **LR-01, informs, not applied.** Closed-loop noise sits mostly in makespan. The denominator
  belongs to the goodput-math lens.
- **LR-14, rejected for this round.** Timing randomization is a separate axis, owned by training
  and test.
