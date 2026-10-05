# Audit phase2-mid, lens "gaming-concentration" (LR-13, A11.2)

- **Auditor:** independent adversarial auditor, 2026-10-03 19:50-21:00 PDT.
- **Verdict: FAIL.** 0 blockers, 2 majors, 4 minors.
- **A11.2 decision: GAMING.** Re-run all three M1 restarts (m1-s1, m1-s2, m1-s3) with sign-constrained load
  coefficients at the same B = 400 before Select+Test (details in F1).
- **Evidence directory:** `CR/runs/audits/phase2-mid-gaming/`. Scripts are in `scripts/`. The decision
  rules in `PRESET.md` were written before any phase-2 M1 per-request row was read. The key numbers are in
  `summary.json`.

Every number below comes from that directory: means over CRN replicates k = 0, 1, 2 on the 14 val cells.
I did not touch val k3-10, which stays fresh for Select+Test, and I did not touch test.

## What I ran (own replays, own scoring)

- **Replays.** 882 replays, 0 errors, through the CR slot pool (16 slots), in a private root
  (`root/`: own result cache and own copy of the E0 v2 table; symlinked config, traces, cells and
  replicates). No replicate file was added to `CR/runs/replicates`.
- **Policies replayed:**
  - the 11 tier-A selected configs (`runs/phase2/selection/tierA.specs.jsonl`);
  - the A12 row (m1-s1 g25 mean);
  - m1-s3's own best checkpoint (g25 mean);
  - default@defaults and round_robin;
  - 7 M1 counterfactuals: theta7 = 0 on each of the three restarts; theta2 = theta7 = 0; theta1 x 0.5; and
    theta5 x 2.
- **Reproduction check.** The replays match the campaign cache record for record: 546 of 546 records are
  identical, including `per_request_canonical_sha256` and `goodput_rps_window`. My recomputed val k0-2
  objectives equal the lr-train history for all 12 selected rows (absolute difference 0.0), and m1-s3 g25
  gives 0.15782 (history 0.157820). `objectives.json`.
- **Scoring.** `scripts/analyze.py` is my own A2 scorer. It uses the E0 table and the window rule; the
  harness is used only for replicate paths. Its per-record window_good and window_requests equal the
  harness's on 882 of 882 records.
- **Guard metrics per (policy, cell, k)** (`guards.json`):
  - worker request share, window basis, against the cap min(2/N, 1/N + 0.25);
  - share of prefill tokens and of input tokens;
  - prefix reuse, per worker and overall;
  - ISL segregation, NMI(worker; ISL quartile), and prefix-group segregation (Mooncake first-block groups);
  - starvation: e2e > 3 S x E0 or never completed, and > 10 S;
  - good fraction by ISL quartile and top decile;
  - per-worker good fractions;
  - engine-state proxies: time-averaged in-system and decoding requests per worker, a KV-token occupancy
    proxy, and ITL/TTFT on the max-share worker versus the others;
  - paired good-to-bad and bad-to-good transitions by request identity.
- **Per-request mechanism scripts:**
  - `scripts/giant.py`: requests with ISL >= 60K, plus "dump" workers, meaning workers on which >= 80% of
    in-window requests are such giants;
  - `scripts/attraction.py`: the rank of the chosen worker in reconstructed outstanding prefill tokens when a
    giant arrives;
  - `scripts/per_worker.py`;
  - `scripts/stragglers.py`.

## Results

### Concentration (window share against the cap)

- **M1, M0 and the baseline contenders.**
  - The selected M1 (m1-s2 g15) exceeds the cap on 4 of the 5 Mooncake val cells:
    - n6-open-L2: 0.369 against 0.333;
    - n8-closed-L2: 0.364 against 0.250;
    - n8-open-L3: 0.402 against 0.250;
    - islu1.25-n6-closed-L2: 0.454 against 0.333.

    The A12 row also exceeds it on 4 Mooncake cells, with a maximum of 0.477, and m1-s3 g25 on 2.
  - M1 stays within the cap on every sessions and AgentX val cell, and so does every tuned policy.
  - M0 and every other contender stay within the cap everywhere: ramjet, llmdpp, stickybounded, twotier,
    lmetric, stickyhard and llmdob. ramjet comes closest, at 0.245 against 0.250 at n8-open-L3.
  - chwbl and dualmap exceed the cap on 3 and 4 Mooncake cells respectively (F5).
- **The concentration is in requests, not in work.** The hottest worker's share of prefill tokens stays
  near 1/N: 0.146 against 0.125 at N8, and 0.183 against 0.167 at N6. The max-share worker carries the
  shared-prefix group: Mooncake w1 group 0 is 1,484 rows sharing a 12-block (6,144-token) prefix, with ISL
  p50 6,578. M1 sends 48-56% of that group to one worker; default sends 15-20%. That worker's prefix reuse
  is 0.64-0.73 against 0.19-0.24 on the other workers.
- **The concentration buys almost no cache over within-cap baselines.** Overall in-window prefix reuse at
  n8-open-L3 is 0.381 for M1, against 0.386 for llmdpp and 0.377 for ramjet. Both baselines stay within the
  cap. Across the 4 violating cells, M1's reuse is at most +0.0065 above the best within-cap baseline
  (lmetric or llmdpp), and below it on 2 of the 4. So the
  cap violation is mainly **segregation**: cheap cached short requests go to one worker and the
  long-prefill traffic to the rest. It is not extra cache affinity.

### Sacrifice and starvation: per-request evidence

n8-open-L3, means over k0-2. The same columns for every violating cell are in `summary.json` (`by_cell`,
`giants_ge_60k`, `giant_attraction`) and in `guards.json` (`paired_by_cell`).

| policy | share (cap 0.25) | good | top-ISL-decile good | ISL >= 60K good | dump workers/k | giant sent to the max-outstanding-prefill worker (chance 0.125) | good-to-bad vs default in ISL q4 | min worker good |
|---|---|---|---|---|---|---|---|---|
| M1 selected (m1-s2 g15) | 0.402 | 0.916 | 0.730 | 0.425 | 1.0 | 0.580 | 0.68 (x2.8) | 0.000 |
| A12 (m1-s1 g25) | 0.477 | 0.930 | 0.844 | 0.874 | 0.7 | 0.283 | 0.58 (x2.4) | 0.642 |
| m1-s3 g25 | 0.332 | 0.900 | 0.865 | 0.782 | 1.0 | 0.442 | 0.47 (x1.9) | 0.573 |
| M1 selected, theta7 = 0 | 0.434 | 0.888 | 0.679 | 0.943 | 0.0 | 0.130 | 0.69 (x2.8) | 0.467 |
| ramjet (selected) | 0.245 | 0.831 | 0.882 | 1.000 | 0.0 | 0.000 | 0.37 (x1.5) | 0.643 |
| llmdpp (selected) | 0.235 | 0.838 | 0.920 | 0.862 | 0.0 | 0.145 | 0.29 (x1.2) | 0.561 |
| M0 (m0-s3 g20) | 0.158 | 0.770 | 0.867 | 1.000 | 0.0 | 0.007 | 0.33 (x1.3) | 0.668 |
| default@defaults | 0.152 | 0.611 | 0.880 | 1.000 | 0.0 | 0.007 | - | 0.487 |

- **A sacrificial worker.** In every replicate at n8-open-L3, the selected M1 leaves one worker receiving
  only 14-16 in-window requests, all very long, with median ISL 98-100K. Not one of them is good:
  - k0: w7, 0 of 16 good;
  - k1: w2, 0 of 14;
  - k2: w4, 0 of 15.

  The hot workers, meanwhile, serve 519-1,014 mostly short cached requests at 0.90-0.99 good
  (`scripts/per_worker.py`).
- **These giants gain nothing from cache.** All 46 rows per replicate trace with ISL >= 60K (29 of them in the
  window) belong to first-block
  group 14, which shares only 1 block (512 tokens) across its members (`prefix_groups.json`). Stacking them
  on one worker therefore has no cache rationale.
- **These giants are not doomed.** At n8-open-L3, default, ramjet, M0, twotier and chwbl serve the ISL >= 60K
  requests 100% good, and llmdpp serves 86%. M1 serves 42.5% at n8-open-L3 and 69.2% at n8-closed-L2.
- **Mechanism: load attracts long requests.**
  - When a giant arrives, M1 sends it to the worker with the most outstanding prefill 58% of the time at N8
    (chance 12.5%; default 0.7-2.2%; ramjet 0-0.7%). In 60-67% of cases that worker already holds an
    outstanding giant; default and ramjet never do this (0%).
  - This is what the selected theta predicts. The total coefficient on active prefill tokens is
    -A_p/P (from theta0 = -1) + theta3/T + theta7·(P/T)/T. With theta3 = -0.724 and theta7 = +0.0878 it turns
    positive above about P = 77K tokens; for the A12 theta the threshold is about 108K.
  - Setting only theta7 = 0 removes the dump worker on all three restarts (0 per replicate). It brings
    giant good back to 0.943 for s2, 1.000 for s1 and 1.000 for s3 at n8-open-L3, and drops the attraction
    to chance or below (0.130, 0.014, 0.022).
- **Long requests carry the losses, beyond the dump worker.** Against default:
  - 56-68% of M1's good-to-bad transitions fall in the top ISL quartile, x2.3-2.8 its 0.25 share, on
    n8-open-L3, n8-closed-L2 and islu1.25-n6-closed-L2.
  - The rescued requests are mostly short: only 8-14% of them are in q4.
  - The baseline contenders put x0.6-1.6 of their losses in q4.
  - The theta7 = 0 projections keep this skew: x2.2-2.9 for s1 and s2, x1.5-2.3 for s3. So it comes from
    segregation and not only from theta7.
- **No starvation in the long-wait sense.** M1's share of slowdowns above 3S is below every baseline's on
  every Mooncake cell: 0.0002-0.0037, against 0.0122-0.0411 for default. No exposed request stays incomplete, and the
  maximum slowdown is 3.2-5.9 S against 9.0-11.2 S for default. The sacrificed requests miss narrowly: the
  median slowdown of a giant at n8-open-L3 is 1.13 S.
- **Pre-set rules** (`rules.json`):
  - The selected M1 is flagged as gaming on 3 of its 4 violating cells: S1 at n8-open-L3, and S4 at
    n8-open-L3, n8-closed-L2 and islu1.25-n6-closed-L2.
  - The A12 row is flagged on 3 cells (S4).
  - m1-s3 g25 trips none of S1-S4. It still has the dump worker at n8-open-L3, with giant good 0.782, so it
    is covered by the mechanism finding.
  - No contender baseline is flagged. chwbl and dualmap trip S3 (F5).

### Engine-state shift

On the four violating cells, M1 runs its max-share worker hot:

- Time-averaged in-system requests are 1.15-1.50 x the worker mean. Baselines are at 1.02-1.10 and default at
  1.10-1.16.
- The KV-token occupancy proxy (an upper bound that ignores sharing) reaches 0.69-0.82 of 301,808 tokens,
  against 0.42-0.77 for the contender baselines.
- The decoding batch on that worker is higher: 9.8 against 7.3-8.2 at n8-closed-L2.
- Its ITL is lower (24-31 ms), while TTFT p90 on the other workers rises: 2.78 s against 1.99 s for
  default and 2.08 s for ramjet at n8-open-L3.

AgentX and sessions show no M1-specific shift:

- On AgentX, most tuned policies lower top-decile good against default. M1 (0.85, 0.85, 0.95, 0.99 on V1,
  V2-n4, V2-n6 and V3) stays within the contender baselines' range.
- The sessions val cells are saturated: every contender is at good >= 0.998.

### Val k0-2 objective (selection-exposed; MDE 0.038)

| config | val k0-2 |
|---|---|
| M1 selected (s2 g15) | 0.16363 |
| theta7 = 0 | 0.15613 |
| theta2 = theta7 = 0 | 0.15806 |
| theta5 x 2 | 0.15542 |
| theta1 x 0.5 | 0.14350 |
| A12 (s1 g25) | 0.16148 |
| A12, theta7 = 0 | 0.15740 |
| m1-s3 g25 | 0.15782 |
| m1-s3, theta7 = 0 | 0.15194 |
| ramjet | 0.14202 |
| llmdpp | 0.14145 |

- About 80% of M1's 0.0216 lead over ramjet comes from the 5 Mooncake cells (0.0174 of 0.0216). Those are
  the cells with the cap violations.
- The theta7 projection shrinks the lead to 0.0141. These are projections, not retrained optima.

## Findings

### F1 (major): M1 games the objective; A11.2 is triggered for all three M1 restarts

- **Claim.**
  - The selected M1's cap violations are not legitimate cache affinity. They combine hot-prefix
    segregation with a load-attracting coefficient (theta7 > 0) that stacks requests of 60K+ tokens onto
    one prefill-busy worker.
  - That sacrifices requests every baseline serves: ISL >= 60K good is 0.425 against 1.000 at
    n8-open-L3. The policy is degenerate: one worker has 0 good requests in every replicate.
  - All three restarts end with theta7 > 0 at their selected checkpoints: s1 +0.0707, s2 +0.0878,
    s3 +0.669. Each shows the dump worker at n8-open-L3, at 0.7, 1.0 and 1.0 per replicate.
- **Why it matters.** The current M1 (and the A12 row) is not a valid headline candidate under A11.2. M2's
  warm start (tier B) would inherit the behavior.
- **Fix (required before Select+Test).**
  - Re-run m1-s1, m1-s2 and m1-s3 with the same inits, seeds, B = 400, popsize 16, K = 2, UH re-evaluation
    and validation schedule.
  - Constrain theta3, theta4, theta5 and theta7 to <= 0 (upper bound 0). With theta0 pinned at -1, that is
    the minimal set under which no load signal can attract for any prompt length.
  - Spaces to change: `runs/phase2/spaces/m1_learned_v1.yaml`, `m1_learned_v1_s2bc.yaml` and
    `m1_learned_v1_s3cache.yaml`.
  - The s2 behavior-clone init itself attracts: theta3 = +1.28 and theta4 = +12.5, so decode load attracts
    above about 24K tokens. Project it to theta3 = theta4 = 0 and disclose the projection under A11.1 and
    A12.
  - theta3 > 0 or theta4 > 0 also appear at validated checkpoints of all three restarts (s1 g5-g20,
    s2 g5-g20, s3 g5). So constraining theta7 alone is not enough.
  - Inits that sit on the new bound (s1, s3) put CMA-ES at the edge of the box. The re-run stage should
    check the bound transform's behavior there.
- **Then:**
  - Re-select on val k0-2 under LR-10.
  - Make the A12 row the constrained s1.
  - Keep the unconstrained M1 only as a disclosed, gaming-flagged row.
  - Start M2 (and the M1-continued comparator) from the constrained selection.
  - Recommendation: apply the same sign rule to M1-noaff (A15) and M1-v2 (A17.1). In v2 the load-ordered
    features are 3, 4, 5, 7, 8, 10, 12-21.

### F2 (major): the sign constraint alone may not remove the long-request sacrifice

- **Claim.** The theta7 = 0 projections remove the dump worker, but they keep the cap violations
  (0.350-0.473 at n8-open-L3) and the skew of losses onto long requests:
  - S1 or S4 still fire on 1-3 cells per restart;
  - q4 takes x1.5-2.9 of the good-to-bad transitions;
  - top-decile good at n8-open-L3 is 0.679 against 0.880 for default.
- **Mechanism (hypothesis, from projections and not from a retrained optimum).** The strong overlap weight
  and request-count repulsion (theta1 of about 61, theta5 of about -146) send short cached group-0 requests
  to one worker. The long requests then compete on the rest.
- **Why it matters.** A retrained constrained M1 is not guaranteed clean. The headline gain over within-cap
  baselines would still partly come from redistributing misses onto long requests, which the request-count
  objective rewards (LR-13).
- **Fix.**
  - Pre-register now that the constrained re-run's selected config is re-audited with this directory's
    scripts (`analyze.py`, `giant.py`, `attraction.py`, `rules.py` with the PRESET.md thresholds) on val
    k0-2 before Select+Test.
  - REPORT carries the concentration, segregation (NMI 0.10-0.24 against <= 0.10 for contenders) and ISL-q4
    sacrifice flags for every M1 row.

### F3 (minor): engine-state shift is a staleness risk, untested

- M1 runs one worker at 1.43-1.50 x mean occupancy, with its KV proxy up to 0.82 of capacity, and depends on
  perfectly fresh state to keep giants off the cached-short worker.
- Hypothesis: router-state lag herds this concentration (LR-14).
- Fix: in the A5 lag pass, also report share, dump workers and ISL-q4 good for M1 and the val-best baseline.

### F4 (minor): the harness LR-13 guards could not see this failure

- `goodput.guard_metrics` computes `worker_share_max` over all rows, warm-up and drain included: 0.316
  against 0.369 in-window at n6-open-L2.
- It has no per-worker good minimum, no top-ISL-decile good against a reference, and no attraction metric.
- lr-train validation records keep no per_request rows. So the pilot, gate and tier-A guard reads flagged
  concentration but not the 0%-good worker.
- Fix: add window-basis `worker_good_frac_min` and top-decile good to the guards, and keep per_request rows
  for validation checkpoints.

### F5 (minor): chwbl and dualmap concentrate and starve more

- The selected chwbl exceeds the cap on 3 Mooncake cells and dualmap on 4 (up to 0.360 against 0.333).
- Their share above 3S at n8-open-L3 is 0.072 and 0.051, against 0.041 for default, so S3 fires.
- They are not contenders (val 0.029 and 0.040). REPORT should list them with concentration flags.

### F6 (minor): completion windows do not score stragglers

- In closed loop, a request that is still running at the full-occupancy end is never scored.
- At n8-closed-L2, 7.3 of 29 giants per replicate are unscored under M1, against 1.0-1.3 for default,
  ramjet, llmdpp and M0.
- The effect is small (about 0.3% of window requests), but it rewards stranding long requests.
- Fix: report unscored exposed requests by ISL with closed-loop results (LR-01).

## Lessons

- **Applied:**
  - LR-13: guards, cap rule, sign constraint, ISL-stretch stress via the islu1.25 val cell;
  - LR-10: the selection is unchanged and re-run restarts are re-selected;
  - LR-11: val k3-10 and test untouched, and the counterfactual configs are counted only here, not as
    learned arms;
  - LR-01: window basis;
  - LR-02 and LR-03: CRN replicates;
  - LR-14: F3.
- **Rejected:** none.

## Files

- `CR/runs/audits/phase2-mid-gaming/`:
  - `PRESET.md`, `scripts/*.py`;
  - `val_k012/results.jsonl` (882 records);
  - `guards.json`, `objectives.json`, `rules.json`, `giant_60000.json`, `attraction.json`,
    `stragglers.json`, `prefix_groups.json`, `summary.json`;
  - `specs_*.jsonl`.
- **Process.** Background lr-eval PIDs 1717630 and 1762526 (recorded in `*.script.pid`) both exited 0.
  Nothing was killed. No WT changes, no commits, no pushes.
