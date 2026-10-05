# Audit: setup checkpoint, lens "determinism-cost-context", round 0

- Auditor: independent dynamic auditor (adversarial), 2026-10-02.
- Verdict: **FAIL**, because of one major finding (F1, noise floor). Every setup claim I re-ran
  reproduced, except the `max_model_len` rejection rule (F2, minor).
- Evidence root: `CR/runs/audit-setup/determinism-cost-context-r0/`. In this file `A/` means that
  directory and `CR` means `<campaign-root>`.
  - Scripts are in `A/scripts/`: `audit_runner.py`, `rss_growth.py`, `long_context.py`,
    `mml_boundary.py`, `summarize.py`. Job specs are in `A/jobs/`, outputs in `A/out/`, and
    `A/audit_summary.json` collects them all.
- Method:
  - All replays ran through my own runner, which uses the same Python API as setup
    (`run_trace_replay` / `run_synthetic_trace_replay`, `MockEngineArgs.from_json(engine.json)`).
  - It canonicalizes **every** per_request field except `uuid`, including `routing_history`,
    `admission_history`, `itl` and `ttst`. That is stricter than setup's compact rows.
  - It also computes setup's exact compact hash, so I can compare runs across sessions.
  - Replays ran strictly one at a time, as there is no slot pool yet (`CR/slots/` does not exist).
    The host was idle (load < 0.3).
- LESSONS.md appeared at 13:32, mid-audit. Lessons applied:
  - **LR-02** (noise floor): tested and extended; see F1.
  - **LR-09** (verify AgentX load modes): applied; see F6.
  - **LR-03** (common random numbers): it is the basis of F1's fix.
  - **LR-01** (makespan denominator): not evaluated. It belongs to the goodput-math lens.

## Claims re-verified (PASS)

| Claim (setup) | My evidence | Result |
|---|---|---|
| WT on `rupei/learned-routing`, base `2be8d9c43a` = `origin/rupei/router-policy-aic-ttft`, head `<commit-01>` | `git merge-base HEAD origin/...` = `2be8d9c43a…`. `git ls-remote origin` still shows the same SHA, so the base is not stale. HEAD is one commit on top: Conventional subject, `Signed-off-by` present, no `Co-Authored-By`. Upstream is unset and the WT tree is clean. | PASS |
| Main checkout untouched | `git status` in `<repo>` shows exactly the 3 pre-existing `M` files under `lib/kv-router` and `lib/llm`, plus `?? notes/`. HEAD is `8cd79d84ee`, unchanged; `notes/learned-routing/` contains only `PLAN.md`. The top-level dir mtime is 13:26:47 with no persistent entry carrying that time (a transient temp file; nothing persisted). | PASS |
| Bindings carry the seed patch | `_core.abi3.so` was built at 13:21:03 (`maturin_release_2.log`). The patched sources are dated 13:17:52 and unchanged since; only the test file changed later, at 13:24. `Parameters` is `deny_unknown_fields`, so `seed` would be rejected if the patch were absent. The venv imports `dynamo` from WT only. | PASS |
| Determinism, pair 1: seeded `dynamo-default-cost-fn` (seed 1), N=8, 2000-row Mooncake | Two fresh processes: full hash `184addcd…` both times. Compact hash `e7ddbd8f…` equals **setup's recorded hash** (cross-session reproducible). Goodput 2.868672871547336 in all runs (`A/out/fresh_seed1_{a,b}`). | PASS |
| Determinism, pair 2: unseeded builtin default is nondeterministic | Two fresh runs gave 2.9296168 vs 2.8146367, with different hashes (`A/out/fresh_kvdef_{a,b}`). | PASS (confirmed) |
| lmetric deterministic | Compact hash `93bcb1f4…` equals setup's recorded hash, at 3.153759121853308 (`A/out/fresh_lmetric`). | PASS |
| In-process repeats deterministic | One process interleaved seed1, lmetric, unseeded, seed1, seed2, rr, seed1, lmetric, N32, N2. Seed1 is identical at positions 0, 3 and 6 and equals fresh-process. lmetric matches at 1 and 7. Seed2 (2.8180453202645364) and rr (2.890342652363542) equal setup's values exactly (`A/out/inproc_interleaved`). | PASS |
| Host speed doesn't affect results (new test) | Replay pinned with `taskset -c 5` beside a busy loop on CPU 5: wall time 3.32 s vs 1.55 s, identical full hash `184addcd…` (`A/out/contended_seed1`). Static: replay decay uses a synthetic clock (`mod.rs:588,853`). The `Instant::now()` calls in `multi_worker.rs:1168,1201` only feed the metrics publisher. | PASS |
| Determinism with router queueing (new test; `router_queue_threshold` is a planned tuning knob) | N=4 overloaded cell with `router_queue_threshold=1.0`: 786 of 1000 requests queued, identical hashes in two fresh runs (`A/out/queue_{a,b}`). The queue orders by `enqueue_seq`, not UUID (`policy_queue.rs:101-113`). | PASS |
| AgentX replay deterministic | 3 in-process repeats give identical hashes. Compact hash `e0bbdc8d…` equals setup's. AgentX UUIDs are deterministic (`driver.rs: Uuid::from_u128(node_index+1)`), which is why its native summaries are bit-identical. | PASS |
| Cost: ~1.43–1.67 s per fresh 2000-row replay, flat in N | Fresh process: N=2 took 1.464 / 1.480 s (setup 1.480), N=32 took 1.658 / 1.665 s (setup 1.666) (`A/out/fresh_open_n{2,32}_{a,b}`). Within 2x: yes, within 1%. | PASS |
| Cost: ~0.95–1.0 s in-process repeat | 0.81–1.07 s across 9 in-process replays. 40 repeats of seed1 N=8 took 0.95–1.02 s (`A/out/rss_growth_seed1_n8.json`). | PASS |
| Memory ≤ 300 MiB (2000 rows) | 203–298 MiB fresh, 330 MiB in a long-lived process after an N=32 replay. No leak: VmRSS flat at ~216 MiB over 40 repeats (+0.03 MiB per replay), all 40 hashes identical. | PASS (see F5 for full-size cells) |
| Replays are single-threaded (new; matters for slot scaling) | `/usr/bin/time`: 100% CPU (2000 rows) and 99% CPU (23.6k rows); 1 OS thread after replay. | PASS: 20 slots scale ~linearly |
| 128K: single 120K request TTFT 17.8 s | Rerun: 17,815.540859 ms through **both** the synthetic and Mooncake trace paths. It equals **my independent** AIS chunked-prefill sum (15 chunks of ≤8192 tokens, `predict_prefill(1, new, prefix)`): 17,815.540858762586 ms. 131,000 tokens gives 20,617.163325 ms, again equal to the chunk sum (`A/out/long_context/long_context.json`). | PASS (exact) |
| 128K TTFT physically plausible | Qwen3-32B (64 layers, 64 query heads × 128 head dim, ~31.2B non-embedding parameters) prefill at 120K: linear FLOPs ≈ 2·31.2e9·1.2e5 = 7.5e15, causal attention ≈ 2·64·8192·L² = 1.5e16, total ≈ 2.3e16. On 2× H100 SXM (2 × 989 TFLOP/s dense BF16) that is 11.4 s at 100% MFU, so 17.8 s means ~64% MFU, plausible for FA3 plus TP2. The scaling also checks out: 1k gives 55.5 ms (~59% MFU), and 131K/120K = 1.157 predicted vs 20.6/17.8 = 1.157 measured. ITL at 120K is 21.2 ms vs a bandwidth estimate of ~15.1 ms of weights + ~4.7 ms of KV (31.5 GB / 2 GPUs at 3.35 TB/s). | PASS ([hypothesis]-level physics check) |
| `session_context` reaches policies | **Static, my own trace.** The same `session_id` flows: replay driver → `place()` (`agg.rs:667-673`) → `PendingRequest.session_id` (`kv_router/mod.rs:454-473,927`) → `SchedulingRequest.session_context` (`:321-324`) → `WorkerSelectionContext.request` (borrowed, `context.rs:12,69`). Mooncake multi-turn emits it when `!trace.is_single_turn()` (`entrypoints.rs:264`). For the agentic/weka path the driver emits with `emit_session_metadata: true` (`driver.rs:1406`); the ready-turn `session_id` (`:1802`) and `replay_context.session_id` (`:1729`) are both `session.session_id.clone()`. **Dynamic:** multi-turn Mooncake 60/60 (20 sessions × 3 turns), synthetic 180/180 (60 sessions), AgentX 85/85 (10 sessions), single-turn 0/2000 (`A/out/sessions`, `A/out/agentx_inproc`). No built-in worker-selection policy reads `session_context()`, so a policy-level dynamic check still waits for the build-stage sticky-session test. | PASS (see F6 for an evidence-chain nit) |
| `expected_output_tokens` = true OSL | Static: `mod.rs:459-463` gives `max_output_tokens = effective_max_output_tokens()` (= trace `output_length`, `protocol.rs:121`), and `:920-923` sets `Some(max_output_tokens)`. Load tracking discards it (`single.rs:260`), and no built-in policy calls `expected_output_tokens()` (rg over `lib/router-plugins`, `lib/kv-router/src`). Dynamic: `output_length == requested_output_length` on 2000/2000, 60/60, 180/180 and 85/85 rows, and the multiset of (ISL, requested OSL) equals the trace's (ISL, OSL) on the 2000-row slice. | PASS: the leak is real, and learned features must not read it |
| Per-request goodput recomputation = native | On every audit run with an SLA, the recomputed goodput (TTFT ≤ T, ITL ≤ I, ITL skipped when output ≤ 1) equals the native `goodput_request_throughput_rps` bit-for-bit. | PASS |

## Findings

### F1 (MAJOR): repeat-based noise floors are illusory; per-cell goodput has ~1.4–1.8% sd from arbitrary arrival order, for every policy

- **Claim being audited.**
  - Setup reports that round-robin, lmetric, two-tier and the seeded default are per-request
    identical on repeats, and that the default's tie noise is about 2%.
  - It advises: "use common seeds across policies and several seeds for noise estimates".
  - The contract's noise floor is "3 × the pooled sd of repeats".
- **Evidence** (`A/out/jitter`, `A/out/speedup_ppm`, `A/audit_summary.json`).
  - I perturbed only the **within-burst order** of the N=8 2000-row Mooncake cell. Each timestamp
    got ±1 ms of jitter (4 seeds), then a stable sort. That moved ~1,800 rows, because the trace has
    only 113 distinct timestamps (see F3).
  - Goodput, unperturbed followed by jitter seeds 1–4:

    | Policy | Values | sd | CV | Range |
    |---|---|---|---|---|
    | lmetric | 3.1538, 3.1249, 3.1082, 3.0257, 3.0363 | 0.056 | **1.8%** | 4.1% |
    | round-robin | 2.8903, 2.8785, 2.9231, 2.9609, 2.9656 | 0.040 | **1.4%** | 3.0% |
    | seeded default | 2.8687, 2.7796, 2.7703, 2.8402, 2.8446 | 0.043 | **1.5%** | 3.5% |

  - Paired normalized ratios, the quantity in the training objective and the gate:
    - lmetric/default: 1.099, 1.124, 1.122, 1.065, 1.067. sd **0.0285**.
    - rr/default: 1.008, 1.036, 1.055, 1.043, 1.043. sd 0.018.
  - The default's tie-RNG spread over 7 realizations matches. Seeds 1/2, setup's unseeded pair and
    my 3 unseeded runs give 2.815–2.930, with sd 0.044 (CV 1.5%) and range 4.0%.
  - Control: order-preserving continuous perturbations do **not** do this. `arrival_speedup_ratio`
    values 0.727272, 0.727274 and 0.727275 move goodput only at the ppm level (lmetric
    3.1537551 / 3.1537632 / 3.1537673). So the mechanism is discrete: routing order and tie events
    within simultaneous arrival bursts.
- **Why it matters.**
  - Deterministic policies (rr, lmetric, two-tier, and learned-choice at temperature 0 without ties)
    have zero repeat sd by construction, and tie seeds don't perturb them either.
  - A repeat-based "3 × pooled sd" threshold therefore uses roughly the default's tie noise alone, a
    false floor. Per-cell differences of 3–6% are within the per-cell noise I measured.
  - Consequences:
    - Per-cell "beyond noise" calls would be overconfident.
    - CMA-ES on a few train cells can fit order artifacts.
    - Pilot gate condition 3 (first clause) can pass on noise.
  - The second clause (mean Δ over validation cells > 2 SE across cells) is the only part that
    stays honest, and only if there are enough cells.
  - LESSONS LR-02 makes the same diagnosis. Its fix ("K tie seeds × workload segments") still gives
    RNG-free policies zero within-segment spread. This audit shows a cheap within-cell replicate
    that perturbs all policies.
- **Fix.**
  1. In `lr-eval`, make `repeat=k` a **common-random-numbers workload perturbation**. Use a seeded,
     deterministic permutation of the order within each equal-timestamp group, or seeded ±1 ms
     jitter, keyed by (cell, k) and identical across policies. Also set the policy `seed = k`.
  2. Estimate noise floors and gate thresholds from these paired replicates, and segments (LR-02),
     never from identical re-runs.
  3. Use ≥ 3 replicates for validation and test decisions, and ≥ 2 inside CMA-ES evaluations, or
     average over longer windows.
  4. Record the measured per-cell sd (about 1.5% absolute, about 0.02–0.03 in the paired ratio, at
     this cell) in facts, and recalibrate it per family and N.

### F2 (MINOR): the `max_model_len` rejection rule is misreported; only ISL ≥ 131072 is rejected, and ISL+OSL > 131072 completes with truncated output

- **Claim being audited.** `facts/setup.json` `long_context.notes` and the issues list say "Requests
  with ISL+OSL above 131072 are rejected". The only evidence was an ISL of 140,000.
- **Evidence** (`A/out/long_context/mml_boundary.json`):

  | ISL + OSL | Sum | Status |
  |---|---|---|
  | 131,009 + 64 | 131,073 | completed |
  | 131,050 + 64 | 131,114 | completed |
  | 131,000 + 73 | 131,073 | completed |
  | 131,071 + 1 | 131,072 | completed |
  | 131,072 + 1 | 131,073 | **rejected** |

  The log line is `rejecting request that exceeds a worker admission limit prompt_tokens=131072
  max_model_len=131072`.
- **Over-length requests are truncated, not rejected** (`A/out/long_context/mml_truncation.json`).
  ISL + realized `output_length` is exactly 131,072 in every case:

  | ISL + requested OSL | Realized output |
  |---|---|
  | 131,050 + 64 | 22 |
  | 131,009 + 64 | 63 |
  | 131,000 + 73 | 72 |
  | 130,000 + 2,000 | 1,072 |

  `terminal_status` is `completed` throughout. That mirrors vLLM's `finish_reason=length` when the
  client doesn't pre-validate `max_tokens`. A client that does would get HTTP 400. The parallel
  ais-and-policy auditor reported the same truncation.
- **Impact.**
  - Small: setup's next-stage notes already say to drop or flag ISL+OSL > 131072 at trace
    preparation.
  - But a harness that relies on replay rejection to exclude those requests will silently serve
    them, truncated.
  - For those rows `requested_output_length` (= `expected_output_tokens`) ≠ realized
    `output_length`.
- **Fix.**
  - Correct the claim in facts.
  - Trace preparation must filter ISL+OSL > 131072 explicitly and log the count. This matters most
    for AgentX: the "intact plays that fit" criterion must be ISL+OSL ≤ 131072, not ISL ≤ 131072.

### F3 (MINOR, unrecorded workload property): Mooncake and toolagent arrivals are 3 s-quantized bursts

- **Evidence.**
  - `mooncake_trace.jsonl` and `toolagent_trace.jsonl` each have 23,608 rows but only **1,180
    distinct timestamps**.
  - Gaps are 3,049–3,053 ms (Mooncake) and 2,997–3,000 ms (toolagent); bursts average 20 and peak
    at 47 simultaneous arrivals.
  - The 2000-row slice has 113 distinct timestamps.
- **Impact.**
  - Open-loop cells are burst-driven. The router places up to 47 requests at the same simulated
    instant, using only its own just-routed accounting.
  - The within-burst order, an arbitrary file order, drives F1's noise.
  - This is a strong structural fact for calibration (knee location) and for feature design (e.g.
    `active_prefill_tokens` within a burst). It isn't in setup facts, PLAN or LESSONS.
- **Fix.**
  - Record it in facts.
  - Calibration should decide deliberately whether open-loop cells keep the bursts or de-quantize
    them, by spreading arrivals uniformly within each 3 s bucket as a seeded transform. Label the
    choice.

### F4 (MINOR, provenance): the in-process cost measurement has no saved script, and its hash doesn't match the fresh-process hash

- **Evidence.**
  - `runs/setup/cost/inprocess/result.json` has no generating script in `runs/setup/scripts/`.
  - Its `open_n8` (seeded) canonical hash is `7edcc2fe…`. The same cell's fresh-process hash under
    `smoke_lib.py` is `e7ddbd8f…`, even though the goodputs are equal.
  - My in-process rerun produced `e7ddbd8f…`, so the replay is fine. The unsaved script evidently
    canonicalized differently.
- **Fix.** Save the script, or annotate the file. The wall-time numbers are corroborated (F-table).

### F5 (MINOR): cost and memory were projected only from 2000-row cells

- **Evidence** (`A/out/full_mooncake_n{8,32}`, `A/out/agentx_inproc`).
  - The full 23,608-row Mooncake trace takes **7.64 s at N=8 and 8.77 s at N=32** per fresh
    replay, which confirms setup's unmeasured "~10 s".
  - Its peak RSS is **1,082–1,136 MiB** in the runner's `ru_maxrss`, and 1,149–1,188 MiB for the
    process (`/usr/bin/time`), not ≤ 300 MiB. 20 slots × about 1.2 GiB ≈ 24 GiB of the host's
    125 GiB, which is fine.
  - AgentX (2 plays, 85 requests, N=2) takes 0.87 s per in-process replay.
- **Projection [hypothesis]:** a 1-request replay costs 0.39 s in-process (setup `ais_probe`). That
  gives a marginal cost of about 5.6 ms per AgentX request against about 0.3 ms per Mooncake
  request, so the full 3,132-request AgentX family is about 18 s per replay. This needs measuring
  once traces exist.
- **Impact.** Setup's "CPU cluster unlikely unless > ~100k replays" holds only for 2000-row cells. At
  20 slots, 8 h buys about 76k full-trace Mooncake replays and plausibly about 32k full AgentX
  replays.
- **Fix.** The pilot must project cost per family using measured per-request marginal cost.

### F6 (MINOR): AgentX evidence chains are partly vacuous

1. **Session evidence.** Setup cites `agg.rs:667-690` as proof that per_request session ids equal
   the placement session ids. For weka rows, though, per_request `session_id` is written by
   `on_request_context` from `replay_context` (`agg.rs:352-353`, `report.rs:1340`).
   `on_session_metadata` is then a set-once no-op (`report.rs:1272-1273`), which is why
   `turn_index` is null. The conclusion is still right; the real link is `driver.rs:1729` and
   `:1802`, which both clone `session.session_id`. But the cited chain did not prove it for AgentX.
2. **Lanes evidence.** "AgentX replays at N=2 with lanes=2 and in timestamp mode" uses 2 plays,
   and `effective_agentic_lanes` = min(lanes, plays). With lanes equal to the play count, the two
   runs are **identical up to a worker relabeling** (same arrivals, TTFTs, goodput 0.04323735422852583
   and duration; workers 54/31 vs 31/54; `A/out/agentx_inproc`). So nothing has yet shown that
   `agentic_lanes` actually constrains load.
- **Fix.** Calibration must exercise lanes < plays (LR-09).

## Not findings (checked, OK)

- **UUIDs.** Random request UUIDs do not affect simulated outcomes. Identical canonical hashes held
  across every repeat. The native mean and std differ by about 1e-15 only because of summation
  order, as setup said.
- **Seeded RNG lifetime.** The seeded default's RNG is per policy instance and per replay call.
  Interleaving other replays in the same process does not shift its stream.

## Process notes

- **Replays.**
  - I ran about 110 short replays (about 0.4–9 s each), strictly sequentially, outside a slot pool
    that does not exist yet.
  - One run deliberately shared CPU 5 with a 60 s busy-loop (PID 185817), which I stopped by exact
    PID.
  - Recorded in `facts/DEVIATIONS.md`.
- **Writes.**
  - I made no changes to WT or to the main checkout.
  - I did run read-only `git status`/`git log` in the main checkout, which may refresh
    `.git/index` stat data.
- **Scratch.** `A/` is about 229 MiB, mostly full-trace per_request JSONL. It is listed in
  `CLEANUP.md`.
