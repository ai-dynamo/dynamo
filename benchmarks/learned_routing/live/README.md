# Live lane: AIPerf inputs and scoring for real-GPU runs

Amendment A13 asks whether the simulated policy ranking holds on a real deployment (Qwen3-32B, vLLM
0.24.0, H100, TP2, aggregated). This directory makes a live run replay the same workload as offline
replay, request for request, and scores it with the harness's own A2 scorer. `CR` below is the
campaign root.

| File | What it does |
|---|---|
| `gen_aiperf_inputs.py` | `cell`: a cell plus CRN replicate `k` gives an AIPerf input file, a per-request table and a manifest. `idle`: an idle-calibration input for the live-fitted E0. `verify-replay`: checks a generated schedule against a cached replay `per_request` file. |
| `score_live.py` | `score`: AIPerf per-request export to the harness scorer (windowed goodput, ITL, E2E slowdown against E0). `fit-e0`: fits the live idle E0. `compare`: joins live rows with replay rows of the same cell and replicate. |
| `run_aiperf.py` | Runs AIPerf with exactly the manifest's argv and environment, and writes `<artifact>.run.json`. |
| `smoke_payload.sh` | The AIPerf payload for `deploy/job_node.sh`: each run under `LR_SMOKE_INPUTS` in order (default `cell idle`), bracketed by metric scrapes, with a background scraper of the frontend and every worker. |
| `scrape_metrics.py` | Prometheus scraper for `vllm:*` and `dynamo_*` series (`loop` or `once`); standard library only. |
| `smoke_compare.py` | Live run against replay of the same cell and replicate: latency distributions, prefix-cache hit rate, per-worker balance, goodput under the AIS and live E0, schedule lateness, idle E0 against AIS. |
| `aiperf_env.sh` | Builds the pinned AIPerf 0.13.0 environment on the GPU cluster's shared filesystem (uv CPython 3.12.3); its freeze must equal the workstation's. Site settings come from `deploy/site.env`. |
| `loadgen_contract.py` | `serve`: a dummy SSE completions server that logs what it receives. `check`: compares that log with the generator's request table. This is the GPU-free check that AIPerf sends what the generator asked for. |
| `tests/` | Unit tests, plus a check that the scorer reproduces cached `lr-eval` records. |
| `audit_parity.py` | Independent loadgen parity audit. `static` compares AIPerf's own reading of an input with a reference written from the replay source and with cached replay `per_request` rows. `stub-prep` and `stub-check` drive AIPerf against a stub that plays back replay's latencies. `summarize` builds one table. |
| `audit_aiperf_view.py` | The audit's AIPerf-side view (AIPerf environment): AIPerf's `MooncakeTrace` model and loader output, 16-token block hashes and sampled token LCPs. |
| `audit_stub_server.py` | The replay-playback completions stub (AIPerf environment). |

## Load generator: AIPerf 0.13.0, upstream

Use `aiperf==0.13.0` from PyPI (wheel SHA-256
`a20100fd127f1d10ea45d4946b1658a186030e86770344c80dfd256a60192f34`), installed in its own
environment, never in the worktree `.venv`. A local copy is at `CR/runs/live/env/aiperf-0.13.0`;
its freeze is next to it. Upstream is enough for every family except AgentX, so no AIPerf fork is
needed. These upstream features carry the semantics:

- `mooncake_trace` rows with a verbatim `payload`, which carries the token-ID prompt;
- `timestamp` on first turns, replayed with `--fixed-schedule`;
- `delay` on later turns: milliseconds after the previous turn's credit returns;
- session concurrency (`--concurrency`), where a session holds its slot across turns and think
  time;
- `--dataset-sampling-strategy sequential` plus `--num-sessions`, so each session runs exactly once;
- `AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID`, which sends Dynamo's session header;
- counting the pretokenized prompt;
- server-reported usage (`--use-server-token-count`).

### What is supported

| Family | `open_speedup` / `open_rate` | `closed_concurrency` |
|---|---|---|
| mooncake | faithful | faithful |
| fast25_conversation, fast25_synthetic | faithful | faithful |
| synthetic_sessions | faithful (multi-turn, think time, prefix accumulation) | faithful |
| agentx (`agentic_lanes`) | not supported | not supported |

AgentX is refused with an error. The lowered Agentic Mooncake v2 cells need all of the following:

- typed `dispatch` and `completion` edges with per-edge delays;
- `join` and `replay_barrier` relations;
- play-relative `not_before_ms` floors;
- `L` lanes, with copies dealt `ordinal % L` in order;
- a lane that activates its next copy the instant the previous play is quiescent.

No available load generator implements that:

- Upstream `agentic_replay` samples trajectory snapshots at a random `t*`, adds a warm-up phase,
  infers cross-stream edges from recorded intervals, and recycles lanes from the sampler.
- Upstream `dag_jsonl` has only completion-triggered spawns and accumulates chat messages.
- A prior finite AgentX fork of AIPerf (unpublished) plays one play per lane with no recycling,
  and releases a root slot on the root's terminal request.

Driving AgentX live needs one of two things: a dedicated driver that executes the lowered graph
with replay's lane rules, or new fork work on lane recycling and the `replay_barrier` relation.

## Semantics, replay against live

| Aspect | Offline replay | Live (this lane) | Evidence |
|---|---|---|---|
| Arrivals, open loop | `(t - min t) / speedup` on the replicate trace (`crn-spread-v1` or `crn-order-v1`) | The same float operations; written as `timestamp` | `verify-replay`: bit-identical on every request (below) |
| Think time | Next turn ready at completion plus `delay / speedup` (open) or `delay` (closed) | `delay` after AIPerf's credit return, already scaled | `verify-replay`: max error 1.2e-10 ms; contract residuals below |
| Closed loop | `C` session slots, activated in replicate order, each held across turns and think time | AIPerf session concurrency `C`, sequential sampling, `--num-sessions S` | `verify-replay`: order and `C` sessions at t = 0; contract: peak `C`, 0 inversions |
| ISL | `input_length` tokens | Token-ID `prompt` of exactly `input_length` tokens (Qwen3 adds no BOS) | Contract: exact digests; scorer: `usage.prompt_tokens` |
| OSL | `output_length`, truncated at `max_model_len` | `max_tokens = min_tokens = OSL`, `ignore_eos`, the same truncation | Contract, and scorer `osl_mismatch` |
| Prefix structure | Interned hash ID repeated per trace block | Hash-path trie blocks: equal prefixes give equal tokens; siblings differ at their first token | Unit tests: token LCP equals replay's for every pair; engine-block (16) hash structure isomorphic |
| Session context | `SessionContext(session_id)`, except all-single-turn open loop | `X-Dynamo-Session-ID` from AIPerf's stable per-session correlation ID; off in the same case | Contract: present exactly when expected, constant within a session, distinct across sessions |
| Warm-up and window | `measure`: arrival basis in replay time, or completion basis with identity warm-up and `full_occupancy` | Same rule through `goodput.compute_metrics`; open-loop first turns keep their generated arrival | Unit test: the scorer reproduces 4 cached `lr-eval` records exactly |
| CRN | Replicate `k`, policy seed `k + 1` | Same replicate trace; the manifest carries `policy_seed` for the router policy YAML | Manifest |
| Cold start | Empty caches | A fresh `--salt` per run, so no run shares a KV block with another | Unit test (salts share no 16-token block) |

Prompt token IDs come from the Qwen3 regular vocabulary `[0, 151643)`. Qwen3-32B's `tokenizer.json`
(SHA-256 `aeb13307...dae4`, the same file as Qwen3-0.6B's) has 151,643 BPE tokens. Its 26
added/special tokens are `151643..151668`, `add_bos_token` is false, and `max_position_embeddings`
is 40,960, so 131,072 needs YaRN in the deployment.

### Known live-only residuals

The live lane does not erase these differences. It measures them per run.

- **Client timing.** Each request starts at the credit issue plus client overhead. Measured on the
  CPU contract smoke, on a host at load about 20:
  - the server saw first turns 1.9 to 5.1 ms after schedule (p50; max 9.4 ms);
  - AIPerf issued credits 0.2 to 0.7 ms after schedule (p50; max 4.2 ms).

  `score` reports this as `schedule_lateness_ms` and `start_lag_ms`.
- **Think time.** It runs from AIPerf's credit return, which follows HTTP EOF and bookkeeping; replay
  starts the delay at the final token. The residual measured at the server was -0.9 to +24 ms
  (p50 2.5 to 4.3 ms). The small negative values are consistent with event-loop timer granularity
  (a hypothesis, not verified). `score` reports it as `think_residual_ms`.
- **Latency base.** `ttft_ms` and `e2e_latency_ms` are measured from the HTTP request start, and
  include the frontend and transport. Use the live-fitted E0 to take those out of the slowdown
  reference.
- **Closed-loop handoff.** Slots are reissued after a credit returns, not at the completion
  instant. The occupancy gap is reported as `window_below_cap_frac` (0.25% in the contract smoke).
- **Router state.** Replay routes on perfectly fresh overlap and load state; live routing sees KV
  events after their real latency (A5).
- **Tie order.** The order of simultaneous arrivals cannot be enforced live. `crn-spread-v1` cells
  have essentially no ties.

## Usage

The worktree `.venv` runs everything that imports the harness (`PY`). The AIPerf environment runs
AIPerf and `loadgen_contract.py serve`.

```bash
PY=$WT/.venv/bin/python   # WT = your checkout of this branch
AIPERF_ENV=$CR/runs/live/env/aiperf-0.13.0
LIVE=$WT/benchmarks/learned_routing/live

# 1. Inputs for one (cell, k); the salt must be unique per live run.
$PY $LIVE/gen_aiperf_inputs.py cell --cells CR/cells/test.jsonl --cell-id mooncake-w3-base-n4-open-L2 \
  --k 0 --salt <run-id> --out-dir <inputs>
#    optional: --max-sessions N for a smoke subset (also writes subset_trace.jsonl for replay)

# 2. Run AIPerf against the Dynamo frontend (needs a fresh artifact directory).
$AIPERF_ENV/bin/python $LIVE/run_aiperf.py --inputs <inputs> --url http://<frontend>:8000 \
  --artifact-dir <artifacts> --aiperf $AIPERF_ENV/bin/aiperf --verify-input

# 3. Score with the AIS E0 (simulator reference) or the live-fitted E0.
$PY $LIVE/score_live.py score --inputs <inputs> --run-dir <artifacts> --policy <name> \
  --out <score.json> --rows-out <rows.jsonl.gz> [--e0 live --e0-live <e0_live.json>]

# Live-fitted idle E0: one request at a time over an ISL grid, then a fit.
$PY $LIVE/gen_aiperf_inputs.py idle --salt <run-id> --out-dir <idle-inputs>
$AIPERF_ENV/bin/python $LIVE/run_aiperf.py --inputs <idle-inputs> --url ... --artifact-dir <idle-artifacts> --aiperf $AIPERF_ENV/bin/aiperf
$PY $LIVE/score_live.py fit-e0 --inputs <idle-inputs> --run-dir <idle-artifacts> --out <e0_live.json>

# Per-request live vs replay latency ratios for the same cell and replicate.
$PY $LIVE/score_live.py compare --inputs <inputs> --live-rows <rows.jsonl.gz> --replay-per-request <per_request.jsonl.gz>
```

`score` exits with code 2 when `fidelity_ok` is false. That covers missing records, an ISL/OSL
or prompt-digest mismatch, a usage mismatch, or a wrong session header. Smoke subsets can pass
`--measure-json '{"basis": "arrival"}'`; the override is recorded in the output.

### E0 modes

- `ais` (default): the harness's `ais-chunked-estimator-v2` table, which is what the simulator
  scores against. The generator precomputes it per request into `requests.jsonl`.
- `live`: idle calibration on the deployment itself, `live-idle-v1`:
  - TTFT(ISL) is piecewise-linear through the per-ISL median idle TTFT;
  - the batch-1 decode step is `d0 + d1 * context`, fitted by least squares on per-request mean
    ITL at mean context `ISL + OSL/2 + 1`;
  - `E0 = TTFT(ISL) + sum_{j=0}^{OSL-2} step(ISL + j + 2)`, replay's decode-context convention.

### Deployment requirements these semantics rely on

These belong to the deployment recipe, but the equivalence above depends on them:

- **Engines:**
  - vLLM `--block-size 16`, prefix caching on, chunked prefill,
    `--max-num-batched-tokens 8192`, `--max-num-seqs 1024`;
  - `--max-model-len 131072` with YaRN (factor 4, original 32,768);
  - `--num-gpu-blocks-override 18863`, which pins replay's KV capacity. That capacity matters is a
    hypothesis; it is recommended, not proven necessary.
- **Router:**
  - a KV router with the policy YAML seeded `k + 1` (`manifest.replicate.policy_seed`);
  - no `--router-session-affinity-ttl-secs`: replay never pins sessions, and the session ID
    reaches policies only as `SessionContext`.
- **Run hygiene:** idle engines at the start of every run, no AIPerf warm-up phase, and one
  artifact directory per (policy, cell, k).

## Validation (2026-10-03)

The evidence summary is `CR/runs/live/contract/SUMMARY.json`.

**`verify-replay` against cached replay `per_request` rows, replicate `k = 0`.** All four cells
are exact:

| Cell | Mode | Requests | Result |
|---|---|---:|---|
| `mooncake-w0-base-n4-open-L1` | open, single-turn | 3,422 | arrival/ISL/OSL tuples bit-identical |
| `mooncake-w0-base-n8-closed-L2` | closed | 3,422 | 61 sessions at t = 0, 0 order inversions |
| `sessions-s0-base-n4-open-L2` | open, multi-turn | 14,451 | first arrivals bit-identical; 11,526 delays within 1.2e-10 ms |
| `sessions-s0-base-n8-closed-L2` | closed, multi-turn | 15,890 | 265 sessions at t = 0, 0 inversions; 12,691 delays within 1.2e-10 ms |

**CPU contract smoke.** AIPerf 0.13.0 drove the dummy server on three inputs:

- a 24-session multi-turn trace, open and closed (`C = 4`);
- the first 40 sessions of `mooncake-w0-base-n4-open-L1`.

All 154 requests were received exactly once, with exact prompts, forced OSL and the correct
session headers. There were no causality violations, and the closed loop peaked at exactly
`C = 4`. Every AIPerf export scored with `fidelity_ok` true.

**Full-size input.** AIPerf parsed a full 204 MB input (3,422 rows, 32.5 M prompt tokens) and
sent its first 5 s of schedule (`--fixed-schedule-end-offset 5000`) in 12 s of wall time, at a
peak RSS of 318 MB.

**Unit tests.** 31 pass, including the four cached-record reproductions:

```bash
cd $WT/benchmarks/learned_routing
PYTHONDONTWRITEBYTECODE=1 $PY -m pytest live/tests -q -p no:cacheprovider
```

## GPU smoke (2026-10-03)

One GPU smoke job (8 x H100 SXM, N = 4 x TP2, `default@defaults` seed 1) ran the full
`mooncake-w2-base-n4-open-L2` replicate `k = 0` (4,002 requests) through `smoke_payload.sh`, then
the idle E0 calibration. `smoke_compare.py` against the cached replay of the same cell and replicate:

- **Request fidelity:** every request matched, with exact prompts, ISL and forced OSL. One OSL = 1
  request whose only token detokenized to empty text was counted as an AIPerf error.
- **Schedule:** credit lateness p50 1.4 ms, p99 2.5 ms, max 9.4 ms, flat over the run.
- **Goodput (window, AIS E0):** 3.4454 live vs 3.4535 replay (2,113 vs 2,118 good of 2,376).
- **Prefix cache:** 0.3599 live (usage and vLLM counters agree) vs 0.3589 replay.
- **Latency:** E2E distributions agree (KS 0.022). Live TTFT is higher (p50 176 vs 130 ms) and live
  ITL lower (p50 22.9 vs 25.6 ms). The idle calibration shows why: the real batch-1 decode step is
  7 to 10% faster than AIS, and real prefill 5 to 10% slower, plus about 17 ms of HTTP path.

Details and the finalist cost model: `CR/facts/live_smoke.json` (mirrored, sanitized, under
`notes/learned-routing/campaign/facts/` when synced).

## Loadgen parity audit (2026-10-03)

An independent audit, with no GPU. Results are in `CR/facts/live_loadgen_parity.json`, and the
evidence is in `CR/runs/live/audit-parity/` (`SUMMARY.json`, `static/*/report.json`,
`stub/*/check.json`).

**`static` audit: 13 cells, all exact.**
- **Coverage:**
  - 9 train/val cells against cached replay rows of 2 policies each: Mooncake open and closed
    (including the osl2.0, root2 and islu1.25 transforms) and synthetic sessions open and closed
    (including think1.5, think2.0 and root2), at N = 4, 6 and 8;
  - 4 FAST25 test cells against the replay-source reference only, so no test cell was replayed.
- **Request identity:** request set and order, bit-identical open-loop arrivals, exact ISL and OSL
  targets.
- **Prefix structure:**
  - 2,250 to 3,000 sampled pairs per cell, where the token LCP equals `min(ISL_a, ISL_b, m * B)`
    and equals replay's token LCP;
  - a bijection between the live and replay 16-token chained block hashes over every full block.
- **Sessions:** think delays are bit-equal to the trace and match replay's gaps to 2.4e-10 ms.
- **Admission:**
  - The session header follows exactly the rule by which replay emits `SessionContext`.
  - Closed loop: replay starts session `C + m` exactly at the m-th session departure.
- **Scoring:** the live scorer reproduces every replay record's window, warm-up exclusions and
  good counts exactly.

**Stub runs: AIPerf 0.13.0 with the manifest argv, against `audit_stub_server.py`.**
- **Runs:** 4 full cells, 45,462 requests.
- **Delivery:** every request arrived exactly once, with the exact prompt, forced OSL and the
  correct session header.
- **Order:** AIPerf issued credits in replay's order, with 0 inversions.
- **Closed loop:** concurrency peaked at exactly `C`.
- **Live-only residuals,** on a shared workstation:
  - open-loop first-turn lateness: p50 2 ms, p99 11 ms;
  - think-time residual: p50 2.4 to 2.7 ms, p99 12 to 14 ms;
  - closed-loop slot handoff: p50 2.5 ms, p99 11 ms;
  - accumulated drift behind replay arrivals: up to 0.56 s;
  - transport reordering of near-simultaneous starts: up to 59 ms.
- **Metric effect:**
  - Window request counts matched replay in all 4 runs.
  - Good counts matched in 3 of 4 runs. In the fourth, 3 of 5,808 flipped at the E2E-slowdown
    boundary, because the client-side latency base adds about 1 to 2 ms.
  - `goodput_rps_window` moved by at most 0.06%.
