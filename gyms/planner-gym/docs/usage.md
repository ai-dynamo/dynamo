<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Planner Gym Usage Guide

Run these commands from `dynamo/gyms/planner-gym` after following the
[getting started guide](getting-started.md). For scoring definitions and
limitations, see the [Planner Gym README](../README.md).

## Build a Golden Set from an external base trace

[`configs/golden-set.example.yaml`](../configs/golden-set.example.yaml) is a
portable recipe for schedule and request-composition experiments. It defines
steady load, step/recovery, ramps, repeated cycles, composition shifts, and
seeded random bursts without naming or locating a source dataset. The source
path is supplied only when the builder runs, so a private or separately
licensed collection does not need to be copied into this repository or named
in a committed config.

Run the builder from `dynamo/gyms/planner-gym` and choose a new output directory:

```bash
python3 scripts/build_golden_set.py configs/golden-set.example.yaml \
  --source /absolute/path/to/base-trace \
  --source-format auto \
  --reference-rps 1.0 \
  --seed 7 \
  --max-source-requests 100000 \
  --max-source-hashes 5000000 \
  --out runs/golden-set-rps1-seed7
```

`--source` accepts a JSONL/JSONL-gzip file or a recursively discovered
directory and may be repeated. `auto` detects either of the two supported
input formats:

- `dynamo_request_trace_v1` contains `dynamo.request.trace.v1` event rows. Its
  replay block size is embedded, so omit `--source-block-size` unless you want
  an explicit consistency check.
- `replay_jsonl` contains `timestamp`, `input_length`, `output_length`, and
  `hash_ids` in each row. Its block size is not encoded, so
  `--source-block-size` is required.

One invocation must describe a homogeneous collection: every discovered shard
must use the same format and one block size. Point `--source` at one logically
coherent model/tokenizer traffic collection; that identity is not inferable
from the replay fields and therefore cannot be checked by the builder. Mixed
formats and mixed block sizes are rejected. `--max-source-requests` and
`--max-source-hashes` are independent request-count and loaded-prefix-reference
guards, not byte-precise memory limits or sampling limits. Narrow the source or
raise them deliberately when the collection is larger. Supplying `0` disables
the corresponding guard.

`--reference-rps` gives meaning to the recipe's `load_factor`: a factor of
`1.0` has that expected offered rate under the Poisson process (the
deterministic process targets the corresponding integrated count). Calibrate
it for the fixed model, engine, hardware, fleet, request mix, and SLO that
define the comparison—for example, from a steady-load capacity run—and keep
the value constant across autoscalers. Recalibrate or version the Golden Set
when that substrate changes. Equal RPS does not imply equal GPU work when a
phase selects different request-length quantiles.

Selection is deterministic for a seed and occurs without replacement within
each generated workload. When the input exposes session IDs and the recipe
uses `session_policy: preserve`, requests from an input-scoped session stay
together, retain their order, and are not split across schedule phases.
Original inter-turn gaps are not retained. A partial source directory can only
preserve the part of a session present there, and rows without a session ID are
treated as independent requests. This is atomic sampling of request
composition, not closed-loop conversational replay: the generated files omit
request and session IDs so every materialized timestamp remains an open-loop
arrival. A future agentic recipe should model completion dependencies and tool
delays explicitly rather than infer them from sparse session labels. Prefix
hashes are compactly remapped while preserving equality relationships within
each generated workload. If two differently named composition pools select
exactly the same request groups for a source, construction fails instead of
silently producing a no-op composition shift. Phase medians in the manifest
show how strongly the selected request mixes differ.

The fresh output directory contains one timestamp-sorted replay JSONL file per
workload, a path-redacted `manifest.json`, and
`match-config.fragment.yaml`. The manifest records the recipe and source
fingerprints, seed, reference RPS, block size, filtering/session coverage,
phase schedules and counts, composition summaries, and output checksums. The
fragment is directly usable only when the complete Match Config is saved beside
the generated traces. If its `evaluations` mapping is merged into a config
elsewhere, rewrite each trace path relative to that Match Config (or use an
absolute path). Generated traces belong under ignored `runs/` or another
external artifact location, not in source control.

`match-config.fragment.yaml` is not a runnable config by itself. For a complete
two-policy example, start from
[`configs/match.golden-set.quickstart.yaml`](../configs/match.golden-set.quickstart.yaml)
and replace its `evaluations` section with the generated fragment. Save the
complete config beside the generated traces, or link the traces into the
config's directory, so the fragment's relative paths continue to resolve.

## Bring your own recorded trace

Recorded traffic stays outside this repository. Match Configs accept one or
more user-supplied replay JSONL files and resolve relative paths from the
config file:

```yaml
evaluations:
  workloads: [flat]
  traces:
    - name: my-recorded-traffic
      path: ../../../private/traces/my-traffic.jsonl
      block_size: 512
      presorted: true
  defaults:
    max_requests: 1000
    arrival_speedup: 1.0
```

Each JSONL row must contain `timestamp`, `input_length`, `output_length`, and
`hash_ids`. Set `block_size` to the token span represented by each hash. Use
`presorted: true` for timestamp-sorted large traces so capped runs can stream
the first records without loading the whole file. The materialized trace is
fingerprinted in result provenance. External traces and transformed copies are
staged under a private system-temporary directory and deleted after execution;
their source paths are redacted from reports.

Simulation Match Configs also accept exact native Dynamo request-trace shards:

```yaml
evaluations:
  traces:
    - name: recorded-requests
      format: dynamo
      paths:
        - ../../../private/traces/requests-00.jsonl.gz
        - ../../../private/traces/requests-01.jsonl.gz
      # block_size: 16  # optional; omission uses embedded trace metadata
  defaults:
    max_requests: null
    arrival_speedup: 1.0
```

The ordered shard list is passed directly to Dynamo without sorting, merging,
conversion, capping, or rewriting. Exact native traces reject `max_requests`
and non-1 `arrival_speedup`; they are simulation-only because the real backend
drives AIPerf with Mooncake JSONL.

For a single simulation, pass the file directly:

```bash
python scripts/run_match.py \
  --trace /absolute/path/to/my-traffic.jsonl \
  --trace-block-size 512
```

## User guide: run a Match Config

A Match Config is a versioned YAML description of a complete comparison
matrix. It selects either the `sim` or `real` backend, the autoscalers or
endpoint names, model and engine configuration, recorded and synthetic
workloads, SLO targets, metrics, execution safeguards, and publication
destinations. Start from
[`configs/match.sim.example.yaml`](../configs/match.sim.example.yaml) or
[`configs/match.real.example.yaml`](../configs/match.real.example.yaml).
The simulation example intentionally expands to a broad matrix; use the
[two-policy quickstart](getting-started.md) for a small first run.

Simulation model and engine settings are first-class and control the replay:

```yaml
backend:
  type: sim
  topology: disagg
  gpu_budget: 32

  model:
    name: openai/gpt-oss-120b
    # Optional; defaults to name.
    ais_model_path: openai/gpt-oss-120b

  engines:
    common:
      system: h200_sxm
      backend: vllm
      backend_version: "0.19.0"
      tp_size: 1
      moe_tp_size: 1
      moe_ep_size: 1
      attention_dp_size: 1
      runtime:
        cold_start_delay_s: 30
      extra_args: {}
    # Optional partial overrides inherited from common.
    prefill:
      runtime:
        kv_transfer_bandwidth_gbps: 100
        kv_bytes_per_token: 131072
    decode: {}

  replay:
    # Simulated-time cadence; defaults to 5 seconds. null disables capture.
    telemetry_sample_interval_s: 5
```

Telemetry sampling has its own simulated-time schedule and does not need to be
a multiple of a Planner or rival autoscaler's tick duration. A positive value
can therefore sample every few seconds or minutes without changing controller
behavior. The default is 5 seconds so Match Config reports have useful charts;
set `telemetry_sample_interval_s: null` when the extra persisted observations
are not wanted. Every enabled simulation cell writes
`runs/<run-id>/telemetry.jsonl` beneath that session's configured artifact
root. The file contains one complete Dynamo telemetry snapshot per line and is
the only replay-telemetry ingestion boundary used by Arena reporting.

For an aggregated topology, use `engines.aggregate` instead of
`engines.prefill`/`engines.decode`. Engine GPU cost is derived as
`tp_size * attention_dp_size`; MoE TP and EP, when supplied, must describe the
same world size. GPU-budget validation weights prefill and decode replica
counts by their independently resolved engine costs. An engine's `backend`
selects DynoSim's scheduler semantics. AISimulate performance lookup defaults to that
backend/version but can be decoupled explicitly with `ais_backend` and
`ais_backend_version`; identity and accounting fields cannot be overridden
through `extra_args`. `cold_start_delay_s` models worker startup. KV handoff
delay requires `kv_transfer_bandwidth_gbps` (GB/s) and `kv_bytes_per_token` together;
for a disaggregated deployment they normally belong on the prefill engine.
Replace the example bytes/token value with the model and KV-cache dtype being
simulated. Other low-level engine runtime behavior remains available through
`extra_args`; the three typed runtime fields cannot be duplicated there.

The compact legacy-compatible form `substrate: gpt_oss` remains available and
resolves to the same complete model and role-engine snapshot. A config must
choose either the preset shorthand or explicit `model` plus `engines`, so
there is never a hidden override order.

Validate a config and inspect its fully expanded matrix before starting work:

```bash
python scripts/run_match_config.py configs/match.sim.example.yaml \
  --validate-only \
  --print-matrix
```

Run the simulation example from the Dynamo environment built above:

```bash
python scripts/run_match_config.py \
  configs/match.sim.example.yaml
```

For a real match, first replace the example endpoint catalog entries with
already-running deployments, then run:

```bash
python scripts/run_match_config.py configs/match.real.example.yaml
```

The nested real deployment metadata is easiest to read in
[`configs/endpoints.example.yaml`](../configs/endpoints.example.yaml); JSON
and YAML endpoint catalogs are accepted by both Match Configs and the endpoint CLI.

All relative paths—including `backend.planner_config`, `endpoint_catalog`,
`publish.artifact_root`, and JSON/HTML destination paths—resolve from the Match
Config's directory. Real autoscaler entries refer to endpoint catalog entries
by `name`; the runner
benchmarks those deployments but does not create or reconfigure them. Pass
`--overwrite` explicitly to replace an existing JSON or HTML destination.
Optional `autoscaler_type` and `declared_config` fields on a real entry are
recorded as controller provenance only. Each real endpoint's `model` is the
operational value passed to AIPerf. Its optional `declared_deployment` block
records unverified model, topology, and engine provenance; Arena does not
apply, discover, or verify that deployment configuration.

| Setting | Authority |
| --- | --- |
| Sim `backend.model` and `backend.engines` | Controls DynoSim |
| Sim `backend.planner_config` | Planner policy from a reusable file or inline mapping |
| Real endpoint `model` | Passed to AIPerf `--model` |
| Real endpoint `declared_deployment` | Unverified deployment provenance only |
| Real autoscaler `declared_config` | Unverified controller-policy provenance only |

Simulation configs can also pin `gpu_budget`, `router.mode`, Planner-specific
settings as an inline `planner_config` mapping or a reusable YAML/JSON path,
and replay controls (`ais_bootstrap` and `concurrency`). See
[`configs/planner.sim.example.yaml`](../configs/planner.sim.example.yaml) for a
reusable Planner policy that deliberately omits runner-owned topology, budget,
engine-cost, and report fields. Preset-based legacy configs may still use
`replay.model_name`; explicit configs must use `model.name`. Real configs can
set AIPerf's `executable`, `tokenizer`, `streaming`, `timeout_s`, and
`extra_args`. Optional top-level
`description` and string-valued `labels` travel with the resolved config and
run provenance in the result JSON.

`evaluations.workloads` selects registry entries directly,
`evaluations.traces` declares external recorded files, and
`evaluations.suites` adds `synthetic`, `recorded`, or `all`. Suite names are
also accepted directly in the workload list.
`evaluations.exclude` is applied after both are combined. Defaults such as
`seed`, `max_requests`, and `arrival_speedup` apply to every selected workload;
`evaluations.overrides` changes them for one workload.

An SLO threshold can be a scalar or a list. Lists within one profile expand as
a Cartesian product: `ttft_ms: [300, 500]` with `itl_ms: [50, 100]` produces
four targets. The complete run count is:

```text
autoscalers x workloads x expanded SLO targets x repetitions
```

`execution.max_runs` rejects an unexpectedly large matrix before execution,
and `fail_fast` controls whether a failed cell stops the remaining sweep.
Repetition zero uses the configured evaluation seed; later repetitions use
`seed + repetition`.
Static, reactive, and KEDA entries can set their polling and policy controls
directly. Reactive and KEDA configs include explicit `min_*`/`max_*` replica
bounds; KEDA also exposes thresholds, tolerance, and stabilization windows.
Omitted adapter settings are expanded to their effective defaults in the
resolved config, and maximum fleets are checked against `gpu_budget`.
`metrics.rank_by` chooses the leaderboard order; `metrics.include` selects the
reported context columns. Publication supports console, JSON, and standalone
HTML destinations, with raw diagnostics under `publish.artifact_root`:

```yaml
publish:
  artifact_root: ../runs/my-match/artifacts
  destinations:
    - type: console
    - type: json
      path: ../runs/my-match/results.json
    - type: html
      path: ../runs/my-match/report.html
```

The HTML report is one self-contained file per Match Config session. Its
leaderboard and eight aligned time lanes update together when you select an
expanded SLO/repetition configuration or workload. A shared checkbox legend
shows or hides each autoscaler across queue depth, TTFT, TPOT, KV utilization,
KV hit/cache reuse, active replicas, and provisioned GPUs. Arriving requests
come from the common workload trace or the replay's telemetry windows and are
shown once because every autoscaler receives the same open-loop traffic. TTFT
and TPOT are raw means from replay-owned windows, independent of policy-specific
observation handling; hover exposes the exact window bounds and saved
latency-sample count. The report timeline is projected from each run's persisted
`telemetry.jsonl`, and each saved result row retains the full traffic aggregate
and raw rank-level scheduler rows so additional charts can be derived without
rerunning the trace. For vLLM, the KV lane
distinguishes active-block pressure from physical residency, which also
includes inactive reusable blocks.
SGLang's legacy occupancy already includes all occupied radix pages, so its
active and physical lines are equal until the backend exposes that split. The
cache lane deliberately shows two different signals: router KV hit rate is a
request-weighted routing-overlap estimate, while scheduler cache reuse is the
backend-native ratio of interval hit tokens to observed cache tokens. They are
not substituted for each other. Older runs can fall back to a saved
completion-attributed timeline or a labeled whole-run aggregate. Capacity lines
show fleet observations at telemetry sample times, while hollow markers use
Planner details to show exact capacity requests at their controller decision
times. Provisioned GPUs include starting and draining replicas and weight
prefill/decode pools by their configured GPUs per worker.

Queue depth is the point-in-time sum of scheduler-waiting requests across all
live ranks plus router-pending requests. The total, scheduler, and router layers
are retained separately, as are the raw rank-level scheduler rows used to
derive them. Scheduled/in-flight work is excluded from queue depth. Legacy
artifacts retain their scheduler-only queue values and explicitly mark router
pending and the new KV time series unavailable instead of inferring zeros.

Open-loop simulation reports show the actual replay arrival schedule. A
closed-loop simulation ignores trace timestamps, so the report marks arrivals
unavailable rather than presenting the trace schedule as observed load. Online
matches retain the leaderboard, while replica/GPU lanes remain unavailable
until deployment telemetry is supplied.

Validation is strict: unknown fields, duplicate YAML keys or names, unsafe
path-bearing identifiers, unsupported backend metrics, missing endpoints, and
invalid replica/SLA ranges are rejected before a run begins.

Simulation runs enable per-request capture. Dynamo returns terminal request
records in `ReplayReport.per_request`, exposed through
`ArenaReplayResult.per_request` for SLO scoring. A sim Match Config still
performs one deterministic replay per expanded SLO target so the configured
runtime SLA, result identity, and artifact set remain aligned. The real backend
likewise runs one AIPerf process per endpoint, workload, and SLO target.

Saved results can be rendered again without replaying traffic:

```bash
python scripts/render_saved_report.py \
  /path/to/results.json \
  --out /path/to/report.html \
  --normalized-out /path/to/normalized-report.json
```

The renderer accepts the normalized `results.json` produced by a Match Config
run. It uses the saved timeline and metrics without reopening source shards,
rereading telemetry JSONL, or invoking Dynamo. The raw JSONL remains in the session artifact tree for audit
and future projections.

## User guide: run one simulated match

`run_match.py` is the quickest simulation smoke test. Its CLI currently exposes
the `static` and `reactive` adapters.

```bash
export TRACE_DIR=../../traces

python scripts/run_match.py \
  --autoscaler static \
  --trace "$TRACE_DIR/mooncake/1000.jsonl" \
  --num-prefill 4 \
  --num-decode 1 \
  --ttft-ms 2000 \
  --itl-ms 50 \
  --report-json runs/static-4p1d-trace-report.json
```

The terminal summary reports mean/p99 TTFT, mean ITL, request throughput,
simulation ticks, and scaling-event count. Compare against the reactive policy
by changing `--autoscaler static` to `--autoscaler reactive`.

Useful options:

- `--trace`: any replay JSONL trace with the fields described below.
- `--substrate`: select a bundled model/engine/hardware preset.
- `--num-prefill`, `--num-decode`: initial worker counts.
- `--ttft-ms`, `--itl-ms`: populate the substrate `PlannerConfig` and
  diagnostics. The current static/reactive adapters ignore them, so these flags
  do not change their decisions or redefine scorecard profiles.
- `--report-json`: write the AIPerf-style trace report.

## User guide: run the offline leaderboard

The headline command sweeps the default roster—Planner, KEDA port, reactive,
and fixed `static-4P4D`—over the requested workloads:

```bash
export DYNAMO_DIR=/absolute/path/to/dynamo
source .venv/bin/activate

python scripts/run_leaderboard.py \
  --workloads staircase flash_crowd mooncake \
  --profile interactive \
  --substrate gpt_oss \
  --seed 0 \
  --out runs/my-leaderboard.json
```

This example runs 4 policies x 3 workloads = 12 matches. Use
`--workloads all` for all eight workloads (32 matches).

Dynamic policies start at `1P1D` and must earn additional capacity. The static
baseline starts and remains at `4P4D`. The supplied leaderboard script uses a
disaggregated topology, TP=1 engine arguments, one GPU per engine, and a
round-robin router. Treat those settings as part of the benchmark definition
when comparing result files.

The table is ranked independently for each workload by SLO-qualified requests
per second per time-averaged allocated GPU, followed by an overall mean of that
metric. `GPU-h` remains in the table as cumulative allocation context; `avg_GPU`
is the efficiency denominator. The full scorecard for every row is written to
JSON.

## User guide: benchmark live endpoints

### 1. Describe the deployments

Copy the example and replace its values with endpoints that are already
running:

```bash
cp configs/endpoints.example.yaml configs/endpoints.local.yaml
```

Each entry has three required operational fields and optional metadata:

```json
{
  "endpoints": [
    {
      "name": "planner",
      "url": "http://planner.example:8000",
      "model": "Qwen/Qwen3-8B",
      "endpoint_type": "chat",
      "description": "Dynamo Planner deployment",
      "declared_deployment": {
        "topology": "disagg",
        "model": {"name": "Qwen/Qwen3-8B"},
        "engines": {
          "common": {
            "backend": "sglang",
            "backend_version": "0.5.10",
            "system": "h200_sxm",
            "tp_size": 1
          }
        }
      }
    }
  ]
}
```

- `name` must identify the result/artifact directory.
- `url` is the URL AIPerf should target.
- `model` must match the model served by that deployment.
- `endpoint_type` defaults to `chat`.
- `description` is optional metadata.
- `declared_deployment` is optional, unverified provenance retained by Match
  Config runs. The legacy endpoint CLI ignores it.

A bare JSON list of endpoint objects is also accepted.

### 2. Run a small matrix first

```bash
python scripts/run_endpoint_bench.py \
  --endpoints configs/endpoints.local.yaml \
  --workloads flat staircase \
  --profiles interactive relaxed \
  --tokenizer gpt2 \
  --timeout-s 900 \
  --artifact-root runs/online \
  --out runs/online-leaderboard.json
```

Replace `gpt2` with the tokenizer appropriate for the served model, or omit
`--tokenizer` when AIPerf can resolve it.

The runner executes one AIPerf process for every endpoint x workload x profile
combination, passing each workload's block size to preserve its prefix hashes.
It is serial, and AIPerf follows each trace's timestamps in real time. A synthetic trace lasts 180 seconds, so the example above is four
three-minute runs per endpoint, plus startup and completion overhead.

Use exact profile names: `interactive`, `agentic`, or `relaxed`. Use
`--workloads all` only after a small matrix succeeds.

### 3. Inspect failures and artifacts

The console leaderboard ranks endpoints by AIPerf goodput for each workload.
The result JSON retains the subprocess return code, the last stderr lines, and
the raw artifact directory for every run. A failed run is marked `[FAILED
rc=N]`; inspect those fields before comparing scores. A timeout is recorded as
return code 124 and a missing AIPerf executable as 127. The sweep continues
and saves all results before exiting with code 1 if any run failed.
`--timeout-s` bounds each AIPerf process; omit it to wait without a time limit.

The live leaderboard does **not** calculate goodput/GPU or scaling
oscillations because the endpoint runner does not collect deployment cost or
controller telemetry.

## Workload catalog

The seven synthetic workloads are deterministic for a given seed and run for
180 seconds. Request counts vary with the Poisson draw and are approximately
1,000 at seed 0. Input/output lengths are sampled rather than fixed.

| Workload | Arrival pattern | Shape/prefix | What it stresses |
| --- | --- | --- | --- |
| `mooncake` | Recorded | Real prefill-heavy trace and prefix reuse | Anchor against a real trace shipped with Dynamo |
| `flat` | Constant 5.6 req/s | Prefill-heavy, no sharing | Steady-state efficiency and unnecessary oscillation |
| `staircase` | 1 req/s, +1 every 20 s | Prefill-heavy, no sharing | Provisioning lag under sustained growth |
| `square_wave` | 1 <-> 10 req/s, 40 s period | Prefill-heavy, no sharing | Thrash and stabilization behavior |
| `flash_crowd` | 3 req/s, then 30 req/s from 80-95 s | Prefill-heavy, no sharing | Burst response and recovery |
| `diurnal` | Sinusoid, mean 5.6 and amplitude 4 req/s | Prefill-heavy, no sharing | Gradual scale-up and scale-down discipline |
| `decode_heavy` | Staircase | Median ~200 input / ~600 output tokens, no sharing | Prefill-versus-decode capacity decisions |
| `shared_prefix` | Constant 5.6 req/s | Median ~1000 input / ~200 output tokens; up to eight shared 512-token blocks, bounded by input length | Prefix-cache reuse |

Every generated record follows the replay JSONL shape:

```json
{"timestamp": 1250, "input_length": 1024, "output_length": 128, "hash_ids": [0, 1]}
```

`timestamp` is in milliseconds. `hash_ids` describes block-level prefix
identity; generated workloads use a 512-token block size.

## Understand the outputs

| Path | Producer | Contents |
| --- | --- | --- |
| `runs/traces/*.jsonl` | Trace generator and leaderboard scripts | Materialized synthetic traces |
| `runs/leaderboard.json` | Offline leaderboard default | One full scorecard per policy/workload pair |
| `planner_reports/arena_<policy>.html` and `.log.jsonl.gz` | Single simulated match | Dynamo diagnostics, relative to the shell working directory |
| `planner_reports/_leaderboard_run.html` and `.log.jsonl.gz` | Offline sweep | Dynamo diagnostics; the fixed filename is overwritten across matches |
| `runs/online/<endpoint>/<workload>_<profile>/` | Online endpoint runner | Raw nested AIPerf artifacts |
| `runs/online_leaderboard.json` | Online endpoint runner default | Parsed metrics, return codes, artifact paths, and stderr tails |
| Configured `publish.artifact_root/<session>/` | Match Config runner | Generated synthetic traces, per-run diagnostics, raw `runs/<run-id>/telemetry.jsonl` streams, and AIPerf artifacts; external traces use ephemeral private staging |
| Configured JSON destination | Match Config runner | Resolved config, expanded matrix, provenance, normalized metrics, raw metrics, and failures |
| Configured HTML destination | Match Config runner | Standalone interactive leaderboard and aligned workload/latency/capacity timelines |

## Extend the Arena

For the experimental TypeSafe Jev decision engine and its five-policy synthetic
pilot, see [the Jev guide](jev.md). It uses the same Planner engine interface
and reports hosted-controller overhead separately from simulation metrics.

### Create a custom synthetic workload

Compose an arrival process, request shape, and prefix policy:

```python
from pathlib import Path

from autoscaling_arena.workloads import Workload, axes

workload = Workload(
    name="my_burst",
    description="A 30-second shared-prefix burst",
    duration_s=30,
    arrival=axes.square_wave(low=1, high=8, period_s=10),
    shape=axes.balanced,
    prefix_factory=lambda: axes.SharedPrefix(shared_blocks=4),
)

trace = workload.materialize(Path("runs/traces"), seed=7)
print(trace)
```

Pass that file to `run_match.py --trace`. To make a name available to
`run_leaderboard.py` and `run_endpoint_bench.py`, add the `Workload` to
`WORKLOADS` in `src/autoscaling_arena/workloads/registry.py`.

For a recorded workload, construct `Workload(name=..., description=...,
static_trace=Path(...))` instead.

### Add an offline autoscaler

An adapter implements three methods:

1. `initial_tick(start_s)` schedules its first observation.
2. `async tick(scheduled_tick, tick_input)` returns `PlannerEffects`, including
   the desired replica counts and next tick.
3. `async shutdown()` releases policy resources.

Register its factory in an `AutoscalerSpec`:

```python
from autoscaling_arena.leaderboard import AutoscalerSpec

spec = AutoscalerSpec(
    name="my-policy",
    factory=lambda planner_config, capabilities: MyAutoscaler(capabilities),
    start_prefill=1,
    start_decode=1,
)
```

The factory receives the shared `PlannerConfig` and derived
`WorkerCapabilities`. Use `run_leaderboard(autoscalers=[spec], ...)` from a
Python driver, or add the policy to `default_autoscalers()`. Do not encode model
or hardware behavior in the adapter; that belongs to the shared substrate.

### Score a custom SLO

```python
from autoscaling_arena.scorecard import SLOProfile, scorecard

chat = SLOProfile(name="chat", ttft_ms=500, itl_ms=75)
result = scorecard(report, profiles=(chat,))
```

For live runs, pass custom `SLOProfile` objects to the Python
`run_endpoint_leaderboard()` API. The supplied endpoint CLI only selects from
the three built-in profiles.

### Main Python modules

| Module | Purpose |
| --- | --- |
| `autoscaling_arena.workloads` | Workload registry, axes, and trace materialization |
| `autoscaling_arena.datasets` | Generic Dynamo request-trace validation and shard discovery |
| `autoscaling_arena.adapters` | Static, reactive, and KEDA/HPA policy adapters |
| `autoscaling_arena.runners.sims` | One DynoSim match via `run_arena_replay()` |
| `autoscaling_arena.runners.real` | Endpoint loading, AIPerf execution, and result parsing |
| `autoscaling_arena.scorecard` | SLO goodput, efficiency, and stability metrics |
| `autoscaling_arena.leaderboard` | Offline sweep orchestration and table formatting |
| `autoscaling_arena.match_config` | Strict YAML schema, validation, and matrix expansion |
| `autoscaling_arena.match_runner` | Config-driven sim/real execution and publication |
| `autoscaling_arena.html_report` | Standalone interactive leaderboard and time-series report rendering |

## Development and tests

Install the test extra:

```bash
python -m pip install -e '.[test]'
```

The scorecard tests are standalone:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests/test_scorecard.py
```

Run the pure-Python suite without writing caches:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q
```

## Troubleshooting

### `ModuleNotFoundError: dynamo` or `dynamo._core`

The offline scripts are running under the wrong Python environment or the
Dynamo bindings have not been built. Activate that Dynamo virtual environment
and verify the import command in
[simulation setup](getting-started.md). Installing
`autoscaling-arena` alone does not install Dynamo.

### `unexpected keyword argument 'telemetry_jsonl_path'`

The active Dynamo binding predates the replay telemetry bridge, or Python is
loading a binding built from a different checkout. Rebuild the native runtime from the same Dynamo checkout and run the capability check in
[Replay integration](replay-integration.md). Dynamo #12334 alone does not
provide the telemetry-file contract. Note that the public
`dynamo.replay.run_trace_replay()` API exposes `telemetry_options`; the direct
file parameters belong to the low-level `_core.run_mocker_trace_replay()`
binding.

### `static trace for 'mooncake' missing`

`gen_traces.py` and the registry resolve the anchor from
`$DYNAMO_DIR/lib/bench/testdata/mooncake_trace_1000.jsonl`. Set `DYNAMO_DIR` to
the correct absolute path and use `gen_traces.py --include-recorded`.

### AISimulate cannot resolve the default model/system/backend

Verify that the installed AISimulate performance data supports the selected
model, system, and backend version. Use the AISimulate version pinned by the
same Dynamo checkout. For legacy simulation commands, select a bundled
combination with `--substrate`. For Match Config runs, set the first-class
`backend.model` and `backend.engines` fields.

### `aiperf` is not found

Install AIPerf in the active environment, or provide the executable explicitly:

```bash
python scripts/run_endpoint_bench.py \
  --aiperf-bin /absolute/path/to/aiperf \
  --endpoints configs/endpoints.local.yaml
```

### An endpoint row has null metrics or `[FAILED rc=N]`

Open the result JSON and inspect `returncode`, `stderr_tail`, and `artifact_dir`.
Confirm that the URL is reachable, the model name and endpoint type are
correct, and the tokenizer is available. Use a fresh `--artifact-root` while
debugging so a previous summary cannot be mistaken for the current run.

### A live sweep takes longer than expected

Online replays run serially and in real time. Estimate the lower bound as:

```text
endpoints x workloads x profiles x trace duration
```

Most synthetic traces are 180 seconds. Start with one endpoint, one workload,
and one profile.

## Current limitations

- The built-in roster is Planner, KEDA/HPA port, reactive, and static. Ray
  Serve, llm-d, and an offline oracle are not implemented.
- The online runner has no deployment, first-class authentication/header,
  parallelism, cost-telemetry, or controller-telemetry layer; Match Config
  `aiperf.extra_args` can pass supported AIPerf authentication flags.
- Match Config results carry resolved provenance in JSON and can be rendered as
  standalone HTML, but there is no persistent run database or cross-run query
  layer.
- The legacy offline leaderboard uses fixed diagnostic filenames that are
  overwritten during a sweep; Match Config runs use isolated session/run paths.
- The legacy offline and endpoint CLIs accept registry workload names; use a
  Match Config for a matrix containing user-supplied recorded traces.
- Golden Set construction currently materializes its coherent source scope in
  memory. `--max-source-requests` and `--max-source-hashes` abort instead of
  sampling, so pre-scope very large inputs to one model/tokenizer, block size,
  and useful time window.
