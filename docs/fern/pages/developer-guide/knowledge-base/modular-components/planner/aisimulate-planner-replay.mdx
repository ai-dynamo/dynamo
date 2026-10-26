---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Simulate Planner Decisions with AISimulate
sidebar-title: Simulate Planner Decisions (AISimulate)
subtitle: Replay a trace through the Planner offline to compare topologies, scaling settings, startup delays, and experimental retune
---

`aisimulate predict --stack dynamo` replays a request trace through simulated engines with the
Dynamo Planner in the loop. Use it to compare aggregated and disaggregated topologies, service-level
objective (SLO) targets, scaling settings, worker startup delays, and
[retune](#simulate-retune-experimental) without a cluster. The `dynamo.replay` Python API runs the
same simulation from a script.

<Note>
The production Planner scales Kubernetes or Global Planner deployments; it does not autoscale a
local deployment. `aisimulate predict --stack dynamo` runs locally but evaluates Planner decisions
for the deployment described by its YAML input.
</Note>

For the general simulation workflow, see
[Run a Simulation](../../../../cli/operations/simulation-with-dynosim/dynosim-replay.mdx). For every
`aisimulate predict` option and configuration section, see the
[replay CLI reference](../../../../reference/components/dynosim-replay-cli-reference.mdx). For how
replay feeds the Planner its observations, see the
[Planner simulation adapter](../../concepts/simulation/dynosim-architecture.md#planner-simulation-adapter).

## Prerequisites

- Install Dynamo, AISimulate, and the Planner dependencies from one checkout, as described in
  [Install](../ai-simulate-experimental/sweeper-experimental/dynamo-integration.md#install).
  AISimulate supports Python 3.11 through 3.13. That procedure builds the bindings in release mode,
  which matters because simulation runs are CPU-bound.
- AISimulate reads each model's Hugging Face configuration on first use, for
  `engine.context_length: max` and the default timing model. The host needs Hugging Face Hub
  access or the configuration in its local Hugging Face cache.

## Configure the Planner Section

Add a `planner` section to the prediction config. This section runs the Planner with its
production defaults:

```yaml
planner: {policy: enabled}
```

`policy` defaults to `disabled`, which replays the deployment with fixed replicas and writes
`planner: null` to the report. With `enabled`, the Dynamo Planner adapter builds a `PlannerConfig`
from the section and validates it as a live Planner would. Omitted keys take the defaults listed in
the [Planner Configuration reference](../../../../reference/components/planner-configuration.mdx).

The section uses AISimulate key names. These differ from the `PlannerConfig` field names:

| `planner` key | `PlannerConfig` field |
|---|---|
| `target` | `optimization_target` |
| `max_num_gpus` | `max_gpu_budget` |
| `min_num_gpus` | `min_gpu_budget` |
| `min_workers` | `min_endpoint` |
| `prefill_min_workers`, `decode_min_workers` | `prefill_min_endpoint`, `decode_min_endpoint`; disaggregated engines only |

The other accepted keys keep their `PlannerConfig` names: `enable_throughput_scaling`,
`enable_load_scaling`, `throughput_adjustment_interval_seconds`,
`load_adjustment_interval_seconds`, `max_throughput_scaling_replicas`, `max_num_fpm_samples`,
`fpm_sample_bucket_size`, `load_scaling_down_sensitivity`, `load_min_observations`,
`load_predictor`, `load_predictor_log1p`, `prophet_window_size`, `kalman_q_level`,
`kalman_q_trend`, `kalman_r`, `kalman_min_points`, `load_predictor_warmup_trace`,
[`advisory`](#replay-in-advisory-mode), and [`retune`](#simulate-retune-experimental). Any other
key fails with `Extra inputs are not permitted`. `load_predictor_warmup_trace` names a file, a
directory of JSONL or JSONL.GZ shards, or a glob; a relative path resolves against the working
directory.

The adapter sets the remaining fields from the rest of the config:

- `mode` follows `engine.mode`: `agg` or `disagg`.
- `prefill_engine_num_gpu` and `decode_engine_num_gpu` are `tensor` × `attention_data` of each
  role's `parallelism`.
- With `target: sla`, `ttft_ms` and `itl_ms` come from `evaluation.sla`.
- With `target: load`, the scaling thresholds are fixed: `decode_scale_up_kv_rate: 90` and
  `decode_scale_down_kv_rate: 70`, plus `prefill_scale_up_queue_tokens: 1` and
  `prefill_scale_down_queue_tokens: 0` for a disaggregated engine. The section cannot change them.
- The Planner writes no HTML diagnostics report, starts no live dashboard, and connects to no
  Prometheus server.

Set `max_num_gpus` above the starting GPU count: when the starting replicas already use the whole
budget, the Planner holds every scale-up it proposes.

A `planner` section requires `--stack dynamo`; the default `engine` stack rejects it as an unknown
section. `aisimulate predict --stack dynamo` runs offline only, so `--online` is rejected.

### Targets and Scaling Modes

- `throughput` (the default), `latency`, and `load` run load-based scaling only and ignore
  `enable_throughput_scaling` and `enable_load_scaling`, as a live Planner does. Retune-only mode is
  the one exception; see [Simulate Retune](#simulate-retune-experimental).
- `target: sla` uses the two flags: throughput scaling is on and load scaling is off unless you set
  them. Throughput scaling requires `evaluation.sla.ttft_ms` and `evaluation.sla.itl_ms`; an SLA
  with only `e2e_ms` is rejected.
- Omitted and `null` adjustment intervals follow the
  [Engine and Adapter Rules](../../../../reference/components/dynosim-replay-cli-reference.mdx#engine-and-adapter-rules);
  explicit intervals follow the `PlannerConfig` constraints in the
  [Planner Configuration reference](../../../../reference/components/planner-configuration.mdx).

### Planner Performance Model

Throughput scaling with `target: sla` needs a performance model of the engines. Predict
bootstraps it from AISimulate forward-pass estimates for `engine.model`, `engine.hardware`, and
`engine.backend`, and records the result in `planner.metadata.bootstrap`: `status: installed`, the
number of prefill and decode samples (`prefill_fpm_count`, `decode_fpm_count`), and an
`fpm_sha256` digest. Other targets do not need the model and record `status: not_required`.

The bootstrap uses AISimulate estimates even when `engine.workers.<role>.timing.type` is
`polynomial` or `fixed`. With those timing models, the Planner's model and the simulated engines
disagree until the Planner has tuned its model on the forward passes it observes. Keep
`timing.type: default` when the Planner's throughput decisions matter.

## Compare Topologies and Startup Delays

The steps below replay one hour of the FAST'25 tool-agent trace with a Planner that targets a
1,500 ms TTFT and a 50 ms ITL and may use up to four GPUs.

<Steps toc={true}>
<Step title="Run the aggregated baseline" id="run-the-aggregated-baseline">

Download the trace:

```bash
mkdir -p traces/mooncake-fast25
curl -sL \
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/traces/toolagent_trace.jsonl \
  -o traces/mooncake-fast25/toolagent_trace.jsonl
```

Save this configuration as `planner-aggregated.yaml`. It starts two aggregated workers and lets the
Planner scale to four:

```yaml
traffic:
  source:
    type: trace
    format: mooncake
    paths: [traces/mooncake-fast25/toolagent_trace.jsonl]
    block_size: 512
  load: {type: trace_timestamps, speedup: 1.0}
  stop: {max_virtual_time_seconds: 3600}
engine:
  mode: aggregated
  model: nvidia/Llama-3.1-8B-Instruct-FP8
  hardware: h200_sxm
  backend: vllm
  context_length: max
  workers:
    aggregated:
      parallelism: {replicas: 2, tensor: 1, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
      scheduler: {max_batched_tokens: 8192, max_sequences: 256}
      kv_cache: {block_size: 64, prefix_caching: true, capacity: {type: default, memory_fraction: 0.9}}
      timing: {type: default}
      startup_seconds: 0
router:
  policy: round_robin
  prefill_load_model: {type: none}
planner:
  policy: enabled
  target: sla
  enable_throughput_scaling: true
  enable_load_scaling: true
  throughput_adjustment_interval_seconds: 300
  load_adjustment_interval_seconds: 10
  max_num_gpus: 4
  min_workers: 1
evaluation:
  sla: {ttft_ms: 1500, itl_ms: 50}
```

Run the prediction:

```bash
aisimulate predict \
  --stack dynamo \
  --config planner-aggregated.yaml \
  --output-dir planner-reports/aggregated
```

The command prints a summary table and writes `planner-reports/aggregated/prediction.json`. The run
replays 23,608 requests in about 20 seconds. The table's last line,
`Timing (runner): unavailable`, is expected with `--stack dynamo`.

</Step>
<Step title="Run the disaggregated comparison" id="run-the-disaggregated-comparison">

Copy the baseline, then replace its `engine` mapping with separate prefill and decode roles:

```yaml
engine:
  mode: disaggregated
  model: nvidia/Llama-3.1-8B-Instruct-FP8
  hardware: h200_sxm
  backend: vllm
  context_length: max
  kv_transfer: {bytes_per_token: auto, bandwidth_gb_per_second: 400, timing_mode: destination_missing}
  workers:
    prefill:
      parallelism: {replicas: 1, tensor: 1, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
      scheduler: {max_batched_tokens: 8192, max_sequences: 64}
      kv_cache: {block_size: 64, prefix_caching: true, capacity: {type: default, memory_fraction: 0.9}}
      timing: {type: default}
      startup_seconds: 0
    decode:
      parallelism: {replicas: 1, tensor: 1, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
      scheduler: {max_batched_tokens: 8192, max_sequences: 256}
      kv_cache: {block_size: 64, prefix_caching: true, capacity: {type: default, memory_fraction: 0.9}}
      timing: {type: default}
      startup_seconds: 0
```

Also replace the `planner` mapping with role-specific minimums:

```yaml
planner:
  policy: enabled
  target: sla
  enable_throughput_scaling: true
  enable_load_scaling: true
  throughput_adjustment_interval_seconds: 300
  load_adjustment_interval_seconds: 10
  max_num_gpus: 4
  prefill_min_workers: 1
  decode_min_workers: 1
```

Save the result as `planner-disaggregated.yaml`, then run it:

```bash
aisimulate predict \
  --stack dynamo \
  --config planner-disaggregated.yaml \
  --output-dir planner-reports/disaggregated
```

Compare GPU time, goodput, latency, and the number of scaling events of the two runs:

```bash
for run in aggregated disaggregated; do
  jq -c --arg run "$run" '{run: $run, gpu_hours: .summary.gpu_hours,
    goodput_requests: .summary.goodput_completed_requests,
    mean_ttft_ms: .summary.mean_ttft_ms,
    scaling_events: (.planner.scaling_events | length)}' \
    "planner-reports/$run/prediction.json"
done
```

</Step>
<Step title="Sweep aggregated startup time" id="sweep-aggregated-startup-time">

`startup_seconds` delays only the workers the Planner adds; the initial replicas are ready at time
zero. Run the aggregated prediction with a sequence of overrides. Each run takes about 20 seconds:

```bash
for startup_seconds in 0 60 120 180 240 300; do
  output_dir=$(printf "planner-reports/startup-%03d" "$startup_seconds")
  aisimulate predict \
    --stack dynamo \
    --config planner-aggregated.yaml \
    --set "engine.workers.aggregated.startup_seconds=$startup_seconds" \
    --output-dir "$output_dir" \
    --overwrite
done
```

With `startup_seconds: 0`, an added worker appears in `planner.lifecycle_operations` directly as a
`worker_ready` transition. With a positive delay, it first appears as `worker_starting` and becomes
`worker_ready` `startup_seconds` later. Count added workers from `planner.scaling_events`, which
does not depend on the delay, and compare GPU time and TTFT as the delay grows. The `starting`
column counts the added workers that went through a startup delay:

```bash
for dir in planner-reports/startup-*; do
  jq -c --arg run "$dir" '{run: $run, gpu_hours: .summary.gpu_hours,
    mean_ttft_ms: .summary.mean_ttft_ms,
    added_workers: ([.planner.scaling_events[] | select(.reason == "scale_up")
      | .to_count - .from_count] | add // 0),
    starting: ([.planner.lifecycle_operations[].transitions[]
      | select(.transition == "worker_starting")] | length)}' \
    "$dir/prediction.json"
done
```

</Step>
</Steps>

Any traffic that `aisimulate predict` accepts works with a Planner, including synthetic traffic and
the other trace formats in [Traffic Rules](../../../../reference/components/dynosim-replay-cli-reference.mdx#traffic-rules).
Retune requires a Dynamo request trace.

## Read the Results

`prediction.json` has these top-level keys:

| Key | Contents |
|---|---|
| `summary` | Request, latency, throughput, goodput, and GPU-time metrics |
| `planner` | Planner details below; `null` with `policy: disabled` |
| `per_request` | One record per request with `--capture-per-request`, which also writes `requests.jsonl`; otherwise `null` |
| `coverage` | Which details the run captured |

`summary.gpu_hours` is the provisioned GPU time: (`prefill_worker_seconds` ×
`prefill_gpus_per_worker` + `decode_worker_seconds` × `decode_gpus_per_worker`) / 3600, with GPUs
per worker taken from each role's `parallelism`.

The `planner` object holds:

| Key | Contents |
|---|---|
| `metadata` | `mode`, `advisory`, `planner_config_digest`, the plugin pipeline, and `bootstrap` (see [Planner Performance Model](#planner-performance-model)) |
| `scaling_events` | One entry per replica change: `at_s`, `component` (`agg`, `prefill`, or `decode`), `from_count`, `to_count`, and `reason` (`scale_up` or `scale_down`) |
| `lifecycle_operations` | Worker transitions (`worker_starting`, `worker_ready`, `worker_draining`, `worker_removed`) with `at_ms`, `pool`, and `cause` |
| `ticks` | One record per Planner tick, described below |
| `total_ticks` | Number of Planner ticks |
| `html_report_path` | Always `null` in predict; see [Run from Python](#run-from-python) for the HTML report |
| `retune_events` | Retune decisions; present whenever `retune` is set, and empty when retune made no decision |
| `recommended_scaling_events` | Replica changes the Planner recommended but did not apply; present only with `advisory: true` (see [Replay in Advisory Mode](#replay-in-advisory-mode)) |

Each `ticks` entry has `ordinal`, `at_ms`, `scheduled_tick` (which scaling loops ran),
`input` (traffic, worker counts, and forward-pass observations), `topology` (active, starting, and
draining workers per role), `effects` (the Planner's output, including `scale_to` and
diagnostics), and `runtime_decision` (`target_prefill`, `target_decode`, `next_tick_ms`, and
`router_parameters`). Aggregated workers appear under `topology.decode` and
`runtime_decision.target_decode`. Predict always records ticks: the one-hour aggregated example
writes about 2 MB.

For the other files in the output directory and the exit codes, see
[Output](../../../../reference/components/dynosim-replay-cli-reference.mdx#output) in the replay
CLI reference.

## Run from Python

`dynamo.replay.run_trace_replay` runs the same Planner-in-the-loop replay from a script and returns
a `ReplayReport` with `summary`, `per_request`, `coverage`, `planner`, and `to_dict()`. The Planner
input is a `PlannerConfig`, not the AISimulate `planner` section: use `PlannerConfig` field names
(`optimization_target`, `max_gpu_budget`, `min_endpoint`, `ttft_ms`, `itl_ms`), and set `mode` to
`agg` with `extra_engine_args` or `disagg` with `prefill_engine_args` and `decode_engine_args`.
This script mirrors the aggregated baseline:

```python
from dynamo.replay import run_trace_replay

ENGINE = {
    "engine": {
        "backend": "vllm",
        "block_size": 64,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 256,
        "timing_model": {
            "type": "external",
            "provider": "ais",
            "config": {
                "model": "nvidia/Llama-3.1-8B-Instruct-FP8",
                "system": "h200_sxm",
                "backend": "vllm",
                "worker_type": "aggregated",
            },
        },
    },
}

PLANNER = {
    "mode": "agg",
    "optimization_target": "sla",
    "enable_throughput_scaling": True,
    "enable_load_scaling": True,
    "throughput_adjustment_interval_seconds": 300,
    "load_adjustment_interval_seconds": 10,
    "max_gpu_budget": 4,
    "ttft_ms": 1500,
    "itl_ms": 50,
    "decode_engine_num_gpu": 1,  # GPU hours in the HTML report
}

report = run_trace_replay(
    "traces/mooncake-fast25/toolagent_trace.jsonl",
    trace_format="mooncake",
    trace_block_size=512,
    extra_engine_args=ENGINE,
    num_workers=2,
    planner_config=PLANNER,
    sla_ttft_ms=1500,
    sla_itl_ms=50,
    max_sim_time_ms=3_600_000,
)
print(report.planner.metadata["bootstrap"]["status"])
print(len(report.planner.scaling_events), report.summary["gpu_hours"])
print(report.planner.html_report_path)
```

- `planner_config` accepts a mapping, an inline JSON string, or the path of a JSON or YAML file.
- Planner replay is offline only. Trace formats other than `dynamo` take exactly one file.
- `sla_ttft_ms`, `sla_itl_ms`, and `sla_e2e_ms` set the SLA for the report's goodput metrics. The
  Planner's own targets are `ttft_ms` and `itl_ms` in `planner_config`.
- Engine arguments use the
  [canonical engine mapping](../../../../reference/components/dynosim-replay-cli-reference.mdx#canonical-python-engine-configuration).
  An AISimulate `timing_model` with `worker_type` gives the same performance-model bootstrap as
  predict. Engine arguments without one, such as the default `polynomial` timing, leave the
  bootstrap at `not_configured_load_only`, and the log warns that throughput-based scaling falls back
  to load-based scaling. The engine argument `startup_time`, in seconds, delays added workers like
  `startup_seconds`.
- With the `PlannerConfig` defaults, the replay writes an HTML diagnostics report to
  `./planner_reports/` and returns its path in `report.planner.html_report_path`; predict never
  writes one. Set `report_interval_hours: None` and `live_dashboard_port: 0` to skip it.
- `capture_planner_details=False` drops `ticks`, `lifecycle_operations`, and the HTML report, and
  keeps `scaling_events`, `total_ticks`, `metadata`, and `retune_events`.
- `report.planner.scaling_events` holds `ScalingEvent` objects; `report.to_dict()` returns the
  same JSON as `prediction.json`.

## Replay in Advisory Mode

Set `advisory: true` to replay the Planner as it runs in [advisory mode](../../../../reference/components/planner-configuration.mdx#advisory-mode)
in production: it makes every decision and changes nothing.

```yaml
planner:
  policy: enabled
  advisory: true
```

In Python, set `"advisory": True` in `planner_config`. A Planner configuration copied from a
deployment that runs in advisory mode replays the same way; set `advisory: false`, or pass
`--set planner.advisory=false` to predict, to simulate the scaling it would do. If that
configuration enables `batch_scheduling`, also set `batch_scheduling.enabled: false`: replay does not
run batch scheduling, and non-advisory batch scheduling requires Redis settings. An advisory replay
logs an `[ADVISORY]` warning at startup. `policy: disabled` rejects `advisory: true`, and
`aisimulate recommend` rejects the `advisory` key.

What runs:

- Every Planner tick, with the same observations, load predictions, and load-based and
  throughput-based decisions as an active Planner.
- Retune sweeps and scoring. With load-based or throughput-based scaling, the candidate replays run
  an active inner Planner, so a candidate is scored on the replicas it would drive; retune-only
  candidates keep the replicas fixed.

What stays fixed:

- The replica counts and router parameters keep their starting values, and `lifecycle_operations`
  is empty. The Planner sees no scaling in progress and its starting replicas as the expected
  counts, as a live advisory Planner sees a deployment that never changes.
- `summary` reports the starting deployment serving the whole trace.

What is reported:

- `planner.metadata.advisory` is `true` and `planner.scaling_events` is empty.
- `planner.recommended_scaling_events` lists the recommendations with the fields of
  `scaling_events`. An entry is added each time a component's recommended count changes to a value
  other than its current count, which is `from_count`. A scaling decision that keeps a component
  at its current count resets that component's recommendation, so the same recommendation after a
  hold is listed again. The recommendation of every tick stays in `ticks[].effects.scale_to`.
- A retune decision that would apply has outcome `advisory`, with the `chosen` point and the
  `planner_changes` it recommends, and `predictor_warmup: null`. Every tick's
  `runtime_decision.router_parameters` is `null`, so the recommended router parameters appear only
  in `retune_events[].chosen.router`.
- With `run_trace_replay` and the `PlannerConfig` report defaults, the HTML diagnostics report shows
  the fixed replica counts with the recommendation markers. Predict writes no HTML report.

<Note>
Each recommendation is one step from the starting deployment, decided on traffic that the starting
deployment served. It is not the path an active Planner would take: after its first scale-up, an
active Planner observes the larger deployment and decides from there, and every later retune sweep
searches from the point it applied. To predict what an active Planner does, replay with
`advisory: false`.
</Note>

## Simulate Retune (Experimental)

<Warning>
**Experimental.** Retune runs only in offline simulation in this release. A live Planner rejects a
configuration that sets `retune` at startup.
</Warning>

Retune makes the Planner retune its router and Planner runtime parameters from the traffic of the
last window. For the concept and the two scenarios, see
[Periodic Retune](planner-guide.md#periodic-retune-experimental). For windows, scoring, and decision
timing, see [Planner Design](planner-design.md#retune-experimental-simulation-only). For every
`retune` field, its default, and the search-domain syntax, see the
[`retune` reference](../../../../reference/components/planner-configuration.mdx#retune-experimental-simulation-only).

### Requirements

With `aisimulate predict --stack dynamo`:

- `planner.policy: enabled`. To retune with fixed replicas, also set both
  `enable_load_scaling: false` and `enable_throughput_scaling: false` (retune-only mode).
- `traffic.source` with `type: trace` and `format: dynamo`, and `traffic.load.type:
  trace_timestamps`; `speedup` is allowed. The `agentic_*` load options and
  `traffic.stop.max_virtual_time_seconds` are rejected.
- `router.policy: kv_router` when `retune.router` is set.
- `aisimulate recommend` rejects `retune`: each recommendation candidate is a static configuration.

Recommended: set `evaluation.sla` with `ttft_ms` and `itl_ms`. Predict fills `retune.sla_ttft_ms`
and `retune.sla_itl_ms` from it when they are unset, so retune scores candidates against the SLA
that AISimulate reports goodput for. Without it, scoring uses the Planner's `ttft_ms` and `itl_ms`,
which then keep their `PlannerConfig` defaults.

With `dynamo.replay.run_trace_replay`:

- `trace_format="dynamo"`. Other trace formats and `run_synthetic_trace_replay` reject retune.
- No `replay_concurrency`, `agentic_*` options, or `max_sim_time_ms`, and no requests with
  `agent_context` (an agentic Dynamo trace). Retune windows drop tool rows. `execution_model` is
  accepted, because it applies only to agentic requests.
- `router_mode="kv_router"` and a router configuration that meets the
  [`retune.router` requirements](../../../../reference/components/planner-configuration.mdx#retune-experimental-simulation-only)
  when `retune.router` is set.
- With throughput scaling, retune reads `load_predictor_warmup_trace` as recorded history. It must
  then be a Dynamo request trace that ends at or before the first replayed request, on the same
  absolute timestamps. Without throughput scaling, retune does not read it.

### Prepare a Dynamo Request Trace

Retune replays windows of a Dynamo request trace (`dynamo.request.trace.v1`). Record one from a
deployment with [request tracing](../../../../cli/operations/observability.mdx#capture-and-replay-requests);
the [record schema](../../../../reference/observability/request-traces.mdx#record-schema) describes
its rows. To try retune without a deployment, convert the Mooncake trace that
[Run the aggregated baseline](#run-the-aggregated-baseline) downloads. This script writes the first 20 minutes to `traces/toolagent-history.jsonl`, the predictor history in
Scenario 2, and the remaining 39 minutes to `traces/toolagent-replay.jsonl`:

```python
import json
import sys

ORIGIN_MS = 1_760_000_000_000  # any epoch-millisecond origin
SPLIT_MS = 20 * 60 * 1000  # rows before this offset become predictor history

with (
    open(sys.argv[1], encoding="utf-8") as rows,
    open("traces/toolagent-history.jsonl", "w", encoding="utf-8") as history,
    open("traces/toolagent-replay.jsonl", "w", encoding="utf-8") as replay,
):
    for index, line in enumerate(rows):
        row = json.loads(line)
        received_ms = ORIGIN_MS + int(row["timestamp"])
        record = {
            "schema": "dynamo.request.trace.v1",
            "event_type": "request_end",
            "event_time_unix_ms": received_ms,
            "request": {
                "request_id": f"toolagent-{index}",
                "request_received_ms": received_ms,
                "output_tokens": row["output_length"],
                "replay": {
                    "trace_block_size": 512,
                    "input_length": row["input_length"],
                    "input_sequence_hashes": row["hash_ids"],
                },
            },
        }
        sink = history if int(row["timestamp"]) < SPLIT_MS else replay
        sink.write(json.dumps(record) + "\n")
```

Save it as `mooncake_to_dynamo.py` and run it:

```bash
python3 mooncake_to_dynamo.py traces/mooncake-fast25/toolagent_trace.jsonl
```

A converted row has no `total_time_ms`, so retune cannot exclude requests that completed after a
window's sweep launched. A recorded trace carries each request's latency.

### Scenario 1: Fixed Replicas, Router Only

Save this configuration as `retune-router.yaml`. Three aggregated workers start with router
temperature 1.0, and retune searches `prefill_load_scale` and `temperature` every 10 minutes of
trace time:

```yaml
traffic:
  source: {type: trace, format: dynamo, paths: [traces/toolagent-replay.jsonl]}
  load: {type: trace_timestamps}
engine:
  mode: aggregated
  model: nvidia/Llama-3.1-8B-Instruct-FP8
  hardware: h200_sxm
  backend: vllm
  context_length: max
  workers:
    aggregated:
      parallelism: {replicas: 3, tensor: 1, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
      scheduler: {max_batched_tokens: 8192, max_sequences: 256}
      kv_cache: {block_size: 64, prefix_caching: true, capacity: {type: default, memory_fraction: 0.9}}
      timing: {type: default}
router: {policy: kv_router, temperature: 1.0}
planner:
  policy: enabled
  enable_load_scaling: false
  enable_throughput_scaling: false
  retune:
    interval_seconds: 600
    max_parallel: 4
    router:
      overlap_score_credit: 1.0
      prefill_load_scale: {choices: [0.5, 1.0, 2.0]}
      temperature: {choices: [0.0, 1.0]}
evaluation:
  sla: {ttft_ms: 1500, itl_ms: 50}
```

- The router domain is 1 × 3 × 2 = 6 grid points. The scalar `overlap_score_credit: 1.0` pins that
  knob; an omitted knob would search its default domain.
- Both scaling flags are `false`, so the Planner never scales. It ticks every
  `load_adjustment_interval_seconds` (5 seconds by default), and `interval_seconds` must be a
  multiple of that tick.
- Window `k` covers `[600k, 600(k+1))` seconds after the first request, and its sweep launches
  60 seconds (`grace_seconds`) after the window ends.

Run the prediction:

```bash
aisimulate predict --stack dynamo --config retune-router.yaml --output-dir retune-reports/router
```

The 39-minute replay makes three decisions. Each sweep replays about 4,100 requests once per
candidate, and the whole run takes about a minute.

### Scenario 2: Autoscaling, Router and Planner

For a Planner that scales and is retuned, copy `retune-router.yaml` to `retune-joint.yaml`, set
`replicas: 2`, and replace the `planner` section:

```yaml
planner:
  policy: enabled
  target: sla
  enable_throughput_scaling: true
  enable_load_scaling: true
  max_num_gpus: 6
  load_predictor_warmup_trace: traces/toolagent-history.jsonl
  retune:
    interval_seconds: 600
    max_parallel: 8
    router:
      overlap_score_credit: 1.0
      prefill_load_scale: 1.0
      temperature: {choices: [0.0, 1.0]}
    planner: default
```

- `planner: default` searches `load_scaling_down_sensitivity` in `{70, 80, 90}`, because load
  scaling is on, and runs a load-predictor pre-sweep over the 11 predictor presets, because
  `target: sla` uses throughput scaling. With two temperatures, the grid has 6 points.
- Each candidate replay runs the current Planner with the candidate's Planner fields, starting from
  the replica counts at the window start.
- `load_predictor_warmup_trace` warms the Planner's load predictor and, because `target: sla`
  scales on throughput, is also retune's trace history. `history_seconds` resolves to the
  throughput interval (180 seconds by default) × 50, the largest `prophet_window_size` among the
  presets: 9,000 seconds. The history file supplies the 20 minutes it has.

Run it:

```bash
aisimulate predict --stack dynamo --config retune-joint.yaml --output-dir retune-reports/joint
```

### Read Retune Decisions

`planner.retune_events` lists every decision whose decision tick falls within the replay. It is
empty when retune made no decision, for example in a replay shorter than one window plus
`grace_seconds`:

```bash
jq -c '.planner.retune_events[] | {window_index, outcome, decided_at_s,
  router: .chosen.router, planner_changes, baseline_score, best_score}' \
  retune-reports/router/prediction.json
```

Each record has these fields. Times are Planner-clock seconds from the start of the replay:

| Field | Meaning |
|---|---|
| `decision_id` | `retune-<start>-w<k>` |
| `outcome` | `applied`, `kept`, `advisory`, `busy`, `stale`, `timeout`, or `failed`; see [outcomes](planner-design.md#scoring-and-outcomes) |
| `reason` | Why a decision was kept or failed, for example `empty window` |
| `window_index`, `window_start_s`, `window_end_s` | The window that was replayed |
| `as_of_s`, `launched_at_s`, `decided_at_s` | Scheduled launch time, the tick that launched the sweep, and the tick that decided; `launched_at_s` is `null` when no sweep ran |
| `num_requests`, `num_candidates` | Requests in the window and candidates in the sweep |
| `previous`, `chosen` | The point before the decision and the point chosen, as `router` and `planner` maps keyed by live names (`router_temperature`, `load_scaling_down_sensitivity`) |
| `planner_changes` | Planner fields the decision changes |
| `baseline_score`, `best_score`, `top` | The current point's score, the best score, and up to five `[candidate, score]` pairs |
| `predictor_presweep` | Load-predictor pre-sweep: `winner`, `fields`, per-candidate `losses`, `eval_steps`, `warmup_steps`, and `wall_s` |
| `predictor_warmup` | Number of traffic observations that warm a newly selected load predictor; `null` unless an `applied` decision replaces the predictor |
| `sweep_wall_s`, `decision_latency_s` | The sweep's wall-clock time, and the delay after launch at which the decision could land; `null` for `busy` and decisions that fail before a sweep runs |

Each tick also records the router state. `runtime_decision.router_parameters` is the desired
parameter map per worker type from that tick on. `router_policies` is the replay router's state,
`revision` and `parameters`, before the tick's update. A tick does not repeat the decisions; match
a decision's `decided_at_s` × 1000 to the tick's `at_ms`. To see the switch around the first
decision of Scenario 1:

```bash
jq -c '.planner.ticks[] | select(.at_ms >= 655000 and .at_ms <= 675000)
  | {at_ms, desired: .runtime_decision.router_parameters.aggregated.router_temperature,
     replay: [.router_policies[] | .parameters.router_temperature]}' \
  retune-reports/router/prediction.json
```

### Decision Latency

A sweep takes time, and the old parameters keep serving until the decision. In replay, virtual time
stops while the sweep runs, so `decision_latency_seconds` sets how much trace time passes between
launch and decision; [Decision Timing](planner-design.md#decision-timing) describes when the
decision lands. Choose the value by what the run is for:

- The default, `0`, decides one tick after launch. It gives the best case.
- A fixed number of seconds gives the same decision ticks on every run. Use it to compare
  configurations, and set it to the sweep time you expect on the host that will run sweeps.
- `measured` charges the sweep's own wall-clock time on the replay host, as the
  [`decision_latency_seconds` field](../../../../reference/components/planner-configuration.mdx#retune-experimental-simulation-only)
  defines it. Decision ticks then depend on the speed and load of the replay host, so two runs of
  the same config can switch at different ticks and report different results. Set
  `decision_latency_scale` above 1 to model a slower sweep host, or below 1 for a faster one. Above
  1, also lower `timeout_seconds` to at most `interval_seconds` / `decision_latency_scale`; the
  default timeout is rejected.

For example, on a host where Scenario 1's sweeps take about 11 seconds, `measured` moves each
decision from one tick after launch (665 seconds for window 0) to the third (675 seconds):

```bash
aisimulate predict --stack dynamo --config retune-router.yaml \
  --set planner.retune.decision_latency_seconds=measured \
  --output-dir retune-reports/router-measured
```

### Measure the Gain

Run the same config without retune and compare `summary.goodput_output_throughput_tok_s`,
`summary.gpu_hours`, and the latency percentiles:

- For Scenario 1, replace the whole `planner` section with
  `--set 'planner={policy: disabled}'`. Do not only remove `retune`: without it, the two `false`
  scaling flags no longer select retune-only mode, and the Planner autoscales with its target's
  defaults.
- For Scenario 2, remove retune with `--set planner.retune=null`.

```bash
aisimulate predict --stack dynamo --config retune-router.yaml \
  --set 'planner={policy: disabled}' --output-dir retune-reports/static
for run in static router; do
  jq -c --arg run "$run" '{run: $run,
    goodput_tok_s: .summary.goodput_output_throughput_tok_s,
    gpu_hours: .summary.gpu_hours, p90_ttft_ms: .summary.p90_ttft_ms}' \
    "retune-reports/$run/prediction.json"
done
```

`--set planner.advisory=true` keeps the sweeps and records a decision that would apply as
`advisory`, with the point it would choose, without applying it. Other decisions keep their
outcomes, such as `kept` when no candidate clears the threshold. Advisory mode also keeps the
replica counts fixed; see [Replay in Advisory Mode](#replay-in-advisory-mode).

Each candidate is replayed once. All candidate replays of a window, the current point's included,
use the same router seed and request ids, so a candidate that changes no routing or scaling
decision scores exactly the current point and is kept. A candidate that changes decisions is still
judged on one sample path of one window. With the pinned AISimulate release, an inner Planner with
`optimization_target: sla` can still decide differently in two replays of the same candidate, so
such a candidate can miss an exact tie; see
[Candidates and the Sweep](planner-design.md#candidates-and-the-sweep).
The outer replay keeps its default, unseeded routing, so two runs of the same config can differ in
the outer replay's results and, with autoscaling, in the replica counts a window starts from. Set
`min_improvement_percentage` (3 by default) to the smallest gain worth a switch, and check a
decision across windows or runs before trusting a small margin.

### Run Retune from Python

`run_trace_replay` takes the same `retune` block inside a `PlannerConfig` mapping. Sweeps start
worker processes with the `spawn` method, so guard the script with `if __name__ == "__main__":`.
This script runs Scenario 1:

```python
from dynamo.llm import KvRouterConfig
from dynamo.replay import run_trace_replay

ENGINE = {
    "engine": {
        "backend": "vllm",
        "block_size": 64,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 256,
        "timing_model": {
            "type": "external",
            "provider": "ais",
            "config": {
                "model": "nvidia/Llama-3.1-8B-Instruct-FP8",
                "system": "h200_sxm",
                "backend": "vllm",
                "worker_type": "aggregated",
            },
        },
    },
}

PLANNER = {
    "mode": "agg",
    "enable_load_scaling": False,
    "enable_throughput_scaling": False,
    "retune": {
        "interval_seconds": 600,
        "max_parallel": 4,
        "sla_ttft_ms": 1500,
        "sla_itl_ms": 50,
        "router": {
            "overlap_score_credit": 1.0,
            "prefill_load_scale": {"choices": [0.5, 1.0, 2.0]},
            "temperature": {"choices": [0.0, 1.0]},
        },
    },
}

if __name__ == "__main__":
    report = run_trace_replay(
        ["traces/toolagent-replay.jsonl"],
        trace_format="dynamo",
        extra_engine_args=ENGINE,
        num_workers=3,
        router_mode="kv_router",
        router_config=KvRouterConfig(router_temperature=1.0),
        planner_config=PLANNER,
        sla_ttft_ms=1500,
        sla_itl_ms=50,
    )
    for event in report.planner.retune_events:
        print(event["window_index"], event["outcome"], event["chosen"]["router"])
```

`router_config` is the startup router. Without `evaluation.sla`, the scoring SLA defaults to the
Planner's `ttft_ms` and `itl_ms`, so the script sets `retune.sla_ttft_ms` and `retune.sla_itl_ms`.
With `retune` set, `report.planner.retune_events` is a list, empty when retune made no decision;
without `retune`, it is `None`.

For Scenario 2, pass `num_workers=2` and write the Scenario 2 `planner` section with
`PlannerConfig` field names. Python does not take the SLA from `evaluation.sla`, so set the
Planner's own `ttft_ms` and `itl_ms`, which `target: sla` scales against and retune scores with:

```python
PLANNER = {
    "mode": "agg",
    "optimization_target": "sla",
    "enable_load_scaling": True,
    "enable_throughput_scaling": True,
    "max_gpu_budget": 6,
    "ttft_ms": 1500,
    "itl_ms": 50,
    "load_predictor_warmup_trace": "traces/toolagent-history.jsonl",
    "retune": {
        "interval_seconds": 600,
        "max_parallel": 8,
        "router": {
            "overlap_score_credit": 1.0,
            "prefill_load_scale": 1.0,
            "temperature": {"choices": [0.0, 1.0]},
        },
        "planner": "default",
    },
}
```

Keep the AISimulate `timing_model` with `worker_type` in `ENGINE`. Without it, the
performance-model bootstrap is `not_configured_load_only`, and throughput scaling falls back to
load scaling, as described in [Run from Python](#run-from-python).

### Size the Run

Each window costs one sweep: one replay of the window per distinct candidate, plus one forecast task
per predictor candidate when the load predictor is searched, and one task that prepares a winning
predictor that differs from the current one. A sweep runs in `max_parallel` worker processes that
it starts and stops, and the replay waits for it. A run therefore takes about the replay without
retune plus the sum of `sweep_wall_s` over its decisions. For the size of the
candidate set, see [Candidates and the Sweep](planner-design.md#candidates-and-the-sweep). In
Scenario 2, every candidate replay also runs a Planner, including its performance-model bootstrap
for `target: sla`.

`timeout_seconds` (by default `interval_seconds`) bounds each sweep in wall-clock seconds. A sweep
that exceeds it is killed, the decision is `timeout`, and nothing changes. If sweeps time out,
narrow the domain, raise `max_parallel`, or lengthen `interval_seconds`.

`aisimulate predict` runs the replay in a supervised execution process and sets no wall-clock limit
on the replay itself. Sweep workers are child processes of the execution process, so they count
against its limits under `execution.resources`:

| Setting | Default | Retune guidance |
|---|---|---|
| `memory_limit_gb` | `auto` | Cover the outer replay plus `max_parallel` window replays |
| `reserve_memory_gb`, `reserve_memory_fraction` | `1.0`, `0.0` | Raise on shared hosts |
| `cpu_limit` | `auto`: host CPUs minus one | At least `max_parallel` + 1 |
| `initialization_timeout_seconds` | `60` | Raise, for example to 600, for multi-day traces |
| `shutdown_timeout_seconds` | `5` | The default is usually enough |

- **Memory.** `auto` is the smaller of 90% of the memory available at start and the available
  memory less the reserve. The resident memory of the whole process tree, sweep workers included,
  must stay below the limit, and the host's available memory must stay above the reserve;
  otherwise the run stops as `resource_limited` with exit code 3. Size the limit as the outer
  replay plus `max_parallel` times the peak of one window replay, measured on a short run:
  `resource-runtime.json` reports the tree's `peak_observed_rss_bytes`.
- **CPUs.** The execution process and its sweep workers are pinned to `cpu_limit` CPUs. AISimulate
  also limits math libraries to one thread (`OMP_NUM_THREADS=1` and similar), which sweep workers
  inherit.
- **Timeouts.** The initialization timeout covers start-up until the replay begins, including a
  pass over the whole trace that sizes the memory plan, about 20 seconds per million requests. The
  shutdown timeout covers process exit after the replay; sweeps finish before the replay returns.

For scale, on a 24-core host, Scenario 1 above peaks at about 2 GB with `max_parallel: 4`, and
Scenario 2 at about 4.5 GB with `max_parallel: 8`.

### Limitations

- **Simulation only.** A live Planner rejects `retune` at startup.
- **Open-loop Dynamo request traces only.** See [Requirements](#requirements).
- **Cold window replays.** Each candidate replay starts with an empty KV cache and the replica
  counts at the window start, so prefix-cache reuse from before the window does not carry over.
- **Recorded completion times.** A window includes a request only if it completed before the sweep
  launched, judged by the latency recorded in the trace rather than the simulated one.
- **One replay per candidate.** Candidate replays share one random seed, but a score is still one
  sample path of one window; see [Measure the Gain](#measure-the-gain).
- **Router cache weights are not retunable.** `host_cache_hit_weight` and `disk_cache_hit_weight`
  also set how the startup router counts cached tokens, which a runtime update does not change, and
  replay reports no shared cache hits for `shared_cache_multiplier`. Retune rejects all three and
  its error says why.
