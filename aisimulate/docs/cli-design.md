---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: AISimulate CLI Design
subtitle: Draft public command, configuration, and output contract for simulation and recommendation
---

This document proposes one public command-line interface (CLI) for AISimulate prediction and
configuration recommendation. It unifies the user-facing concepts currently split across Replay and
Sweeper while keeping execution-stack details out of the configuration schema.

## Scope

The design covers:

- The `aisimulate simulate` and `aisimulate recommend` commands.
- Their command-line flags and precedence rules.
- The shared YAML schema and recommendation-only extensions.
- Search-domain syntax and validation.
- User-visible files, standard output, errors, and exit codes.
- The conceptual replacement of the existing Replay and Sweeper entry points.

The design does not cover:

- Python APIs, `ReplaySpec`, runners, factories, or adapters.
- How the `engine` and `dynamo` stacks execute a simulation.
- AIConfigurator (AIC), Planner, router, optimizer, worker-pool, caching, or timeout internals.
- Compatibility shims, migration code, or implementation sequencing.
- Online replay. Both commands are offline-only in version 1.

## Commands

### Simulate

Predict the behavior of one concrete deployment configuration:

```bash
aisimulate simulate --config simulation.yaml
```

`simulate` accepts only concrete configuration values. It rejects search domains and the
recommendation-only `optimization` and `optimizer` sections.

### Recommend

Search a configuration space and recommend concrete deployment configurations:

```bash
aisimulate recommend --config recommendation.yaml
```

`recommend` accepts the complete simulation schema plus search domains, `optimization`, and
`optimizer`. Every recommended YAML file is a concrete configuration that can be passed directly to
`aisimulate simulate`.

### Common Options

| Option | Type | Default | Meaning |
|---|---|---:|---|
| `-c`, `--config PATH` | path | Required | Input YAML file. |
| `--stack engine\|dynamo` | enum | `engine` | Execution stack. This selection is CLI-only and is never written into YAML. |
| `--set PATH=YAML_VALUE` | repeatable assignment | None | Override an existing configuration path after loading YAML. |
| `--output-dir PATH` | path | `./aisimulate-output` | Directory for durable results. |
| `--overwrite` | flag | `false` | Replace known AISimulate output files in an existing output directory. |
| `--format table\|json` | enum | `table` | Standard-output presentation. It does not change durable output files. |

`simulate` also accepts:

| Option | Type | Default | Meaning |
|---|---|---:|---|
| `--capture-per-request` | flag | `false` | Write per-request prediction records to `requests.jsonl`. |

The CLI deliberately does not expose field-specific flags such as `--request-per-second` or
`--num-workers`. YAML is the authoritative semantic configuration surface.

### Override Semantics

`--set` uses a dot-separated path and parses its value as YAML:

```bash
aisimulate simulate \
  --config simulation.yaml \
  --set traffic.load.requests_per_second=16 \
  --set engine.workers.aggregated.parallelism.replicas=4
```

The following rules apply:

- The path must already exist in the input schema. An override cannot create a new or unknown field.
- Overrides are applied from left to right. The last assignment to a path wins.
- YAML scalar, sequence, and mapping syntax is accepted on the right-hand side.
- Sequence-index paths are not supported in version 1. Override the complete sequence instead.
- Normal schema and cross-field validation runs after all overrides are applied.
- The resolved input written to the output directory includes the overrides.

## Configuration Model

Both commands use one strict YAML model:

```yaml
traffic: {}
engine: {}
router: {}
planner: {}
evaluation: {}
```

`recommend` extends that model with:

```yaml
optimization: {}
optimizer: {}
```

The command determines the document type. There is no top-level `kind` or stack field.

| Section | `simulate` | `recommend` | Purpose |
|---|---|---|---|
| `traffic` | Optional | Optional | Request source, load shape, and stopping condition. Uses the default synthetic request traffic when omitted. |
| `engine` | Required | Required | Model, hardware, backend, topology, and worker roles. |
| `router` | Optional | Optional | Request routing policy. Defaults to round robin. |
| `planner` | Optional | Optional | Runtime scaling policy. Defaults to disabled. |
| `evaluation` | Optional | Optional | Service-level objective (SLA) thresholds used for reporting and goals. |
| `optimization` | Rejected | Required | Recommendation objective and candidate GPU constraints. |
| `optimizer` | Rejected | Optional | Public search controls. |

Unknown fields are rejected everywhere. Every semantic configuration knob is an explicit, typed YAML
field. The selected stack, backend, policy, timing model, or capacity model determines which
conditional fields are legal; there is no generic configuration passthrough mapping.

## Traffic

Traffic is always expressed as three concepts:

```yaml
traffic:
  source: {}
  load: {}
  stop: {}
```

`source` defines requests or sessions, `load` defines when they begin, and `stop` defines when the run
ends. The load and stop unit follows the source type.

Omitting `traffic` is equivalent to this concrete default:

```yaml
traffic:
  source:
    type: synthetic
    input_tokens: 1024
    output_tokens: 128
  load:
    type: concurrency
    concurrency: 1
  stop:
    requests: 100
```

The default source contains independent requests with fixed input sequence length (ISL) and output
sequence length (OSL); it does not sample token-length distributions or create sessions. A supplied
`traffic` mapping does not merge recursively with this example. Once `traffic` is present, its normal
source, load, and stop validation applies.

### Synthetic Request Source

```yaml
traffic:
  source:
    type: synthetic
    input_tokens: 1024
    output_tokens: 128
  load:
    type: poisson
    requests_per_second: 8
    seed: 42
  stop:
    requests: 100
```

`synthetic` generates independent single requests. It does not accept a `session` mapping.

| Path | Type | Default | Constraints |
|---|---|---:|---|
| `traffic.source.type` | enum | Required | `synthetic`. |
| `traffic.source.input_tokens` | integer | Required | Greater than zero. |
| `traffic.source.output_tokens` | integer | Required | Greater than zero. |

### Synthetic Session Source

```yaml
traffic:
  source:
    type: synthetic-session
    new_input_tokens_per_turn: 1024
    output_tokens_per_turn: 128
    session:
      turns: 4
      shared_prefix_ratio: 0.5
      prefix_groups: 16
      inter_turn_delay_ms: 1000
  load:
    type: constant_rate
    sessions_per_second: 8
  stop:
    sessions: 100
```

`synthetic-session` generates multi-turn sessions. It requires a `session` mapping, and requests
inside one session execute in turn order.

| Path | Type | Default | Constraints |
|---|---|---:|---|
| `traffic.source.type` | enum | Required | `synthetic-session`. |
| `traffic.source.new_input_tokens_per_turn` | integer | Required | Greater than zero. New input contributed by each turn. |
| `traffic.source.output_tokens_per_turn` | integer | Required | Greater than zero. Output generated by each turn. |
| `traffic.source.session.turns` | integer | Required | At least `2`; use `synthetic` for one request. |
| `traffic.source.session.shared_prefix_ratio` | number | `0` | From `0` through `1`. |
| `traffic.source.session.prefix_groups` | integer | `0` | Nonnegative. Must be positive when `shared_prefix_ratio` is positive. |
| `traffic.source.session.inter_turn_delay_ms` | number | `0` | Nonnegative. |

### Synthetic Load and Stop

Both synthetic source types use exactly one of the following load shapes:

| `traffic.load.type` | Required Fields | Semantics |
|---|---|---|
| `concurrency` | `concurrency` | Closed loop with at most this many source units active. |
| `poisson` | Source-specific `*_per_second`; optional `seed` | Open loop with exponentially distributed inter-arrival times. |
| `constant_rate` | Source-specific `*_per_second` | Open loop with constant inter-arrival times. |
| `kv_capacity_fraction` | `fraction` | Recommendation-only closed-loop load resolved relative to each candidate's KV capacity. |

The source-specific open-loop rate field is:

- `requests_per_second` for `source.type: synthetic`.
- `sessions_per_second` for `source.type: synthetic-session`.

The rate is a positive finite number. `concurrency` is a positive integer and counts independent
requests for `synthetic` or active sessions for `synthetic-session`. At most one turn of a session is
active at a time. `seed` is a nonnegative integer and defaults to `42`. `fraction` is greater than `0`
and finite; it has no upper bound. `fraction: 1` targets a concurrency whose estimated KV working set
equals the candidate's usable KV capacity. Values greater than `1` intentionally oversubscribe that
capacity. They remain valid because excess work can queue; they do not mean that the engine has more
physical KV memory.

The stop fields also follow the source unit:

| Source Type | Fixed Count | Load-Relative Count |
|---|---|---|
| `synthetic` | `requests` | `requests_per_load_unit` |
| `synthetic-session` | `sessions` | `sessions_per_load_unit` |

All four fields are positive. The fixed-count field is a positive integer. The load-relative field is
a positive number and resolves to `max(1, round(count_per_load_unit * load_unit))`. The load unit is
concurrency for `concurrency` and resolved `kv_capacity_fraction` traffic, requests per second for a
`synthetic` open-loop source, or sessions per second for a `synthetic-session` open-loop source.

For `synthetic-session`, a session with four turns contributes four requests but only one unit to the
load and stopping condition.

`sessions_per_second` controls the arrival rate of new sessions. For a multi-turn session, it schedules
the first turn; later turns follow that session's completion and `inter_turn_delay_ms` rules and do not
count as new load arrivals. `requests_per_second` schedules independent single requests.

In a recommendation input, source type, token fields, session shape, and stopping condition stay
concrete. Only numeric fields under `traffic.load` can be search domains. A
`kv_capacity_fraction` recommendation is materialized as a concrete `concurrency` load in each
recommended simulation YAML.

### Trace Source

```yaml
traffic:
  source:
    type: trace
    paths:
      - traces/requests.jsonl
    format: mooncake
    block_size: 512
  load:
    type: trace_timestamps
    speedup: 1.0
  stop:
    max_virtual_time_seconds: 300
```

| Path | Type | Default | Constraints |
|---|---|---:|---|
| `traffic.source.type` | enum | Required | `trace`. |
| `traffic.source.paths` | list of paths | Required | Nonempty. Exactly one path except for `dynamo`, which permits one or more. Path order is preserved. |
| `traffic.source.format` | enum | `mooncake` | `mooncake`, `mooncake-delta`, `agentic_mooncake`, `applied_compute_agentic`, or `dynamo`. |
| `traffic.source.block_size` | integer or null | `512` except `dynamo`; embedded for `dynamo` | Greater than zero when set. |
| `traffic.load.type` | enum | Required | `trace_timestamps` or `concurrency`. |
| `traffic.load.speedup` | number | `1` | Positive; valid only with `trace_timestamps`. |
| `traffic.load.concurrency` | integer | None | Positive; required only with `concurrency`. |
| `traffic.stop.max_virtual_time_seconds` | number | None | Positive and valid only for supported trace formats; see the compatibility table. |

Omitting `traffic.stop` for a trace runs to end of trace. `max_virtual_time_seconds` is trace-only and
cannot be used for synthetic traffic. Trace source, format, and token/session content stay concrete in
a recommendation input; only a numeric trace-load field can be a domain.

`speedup: N` divides authored timing by `N`; for example, `2` replays the timing twice as fast. For
Mooncake session traces it scales both first-turn arrival timestamps and inter-turn delays. For
agentic traces it scales root-node timestamps and the combined dependency delay and tool wait.
`speedup` is deliberately rejected with `load.type: concurrency`: concurrency replaces authored
first-arrival pacing, and inter-turn or dependency delays remain unscaled.

### Trace Format Compatibility

| Format | JSONL Unit | Allowed Load | `speedup` | `max_virtual_time_seconds` | Other Version 1 Constraints |
|---|---|---|---|---|---|
| `mooncake` | One request or session turn with a full prompt | `trace_timestamps`, `concurrency` | Timestamp load only | Supported | None specific to the format. |
| `mooncake-delta` | One session turn; follow-up input is only the new input delta | `trace_timestamps`, `concurrency` | Timestamp load only | Supported | Aggregated deployment only; `planner.policy` must be `disabled`. |
| `agentic_mooncake` | One request node in a dependency graph | `trace_timestamps` | Supported | Not supported; omit it | Aggregated deployment only; `planner.policy` must be `disabled`. |
| `applied_compute_agentic` | One complete session, expanded into `num_turns + 1` requests | `concurrency` | Not supported; omit it | Supported | Source rows have no first-turn timestamps. |
| `dynamo` standard trace | Native request-trace records, possibly across multiple files | `trace_timestamps`, `concurrency` | Timestamp load only | Supported | The embedded trace block size is authoritative. |
| `dynamo` agentic trace | Native agentic request-trace records, possibly across multiple files | `trace_timestamps` | Supported | Not supported; omit it | Aggregated deployment only; `planner.policy` must be `disabled`. |

The `dynamo` loader detects whether its records are standard or agentic and applies the corresponding
row above. If `traffic.source.block_size` is supplied for `dynamo`, it must match the embedded block
size. For the other formats, `block_size` is the trace hash-block size used to reconstruct prompts.

### Mooncake and Mooncake Delta JSONL

Both formats use the same row schema. Rows with the same `session_id` are turns in file order.

| Field | Required | Semantics |
|---|---|---|
| `request_id` | No | Request identity. |
| `session_id` | No | Groups rows into a session; an omitted value creates a one-row session. |
| `input_length` or `input_tokens` | No | Input token count; defaults to the capacity represented by `hash_ids`. |
| `output_length` or `output_tokens` | Yes | Output token count. |
| `output_token_ids` | No | Exact output tokens; its length must equal the output token count. |
| `hash_ids` | Yes | Prompt hash blocks at `traffic.source.block_size`. |
| `timestamp` or `created_time` | Conditional | Virtual timestamp in milliseconds. Required for the first row of every session under `trace_timestamps`. |
| `delay` or `delay_ms` | No | Inter-turn delay in milliseconds. It must be omitted or zero on a session's first row; later rows use it instead of a timestamp difference. |
| `priority`, `strict_priority`, `policy_class` | No | Scheduling metadata. |

With `format: mooncake`, every row describes that turn's complete prompt. With
`format: mooncake-delta`, the first row describes the initial prompt, while each later row's input and
`hash_ids` describe only new input for that turn. Replay builds the next complete prompt by appending
the prior generated output and the new delta. Using `mooncake-delta` on a full-prompt trace would
double-count prior context.

### Agentic Mooncake JSONL

`agentic_mooncake` includes all Mooncake request fields above, but `request_id` is required, nonempty,
and unique. Each row is an independently schedulable request node rather than a turn inferred only
from session order. It adds these fields:

| Field | Default | Semantics |
|---|---:|---|
| `wait_for` | `[]` | Request IDs that must all complete first. Unknown IDs, self-dependencies, and cycles are rejected. |
| `delay` / `delay_ms` | `0` | Delay after the last dependency completes. |
| `tool_wait_ms` | `0` | Additional tool wait after dependencies; scheduling delay is `delay + tool_wait_ms`. |
| `timestamp` / `created_time` | `0` for roots | Ready time for a node with an empty `wait_for`; dependent-node timestamps do not control release. |
| `request_kind`, `branches`, `prefix_reset`, `tool_events` | Empty | Producer metadata accepted by the format. Version 1 scheduling is controlled by `wait_for`, and cache identity is carried by `hash_ids`; these metadata fields do not independently change replay behavior. |

A dependent node becomes ready after the latest request in `wait_for` completes, plus its `delay` and
`tool_wait_ms`. `speedup` therefore scales both authored root arrivals and those post-dependency waits.

### Applied Compute Agentic JSONL

Each row is one session with `num_turns`, `input_prompt_length`, arrays
`assistant_response_length`, `tool_call_output_length`, and `tool_call_latency`, plus
`final_assistant_response_length`. Each array length must equal `num_turns`; tool latency is in
seconds. The row expands to `num_turns` assistant/tool turns plus one final assistant request. The
input grows cumulatively by each assistant response and tool output. Because the format has no
first-session arrival timestamps, it requires `load.type: concurrency` and rejects `speedup`.

> [!NOTE]
> `max_virtual_time_seconds` limits the total simulated virtual time of one simulation or recommendation
> candidate. It is not a separate processing-time limit for each path in `traffic.source.paths`, and it
> is not a real wall-clock timeout. The limit is a soft scheduling cutoff: events at the cutoff are
> processed, replay stops before the first event after the cutoff, and requests still in flight can be
> reported as incomplete. The reported duration can extend slightly past the cutoff while an already
> running engine pass finishes.

## Engine

`engine` owns topology, model and runtime identity, worker pools, scheduling, KV cache, timing, and
prefill-to-decode transfer behavior.

```yaml
engine:
  mode: aggregated
  model: meta-llama/Llama-3.1-8B-Instruct
  hardware: H100-SXM-80GB
  backend: vllm
  backend_version: null
  context_length: 32768
  workers:
    aggregated:
      parallelism:
        replicas: 2
        tensor: 1
        pipeline: 1
        attention_data: 1
        moe_tensor: 1
        moe_expert: 1
      scheduler:
        max_batched_tokens: 8192
        max_sequences: 256
      kv_cache:
        block_size: 64
        prefix_caching: true
        capacity:
          type: default
          memory_fraction: 0.9
      timing:
        type: default
      startup_seconds: 0
```

### Engine Fields

| Path | Type | Simulate Default | Constraints |
|---|---|---:|---|
| `engine.mode` | enum | Required | `aggregated` or `disaggregated`. |
| `engine.model` | string | Required | Nonempty model identifier. Fixed during recommendation. |
| `engine.hardware` | string, role mapping, or `auto` | Required | A concrete hardware assignment; `recommend` also accepts `auto`. |
| `engine.backend` | enum | Required | `vllm`, `sglang`, or `trtllm`. |
| `engine.backend_version` | string or null | `null` | Fixed when set. |
| `engine.context_length` | integer | Required | Greater than zero. |
| `engine.workers` | mapping | Required | Must contain the roles required by the selected mode. |
| `engine.kv_transfer` | mapping | None | Disaggregated mode only. |

`engine.hardware: auto` is valid only in `recommend` and requires the hardware inventory under
`optimization.hardware`. The recommender searches the listed hardware SKUs subject to their available
GPU counts. Every recommended simulation YAML replaces `auto` with a concrete assignment. Aggregated
mode resolves to one hardware identifier. Disaggregated mode resolves either to one shared identifier
or to `{prefill: <sku>, decode: <sku>}`; each worker pool is homogeneous, but the two pools may use
different SKUs.

An aggregated configuration uses `workers.aggregated`. A disaggregated configuration uses
`workers.prefill` and `workers.decode`:

```yaml
engine:
  mode: disaggregated
  model: meta-llama/Llama-3.1-8B-Instruct
  hardware: H100-SXM-80GB
  backend: vllm
  context_length: 32768
  kv_transfer:
    bytes_per_token: auto
    bandwidth_gb_per_second: 400
    timing_mode: destination_missing
  workers:
    prefill:
      parallelism: {replicas: 2, tensor: 2, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
      scheduler: {max_batched_tokens: 8192, max_sequences: 64}
      kv_cache:
        block_size: 64
        prefix_caching: true
        capacity: {type: default, memory_fraction: 0.9}
      timing: {type: default}
      startup_seconds: 0
    decode:
      parallelism: {replicas: 4, tensor: 1, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
      scheduler: {max_batched_tokens: 8192, max_sequences: 256}
      kv_cache:
        block_size: 64
        prefix_caching: false
        capacity: {type: default, memory_fraction: 0.9}
      timing: {type: default}
      startup_seconds: 0
```

All worker roles share the top-level model, backend, backend version, and context length in version 1.
A disaggregated concrete hardware mapping may select different SKUs for prefill and decode; other
role-specific model or backend selection is rejected. If `engine.mode` is a recommendation domain
containing both modes, `workers` declares all three roles. Each concrete candidate retains only the
role or roles active for its selected mode.

### Worker Fields

The table lists both the concrete value used by `simulate` when a field is omitted and the domain
used by `recommend` when no preset, concrete value, or explicit domain supplies that knob.

| Path Under a Worker | Type | Simulate Default | Recommend Default Domain | Constraints |
|---|---|---:|---|---|
| `parallelism` | mapping or `auto` | All fields `1` | `auto` | `auto` is recommend-only and resolves the complete parallel configuration. |
| `parallelism.preset` | string or null | `null` | `null` | One backend-declared whole-parallelism preset; see [Preset and Default-Domain Rules](#preset-and-default-domain-rules). |
| `parallelism.replicas` | integer | `1` | Feasible positive values within the GPU budget | Greater than zero. |
| `parallelism.tensor` | integer | `1` | Feasible values from the parallelism registry | Greater than zero and feasible for the model, backend, hardware, and budget. |
| `parallelism.pipeline` | integer | `1` | Feasible values from the parallelism registry | Greater than zero and feasible for the model, backend, hardware, and budget. |
| `parallelism.attention_data` | integer | `1` | Feasible values from the parallelism registry | Greater than zero and feasible for the model, backend, hardware, and budget. |
| `parallelism.moe_tensor` | integer | `1` | Feasible values from the parallelism registry | Greater than zero and compatible with the model/backend. |
| `parallelism.moe_expert` | integer | `1` | Feasible values from the parallelism registry | Greater than zero and compatible with the model/backend. |
| `scheduler.max_batched_tokens` | integer | `8192` | Prefill/aggregated: `{choices: [8192, 16384, 32768]}`; decode: `{choices: [8192]}` | Greater than zero. |
| `scheduler.max_sequences` | integer | `256` | Prefill: `{choices: [1, 2, 4, 8, 16, 32, 64, 128, 256]}`; aggregated/decode: `{choices: [256, 512, 1024]}` | Greater than zero. |
| `kv_cache.block_size` | integer | vLLM `64`; SGLang `1`; TensorRT-LLM `32` | Fixed at the backend default | Greater than zero and supported by the backend. |
| `kv_cache.prefix_caching` | boolean | Aggregated/prefill `true`; decode `false` | Fixed at the role default | Must be supported by the backend. |
| `kv_cache.capacity.type` | enum | `default` | `{choices: [default]}` | `default` or `fixed`. |
| `kv_cache.capacity.memory_fraction` | number | vLLM/TensorRT-LLM `0.9`; SGLang `0.88` | Fixed at the backend default | Greater than `0` and no greater than `1`; valid only for `type: default`. |
| `kv_cache.capacity.blocks` | integer | None | No default domain | Required and greater than zero for `type: fixed`; physical KV blocks per worker rank. |
| `timing.type` | enum | `default` | `{choices: [default]}` | `default`, `fixed`, or `polynomial`. |
| `timing.prefill_ms` | number | None | No default domain | Required, finite, and nonnegative for `type: fixed`. |
| `timing.decode_ms` | number | None | No default domain | Required, finite, and nonnegative for `type: fixed`. |
| `startup_seconds` | number | `0` | Fixed at `0` | Nonnegative. |

`parallelism` uses exactly one of these shapes:

```yaml
# Complete automatic search in recommend.
parallelism: auto
```

```yaml
# One backend-declared complete parallelism preset.
parallelism:
  preset: throughput
```

```yaml
# Explicit values or per-leaf recommendation domains.
parallelism:
  replicas: {range: {min: 1, max: 8, step: 1}}
  tensor: {choices: [1, 2, 4, 8]}
  pipeline: 1
  attention_data: 1
  moe_tensor: 1
  moe_expert: 1
```

A `parallelism` preset expands all six fields. It cannot be combined with a domain on a covered leaf;
the common preset rules still permit an intentional concrete leaf override.

`kv_cache.capacity.type: default` and `timing.type: default` replace the previous public name `aic`.
They select the stack's default capacity estimator and timing provider. The initial default registry
preserves current replay and Sweeper behavior, including the backend and role defaults in the table.
Backend-version-specific defaults are deferred beyond version 1; adding them changes the registry, not
the YAML shape.

`kv_cache.capacity.type: fixed` requires `blocks`, so users can directly provide cache size. It rejects
`memory_fraction`. Conversely, `type: default` rejects `blocks` and derives block count from model,
hardware, parallelism, block size, backend, and memory fraction.

The physical GPU count of a worker role is:

```text
parallelism.replicas * parallelism.tensor * parallelism.pipeline * parallelism.attention_data
```

`moe_tensor` and `moe_expert` describe partitioning within that physical shape and do not multiply
the GPU count again. Aggregated candidate GPU count is the aggregated worker count. Disaggregated
candidate GPU count is the sum of the prefill and decode worker counts.

### Prefill-to-Decode KV Transfer

| Path | Type | Simulate Default | Recommend Default Domain | Constraints |
|---|---|---:|---|---|
| `engine.kv_transfer.bytes_per_token` | integer or `auto` | `auto` | Fixed at `auto` | Positive when concrete. `auto` derives the KV footprint from the model, KV dtype, and parallelism. |
| `engine.kv_transfer.bandwidth_gb_per_second` | number or null | `null` | Fixed at `null` | Positive and finite when set. `null` disables modeled transfer delay. |
| `engine.kv_transfer.timing_mode` | enum | `full_prompt` | `{choices: [full_prompt, destination_missing]}` | `full_prompt` or `destination_missing`. |

`full_prompt` charges transfer for the complete prompt KV footprint. `destination_missing` charges
only the prompt KV not already present at the selected decode worker. `kv_transfer` is rejected for
aggregated mode. Its fields can be concrete or recommendation domains under the same rules as other
Engine fields.

## Router

```yaml
router:
  policy: round_robin
  prefill_load_model:
    type: none
```

| Path | Type | Simulate Default | Recommend Default Domain | Constraints |
|---|---|---:|---|---|
| `router.preset` | string or null | `null` | `null` | `default` or `load_aware`, or another stack-declared whole-Router preset. |
| `router.policy` | enum | `round_robin` | `{choices: [round_robin, kv_router]}` | `round_robin` or `kv_router`. |
| `router.prefill_load_model.type` | enum | `none` | `{choices: [none, aic]}` for `kv_router`; fixed `none` otherwise | `none` or `aic`. |
| `router.overlap_score_credit` | number | `1.0` | `{choices: [0.0, 0.5, 1.0]}` | Finite and nonnegative; `kv_router` only. |
| `router.prefill_load_scale` | number | `1.0` | `{choices: [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]}` | Finite and nonnegative; `kv_router` only. |
| `router.temperature` | number | `0.0` | `{choices: [0.0, 0.2, 0.5, 1.0]}` | Finite and nonnegative; `kv_router` only. |

These are all Router knobs exposed by version 1 of the simulation contract. Production-only Router
startup, indexer, cryptographic tracking, and admission-control fields are intentionally not generic
passthroughs. Adding one requires a typed public field and support from both execution stacks.

In `simulate`, every supplied knob is concrete. In `recommend`, a knob can use `choices` or `range`,
or it can inherit its table-defined default domain:

```yaml
router:
  policy: kv_router
  prefill_load_model:
    type: aic
  prefill_load_scale:
    choices: [0.5, 1.0, 2.0, 4.0]
  temperature:
    range: {min: 0.0, max: 1.0, step: 0.2}
```

`round_robin` requires `prefill_load_model.type: none` and has no KV-router-only knobs. The
`kv_router` policy may use either load model. Conditional domains apply only to the policy branch that
owns them; a round-robin candidate drops KV-router-only fields from its concrete output.

## Planner

```yaml
planner:
  policy: disabled
```

| Path | Type | Simulate Default | Recommend Default Domain | Constraints |
|---|---|---:|---|---|
| `planner.policy` | enum | `disabled` | `{choices: [disabled, planner]}` | `disabled` or `planner`. |
| `planner.target` | enum | `throughput` | Derived from `optimization.target` | `throughput`, `latency`, `load`, or `sla`; Planner policy only. |
| `planner.preset` | string or null | `null` | `null` | Whole-Planner preset; see the preset list below. |
| `planner.enable_throughput_scaling` | boolean | `true` | `{choices: [false, true]}` | Planner policy only. |
| `planner.enable_load_scaling` | boolean | `false` | `{choices: [false, true]}` | Planner policy only. |
| `planner.throughput_adjustment_interval_seconds` | integer | `180` | `{choices: [180, 600]}` | Positive; used when throughput scaling is enabled. |
| `planner.load_adjustment_interval_seconds` | integer | `5` | `{choices: [5, 10]}` | Positive and shorter than the throughput interval when throughput scaling is enabled. |
| `planner.max_num_fpm_samples` | integer | `64` | `{choices: [32, 64, 128]}` | Positive. |
| `planner.fpm_sample_bucket_size` | integer | `16` | `{choices: [4, 16, 64]}` | Positive perfect square. |
| `planner.load_scaling_down_sensitivity` | integer | `80` | `{choices: [70, 80, 90]}` | From `0` through `100`; used when load scaling is enabled. |
| `planner.load_min_observations` | integer | `5` | `{choices: [3, 5, 8]}` | Positive; used when load scaling is enabled. |
| `planner.load_predictor` | enum | `arima` | `{choices: [constant, arima, prophet, kalman]}` | Used when throughput scaling is enabled. |
| `planner.load_predictor_log1p` | boolean | `false` | `{choices: [false, true]}` | Used when throughput scaling is enabled. |
| `planner.prophet_window_size` | integer | `50` | `{choices: [20, 50]}` | Positive; `load_predictor: prophet` only. |
| `planner.kalman_q_level` | number | `1.0` | `{choices: [1.0, 10.0]}` | Positive; `load_predictor: kalman` only. |
| `planner.kalman_q_trend` | number | `0.1` | `{choices: [0.1, 1.0]}` | Positive; `load_predictor: kalman` only. |
| `planner.kalman_r` | number | `10.0` | `{choices: [5.0, 10.0]}` | Positive; `load_predictor: kalman` only. |
| `planner.kalman_min_points` | integer | `5` | `{choices: [3, 5]}` | Positive; `load_predictor: kalman` only. |
| `planner.min_workers` | integer | `1` | Fixed at `1` | Nonnegative; applies to aggregated mode and as the disaggregated fallback. |
| `planner.prefill_min_workers` | integer or null | `null` | Fixed at `null` | Positive when set; overrides `min_workers` for prefill. |
| `planner.decode_min_workers` | integer or null | `null` | Fixed at `null` | Positive when set; overrides `min_workers` for decode. |

These are all Planner knobs exposed by version 1. `simulate` may set a concrete `planner.target` and
otherwise uses `throughput`. In `recommend`, the target is not a search dimension: throughput targets
and Pareto map to `throughput`, `ttft` and `e2e_latency` map to `latency`, and goodput targets map to
`sla`. An explicitly supplied Planner target must equal that derived value.

The public whole-Planner presets are `disabled`, `throughput_180_5`, `throughput_600_5`,
`load_180_5`, `load_180_10`, `hybrid_180_5`, and `hybrid_600_5`. Each expands to a complete Planner
configuration, including default FPM sampling, load sensitivity, and predictor settings.

A Planner recommendation can select a whole-Planner preset:

```yaml
planner:
  policy: planner
  preset:
    choices: [throughput_180_5, throughput_600_5, hybrid_180_5]
```

Or, without a preset, it can provide only the domains it wants to change; every other knob receives
the default domain from the table:

```yaml
planner:
  policy: planner
  load_adjustment_interval_seconds:
    choices: [5, 10]
```

Planner runtime minimums and recommendation candidate GPU constraints are separate controls:

- `planner.min_workers`, `prefill_min_workers`, and `decode_min_workers` constrain runtime scaling
  during one simulated candidate run.
- `optimization.constraints` constrains which static candidate deployments the recommender evaluates.

When `planner.policy: disabled`, `preset` and all Planner knobs must be absent. Invalid conditional
combinations are rejected before search instead of being silently removed.

### Preset and Default-Domain Rules

Each lowest-level tunable object, `engine.workers.<role>.parallelism`, `router`, or `planner`, accepts
at most one whole-object `preset`. A preset value is either one concrete identifier or a `choices`
domain of identifiers. Presets do not support `range` or `auto`. Parallelism preset identifiers are
declared by the selected backend; Router and Planner identifiers are listed in their sections above.

Every preset covers every public knob in its object, including knobs for which it simply selects the
component default. It is therefore a complete baseline rather than a partial bundle or an ordered
collection of preset families.

For `simulate`, an omitted knob takes its table-defined concrete default. For `recommend`, values
resolve as follows:

1. If a preset is selected, expand it into the complete object.
2. Apply explicitly configured concrete values after preset expansion.
3. Reject any `choices`, `range`, or `auto` domain on a knob covered by the selected preset.
4. If no preset is selected, use an explicit concrete value to pin a knob, an explicit domain to
   replace its default domain, and the table-defined default domain for every omitted tunable knob.
5. Validate each resulting concrete component configuration.

If the entire optional `router` or `planner` section is omitted, it stays fixed at its simulation
default (`round_robin` or `disabled`). Default recommendation domains apply only inside a tunable
object that is explicitly present in the input.

For example, a recommendation that supplies a range only for
`planner.load_adjustment_interval_seconds` searches that range and the default domains of all other
Planner knobs. A user who wants a knob fixed at its simulate default supplies that concrete value.

The following is invalid because the selected parallelism preset already controls `replicas`:

```yaml
engine:
  workers:
    decode:
      parallelism:
        preset: throughput
        replicas: {range: {min: 1, max: 8, step: 1}}
```

The same field set to one concrete integer remains a permitted intentional override. Errors name the
preset and every conflicting domain path. If domains produce duplicate concrete configurations, the
result records one candidate and emits a warning.

`simulation.resolved.yaml` and each YAML under `recommendations/` omit `preset` and contain the fully
expanded concrete fields. `trials.jsonl` records both the selected preset identifier and expanded
fields for audit. `recommendation.resolved.yaml` retains the original preset selector and domains.

## Evaluation

```yaml
evaluation:
  sla:
    ttft_ms: 500
    itl_ms: 50
```

An SLA uses one of these forms:

```yaml
evaluation:
  sla:
    ttft_ms: 500
    itl_ms: 50
```

```yaml
evaluation:
  sla:
    e2e_ms: 3000
```

All thresholds are positive finite numbers. `ttft_ms` and `itl_ms` must be supplied together. They
cannot be combined with `e2e_ms` in version 1.

`goodput` and `goodput_per_gpu` optimization require either SLA form. Planner throughput scaling uses
the `ttft_ms` plus `itl_ms` form when the recommendation target is SLA-based.

Evaluation fields are always concrete and are not search dimensions.

## Recommendation Domains

A field in a recommendation input is either a concrete value or one explicit domain object.

### Choices

```yaml
engine:
  backend:
    choices: [vllm, sglang]
```

`choices` must be nonempty and contain unique values valid for the field. A bare YAML sequence never
means a search domain.

### Numeric Range

```yaml
engine:
  workers:
    aggregated:
      parallelism:
        replicas:
          range:
            min: 1
            max: 8
            step: 1
            scale: linear
```

| Range Field | Type | Default | Constraints |
|---|---|---:|---|
| `min` | integer or number | Required | Finite and no greater than `max`. |
| `max` | same as `min` | Required | Finite and no less than `min`. |
| `step` | same as range | None | Positive; allowed only for `linear`. Required for integer linear ranges. |
| `scale` | enum | `linear` | `linear` or `log`. |

For `log`, `min` must be positive and `step` is rejected. Integer-valued fields always materialize
integers, including when sampled from a log range.

### Feasible Parallelism

```yaml
engine:
  workers:
    aggregated:
      parallelism: auto
```

The scalar string `auto` is accepted only for the complete worker `parallelism` object in `recommend`.
It requests feasible tuples of `replicas`, tensor parallelism, pipeline parallelism, attention data
parallelism, MoE tensor parallelism, and MoE expert parallelism. This preserves compatibility among
the dimensions instead of taking an invalid Cartesian product of independently automatic leaves.

An omitted `parallelism` object also uses `auto`, its default recommendation domain. In `simulate`,
`parallelism` must resolve from a concrete mapping or preset; `auto` is rejected. An explicit mapping
may put `choices` or `range` on its leaves, but a leaf cannot itself be `auto`.

### Domain Validation

A field accepts at most one domain form. `choices` and `range` use mappings; feasible parallelism uses
the scalar `auto`. Domains are allowed only at documented searchable leaves:

- Engine mode, backend, `hardware: auto`, parallelism preset or `auto`, parallelism leaves, scheduler, KV cache,
  transfer, timing, and supported backend-specific
  fields.
- Router policy, load model, preset, and supported policy-specific fields.
- Planner policy, preset, and supported Planner-specific fields.
- Numeric fields under `traffic.load`.

The following stay concrete in version 1:

- Model, concrete hardware values, backend version, and context length.
- Traffic source, token lengths, session shape, trace contents, and stopping condition.
- Evaluation thresholds.
- Optimization target, hardware inventory, constraints, and optimizer controls.

## Optimization Goal

`optimization` exists only in a recommendation input:

```yaml
optimization:
  target: goodput_per_gpu
  constraints:
    min_candidate_gpus: null
    max_candidate_gpus: 32
```

### Targets

| Target | Direction | SLA Required |
|---|---|---|
| `throughput` | Maximize | No |
| `throughput_per_gpu` | Maximize | No |
| `throughput_per_user` | Maximize | No |
| `ttft` | Minimize | No |
| `e2e_latency` | Minimize | No |
| `goodput` | Maximize | Yes |
| `goodput_per_gpu` | Maximize | Yes |
| `pareto` | Fixed two-objective frontier | No |

`target: pareto` always computes the nondominated frontier over exactly
`throughput_per_gpu` and `throughput_per_user`. There is no `pareto_objectives` field and users cannot
replace either axis.

### Hardware Inventory

When `engine.hardware` is concrete, `optimization.hardware` is omitted. When
`engine.hardware: auto`, `optimization.hardware` is a required nonempty mapping from hardware
identifier to the number of available GPUs:

```yaml
optimization:
  hardware:
    H100-SXM-80GB: 16
    H200-SXM-141GB: 8
```

Every count is a positive integer. Aggregated candidates choose one SKU. Disaggregated candidates may
choose different prefill and decode SKUs, and their per-SKU GPU totals cannot exceed the corresponding
inventory counts.

### Candidate GPU Constraints

`min_candidate_gpus` and `max_candidate_gpus` are optional positive integers. The minimum cannot
exceed the maximum. They constrain the concrete GPU count defined under [Engine](#engine); they do not
limit Planner scaling during a candidate run.

## Optimizer Controls

```yaml
optimizer:
  algorithm: bayesian
  max_trials: 320
  parallelism: 16
  candidate_timeout_seconds: 600
  seed: 42
```

| Path | Type | Default | Constraints |
|---|---|---:|---|
| `optimizer.algorithm` | enum | `bayesian` | `bayesian` or `random`. |
| `optimizer.max_trials` | integer | `320` | Greater than zero. |
| `optimizer.parallelism` | integer | `16` | Greater than zero. |
| `optimizer.candidate_timeout_seconds` | number | `600` | Greater than zero. |
| `optimizer.seed` | integer | `42` | Nonnegative. |

Algorithm-specific public controls, when supported, are explicit typed fields directly under
`optimizer`. Unknown controls or controls incompatible with the selected algorithm are rejected.

`max_trials` is the total recommendation-run budget. This contract does not define optimizer
internals, how trials are allocated among conditional branches, process or thread models, result
caching, or timeout enforcement.

## Complete Simulation Example

```yaml
traffic:
  source:
    type: synthetic
    input_tokens: 1024
    output_tokens: 128
  load:
    type: poisson
    requests_per_second: 8
    seed: 42
  stop:
    requests: 100

engine:
  mode: aggregated
  model: meta-llama/Llama-3.1-8B-Instruct
  hardware: H100-SXM-80GB
  backend: vllm
  backend_version: null
  context_length: 32768
  workers:
    aggregated:
      parallelism:
        replicas: 2
        tensor: 1
        pipeline: 1
        attention_data: 1
        moe_tensor: 1
        moe_expert: 1
      scheduler:
        max_batched_tokens: 8192
        max_sequences: 256
      kv_cache:
        block_size: 64
        prefix_caching: true
        capacity:
          type: default
          memory_fraction: 0.9
      timing:
        type: default
      startup_seconds: 0

router:
  policy: round_robin
  prefill_load_model: {type: none}

planner:
  policy: disabled

evaluation:
  sla:
    ttft_ms: 500
    itl_ms: 50
```

## Scalar Recommendation Example

```yaml
traffic:
  source:
    type: synthetic-session
    new_input_tokens_per_turn: 1024
    output_tokens_per_turn: 128
    session: {turns: 4, shared_prefix_ratio: 0, prefix_groups: 0, inter_turn_delay_ms: 1000}
  load:
    type: poisson
    sessions_per_second: {range: {min: 4, max: 32, step: 4, scale: linear}}
    seed: 42
  stop:
    sessions_per_load_unit: 10

engine:
  mode: {choices: [aggregated, disaggregated]}
  model: meta-llama/Llama-3.1-8B-Instruct
  hardware: auto
  backend: {choices: [vllm, sglang]}
  backend_version: null
  context_length: 32768
  workers:
    aggregated:
      parallelism: auto
      scheduler: {max_batched_tokens: {choices: [8192, 16384]}, max_sequences: {choices: [256, 512]}}
    prefill:
      parallelism: auto
      scheduler: {max_batched_tokens: 8192, max_sequences: 64}
    decode:
      parallelism: auto
      scheduler: {max_batched_tokens: 8192, max_sequences: 256}

router:
  policy: {choices: [round_robin, kv_router]}
  prefill_load_model: {type: none}

planner:
  policy: disabled

evaluation:
  sla: {ttft_ms: 500, itl_ms: 50}

optimization:
  target: goodput_per_gpu
  hardware:
    H100-SXM-80GB: 16
    H200-SXM-141GB: 32
  constraints:
    min_candidate_gpus: 1
    max_candidate_gpus: 32

optimizer:
  algorithm: bayesian
  max_trials: 320
  parallelism: 16
  candidate_timeout_seconds: 600
  seed: 42
```

Conditional validation applies after a domain is materialized. For example, a round-robin candidate
must resolve the load model to `none`; a recommendation must not rely on an invalid combination being
silently ignored.

## Pareto Recommendation Example

The following replaces the `optimization` mapping from the scalar example. The hardware inventory is
still required because that example uses `engine.hardware: auto`:

```yaml
optimization:
  target: pareto
  hardware:
    H100-SXM-80GB: 16
    H200-SXM-141GB: 32
  constraints:
    min_candidate_gpus: 1
    max_candidate_gpus: 32
```

A scalar recommendation ranks every feasible candidate and saves all of them. A Pareto
recommendation saves the complete nondominated front. Neither mode silently drops failed or
infeasible trials from the trial ledger.

## Outputs

Output controls are CLI-only. They never appear in an input or recommended YAML file.

### Simulation Directory

```text
<output-dir>/
├── run.json
├── simulation.resolved.yaml
├── prediction.json
└── requests.jsonl                 # only with --capture-per-request
```

- `run.json` records command identity, stack, timestamps, status, and artifact names.
- `simulation.resolved.yaml` is the validated concrete input after `--set` overrides and defaults.
- `prediction.json` contains aggregate predicted metrics and configuration-independent units.
- `requests.jsonl` contains one record per request when explicitly enabled.

### Recommendation Directory

```text
<output-dir>/
├── run.json
├── recommendation.resolved.yaml
├── trials.jsonl
└── recommendations/
    ├── index.json
    ├── 0001.yaml
    ├── 0002.yaml
    └── ...
```

- `recommendation.resolved.yaml` retains the validated domains and resolved optimizer defaults.
- `trials.jsonl` records every successful, failed, timed-out, and infeasible trial with its concrete
  candidate, status, metrics when available, and structured error when unavailable.
- `recommendations/index.json` provides rank or Pareto membership, objective values, constraints,
  and the corresponding YAML file.
- Each numbered YAML is a concrete simulation config. It excludes `optimization`, `optimizer`, and
  `preset`, contains no domains or `auto` values, and can be passed directly to
  `aisimulate simulate`.

For scalar optimization, `index.json` orders all feasible candidates from best to worst. For Pareto
optimization, it lists the complete nondominated front in a deterministic display order; that order
does not imply a scalar ranking.

### Existing Output Directories

Without `--overwrite`, the CLI rejects an existing nonempty output directory. With `--overwrite`, it
may replace only the known files and directories listed above. It must preserve unrelated files and
must not recursively clear an arbitrary directory.

### Standard Output

`--format table` prints a concise human-readable summary. `--format json` prints the same summary as
one JSON value for shell automation. Durable artifact formats do not change with this option.

## Errors and Exit Codes

| Exit Code | Meaning |
|---:|---|
| `0` | Successful simulation or recommendation. |
| `1` | Execution failure or a completed recommendation with no feasible candidate. |
| `2` | CLI syntax, YAML parsing, schema, domain, override, or unsupported-combination error. |
| `130` | Interrupted by the user. |

Configuration errors identify the input file and full field path:

```text
recommendation.yaml: traffic.load.sessions_per_second.range.min:
must be greater than 0, got 0
```

Combination errors name conflicting values and explain the supported contract:

```text
recommendation.yaml: router.prefill_load_model.type:
'aic' is incompatible with router.policy='round_robin'; use policy='kv_router' or type='none'
```

An unsupported stack/backend/policy option must fail explicitly. It must not be ignored or silently
translated to a different behavior.

## Legacy Command Mapping

The new interface conceptually replaces the existing entry points:

| Existing Surface | New Surface |
|---|---|
| `python -m aisimulate.replay` | `aisimulate simulate --stack engine` |
| `python -m dynamo.replay` | `aisimulate simulate --stack dynamo` |
| Engine-backed Sweeper wrapper | `aisimulate recommend --stack engine` |
| Dynamo-backed Sweeper wrapper | `aisimulate recommend --stack dynamo` |
| `python -m aisimulate.sweeper` validation entry point | `aisimulate recommend` validation before execution |

This is a command and schema mapping only. Compatibility aliases, deprecation periods, config
conversion, and code migration are outside this design.

## Review Items

The major structure and defaults above are proposed as the version 1 contract. Review should focus
on the remaining narrow naming and boundary decisions:

1. Confirm `planner.policy: disabled | planner`, or choose a more descriptive enabled-policy name.
2. Confirm `./aisimulate-output` as the default output directory rather than requiring
   `--output-dir` or generating a timestamped directory.

These are CLI schema questions. Their resolution does not require specifying underlying execution or
optimizer architecture.
