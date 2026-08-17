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

## Presets and Default Ranges

The reference tables below use four core columns:

- **Knob** is the complete YAML path.
- **Default** is the concrete value used by `simulate` when the knob is omitted.
- **Default Range** is the recommendation domain used when preset search is disabled. `x` means the
  knob is non-sweepable and rejects any domain. `-` means the knob is sweepable, but its default
  domain is the singleton concrete default. Any displayed `choices` or `range` is searched by
  default.
- **Preset** names the smallest configuration object whose preset covers the knob. `-` means no
  preset covers it.
- **Rules** carries the type, conditional availability, and validation that would otherwise require
  repeated prose below the table.

A preset is a list of complete mappings. Each mapping must specify every knob belonging to the
smallest preset class shown in the table, including a `null` value for a conditionally inactive knob.
A mapping is one atomic candidate choice; values inside it are not independently combined.

For any preset-capable object in `recommend`, `preset` has these forms:

```yaml
# Omitted, or written explicitly: use the component's built-in default preset list.
preset: default
```

```yaml
# Replace the default preset list with complete atomic choices.
router:
  preset:
    - policy: round_robin
      prefill_load_model: {type: none}
      overlap_score_credit: null
      prefill_load_scale: null
      temperature: null
    - policy: kv_router
      prefill_load_model: {type: none}
      overlap_score_credit: 1.0
      prefill_load_scale: 1.0
      temperature: 0.0
```

```yaml
# Disable preset search. These two spellings are equivalent.
preset: false
# preset: {}
```

When `preset` is omitted or `default`, the built-in preset list is the default sweep space. A custom
list replaces it. A list entry missing any covered knob, containing an unknown knob, or containing a
`choices`, `range`, or `auto` domain is rejected.

The built-in list is versioned public configuration data owned by the component provider. It follows
the same complete-mapping validation as a user-provided list; it is not an opaque runtime mode.

When `preset` is `false` or `{}`, every covered sweepable knob becomes an independent sweep dimension.
An explicit concrete value pins the knob; an explicit `choices` or `range` replaces its table-defined
default range. An omitted `-` knob uses the singleton concrete default. An `x` knob stays pinned and
rejects a domain. The Sweeper evaluates the Cartesian product and rejects infeasible concrete
combinations. A preset and independent domains cannot be active on the same object.

Preset controls are recommendation-only. `simulation.resolved.yaml` and recommended simulation YAMLs
contain only the expanded concrete knobs.

If the optional `router` or `planner` section is absent, that component stays fixed at its concrete
default. Its default preset sweep is activated only when the section is present in a recommendation.

### Parallelism Preset Behavior

`engine.workers.<role>.parallelism` uses the same `preset: default` spelling as every other
preset-capable object:

```yaml
parallelism:
  preset: default
```

The built-in default follows the existing Sweeper projection algorithm below. It does not expose the
six YAML leaves as six independent optimizer parameters.

First, the Sweeper builds the legal configuration pool for each deployment-mode branch. It enumerates
worker sizes from the current `1, 2, 4, 8, 16` GPU ladder, with pipeline parallelism fixed at `1`, then
enumerates legal tensor, attention-data, MoE-tensor, and MoE-expert shapes. It applies model-width,
backend, real-silicon, KV-capacity, GPU-budget, and runner-capability filters. For every surviving
worker shape, it enumerates positive replica counts that fit the budget. A disaggregated pool contains
prefill/decode pairs whose combined GPU count fits the same budget. Aggregated and disaggregated modes
use separate optimizer studies; backend remains a categorical parameter within each study.

Second, each complete mapping is encoded into a smaller latent search space:

| Deployment | Latent Parameter | Optimizer Type | Encoding |
|---|---|---|---|
| Both | `used_gpu_ratio` | Continuous float | Total GPUs divided by the branch GPU budget; range is the minimum and maximum ratio in the legal pool, default clamped from `1.0`. |
| Aggregated | `agg_num_gpus_per_engine_target` | Log-scale discrete | GPUs per worker, `tensor * pipeline * attention_data`; feasible values come from the legal pool and the default is the pool value nearest its geometric midpoint. |
| Aggregated | `agg_attention_mode` | Categorical | `tp` when attention data parallelism is `1`, otherwise `dp`. |
| Aggregated MoE | `agg_ffn_mode` | Categorical | `ep` when MoE expert parallelism is greater than `1`, otherwise `tp`. |
| Disaggregated | `prefill_gpu_share` | Continuous float | Prefill-pool GPUs divided by total candidate GPUs; range comes from the legal pool, default clamped from `0.5`. |
| Disaggregated | `prefill_num_gpus_per_engine_target` | Log-scale discrete | Prefill GPUs per worker. |
| Disaggregated | `decode_num_gpus_per_engine_target` | Log-scale discrete | Decode GPUs per worker. |
| Disaggregated | `prefill_attention_mode`, `decode_attention_mode` | Categorical | Per-role `tp` or `dp`. |
| Disaggregated MoE | `prefill_ffn_mode`, `decode_ffn_mode` | Categorical | Per-role `ep` or `tp`. |

The latent parameter names retain the existing Sweeper's `engine` wording; in this public schema,
`num_gpus_per_engine_target` means GPUs per worker.

Only `used_gpu_ratio` and, for disaggregated mode, `prefill_gpu_share` are continuous parallelism
parameters. GPUs per worker are discrete values sampled on a log scale; attention and FFN modes are
categorical. Replica count is not sampled directly: together, total GPU ratio and GPUs-per-worker
targets express the desired replica footprint. Constant latent parameters are omitted from the study
and injected at their defaults.

Third, every optimizer suggestion is snapped back to one complete mapping from the legal pool:

1. Remove mappings that do not support the suggested backend.
2. Count categorical mismatches for attention and FFN modes, and retain only mappings with the minimum
   mismatch count. An exact mode match wins whenever one exists.
3. Compute normalized squared distance over the numeric latent parameters. Ratios use linear values;
   each GPUs-per-worker target uses `log2`. Each dimension is normalized by its backend-compatible
   minimum-to-maximum span, and a constant dimension contributes zero:

   ```text
   distance = sum(((transform(actual) - transform(requested)) / span) ^ 2)
   ```

4. Select the mapping with minimum distance. Ties are deterministic: compare
   `(tensor, pipeline, attention_data, moe_tensor, moe_expert, replicas)` for aggregated mode, or the
   concatenated prefill tuple followed by the decode tuple for disaggregated mode.

The selected mapping supplies the concrete six YAML fields. Trial metadata records requested latent
features, actual snapped features, projection distance, whether a categorical mode was projected, and
the final complete parallel configuration.

A user-provided preset is a list of complete parallelism mappings:

```yaml
parallelism:
  preset:
    - {replicas: 1, tensor: 1, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
    - {replicas: 2, tensor: 2, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
```

Unlike the built-in default preset, this list is kept flat: each complete mapping is one categorical
choice and the Sweeper does not decompose it. To search independent dimensions, disable the preset
and provide zero or more per-knob domains:

```yaml
parallelism:
  preset: false
  replicas: {range: {min: 1, max: 8, step: 1}}
  tensor: {choices: [1, 2, 4, 8]}
```

Omitted parallelism knobs then use their table-defined default ranges, and the Sweeper evaluates the
Cartesian product before feasibility filtering.

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

### Traffic Fields

| Knob | Default | Default Range | Preset | Rules |
|---|---:|---|---|---|
| `traffic.source.type` | `synthetic` | `x` | `-` | `synthetic`, `synthetic-session`, or `trace`. |
| `traffic.source.input_tokens` | `1024` | `x` | `-` | Positive; `synthetic` only. |
| `traffic.source.output_tokens` | `128` | `x` | `-` | Positive; `synthetic` only. |
| `traffic.source.new_input_tokens_per_turn` | `1024` | `x` | `-` | Positive; `synthetic-session` only. |
| `traffic.source.output_tokens_per_turn` | `128` | `x` | `-` | Positive; `synthetic-session` only. |
| `traffic.source.session.turns` | `4` | `x` | `-` | At least `2`. |
| `traffic.source.session.shared_prefix_ratio` | `0` | `x` | `-` | From `0` through `1`. |
| `traffic.source.session.prefix_groups` | `0` | `x` | `-` | Nonnegative; positive when prefix ratio is positive. |
| `traffic.source.session.inter_turn_delay_ms` | `0` | `x` | `-` | Nonnegative. |
| `traffic.source.paths` | Required for trace | `x` | `-` | One path except `dynamo`, which permits multiple. |
| `traffic.source.format` | `mooncake` | `x` | `-` | See [Trace Format Compatibility](#trace-format-compatibility). |
| `traffic.source.block_size` | `512`; embedded for `dynamo` | `x` | `-` | Positive. |
| `traffic.load.type` | `concurrency` | `x` | `-` | Synthetic: `concurrency`, `poisson`, `constant_rate`, or `kv_capacity_fraction`; trace: `trace_timestamps` or `concurrency`. |
| `traffic.load.concurrency` | `1` | `-` | `-` | Positive integer; explicit domains are allowed in `recommend`. |
| `traffic.load.requests_per_second` | `null` | `-` | `-` | Positive; synthetic request open-loop load only. |
| `traffic.load.sessions_per_second` | `null` | `-` | `-` | Positive; synthetic session open-loop load only. |
| `traffic.load.seed` | `42` | `x` | `-` | Nonnegative; `poisson` only. |
| `traffic.load.fraction` | `null` | `-` | `-` | Positive finite number; `kv_capacity_fraction` only and may exceed `1`. |
| `traffic.load.speedup` | `1` | `-` | `-` | Positive; trace timestamp load only. |
| `traffic.stop.requests` | `100` for default traffic | `x` | `-` | Positive integer; synthetic request source only. |
| `traffic.stop.requests_per_load_unit` | `null` | `x` | `-` | Positive; synthetic request source only. |
| `traffic.stop.sessions` | `null` | `x` | `-` | Positive integer; synthetic session source only. |
| `traffic.stop.sessions_per_load_unit` | `null` | `x` | `-` | Positive; synthetic session source only. |
| `traffic.stop.max_virtual_time_seconds` | `null` | `x` | `-` | Positive; supported trace formats only. |

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

### Synthetic Load and Stop

Both synthetic source types support closed-loop concurrency, Poisson arrivals, constant-rate
arrivals, and recommendation-only KV-capacity-relative load as listed in the Traffic table.

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

The stop fields follow the source unit. The fixed-count field is a positive integer. The
load-relative field is
a positive number and resolves to `max(1, round(count_per_load_unit * load_unit))`. The load unit is
concurrency for `concurrency` and resolved `kv_capacity_fraction` traffic, requests per second for a
`synthetic` open-loop source, or sessions per second for a `synthetic-session` open-loop source.

For `synthetic-session`, a session with four turns contributes four requests but only one unit to the
load and stopping condition.

`sessions_per_second` controls the arrival rate of new sessions. For a multi-turn session, it schedules
the first turn; later turns follow that session's completion and `inter_turn_delay_ms` rules and do not
count as new load arrivals. `requests_per_second` schedules independent single requests.

In a recommendation input, source type, token fields, session shape, and stopping condition stay
concrete. Only `traffic.load` rows whose Default Range is not `x` can be search domains. A
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

`<role>` is `aggregated`, `prefill`, or `decode` as selected by `engine.mode`.

| Knob | Default | Default Range | Preset | Rules |
|---|---:|---|---|---|
| `engine.mode` | `aggregated` | `{choices: [aggregated, disaggregated]}` | `-` | `aggregated` or `disaggregated`. |
| `engine.model` | Required | `x` | `-` | Nonempty and fixed during recommendation. |
| `engine.hardware` | Required | `auto` | `-` | Concrete assignment; `recommend` also accepts `auto` with `optimization.hardware`. |
| `engine.backend` | `vllm` | `{choices: [vllm]}` | `-` | `vllm`, `sglang`, or `trtllm`; explicit choices may include supported alternatives. |
| `engine.backend_version` | `null` | `x` | `-` | Fixed when set. |
| `engine.context_length` | Required | `x` | `-` | Positive. |
| `engine.workers` | Required | `x` | `-` | Aggregated role or prefill plus decode roles. |
| `engine.workers.<role>.parallelism.preset` | `default` in `recommend` | `x` | `-` | Default list, complete mapping list, `false`, or `{}`. |
| `engine.workers.<role>.parallelism.replicas` | `1` | Feasible positive values within GPU budget | `parallelism` | Positive. |
| `engine.workers.<role>.parallelism.tensor` | `1` | Feasible registry values | `parallelism` | Positive and model/backend compatible. |
| `engine.workers.<role>.parallelism.pipeline` | `1` | Feasible registry values | `parallelism` | Positive and model/backend compatible. |
| `engine.workers.<role>.parallelism.attention_data` | `1` | Feasible registry values | `parallelism` | Positive and model/backend compatible. |
| `engine.workers.<role>.parallelism.moe_tensor` | `1` | Feasible registry values | `parallelism` | Positive and model/backend compatible. |
| `engine.workers.<role>.parallelism.moe_expert` | `1` | Feasible registry values | `parallelism` | Positive and model/backend compatible. |
| `engine.workers.<role>.scheduler.max_batched_tokens` | `8192` | Prefill/aggregated: `{choices: [8192, 16384, 32768]}`; decode: `-` | `-` | Positive. |
| `engine.workers.<role>.scheduler.max_sequences` | `256` | Prefill: `{choices: [1, 2, 4, 8, 16, 32, 64, 128, 256]}`; aggregated/decode: `{choices: [256, 512, 1024]}` | `-` | Positive. |
| `engine.workers.<role>.kv_cache.block_size` | vLLM `64`; SGLang `1`; TensorRT-LLM `32` | `-` | `-` | Positive and backend-supported. |
| `engine.workers.<role>.kv_cache.prefix_caching` | Aggregated/prefill `true`; decode `false` | `-` | `-` | Backend-supported. |
| `engine.workers.<role>.kv_cache.capacity.type` | `default` | `-` | `-` | `default` or `fixed`. |
| `engine.workers.<role>.kv_cache.capacity.memory_fraction` | vLLM/TensorRT-LLM `0.9`; SGLang `0.88` | `-` | `-` | `(0, 1]`; `default` capacity only. |
| `engine.workers.<role>.kv_cache.capacity.blocks` | `null` | `-` | `-` | Positive and required for `fixed` capacity. |
| `engine.workers.<role>.timing.type` | `default` | `-` | `-` | `default`, `fixed`, or `polynomial`. |
| `engine.workers.<role>.timing.prefill_ms` | `null` | `-` | `-` | Nonnegative and required for `fixed` timing. |
| `engine.workers.<role>.timing.decode_ms` | `null` | `-` | `-` | Nonnegative and required for `fixed` timing. |
| `engine.workers.<role>.startup_seconds` | `0` | `-` | `-` | Nonnegative. |
| `engine.kv_transfer.bytes_per_token` | `auto` | `-` | `-` | Positive when concrete; disaggregated mode only. |
| `engine.kv_transfer.bandwidth_gb_per_second` | `null` | `-` | `-` | Positive when set; `null` disables transfer delay. |
| `engine.kv_transfer.timing_mode` | `full_prompt` | `{choices: [full_prompt, destination_missing]}` | `-` | Disaggregated mode only. |

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

| Knob | Default | Default Range | Preset | Rules |
|---|---:|---|---|---|
| `router.preset` | `default` in `recommend` | `x` | `-` | Default list, complete mapping list, `false`, or `{}`. |
| `router.policy` | `round_robin` | `{choices: [round_robin, kv_router]}` | `router` | `round_robin` or `kv_router`. |
| `router.prefill_load_model.type` | `none` | `{choices: [none, aic]}` | `router` | `aic` is KV-router-only. |
| `router.overlap_score_credit` | `1.0` | `{choices: [0.0, 0.5, 1.0]}` | `router` | Finite, nonnegative, and KV-router-only. |
| `router.prefill_load_scale` | `1.0` | `{choices: [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]}` | `router` | Finite, nonnegative, and KV-router-only. |
| `router.temperature` | `0.0` | `{choices: [0.0, 0.2, 0.5, 1.0]}` | `router` | Finite, nonnegative, and KV-router-only. |

`round_robin` requires `prefill_load_model.type: none` and has no KV-router-only knobs. The
`kv_router` policy may use either load model. A complete Router preset uses `null` for KV-router-only
knobs in a round-robin mapping. Production-only Router fields remain outside the version 1 contract.

## Planner

```yaml
planner:
  policy: disabled
```

| Knob | Default | Default Range | Preset | Rules |
|---|---:|---|---|---|
| `planner.preset` | `default` in `recommend` | `x` | `-` | Default list, complete mapping list, `false`, or `{}`. |
| `planner.policy` | `disabled` | `{choices: [disabled, planner]}` | `planner` | `disabled` or `planner`. |
| `planner.target` | `throughput` | `x` | `planner` | Derived from `optimization.target` in `recommend`. |
| `planner.enable_throughput_scaling` | `true` | `{choices: [false, true]}` | `planner` | Planner policy only. |
| `planner.enable_load_scaling` | `false` | `{choices: [false, true]}` | `planner` | Planner policy only. |
| `planner.throughput_adjustment_interval_seconds` | `180` | `{choices: [180, 600]}` | `planner` | Positive; throughput scaling only. |
| `planner.load_adjustment_interval_seconds` | `5` | `{choices: [5, 10]}` | `planner` | Positive and shorter than throughput interval when used. |
| `planner.max_num_fpm_samples` | `64` | `{choices: [32, 64, 128]}` | `planner` | Positive. |
| `planner.fpm_sample_bucket_size` | `16` | `{choices: [4, 16, 64]}` | `planner` | Positive perfect square. |
| `planner.load_scaling_down_sensitivity` | `80` | `{choices: [70, 80, 90]}` | `planner` | From `0` through `100`; load scaling only. |
| `planner.load_min_observations` | `5` | `{choices: [3, 5, 8]}` | `planner` | Positive; load scaling only. |
| `planner.load_predictor` | `arima` | `{choices: [constant, arima, prophet, kalman]}` | `planner` | Throughput scaling only. |
| `planner.load_predictor_log1p` | `false` | `{choices: [false, true]}` | `planner` | Throughput scaling only. |
| `planner.prophet_window_size` | `50` | `{choices: [20, 50]}` | `planner` | Positive; Prophet only. |
| `planner.kalman_q_level` | `1.0` | `{choices: [1.0, 10.0]}` | `planner` | Positive; Kalman only. |
| `planner.kalman_q_trend` | `0.1` | `{choices: [0.1, 1.0]}` | `planner` | Positive; Kalman only. |
| `planner.kalman_r` | `10.0` | `{choices: [5.0, 10.0]}` | `planner` | Positive; Kalman only. |
| `planner.kalman_min_points` | `5` | `{choices: [3, 5]}` | `planner` | Positive; Kalman only. |
| `planner.min_workers` | `1` | `-` | `planner` | Nonnegative. |
| `planner.prefill_min_workers` | `null` | `-` | `planner` | Positive when set. |
| `planner.decode_min_workers` | `null` | `-` | `planner` | Positive when set. |

These are all Planner knobs exposed by version 1. `simulate` may set a concrete `planner.target` and
otherwise uses `throughput`. In `recommend`, the target is not a search dimension: throughput targets
and Pareto map to `throughput`, `ttft` and `e2e_latency` map to `latency`, and goodput targets map to
`sla`. A complete recommendation preset uses `target: null`; materialization writes the derived
concrete value.

Planner runtime minimums and recommendation candidate GPU constraints are separate:

- `planner.min_workers`, `prefill_min_workers`, and `decode_min_workers` constrain runtime scaling
  during one simulated candidate run.
- `optimization.constraints` constrains which static candidate deployments the recommender evaluates.

When `planner.policy: disabled`, conditionally inactive Planner knobs are `null` in a complete preset
mapping and are omitted from the concrete simulation output.

## Evaluation

```yaml
evaluation:
  sla:
    ttft_ms: 500
    itl_ms: 50
```

| Knob | Default | Default Range | Preset | Rules |
|---|---:|---|---|---|
| `evaluation.sla.ttft_ms` | `null` | `x` | `-` | Positive; supplied with `itl_ms`. |
| `evaluation.sla.itl_ms` | `null` | `x` | `-` | Positive; supplied with `ttft_ms`. |
| `evaluation.sla.e2e_ms` | `null` | `x` | `-` | Positive; mutually exclusive with TTFT plus ITL. |

`goodput` and `goodput_per_gpu` optimization require either SLA form. Planner throughput scaling uses
the `ttft_ms` plus `itl_ms` form when the recommendation target is SLA-based.

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
        preset: false
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

### Domain Validation

A field accepts at most one domain form. Domains are allowed only where **Default Range** is not `x`:

- Engine mode, backend, `hardware: auto`, parallelism leaves, scheduler, transfer timing, and
  supported backend-specific fields.
- Router policy, load model, and supported policy-specific fields.
- Planner policy and supported Planner-specific fields.
- Traffic load intensity and timing fields marked `-` in the table.

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

| Knob | Default | Default Range | Preset | Rules |
|---|---:|---|---|---|
| `optimization.target` | `throughput` | `x` | `-` | Maximize `throughput`, `throughput_per_gpu`, `throughput_per_user`, `goodput`, or `goodput_per_gpu`; minimize `ttft` or `e2e_latency`; or compute `pareto`. |
| `optimization.hardware` | `null` | `x` | `-` | Required inventory mapping for `engine.hardware: auto`; positive GPU count per SKU. |
| `optimization.constraints.min_candidate_gpus` | `null` | `x` | `-` | Positive when set and no greater than the maximum. |
| `optimization.constraints.max_candidate_gpus` | `32` | `x` | `-` | Positive and bounded by hardware inventory. |

`pareto` is always the fixed `throughput_per_gpu` and `throughput_per_user` frontier. Goodput targets
require `evaluation.sla`. Aggregated candidates choose one hardware SKU; disaggregated candidates may
choose different prefill and decode SKUs without exceeding per-SKU inventory.

## Optimizer Controls

```yaml
optimizer:
  algorithm: bayesian
  max_trials: 320
  parallelism: 16
  candidate_timeout_seconds: 600
  seed: 42
```

| Knob | Default | Default Range | Preset | Rules |
|---|---:|---|---|---|
| `optimizer.algorithm` | `bayesian` | `x` | `-` | `bayesian` or `random`. |
| `optimizer.max_trials` | `320` | `x` | `-` | Positive total trial budget. |
| `optimizer.parallelism` | `16` | `x` | `-` | Positive. |
| `optimizer.candidate_timeout_seconds` | `600` | `x` | `-` | Positive wall-clock limit per candidate. |
| `optimizer.seed` | `42` | `x` | `-` | Nonnegative. |

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
      parallelism: {preset: default}
      scheduler: {max_batched_tokens: {choices: [8192, 16384]}, max_sequences: {choices: [256, 512]}}
    prefill:
      parallelism: {preset: default}
      scheduler: {max_batched_tokens: 8192, max_sequences: 64}
    decode:
      parallelism: {preset: default}
      scheduler: {max_batched_tokens: 8192, max_sequences: 256}

router:
  preset: false
  policy: {choices: [round_robin, kv_router]}
  prefill_load_model: {type: none}

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

- `recommendation.resolved.yaml` retains the effective validated preset mapping lists, independent
  domains, and resolved optimizer defaults.
- `trials.jsonl` records every successful, failed, timed-out, and infeasible trial with its concrete
  candidate, selected atomic preset mappings, status, metrics when available, and structured error
  when unavailable.
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
