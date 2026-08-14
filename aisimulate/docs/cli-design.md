---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: AISimulate CLI Design
subtitle: Draft public command, configuration, and output contract for simulation and recommendation
---

This document proposes one public command-line interface (CLI) for AISimulate prediction and
configuration recommendation. It unifies the user-facing concepts currently split across Replay and
Sweeper while keeping execution-stack details out of the configuration schema.

> [!WARNING]
> **Design draft.** This document defines a CLI contract for review. It is not an implementation plan
> and does not commit the project to compatibility before the design is approved.

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
  --set deployment.engines.aggregated.replicas=4
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
deployment: {}
router: {}
planner: {}
evaluation: {}
metadata: {}
```

`recommend` extends that model with:

```yaml
optimization: {}
optimizer: {}
```

The command determines the document type. There is no top-level `kind` or stack field.

| Section | `simulate` | `recommend` | Purpose |
|---|---|---|---|
| `traffic` | Required | Required | Request source, load shape, and stopping condition. |
| `deployment` | Required | Required | Model, hardware, backend, topology, and engine roles. |
| `router` | Optional | Optional | Request routing policy. Defaults to round robin. |
| `planner` | Optional | Optional | Runtime scaling policy. Defaults to disabled. |
| `evaluation` | Optional | Optional | Service-level objective (SLA) thresholds used for reporting and goals. |
| `metadata` | Optional | Optional | User-owned annotations with no simulation or search semantics. |
| `optimization` | Rejected | Required | Recommendation objective and candidate GPU constraints. |
| `optimizer` | Rejected | Optional | Public search controls. |

Unknown fields are rejected everywhere except inside `metadata`. Every semantic configuration knob is
an explicit, typed YAML field. The selected stack, backend, policy, timing model, or capacity model
determines which conditional fields are legal; there is no generic configuration passthrough mapping.

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
    input_tokens: 1024
    output_tokens: 128
    session:
      turns: 4
      shared_prefix_ratio: 0.5
      prefix_groups: 16
      inter_turn_delay_ms: 1000
  load:
    type: request_rate
    sessions_per_second: 8
  stop:
    sessions: 100
```

`synthetic-session` generates multi-turn sessions. It requires a `session` mapping, and requests
inside one session execute in turn order.

| Path | Type | Default | Constraints |
|---|---|---:|---|
| `traffic.source.type` | enum | Required | `synthetic-session`. |
| `traffic.source.input_tokens` | integer | Required | Greater than zero. |
| `traffic.source.output_tokens` | integer | Required | Greater than zero. |
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
| `request_rate` | Source-specific `*_per_second` | Open loop with constant inter-arrival times. |
| `kv_capacity_fraction` | `fraction` | Recommendation-only closed-loop load resolved relative to each candidate's KV capacity. |

The source-specific open-loop rate field is:

- `requests_per_second` for `source.type: synthetic`.
- `sessions_per_second` for `source.type: synthetic-session`.

The rate is a positive finite number. `concurrency` is a positive integer and counts independent
requests for `synthetic` or active sessions for `synthetic-session`. At most one turn of a session is
active at a time. `seed` is a nonnegative integer and defaults to `42`. `fraction` is greater than `0`
and no greater than `1`.

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
| `traffic.source.paths` | list of paths | Required | Nonempty. Path order is preserved. |
| `traffic.source.format` | enum | `mooncake` | `mooncake`, `dynamo`, or `applied_compute_agentic`. |
| `traffic.source.block_size` | integer or null | `null` | Greater than zero when set. |
| `traffic.load.type` | enum | Required | `trace_timestamps` or `concurrency`. |
| `traffic.load.speedup` | number | `1` | Positive; valid only with `trace_timestamps`. |
| `traffic.load.concurrency` | integer | None | Positive; required only with `concurrency`. |
| `traffic.stop.max_virtual_time_seconds` | number | None | Positive and valid only for trace traffic. |

Omitting `traffic.stop` for a trace runs to end of trace. `max_virtual_time_seconds` is trace-only and
cannot be used for synthetic traffic. Trace source, format, and token/session content stay concrete in
a recommendation input; only a numeric trace-load field can be a domain.

> [!NOTE]
> `max_virtual_time_seconds` limits the total simulated virtual time of one simulation or recommendation
> candidate. It is not a separate processing-time limit for each path in `traffic.source.paths`, and it
> is not a real wall-clock timeout. The limit is a soft scheduling cutoff: events at the cutoff are
> processed, replay stops before the first event after the cutoff, and requests still in flight can be
> reported as incomplete. The reported duration can extend slightly past the cutoff while an already
> running engine pass finishes.

## Deployment

`deployment` owns the engines so topology, role count, parallelism, scheduling, cache, and timing are
described together.

```yaml
deployment:
  mode: aggregated
  model: meta-llama/Llama-3.1-8B-Instruct
  hardware: H100-SXM-80GB
  backend: vllm
  backend_version: null
  context_length: 32768
  engines:
    aggregated:
      presets: {}
      replicas: 2
      parallelism:
        tensor: 1
        pipeline: 1
        attention_data: 1
        moe_tensor: 1
        moe_expert: 1
      scheduler:
        max_batched_tokens: 8192
        max_sequences: 256
      cache:
        block_size: 64
        prefix_caching: true
        capacity:
          type: aic
          memory_fraction: 0.9
      timing:
        type: aic
      startup_seconds: 0
```

### Deployment Fields

| Path | Type | Default | Constraints |
|---|---|---:|---|
| `deployment.mode` | enum | Required | `aggregated` or `disaggregated`. |
| `deployment.model` | string | Required | Nonempty model identifier. Fixed during recommendation. |
| `deployment.hardware` | string | Required | Nonempty hardware identifier. Fixed during recommendation. |
| `deployment.backend` | enum | Required | `vllm`, `sglang`, or `trtllm`. |
| `deployment.backend_version` | string or null | `null` | Fixed when set. |
| `deployment.context_length` | integer | Required | Greater than zero. |
| `deployment.engines` | mapping | Required | Must contain the roles required by the selected mode. |

An aggregated deployment uses `engines.aggregated`. A disaggregated deployment uses
`engines.prefill` and `engines.decode`:

```yaml
deployment:
  mode: disaggregated
  model: meta-llama/Llama-3.1-8B-Instruct
  hardware: H100-SXM-80GB
  backend: vllm
  context_length: 32768
  engines:
    prefill:
      presets: {}
      replicas: 2
      parallelism: {tensor: 2, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
      scheduler: {max_batched_tokens: 8192, max_sequences: 64}
      cache:
        block_size: 64
        prefix_caching: true
        capacity: {type: aic, memory_fraction: 0.9}
      timing: {type: aic}
      startup_seconds: 0
    decode:
      presets: {}
      replicas: 4
      parallelism: {tensor: 1, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
      scheduler: {max_batched_tokens: 8192, max_sequences: 256}
      cache:
        block_size: 64
        prefix_caching: true
        capacity: {type: aic, memory_fraction: 0.9}
      timing: {type: aic}
      startup_seconds: 0
```

All roles share the top-level model, hardware, backend, backend version, and context length in version
1. Role-specific model or hardware selection is rejected.

If `deployment.mode` is a recommendation domain containing both modes, `engines` must declare all
three roles. Each concrete candidate retains only the role or roles active for its selected mode.

### Engine Role Fields

| Path Under a Role | Type | Default | Constraints |
|---|---|---:|---|
| `presets` | mapping | `{}` | Named Engine knob bundles, grouped by preset family. |
| `replicas` | integer | `1` | Greater than zero. |
| `parallelism.tensor` | integer | `1` | Greater than zero. |
| `parallelism.pipeline` | integer | `1` | Greater than zero. |
| `parallelism.attention_data` | integer | `1` | Greater than zero. |
| `parallelism.moe_tensor` | integer | `1` | Greater than zero and compatible with the model/backend. |
| `parallelism.moe_expert` | integer | `1` | Greater than zero and compatible with the model/backend. |
| `scheduler.max_batched_tokens` | integer | Backend default | Greater than zero when set. |
| `scheduler.max_sequences` | integer | Backend default | Greater than zero when set. |
| `cache.block_size` | integer | Backend default | Greater than zero and supported by the backend. |
| `cache.prefix_caching` | boolean | `false` | Must be supported by the backend. |
| `cache.capacity.type` | enum | `aic` | `aic` or `fixed`. |
| `cache.capacity.memory_fraction` | number | `0.9` | Greater than `0` and no greater than `1`; AIC capacity only. |
| `timing.type` | enum | `aic` | `aic`, `fixed`, or `polynomial`. |
| `startup_seconds` | number | `0` | Nonnegative. |

The physical GPU count of a role is:

```text
replicas * tensor * pipeline * attention_data
```

`moe_tensor` and `moe_expert` describe partitioning within that physical shape and do not multiply
the GPU count again. Aggregated candidate GPU count is the aggregated role count. Disaggregated
candidate GPU count is the sum of the prefill and decode role counts.

Portable and backend-specific fields are written directly under the engine role, `cache.capacity`, or
`timing`, according to ownership. Every field is typed and validated. The selected `--stack`, backend,
capacity type, and timing type determine which conditional fields are legal.

Engine presets are scoped to one role. An aggregated, prefill, or decode role can independently select
backend-defined preset families for its parallelism, scheduler, cache, timing, or other Engine knobs.
In `recommend`, an Engine preset family can use `choices` just like Router and Planner presets:

```yaml
deployment:
  engines:
    decode:
      presets:
        scheduler:
          choices: [latency, throughput]
      scheduler:
        max_sequences: 256
```

Preset family names and identifiers are declared by the selected backend. The example names illustrate
the schema and are valid only when that backend declares them. The direct
`scheduler.max_sequences` field follows the common preset override and conflict rules below.

## Router

```yaml
router:
  policy: round_robin
  prefill_load_model:
    type: none
  presets: {}
```

| Path | Type | Default | Constraints |
|---|---|---:|---|
| `router.policy` | enum | `round_robin` | `round_robin` or `kv_router`. |
| `router.prefill_load_model.type` | enum | `none` | `none` or `aic`. |
| `router.presets` | mapping | `{}` | Named Router knob bundles, grouped by preset family. |

Router knobs are direct fields under `router`; load-model knobs are direct fields under
`router.prefill_load_model`. The names are stable configuration keys and do not have to copy existing
Router CLI flag spelling. Every supported key has a declared type, default, and validation rule.
Unknown keys and keys unsupported by the selected `--stack`, policy, or load model are rejected.

In `simulate`, every knob must be concrete. In `recommend`, a supported knob leaf can use
`choices` or `range`:

```yaml
router:
  policy: kv_router
  prefill_load_model:
    type: aic
  presets:
    behavior:
      choices: [null, load_aware]
  prefill_load_scale:
    choices: [0.5, 1.0, 2.0, 4.0]
  temperature:
    range: {min: 0.0, max: 1.0, step: 0.2}
```

The example also searches the optional `behavior` preset family; `null` means that no preset from
that family is applied.

`round_robin` requires `prefill_load_model.type: none`, empty `presets`, and no KV-router-only knobs.
The `kv_router` policy may use either load model. When `router.policy` is itself a domain, every
configured preset and direct knob must be legal for every policy choice. Use separate recommendation
inputs when policy branches require different preset or field shapes.

## Planner

```yaml
planner:
  policy: disabled
  target: throughput
  limits: {}
  presets: {}
```

| Path | Type | Default | Constraints |
|---|---|---:|---|
| `planner.policy` | enum | `disabled` | `disabled` or `planner`. |
| `planner.target` | enum | `throughput` | `throughput`, `latency`, or `sla`; `recommend` also accepts `auto`. |
| `planner.limits` | mapping | Planner defaults | Runtime scaling limits. Values must be concrete. |
| `planner.presets` | mapping | `{}` | Named Planner knob bundles, grouped by preset family. |

When the policy is `disabled`, planner-only `limits`, `presets`, and direct Planner knobs must be
absent. `simulate` requires a concrete target when the Planner is enabled. `recommend` may use
`target: auto`; each recommended YAML contains a concrete target.

Planner-specific knobs are direct fields under `planner`, alongside the common `policy`, `target`,
`limits`, and `presets` fields. Their names do not have to match existing CLI flags. Every key is typed
and validated, `simulate` requires concrete values, and supported leaves can be domains in `recommend`.

A Planner recommendation can search independent preset families and individual knobs together:

```yaml
planner:
  policy: planner
  target: sla
  limits: {}
  presets:
    scaling_policy:
      choices: [throughput_180_5, throughput_600_5, hybrid_180_5]
    fpm_sampling:
      choices: [small, default, large]
    load_sensitivity:
      choices: [aggressive, default, conservative]
  load_adjustment_interval_seconds:
    choices: [5, 10]
```

Planner runtime scaling limits and recommendation candidate GPU constraints are separate controls:

- `planner.limits` constrains scaling during one simulated candidate run.
- `optimization.constraints` constrains which static candidate deployments the recommender evaluates.

Omitting `planner.limits` uses Planner defaults. This CLI design does not duplicate the Planner's
internal configuration model.

### Preset Expansion and Conflicts

Engine-role `presets`, `router.presets`, and `planner.presets` use component-defined preset families.
Each family value is either one concrete preset identifier or a `choices` domain of preset
identifiers. Presets do not support `range` or `auto`. A nullable preset family can include `null` in
`choices` to mean that no preset from that family is applied.

Preset identifiers are public names validated against the selected stack, backend, Engine role,
policy, and target. A preset expands to a documented set of concrete component fields. It is a
convenience and search-space shorthand, not an opaque runtime mode.

For each candidate, values resolve in this order:

1. Start with the selected component defaults.
2. Expand the selected preset in each preset family.
3. Apply explicitly configured direct fields, which override values supplied by presets.
4. Validate the resulting concrete component configuration.

This precedence permits a named baseline with one or more intentional knob overrides. It also means
an explicit field can remove the practical difference between two presets; the input remains valid,
but the resulting concrete candidates can be identical.

Conflicts use these rules for Engine, Router, and Planner alike:

- A direct field and a preset can set the same knob. The direct field wins.
- Two selected presets can set the same knob to the same value. This is allowed.
- Two selected presets that set the same knob to different values are a configuration error. Preset
  family order and YAML declaration order never establish precedence.
- The fully expanded and overridden configuration must still satisfy all conditional rules. An
  invalid final combination is a configuration error even if each individual value is valid.

Errors name every preset and field participating in the conflict. In `recommend`, a preset domain
must not admit conflicting preset combinations; the CLI rejects that search space before evaluation.
If explicit domains produce duplicate concrete configurations after preset expansion and overrides,
the result records one concrete candidate and emits a warning.

`simulation.resolved.yaml` and each YAML under `recommendations/` omit `presets` and contain the fully
expanded concrete component fields. `trials.jsonl` records both the selected preset identifiers and
expanded fields for audit. `recommendation.resolved.yaml` retains the original preset selectors and
domains.

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

`goodput` and `goodput_per_gpu` optimization require either SLA form. `planner.target: sla` requires
the `ttft_ms` plus `itl_ms` form because the Planner consumes both thresholds.

Evaluation fields are always concrete and are not search dimensions.

## Recommendation Domains

A field in a recommendation input is either a concrete value or one explicit domain object.

### Choices

```yaml
deployment:
  backend:
    choices: [vllm, sglang]
```

`choices` must be nonempty and contain unique values valid for the field. A bare YAML sequence never
means a search domain.

### Numeric Range

```yaml
deployment:
  engines:
    aggregated:
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
parallelism:
  tensor: {auto: feasible}
  pipeline: 1
  attention_data: {auto: feasible}
  moe_tensor: {auto: feasible}
  moe_expert: {auto: feasible}
```

`{auto: feasible}` is accepted only on engine parallelism fields in version 1. It requests all
model-, hardware-, backend-, and budget-compatible values. It is not equivalent to omitting a field;
an omitted parallelism field defaults to `1`.

`planner.target: auto` is a separate Planner value, not a domain object.

### Domain Validation

A mapping cannot combine `choices`, `range`, and `auto`. Domains are allowed only at documented
searchable leaves:

- Deployment mode and backend.
- Engine preset families, replicas, parallelism, scheduler, cache, and supported backend-specific
  fields.
- Router policy, load model, preset families, and supported policy-specific fields.
- Planner policy, target, preset families, and supported Planner-specific fields.
- Numeric fields under `traffic.load`.

The following stay concrete in version 1:

- Model, hardware, backend version, and context length.
- Traffic source, token lengths, session shape, trace contents, and stopping condition.
- Evaluation thresholds.
- Optimization target, Pareto objective list, constraints, and optimizer controls.

## Optimization Goal

`optimization` exists only in a recommendation input:

```yaml
optimization:
  target: goodput_per_gpu
  pareto_objectives: null
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
| `e2e_latency` | Minimize | No |
| `goodput` | Maximize | Yes |
| `goodput_per_gpu` | Maximize | Yes |
| `pareto` | Per objective | If any objective is goodput-based |

For scalar targets, `pareto_objectives` must be null or omitted. For `target: pareto`, it is either:

- Null or omitted, which selects `[throughput_per_gpu, throughput_per_user]`.
- A list of at least two unique scalar targets.

The list cannot contain `pareto` itself.

### Candidate GPU Constraints

`min_candidate_gpus` and `max_candidate_gpus` are optional positive integers. The minimum cannot
exceed the maximum. They constrain the concrete deployment GPU count defined under
[Deployment](#deployment); they do not limit Planner scaling during a candidate run.

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

deployment:
  mode: aggregated
  model: meta-llama/Llama-3.1-8B-Instruct
  hardware: H100-SXM-80GB
  backend: vllm
  backend_version: null
  context_length: 32768
  engines:
    aggregated:
      presets: {}
      replicas: 2
      parallelism:
        tensor: 1
        pipeline: 1
        attention_data: 1
        moe_tensor: 1
        moe_expert: 1
      scheduler:
        max_batched_tokens: 8192
        max_sequences: 256
      cache:
        block_size: 64
        prefix_caching: true
        capacity:
          type: aic
          memory_fraction: 0.9
      timing:
        type: aic
      startup_seconds: 0

router:
  policy: round_robin
  prefill_load_model: {type: none}
  presets: {}

planner:
  policy: disabled
  target: throughput
  limits: {}
  presets: {}

evaluation:
  sla:
    ttft_ms: 500
    itl_ms: 50

metadata:
  name: baseline-8-rps
```

## Scalar Recommendation Example

```yaml
traffic:
  source:
    type: synthetic-session
    input_tokens: 1024
    output_tokens: 128
    session: {turns: 4, shared_prefix_ratio: 0, prefix_groups: 0, inter_turn_delay_ms: 1000}
  load:
    type: poisson
    sessions_per_second: {range: {min: 4, max: 32, step: 4, scale: linear}}
    seed: 42
  stop:
    sessions_per_load_unit: 10

deployment:
  mode: {choices: [aggregated, disaggregated]}
  model: meta-llama/Llama-3.1-8B-Instruct
  hardware: H100-SXM-80GB
  backend: {choices: [vllm, sglang]}
  backend_version: null
  context_length: 32768
  engines:
    aggregated:
      presets: {}
      replicas: {range: {min: 1, max: 8, step: 1}}
      parallelism: {tensor: {auto: feasible}, pipeline: 1, attention_data: {auto: feasible}, moe_tensor: 1, moe_expert: 1}
      scheduler: {max_batched_tokens: {choices: [4096, 8192]}, max_sequences: {choices: [128, 256]}}
      cache:
        block_size: 64
        prefix_caching: {choices: [false, true]}
        capacity: {type: aic, memory_fraction: 0.9}
      timing: {type: aic}
      startup_seconds: 0
    prefill:
      presets: {}
      replicas: {range: {min: 1, max: 4, step: 1}}
      parallelism: {tensor: {auto: feasible}, pipeline: 1, attention_data: {auto: feasible}, moe_tensor: 1, moe_expert: 1}
      scheduler: {max_batched_tokens: 8192, max_sequences: 64}
      cache: {block_size: 64, prefix_caching: true, capacity: {type: aic, memory_fraction: 0.9}}
      timing: {type: aic}
      startup_seconds: 0
    decode:
      presets: {}
      replicas: {range: {min: 1, max: 8, step: 1}}
      parallelism: {tensor: {auto: feasible}, pipeline: 1, attention_data: {auto: feasible}, moe_tensor: 1, moe_expert: 1}
      scheduler: {max_batched_tokens: 8192, max_sequences: 256}
      cache: {block_size: 64, prefix_caching: true, capacity: {type: aic, memory_fraction: 0.9}}
      timing: {type: aic}
      startup_seconds: 0

router:
  policy: {choices: [round_robin, kv_router]}
  prefill_load_model: {type: none}
  presets: {}

planner:
  policy: disabled
  target: throughput
  limits: {}
  presets: {}

evaluation:
  sla: {ttft_ms: 500, itl_ms: 50}

optimization:
  target: goodput_per_gpu
  constraints:
    min_candidate_gpus: 1
    max_candidate_gpus: 32

optimizer:
  algorithm: bayesian
  max_trials: 320
  parallelism: 16
  candidate_timeout_seconds: 600
  seed: 42

metadata:
  name: goodput-per-gpu-search
```

Conditional validation applies after a domain is materialized. For example, a round-robin candidate
must resolve the load model to `none`; a recommendation must not rely on an invalid combination being
silently ignored.

## Pareto Recommendation Example

The following replaces only the goal from the scalar example:

```yaml
optimization:
  target: pareto
  pareto_objectives:
    - throughput_per_gpu
    - throughput_per_user
  constraints:
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
  `presets`, contains no domains or `auto` values, and can be passed directly to
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
3. Confirm the direct typed fields for `cache.capacity.type: fixed` and the `fixed` and `polynomial`
   timing providers.
4. Confirm whether version 1 should list all three trace formats or expose only `mooncake` in the new
   public schema.

These are CLI schema questions. Their resolution does not require specifying underlying execution or
optimizer architecture.
