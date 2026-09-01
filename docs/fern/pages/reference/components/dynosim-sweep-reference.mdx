---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: DynoSim Sweep Reference
subtitle: Configuration domains, optimizer controls, and output for aisimulate recommend
---

`aisimulate recommend --stack dynamo` searches AISimulate engine configuration plus Dynamo Router
and Planner domains. Each selected result is a concrete prediction YAML. For a guided workflow, see
[Sweep DynoSim Configurations](../../cli/operations/simulation-with-dynosim/dynosim-sweeps.mdx).

## Command

```bash
aisimulate recommend --stack dynamo --config recommendation.yaml
```

| Option | Default | Purpose |
|---|---|---|
| `-c`, `--config` | Required | Recommendation YAML |
| `--stack` | `engine` | Set to `dynamo` to load the Dynamo runner and adapters |
| `--set PATH=YAML_VALUE` | None | Override a configuration path after loading YAML; repeatable |
| `--output-dir` | `./aisimulate-output` | Recommendation output directory |
| `--overwrite` | `false` | Replace known AISimulate output files while preserving unrelated files |
| `--format` | `table` | Standard-output format: `table` or `json` |

## Configuration Sections

Recommendation accepts the prediction sections plus search controls:

| Section | Required | Purpose |
|---|---|---|
| `traffic` | No | Workload source, load pattern, and stopping condition |
| `engine` | Yes | Backend, topology, parallelism, scheduler, KV cache, and timing domains |
| `router` | No | Dynamo routing domains |
| `planner` | No | Dynamo Planner domains and presets |
| `evaluation` | No | Service-level objective (SLO) thresholds |
| `optimization` | Yes | Objective, hardware resolution, and candidate GPU constraints |
| `optimizer` | No | Algorithm, trial budget, worker fan-out, timeout, and seed |

Unknown fields are rejected. Conditional validation runs after each candidate is materialized.

When `traffic` is omitted, recommendation uses the same default as prediction: 100 independent
synthetic requests at concurrency 10, with 1,024 input tokens and 128 output tokens per request.

## Domain Syntax

### Choices

Use `choices` for categorical or explicitly enumerated values:

```yaml
engine:
  backend: {choices: [vllm, sglang]}
router:
  policy: {choices: [round_robin, kv_router]}
```

### Numeric Ranges

Use `range` for numeric domains:

```yaml
traffic:
  load:
    type: constant_rate
    requests_per_second: {range: {min: 4, max: 32, step: 4, scale: linear}}
```

`scale` is `linear` by default. Use `log` only for positive bounds. A stepped integer range includes
its upper bound when the step lands on it.

### Presets

A preset covers every knob in its smallest configuration group.

- Omit `preset` or set it to `default` to use the built-in complete mappings.
- Supply a list of names or complete mappings to search those mappings as categorical choices.
- Set `preset: false` or `preset: {}` to search the group's independent knob domains.
- Do not combine an active preset with an independent domain covered by that preset.

Parallelism presets are special. The default invokes AISimulate's feasible parallel-configuration
enumeration and projection. A custom list treats each complete parallelism mapping as one flat
choice. With `preset: false`, each parallelism knob becomes an independent dimension before
feasibility filtering.

## Traffic Domains

Recommendation accepts `choices` or `range` on `traffic.load.concurrency`, request or session
rate, `traffic.load.speedup` for trace timestamps, and `traffic.load.fraction` for
KV-capacity-relative load. The
`kv_capacity_fraction` load type is recommendation-only: it derives candidate concurrency from
that candidate's estimated KV capacity. Fractions above 1 are valid and model oversubscribed load.

For a swept load, use `stop.requests_per_load_unit` or `stop.sessions_per_load_unit` to scale the
simulation length with each concrete candidate. Fixed `stop.requests` and `stop.sessions` keep the
same simulation length across the load domain.

## Optimization

`optimization.target` accepts:

| Target | Ordering |
|---|---|
| `throughput` | Highest output throughput |
| `throughput_per_gpu` | Highest output throughput per simulated GPU |
| `throughput_per_user` | Highest per-user output throughput |
| `goodput` | Highest SLO-compliant request throughput |
| `goodput_per_gpu` | Highest goodput per simulated GPU |
| `ttft` | Lowest Time to First Token (TTFT) |
| `e2e_latency` | Lowest end-to-end latency |
| `pareto` | Complete nondominated front over the public CLI's fixed pair: throughput per GPU and throughput per user |

`goodput` and `goodput_per_gpu` require `evaluation.sla` with either `e2e_ms` or both `ttft_ms` and
`itl_ms`.

The public recommendation schema does not expose `pareto_objectives`. Custom objective lists remain
available only through the legacy Sweeper Python SDK.

`optimization.constraints.min_candidate_gpus` and `max_candidate_gpus` bound the simulated GPU
footprint. When `engine.hardware: auto`, set one concrete `optimization.hardware` value.

Recommendation searches aggregated and disaggregated modes by default, and vLLM and SGLang
backends by default. TensorRT-LLM can be selected explicitly for aggregated mode. The default
`engine.context_length: max` resolves from the model's Hugging Face configuration.

## Optimizer Controls

| Field | Default | Purpose |
|---|---:|---|
| `optimizer.algorithm` | `bayesian` | `bayesian` or `random` search |
| `optimizer.max_trials` | `320` | Hard total suggestion budget across active topology branches |
| `optimizer.parallelism` | `16` | Concurrent candidate workers |
| `optimizer.candidate_timeout_seconds` | `600` | Wall-clock timeout per candidate |
| `optimizer.seed` | `42` | Stable base seed for branch samplers |

Failed, infeasible, unsupported, and cached suggestions consume the trial budget. Every active
topology branch must receive at least one trial.

## Output

Recommendation writes:

```text
aisimulate-output/
└── recommendations/
    ├── 0001.yaml
    ├── 0002.yaml
    └── ...
```

Each numbered YAML contains one concrete prediction configuration with no `optimization`,
`optimizer`, preset, domain, or `auto` value. Run any result directly:

```bash
aisimulate predict \
  --stack dynamo \
  --config aisimulate-output/recommendations/0001.yaml
```

Scalar objectives write all feasible candidates in deterministic best-to-worst order. Pareto writes
the complete nondominated front; its deterministic file order is not a scalar ranking.

## Related Pages

- [Run a DynoSim Simulation](../../cli/operations/simulation-with-dynosim/dynosim-replay.mdx)
- [DynoSim Replay CLI Reference](dynosim-replay-cli-reference.mdx)
- [DynoSim Architecture](../../developer-guide/knowledge-base/concepts/simulation/dynosim-architecture.md)
