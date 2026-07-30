---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Spica Candidate Materialization
subtitle: How a search suggestion becomes a backend deployment and replay specification
---

> [!WARNING]
> **Experimental.** Spica is intended for evaluation and feedback, not production capacity
> planning. Its API, configuration schema, search results, and deployment output may change
> without a standard deprecation period.

Spica materializes each optimizer suggestion in the main process. It first unrolls the backend
selection, then asks each configured adapter to build its concrete replay configuration and runtime
hooks.

## Backend Sample

`unroll_sample` produces one flat backend sample. It contains:

- `deployment_mode`, `backend`, and the resolved `backend_version`;
- model, hardware, GPU-budget, context-length, and startup fields;
- the projected parallel shape, replica counts, and `used_gpus`;
- batching, block-size, memory-utilization, and prefix-caching fields for the active engine roles;
- concrete concurrency and KV-capacity diagnostics when `kv_load_ratio` is used.

An aggregated parallel selection expands to:

```yaml
deployment_mode: agg
tp: 4
pp: 1
attention_dp: 2
moe_tp: 1
moe_ep: 8
strategy: dep
replicas: 2
used_gpus: 16
```

A disaggregated selection uses `prefill_` and `decode_` prefixes and records both replica counts.

The backend sample contains no Planner, Router, or KVBM fields.

## Backend Deployment Specification

`build_backend_deployment` turns the sample into `BackendDeploymentSpec`. The specification
contains:

- backend name and resolved performance-model version;
- aggregated or prefill/decode engine-argument payloads;
- aggregated or prefill/decode worker counts;
- the concrete parallel configuration.

Engine argument payloads preserve the current replay inputs, including `aic_backend`,
`aic_system`, `aic_model_path`, tensor and attention-data parallelism, batching settings, and
optional startup or speculative-decoding values.

## Adapter Materialization

For each configured adapter, Spica passes:

- the prepared `AdapterSearchPlan`;
- only that adapter's decoded local optimizer selection;
- a `CandidateContext` containing the backend sample, `BackendDeploymentSpec`, and concrete
  concurrency.

The adapter returns `AdapterReplaySpec(config=..., runtime_hooks=...)`. Adapter configuration is
stored by adapter name; it is not flattened into the backend sample.

## Replay Specification

The complete version 1 replay payload is equivalent to:

```yaml
api_version: 1
backend_deployment:
  deployment_mode: agg
  backend: vllm
  backend_version: 0.11.0
  agg_engine_args: {}
  num_workers: 2
workload: {}
goal: {}
concurrency: null
adapters:
  dynamo.router:
    config:
      mode: kv_router
      overlap_score_credit: 0.5
      prefill_load_scale: 1.0
      router_temperature: 0.2
    runtime_hooks:
      - provider: dynamo.router
        kind: placement_policy
        api_version: 1
        config: {}
```

`RunnerCapabilities.require_compatible` validates the version, backend/topology pair, and hooks
before execution.

## Candidate Output

After replay and scoring, `Candidate.config` contains the backend sample and a nested adapter
mapping:

```yaml
backend: vllm
backend_version: 0.11.0
deployment_mode: agg
used_gpus: 16
adapters:
  dynamo.router:
    mode: kv_router
    overlap_score_credit: 0.5
    prefill_load_scale: 1.0
    router_temperature: 0.2
```

`Candidate.metrics` contains normalized replay metrics, while `Candidate.score` and optional
`Candidate.objectives` contain the optimization result.

## Serialization

`canonical_json` converts replay dataclasses, Pydantic models, enumerations, mappings, and sequences
to deterministic strict JSON. It sorts mapping keys and rejects non-finite values. The current
search cache intentionally remains keyed by the raw suggestion to preserve the pre-refactor
optimizer trajectory.
