---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Spica Search Space
subtitle: Backend fields and optional simulation-adapter search spaces
---

> [!WARNING]
> **Experimental.** Spica is intended for evaluation and feedback, not production capacity
> planning. Its API, configuration schema, search results, and deployment output may change
> without a standard deprecation period.

`SmartSearchConfig.search_space` contains only backend and deployment fields. Optional
feature-specific search spaces are mappings under `SmartSearchConfig.adapters`.

## Top-Level Shape

```yaml
search_space:
  model_name: deepseek-ai/DeepSeek-V3
  hardware_sku: h200_sxm
  gpu_budget: 32
  deployment_mode: [disagg, agg]
  backend: [vllm, sglang]

adapters:
  dynamo.router:
    search_space:
      mode: [round_robin, kv_router]
  dynamo.planner:
    search_space:
      scaling_policy: [disabled, load_180_5]
```

Adapter names are package entry-point names. The dot in `dynamo.router` or `dynamo.planner` is part
of the key.

## Backend and Deployment Fields

| Field | Type | Default | Role |
|---|---|---|---|
| `deployment_mode` | `list[str]` | `[disagg, agg]` | searched branch; `agg` or `disagg` |
| `backend` | `list[str]` | `[vllm]` | searched; `vllm`, `sglang`, or `trtllm` |
| `parallel_configs` | `list[dict]` | `[]` | optional pinned parallel menu; empty generates legal configurations |
| `model_name` | `str` | required | pinned model identifier |
| `hardware_sku` | `str` | required | pinned AI Configurator system identifier |
| `gpu_budget` | `int` | `32` | maximum GPUs per candidate |
| `min_gpu_budget` | `int?` | `None` | optional lower bound for enumerated configurations |
| `context_length` | `int?` | `None` | optional KV-feasibility sequence length |
| `startup_time` | `float?` | `None` | optional simulated worker startup time |
| `aic_nextn` | `int?` | `None` | optional speculative-decoding depth |

Backends that have no performance database, no legal parallel configuration, or no support from
the selected runner are removed before sampling. The transitional Dynamo runner supports vLLM and
SGLang aggregated and disaggregated replay, and TensorRT-LLM aggregated replay.

## Engine Fields

Each searched engine list must be a non-empty subset of its allowed choices. A one-item list pins
the field.

| Field | Default | Allowed |
|---|---|---|
| `prefill_max_num_batched_tokens` | `[8192, 16384, 32768]` | `8192`, `16384`, `32768` |
| `prefill_max_num_seqs` | `[1, 2, 4, 8, 16, 32, 64, 128, 256]` | listed default values |
| `decode_max_num_batched_tokens` | `[8192]` | `8192` |
| `decode_max_num_seqs` | `[256, 512, 1024]` | `256`, `512`, `1024` |
| `agg_max_num_batched_tokens` | `[8192, 16384, 32768]` | `8192`, `16384`, `32768` |
| `agg_max_num_seqs` | `[256, 512, 1024]` | `256`, `512`, `1024` |

Pinned role fields:

| Role | Fields and defaults |
|---|---|
| prefill | `prefill_block_size: 64`, `prefill_gpu_memory_utilization: 0.9`, `prefill_enable_prefix_caching: true` |
| decode | `decode_block_size: 64`, `decode_gpu_memory_utilization: 0.9`, `decode_enable_prefix_caching: false` |
| aggregated | `agg_block_size: 64`, `agg_gpu_memory_utilization: 0.9`, `agg_enable_prefix_caching: true` |

Only fields for the active deployment branch enter the optimizer study.

## Pinned Parallel Configurations

An aggregated entry is a shape with an optional replica count:

```yaml
search_space:
  deployment_mode: [agg]
  parallel_configs:
    - tp: 4
      attention_dp: 2
      replicas: 2
```

A disaggregated entry contains prefill and decode shapes:

```yaml
search_space:
  deployment_mode: [disagg]
  parallel_configs:
    - prefill: {tp: 4, replicas: 1}
      decode: {tp: 8, attention_dp: 2, replicas: 2}
```

Pinning `parallel_configs` requires exactly one deployment mode. Every pinned shape must be legal,
KV-feasible, and supported by at least one configured backend.

## Adapter Contract

Each adapter entry has one field:

```yaml
adapters:
  example.adapter:
    search_space:
      policy: [a, b]
      weight: [0.25, 0.5]
```

Spica passes the complete `search_space` mapping to `generate_search_space`. The adapter returns a
`SearchSpaceFragment` for each deployment branch. Spica then namespaces the local parameters before
merging them into the study.

An adapter can be supplied through the `aisimulate.adapters` entry-point group or injected directly
through the Python API. Spica loads only configured adapters.

## Dynamo Router Adapter

`dynamo.router` accepts:

| Field | Type | Default |
|---|---|---|
| `mode` | `list[str]` | `[kv_router, round_robin]` |
| `overlap_score_credit` | `list[float]` | `[0.0, 0.5, 1.0]` |
| `prefill_load_scale` | `list[float]` | `[0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]` |
| `temperature` | `list[float]` | `[0.0, 0.2, 0.5, 1.0]` |

When `mode` contains only `round_robin`, the adapter removes the other Router dimensions. A
round-robin candidate produces no runtime hook. A KV-router candidate produces a
`dynamo.router:placement_policy@1` hook.

The current replay API does not support Router admission-control pins. The adapter rejects
`active_decode_blocks_threshold`, `active_prefill_tokens_threshold`,
`active_prefill_tokens_threshold_frac`, or `no_admission_control` when KV routing is enabled.

## Dynamo Planner Adapter

`dynamo.planner` accepts:

| Field | Type | Default |
|---|---|---|
| `scaling_policy` | `list[str\|dict]` | all built-in scaling policies |
| `fpm_sampling` | `list[str\|dict]` | all FPM presets |
| `load_sensitivity` | `list[str\|dict]` | all load-sensitivity presets |
| `load_predictor_candidates` | `list[str\|dict]` | all load-predictor presets |
| `min_endpoint` | `int?` | `None` |

The adapter filters policies against the optimization target and SLA before creating the study.
For every distinct throughput-scaling interval, it derives and runs the load-predictor pre-sweep
from the complete configured search space. A disabled Planner candidate produces no runtime hook;
an enabled candidate produces a `dynamo.planner:scaling_policy@1` hook containing a concrete
`PlannerConfig` payload.

## Removed KVBM Fields

Spica rejects old KVBM fields such as `num_g2_blocks`, transfer bandwidth, offload batch size, and
host/disk cache-hit weights. Native G2 replaces KVBM, so these fields have no adapter migration.

Old flat Planner and Router fields are also rejected. Move them under
`adapters.dynamo.planner.search_space` or `adapters.dynamo.router.search_space` and use the adapter
field names in this reference.

## Source of Truth

The backend schema is defined in
[`aisimulate/src/aisimulate/spica/config.py`](https://github.com/ai-dynamo/dynamo/blob/main/aisimulate/src/aisimulate/spica/config.py).
The Dynamo adapter schemas are defined in `components/src/dynamo/planner/simulation/adapter.py`
and `components/src/dynamo/router/simulation/adapter.py`.
