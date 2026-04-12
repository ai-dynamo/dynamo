<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Global Router

A hierarchical routing service that sits between the Dynamo frontend and local routers in different pool namespaces. The global router supports both disaggregated and aggregated serving with flexible pool selection based on request characteristics.

## Overview

The Global Router supports two modes:

- **Disagg mode** (default): Registers as both prefill and decode worker. Routes prefill requests based on (ISL, TTFT) and decode requests based on (context_length, ITL) to separate pool types.
- **Agg mode**: Registers as a single generate worker. Routes all requests based on (TTFT target, ITL target) to unified pools that handle both prefill and decode.

Both modes support priority-based pool overrides from agent hints.

## Supported Backends

- **vLLM** - Uses synchronous prefill path (frontend waits for prefill to complete)
- **Mocker** - Uses same synchronous path as vLLM

**Not supported:**
- **SGLang** - Bootstrap path (async KV transfer) not implemented
- **TensorRT-LLM** - Bootstrap path not implemented

## Architecture

### Disagg Mode

```
Frontend
    |
    v
Global Router (registers as both prefill + decode)
    |
    +---> Prefill Pool 0 (namespace: prefill_pool_0)
    |         |
    |         +---> Local Router ---> Prefill Worker 0
    |                           +---> Prefill Worker 1
    |
    +---> Prefill Pool ...
    |
    +---> Decode Pool 0 (namespace: decode_pool_0)
    |         |
    |         +---> Local Router ---> Decode Worker 0
    |                           +---> Decode Worker 1
    |
    +---> Decode Pool ...
```

### Agg Mode

```
Frontend
    |
    v
Global Router (registers as Chat + Completions)
    |
    +---> Agg Pool 0 (namespace: agg_pool_0)
    |         |
    |         +---> Local Router ---> Worker 0 (prefill + decode)
    |                           +---> Worker 1 (prefill + decode)
    |
    +---> Agg Pool 1 (namespace: agg_pool_1)
    |         |
    |         +---> Local Router ---> Worker 0 (prefill + decode)
    |                           +---> Worker 1 (prefill + decode)
    |
    +---> Agg Pool ...
```

## Usage

```bash
python -m dynamo.global_router \
  --config path/to/global_router_config.json \
  --model-name Qwen/Qwen3-0.6B \
  --namespace dynamo
```

### Arguments

All options can be set via CLI flags or environment variables. CLI flags take precedence over environment variables.

| Argument | Required (CLI or env) | Env var | Default | Description |
|----------|----------------------|---------|---------|-------------|
| `--config` | Yes | `DYN_GLOBAL_ROUTER_CONFIG` | - | Path to JSON configuration file |
| `--model-name` | Yes | `DYN_GLOBAL_ROUTER_MODEL_NAME` | - | Model name for registration (must match workers) |
| `--namespace` | No | `DYN_NAMESPACE` | "dynamo" | Namespace for global router |
| `--component-name` | No | `DYN_GLOBAL_ROUTER_COMPONENT_NAME` | "global_router" | Component name |
| `--default-ttft-target` | No | `DYN_GLOBAL_ROUTER_DEFAULT_TTFT_TARGET` | None | Default TTFT target (ms) for prefill pool selection |
| `--default-itl-target` | No | `DYN_GLOBAL_ROUTER_DEFAULT_ITL_TARGET` | None | Default ITL target (ms) for pool selection |

## Configuration

The configuration file format depends on the mode. The `mode` field determines which mode is used; if omitted, it defaults to `"disagg"`.

### Disagg Mode Configuration

```jsonc
{
    "mode": "disagg",                     // Optional, defaults to "disagg"
    "num_prefill_pools": <int>,
    "num_decode_pools": <int>,
    "prefill_pool_dynamo_namespaces": [],
    "decode_pool_dynamo_namespaces": [],

    "prefill_pool_selection_strategy": {
        "isl_min": <int>,
        "isl_max": <int>,
        "isl_resolution": <int>,
        "ttft_min": <float>,
        "ttft_max": <float>,
        "ttft_resolution": <int>,
        "prefill_pool_mapping": [[]],     // 2D array [isl_resolution][ttft_resolution] -> pool index
        "priority_overrides": []          // Optional
    },

    "decode_pool_selection_strategy": {
        "context_length_min": <int>,
        "context_length_max": <int>,
        "context_length_resolution": <int>,
        "itl_min": <float>,
        "itl_max": <float>,
        "itl_resolution": <int>,
        "decode_pool_mapping": [[]],      // 2D array [context_length_resolution][itl_resolution] -> pool index
        "priority_overrides": []          // Optional
    }
}
```

### Agg Mode Configuration

```jsonc
{
    "mode": "agg",
    "num_agg_pools": <int>,
    "agg_pool_dynamo_namespaces": [],

    "agg_pool_selection_strategy": {
        "ttft_min": <float>,              // Minimum TTFT target (ms)
        "ttft_max": <float>,              // Maximum TTFT target (ms)
        "ttft_resolution": <int>,         // Number of grid rows for TTFT dimension
        "itl_min": <float>,              // Minimum ITL target (ms)
        "itl_max": <float>,              // Maximum ITL target (ms)
        "itl_resolution": <int>,          // Number of grid columns for ITL dimension
        "agg_pool_mapping": [[]],         // 2D array [ttft_resolution][itl_resolution] -> pool index
        "priority_overrides": []          // Optional
    }
}
```

### Why TTFT x ITL for Agg Mode

In aggregated mode, the same pool handles both prefill and decode. Both SLA targets matter for a single routing decision:

- **TTFT target** captures the user's prefill latency requirement. ISL is implicitly accounted for — a user sending a large prompt with a tight TTFT target is saying "I need a fast pool."
- **ITL target** captures the user's decode latency requirement. With chunked prefill, ITL reflects the combined prefill+decode contention. Without chunked prefill, ITL reflects pure decode performance.

This creates natural pool separation:
- Tight TTFT + tight ITL -> premium interactive pool
- Relaxed TTFT + tight ITL -> decode-optimized pool
- Tight TTFT + relaxed ITL -> prefill-optimized pool
- Relaxed TTFT + relaxed ITL -> batch/throughput pool

### Pool Selection

The pool selection uses a 2D grid lookup. Each dimension is divided into buckets based on the resolution.

**Prefill Pool Selection** (disagg mode, based on ISL and TTFT target):

1. Compute `isl_step = (isl_max - isl_min) / isl_resolution`
2. Compute `ttft_step = (ttft_max - ttft_min) / ttft_resolution`
3. For a request with input sequence length `ISL` and target TTFT:
   - `isl_idx = clamp((ISL - isl_min) / isl_step, 0, isl_resolution - 1)`
   - `ttft_idx = clamp((ttft_target - ttft_min) / ttft_step, 0, ttft_resolution - 1)`
4. Lookup pool: `pool_index = prefill_pool_mapping[isl_idx][ttft_idx]`

**Decode Pool Selection** (disagg mode, based on context length and ITL target):

Same logic but using `context_length` and `itl_target` with `decode_pool_mapping`.

**Agg Pool Selection** (agg mode, based on TTFT and ITL targets):

Same grid logic using `ttft_target` and `itl_target` with `agg_pool_mapping`.

### Priority-Based Pool Override

All strategies support optional `priority_overrides` rules. When a request carries a priority value (from `nvext.agent_hints.priority`), the global router evaluates the override rules after the grid lookup. The first rule whose `[min_priority, max_priority]` range contains the request priority wins, and the request is routed to that rule's `target_pool` instead of the grid result. If no rule matches (or no priority is present), the grid result is used as normal.

This is useful for straggler mitigation in RL workloads: the RL framework can tag slow requests with a high priority, and the global router redirects them to a dedicated min-latency pool.

```jsonc
"priority_overrides": [
    {
        "min_priority": 10,     // inclusive lower bound
        "max_priority": 100,    // inclusive upper bound
        "target_pool": 1        // pool index to route to
    }
]
```

Priority is set by the client via the NVIDIA OpenAI extension:

```json
{
    "messages": [...],
    "nvext": {
        "agent_hints": {
            "priority": 50
        }
    }
}
```

### Priority-Based Pool Override

Both prefill and decode strategies support optional `priority_overrides` rules.
When a request carries a priority value (from `nvext.agent_hints.priority`), the
global router evaluates the override rules **after** the grid lookup. The first
rule whose `[min_priority, max_priority]` range contains the request priority
wins, and the request is routed to that rule's `target_pool` instead of the
grid result. If no rule matches (or no priority is present), the grid result
is used as normal.

This is useful for straggler mitigation in RL workloads: the RL framework can
tag slow requests with a high priority, and the global router redirects them to
a dedicated min-latency pool.

```jsonc
"priority_overrides": [
    {
        "min_priority": 10,     // inclusive lower bound
        "max_priority": 100,    // inclusive upper bound
        "target_pool": 1        // pool index to route to
    }
]
```

Priority is set by the client via the NVIDIA OpenAI extension:

```json
{
    "messages": [...],
    "nvext": {
        "agent_hints": {
            "priority": 50
        }
    }
}
```

### Passing SLA Targets

Clients can pass TTFT and ITL targets via `extra_args` in the request:

```json
{
    "messages": [...],
    "extra_args": {
        "ttft_target": 100,
        "itl_target": 20
    }
}
```

If not provided, the middle of the configured range is used as default. For disagg mode, `ttft_target` drives prefill pool selection and `itl_target` drives decode pool selection. For agg mode, both `ttft_target` and `itl_target` drive pool selection.

## Request Flow

### Disagg Mode

1. Frontend receives request and sends to Global Router (registered as prefill)
2. Global Router selects prefill pool based on (ISL, TTFT_target, priority)
3. Request is forwarded to local router in the selected prefill pool namespace
4. Local router forwards to a prefill worker
5. Prefill response returns with `disaggregated_params`
6. Frontend sends decode request to Global Router (registered as decode)
7. Global Router selects decode pool based on (context_length, ITL_target, priority)
8. Request is forwarded to local router in the selected decode pool namespace
9. Tokens stream back through the chain

### Agg Mode

1. Frontend receives request and sends to Global Router (registered as Chat + Completions)
2. Global Router selects agg pool based on (TTFT_target, ITL_target, priority)
3. Request is forwarded to local router in the selected agg pool namespace
4. Local router forwards to a worker that handles both prefill and decode
5. Tokens stream back through the chain

## Example

See `examples/global_planner/` for a complete example with:
- Global router configuration
- Local router setup for each pool
- Mocker workers for testing
