<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Global Router

A hierarchical routing service that sits between the Dynamo frontend and local routers in different pool namespaces. The global router enables disaggregated serving with flexible pool selection based on request characteristics.

## Overview

The Global Router acts as both a prefill and decode worker from the frontend's perspective:
- Registers with `ModelType.Prefill` for prefill requests
- Registers with `ModelType.Chat | ModelType.Completions` for decode requests

Internally, it routes requests to local routers in different namespaces based on a configurable grid-based selection strategy.

## Supported Backends

- **vLLM** - Uses synchronous prefill path (frontend waits for prefill to complete)
- **Mocker** - Uses same synchronous path as vLLM

**Not supported:**
- **SGLang** - Bootstrap path (async KV transfer) not implemented
- **TensorRT-LLM** - Bootstrap path not implemented

## Architecture

```
Frontend
    |
    v
Global Router (registers as both prefill + decode)
    |
    +---> Prefill Pool 0 (namespace: prefill_pool_0)
    |         |
    |         +---> Local Router ---> Prefill Worker 0
    |                           |
    |                           +---> Prefill Worker 1
    |                           |
    |                           +---> ...
    +---> Prefill Pool ...
    |
    +---> Decode Pool 0 (namespace: decode_pool_0)
    |         |
    |         +---> Local Router ---> Decode Worker 0
    |                           |
    |                           +---> Decode Worker 1
    |                           |
    |                           +---> ...
    +---> Decode Pool ...
```

## Usage

```bash
python -m dynamo.global_router \
  --config path/to/global_router_config.json \
  --model-name Qwen/Qwen3-0.6B \
  --namespace dynamo
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--config` | Yes | - | Path to JSON configuration file |
| `--model-name` | Yes | - | Model name for registration (must match workers) |
| `--namespace` | No | `DYN_NAMESPACE` env var or "dynamo" | Namespace for global router |
| `--component-name` | No | "global_router" | Component name |
| `--default-ttft-target` | No | None | Default TTFT target (ms) for prefill pool selection |
| `--default-itl-target` | No | None | Default ITL target (ms) for decode pool selection |

## Configuration

The configuration file defines pool namespaces and selection strategies:

```jsonc
{
    "num_prefill_pools": <int>,           // Number of prefill pools
    "num_decode_pools": <int>,            // Number of decode pools
    "prefill_pool_dynamo_namespaces": [], // List of Dynamo namespaces for each prefill pool
    "decode_pool_dynamo_namespaces": [],  // List of Dynamo namespaces for each decode pool

    "prefill_pool_selection_strategy": {
        "isl_min": <int>,                 // Minimum input sequence length (tokens)
        "isl_max": <int>,                 // Maximum input sequence length (tokens)
        "isl_resolution": <int>,          // Number of grid rows for ISL dimension
        "ttft_min": <float>,              // Minimum TTFT target (ms)
        "ttft_max": <float>,              // Maximum TTFT target (ms)
        "ttft_resolution": <int>,         // Number of grid columns for TTFT dimension
        "prefill_pool_mapping": [[]]      // 2D array [isl_resolution][ttft_resolution] -> pool index
    },

    "decode_pool_selection_strategy": {
        "context_length_min": <int>,      // Minimum context length (tokens)
        "context_length_max": <int>,      // Maximum context length (tokens)
        "context_length_resolution": <int>, // Number of grid rows for context length
        "itl_min": <float>,               // Minimum ITL target (ms)
        "itl_max": <float>,               // Maximum ITL target (ms)
        "itl_resolution": <int>,          // Number of grid columns for ITL dimension
        "decode_pool_mapping": [[]]       // 2D array [context_length_resolution][itl_resolution] -> pool index
    }
}
```

### Pool Selection

The pool selection uses a 2D grid lookup. Each dimension is divided into buckets based on the resolution.

**Prefill Pool Selection** (based on ISL and TTFT target):

1. Compute `isl_step = (isl_max - isl_min) / isl_resolution`
2. Compute `ttft_step = (ttft_max - ttft_min) / ttft_resolution`
3. For a request with input sequence length `ISL` and target TTFT:
   - `isl_idx = clamp((ISL - isl_min) / isl_step, 0, isl_resolution - 1)`
   - `ttft_idx = clamp((ttft_target - ttft_min) / ttft_step, 0, ttft_resolution - 1)`
4. Lookup pool: `pool_index = prefill_pool_mapping[isl_idx][ttft_idx]`

**Decode Pool Selection** (based on context length and ITL target):

Same logic but using `context_length` and `itl_target` with `decode_pool_mapping`.

**Example**: With `isl_min=0`, `isl_max=32000`, `isl_resolution=2`:
- ISL in [0, 16000) → `isl_idx = 0`
- ISL in [16000, 32000] → `isl_idx = 1`

If `prefill_pool_mapping = [[0, 1], [0, 1]]` and `ttft_resolution=2`:
- Low ISL + Low TTFT target → pool 0
- Low ISL + High TTFT target → pool 1
- High ISL + Low TTFT target → pool 0
- High ISL + High TTFT target → pool 1

### Passing SLA Targets

Clients can pass TTFT and ITL targets via `extra_args` in the request:

```json
{
    "messages": [...],
    "extra_args": {
        "ttft_target": 100,  // Target TTFT in ms for prefill pool selection
        "itl_target": 20     // Target ITL in ms for decode pool selection
    }
}
```

If not provided, the middle of the configured range is used as default.

## Request Flow

1. Frontend receives request and sends to Global Router (registered as prefill)
2. Global Router selects prefill pool based on (ISL, TTFT_target)
3. Request is forwarded to local router in the selected prefill pool namespace
4. Local router forwards to a prefill worker
5. Prefill response returns with `disaggregated_params`
6. Frontend sends decode request to Global Router (registered as decode)
7. Global Router selects decode pool based on (context_length, ITL_target)
8. Request is forwarded to local router in the selected decode pool namespace
9. Tokens stream back through the chain

## Example

See `examples/hierarchical_planner/` for a complete example with:
- Global router configuration
- Local router setup for each pool
- Mocker workers for testing
