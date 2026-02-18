---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Custom Request Arguments (extra_args)
---

# Custom Request Arguments (`extra_args`)

`extra_args` is an optional JSON object on the request body that is passed through the frontend directly to backend workers. Unlike `nvext`, which contains NVIDIA extensions to the OpenAI request format consumed by the Dynamo frontend and router (e.g., routing hints, annotations, greedy sampling overrides), `extra_args` is opaque to the frontend — it is forwarded as-is to downstream components where backend-specific logic can extract the keys it needs.

## Usage

Include `extra_args` as a top-level field alongside standard OpenAI-compatible fields:

```json
{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Hello"}],
    "extra_args": {
        "priority": 10,
        "ttft_target": 100,
        "itl_target": 20
    }
}
```

The object is forwarded as-is through the request pipeline. Each component extracts only the keys it recognizes and ignores the rest, so unrecognized keys are safe to include for experimentation or custom backend extensions.

## Supported Keys

| Key | Type | Default | Consumed By | Description |
|-----|------|---------|-------------|-------------|
| `priority` | `int` | `0` | vLLM and SGLang backends | Request priority for queue ordering, KV cache eviction, and queue admission. |
| `ttft_target` | `float` | config default | Global Router | Target time-to-first-token in milliseconds. Used to select the prefill worker pool. |
| `itl_target` | `float` | config default | Global Router | Target inter-token latency in milliseconds. Used to select the decode worker pool. |

### `priority`

Extracted by the vLLM and SGLang worker handlers and passed to the engine's `generate` call. Priority influences queue ordering, KV cache eviction under memory pressure, and queue admission when the waiting queue is full.

The semantics of the priority value differ between backends:

- **vLLM**: Smaller values = higher priority. A request with `priority: 0` is scheduled before `priority: 10`. Ties are broken by arrival time. Requires `--scheduling-policy priority` on the engine.
- **SGLang**: By default, larger values = higher priority. This can be inverted with `--schedule-low-priority-values-first` to match vLLM's convention. Requires `--enable-priority-scheduling` on the engine.

When omitted, vLLM defaults to `0`; SGLang defaults to `None` (engine default).

### `ttft_target`

Used by the Global Router to select which prefill worker pool handles the request. The router maps the target against configured TTFT resolution buckets. If not provided, the midpoint of the configured range is used. See [Global Router documentation](../../../../components/src/dynamo/global_router/README.md) for pool mapping details.

### `itl_target`

Used by the Global Router to select which decode worker pool handles the request, analogous to `ttft_target` for the decode phase. If not provided, the midpoint of the configured range is used.

## See Also

| Document | Description |
|----------|-------------|
| [NVIDIA Request Extensions (`nvext`)](nvext.md) | Routing, preprocessing, and response metadata extensions consumed by the frontend |
| [Frontend Guide](frontend-guide.md) | KServe gRPC configuration and integration |
| [Global Router](../../../../components/src/dynamo/global_router/README.md) | SLA-aware pool routing with `ttft_target` / `itl_target` |
