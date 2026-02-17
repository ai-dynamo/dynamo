---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: NVIDIA Request Extensions (nvext)
---

# NVIDIA Request Extensions (`nvext`)

`nvext` is a top-level JSON object on the request body that provides NVIDIA-specific extensions to the OpenAI-compatible API. Unlike [`extra_args`](extra-args.md), which is an opaque pass-through to backend workers, `nvext` fields are consumed by the Dynamo frontend, preprocessor, and router to control routing, preprocessing, response metadata, and scheduling behavior.

## Usage

Include `nvext` as a top-level field alongside standard OpenAI-compatible fields:

```json
{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Hello"}],
    "nvext": {
        "greed_sampling": true,
        "extra_fields": ["worker_id", "timing"],
        "agent_hints": {
            "latency_sensitivity": 5.0,
            "osl": 1024
        }
    }
}
```

## Field Reference

| Field | Type | Default | Consumed By | Description |
|-------|------|---------|-------------|-------------|
| `greed_sampling` | `bool` | `None` | Preprocessor | Forces greedy sampling regardless of other sampling parameters. |
| `use_raw_prompt` | `bool` | `None` | Preprocessor | Bypasses the prompt template and passes the prompt directly to the tokenizer. |
| `annotations` | `string[]` | `None` | Preprocessor | Triggers out-of-band information in the SSE stream via the `event:` field. |
| `backend_instance_id` | `u64` | `None` | Router | Routes the request to a specific backend instance. |
| `token_data` | `u32[]` | `None` | Preprocessor | Pre-tokenized prompt tokens. When provided with `backend_instance_id`, tokenization is skipped. |
| `max_thinking_tokens` | `u32` | `None` | Backend | Maximum thinking tokens allowed (passed through to backends). |
| `extra_fields` | `string[]` | `None` | Response builder | Fields to include in the response `nvext`. Supported: `"worker_id"`, `"timing"`. |
| `prefill_worker_id` | `u64` | `None` | Router | Routes the request to a specific prefill worker (disaggregated serving). |
| `decode_worker_id` | `u64` | `None` | Router | Routes the request to a specific decode worker (disaggregated serving). |
| `agent_hints` | object | `None` | Router | Per-request hints for scheduling and load balancing. See [Agent Hints](#agent-hints). |

### Header Overrides

Routing fields can also be set via HTTP headers, which take priority over `nvext` values:

| Header | Overrides |
|--------|-----------|
| `x-worker-instance-id` | `backend_instance_id` and `decode_worker_id` |
| `x-prefill-instance-id` | `prefill_worker_id` |

## Agent Hints

The `agent_hints` sub-object carries per-request hints that the router uses for scheduling, load balancing, and KV cache optimization.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `latency_sensitivity` | `f64` | `None` | Priority scheduling hint in seconds. Shifts the request's effective arrival time earlier in the router queue. Requires `--router-queue-threshold`. |
| `osl` | `u32` | `None` | Expected output sequence length (tokens). Used for output block tracking and resource estimation. |
| `speculative_prefill` | `bool` | `false` | When `true`, speculatively prefills the predicted next-turn prompt after the current turn completes to warm the KV cache. |

### `latency_sensitivity`

When `--router-queue-threshold` is set and the queue is active, this value shifts the request's effective arrival time earlier in the queue, giving it priority over requests with lower (or no) `latency_sensitivity`. A value of `5.0` means the request is treated as if it arrived 5 seconds earlier than it actually did. Has no effect when queueing is disabled.

```json
{
    "nvext": {
        "agent_hints": {
            "latency_sensitivity": 5.0
        }
    }
}
```

### `osl`

Expected output sequence length — the estimated number of output tokens the request will generate. The router uses this hint in two ways:

1. **Output block tracking**: When `--track-output-blocks` is enabled, the router adds placeholder blocks during generation and applies fractional decay based on progress toward `osl`.
2. **Resource estimation**: Helps the router estimate total resource requirements when making routing decisions.

```json
{
    "nvext": {
        "agent_hints": {
            "osl": 1024
        }
    }
}
```

### `speculative_prefill`

When set to `true`, the system speculatively prefills the predicted next-turn prompt after the current assistant turn completes. This is designed for multi-turn agentic workloads where the next request's prefix is predictable.

How it works:

1. As the assistant response streams, the system accumulates the full response text.
2. Once the response finishes, a background task constructs the next-turn prompt by appending the assistant response to the conversation history (with thinking content stripped for non-last turns).
3. The constructed prompt is tokenized and sent as a `max_tokens=1` request to warm the KV cache on a worker.
4. When the actual next request arrives, it benefits from the already-warm KV cache, reducing TTFT.

```json
{
    "nvext": {
        "agent_hints": {
            "speculative_prefill": true
        }
    }
}
```

## Response Extensions

When the client requests response metadata via `extra_fields`, the response includes an `nvext` object with the requested fields:

| Field | Requested Via | Description |
|-------|---------------|-------------|
| `worker_id` | `extra_fields: ["worker_id"]` | Prefill/decode worker IDs and data parallel ranks that processed the request. |
| `timing` | `extra_fields: ["timing"]` | Per-request timing information (TTFT, ITL, queue time, etc.). |
| `token_ids` | Automatic (GAIE Stage 1) | Tokenized prompt for reuse in Stage 2 query-only mode. |

### Example response `nvext`

```json
{
    "nvext": {
        "worker_id": {
            "prefill_worker_id": 1,
            "prefill_dp_rank": 0,
            "decode_worker_id": 2,
            "decode_dp_rank": 0
        },
        "timing": {
            "ttft_ms": 45.2,
            "itl_ms": 12.1
        }
    }
}
```

## See Also

| Document | Description |
|----------|-------------|
| [Custom Request Arguments (`extra_args`)](extra-args.md) | Opaque pass-through arguments for backend workers |
| [Frontend Guide](frontend-guide.md) | KServe gRPC configuration and integration |
| [Router Guide](../router/router-guide.md) | Full router configuration and CLI arguments |
