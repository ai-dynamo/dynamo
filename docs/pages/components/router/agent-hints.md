---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Agent Hints
subtitle: Per-request hints for scheduling, load balancing, and KV cache optimization
---

# Agent Hints

Agent hints are optional per-request hints passed via the `nvext.agent_hints` field in the request body. They allow the calling agent or application to communicate request-level metadata that the router uses to improve scheduling, load balancing, and KV cache utilization.

```json
{
  "nvext": {
    "agent_hints": {
      "latency_sensitivity": 5.0,
      "osl": 512,
      "speculative_prefill": true
    }
  }
}
```

All three fields are optional and independent — you can use any combination.

## `latency_sensitivity`

Priority scheduling hint, specified in seconds. When `--router-queue-threshold` is set and the queue is active, this value shifts the request's effective arrival time earlier in the queue, giving it priority over requests with lower (or no) `latency_sensitivity`. A value of `5.0` means the request is treated as if it arrived 5 seconds earlier than it actually did. Has no effect when queueing is disabled.

- **Type**: `f64` (optional)
- **Recommended default**: `1.2` for latency-sensitive agentic requests
- **Requires**: `--router-queue-threshold` to be set

### Example

```json
{
  "nvext": {
    "agent_hints": {
      "latency_sensitivity": 5.0
    }
  }
}
```

A request with `latency_sensitivity: 5.0` arriving at time `T` is treated as if it arrived at `T - 5s`, so it will be scheduled ahead of requests that arrived within the last 5 seconds (unless they have even higher sensitivity).

## `osl`

Expected output sequence length — the estimated number of output tokens the request will generate. The router uses this hint in two ways:

1. **Output block tracking**: When output block tracking is enabled (`--router-track-output-blocks`), the router adds placeholder blocks during generation and applies fractional decay based on progress toward `osl`. This gives the router a more accurate picture of each worker's KV cache utilization for long-running requests.
2. **Resource estimation**: Helps the router estimate total resource requirements when making routing decisions.

- **Type**: `u32` (optional)
- **Requires**: `--router-track-output-blocks` (frontend or standalone router) for output block tracking behavior

### Example

```json
{
  "nvext": {
    "agent_hints": {
      "osl": 1024
    }
  }
}
```

If the request is expected to generate ~1024 tokens, providing `osl: 1024` lets the router account for the output-side KV cache growth when balancing load across workers.

## `speculative_prefill`

When set to `true`, the system speculatively prefills the predicted next-turn prompt after the current assistant turn completes. This is designed for multi-turn agentic workloads where the next request's prefix is predictable.

- **Type**: `bool` (optional, defaults to `false`)
- **No additional CLI flags required**; works automatically when the hint is set in the request

### How it works

1. As the assistant response streams, the system accumulates the full response text.
2. Once the response finishes (indicated by `finish_reason`), a background task constructs the next-turn prompt by appending the assistant response to the conversation history (with thinking content stripped by the chat template for non-last assistant turns).
3. The constructed prompt is tokenized and sent through the pipeline as a `max_tokens=1` request to warm the KV cache on a worker.
4. When the actual next request arrives, it benefits from the already-warm KV cache, reducing TTFT.

### Example

```json
{
  "nvext": {
    "agent_hints": {
      "speculative_prefill": true
    }
  }
}
```

This is most effective for reasoning models in agentic loops, where the conversation grows incrementally and the next turn's prefix (everything up to the new user message) is the same as the current conversation.

## See Also

- **[Router Guide](router-guide.md)**: Full router configuration and CLI arguments
- **[Router Examples](router-examples.md)**: Usage patterns and benchmarking
