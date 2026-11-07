<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Experimental Native Retained-Prefix Requests

This request example requires a disaggregated TensorRT-LLM deployment with its native retained-prefix provider enabled. Its build must expose `tensorrt_llm.retained_prefix.parse_retained_prefix` and the native retained-prefix fields. Published builds without that API reject opt-in requests. Use the native engine's deployment and compatibility requirements; this example supplies no standalone serving configuration or supported topology claim.

For the initial request, include these fields alongside the usual model, prompt and sampling options:

```json
{
  "nvext": {
    "extra_fields": ["engine_data"],
    "trtllm_retained_prefix": {"retain_prefix": true}
  }
}
```

Extract the complete reference from the actual response, then include it on the next request. The Python fragment assumes `first_response` is the parsed response from the initial request and `next_request` already contains the correct model, sampling and actual trajectory token history:

```python
reference = first_response["nvext"]["engine_data"][
    "trtllm_retained_prefix"
]["retained_prefix"]
next_request["nvext"] = {
    "extra_fields": ["engine_data"],
    "trtllm_retained_prefix": {
        "retain_prefix": False,
        "retained_prefix": reference,
    },
}
```

Send `next_request` to the same deployment through its ordinary completions API. Use only engine-generated references, and append that trajectory's actual output IDs when constructing the next prompt. The engine decides whether its materialized prefix is compatible and still available. To retain another completed decoder prefix, explicitly set `retain_prefix` to `true` on that request.

A successful HTTP response or exported reference is not proof of reverse reuse. Validate the complete prefill→decode→next-prefill cycle using native import/coverage and source-retirement evidence. Cache-hit counters and the NIXL backend name do not establish retained GPU memory, auxiliary-state coverage, or a direct-memory transport route.

See [KV Cache Transfer](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/developer-guide/additional-resources/tensorrt-llm-details/kv-cache-transfer.md) for the request contract and compatibility boundary.
