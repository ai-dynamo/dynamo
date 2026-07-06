---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: SGLang for Agentic Workloads
subtitle: Priority scheduling and KV cache tuning for agentic serving
---

This guide covers SGLang-specific configuration for agentic serving with Dynamo. It explains which SGLang engine flags to enable, how Dynamo's [agent hints](../../components/frontend/nvext.md#agent-hints) map to SGLang behavior, and how session headers interact with SGLang 0.5.14.

## Overview

Agentic workloads (tool-calling loops, multi-turn reasoning, code generation pipelines) have different performance characteristics than batch inference:

- **Prefix-heavy**: Successive turns share a growing conversation prefix. KV cache reuse is critical for low TTFT.
- **Priority-sensitive**: Some requests (user-facing agent turns) matter more than background tasks.
- **Long-lived**: Conversations span minutes to hours. Cache eviction under memory pressure can destroy accumulated KV state.

Dynamo's agent hints give the router per-request metadata. SGLang's engine flags control how that metadata affects scheduling and eviction on the worker. For the cross-layer Dynamo priority semantics, see [Priority Scheduling](../../components/router/priority-scheduling.md).

## SGLang Engine Flags

### Priority Scheduling

Enable priority-based scheduling so the engine respects the `priority` value from `nvext.agent_hints.priority`:

```bash
python -m dynamo.sglang \
  --model-path <model> \
  --enable-priority-scheduling \
  ...
```

| Flag                           | Description                                                |
| ------------------------------ | ---------------------------------------------------------- |
| `--enable-priority-scheduling` | Enables priority-based request scheduling instead of FCFS. |

When priority scheduling is enabled, the engine uses the `priority` field from `nvext.agent_hints` to order requests in its internal queue. Requests with higher effective priority are scheduled before lower-priority ones. Ties are broken by arrival time.

Router queue priority is configured separately on the frontend with
`--router-queue-threshold`; see
[Router Configuration and Tuning](../../components/router/router-configuration.md#routing-behavior).

### Priority-Based KV Cache Eviction

By default, SGLang evicts radix tree nodes using LRU. You can switch to priority-based eviction so that low-priority cache entries are evicted before high-priority ones:

```bash
python -m dynamo.sglang \
  --model-path <model> \
  --radix-eviction-policy priority \
  ...
```

| Flag                      | Values            | Default | Description                                                                                                |
| ------------------------- | ----------------- | ------- | ---------------------------------------------------------------------------------------------------------- |
| `--radix-eviction-policy` | `lru`, `priority` | `lru`   | Eviction strategy for the GPU radix cache. `priority` uses a heap ordered by the request's priority value. |

This does **not** require HiCache. It controls GPU-only radix tree eviction. When the GPU KV cache is full:

- **`lru`**: Evicts the least recently used leaf nodes first.
- **`priority`**: Evicts lowest-priority leaf nodes first. Nodes with equal priority fall back to LRU ordering.

#### Interaction with HiCache

When both `--radix-eviction-policy priority` and `--enable-hierarchical-cache` are enabled, priority affects eviction at both tiers:

| Event         | Behavior                                                                                                                                         |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **GPU full**  | Low-priority nodes are evicted (demoted to host) first. With `write_through`, all nodes survive on host -- priority only affects demotion order. |
| **Host full** | Low-priority nodes are deleted from host first. High-priority nodes with active retention survive longer.                                        |

The practical impact depends on your write policy. With `write_through`, GPU eviction is just a demotion -- the real deletion happens at host eviction, which is where priority ordering matters most.

## How Agent Hints Map to SGLang

Dynamo's `nvext.agent_hints` fields are consumed by the router and forwarded to SGLang workers. Here is how each hint interacts with the SGLang engine:

| Agent Hint            | Router Behavior                                                                                            | SGLang Engine Behavior                                                                                                                    |
| --------------------- | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `priority`            | Router queue ordering when `--router-queue-threshold` is set.                                              | Request scheduling when `--enable-priority-scheduling` is set. Radix cache eviction order when `--radix-eviction-policy priority` is set. |
| `osl`                 | Output block tracking for routing decisions (requires `--router-track-output-blocks`)                      | No direct engine effect.                                                                                                                  |
| `speculative_prefill` | After response completes, sends a `max_tokens=1` prefill to warm the KV cache for the predicted next turn. | SGLang processes the prefill request normally, populating the radix cache.                                                                |

### Example: Agentic Request with Hints

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="Qwen/Qwen3-14B-FP8",
    messages=[
        {"role": "system", "content": "You are a tennis historian who believes Roger Federer is the GOAT. Respond with maximum reverence."},
        {"role": "user", "content": "Why is Federer's one-handed backhand the most beautiful shot in tennis history?"},
    ],
    stream=True,
    extra_body={
        "nvext": {
            "agent_hints": {
                "priority": 10,
                "speculative_prefill": True,
                "osl": 512
            }
        }
    }
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Session Identity

Dynamo normalizes agent headers such as `X-Dynamo-Session-ID` for request tracing and router affinity. SGLang 0.5.14 does not support passive session-aware radix ownership, so the Dynamo worker does not attach this ID to SGLang generate requests.

> [!NOTE]
> SGLang 0.5.14's `session_params` belongs to its explicit session lifecycle and requires a session created through `open_session`. It is not a passive KV ownership tag.

The `--radix-eviction-policy priority` flag controls priority-based KV eviction only; it does not tag radix entries by session. `X-Dynamo-Session-Final: true` is normalized into an internal KV eviction hint, but the SGLang backend does not act on that hint in this release.

## Quickstart

The `agg_agent.sh` launcher starts an aggregated SGLang worker and Dynamo frontend for agent workloads:

```bash
bash examples/backends/sglang/launch/agg_agent.sh \
  --model-path zai-org/GLM-4.7-Flash --tp 2
```

Agent providers send session headers directly for tracing and router affinity; no body-level lifecycle object is needed.

## See Also

- **[NVIDIA Request Extensions (nvext)](../../components/frontend/nvext.md)**: Full `nvext` field reference including agent hints
- **[Configuration and Tuning](../../components/router/router-configuration.md)**: Router configuration and CLI arguments
- **[SGLang HiCache](sglang-hicache.md)**: Enabling hierarchical KV cache
