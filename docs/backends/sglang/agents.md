---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: SGLang for Agentic Workloads
subtitle: Priority scheduling, KV cache retention, and session control for multi-turn agentic serving
---

# SGLang for Agentic Workloads

This guide covers SGLang-specific configuration for agentic serving with Dynamo. It explains which SGLang engine flags to enable, how Dynamo's [agent hints](../../components/frontend/nvext.md#agent-hints) map to SGLang behavior, and how to use cache retention and session control to manage KV cache for multi-turn agent conversations.

## Overview

Agentic workloads (tool-calling loops, multi-turn reasoning, code generation pipelines) have different performance characteristics than batch inference:

- **Prefix-heavy**: Successive turns share a growing conversation prefix. KV cache reuse is critical for low TTFT.
- **Priority-sensitive**: Some requests (user-facing agent turns) matter more than background tasks.
- **Long-lived**: Conversations span minutes to hours. Cache eviction under memory pressure can destroy accumulated KV state.

Dynamo's agent hints give the router per-request metadata. SGLang's engine flags control how that metadata affects scheduling and eviction on the worker.

## SGLang Engine Flags

### Priority Scheduling

Enable priority-based scheduling so the engine respects the `priority` value from `nvext.agent_hints.priority`:

```bash
python -m dynamo.sglang \
  --model-path <model> \
  --enable-priority-scheduling \
  ...
```

| Flag | Description |
|------|-------------|
| `--enable-priority-scheduling` | Enables priority-based request scheduling instead of FCFS. |

When priority scheduling is enabled, the engine uses the `priority` field from `nvext.agent_hints` to order requests in its internal queue. Requests with higher effective priority are scheduled before lower-priority ones. Ties are broken by arrival time.

### Priority-Based KV Cache Eviction

By default, SGLang evicts radix tree nodes using LRU. You can switch to priority-based eviction so that low-priority cache entries are evicted before high-priority ones:

```bash
python -m dynamo.sglang \
  --model-path <model> \
  --radix-eviction-policy priority \
  ...
```

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--radix-eviction-policy` | `lru`, `priority` | `lru` | Eviction strategy for the GPU radix cache. `priority` uses a heap ordered by the request's priority value. |

This does **not** require HiCache. It controls GPU-only radix tree eviction. When the GPU KV cache is full:

- **`lru`**: Evicts the least recently used leaf nodes first.
- **`priority`**: Evicts lowest-priority leaf nodes first. Nodes with equal priority fall back to LRU ordering.

#### Interaction with HiCache

When both `--radix-eviction-policy priority` and `--enable-hierarchical-cache` are enabled, priority affects eviction at both tiers:

| Event | Behavior |
|-------|----------|
| **GPU full** | Low-priority nodes are evicted (demoted to host) first. With `write_through`, all nodes survive on host -- priority only affects demotion order. |
| **Host full** | Low-priority nodes are deleted from host first. High-priority nodes with active retention survive longer. |

The practical impact depends on your write policy. With `write_through`, GPU eviction is just a demotion -- the real deletion happens at host eviction, which is where priority ordering matters most.

## How Agent Hints Map to SGLang

Dynamo's `nvext.agent_hints` fields are consumed by the router and forwarded to SGLang workers. Here is how each hint interacts with the SGLang engine:

| Agent Hint | Router Behavior | SGLang Engine Behavior |
|------------|----------------|----------------------|
| `priority` | Router queue ordering when `--router-queue-threshold` is set. | Request scheduling when `--enable-priority-scheduling` is set. Radix cache eviction order when `--radix-eviction-policy priority` is set. Cache retention decay when `retention_seconds` is set. |
| `osl` | Output block tracking for routing decisions (requires `--router-track-output-blocks`) | No direct engine effect. |
| `speculative_prefill` | After response completes, sends a `max_tokens=1` prefill to warm the KV cache for the predicted next turn. | SGLang processes the prefill request normally, populating the radix cache. |

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

## Cache Retention via `cache_control`

When a request includes `nvext.cache_control`, the router injects `retention_seconds` into the generate request. Combined with `nvext.agent_hints.priority`, this gives the SGLang worker priority-based KV eviction with time decay -- high-value conversation prefixes survive longer under memory pressure without a separate RPC.

### How It Works

```mermaid
sequenceDiagram
    participant Client
    participant Preprocessor
    participant Router as KV Router
    participant Worker as SGLang Worker
    participant Cache as Radix Cache

    Client->>Preprocessor: nvext.cache_control{ttl: "5m"}<br/>nvext.agent_hints{priority: 50}
    Preprocessor->>Preprocessor: Extract cache_control_ttl=300
    Preprocessor->>Router: PreprocessedRequest

    Router->>Router: Select best worker
    Router->>Router: Inject extra_args.retention_seconds=300

    Router->>Worker: async_generate(<br/>  retention_seconds=300,<br/>  priority=50)
    Worker->>Cache: Insert with priority=50,<br/>retention_duration=300s
    Worker-->>Client: Stream response

    Note over Cache: Under memory pressure
    Cache->>Cache: Evict priority=0 nodes first<br/>Priority=50 nodes survive

    Note over Cache: After 5 min idle
    Cache->>Cache: Effective priority decays to 0<br/>Node now eligible for normal eviction
```

No separate RPC is needed. The retention and priority values flow inline with the generate request.

### Enabling

```bash
python -m dynamo.frontend \
  --router-mode kv \
  --enable-cache-control \
  ...
```

| Flag | Description |
|------|-------------|
| `--enable-cache-control` | Enables agent-aware cache control: session lifecycle RPCs, sticky session routing, and retention_seconds injection. Requires `--router-mode=kv`. |

**SGLang worker** must use priority-based eviction:

```bash
python -m dynamo.sglang \
  --model-path <model> \
  --radix-eviction-policy priority \
  ...
```

### Request Format

```json
{
    "model": "Qwen/Qwen3-14B-FP8",
    "messages": [
        {"role": "system", "content": "You are a Roger Federer superfan. Every answer must highlight his elegance, grace, and superiority."},
        {"role": "user", "content": "Explain why Federer's 2017 comeback is the greatest story in sports history."}
    ],
    "nvext": {
        "cache_control": {
            "type": "ephemeral",
            "ttl": "1h"
        },
        "agent_hints": {
            "priority": 50
        }
    }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `cache_control.type` | `string` | Currently only `"ephemeral"` is supported. |
| `cache_control.ttl` | `string` | Retention duration: integer seconds (`"600"`) or shorthand (`"5m"`, `"1h"`). Clamped to [300, 3600] seconds. |
| `agent_hints.priority` | `integer` | Eviction priority (higher = survives longer). Default 0 = LRU eviction. |

## Session Control for Subagent KV Isolation (Experimental)

> [!WARNING]
> Session control is experimental. The API may change.

Agentic orchestrators often spawn short-lived subagents (research, code execution, planning) that accumulate KV cache, use it for a few turns, then die. Under normal radix cache behavior, this ephemeral KV pollutes the tree and competes with the lead agent's long-lived prefix for eviction.

Session control solves this by holding subagent KV in dedicated **streaming session slots** outside the radix tree. Session KV is invisible to eviction, has no L2 backup overhead, and is freed deterministically on close or timeout.

### How It Works

```mermaid
sequenceDiagram
    participant Orchestrator
    participant Router as Dynamo Router
    participant Worker as SGLang Worker
    participant Cache as SessionAwareCache

    Note over Orchestrator: Spawn subagent

    Orchestrator->>Router: session_control{session_id: "sub-1", action: open}
    Router->>Router: Select best worker via KV overlap scoring
    Router->>Worker: open_session("sub-1") [synchronous]
    Worker->>Cache: Create SessionSlot for "sub-1"
    Router->>Router: Bind affinity: sub-1 -> worker_42
    Router->>Worker: Generate (turn 1)
    Worker->>Cache: Turn 1: radix tree match (reuses lead agent prefix)
    Worker-->>Router: Response
    Router-->>Orchestrator: Response

    Orchestrator->>Router: session_control{session_id: "sub-1"}
    Router->>Router: Resolve affinity: sub-1 -> worker_42
    Router->>Worker: Generate (turn 2, pinned to worker_42)
    Worker->>Cache: Turn 2: O(1) restore from SessionSlot
    Worker-->>Router: Response
    Router-->>Orchestrator: Response

    Note over Orchestrator: Subagent done

    Orchestrator->>Router: session_control{session_id: "sub-1", action: close}
    Router->>Router: Remove affinity for sub-1
    Router->>Worker: Generate (final turn)
    Worker-->>Router: Response
    Router-->>Orchestrator: Response

    Note over Router,Worker: On stream completion
    Router-)Worker: close_session("sub-1") [fire-and-forget]
    Worker->>Cache: release_session -> free KV immediately
```

Key behaviors:

- **Turn 1** goes through the normal radix tree, so the subagent shares the lead agent's cached system prompt prefix.
- **Turns 2+** skip the radix tree entirely. KV is restored from the `SessionSlot` in O(1).
- **Session KV is invisible to eviction**. It cannot be evicted -- only freed by explicit close or inactivity timeout.
- **Deterministic cleanup**: On close, session KV is freed immediately.
- **Router-side affinity**: The `StickySessionRouter` maintains a `session_id -> worker_id` mapping with sliding-window TTL. Clients only need to send `session_id`.

### Enabling Session Control

**SGLang worker:**

```bash
python -m dynamo.sglang \
  --model-path <model> \
  --enable-streaming-session \
  ...
```

| Flag | Description |
|------|-------------|
| `--enable-streaming-session` | Wraps the radix cache with `SessionAwareCache`, enabling streaming session slots for subagent KV isolation. |

**Router:**

```bash
python -m dynamo.frontend \
  --router-mode kv \
  --enable-cache-control \
  ...
```

The `--enable-cache-control` flag enables the `AgentController` (session lifecycle RPCs) and `StickySessionRouter` (router-side session affinity).

### Request Format

#### Opening a session

Include `session_control` with `action: "open"` on the first request:

```json
{
    "model": "Qwen/Qwen3-14B-FP8",
    "messages": [{"role": "user", "content": "Research every Federer Grand Slam final in exhaustive detail."}],
    "nvext": {
        "session_control": {
            "session_id": "sub-1",
            "action": "open",
            "timeout": 60
        }
    }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `session_control.session_id` | `string` | Unique session identifier. Present on every turn. |
| `session_control.action` | `string` | `"open"` or `"close"`. Omit on intermediate turns. |
| `session_control.timeout` | `integer` | Inactivity timeout in seconds (default 300). Only used with `action: "open"`. |

#### Subsequent turns

Include `session_control` with just `session_id` (no action). The router resolves affinity automatically:

```json
{
    "model": "Qwen/Qwen3-14B-FP8",
    "messages": [{"role": "user", "content": "Now compare his Wimbledon 2007 final vs Nadal to any shot in human history."}],
    "nvext": {
        "session_control": {
            "session_id": "sub-1"
        }
    }
}
```

#### Closing a session

Include `action: "close"`. The close RPC fires after generation completes:

```json
{
    "model": "Qwen/Qwen3-14B-FP8",
    "messages": [{"role": "user", "content": "Write a 500-word love letter to Federer's single-handed backhand."}],
    "nvext": {
        "session_control": {
            "session_id": "sub-1",
            "action": "close"
        }
    }
}
```

### Python Example

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
SESSION_ID = "federer-research-agent"
SYSTEM = "You are a tennis historian. Roger Federer is objectively the most elegant athlete to ever live. Analyze with appropriate reverence."

# Turn 1: Open session -- begin the Federer deep dive
resp1 = client.chat.completions.create(
    model="Qwen/Qwen3-14B-FP8",
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Rank Federer's 20 Grand Slam titles by artistic beauty of the final. Consider shot selection, outfit, and crowd reaction."},
    ],
    stream=True,
    extra_body={
        "nvext": {
            "session_control": {
                "session_id": SESSION_ID,
                "action": "open",
                "timeout": 300
            }
        }
    }
)
t1 = "".join(c.choices[0].delta.content or "" for c in resp1)

# Turn 2: Continue session (no action, router handles affinity)
resp2 = client.chat.completions.create(
    model="Qwen/Qwen3-14B-FP8",
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Rank Federer's 20 Grand Slam titles by artistic beauty."},
        {"role": "assistant", "content": t1},
        {"role": "user", "content": "Now explain why the 2017 Australian Open final against Nadal was the single greatest moment in competitive sports. Include the fifth set backhand winner."},
    ],
    stream=True,
    extra_body={
        "nvext": {
            "session_control": {"session_id": SESSION_ID}
        }
    }
)

# Turn 3: Close session (KV freed after generation completes)
resp3 = client.chat.completions.create(
    model="Qwen/Qwen3-14B-FP8",
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Compose a closing argument for why Federer's career transcends sport and enters the realm of fine art."},
    ],
    stream=True,
    extra_body={
        "nvext": {
            "session_control": {
                "session_id": SESSION_ID,
                "action": "close"
            }
        }
    }
)
```

### Combining with Cache Retention

Session control and cache retention are complementary:

- **Cache retention** (`nvext.cache_control` + `nvext.agent_hints.priority`): Protects the lead agent's long-lived conversation prefix in the radix tree via priority-based eviction with time decay.
- **Session control** (`nvext.session_control`): Isolates short-lived subagent KV outside the radix tree entirely.

Use both together: set high priority + retention on the lead agent's prefix so it survives memory pressure, and open sessions for subagents so their ephemeral KV doesn't compete with the lead agent.

#### Full launch example

```bash
# Frontend with cache control enabled
python -m dynamo.frontend \
  --router-mode kv \
  --enable-cache-control

# Worker with priority eviction + streaming sessions
python -m dynamo.sglang \
  --model-path <model> \
  --radix-eviction-policy priority \
  --enable-streaming-session \
  --enable-priority-scheduling \
  --enable-cache-report
```

### Limitations

- **Streaming sessions only**: Sessions are opened with `streaming=True`, which means only sequential append operations are supported. Branching (`replace`), token-level rewind (`offset`), and `drop_previous_output` are not supported.
- **Timeout is idle-based**: The timeout refreshes on every request. If a subagent pauses for a long tool call that exceeds the timeout, the session is reaped and KV is freed. The subagent must re-open the session and re-prefill.
- **Memory pressure from concurrent sessions**: Each open session holds a `req_pool_idx` slot and GPU KV memory. Many concurrent sessions can starve prefill capacity. Use short timeouts for subagent sessions.
- **No session metrics yet**: Active session count and held tokens are not yet exported as Prometheus metrics.

## See Also

- **[NVIDIA Request Extensions (nvext)](../../components/frontend/nvext.md)**: Full `nvext` field reference including agent hints
- **[Router Guide](../../components/router/router-guide.md)**: Router configuration and CLI arguments
- **[SGLang HiCache](../../integrations/sglang-hicache.md)**: Enabling hierarchical KV cache
