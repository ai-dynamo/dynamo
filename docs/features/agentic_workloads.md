---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Agentic Workloads
subtitle: Workload-aware inference with agentic hints for routing, scheduling, and KV cache Management
---

### Gaps with workload-agnostic inference

Agentic LLM inference is [dominated by KV-cache storage and I/O](https://arxiv.org/abs/2602.21548) rather than computation; without leveraging the predictable structure of agent lifecycles, we leave significant optimizations on the table. 
Three gaps stand out with current workflows:

1. **Reactive vs. proactive:** Current runtimes do not use signals from the harness about what will happen next—e.g. that a "Plan" step is done and "Execute" steps are coming—so they cannot prefetch, pin, or schedule proactively.

2. **All KV-cache blocks treated equally:** Generic eviction (e.g. LRU) does not distinguish high-value, long-lived context (system prompt, tool definitions) from ephemeral context (chain-of-thought, scratchpad). 

3. **Workload-agnostic scheduling:** Agents have predictable structure. Tools and system prompts repeat across turns, shallow vs. deep research have different latency needs, and the orchestrator knows which phase comes next.



## Dynamo as an Agentic Runtime

Dynamo exposes **agentic hints** and uses them at three layers: frontend API, router, and KV cache management. Together, these enable workload-aware inference instead of generic, state-of-the-moment optimization.

### Agentic Hints

Agentic hints are per-request metadata that the agent client (e.g. Claude Code, Codex, [NeMo Agent Toolkit](https://github.com/NVIDIA/NeMo-Agent-Toolkit)) sends to Dynamo's frontend. They are carried in the request body under [**nvext**](../components/frontend/nvext.md#agent-hints) on chat completions. The frontend parses them and passes them to the KV router and, where applicable, to the KV cache manager and backends.

- **Flow:** Harness sets hints in the request → Dynamo frontend parses `nvext` into routing hints → KV router uses them for queue ordering and worker selection → backends use them for priority scheduling and cache eviction.

![Agentic workflow: Harness → hints in request → Dynamo frontend → routing hints → KV router (queue order, worker choice) → backend](../diagrams/agentic_workflow.png)

The request body includes `nvext.agent_hints` (routing, scheduling) and `nvext.cache_control` (TTL-based pinning); the frontend passes the former to the KV router and the latter to the KV block manager for cache pinning, prefetching, and eviction.

| Hint | Description |
|------|-------------|
| `latency_sensitivity` | Router queue priority (requires `--router-queue-threshold`). Higher values shift the request earlier in the queue so user-facing turns run before background work. |
| `priority` | Engine queue ordering and KV cache eviction. Forwarded to the backend for scheduling and priority-based eviction. |
| `osl` | Expected output sequence length (tokens). Used by the router for output block tracking and load-balancing accuracy when `--router-track-output-blocks` is enabled. |
| `speculative_prefill` | When true, after the assistant turn completes the system prefills the predicted next-turn prefix (conversation history + assistant text, e.g. thinking stripped) to warm the KV cache for the next request. |
| `program_id` | (Planned) Identifies the agentic program for program-level metrics and cache affinity. |
| `context_type` | (Planned) Semantic type (e.g. system prompt, tool definition, reasoning branch) for context-aware eviction. |

**`nvext.cache_control`** (sibling of `agent_hints`, not inside it) provides TTL-based KV cache pinning. Pinned prefixes resist eviction for the specified duration. See [SGLang for Agentic Workloads — Cache Pinning](../backends/sglang/agents.md#cache-pinning-experimental).


## Feature matrix

| Feature | vLLM | SGLang | TensorRT-LLM |
|---------|:----:|:------:|:-------------:|
| Priority-based cache eviction | 🚧 | ✅ | 🚧 |
| Cache pinning | | ✅ | 🚧  |
| Cache prefetching | | 🚧 | |
| Subagent / thinking-aware cache eviction | | 🚧 | |
| Speculative prefill | ✅ | ✅ | ✅ |
| Latency-sensitivity–aware routing | ✅ | ✅ | ✅ |

🚧 = Work in progress or experimental.

### Using Dynamo from LangChain

Dynamo is now supported directly in LangChain using the [NVIDIA AI Endpoints integration](https://docs.langchain.com/oss/python/integrations/chat/nvidia_ai_endpoints#use-with-nvidia-dynamo). Configure the chat model to use the Dynamo endpoint and pass agent hints directly from the LangChain client.



## Features (experimental)

### KV cache optimizations 

- **Priority-based KV cache eviction:** Instead of evicting by LRU alone, the backend can evict **low-priority** cache entries first when the GPU (and, with HiCache, host) cache is full. The `priority` value in `nvext.agent_hints` is forwarded to the engine; with SGLang, enable `--enable-priority-scheduling` and `--radix-eviction-policy priority`.

- **Cache pinning (experimental):** [Anthropic's v1/messages](https://docs.anthropic.com/en/docs/build-with-claude/caching) includes a `cache_control` field that tells servers how long to keep KV cache for specific blocks. Dynamo implements an OSS version with SGLang's HiCache: users can set `cache_control` via the same API as Anthropic or as an `nvext` field on chat completions. When set, the Dynamo router calls a hook in HiCache after the request completes to **pin** the blocks created by those tokens for the user-specified TTL. Pinned nodes resist eviction (demoting to host memory rather than being deleted).  
    In the Nemo Agentic toolkit and Dynamo integration, TTL is dynamically computed as the product of how many times a block is expected to be reused and the time between those requests; the NAT profiler pre-computes these expectations during agent evaluations and stores them in a data structure per agent, then injects `nvext.cache_control` with the derived TTL (see [dynamo_llm.py](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/develop/packages/nvidia_nat_core/src/nat/llm/dynamo_llm.py)). 

    **Future work:** TTL could be determined dynamically by context type—e.g. think tokens or scratchpad content could use a lower TTL than system prompt or tool definitions, so high-value static context is retained longer while ephemeral context expires sooner. 

- **Cache prefetching (future work):** Using the predictable agentic lifecycle (e.g. parent-child subagents, known next turn), Dynamo could proactively prefetch or move KV cache to a different worker so that the next request hits warm cache. 

### Speculative prefill 

After a turn finishes, the system can send a **speculative** `max_tokens=1` prefill with the **predicted next-turn prefix** (conversation history + assistant text, e.g. thinking stripped) to the same worker. When the real next request arrives, it hits a warm KV cache. Per-turn TTFT on turns 2+ can drop significantly (e.g. up to ~3× in [multiturn benchmarks](https://github.com/ai-dynamo/dynamo/blob/main/lib/bench/src/bin/README.md)). This can be extended so that Dynamo automatically sends tools and system prompt for subagents to a worker in advance, so subagent requests always hit warm cache.

### Latency-sensitivity–aware routing

When `--router-queue-threshold` is set, the router maintains a priority queue. Requests with higher `latency_sensitivity` are treated as if they arrived earlier, so they are scheduled ahead of bulk or background work. Under load, this keeps median latency low for user-facing agent turns while background work can tolerate higher latency. For a runnable demo and results, see [NeMo Agent Toolkit latency sensitivity demo](https://github.com/NVIDIA/NeMo-Agent-Toolkit/tree/develop/examples/dynamo_integration/latency_sensitivity_demo).

---

## See also

- [NeMo Agent Toolkit — Dynamo integration](https://github.com/NVIDIA/NeMo-Agent-Toolkit/tree/develop/examples/dynamo_integration)
- [Context engineering for AI agents (Manus)](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- [Stateful runtime for agents (OpenAI/Bedrock)](https://openai.com/index/introducing-the-stateful-runtime-environment-for-agents-in-amazon-bedrock/)
- [Claude Code's Prompt Caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)