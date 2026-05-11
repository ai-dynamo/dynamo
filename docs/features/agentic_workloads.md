---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Agents
subtitle: Workload-aware inference with agentic hints for routing, scheduling, and KV cache Management
---

### Gaps with workload-agnostic inference

Agentic LLM inference is [dominated by KV-cache storage and I/O](https://arxiv.org/abs/2602.21548) rather than computation; without leveraging the predictable structure of agent lifecycles, we leave significant optimizations on the table.
Three gaps stand out with current workflows:

1. **Reactive vs. proactive:** Current runtimes do not use signals from the harness about what will happen next—e.g. that a "Plan" step is done and "Execute" steps are coming—so they cannot prefetch, pin, or schedule proactively.

2. **All KV-cache blocks treated equally:** Generic eviction (e.g. LRU) does not distinguish high-value, long-lived context (system prompt, tool definitions) from ephemeral context (chain-of-thought, scratchpad).

3. **Workload-agnostic scheduling:** Agents have predictable structure. Tools and system prompts repeat across turns, shallow vs. deep research have different latency needs, and the orchestrator knows which phase comes next.



## Dynamo as an Agentic Runtime

Dynamo exposes **agentic hints** and uses them at the frontend API, router, and backend scheduling layers. Together, these enable workload-aware inference instead of generic, state-of-the-moment optimization.

### Agent Context

Agent context is passive workflow and program identity carried in
[`nvext.agent_context`](../components/frontend/nvext.md#agent-context). It lets
an agent harness label each request with `workflow_type_id`, `workflow_id`,
`program_id`, and optional `parent_program_id` so Dynamo request traces can be
joined with harness-side events. It does not change routing or scheduling.

See [Agent Context and Tracing](../agents/agent-context.md) for the request
contract, trace sink configuration, and JSONL trace schema.

### Agentic Hints

Agentic hints are per-request metadata that the agent client (e.g. Claude Code, Codex, [NeMo Agent Toolkit](https://github.com/NVIDIA/NeMo-Agent-Toolkit)) sends to Dynamo's frontend. They are carried in the request body under [**nvext**](../components/frontend/nvext.md#agent-hints) on chat completions. The frontend parses them and passes them to the KV router and, where applicable, to backends.

- **Flow:** Harness sets hints in the request → Dynamo frontend parses `nvext` into routing hints → KV router uses them for queue ordering and worker selection → backends use them for priority scheduling and cache eviction.

![Agentic workflow: Harness → hints in request → Dynamo frontend → routing hints → KV router (queue order, worker choice) → backend](../assets/img/agentic-hints-workflow.svg)

The request body includes `nvext.agent_hints` for routing and scheduling metadata that the frontend passes through to the KV router and backend runtime.

| Hint | Description |
|------|-------------|
| `priority` | Unified request priority. Higher values move the request earlier in the router queue and are forwarded to the backend for scheduling and priority-based eviction. |
| `osl` | Expected output sequence length (tokens). Used by the router for output block tracking and load-balancing accuracy when `--router-track-output-blocks` is enabled. |
| `speculative_prefill` | When true, after the assistant turn completes the system prefills the predicted next-turn prefix (conversation history + assistant text, e.g. thinking stripped) to warm the KV cache for the next request. |
| `program_id` | (Planned) Identifies the agentic program for program-level metrics and cache affinity. |
| `context_type` | (Planned) Semantic type (e.g. system prompt, tool definition, reasoning branch) for context-aware eviction. |

## Feature matrix

| Feature | vLLM | SGLang | TensorRT-LLM |
|---------|:----:|:------:|:-------------:|
| Priority-based cache eviction | 🚧 | ✅ | 🚧 |
| Subagent KV isolation (session control) | | 🚧 | |
| Cache prefetching | | 🚧 | |
| Subagent / thinking-aware cache eviction | | 🚧 | |
| Speculative prefill | ✅ | ✅ | ✅ |
| Priority-aware routing | ✅ | ✅ | ✅ |

🚧 = Work in progress or experimental.

### Using Dynamo from LangChain

Dynamo is now supported directly in LangChain using the [NVIDIA AI Endpoints integration](https://docs.langchain.com/oss/python/integrations/chat/nvidia_ai_endpoints#use-with-nvidia-dynamo). Configure the chat model to use the Dynamo endpoint and pass agent hints directly from the LangChain client.



## Features (experimental)

### KV cache optimizations

- **Priority-based KV cache eviction:** Instead of evicting by LRU alone, the backend can evict **low-priority** cache entries first when the GPU (and, with HiCache, host) cache is full. The `priority` value in `nvext.agent_hints` is forwarded to the engine; with SGLang, enable `--enable-priority-scheduling` and `--radix-eviction-policy priority`.

- **Subagent KV isolation (experimental):** Session control holds subagent KV in dedicated streaming session slots outside the radix tree. Session KV is invisible to eviction and freed deterministically on close or timeout. The router manages sticky session affinity so subsequent turns always hit the same worker. See [SGLang for Agentic Workloads -- Session Control](../backends/sglang/agents.md#session-control-for-subagent-kv-isolation-experimental).

- **Cache prefetching (future work):** Using the predictable agentic lifecycle (e.g. parent-child subagents, known next turn), Dynamo could proactively prefetch or move KV cache to a different worker so that the next request hits warm cache.

### Speculative prefill

After a turn finishes, the system can send a **speculative** `max_tokens=1` prefill with the **predicted next-turn prefix** (conversation history + assistant text, e.g. thinking stripped) to the same worker. When the real next request arrives, it hits a warm KV cache. Per-turn TTFT on turns 2+ can drop significantly (e.g. up to ~3× in [multiturn benchmarks](https://github.com/ai-dynamo/dynamo/blob/main/lib/bench/README.md)). This can be extended so that Dynamo automatically sends tools and system prompt for subagents to a worker in advance, so subagent requests always hit warm cache.

### Priority-aware routing

When `--router-queue-threshold` is set, the router maintains a priority queue. Requests with higher `priority` are treated as if they arrived earlier, so they are scheduled ahead of bulk or background work. Under load, this keeps median latency low for user-facing agent turns while background work can tolerate higher latency. For a runnable demo and results, see [NeMo Agent Toolkit priority demo](https://github.com/NVIDIA/NeMo-Agent-Toolkit/tree/develop/examples/dynamo_integration/latency_sensitivity_demo).

---

## See also

- [NeMo Agent Toolkit — Dynamo integration](https://github.com/NVIDIA/NeMo-Agent-Toolkit/tree/develop/examples/dynamo_integration)
- [Context engineering for AI agents (Manus)](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- [Stateful runtime for agents (OpenAI/Bedrock)](https://openai.com/index/introducing-the-stateful-runtime-environment-for-agents-in-amazon-bedrock/)
- [Claude Code's Prompt Caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
