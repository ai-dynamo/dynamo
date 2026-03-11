---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
subtitle: "Ishan Dhanani, Alec Flowers — March 2026"
description: "How Dynamo optimizes for agentic workloads at three layers: frontend API, router, and KV cache management."
keywords: agentic inference, KV cache, prefix caching, agent hints, disaggregated serving, Dynamo
last-updated: March 10, 2026
---

# Full-Stack Optimizations for Agentic Inference with Dynamo

Coding agents are starting to write production code at scale. [Stripe’s agents generate 1,300+ PRs per week](https://stripe.dev/blog/minions-stripes-one-shot-end-to-end-coding-agents). [Ramp attributes 30% of merged PRs to agents](https://www.infoq.com/news/2026/01/ramp-coding-agent-platform/). [Spotify reports 650+ agent-generated PRs per month](https://engineering.atspotify.com/2025/11/spotifys-background-coding-agent-part-1). Tools like Claude Code and Codex make hundreds of API calls per coding session, each carrying the full conversation history. Behind every one of these workflows is an inference stack under significant KV cache pressure. Agentic systems are the most cache-intensive workloads in production LLM inference today.

![Cumulative cache reads vs writes across a 42-call Claude Code session. Cache reads (891K tokens) grow steeply while writes (76K) and uncached input stay flat -- an 11.7x read/write ratio.](./cumulative-reads-writes.png)

After the first API call that writes the prefix to KV cache, every subsequent call to the same worker hits 85-97% cache. Agent teams (or swarms) push this further: 97.2% aggregate cache rate across 4 Opus teammates (measured in Claude Code team sessions). An 11.7x read/write ratio means the system reads from cache nearly 12 times for every token it writes.

These numbers come from managed API infrastructure where the provider controls prefix matching, cache placement, and eviction. For teams running open-source models on their own GPUs, none of this exists out of the box. We have been building Dynamo to close that gap. This post walks through how we are making Dynamo agent-native at three layers: the frontend API, the router, and KV cache management.

Throughout this post, we use three terms consistently:
- **Harness**: the agent framework that drives the workflow (Claude Code, Codex, OpenClaw, OpenCode, etc.)
- **Orchestrator**: Dynamo's routing, scheduling, and cache management layer
- **Runtime**: the inference engine that executes the model and owns the kv cache manager (SGLang, vLLM, TRT-LLM)

## Layer 1: The Frontend

### Multi-Protocol Support

Agent harnesses are increasingly adopting `v1/responses` and `v1/messages` over `v1/chat/completions` to cleanly handle new patterns including interleaved thinking and tool calls. The key difference in these APIs is structural. In `v1/chat/completions`, message content is a flat string and tool calls are bolted on as a separate field. As an example, notice how [GLM](https://docs.z.ai/guides/capabilities/thinking-mode#example-usage) and [MiniMax](https://platform.minimax.io/docs/guides/text-m2-function-call#important-note) API handle interleaved thinking differently when hosting their model behind the `v1/chat/completions` endpoint. The `v1/responses` and `v1/messages` APIs use typed content blocks, so a single assistant turn can contain thinking, tool calls, and text as distinct objects. This matters for inference because the orchestrator can see block boundaries, perform prompt optimizations, and apply different cache and scheduling policies per block type. Dynamo serves all three endpoints through a common internal representation, so a single deployment can act as the inference backend for any harness. Our team has internally been running a Dynamo deployment of GLM-5 and Minimax2.5 to power our Codex and Claude Code harnesses allowing us to understand and optimize our backend implementations to match closed-source inference and kv cache reuse performance. We will be sharing a full write-up and some optimized recipes for deploying both models in the upcoming weeks.

<table>
<tr>
<td width="50%">

**Serving Claude Code with Dynamo**

<video controls width="100%">
  <source src="https://github.com/user-attachments/assets/5fa8a224-44e8-4acb-943b-9b8af84815e6" type="video/mp4" />
</video>

</td>
<td width="50%">

**Serving Codex with Dynamo**

<video controls width="100%">
  <source src="https://github.com/user-attachments/assets/8694b544-3fd1-4931-9fd0-7d9a3a2fb78f" type="video/mp4" />
</video>

</td>
</tr>
</table>

We have also invested in day-0 tool call and reasoning parsing support for various open-source models. If you find that a model is not supported, please [open an issue](https://github.com/ai-dynamo/dynamo/issues) or use the [tool-call-parser-generator](https://github.com/ai-dynamo/dynamo/blob/main/.claude/skills/tool-parser-generator/SKILL.md) skill to generate it with your harness of choice.

### Agent Hints: The Harness-Orchestrator Interface

Agent workloads have a distinct set of characteristics when compared to traditional multi-turn chat workloads. When using coding agents, the user waits for a final result, not individual token streams, so the orchestrator can reorder and prioritize requests across agents without affecting the end-user experience. Sessions run for minutes to [even days](https://factory.ai/news/missions) with long tool-call pauses, and the harness has global context: which agents are mid-task and waiting on tool calls, which just spawned, how many turns remain in a session, and whether the current call is a quick lookup or a long synthesis. This is enough to optimize inference scheduling in ways that traditional serving cannot.

Today, inference servers see anonymous tokenized requests with none of this "global context".

![Where nvext fits in the agentic protocol stack alongside MCP and A2A](./protocol-stack.svg)

Dynamo’s new agent hints extension was designed to bridge this gap. It allows any harness to attach structured hints to a request across all three API endpoints, giving the router and runtime the context they need to make agent aware scheduling and caching decisions. This is a v1 API that we are actively co-designing with the community and would love feedback from teams building agent harnesses on what signals are most useful. Please reach out to us if you have any ideas or feedback!

```json
{
  "model": "MiniMaxAI/MiniMax-M2.5",
  "messages": [...],
  "tools": [...],
  "nvext": {
    "agent_hints": {
      "latency_sensitivity": 0.9,
      "osl": 256,
      "speculative_prefill": true,
      "priority": 10
    },
    "cache_control": {
      "type": "ephemeral",
      "ttl": "1h"
    }
  }
}
```

The `agent_hints` fields:

- **`latency_sensitivity`** and **`priority`** control scheduling at the router and engine respectively. We cover both in detail in the Priority Scheduling section below.
- **`osl`** (output sequence length) is the harness's estimate of how many tokens this request will generate. The router uses this to gauge how long a worker will be occupied, which improves load balancing. A harness can learn this over time by tracking average output lengths per tool call type.
- **`speculative_prefill`** signals the orchestrator to begin caching this request's prefix on a likely worker before the full request is ready. This is useful when the harness knows a tool call is about to return and wants to warm the cache ahead of time.

The `cache_control` field will look familiar to anyone who has used Anthropic's prompt caching API. It tells the orchestrator to pin the computed prefix on the worker for the specified TTL, protecting it from eviction during tool call gaps. Currently `ephemeral` is the only supported type (to match Anthropic's API). We discuss how this works in the cache retention section below. You can find complete documentation on agent hints [here](https://docs.nvidia.com/ai-enterprise/dynamo/latest/components/frontend/nvext.html#cache-control).

## Layer 2: The Router

### KV-Aware Placement

Without cache-aware routing, turn 2 of a conversation has a ~1/N chance of landing on the same worker as turn 1. Every miss is a full prefix recomputation which is a significant performance bottleneck and extremely costly for an end user. Dynamo's router maintains a global index of which KV cache blocks exist on which workers. The [Flash Indexer post](https://developer.nvidia.com/blog/building-a-high-performance-kv-cache-index-for-llm-inference-with-nvidia-dynamo/) covers the six iterations that got this indexer to 170M ops/s (**planetary** scale KV routing). On every request, the router queries the index for per-worker overlap scores and selects the worker that minimizes the combined cost of cache miss and current decode load. This cost function is tunable, and we show below how teams can build custom agent aware routing strategies on top of it.

### Priority Scheduling

Two fields in `agent_hints` control scheduling. They are separate knobs because they solve different problems at different layers:

- **`latency_sensitivity`** (0.0-1.0) controls **how soon the request gets dispatched** from the router queue. It answers: "how urgently does this request need to reach a worker?" A user-facing interactive turn (e.g., a lead agent responding to the developer) needs low latency; a background subagent doing a code search does not. The harness knows the difference and sets accordingly.

- **`priority`** (integer) controls **how the engine treats the request once it arrives**: scheduling order within the engine's batch and KV cache eviction policy. It answers: "how important is this request's compute and cache relative to other active requests on this worker?" A long-running synthesis request whose KV cache should survive memory pressure gets a high priority; a short lookup whose cache is disposable gets a low one.

At the router, incoming requests enter a `BinaryHeap<QueueEntry>` ordered by effective arrival time. A higher `latency_sensitivity` makes the request appear as if it arrived earlier, placing it ahead of lower-priority work. Requests only enter the queue when all workers exceed a configurable load threshold. Below that threshold, they bypass the queue entirely and go straight to worker selection. When capacity frees up (prefill completes or a request finishes), the queue drains highest-priority entries first.

Once dispatched, Dynamo passes `priority` through to the engine directly. SGLang, vLLM, and TRT-LLM all support priority-based request scheduling, and engines like SGLang support priority-based radix cache eviction where lower-priority blocks are evicted first under memory pressure.

![How latency_sensitivity and priority flow from harness through router dispatch to engine treatment](./two-gates.svg)

### Agentic Workload Routing Strategies

A coding agent follows a sequential pattern: long prefill, tool call, extend prefix, repeat. A multi-agent harness fans out work across parallel subagents with short, independent contexts. A research agent with a 200K context window needs workers with enough free KV capacity to hold its full state. The router's default cost function (overlap score + decode load) handles the common case, but teams with domain-specific workloads can easily leverage the router's Python bindings to implement custom routing strategies. The core `KvRouter` class provides `best_worker()` for querying routing decisions, `get_potential_loads()` for per-worker load inspection, and `generate()` for routing + dispatch in one call. Custom routers register on the same service mesh as the default components and can override routing config per-request:

```python
# Query per-worker load and overlap for custom routing logic
loads = await router.get_potential_loads(token_ids)

# Override routing config based on request properties
# Long contexts benefit from heavier overlap weighting
config = {"overlap_score_weight": 2.0} if len(token_ids) > 8192 else {}
worker_id, dp_rank, overlap = await router.best_worker(
    token_ids,
    request_id="req-123",
    router_config_override=config
)

# Or bypass the default selector entirely when the harness
# has its own worker selection logic (e.g., session affinity)
stream = await router.generate(
    token_ids, model=model, worker_id=chosen_worker
)
```

The [NeMo Agent Toolkit (NAT)](https://github.com/NVIDIA/NeMo-Agent-Toolkit/tree/develop/examples/dynamo_integration) team used these APIs to build a custom online-learning agentic router. Their router extracts session metadata from `nvext` annotations and feeds it to a [Thompson Sampling](https://en.wikipedia.org/wiki/Thompson_sampling) bandit style cost function that learns which workers perform best for which prefix patterns under load. Compared to Dynamo's default routing, they measured 4x reduction in p50 TTFT and 1.5x increase in p50 tokens-per-second. Priority tagging of latency-sensitive requests achieved up to 63% p50 TTFT reduction under moderate memory pressure. See the [NAT Dynamo integration example](https://github.com/NVIDIA/NeMo-Agent-Toolkit/tree/develop/examples/dynamo_integration) for implementation details. We will be making this available as a routing strategy in Dynamo soon.


## Layer 3: KV Cache Management

### The Problem with Uniform Eviction

Agentic workloads produce blocks with vastly different reuse value but default LRU eviction policies treat all KV blocks identically.

| Block Type | Reuse Pattern | Value |
|------------|---------------|-------|
| System prompt + tool definitions | Every turn | Highest |
| Conversation history | Subsequent turns, growing monotonically | High |
| Thinking/reasoning tokens | Typically zero reuse after reasoning loop closes (a significant portion of output) | Near-zero |
| Subagent KV | Multiple turns then agent dies. No need to retain | Near-zero |

LRU sees only recency. In a high traffic environment, a wait for the completion of a called tool (2-30 seconds while the agent waits for an external API) might cause the agent's blocks to age out and when the agent resumes, the entire prefix must be recomputed. To solve this, we need to provide the orchestrator APIs to control which blocks should be retained, where they should live, and for how long.

### KV Cache as a Shared Resource

Today, KV cache is treated as a local, ephemeral resource on each worker. An agent's ~32K-token system prompt and tool definitions are computed independently on every worker that serves its requests. When a lead agent spawns 4 subagents, each with overlapping tool definitions, that shared prefix is recomputed 4 times if the subagents land on different workers. In our analysis of Claude Code team sessions, we measured this directly: teammates averaged 79.4% cache hit rate vs. 91.3% for the lead agent's explore subagents (5.0x vs. 11.7x read/write ratio), with the gap driven almost entirely by cold-start writes on each teammate's first call. The goal is to make high value KV cache blocks available to all workers in the cluster. Essentially, they are written once during cold start and then read by any worker at all times.

Solutions like SGLang's HiCache and Dynamo's KV Block Manager (KVBM) are building toward a 4-tier memory hierarchy:

![KV cache memory hierarchy: GPU (HBM) at ~ns, CPU (pinned DRAM) at ~us, Local NVMe at ~ms, Remote Storage (NIXL) at ~ms via RDMA](./kv-memory-hierarchy.svg)

Blocks follow a write-through path: when a worker computes KV for a prefix, the blocks flow from GPU to CPU to disk automatically. Each block is deduplicated by sequence hash in a global registry. Once a block is registered, it is immutable and addressable by any worker that can reach the storage tier.

This directly solves the subagent cold-start problem. When the lead agent computes tool definitions and system prompt, those blocks write through to shared storage. When subagent 1 spawns on a different worker, the router queries the Flash Indexer, finds the blocks in shared storage, and the worker loads them via NIXL (RDMA read) instead of recomputing from scratch. Subagent 2 does the same. Four redundant prefill computations become one compute and three loads. The same mechanism addresses cache coherence in disaggregated prefill-decode serving. In disagg mode, the prefill worker computes KV and transfers it to the decode worker via NIXL. The decode worker generates tokens, producing new KV state. On the next turn, a prefill worker needs both the original prefix and the generated tokens from turn 1, but those live only on the decode worker. With shared storage, the decode worker writes its new blocks to the common tier and any prefill worker can fetch them on the next turn.

Multi-tier storage solves sharing and persistence, but blocks still arrive on GPU only after the request hits the worker. The missing piece for agentic systems is prefetch: the harness can leverage historical timing data to predict when an agent's tool call might return, which means it knows which blocks will be needed and when. We are building prefetch hooks so the harness can signal "bring these blocks from storage to GPU ahead of the next request." Combined with the retention APIs (below), this gives the harness full lifecycle control: pin blocks to prevent eviction, set priority to control eviction ordering, and prefetch blocks proactively before they are needed.

![During tool calls, KV blocks offload to host memory and storage, then prefetch back to GPU before the next LLM call.](./tool-call-offload-prefetch.svg)

### Selective Cache Retention

Making blocks globally available solves the sharing problem, but does not solve eviction. SGLang and vLLM both support priority-based eviction via a priority heap where the harness assigns a numeric priority per request and lower-priority blocks are evicted first. TensorRT-LLM takes this further with `TokenRangeRetentionConfig` (designed and implemented by a Dynamo team member[@jthomson04](https://github.com/jthomson04)) which allows per-region control within a single request.

A request carries zero or more directives. Blocks without directives follow the default LRU path with zero overhead. The evictor becomes a two-structure system: an LRU free list for unprioritized blocks (O(1), unchanged) and a priority queue for annotated blocks. The harness can express "system prompt blocks are evicted last (priority: 100); conversation context survives a 30-second tool call (duration: 45s); decode tokens are first to go (priority: 1)" without the engine needing to understand why.

Anthropic's prompt caching lets you mark prefixes as cacheable on their infrastructure. Dynamo's `cache_control` API brings the same semantics to self-hosted inference. When a request includes `cache_control: { type: "ephemeral", ttl: "1h" }`, the router pins the matching prefix nodes in the worker's radix tree for that TTL, protecting them from eviction in the worker's L2 storage.

The next step is connecting retention with the distributed cache. Today, retention directives apply to a single worker's local cache. When a block is pinned on worker A but the next request routes to worker B, the pin does not follow. Extending retention semantics across HiCache/KVBM's shared storage tier means the harness can pin a block once and have it survive across workers: the priority and TTL metadata travel with the block through the write-through path, and any worker that loads the block from shared storage inherits the retention policy. Combined with the prefetch hooks described above, this gives the harness end-to-end lifecycle control across the full memory hierarchy.

### Agent Lifecycle Awareness

Consider a typical Claude Code session. The lead agent runs for 20+ turns, accumulating a growing conversation prefix. Along the way it spawns explore subagents that each run 1-3 turns and terminate. It might spawn a team of 4 specialists that work in parallel on different subtasks and then terminate. Midway through, the agent hits a context limit and summarizes its history, compressing ~175K tokens down to ~40K. Each of these events produces ephemeral KV: blocks that will never be referenced again. Subagent termination, context summarization, and closed reasoning loops all generate ephemeral KV that occupies the same memory as high-value blocks like the system prompt. Reasoning models amplify this: `<think>...</think>` blocks account for ~40% of generated tokens but become ephemeral the moment the reasoning loop closes. Without lifecycle awareness, the cache treats all of these blocks identically.

![Lead agent conversation flow branches to a sub-agent. The sub-agent's ephemeral KV is evicted on session end.](./subagent-lifecycle-vertical.svg)

The retention primitives from above (priority, TTL, token ranges) give us the building blocks. What is missing is the ability to associate them with sessions. If the harness can tag a subagent's requests as belonging to a session and mark that session's KV as ephemeral, the evictor can target those blocks first and skip writing them to shared storage entirely. When the subagent terminates, its session's blocks are the first to reclaim. The same mechanism applies to thinking tokens: the engine can detect `<think>` boundaries during generation and tag those blocks as ephemeral at insertion time, so they skip L2 write-back and evict before normal blocks without any external signal. The design space here is wide: harness-driven session tagging, engine-native semantic detection, hybrid approaches that combine both. We are actively exploring multiple directions and expect the right answer will vary by workload and framework.

## Closing the Gap

The biggest optimization surface in agentic inference is the gap between what the harness knows and what the infrastructure can see. Which agents are blocked, which are about to resume, which KV is worth keeping, which can be thrown away -- all of this context exists at the harness layer but never crosses the API boundary. `nvext.agent_hints` is our first cut at closing that gap: a small set of structured signals that let the orchestrator make informed routing, scheduling, and cache management decisions instead of treating every request as anonymous tokens. This is a v1 API and we are actively evolving it. If you are building agent harnesses, running open-source models for agentic workloads, or thinking about cache-aware inference, we want to hear what signals matter most for your use case. Reach out on [GitHub](https://github.com/ai-dynamo/dynamo) or tag us on X: [@0xishand](https://x.com/0xishand), [@KranenKyle](https://x.com/KranenKyle), [@flowpow123](https://x.com/flowpow123).
