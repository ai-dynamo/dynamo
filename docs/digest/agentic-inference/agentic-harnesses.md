---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
subtitle: "Matej Kosec, Ishan Dhanani, Benjamin Klieger, Dan Gil and Alec Flowers — April 2026"
description: "Streaming Tokens and Tools: Multi-Turn Agentic Harness Support in Dynamo"
keywords: agentic inference, responses api, messages api, tool calling, interleaved reasoning, prefix caching, agent hints, Dynamo
last-updated: Apr 30, 2026
---

# Streaming Tokens and Tools: Multi-Turn Agentic Harness Support in Dynamo

An agentic exchange must preserve a structured interaction: assistant turns interleave reasoning with one or more tool calls, and subsequent user turns return the corresponding tool results to the model context. Reasoning replay is model- and turn-dependent: some reasoning should be retained, while some should be dropped. The inference engine is responsible for this more expressive interaction and for producing correctly segmented API results. Tool-call parsing and reasoning parsing need to happen before the attached harness consumes the response. High-value agentic workflows such as coding also depend on a responsive harness experience: reasoning segments, tool-call events, and request metadata need to stream back as the turn unfolds instead of arriving after a final text response. This post covers lessons from running real agentic clients against Dynamo: how we hardened parser and API coverage and how those parser layers became standalone reusable crates.

These changes build on the performance considerations outlined in our [first post](./agentic-inference.md), which focused on the serving architecture underneath agentic inference: the frontend, router, and KV cache management. This follow-up focuses on correctness, user-experience equivalence, and performance.

Agentic harnesses are still evolving quickly. Claude Code, Codex, and OpenClaw expose the same pressure points from different API surfaces, so the examples below focus on the core behaviors that custom serving stacks need to reproduce.

![Standard server vs Dynamo across two turns of an agent loop. Each turn crosses the network at two edges: harness-to-server (where prompt stability decides cache reuse) and server-to-harness (where streaming tool dispatch decides when the harness can act). Dynamo changes both edges.](./images/fig-1-agent-loop.svg)

## Harness-Facing Dynamo Settings

Our experiments used the newly released `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` model, though the same issues apply across models, reasoning parsers, and tool-call parsers.

To reproduce our results, configure the frontend with the Anthropic-compatible API and the flags that preserve prompt, reasoning, and tool state:

- `--enable-anthropic-api` exposes the Anthropic Messages API to harnesses. Many harnesses can fall back to the default Messages API, but the experience is degraded.
- `--strip-anthropic-preamble` removes the Anthropic billing header that can destabilize KV reuse.
- `--enable-streaming-tool-dispatch` lets complete tool calls start executing as soon as they are decoded, rather than waiting for the end of the turn.

Putting all of this together:
```bash
python -m dynamo.frontend \
  --http-port 8000 \
  --enable-anthropic-api \
  --strip-anthropic-preamble \
  --enable-streaming-tool-dispatch
```

On the worker side, the important settings in this deployment are:

- `--dyn-tool-call-parser <parser>` and `--dyn-reasoning-parser <parser>` reconstruct tool calls and reasoning blocks in the model-specific format the harness expects. Those parsers also control whether reasoning from previous turns should be retained, transformed, or dropped.

## Prompt Stability Is Key for Cache Reuse

Claude Code sends thousands of tokens of reusable prompt scaffolding, much of which is designed to be the same across different users and sessions. However, at the very front of each prompt is a session-specific billing header which causes cache misses when pointed at custom endpoints that do not strip it out:

```text
x-anthropic-billing-header: cc_version=0.2.93; cch=abc123def456==;
You are Claude Code, an interactive CLI tool...
```

These headers poison the KV cache and prevent it from being reused, even across sessions by the same user. A varying line at position zero means every new session starts from a different token prefix, so the stable instructions and tool definitions behind it never line up cleanly for reuse.

To restore KV-cache reuse, Dynamo added `--strip-anthropic-preamble`. The fix is mechanically small and operationally important: remove the unstable billing header before tokenization so that the stable prompt starts at token zero.

The measured impact was large. On a Dynamo B200 deployment with a 52K-token prompt, a stable prefix landed at `168ms` TTFT. Keeping a varying per-session header in the prefix pushed that to `912ms`. Removing the billing header before tokenization brought it back to `169ms`. On this workload, the unstable header costs `744ms` per request and turns a reusable system prompt into a cold prefill. That is about a `5x` reduction in TTFT for new users hitting the same deployment or for the same user opening a new session.

![TTFT across three prompt-prefix conditions on a 52K-token prompt: a stable prefix lands at 168 ms, the stripped Anthropic preamble at 169 ms, while a varying per-session billing header pushes TTFT to 912 ms — a single token at position zero is the difference between hot KV reuse and a full cold prefill.](./images/fig-2-ttft-prefix-stability.svg)

## The Nuances of Reasoning and Tool Parsing

Reasoning replay into the next turn does not have one universal correct form. Some models intentionally drop prior thinking on ordinary assistant turns. Agentic turns with interleaved tool calls are different: there, the reasoning spans often need to remain attached to the tool calls they explain. The real contract is model-specific and turn-specific.

Anthropic's [April 23 Claude Code postmortem](https://www.anthropic.com/engineering/april-23-postmortem) gives a concrete production example of this policy: thinking from previous turns can be cleared on session resume to reduce the prefill burden after the cached prompt has expired.

Contemporary reasoning models tend to produce two different kinds of assistant turns:
- reasoning followed by a direct response to the user
- reasoning followed by one or more tool calls

Agentic models are especially good at producing turns where many reasoning and tool-call segments appear within a single response in the pattern of:

```text
<think>reasoning_0</think> tool_call_0 <think>reasoning_1</think> tool_call_1
```

On the next turn each reasoning span needs to stay attached to the tool call it explains. Dynamo now supports this interleaved format fully. Previously, the same turn could be reconstructed as:

```text
<think>reasoning_0 reasoning_1</think> tool_call_0 tool_call_1
```

If the assistant turn is reconstructed as one generic reasoning block followed by one blob of tool calls, the model still has all the same tokens but loses the sequence and delimiters that made them meaningful. This grouped ordering came from legacy models that emitted only a single reasoning span and a single tool-call pass per turn.

In addition to the reordering bug, we also found that reasoning was often being dropped too aggressively before the next turn. For some models, dropping prior thinking on turns without tool calls is an established behavior and part of the model's fine-tuning (DeepSeek-R1 is the clearest example). But that same behavior is wrong for interleaved agentic turns where the prior reasoning explains the tool sequence. This issue was difficult to spot because users could see reasoning being decoded correctly in the outgoing response while it was still being *silently malformed* or dropped before the next turn.

We validated this against a Dynamo + TRT-LLM deployment: Nemotron-3-Super-120B-A12B-NVFP4 on 4x B200 with TP=4, with `--enable-anthropic-api`, `--strip-anthropic-preamble`, `--enable-streaming-tool-dispatch`, the `nemotron_deci` reasoning parser, and the `qwen3_coder` tool call parser.

### Combined Reasoning and Tool Calls

A model that reasons before calling a tool generates a response where `<think>` content flows first, followed by `<tool_call>` XML. In the case of Nemotron, two different parsers, `nemotron_deci` for reasoning and `qwen3_coder` for tool calls, have to split that stream into the correct Anthropic Messages API content blocks without interfering with each other.

We sent the same prompt five times through the Anthropic Messages API: a system prompt instructing the model to think step by step, two tool definitions (calculator and weather), and the user message "Think carefully about what 15 * 23 equals, then use the calculator to verify." The response structure from a representative round:

```json
{
  "content": [
    {
      "type": "thinking",
      "thinking": "I need to calculate 15 * 23. Let me think: 15 * 20 = 300, and 15 * 3 = 45, so 300 + 45 = 345. I'll use the calculator to verify.\n"
    },
    {
      "type": "tool_use",
      "id": "call-a3364797-3160-4e84-b567-5c495694d502",
      "name": "calculator",
      "input": { "expression": "15 * 23" }
    }
  ],
  "stop_reason": "tool_use",
  "usage": { "input_tokens": 403, "output_tokens": 95 }
}
```

### Streaming Two Parsers at Once

The streaming path makes the parser interaction more visible. A streaming request produces a sequence of SSE events, and the event type sequence shows exactly how the two parsers carve up the token stream:

```text
   1ms  message_start
  82ms  content_block_start  type=thinking
  82ms  content_block_delta  (thinking tokens stream here, ~7ms apart)
   ...  (~70 thinking deltas over ~520ms)
 602ms  content_block_stop
 602ms  content_block_start  type=text
 602ms  content_block_delta
 800ms  content_block_stop
 800ms  content_block_start  type=tool_use
 800ms  content_block_delta
 800ms  content_block_stop
 814ms  message_delta        stop_reason=tool_use
 814ms  message_stop
```

The thinking block streams token by token from `82ms` to `602ms`. Then a brief text block appears (the whitespace between the thinking and tool call regions of the raw token stream). Then the tool_use block arrives at `800ms` as a single structured unit. The `message_stop` follows at `814ms`.

This round-trip did not produce the correct Anthropic event sequence until [PR #7358](https://github.com/ai-dynamo/dynamo/pull/7358). The fix had three parts:

1. **One owner for reasoning parsing**: reasoning parsing used to happen at multiple competing layers. The backend parser could split model output into `reasoning_content` and normal `content`, while the Anthropic streaming converter still tried to infer `<think>` boundaries when mapping the same stream into Anthropic content blocks. PR #7358 made ownership explicit. If a backend path has already produced structured reasoning deltas, the Anthropic converter trusts them and only maps them into the response format.

2. **Template-native reasoning when available**: Dynamo now checks whether the active chat template knows how to read `reasoning_content`. Templates like Nemotron and Qwen3 read that field directly, so Dynamo leaves it alone and lets the template decide how much prior thinking to keep. If the template only understands `content`, Dynamo falls back to the legacy representation: preserve reasoning by inserting `<think>` blocks into `content`, or leave it out when the model/parser policy says prior thinking should not carry into the next turn. Both the Rust preprocessor path (`ModelInput::Tokens`) and the Python worker path (`ModelInput::Text`) use this same conditional rule.

3. **Respect per-request thinking controls**: Many templates default `truncate_history_thinking=true` to save context. That is reasonable for ordinary chat, but it removes the reasoning behind prior tool calls in agent workflows. Dynamo now changes that behavior only for requests where reasoning is actually in play: when a reasoning parser is configured and the client has not disabled thinking, the Anthropic path sets `enable_thinking=true` and `truncate_history_thinking=false`. That keeps the next-turn context agents need without changing the default for requests or models that should run without thinking.

In our B200 experiment with a 52K-token system prompt and an assistant turn containing about 500 tokens of thinking, the unchanged next-turn prefix landed at `167ms` TTFT while mutated thinking landed at `322ms`. That is a `1.9x` increase, or about `155ms` per request, from changing the reasoning content inside the next-turn prefix.

The key takeaway is that the harness, parser, and template path must agree on each model's expected reasoning behavior. Dropping thinking on ordinary turns may be correct for one model and wrong for another. Preserving interleaved reasoning on tool-calling turns may be essential even when ordinary turns are allowed to strip it. **In practice, you should not assume that the tokens produced on turn `N` will automatically arrive unchanged as the prefix of turn `N+1`.** Whether that is true depends on the reasoning parser, tool parser, and chat template for the model you are serving.

## Streaming Tool Calls

Streaming tokens make the user experience feel more responsive and dynamic. The hard part is preserving that streaming behavior while still emitting tool calls as coherent blocks. In the older Dynamo path, reasoning tokens streamed back normally, but tool calls stayed buffered until the very end of the turn before being released all at once to the harness. That reduces responsiveness and delays tool execution even when the model has already decided what to call.

| State | What the harness sees | When tool readiness becomes visible |
|-------|------------------------|-------------------------------------|
| Buffered | tool-call chunks withheld | only at `finish_reason: "tool_calls"` |
| Inline streaming | regular tool-call deltas | as soon as the model emits them |
| Dispatch | typed `event: tool_call_dispatch` side channel | at the same structural completion point, but already parsed |

The important change is from the first row to the latter two. That is where the harness stops waiting for stream end to learn that it needs to act.
Without dispatch, the harness sees a regular token stream and has to infer when a tool call is complete by accumulating deltas and waiting for enough structure to be present. With dispatch enabled, Dynamo can emit a typed SSE side channel:

```text
event: tool_call_dispatch
data: {"choice_index":0,"tool_call":{"index":0,"id":"call-...","type":"function","function":{"name":"calculator","arguments":"{\"expression\":\"42 * 17\"}"}}}
```

That event tells the harness, in one shot, that the tool call is ready to execute. No harness-side delta assembly, no guessing whether the arguments are complete, and no custom parser living inside the harness. This makes Dynamo more easily compatible with custom harnesses.

![Tool dispatch timing on a single turn. Standard servers surface the tool call only after the entire stream finishes; Dynamo emits a typed tool_call_dispatch event the moment the call is parsed, so the tool can run in parallel with the rest of the stream. Δt is the time saved per tool call.](./images/fig-3-streaming-dispatch-timeline.svg)

## Anthropic API Fidelity for Claude Code and OpenClaw

Claude Code and OpenClaw both exercise the Anthropic Messages API rather than only text generation behind an endpoint. Matching the harness experience depends on a collection of smaller behaviors that are easy to miss in ad hoc testing:

- model metadata at both `GET /v1/models` and `GET /v1/models/{model_id}`
- correct handling of slashed model IDs
- useful `input_tokens` in `message_start`
- acceptance of `cache_control`

Once the frontend is reachable and compliant, both harnesses can point at Dynamo's Anthropic-compatible endpoint:

```bash
ANTHROPIC_API_KEY=local-dev-token \
ANTHROPIC_BASE_URL=http://localhost:8000 \
ANTHROPIC_CUSTOM_MODEL_OPTION=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="Dynamo NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4" \
claude --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4

ANTHROPIC_API_KEY=local-dev-token \
ANTHROPIC_BASE_URL=http://localhost:8000 \
npx openclaw agent --local -m "Say ok" --json
```

The fixes in this area brought the custom deployment closer to the native backend behavior. One concrete example shows the flavor of these bugs better than a long checklist. During startup, the harness asks for details about the selected model directly, but Dynamo did not yet serve that endpoint:

```text
GET /v1/models/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
HTTP/1.1 404 Not Found
```

Another example is `message_start` reporting `input_tokens: 0` even when the final response later contains the real count. This can make the token count in the harness temporarily drop to `0` every time a new turn starts. [PR #7234](https://github.com/ai-dynamo/dynamo/pull/7234) fixed that Anthropic path by populating `input_tokens` before the stream begins. Those counts are also control-plane data for long sessions: harnesses use context length to decide when to compact the conversation before the next request would exceed the model window. The broader tokenizer-service work landed separately in [PR #7699](https://github.com/ai-dynamo/dynamo/pull/7699), which added `/v1/tokenize` and `/v1/detokenize` endpoints for accurate token counts before a request is processed by the engine.

## Responses and Codex Fidelity

The Codex-facing version of the same problem lives on the `v1/responses` side. Passing compliance tests is not enough to provide parity in user experience.
We found that a Responses API request could not survive an internal round-trip without losing the fields that made it a Responses request rather than a chat completions request. Preserving those fields required architectural changes in Dynamo's `ResponseParams` path, together with the upstream type-alignment work in [PR #6089](https://github.com/ai-dynamo/dynamo/pull/6089).

Codex should point at Dynamo through the OpenAI-compatible Responses API with request compression enabled:

```bash
OPENAI_API_KEY=local-dev-token \
codex exec \
  -c 'openai_base_url="http://localhost:8000/v1"' \
  -c 'features.enable_request_compression=true' \
  -m nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  "Say ok"
```

### Codex Model Metadata Shapes the Request

Codex parity begins before the first `POST /v1/responses`. The CLI resolves the configured model string into a local model-catalog record, and the resulting `ModelInfo` controls the harness state built around the model: base instructions, history formatting, tool registry, reasoning parameters, verbosity controls, image support, context accounting, tool-output truncation, `parallel_tool_calls`, and the final Responses payload.

Two endpoints can serve the same underlying model and still drive different agent behavior if Codex attaches different catalog metadata. The request may validate against the schema while the harness around it has changed.

Tool-output truncation is a useful example. Codex does not replay unlimited command output into the next model turn. Shell and tool observations are truncated according to the selected model's catalog policy before they re-enter context. In the catalog snapshot we tested, `gpt-5.5` used:

```json
{ "mode": "tokens", "limit": 10000 }
```

By contrast, `openai/openai/gpt-5.5` on a custom endpoint used fallback metadata:

```json
{ "mode": "bytes", "limit": 10000 }
```

Those budgets are not equivalent. A `10,000`-byte limit cuts off structured logs, tracebacks, JSON, or test output much earlier than a `10,000`-token limit for ASCII-heavy coding output. For a coding agent, that changes what the model can inspect after a failed test, a search command, or a compiler error. The model may need additional tool calls to recover context that the intended catalog profile would have preserved.

Reasoning settings are also catalog-derived. Codex sends a Responses `reasoning` object when the selected model metadata says reasoning summaries are supported. In that path, Codex also requests `reasoning.encrypted_content` so reasoning state can be replayed across turns. Fallback metadata removes that path.

Prompting changes too. In Codex, switching from the fallback/default profile to the `gpt-5.5` catalog profile changes the system prompt. The fallback prompt is organized around generic Codex operation (`# How you work`, `# AGENTS.md spec`, `# Tool Guidelines`) and emphasizes `AGENTS.md` precedence, planning, validation, and shell-search habits. The `gpt-5.5` prompt is a different instruction document (`# Personality`, `# General`, `# Working with the user`) that frames the agent as a pragmatic software engineer and adds stronger guidance on codebase reading, local-pattern reuse, scoped edits, dirty worktrees, `apply_patch`, collaboration updates, and final-answer formatting. Catalog aliasing therefore affects base behavioral policy as well as request fields such as truncation and reasoning.

We saw this directly in a 50-task subset of SWE-Bench Verified. In this setup, both routes reached OpenAI-served GPT-5.5; the difference was the endpoint and the model-catalog record Codex attached to it. When the custom endpoint used model ID `openai/openai/gpt-5.5` without being associated with the `gpt-5.5` catalog profile, Codex used generic fallback behavior. In one run, the fallback profile issued roughly half as many tool calls:

| Catalog profile | Total tool calls | Per task |
|-----------------|------------------|----------|
| `gpt-5.5` profile | 2,087 | 41.7 |
| Fallback profile | 1,048 | 21.0 |
| Delta | -1,039 | -20.8 |

The paired comparison pointed in the same direction on every task: the `gpt-5.5` profile used more tools in `50 / 50` tasks, while the fallback profile used more tools in `0 / 50`. A permutation test put the difference below `p < 0.001`.

After adding a model-catalog alias so `openai/openai/gpt-5.5` inherited the intended `gpt-5.5` profile, the same 50-task setup became much closer:

| Catalog profile | Total tool calls | Per task |
|-----------------|------------------|----------|
| `gpt-5.5` profile | 2,081 | 41.6 |
| Alias-backed custom profile | 2,205 | 44.1 |
| Delta | +124 | +2.5 |

The remaining difference was not statistically significant in this run: the permutation test was about `p = 0.22`, and the paired directions were mixed (`20 / 50` tasks favored the native profile, `28 / 50` favored the alias-backed profile, and `2 / 50` tied).

For Dynamo, the implication is that Codex compatibility needs to be evaluated at the catalog and request-shaping layer as well as the HTTP schema layer. If Codex cannot resolve a model ID into the intended profile, fallback defaults may change truncation, search-tool availability, verbosity controls, reasoning-summary support, and parallel tool-call support before Dynamo receives the request.

## What's Next

Dynamo now has `nvext.agent_hints`: `latency_sensitivity`, `priority`, `osl`, and `speculative_prefill`. Those fields give the harness a way to say more about the turn than the prompt alone. A session waiting on a user reply is not the same as one working through a long background tool sequence, and the API can now carry some of that difference.

In the v1.1.0 line, Dynamo is also making more of the agent stack available as reusable pieces. The protocol, parser, and tokenizer layers are versioned as standalone crates, including `dynamo-protocols`, `dynamo-parsers`, and `dynamo-tokenizers`. That gives teams a way to build or customize a harness-facing serving path without copying Dynamo internals into a separate project.

This is also the bridge to longer-running systems such as AutoResearch. The first post explained why agentic workloads stress the serving stack. This post shows the harness-facing contract needed to run those workloads correctly and sets the stage for efficient long-running agents backed by Dynamo endpoints.
