# Streaming Tokens and Tools: Multi-Turn Agentic Harness Support in Dynamo

A turn in an agentic conversation is more than just a prompt followed by a completion. Inference engines are expected to produce correctly segmented API results that are directly consumable by the attached agentic harness i.e. the parsing of tool calls and reasoning falls on the engine and not the harness. Additionally, inference endpoints must stream back partial results, not just in terms of tokens, but also during tool calling. This post covers what broke when real agentic clients running against Dynamo, how we hardened our parsers and API coverage, and how the parsers have been abstracted into independent crates to make them reusable.

Our work here builds upon the performance considerations outline in our [first post](./agentic-inference.md) which focused on the architecture underneath agentic inference: the frontend, the router, and KV cache management. The focus of our fixes in this follow up blog was threefold: correctness, user-experience equivalence, as well as performance.

New agentic harnesses are constantly coming onto the scene and existing ones evolving, maturing and adding more elaborate features. So the focus is on the core features of popular harnesses such as Claude Code and Codex and OpenClaw, in the hopes that any learnings will be useful and transferrable.


## Harness-Facing Dynamo Settings

Our experiments used the newly released `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` model, thought they are applicable across different models, reasoning and tool calling parsers.

To reproduce our results, the frontend must be configured with newly the exposed Anthropic-compatible API and enabled with the switches that preserve prompt, reasoning, and tool state:
- `--enable-anthropic-api` so harnesses can talk to Dynamo over the Anthropic API. Most harnesses can fall back to the default Messages API but that leads to a degraded experience.
- `--strip-anthropic-preamble` this looks for, and strps the Anthropic billing header that can cause kv-reuse instability.
- `--enable-streaming-tool-dispatch` this *new* feature means tool calls can start executing as soon as they are decoded, rather than waiting till the end of turn.

Putting all of this together:
```bash
python -m dynamo.frontend \
  --http-port 8000 \
  --enable-anthropic-api \
  --strip-anthropic-preamble \
  --enable-streaming-tool-dispatch
```

On the worker side, the important settings in this deployment are:
- `--dyn-tool-call-parser <parser>` and `--dyn-reasoning-parser <parser>` so tool calls and reasoning blocks are reconstructed in the model-specific format the harness actually needs. For instance, a common pattern is to drop all reasoning tokens from previous turns.

## Prompt Stability Is Key for Cache Reuse

Claude Code sends thousands of tokens of reusable prompt scaffolding, much of which is designed to be the same across different users and sessions. However, at the very front of each prompt is a session-specific billing header which causes cache misses when pointed at custom endpoints that do not strip it out:

```text
x-anthropic-billing-header: cc_version=0.2.93; cch=abc123def456==;
You are Claude Code, an interactive CLI tool...
```

These headers poison the KV cache and prevent it from being reused, even across sessions by the same user. A varying line at position zero means every new session starts from a different token prefix, so the stable instructions and tool definitions behind it never line up cleanly for reuse.

TO restore proper kv-cache hits, Dynamo added `--strip-anthropic-preamble`. The fix is mechanically small and operationally important: remove the unstable billing header before tokenization so that the stable prompt starts at token zero.

The measured impact was large. On a Dynamo B200 deployment with a 52K-token prompt, a stable prefix landed at `168ms` TTFT. Keeping a varying per-session header in the prefix pushed that to `912ms`. Removing the billing header before tokenization brought it back to `169ms`. On this workload, the unstable header costs `744ms` per request and turns a reusable system prompt into a cold prefill. That is about a `5x` reduction in TTFT for new users hitting the same deployment or for the same user opening a new session.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 380" style="max-width:720px;width:100%;font-family:system-ui,-apple-system,sans-serif;background:#fafafa;border:1px solid #e5e7eb;border-radius:8px">
<line x1="72" y1="324.0" x2="696" y2="324.0" stroke="#e5e7eb" stroke-width="1"/>
<text x="64" y="328.0" text-anchor="end" fill="#6b7280" font-size="11">0</text>
<line x1="72" y1="268.4" x2="696" y2="268.4" stroke="#e5e7eb" stroke-width="1"/>
<text x="64" y="272.4" text-anchor="end" fill="#6b7280" font-size="11">200</text>
<line x1="72" y1="212.8" x2="696" y2="212.8" stroke="#e5e7eb" stroke-width="1"/>
<text x="64" y="216.8" text-anchor="end" fill="#6b7280" font-size="11">400</text>
<line x1="72" y1="157.1" x2="696" y2="157.1" stroke="#e5e7eb" stroke-width="1"/>
<text x="64" y="161.1" text-anchor="end" fill="#6b7280" font-size="11">600</text>
<line x1="72" y1="101.5" x2="696" y2="101.5" stroke="#e5e7eb" stroke-width="1"/>
<text x="64" y="105.5" text-anchor="end" fill="#6b7280" font-size="11">800</text>
<line x1="72" y1="45.9" x2="696" y2="45.9" stroke="#e5e7eb" stroke-width="1"/>
<text x="64" y="49.9" text-anchor="end" fill="#6b7280" font-size="11">1000</text>
<text x="16" y="178.0" text-anchor="middle" fill="#374151" font-size="12" font-weight="500" transform="rotate(-90,16,178.0)">TTFT (ms)</text>
<text x="384.0" y="372" text-anchor="middle" fill="#374151" font-size="12" font-weight="500">Request (3 rounds × 15 requests)</text>
<line x1="72" y1="277.2" x2="696" y2="277.2" stroke="#3b82f6" stroke-width="1" stroke-dasharray="6,4" opacity="0.5"/>
<line x1="72" y1="70.5" x2="696" y2="70.5" stroke="#ef4444" stroke-width="1" stroke-dasharray="6,4" opacity="0.5"/>
<line x1="72" y1="277.0" x2="696" y2="277.0" stroke="#22c55e" stroke-width="1" stroke-dasharray="6,4" opacity="0.5"/>
<text x="698" y="74.5" fill="#ef4444" font-size="10" font-weight="600">911ms</text>
<text x="698" y="271.2" fill="#3b82f6" font-size="10" font-weight="600">168ms</text>
<text x="698" y="291.0" fill="#22c55e" font-size="10" font-weight="600">169ms</text>
<circle cx="78.9" cy="69.0" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="78.9" cy="275.3" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="78.9" cy="275.6" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="92.8" cy="71.4" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="92.8" cy="277.1" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="92.8" cy="277.8" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="106.7" cy="68.8" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="106.7" cy="276.2" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="106.7" cy="278.1" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="120.5" cy="71.7" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="120.5" cy="278.0" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="120.5" cy="275.6" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="134.4" cy="69.1" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="134.4" cy="275.9" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="134.4" cy="275.5" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="148.3" cy="71.1" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="148.3" cy="277.9" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="148.3" cy="276.3" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="162.1" cy="69.2" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="162.1" cy="278.2" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="162.1" cy="278.5" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="176.0" cy="71.5" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="176.0" cy="275.0" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="176.0" cy="275.9" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="189.9" cy="69.4" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="189.9" cy="278.4" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="189.9" cy="275.1" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="203.7" cy="71.4" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="203.7" cy="275.7" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="203.7" cy="278.7" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="217.6" cy="69.4" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="217.6" cy="278.3" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="217.6" cy="275.5" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="231.5" cy="71.8" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="231.5" cy="276.1" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="231.5" cy="275.5" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="245.3" cy="72.0" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="245.3" cy="278.0" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="245.3" cy="278.1" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="259.2" cy="68.6" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="259.2" cy="275.5" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="259.2" cy="275.9" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="273.1" cy="71.7" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="273.1" cy="278.3" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="273.1" cy="278.5" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="286.9" cy="69.0" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="286.9" cy="278.1" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="286.9" cy="277.9" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="300.8" cy="71.7" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="300.8" cy="278.2" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="300.8" cy="275.8" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="314.7" cy="72.2" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="314.7" cy="275.8" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="314.7" cy="278.4" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="328.5" cy="68.5" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="328.5" cy="278.6" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="328.5" cy="276.0" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="342.4" cy="71.9" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="342.4" cy="276.0" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="342.4" cy="278.4" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="356.3" cy="69.5" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="356.3" cy="278.4" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="356.3" cy="275.9" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="370.1" cy="71.7" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="370.1" cy="276.0" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="370.1" cy="278.6" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="384.0" cy="69.3" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="384.0" cy="278.6" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="384.0" cy="275.7" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="397.9" cy="72.2" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="397.9" cy="275.8" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="397.9" cy="277.8" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="411.7" cy="71.6" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="411.7" cy="278.8" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="411.7" cy="276.1" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="425.6" cy="68.1" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="425.6" cy="276.1" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="425.6" cy="277.8" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="439.5" cy="71.3" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="439.5" cy="278.7" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="439.5" cy="275.5" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="453.3" cy="68.7" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="453.3" cy="276.3" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="453.3" cy="278.6" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="467.2" cy="72.0" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="467.2" cy="274.7" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="467.2" cy="275.6" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="481.1" cy="69.2" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="481.1" cy="278.4" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="481.1" cy="278.3" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="494.9" cy="69.1" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="494.9" cy="278.4" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="494.9" cy="275.6" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="508.8" cy="72.2" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="508.8" cy="276.2" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="508.8" cy="278.4" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="522.7" cy="69.2" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="522.7" cy="278.7" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="522.7" cy="275.8" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="536.5" cy="71.4" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="536.5" cy="278.7" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="536.5" cy="278.6" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="550.4" cy="69.5" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="550.4" cy="275.8" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="550.4" cy="275.8" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="564.3" cy="72.3" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="564.3" cy="278.7" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="564.3" cy="278.8" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="578.1" cy="69.4" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="578.1" cy="276.0" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="578.1" cy="278.8" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="592.0" cy="71.9" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="592.0" cy="278.4" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="592.0" cy="275.6" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="605.9" cy="72.2" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="605.9" cy="276.0" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="605.9" cy="278.8" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="619.7" cy="68.3" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="619.7" cy="278.6" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="619.7" cy="276.1" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="633.6" cy="72.1" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="633.6" cy="275.9" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="633.6" cy="277.9" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="647.5" cy="69.5" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="647.5" cy="278.6" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="647.5" cy="276.2" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="661.3" cy="72.4" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="661.3" cy="278.9" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="661.3" cy="278.6" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="675.2" cy="69.6" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="675.2" cy="275.6" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="675.2" cy="276.1" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="689.1" cy="72.0" r="3.5" fill="#ef4444" opacity="0.7"/>
<circle cx="689.1" cy="278.8" r="3.5" fill="#3b82f6" opacity="0.7"/>
<circle cx="689.1" cy="278.3" r="3.5" fill="#22c55e" opacity="0.7"/>
<circle cx="80" cy="40" r="5" fill="#3b82f6" opacity="0.8"/>
<text x="90" y="44" fill="#374151" font-size="12">Stable prefix</text>
<circle cx="240" cy="40" r="5" fill="#ef4444" opacity="0.8"/>
<text x="250" y="44" fill="#374151" font-size="12">Varying prefix</text>
<circle cx="400" cy="40" r="5" fill="#22c55e" opacity="0.8"/>
<text x="410" y="44" fill="#374151" font-size="12">Stripped prefix</text>
</svg>

<p style="text-align:center;color:#6b7280;font-size:0.9em;margin-top:0.5em"><em>Prompt stability versus TTFT on a 52K-token prefix. Stable and stripped prefixes land at ~168–169 ms TTFT; a varying prefix jumps to ~911 ms.</em></p>

## The Nuances of Reasoning and Tool Parsing

Carrying reasoning into the next turn does not have one universal correct form. Some models intentionally drop prior thinking on ordinary assistant turns. Agentic turns with interleaved tool calls are different: there, the reasoning and tool sequence often needs to persist to the next turn in the same order. The real contract is model-specific and turn-specific.

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

This round-trip was broken until [PR #7358](https://github.com/ai-dynamo/dynamo/pull/7358). The fix had three parts:

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

## Anthropic API Fidelity

Claude Code compatibility is more than text generation behind an Anthropic endpoint. For matching the user experience, the harness depends on a collection of smaller behaviors that are easy to miss in ad hoc testing:

- model metadata at both `GET /v1/models` and `GET /v1/models/{model_id}`
- correct handling of slashed model IDs
- useful `input_tokens` in `message_start`
- acceptance of `cache_control`

Once the frontend is reachable and compliant, pointing a harness at Dynamo is straightforward and does not have to replace any existing model configuration:

```bash
ANTHROPIC_API_KEY=local-dev-token \
ANTHROPIC_BASE_URL=http://localhost:8000 \
ANTHROPIC_CUSTOM_MODEL_OPTION=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="Dynamo NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4" \
claude --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
```

The fixes in this area brought the custom deployment closer to the native backend behavior.
One concrete example shows the flavor of these bugs better than a long checklist. During startup, the harness asks for details about the selected model directly, but Dynamo did not yet serve that endpoint:

```text
GET /v1/models/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
HTTP/1.1 404 Not Found
```
Another example is `message_start` reporting `input_tokens: 0` even when the final response later contains the real count. This can make the token count in the harness temporarily drop to `0` every time a new turn starts. [PR #7234](https://github.com/ai-dynamo/dynamo/pull/7234) fixed that Anthropic path by populating `input_tokens` before the stream begins. The broader tokenizer-service work landed separately in [PR #7699](https://github.com/ai-dynamo/dynamo/pull/7699), which added `/v1/tokenize` and `/v1/detokenize` endpoints for accurate token counts before a request is processed by the engine.

## Responses and Codex Fidelity

The Codex-facing version of the same problem lives on the `v1/responses` side. Passing compliance tests is not enough to provide parity in user experience.
We found that a Responses API request could not survive an internal round-trip without losing the fields that made it a Responses request rather than a chat completions request. Preserving those fields turned out to be an architectural concern, not just a serializer concern. The relevant changes live in Dynamo's `ResponseParams` path and the upstream type-alignment work in [PR #6089](https://github.com/ai-dynamo/dynamo/pull/6089).

Codex should point at Dynamo through the OpenAI-compatible Responses API:

```bash
OPENAI_API_KEY=local-dev-token codex exec -c 'model_providers.dynamo={ name = "Dynamo", base_url = "http://localhost:8000/v1", wire_api = "responses", env_key = "OPENAI_API_KEY" }' -c model_provider="dynamo" -m nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 "Say ok"
```

## OpenClaw over the Anthropic Messages API

The same Anthropic API work also translates to a smooth OpenClaw experience. OpenClaw does not need a Dynamo-specific adapter; it can use Dynamo's Anthropic-compatible Messages API for thinking blocks, tool-use blocks, and streaming tool-call dispatch. With `--enable-anthropic-api` on the frontend, pointing OpenClaw at Dynamo is a one-line change:

```bash
ANTHROPIC_BASE_URL=http://localhost:8000 ANTHROPIC_API_KEY=local-dev-token npx openclaw
```

We ran experiments against a Dynamo + TRT-LLM deployment: Nemotron-3-Super-120B-A12B-NVFP4 on 4x B200 with TP=4, with `--enable-anthropic-api`, `--strip-anthropic-preamble`, `--enable-streaming-tool-dispatch`, the `nemotron_deci` reasoning parser, and the `qwen3_coder` tool call parser.

### Combined Reasoning and Tool Calls

The hardest parsing test is when both parsers have to operate on the same token stream. A model that reasons before calling a tool generates a response where `<think>` content flows first, followed by `<tool_call>` XML. Two different parsers, `nemotron_deci` for reasoning and `qwen3_coder` for tool calls, have to split that stream into the correct Anthropic Messages API content blocks without interfering with each other.

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

Because OpenClaw makes the event stream easy to inspect, it helped turn a parser concern into a repeatable integration test. The question was no longer "did the model answer?" It was "did every thinking, text, and tool-use boundary arrive in the shape the harness needs?"

## What's Next

Dynamo now has `nvext.agent_hints`: `latency_sensitivity`, `priority`, `osl`, and `speculative_prefill`. Those fields give the harness a way to say more about the turn than the prompt alone. A session waiting on a user reply is not the same as one working through a long background tool sequence, and the API can now carry some of that difference.

In the v1.1.0 line, Dynamo is also making more of the agent stack available as reusable pieces. The protocol, parser, LLM, and tokenizer layers are versioned as standalone crates, including `dynamo-protocols`, `dynamo-parsers`, `dynamo-llm`, and `dynamo-tokens`. That gives teams a way to build or customize a harness-facing serving path without copying Dynamo internals into a separate project.

This is also the bridge to longer-running systems such as AutoResearch. The first post explained why agentic workloads stress the serving stack. This post shows the harness-facing contract needed to run those workloads correctly and sets the stage for efficient long-running agents backed by Dynamo endpoints.
