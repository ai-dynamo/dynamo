# Full-Stack Optimizations for Agentic Harnesses with Dynamo

Claude Code, OpenClaw, and Codex all depend on details that live above raw token generation: model metadata, and tool-call readiness. Get those details wrong and the failure mode is not just uglier JSON. It is broken cache reuse, idle tool loops, and clients that behave as if the model is slower or less reliable than it really is.

Our [first post](./agentic-inference.md) focused on the architecture underneath agentic inference: the frontend, the router, and KV cache management. This post stays closer to the harness boundary. The question here is simpler and more practical: what had to change in Dynamo to make real agent harnesses like Claude Code, OpenClaw, and Codex feel correct, cache-efficient, and fast?

Claude Code is the main anchor throughout. It puts pressure on nearly every layer at once: a large reusable system prompt, Anthropic-flavored API expectations, interleaved reasoning and tool calls, and long-running sessions with complex event-based patterns such as those created by background commands and agent teams. OpenClaw broadens the story to long-lived and background loops while Codex gives us the `v1/responses` side of the same problem.

## Tiny Setup

The following minimal example helps make the integration concrete.
For Claude Code, we used a simple SSH tunnel to a frontend that exposes the Anthropic-compatible API:

```bash
autossh -M 0 -f -N \
  -L 8000:localhost:8000 \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -o StrictHostKeyChecking=no \
  <gpu-node> && \
ANTHROPIC_BASE_URL=http://localhost:8000 \
ANTHROPIC_CUSTOM_MODEL_OPTION=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="Dynamo NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4" \
claude --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
```

For OpenClaw, the Anthropic-compatible path is even thinner:

```bash
autossh -M 0 -f -N \
  -L 8000:localhost:8000 \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -o StrictHostKeyChecking=no \
  <gpu-node> && \
ANTHROPIC_BASE_URL=http://localhost:8000 \
npx openclaw
```

All experiments in this post were ran against `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` on a single B200 in aggregated serving mode. Some of the strongest results below are correctness results with clear systems implications; others are quantitative.

## The Dynamo Graph Deployment (DGD) Settings

We started the frontend with the Anthropic API compatibility flags shown below:

```bash
python -m dynamo.frontend \
  --http-port 8000 \
  --enable-anthropic-api \
  --strip-anthropic-preamble
```
If the frontend and worker are missing these harness-facing switches, the deployment can benchmark well in terms of metrics and still feel wrong and clunky. 

On the frontend side, the key settings to watch for are:

- `--enable-anthropic-api` so Claude Code and OpenClaw can talk to Dynamo over the Antrhopic API the default messages API results in a degraded experience and is best avoided.
- `DYN_STRIP_ANTHROPIC_PREAMBLE=1` so Claude Code's billing header does not destroy prefix stability.
- `DYN_ENABLE_STREAMING_TOOL_DISPATCH=1` so tool readiness is emitted as structured stream state rather than inferred from deltas. This means that decoded tool calls can start executing as soon as they are decoded, rather than waiting till the end of turn.

On the worker side, the important settings in this deployment are:

- `--dyn-tool-call-parser <parser>` and `--dyn-reasoning-parser <parser>` so tool calls and reasoning blocks are reconstructed in the model-specific format the harness actually needs. For instance, a common pattern is to drop all reasoning tokens from previous turns. Parsers need to handle this correctly, as well as the case of not dropping reasoning tokens for turns with interleaved tool calls, where the reasoning provides crucial context.


## Prompt Stability Is Key for Cache Re-use

Claude Code sends thousands of tokens of reusable prompt scaffolding, much of which is designed to be the same acorss different users and sessions. However, at the very front of each prompt is a session-specific billing header which causes cache misses when pointed at custom endpoints that do not strip it out:

```text
x-anthropic-billing-header: cc_version=0.2.93; cch=abc123def456==;
You are Claude Code, an interactive CLI tool...
```

These headers poison the KV cache and prevent it from being reused, even across sessions by the same user. A varying line at position zero means every new session starts from a different token prefix, so the stable instructions and tool definitions behind it never line up cleanly for reuse.

That is why Dynamo added `--strip-anthropic-preamble`. The fix is mechanically small and operationally important: remove the unstable billing header before tokenization so that the stable prompt starts at token zero.

The impact of this simple fix is significant. On a Dynamo B200 deploy run with a 52K-token prompt, a stable prefix landed at `168ms` TTFT. Keeping a varying per-session header in the prefix pushed that to `912ms`. Removing the billing header before tokenization brought it back to `169ms`. On this workload, the unstable header costs `744ms` per request and turns a reusable system prompt into a cold prefill. That is a redeuction of 5x in TTFT that would be experienced directly by new users hitting the same deployment or the same user opening up a new session.

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

## The Nuances of Reasoning Parsing

Reasoning replay does not have one universal correct form. Some models intentionally drop prior thinking on ordinary assistant turns. Agentic turns with interleaved tool calls are different: there, the reasoning and tool sequence often needs to survive the round-trip in the same order. The real contract is model-specific and turn-specific.

Contemporary reasoning models tend to produce two different kinds of assistant turns:
- reasoning followed by a direct response to the user
- reasoning followed by one or more tool calls

Agentic models are especially good at producing turns where many reasoning and tool-call segments appear within a single response. In those cases, the interleaving matters:

```text
<think>reasoning_0</think> tool_call_0 <think>reasoning_1</think> tool_call_1
```

That structure needs to remain intact on replay when the model is expected to remember why each tool was called. Dynamo now supports this interleaved format. Previously, the same turn could be reconstructed as:

```text
<think>reasoning_0 reasoning_1</think> tool_call_0 tool_call_1
```

That older shape came from legacy models that emitted only a single reasoning span and a single tool-call pass per turn.

We also found a second class of bug: reasoning was often being dropped too aggressively on replay. For some models, dropping prior thinking on turns without tool calls is an established behavior. DeepSeek-R1 is the clearest example. But that same behavior is wrong for interleaved agentic turns where the prior reasoning explains the tool sequence. This was difficult to spot because users could see reasoning being decoded correctly in the outgoing response while it was still being silently malformed or dropped before the next turn.

This round-trip was broken until [PR #7358](https://github.com/ai-dynamo/dynamo/pull/7358). The bug had three layers:

1. **Double parsing**: the Anthropic streaming handler applied a second reasoning parser on top of the engine stream, which already had reasoning correctly split. The second parser re-classified all content as reasoning.

2. **Silent drop**: chat templates only reference `{{ message.content }}` — they ignore `reasoning_content`. Without explicit injection, the model never saw its own prior chain-of-thought. The fix injects `reasoning_content` back into `content` as `<think>` blocks before template rendering, on both the Rust preprocessor path (`ModelInput::Tokens`) and the Python worker path (`ModelInput::Text`). Templates that natively handle `reasoning_content` (Nemotron, Qwen3) are detected at load time and left alone.

3. **Template truncation**: Nemotron's chat template defaults `truncate_history_thinking` to `true`, which strips `<think>` content from all assistant turns before the last user message. That is reasonable for ordinary chat, but wrong for agentic turns where prior reasoning explains why tools were called. Some parsers made this even harder to notice. For example, the DeepSeek-R1 parser preserves reasoning on tool-calling turns but drops it on turns without tool calls, which is exactly the wrong shape for a harness that needs the model to remember why those tool calls happened. NVIDIA's own SWE training pipeline sets `truncate_history_thinking: false`; the Anthropic handler now passes that flag automatically when a reasoning parser is configured.

Once the parser and template path were fixed, the expected behavior finally appeared. In our B200 experiment with a 52K-token system prompt and an assistant turn containing about 500 tokens of thinking, exact replay landed at `167ms` TTFT while mutated thinking landed at `322ms`. That is a `1.9x` increase, or about `155ms` per request, from changing the reasoning content inside the replayed prefix. The point of that measurement is not that every model should preserve all prior reasoning. It is that, in the cases where reasoning remains part of the replayed prefix, mutating it has a measurable cost.

The key takeaway is that the harness, parser, and template path must agree on each model's expected reasoning behavior. Dropping thinking on ordinary turns may be correct for one model and wrong for another. Preserving interleaved reasoning on tool-calling turns may be essential even when ordinary turns are allowed to strip it. In practice, you should not assume that the tokens produced on turn `N` will automatically arrive unchanged as the prefix of turn `N+1`. Whether that is true depends on the reasoning parser, tool parser, and chat template for the model you are serving.

## Streaming Actionable State

Streaming tokens is not enough for harnesses. If the client only learns that a tool call is ready at stream end, the harness stays artificially blind even when the model has already made the decision.


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

That event tells the harness, in one shot, that the tool call is ready to execute. No client-side delta assembly, no guessing whether the arguments are complete, and no custom parser living inside the harness.

The timing result needs to be stated carefully. In our B200 runs, dispatch did not create a large end-to-end wall-time win by itself. The dispatch event fires at essentially the same moment as the inline tool delta. What changed is that the server now exposes tool readiness as a typed protocol event instead of forcing the client to reconstruct it from partial deltas.

It is still a real systems improvement. A tool call is a state transition, not just another substring in the stream. In the old buffered path, the harness learned about that transition only at stream end. In the current path, it learns about it as soon as the tool call is structurally complete. With dispatch enabled, it learns the same fact in a cleaner form: parsed, typed, and ready to act on.

The multi-turn measurements make the right claim narrower, not weaker. On the 30-city workload, the harness learned about each tool call about `9-10ms` before `finish_reason`, and the cumulative earlier feedback over many turns was real but modest. That is not a dramatic latency chart. It is a protocol improvement that removes blind buffering and gives the harness actionable state immediately.

The stream now carries the state transitions an agent harness actually needs, and that `tool_call_dispatch` turns those transitions into something a client can consume without maintaining its own parser.

## Anthropic and Claude Code API Fidelity

Claude Code compatibility is more than text generation behind an Anthropic-shaped endpoint. The harness depends on a collection of smaller behaviors that are easy to miss in ad hoc testing:

- model metadata at both `GET /v1/models` and `GET /v1/models/{model_id}`
- correct handling of slashed model IDs
- useful `input_tokens` in `message_start`
- proper thinking blocks
- acceptance of `cache_control`
- response shapes that track the Anthropic API closely enough for clients not to trip over them

The fixes in this area were not glamorous, but they mattered:

| Surface | Why the harness cares | Failure mode |
|---------|-----------------------|--------------|
| `GET /v1/models/{model_id}` | resolves the connected model before use | slashed IDs 404 |
| `message_start.input_tokens` | does context accounting while the stream is live | early token count stays `0` |
| thinking blocks | reconstructs typed assistant state | reasoning is flattened or dropped |
| `cache_control` | preserves Anthropic prompt-pinning semantics | accepted shape diverges from expected behavior |

This is a good example of harness compatibility being more than "the field exists somewhere." Retrieval path, identifier handling, response shape, and timing all matter. A backend can be broadly Anthropic-flavored and still be just off enough to make a harness feel brittle.

One concrete example shows the flavor of these bugs better than a long checklist. Claude Code asks for the connected model directly. On the measured deployment, that lookup failed:

```text
GET /v1/models/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
HTTP/1.1 404 Not Found
```

That is not the kind of bug that makes a conference talk, but it is exactly the kind that makes a backend feel brittle to a harness. The same section of captures shows a second class of issue: `message_start` reports `input_tokens: 0` even when the final response later contains the real count. Those details sit well below "generate the next token," but they are still part of whether the harness can reason correctly about the backend.

## Responses and Codex Fidelity

The Codex-facing version of the same problem lives on the `v1/responses` side. Passing compliance tests is not enough if realistic replay and field preservation are lossy.

The clean architectural idea here is Dynamo's `ResponseParams` path. Instead of letting a Responses request collapse into chat completions and then trying to reconstruct the missing pieces afterward, Dynamo extracts the client-facing response parameters up front, preserves them through the internal conversion, and merges them back into the final response object.

That turns the internal conversion path from an hourglass into a controlled translation layer. Fields that the engine does not care about, such as `instructions`, `store`, `truncation`, or input-item metadata, do not silently vanish just because the internal runtime speaks a chat-completions-shaped dialect.

The easiest way to see the difference is to look at the request path itself:

```text
Client Request (Responses API)
│
├─► ResponseParams --------------------------► Response echo
│   (model, tools, instructions, etc.)         (preserved verbatim)
│
└─► Internal conversion
    │
    └─► ChatCompletions-shaped request
        │
        │ Fields that would otherwise get lost here:
        │ - input item `id` and `status`
        │ - `previous_response_id`
        │ - `store`, `truncation`, `service_tier`
        │ - original `reasoning` and tool config shape
        │
        └─► Engine output
            │
            └─► Merge engine output + ResponseParams
                │
                └─► Final Responses object
```

Codex surfaced a different failure mode than Claude Code. The issue was not whether Dynamo could generate the next token. It was whether a realistic Responses request could survive an internal round-trip without losing the fields that made it a Responses request in the first place. Preserving those fields turned out to be an architectural concern, not just a serializer concern. On this side of the stack, the main failure mode is quiet loss: the server answers, but the harness loses the metadata it was using to manage the conversation.

The simple text-generation capture is a useful example because it shows both preservation and the remaining gaps in one response. A request as small as:

```json
{
  "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
  "input": "Say hello briefly.",
  "max_output_tokens": 50
}
```

comes back with the Responses-shaped structure intact: a reasoning item, a message item, the original `max_output_tokens`, and the usual top-level fields. At the same time, it shows an important remaining edge: `output_tokens_details.reasoning_tokens` is `0` even though reasoning content is clearly present in the response. That is exactly the kind of detail a Codex-style client notices.

## OpenClaw: Integration Testing and Parser Discovery

Claude Code stress-tests the system prompt and single-session tool loop. OpenClaw stress-tests what happens when sessions stay alive across channels, turns accumulate over hours, and the client is a background loop rather than an interactive terminal. But the most useful thing OpenClaw gave us was a structured integration test for the hardest parsing problem in the stack: combined reasoning and tool calls in a single response.

OpenClaw is a multi-channel AI chat client that connects to the same backend over Telegram, WhatsApp, and web chat simultaneously. Unlike Claude Code, which starts a fresh session per task and sends a large system prompt once, OpenClaw keeps conversations alive indefinitely. That makes it a good vehicle for testing whether the full parsing pipeline holds up under realistic multi-turn conditions with tools and thinking interleaved.

We ran experiments against a Dynamo + TRT-LLM deployment: Nemotron-3-Super-120B-A12B-NVFP4 on 4x B200 with TP=4, with `--enable-anthropic-api`, `--strip-anthropic-preamble`, `--enable-streaming-tool-dispatch`, the `nemotron_deci` reasoning parser, and the `qwen3_coder` tool call parser.

### Parser Names Map to Output Format, Not Model Family

The first thing we had to learn was that parser selection is about the token format the model emits, not the model's name or family.

Nemotron-3-Super ships with a reasoning format that uses `<think>` tags and a tool call format that uses `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>` XML. The reasoning parser that handles `<think>` tags is called `nemotron_deci`. That name makes sense. But the tool call parser that handles the XML format is called `qwen3_coder`, because Qwen models popularized that output shape. An earlier attempt used `--dyn-tool-call-parser nemotron_deci` for both, which looks for `<TOOLCALL>` tags that Nemotron-3-Super never emits. The model was generating perfectly valid tool calls; the wrong parser was silently discarding them.

The naming is confusing because it conflates two different things. Parser names describe the output format they consume, not the model family they were designed for. Once you know that, the correct configuration falls out: `--dyn-reasoning-parser nemotron_deci` (for `<think>` tags) and `--dyn-tool-call-parser qwen3_coder` (for the XML tool call format). Getting one wrong does not produce an error. It produces silence where structured output should be.

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

Across all five non-streaming rounds: `5/5` produced both a `thinking` block and a `tool_use` block, `5/5` had them in the correct order (thinking before tool_use), `5/5` returned `stop_reason: "tool_use"`, and all five correctly parsed the tool name (`calculator`) and input (`{"expression": "15 * 23"}`). Mean latency was `1,187ms`, with the first request at `2,216ms` (cold) and subsequent requests settling to `771-1,097ms`.

The thinking content itself shows the model doing real chain-of-thought: decomposing the multiplication into `15 * 20 = 300` and `15 * 3 = 45`, arriving at `345`, then deciding to call the calculator to verify. That reasoning appears as a structured `thinking` block, not as inline text that a client would have to parse.

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

Across five streaming rounds: `5/5` had both thinking and tool_use blocks in the correct order. Mean time to first thinking token was `83ms`. Mean time to tool_use block was `919ms`. Mean total stream time was `933ms`. The gap between the thinking block starting and the tool_use block arriving (mean `836ms`) is the model's generation time for the reasoning content plus the tool call tokens.

One notable observation: the `tool_call_dispatch` event did not fire in any of the five streaming rounds (`0/5`). The `--enable-streaming-tool-dispatch` flag was set on the frontend, but dispatch events appear to require a different interaction between the tool call parser and the streaming path than what the current TRT-LLM backend produces. The tool_use block still arrives correctly as a `content_block_start` event, which is what clients like OpenClaw consume. Dispatch is an optimization for earlier notification; its absence does not affect correctness.

### Cache Stability Across Turns

A separate experiment measured TTFT across 8-turn multi-turn conversations under three conditions: a stable system prompt, a varying prompt (random preamble injected each turn), and a billing-header-style prefix that Dynamo strips before tokenization. Three rounds of each condition.

On TRT-LLM with B200, all three conditions converged to the same TTFT band. Stable-prefix turns averaged `118ms`, varying-prefix turns averaged `112ms`, and stripped-prefix turns averaged `116ms`. The differences are within noise. After the first turn or two of each session (which ran slightly higher at `120-136ms` as the cache warmed), subsequent turns settled into a `101-116ms` band regardless of condition.

The story here is simpler than the Claude Code result but still useful. OpenClaw's shorter, static system prompt does not benefit as dramatically from preamble stripping because there is less prefix to invalidate. The billing header that costs `744ms` on a 52K-token Claude Code prompt has a proportionally smaller impact on a 2K-token OpenClaw prompt. But the invariant holds: stable prefixes stay in the fast path, and Dynamo's stripping keeps the stripped condition indistinguishable from stable.

### What OpenClaw Adds to the Story

The Claude Code experiments established that prefix stability, replay fidelity, and stream semantics matter for agentic serving. OpenClaw added two things.

First, a concrete integration test for the hardest parsing case: two different parsers (`nemotron_deci` for reasoning, `qwen3_coder` for tool calls) splitting one token stream into correctly ordered Anthropic content blocks. That test passed `10/10` across both streaming and non-streaming paths, with correct block types, correct ordering, correct stop reasons, and correctly parsed tool arguments.

Second, a deployment lesson that is easy to miss in documentation. Parser names in Dynamo describe the output format they consume, not the model they were designed for. A model called Nemotron uses a parser called `qwen3_coder` for its tool calls because the output format matches, not because of any Qwen lineage. When a model appears to reason about tools inline rather than emitting structured tool-call blocks, the first thing to check is whether the configured parser matches the model's actual output format.

## What To Track Now

There are four active fronts where the harness-facing surface is expanding and where future benchmarks will get more interesting.

| Front | What changed | Why rerun it |
|-------|--------------|--------------|
| `nvext.agent_hints` | Dynamo now documents `latency_sensitivity`, `priority`, `osl`, and `speculative_prefill` as per-request routing hints | tests whether the harness can steer scheduling rather than only consume it |
| parser and template coverage | broader parser matrix plus `chat_template.json` prompt formatter support | parser selection errors will show up on more models and more backends |
| tool-choice edge handling | recent fixes for `tool_choice=required` bypassing format-specific parsers | catches the cases that only appear once a real harness leans on the API contract |
| tracing and request IDs | request spans and request-ID propagation across the stack | makes fidelity failures attributable instead of anecdotal |

## Closing the Loop

These subtle changes to the harness integration, ensure that the architecture improvements from the first post pay off. 

Dynamo now ships several of these layers as standalone Rust crates, including `dynamo-protocols`, `dynamo-parsers`, `dynamo-llm`, and `dynamo-tokens`. If you want to build a custom serving pipeline rather than deploy the full stack, you can pull in just the pieces you need. Those crates carry the same harness-facing correctness concerns described above without requiring a full Dynamo deployment.
