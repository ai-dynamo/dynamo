# Full-Stack Optimizations for Agentic Harnesses with Dynamo

This post should read as a direct follow-up to the original agentic inference post, but only briefly. That post explained the architecture: frontend, router, and KV cache management. This one should stay closer to the harness boundary and answer a more practical question:

What had to change in Dynamo to make real agent harnesses like Claude Code, OpenClaw, and Codex feel correct, cache-efficient, and fast?

The easy part is pointing a harness at Dynamo. The hard part is preserving the semantics that the harness quietly depends on: prompt shape, replay order, stream structure, metadata lookups, and tool-call readiness. Small mismatches here are not cosmetic. They turn into cache misses, added latency, and client behavior that looks flaky even when the model itself is fine.

Claude Code should be the main narrative anchor. It puts pressure on nearly every layer at once: long reusable system prompts, strict expectations around Anthropic-flavored API behavior, interleaved reasoning and tools, and long-running sessions where small compatibility gaps compound quickly. OpenClaw broadens the story to background loops and long-lived agents. Codex keeps the piece from becoming only an Anthropic post and lets us show the same problem from the `v1/responses` side.

<put link here: original agentic inference post>

## Tiny Setup

Keep this short. The point is not setup; it is to show the shape of the integration and the knobs that mattered in the experiments.

```bash
# Claude Code against Dynamo
export ANTHROPIC_BASE_URL=http://localhost:8000
claude
```

```bash
# Codex / Responses-style client against Dynamo
curl -s http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model>",
    "input": "Summarize the router design briefly."
  }'
```

```bash
# Flags used in the Anthropic-facing experiments
python -m dynamo.frontend \
  --http-port 8000 \
  --enable-anthropic-api \
  --strip-anthropic-preamble \
  --dyn-reasoning-parser nemotron_nas
```

All experiments in the artifact set ran against `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` on a single B200 in aggregated serving mode. That matters. Some of the correctness claims below are stronger than the measured latency claims on this deployment, and the post should say so explicitly.

## Section 1: Prompt Stability Is Cache Work

Claude Code sends a large amount of reusable prompt scaffolding. That is exactly what you want for KV reuse, as long as the prefix remains stable. The catch is that Claude Code also prepends a session-specific billing header near the very front of the system prompt:

```text
x-anthropic-billing-header: cc_version=0.2.93; cch=abc123def456;
You are Claude Code, an interactive CLI tool...
```

For Anthropic's managed prompt caching, this is fine. For prefix-based KV reuse, it is poison. A varying line at position zero means every new session starts from a different token prefix, so the stable instructions and tool definitions behind it never line up cleanly for reuse.

That is why Dynamo added `--strip-anthropic-preamble`. The fix is mechanically small and operationally important: remove the unstable billing header before tokenization so that the stable prompt starts at token zero.

```rust
fn strip_billing_preamble(system: &mut Option<SystemContent>) {
    if let Some(content) = system {
        let trimmed = content.text.trim_start();
        if trimmed.starts_with("x-anthropic-billing-header:")
            && let Some(newline_pos) = trimmed.find('\n')
        {
            content.text = trimmed[newline_pos + 1..].to_string();
        }
    }
}
```

The artifact story here is stronger than the original skeleton suggested, and the post should lean into that.

- Anthropic baseline via cc-proxy: `53,992` cache creation tokens and `215,102` cache read tokens across 6 requests, which works out to essentially complete reuse after the first request.
- Dynamo localhost measurement on a 52K-token prompt: stable prefix `168ms` TTFT, varying prefix `912ms`, stripped preamble `169ms`.
- That is a `5.4x` slowdown when the prefix varies, and stripping restores the cache hit almost perfectly on this workload.

The framing should be precise. Anthropic is the baseline for what good harness behavior looks like. Dynamo's result is the systems lesson: harness quirks that look incidental at the API boundary can destroy cache reuse if they perturb the prefix too early.

Suggested paragraph:

Claude Code gave us a clean example of how harness semantics become serving semantics. On Anthropic's API, the billing preamble is absorbed into managed prompt caching and effectively disappears as an operational concern. On Dynamo, the same line sits at the front of a prefix-matched KV cache. Left untouched, it turns every session into a new prompt. Stripping it before tokenization is not an API polish item. It is the difference between a 168ms cache hit and a 912ms full-prefill miss on a 52K-token prompt.

Use these artifacts in this section:

- `agentic-harnesses-artifacts/prompt-instability/plots/cache-effect-v2.png` as the hero visual.
- `agentic-harnesses-artifacts/prompt-instability/raw/anthropic-baseline-stats.json` for the Anthropic baseline.
- The ASCII prefix-diff from `agentic-harnesses-artifacts/prompt-instability/README.md`.

Do not lean on the older noisy tunnel-based TTFT data. The localhost B200 result is the one with real signal.

## Section 2: Reasoning Fidelity Is KV Correctness

Interleaved reasoning is easy to misclassify as a rendering problem. It is not. It is a prompt reconstruction problem, which makes it a KV correctness problem.

If a model generates:

```text
<think>reasoning_0</think> tool_call_0 <think>reasoning_1</think> tool_call_1
```

then the next turn has to replay that assistant output in the same structural order. If the replay path flattens all reasoning before all tool calls:

```text
<think>reasoning_0 reasoning_1</think> tool_call_0 tool_call_1
```

the visible meaning may look similar, but the token sequence is different. That means the KV prefix computed during generation no longer matches the prefix seen on replay.

Dynamo's fix was to preserve reasoning as ordered segments rather than one flattened string:

```rust
pub enum ReasoningContent {
    Text(String),
    Segments(Vec<String>),
}
```

The semantic contract is the important part: `segments[i]` is the reasoning that appeared before `tool_calls[i]`, and `segments[N]` is any trailing reasoning after the last tool call. That preserves the original token order instead of reconstructing a lossy approximation.

This section should stay structural. The artifact set supports the claim that incorrect reconstruction breaks the prefix. It does not support a strong latency number on the measured deployment. The prompt in the test is small, the system is aggregated, and the timing data is noisy. The post should say that directly instead of reaching for a weak benchmark.

Suggested paragraph:

The hardest bugs here are the ones that look harmless from the outside. A flattened replay can render correctly, pass a casual eyeball test, and still be wrong for caching. KV reuse depends on token order, not on whether two prompts feel semantically equivalent to a human. Preserving interleaved reasoning and tool calls was therefore less about pretty transcripts and more about making turn `N+1` look exactly like turn `N` did to the cache.

Use these artifacts in this section:

- `agentic-harnesses-artifacts/reasoning-order/raw/trace-example.json`
- The token-order diagram from `agentic-harnesses-artifacts/reasoning-order/README.md`

Do not plot the TTFT CSV for this section. It does not carry the argument.

## Section 3: Streaming Actionable State

Streaming tokens is not enough for harnesses. Agent loops need actionable state as soon as it exists: completed tool calls, completed reasoning blocks, and token accounting that clients can trust while the stream is still in flight.

The original intuition here was that early dispatch might save latency. The artifact set supports a more careful and more interesting claim. Dynamo's streaming dispatch work is primarily about structure, not about a large measured wall-time win on this workload.

Without dispatch, the harness sees a regular token stream and has to infer when a tool call is complete by accumulating deltas and waiting for enough structure to be present. With dispatch enabled, Dynamo can emit a typed SSE side channel:

```text
event: tool_call_dispatch
data: {"choice_index":0,"tool_call":{"index":0,"id":"call-...","type":"function","function":{"name":"calculator","arguments":"{\"expression\":\"42 * 17\"}"}}}
```

That event tells the harness, in one shot, that the tool call is ready to execute. No client-side delta assembly, no guessing whether the arguments are complete, no custom parser living inside the harness.

The important nuance is that the current measurements do not show a meaningful end-to-end speedup on this particular model and workload. The strongest version of the claim is:

- dispatch gives the harness structured notification at the moment a tool call is parseable;
- it simplifies harness logic and makes tool readiness explicit;
- on the measured workload, it does not materially reduce stream completion time by itself.

Suggested paragraph:

The real win in streaming dispatch is not that the stream suddenly ends earlier. It is that the harness stops having to reverse-engineer server intent from a sequence of token deltas. A tool call is a state transition, not just another substring in the stream. Making that transition explicit gives harnesses something they can build against reliably, which matters more than shaving a few milliseconds off a toy calculator example.

Use these artifacts in this section:

- `agentic-harnesses-artifacts/streaming-actionable-state/plots/timeline-no-dispatch.png` as the main visual.
- `agentic-harnesses-artifacts/streaming-actionable-state/plots/dispatch-comparison.png` if you want to show the structural-vs-latency distinction explicitly.
- The event capture in `agentic-harnesses-artifacts/streaming-actionable-state/README.md`.

Avoid claiming "dispatch saves 31ms" or similar. The artifact README is clear that the more durable story is typed, actionable state.

## Section 4: Anthropic and Claude Code API Fidelity

Claude Code compatibility is more than text generation behind an Anthropic-shaped endpoint. The harness depends on a collection of smaller behaviors that are easy to miss in ad hoc testing:

- model metadata at both `GET /v1/models` and `GET /v1/models/{model_id}`
- correct handling of slashed model IDs
- useful `input_tokens` in `message_start`
- proper thinking blocks
- acceptance of `cache_control`
- response shapes that track the Anthropic API closely enough for clients not to trip over them

This section should be short and concrete. It is best as a table plus a few carefully chosen examples.

The most useful implementation detail to call out is that Claude Code does not stop at `GET /v1/models`. It also retrieves the specific connected model, which means the route has to handle model IDs like `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` without treating the slash as a path separator bug. This is a good example of harness compatibility being more than "the field exists somewhere."

Suggested paragraph:

What made Claude Code support feel real was not one big switch. It was a collection of small fidelity fixes that moved Dynamo from "mostly Anthropic-shaped" to "credible as an Anthropic backend." Some of them were obvious, like thinking blocks. Some were annoyingly specific, like wildcard model retrieval for slashed IDs and the need for non-zero `input_tokens` in `message_start`. Together, they determined whether Claude Code could reason about the backend the way it reasons about Anthropic's own API.

Use these artifacts in this section:

- The expected-vs-returned table from `agentic-harnesses-artifacts/anthropic-fidelity/README.md`
- One model-retrieve example showing why slashed IDs matter
- One `message_start` example if you want to talk about token accounting

Keep the tone checklist-driven rather than benchmark-driven.

## Section 5: Responses / Codex Fidelity

The Codex-facing version of the same story lives on the `v1/responses` side. Passing compliance tests is not enough if realistic replay and field preservation are lossy.

The clean architectural idea here is Dynamo's `ResponseParams` path. Instead of letting a Responses request collapse into chat completions and then trying to reconstruct the missing pieces afterward, Dynamo extracts the client-facing response parameters up front, preserves them through the internal conversion, and merges them back into the final response object.

That turns the internal conversion path from an hourglass into a controlled translation layer. Fields that the engine does not care about, such as `instructions`, `store`, `truncation`, or input-item metadata, do not silently vanish just because the internal runtime speaks a chat-completions-shaped dialect.

Suggested paragraph:

Codex surfaced a different failure mode than Claude Code. The issue was not whether Dynamo could generate the next token. It was whether a realistic Responses request could survive an internal round-trip without losing the fields that made it a Responses request in the first place. Preserving those fields turned out to be an architectural concern, not just a serializer concern.

Use these artifacts in this section:

- The field-preservation diagram from `agentic-harnesses-artifacts/responses-fidelity/README.md`
- One concrete replay example

This section should stay shorter than the Claude Code sections. One diagram and one request example are enough.

## Close

The architecture from the first post only pays off if the harness-facing layer preserves enough structure for the router and the cache to exploit it. That is the connective tissue between these two posts.

Prompt stability affects KV reuse. Replay fidelity affects whether the next turn can hit cache at all. Stream semantics affect when the harness can act. Metadata fidelity affects whether the client can manage context and model selection correctly. None of that is a thin compatibility shim over the "real" serving stack. For agentic workloads, it is part of the serving stack.

Good closing line:

For agentic workloads, protocol fidelity is performance work.

## Primary Assets To Pull Into The Draft

- `agentic-harnesses-artifacts/prompt-instability/plots/cache-effect-v2.png`
- `agentic-harnesses-artifacts/reasoning-order/raw/trace-example.json`
- `agentic-harnesses-artifacts/streaming-actionable-state/plots/timeline-no-dispatch.png`
- `agentic-harnesses-artifacts/anthropic-fidelity/README.md`
- `agentic-harnesses-artifacts/responses-fidelity/README.md`

## Guardrails While Turning This Into The Final Post

- Keep the callback to the first post brief.
- Keep Claude Code as the narrative anchor.
- Use quantitative claims only where the artifacts are strong.
- Where the evidence is structural rather than benchmark-driven, say so plainly.
- Do not overclaim streaming dispatch as a latency win on the current workload.
- Keep the Anthropic and Responses fidelity sections crisp and specific.
