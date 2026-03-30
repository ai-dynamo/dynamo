# Section 2: Reasoning Fidelity and KV Correctness

## Deployment Details

- **Model**: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`
- **Node**: B200 GPU
- **Serving mode**: Aggregated (single worker)
- **Date**: 2026-03-19

## Experiment Design

### Method: Trace mutation (Method 3)

The current codebase has the corrected behavior (`ReasoningContent::Segments`). We create a "broken" variant by flattening all reasoning segments into a single string (`ReasoningContent::Text`), which represents the old behavior where interleaving order was lost.

### The Two Reconstruction Forms

**Original model output** (during generation):
```
<think>sqrt(144) = 12. I should use calculator.</think>
[tool_call: calculator("sqrt(144)")]
<think>Got 12. Now multiply by 7.</think>
[tool_call: calculator("12 * 7")]
```

**Correct reconstruction** (Segments — preserves interleaving):
```json
{
  "reasoning_content": [
    "sqrt(144) = 12. I should use calculator.",
    "Got 12. Now multiply by 7.",
    ""
  ],
  "tool_calls": [
    {"function": {"name": "calculator", "arguments": "{\"expression\":\"sqrt(144)\"}"}},
    {"function": {"name": "calculator", "arguments": "{\"expression\":\"12 * 7\"}"}}
  ]
}
```

Token sequence: `<think>reasoning_0</think> tool_call_0 <think>reasoning_1</think> tool_call_1`

**Incorrect reconstruction** (Flattened — all reasoning before all tools):
```json
{
  "reasoning_content": "sqrt(144) = 12. I should use calculator.\nGot 12. Now multiply by 7.\n",
  "tool_calls": [
    {"function": {"name": "calculator", "arguments": "{\"expression\":\"sqrt(144)\"}"}},
    {"function": {"name": "calculator", "arguments": "{\"expression\":\"12 * 7\"}"}}
  ]
}
```

Token sequence: `<think>reasoning_0 reasoning_1</think> tool_call_0 tool_call_1`

### Why the order matters for KV cache

When the model generated the original response, the KV cache entries were computed for the interleaved token sequence. On the next turn, the prompt includes the previous assistant message as context. If the reconstruction uses a **different token order**, the KV cache prefix from the original generation no longer matches — forcing a full recomputation of the prefix.

```
Original generation:    [think][r0][/think][tool0][think][r1][/think][tool1]
                                    ↕ KV cache stores this exact sequence

Correct reconstruction: [think][r0][/think][tool0][think][r1][/think][tool1]
                        ✅ Prefix matches — cache reuse possible

Flat reconstruction:    [think][r0][r1][/think][tool0][tool1]
                        ❌ Prefix diverges at position of [/think] — no cache reuse
```

### Implementation

The fix is in `ReasoningContent` (defined in `lib/async-openai/src/types/chat.rs:470-488`):

```rust
pub enum ReasoningContent {
    /// Flat string — single reasoning block or legacy form.
    Text(String),
    /// Interleaved segments. segments[i] precedes tool_calls[i];
    /// segments[N] is trailing reasoning after the last tool call.
    /// segments.len() == tool_calls.len() + 1.
    Segments(Vec<String>),
}
```

And in the Anthropic conversion (`lib/llm/src/protocols/anthropic/types.rs:1055-1158`), `convert_assistant_blocks()` accumulates thinking text per tool call into ordered segments.

## TTFT Measurement Results

Ran 10 comparisons of segmented vs flattened reasoning, measuring TTFT on follow-up turns.

### Raw Data (ms)

| Run | Warmup (seg) | Followup (seg) | Warmup (flat) | Followup (flat) |
|-----|-------------|----------------|---------------|-----------------|
| 1 | 216.7 | 1483.2* | 106.9 | 122.4 |
| 2 | 112.0 | 114.8 | 122.1 | 114.9 |
| 3 | 111.7 | 123.6 | 110.6 | 116.0 |
| 4 | 114.0 | 115.5 | 119.6 | 109.3 |
| 5 | 119.2 | 119.7 | 119.4 | 113.6 |
| 6 | 111.5 | 139.6 | 107.9 | 126.9 |
| 7 | 398.7 | 163.5 | 205.3 | 225.2 |
| 8 | 335.1 | 197.1 | 193.0 | 116.4 |
| 9 | 120.3 | 112.8 | 216.2 | 115.8 |
| 10 | 173.4 | 179.7 | 137.3 | 118.3 |

*Run 1 outlier (1483ms) is a cold-start artifact.

### Assessment: No clean TTFT signal

**The TTFT data does not show a statistically significant difference between segmented and flattened reasoning on this deployment.**

Reasons:
1. **Aggregated serving** — no disaggregated prefill/decode separation, so KV cache transfer between workers is not a factor
2. **Small prompt** (~387 tokens) — cache benefit is minimal at this scale
3. **Prefix caching may not be active** — SGLang's prefix caching has conditions that may not be met for these short prompts
4. **Network noise** — the SSH tunnel adds ~100ms baseline latency that masks small cache effects

### What this means for the blog

The **structural argument is sound** — the token order demonstrably differs, and mismatched prefixes prevent cache reuse. But the **quantitative signal is not measurable** on this particular deployment at this prompt scale.

The blog should present this as:
1. A concrete code-level example of how reconstruction order matters (the side-by-side diff above)
2. An architectural argument about why prefix order affects KV cache (the diagram)
3. An honest note that the effect scales with prompt length and is most impactful in disaggregated serving where KV cache transfer has real cost

**Do not present the TTFT data as evidence of a measurable improvement** — it isn't, on this setup.

## Caveats

- **Decode KV not returned to prefill**: In disaggregated serving, decode workers do not currently return KV to prefill workers. This means the cache benefit of correct reconstruction is limited to the prefill-side prefix cache only. The blog should note this limitation.
- **Prompt scale**: The effect would be larger with longer conversations (thousands of tokens) where the prefix is a significant portion of total computation.
- **Model sensitivity**: Different models may be more or less sensitive to token reordering in the prompt template.

## Blog Recommendation

**Include with structural focus.** The side-by-side reconstruction diff and the prefix-match diagram are clear and compelling. The architectural argument is correct. Do not include the TTFT chart — the data doesn't support a quantitative claim on this deployment.

Best presentation: one code-level before/after showing the `ReasoningContent::Segments` fix, plus the token-order diagram showing why the prefix breaks.

## V2 Results: Localhost Experiments on B200 (2026-03-20)

Ran three experiments directly on the compute node (no SSH tunnel). All show zero TTFT delta.

### Exp 1: Segments vs Flat (30-city trace, 62 messages, ~10K tokens)
- **Result:** 107.3ms vs 107.3ms (delta: 0.0ms)
- **Why:** Backend generates identical tokens for Segments vs Flat — the distinction is a frontend API concept

### Exp 2: Exact vs Mutated reasoning (random chars prepended, ~10K tokens)
- **Result:** 107.2ms vs 107.1ms (delta: -0.1ms)
- **Why:** 10K tokens prefill in <10ms on B200 — below noise floor

### Exp 3: 52K system prompt + 30-city conversation (~62K tokens)
- **Result:** 177.6ms vs 177.8ms (delta: +0.2ms)
- **Why:** Prefix cache matches the 52K system prompt (identical in both conditions). Mutations only affect ~10K conversation tokens = ~10-20ms recomputation, invisible.

### Interpretation

1. **Prefix caching works** — confirmed by Section 1's 5.4× TTFT result
2. **Segments vs Flat produces identical tokens** on the current backend — the chat template wraps all reasoning in one `<think>` block regardless
3. **Conversation-level mutations are invisible** at ~10K tokens — need 50K+ conversation tokens for measurable effect
4. **Present Segments as API format correctness** (Anthropic thinking blocks must interleave with tool_use), not as a KV cache optimization

## Raw Data

- `raw/trace-example.json` — correct vs incorrect reconstruction (structural artifact)
- `raw/reasoning-order-30cities-localhost.jsonl` — Exp 1: Segments vs Flat
- `raw/reasoning-order-chat-52k-localhost.jsonl` — Exp 1 with 52K system prompt
- `raw/reasoning-cache-big-system-localhost.jsonl` — Exp 3: exact vs mutated + big system prompt
- `raw/ttft-comparison.jsonl` — V1 data (SSH tunnel, noisy — DO NOT PLOT)
- `derived/ttft-comparison.csv` — V1 tabular (DO NOT PLOT)
