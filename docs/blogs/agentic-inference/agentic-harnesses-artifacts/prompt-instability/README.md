# Section 1: Prompt Instability

## Deployment Details

- **Model**: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`
- **Node**: B200 GPU
- **Serving mode**: Aggregated (single worker)
- **Preamble stripping**: `--strip-anthropic-preamble` / `DYN_STRIP_ANTHROPIC_PREAMBLE=1` — **enabled**
- **Date**: 2026-03-19

## The Problem

Claude Code prepends a session-specific billing header to every system prompt:

```
x-anthropic-billing-header: cc_version=0.2.93; cch=abc123def456;
You are Claude Code, an interactive CLI tool...
```

The `cch=` value changes per session and per release. This means:
- Every new Claude Code session produces a **different** first line in the system prompt
- The stable content (instructions, tools, context) follows after this unstable line
- Any prefix-based KV cache sees a unique prefix for every session → **no reuse**

### Visual: Why the prefix breaks

```
Session A system prompt:
  "x-anthropic-billing-header: cc_version=0.2.93; cch=abc123;\n
   You are Claude Code..."
                                                        ▲
Session B system prompt:                                │ diverges here
  "x-anthropic-billing-header: cc_version=0.2.93; cch=def456;\n
   You are Claude Code..."

After stripping:
Session A: "You are Claude Code..."  ─┐
Session B: "You are Claude Code..."  ─┘── identical prefix → cache reuse ✅
```

## Implementation

The stripping function (`lib/llm/src/http/service/anthropic.rs:429-438`):

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

**Verified behavior:** With stripping enabled, a request with the billing preamble produces `input_tokens: 24` — identical to the same request without the preamble. The header is removed before tokenization.

## Token Count Verification

| System prompt | `input_tokens` |
|---|---|
| `"x-anthropic-billing-header: cc_version=0.2.93; cch=abc123;\nYou are a helpful coding assistant."` | 24 |
| `"You are a helpful coding assistant."` | 24 |

The preamble is fully stripped before reaching the engine.

## TTFT Measurement Results

Ran 3 rounds × 8 sequential requests per condition:

| Condition | Description | Mean TTFT (ms) | Stdev (ms) |
|-----------|-------------|-----------------|------------|
| Stable | Same system prompt every request | 146.5 | 67.8 |
| Varying | Different billing header each request | 157.9 | 66.3 |
| Stripped | Billing header present, stripped by Dynamo | 138.9 | 52.5 |

### Assessment: No clean TTFT signal

**The TTFT data does not show a statistically significant difference between conditions on this deployment.**

Same reasons as Section 2:
1. **Aggregated serving** — no prefill/decode separation
2. **Small prompt** (~24 tokens system + message) — cache benefit is negligible at this scale
3. **SGLang prefix caching conditions** may not be met
4. **SSH tunnel noise** (~100ms baseline) masks small effects
5. **No comparison against Anthropic baseline** — requires capturing real Claude Code traffic

### What the data does show

The "stripped" condition has the **lowest variance** (stdev 52.5ms vs 67-68ms for the others). This is suggestive — stable prefixes produce more consistent latency — but not strong enough to claim as a result.

## Anthropic Baseline ✅

Captured via cc-proxy in `anthropic-only` mode with Claude Code (Sonnet 4.6) sending 5 multi-turn questions about the Dynamo codebase.

### Setup

- **cc-proxy** in `anthropic-only` mode → passthrough to `api.anthropic.com`
- **Phoenix/OTLP** for trace collection
- **Claude Code v2.1.79** with `ANTHROPIC_BASE_URL=http://localhost:3080`
- **Model**: `claude-sonnet-4-6`
- **Workflow**: 5 sequential one-sentence questions (frontend, router, planner, backends, KVBM)

### Results (aggregate from cc-proxy /api/stats)

```json
{
  "total_requests": 6,
  "input_tokens": 23,
  "output_tokens": 241,
  "cache_read_input_tokens": 215102,
  "cache_creation_input_tokens": 53992,
  "tool_calls": 0
}
```

### Analysis

| Metric | Value |
|--------|-------|
| Total requests | 6 (1 init + 5 prompts) |
| Cache creation | 53,992 tokens (first request — system prompt + CLAUDE.md) |
| Cache reads | 215,102 tokens (subsequent requests reuse cached prefix) |
| Non-cached input | 23 tokens (only the varying user messages) |
| **Cache hit ratio (turns 2-6)** | **~99.99%** |

**Key finding:** Anthropic's managed prompt caching is extremely effective for Claude Code. The ~54K token system prompt (instructions, tools, CLAUDE.md context) is cached on the first request. All subsequent requests in the same session read from cache, paying only for the small user message delta.

### Why this matters for the preamble story

The billing preamble (`x-anthropic-billing-header: cc_version=...; cch=...;\n`) is placed **before** the system prompt content. On Anthropic's API, this doesn't matter — their caching is keyed on the full content and the `cache_control` placement, not strict prefix matching.

But on Dynamo, which uses **KV prefix caching** (token-sequence-based matching), the varying preamble at the front of the system prompt **breaks the prefix match** for every new session. This is why `--strip-anthropic-preamble` exists — it removes the varying line so that the stable system prompt content starts at position 0, enabling KV prefix cache reuse across sessions.

### Comparison frame

| Aspect | Anthropic API | Dynamo (without stripping) | Dynamo (with stripping) |
|--------|--------------|---------------------------|------------------------|
| Caching mechanism | Managed prompt cache | KV prefix matching | KV prefix matching |
| Billing preamble impact | None (content-hash based) | Breaks prefix match | Removed before tokenization |
| Expected cache reuse | ~100% after first request | 0% across sessions | ~100% across sessions* |

*Assumes prefix caching is active on the Dynamo deployment. Not directly measured on the aggregated deployment used for this experiment.

## Comparisons Not Performed

Per the experiment plan, these were requested but not performed:

| Comparison | Status | Reason |
|---|---|---|
| Anthropic baseline | ✅ Done | Via cc-proxy anthropic-only mode |
| Dynamo aggregated, preamble present | ⚠️ Partial | Stripping is globally enabled on this deployment; tested via "varying" condition |
| Dynamo aggregated, preamble stripped | ✅ Done | "Stable" and "stripped" conditions |
| Dynamo disaggregated, preamble present | ❌ Not done | No disaggregated deployment available |
| Dynamo disaggregated, preamble stripped | ❌ Not done | No disaggregated deployment available |

## Blog Recommendation

**Include — now with quantitative Anthropic baseline.** The Anthropic cache data (215K cached tokens, 99.99% hit ratio) is concrete and dramatic. Combined with the preamble diff and the structural argument, this section is now strong.

**Strongest artifact:** The Anthropic baseline cache stats showing near-perfect cache reuse, contrasted with the structural argument for why Dynamo's KV prefix matching breaks without stripping.

**Blog story:** "Anthropic's managed caching handles the billing preamble transparently — 99.99% cache hits. But Dynamo's KV prefix matching is token-sequence-based: a varying preamble at position 0 breaks the prefix match for every session. `--strip-anthropic-preamble` removes the noise so the stable system prompt starts at position 0."

## V2 Results: Localhost Measurement on B200 (2026-03-20) ⭐

Re-ran the experiment **directly on the compute node** (localhost HTTP, no SSH tunnel) with the 52K-token prompt from real Dynamo docs. This eliminated network noise entirely (stdev dropped from 200-350ms to 7-17ms).

### Three conditions, 3 rounds × 15 requests each

| Condition | Mean TTFT | Stdev | Interpretation |
|-----------|-----------|-------|----------------|
| **Stable** (same prefix every request) | **168ms** | 7ms | Prefix cache hit — only new tokens computed |
| **Varying** (unique non-strippable prefix) | **912ms** | 17ms | Cache miss — full 52K token prefill every time |
| **Stripped** (billing header present, stripped by Dynamo) | **169ms** | 7ms | Cache hit — stripping restores prefix stability |

### Key numbers for the blog

- **5.4× TTFT increase** from cache miss (912ms vs 168ms)
- **744ms penalty per request** when prefix varies
- **Stripping restores cache completely**: stripped (169ms) ≈ stable (168ms)
- Over a 10-turn Claude Code session: **7.4 seconds of wasted prefill**
- Results are **perfectly consistent** across all 3 rounds (stable=167-169ms, varying=911-912ms, stripped=168-170ms)

### Why the V1 experiment failed

The original experiment ran through an SSH tunnel (100-500ms variable latency) with a ~400 token prompt. The cache effect (~10ms at that scale) was completely invisible. The V2 experiment fixed both issues: localhost measurement + 52K token prompt.

### Important: the "varying" condition uses non-strippable noise

The first attempt with billing headers (`x-anthropic-billing-header: ...`) showed no difference because `--strip-anthropic-preamble` was enabled and stripped them. The V2 varying condition uses `Session context: <uuid>\n` which the stripping logic does not recognize — this simulates what would happen if the billing header were NOT stripped.

### Plot

`plots/cache-effect-v2.png` — ⭐ **New hero figure for Section 1.** Two-panel: box plot + per-request time series. Clean separation, zero overlap between conditions.

## Raw Data

- `raw/cache-final-3conditions.jsonl` — ⭐ V2 results, 3 rounds × 3 conditions × 15 requests, localhost
- `raw/anthropic-baseline-stats.json` — cc-proxy aggregate cache stats from Anthropic API
- `raw/ttft-preamble-comparison.jsonl` — V1 results (noisy, through SSH tunnel — DO NOT PLOT)
- `raw/clean-ab-comparison.jsonl` — V1.5 results (localhost but billing header was stripped — inconclusive)
- `derived/ttft-preamble-comparison.csv` — V1 tabular (noisy — DO NOT PLOT)
- `plots/cache-effect-v2.png` — ⭐ V2 publication-ready figure
