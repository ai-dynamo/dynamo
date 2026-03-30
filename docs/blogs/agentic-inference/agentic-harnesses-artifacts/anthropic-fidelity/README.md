# Section 4: Anthropic / Claude Code API Fidelity

## Deployment Details

- **Model**: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`
- **Node**: B200 GPU
- **Access**: SSH tunnel `localhost:8000`
- **Dynamo flags**: `--enable-anthropic-api`, `--dyn-reasoning-parser nemotron_nas`
- **Date**: 2026-03-19

## Summary of Findings

### 1. Model List Endpoint (`GET /v1/models`)

**No content negotiation.** The `anthropic-version` header has no effect on the response shape — both return OpenAI-format JSON:

```json
{
    "object": "list",
    "data": [
        {
            "id": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
            "object": "model",
            "created": 1773884843,
            "owned_by": "nvidia"
        }
    ]
}
```

**Missing fields expected by Claude Code:**
- `context_window` — Claude Code uses this to determine how much context to send
- `max_output_tokens` — Claude Code uses this for completion budget
- `created_at` — Anthropic format uses ISO 8601 string, not Unix epoch integer
- `type` — Anthropic format uses `"type": "model"` not `"object": "model"`

### 2. Individual Model Retrieval (`GET /v1/models/{model_id}`)

**Returns 404.** No individual model retrieval endpoint exists.

```
GET /v1/models/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
HTTP/1.1 404 Not Found
content-length: 0
```

This is required by Claude Code for model metadata lookup. The slashed model ID (`nvidia/NVIDIA-...`) also requires wildcard route handling to avoid being split across path segments.

### 3. Streaming `message_start` Usage

**`input_tokens` is 0 at stream start:**

```json
{
  "type": "message_start",
  "message": {
    "usage": {"input_tokens": 0, "output_tokens": 0}
  }
}
```

Anthropic's API returns non-zero `input_tokens` in `message_start`. Claude Code uses this for context window accounting mid-stream. The actual token count only appears in the final `message_delta` usage event.

### 4. Thinking/Reasoning Blocks

**Working correctly.** Both streaming and non-streaming paths emit proper `thinking` content blocks:

**Streaming:**
```
event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"User"}}
```

**Non-streaming:**
```json
{
  "content": [
    {"type": "thinking", "thinking": "...", "signature": ""}
  ]
}
```

### 5. `cache_control` in Requests

**Accepted without error** but no cache metrics returned in response. The `usage` block does not include `cache_creation_input_tokens` or `cache_read_input_tokens`:

```json
{"usage": {"input_tokens": 17, "output_tokens": 20}}
```

This is expected — Dynamo's KV cache is an internal optimization, not exposed as Anthropic-style prompt caching semantics. But Claude Code checks these fields.

### 6. Preamble Stripping (`strip_billing_preamble`)

**Implementation verified in source** (`lib/llm/src/http/service/anthropic.rs:429-438`):

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

Strips the per-session `x-anthropic-billing-header: cc_version=...; cch=...;\n` line that Claude Code prepends to every system prompt. Controlled by `--strip-anthropic-preamble` / `DYN_STRIP_ANTHROPIC_PREAMBLE=1`.

## Expected vs Returned (Summary Table)

| Field / Behavior | Expected by Claude Code | Returned by Dynamo | Status |
|---|---|---|---|
| `GET /v1/models` list | Yes | Yes (OpenAI format) | ⚠️ No Anthropic format |
| `GET /v1/models/{id}` retrieve | Yes | 404 | ❌ Missing |
| `context_window` in model | Yes | Not present | ❌ Missing |
| `max_output_tokens` in model | Yes | Not present | ❌ Missing |
| `input_tokens` in `message_start` | Non-zero | 0 | ⚠️ Deferred |
| `cache_read_input_tokens` | Yes (when caching) | Not present | ⚠️ Not applicable |
| Thinking blocks (streaming) | Yes | Yes | ✅ Working |
| Thinking blocks (non-streaming) | Yes | Yes | ✅ Working |
| `cache_control` accepted | Yes | Yes (no error) | ✅ Accepted |
| Preamble stripping | N/A (Dynamo feature) | Implemented | ✅ Working |
| `anthropic-version` negotiation | Expected on model endpoints | Not implemented | ⚠️ Missing |

## Blog Recommendation

**Include with caveat.** This section has concrete curl-verifiable evidence for several compatibility gaps that directly affected Claude Code usability. The most impactful items are:

1. Missing model retrieval endpoint (404) — blocks Claude Code from resolving model metadata
2. Missing `context_window` / `max_output_tokens` — Claude Code defaults to conservative values
3. `input_tokens: 0` in `message_start` — breaks mid-stream context accounting

These are good blog material because they are specific, verifiable, and have clear fixes.

## Raw Data

See `raw/` for captured curl responses.
