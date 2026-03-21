# Section 5: Responses / Codex Fidelity

## Deployment Details

- **Model**: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`
- **Node**: B200 GPU
- **Access**: SSH tunnel `localhost:8000`
- **Date**: 2026-03-19

## Summary of Findings

### Architecture: Request Echo via ResponseParams

Dynamo preserves original request parameters through a `ResponseParams` struct extracted **before** converting `NvCreateResponse` → `NvCreateChatCompletionRequest`. This avoids the lossy hourglass problem where fields are dropped during the internal chat completion roundtrip.

**Preserved fields (echoed back in response):**
- `model`, `temperature`, `top_p`, `max_output_tokens`
- `store`, `tools`, `tool_choice`, `instructions`
- `reasoning`, `text`, `service_tier`, `include`, `truncation`

**Fields set to null/default (not round-tripped from input):**
- `billing`, `conversation`, `previous_response_id`, `prompt`, `prompt_cache_key`, `prompt_cache_retention`, `safety_identifier`, `max_tool_calls`, `incomplete_details`

**Hardcoded spec defaults:**
- `background: false`, `frequency_penalty: 0.0`, `presence_penalty: 0.0`, `parallel_tool_calls: true`

### Test 1: Simple Text Generation

```bash
curl -s -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
       "input":"Say hello briefly.","max_output_tokens":50}'
```

**Result:** ✅ Correct structure. Response contains:
- `output[0]`: reasoning block with `summary_text`
- `output[1]`: message block with `output_text`
- All request params echoed back

### Test 2: Multi-Turn Input Items with Tools

```bash
curl -s -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
       "input":[
         {"type":"message","role":"user","content":"What is 2+2?"},
         {"type":"message","role":"assistant","content":"4"},
         {"type":"message","role":"user","content":"And 3+3?"}
       ],
       "max_output_tokens":50,
       "tools":[{"type":"function","name":"calculator","description":"Calculate",
                 "parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}],
       "tool_choice":"auto"}'
```

**Result:** ✅ Multi-turn input items correctly converted to messages. Tools echoed back with `strict: true` appended.

### Key Observations

#### 1. Reasoning Token Accounting Gap

`output_tokens_details.reasoning_tokens: 0` even when reasoning content is present in the output. The reasoning tokens are counted in `output_tokens` total but not broken out in the detail field. This matters for Codex-style harnesses that budget reasoning vs non-reasoning tokens.

#### 2. `strict: true` Added to Tools

Tools echoed back with `strict: true` appended even when not provided in the request. This is a normalization behavior — not harmful but observable.

#### 3. Input Items → Messages Conversion

The `InputParam::Items` path converts multi-turn conversation items to chat completion messages. Fields like `id` and `status` on input items are **not preserved** through the conversion — they exist only in the Responses API layer, not in the chat completion internal representation.

This is the key "field preservation" story: without `ResponseParams`, these fields would be lost entirely. With it, the response echoes back the parameters the client sent, even though the internal engine never saw them.

#### 4. `nvext` Timing Extension

Dynamo adds `nvext.timing` with `request_received_ms` and `total_time_ms` — useful for harness-level latency measurement. This is a Dynamo extension not in the OpenAI spec.

## Field Preservation Diagram

```
Client Request (NvCreateResponse)
│
├─► ResponseParams ──────────────────────► Response echo
│   (model, tools, instructions, etc.)     (preserved verbatim)
│
└─► TryFrom conversion
    │
    └─► NvCreateChatCompletionRequest
        │ (internal engine format)
        │
        │ Fields LOST in conversion:
        │ - input item `id`, `status`
        │ - `previous_response_id`
        │ - `store`, `truncation` (not engine concepts)
        │
        └─► Engine generates stream
            │
            └─► chat_completion_to_response()
                │ merges engine output + ResponseParams
                │
                └─► Final Response object
                    (all client-facing fields present)
```

## Before/After Comparison

The "before" state would be a naive conversion that drops fields through the chat completion roundtrip. The current implementation avoids this by extracting `ResponseParams` before conversion.

**Without ResponseParams (hypothetical lossy path):**
- `instructions` → lost (converted to system message, not recoverable)
- `tools` → partially lost (converted to chat completion tool format, original shape not echoed)
- `store`, `truncation`, `service_tier` → lost (no chat completion equivalent)
- `reasoning` → lost (converted to `reasoning_effort`, original not recoverable)

**With ResponseParams (current implementation):**
- All above fields preserved and echoed back correctly ✅

## Blog Recommendation

**Include as a short section.** The ResponseParams echo-back pattern is a clean architectural story that contrasts with the naive "hourglass" conversion approach. The field preservation diagram is a good visual for the blog.

Keep it brief — this doesn't need benchmark data. One request/response example plus the diagram is sufficient.

## Raw Data

See `raw/` for captured responses.
