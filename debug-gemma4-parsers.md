# Debug: Gemma 4 tool calls leak into reasoning content

**Date**: 2026-07-09
**Source**: User report
**Status**: complete
**Environment**: GPU 1, NVIDIA RTX 5880 Ada Generation, 49,140 MiB; driver 590.48.01
**Baseline**: `origin/main` at `485c298fff259ea228d4d86e23fe4153aead3c8a`

## Problem

Run `nvidia/Gemma-4-31B-IT-NVFP4` through `dynamo.vllm` and
`dynamo.sglang` with Dynamo's Gemma 4 tool and reasoning parsers plus the
backend-native reasoning parser. A required tool request containing literal Gemma tool
markers, braces, brackets, and the string delimiter can leak the generated tool call into
`reasoning_content` and exhaust `max_tokens` instead of returning one structured call.

## Reproduction Steps

1. Build current Dynamo source in a backend-specific virtual environment.
2. Start the discovery plane, Dynamo frontend, and one worker on GPU 1.
3. Configure `--dyn-tool-call-parser`, `--dyn-reasoning-parser`, and the worker's native
   `--reasoning-parser` for Gemma 4.
4. Submit the exact non-streaming required-tool request from the user report at temperature 0.
5. Repeat with vLLM 0.24.0 and SGLang 0.5.14 in separate environments.

## Expected vs Actual

- **Expected**: `finish_reason=tool_calls`; exactly one `run_query` call; exact `sql`
  argument; no assistant content; no raw tool syntax in reasoning.
- **Actual (reported)**: `finish_reason=length`; no structured `tool_calls`; generated
  tool syntax and argument text appear inside `reasoning_content`; completion consumes all
  1,024 requested tokens.

## Investigation Log

### 2026-07-09 — setup

- Confirmed source pins: vLLM 0.24.0 and SGLang 0.5.14.
- Confirmed GPU 1 is the requested RTX 5880 and is initially free.
- Confirmed Hugging Face authentication; exact Gemma checkpoint is not cached.
- Created `rmccormick/gemma4-parsers` directly from latest `origin/main`.
- Unrelated untracked debug logs pre-existed and are excluded from this work.

### 2026-07-09 — parser-level reproduction

- Confirmed the model tokenizer encodes `<|tool_call>`, `<tool_call|>`, `<|"|>`,
  `<|channel>`, and `<channel|>` as individual special tokens.
- Confirmed vLLM 0.24.0's native `gemma4` reasoner treats `<|tool_call>` as an
  implicit reasoning end and activates guided decoding at that token.
- Confirmed SGLang 0.5.14's `gemma4` response detector recognizes only the explicit
  `<channel|>` end token; its guided-reasoning grammar also transitions only on that token.
- Added a CPU stream replay with the exact adversarial SQL, a split `<|tool_call>`
  boundary, and a named `run_query` choice. Latest main failed: all reasoning, the marker,
  and the complete guided JSON accumulated in `reasoning_content`; no tool call was emitted.
- Implemented a one-shot, choice-local Gemma guided-tool handoff. It recognizes an
  explicit close or replaces the first implicit tool marker with the close marker, then
  stops scanning so identical markers in the JSON argument remain untouched.
- The focused replay passes with the exact SQL argument and `finish_reason=tool_calls`.

## Root Cause

The backend-native reasoning gate and Dynamo's response reasoner implement different
Gemma 4 state machines. vLLM treats `<|tool_call>` as an implicit reasoning close, while
the pinned `dynamo-parsers` 3.1.0 Gemma reasoner waits only for `<channel|>`. When the model
uses the tool token directly from its thought channel, backend guided JSON begins but
Dynamo continues classifying every byte as reasoning.

## Fix

Normalize the first implicit Gemma tool boundary before Dynamo reasoning parsing on
required/named, non-structural requests with thinking enabled. Preserve later marker text
verbatim because it belongs to the guided tool argument.

### 2026-07-09 — vLLM 0.24.0 live validation

- Stock vLLM 0.24.0 could not load this ModelOpt checkpoint because an excluded tied
  `ParallelLMHead` received `UnquantizedLinearMethod`, whose `tie_weights` method is not
  implemented. Applied an environment-only compatibility patch selecting
  `UnquantizedEmbeddingMethod` for that case; this patch is not part of the Dynamo diff.
- Launched the text architecture on GPU 1 with all three parsers set to `gemma4`.
- Baseline frontend reproduced the leak: HTTP 200, `finish_reason=stop`, 93 output tokens,
  331 reasoning characters containing `<|tool_call>`, and zero structured tool calls.
- Rebuilt the frontend with the fix and replayed the identical request against the same
  worker: HTTP 200, `finish_reason=tool_calls`, 93 output tokens, 165 reasoning characters
  with no tool marker or SQL, and exactly one `run_query` call.
- The model itself replaced the requested literal `<|"|>.` suffix with U+00A0 at
  temperature 0. This was already present in the generated JSON, so the live argument is
  not character-exact. The focused synthetic parser replay proves parser preservation when
  the backend produces the exact string.
- Wrote the first backend findings to `debug-gemma4-parsers.html`.

### 2026-07-09 — SGLang 0.5.14 live validation

- Launched the full Gemma 4 conditional-generation architecture in the separate SGLang
  environment with all three parser selectors set to `gemma4`.
- Stock 0.5.14 transitions guided decoding only on `<channel|>`. The model instead emitted
  `<|tool_call>`, so no named-tool grammar activated; Dynamo dropped the invalid jail
  output after the embedded `get_time` marker was misread as a call.
- Added a version-scoped compatibility hook that wraps SGLang's spawned scheduler target,
  recognizes `<|tool_call>` as an alternate Gemma 4 reasoning end, and preserves the token
  across request-local grammar copies.
- The transition alone allowed xgrammar to generate whitespace until the 1,024-token limit.
  Added `--constrained-json-disable-any-whitespace` to the verified launch configuration so
  the post-reasoning named-tool JSON begins immediately.
- Final stock-environment branch replay: HTTP 200, `finish_reason=tool_calls`, 180 output
  tokens, 165 clean reasoning characters, and one `run_query` call in 14.943 seconds.
- SGLang also failed character-exact SQL generation, preserving the embedded tool marker
  but replacing the delimiter suffix and continuing with prose inside the JSON string.
  This is generated-model output rather than a parser transformation.
- Completed both backend sections in `debug-gemma4-parsers.html`.
