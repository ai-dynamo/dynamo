# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end parity wrappers — Method 3 of DIS-1906 (NOT YET IMPLEMENTED).

Method 2 lives in `tests/parser_parity/impls/parser/`: in-process Python
imports of vLLM's `ToolParser` classes and SGLang's per-module detectors,
exercising parser logic only.

Method 3 lives here: HTTP wrappers around real `vllm serve` and
`python -m sglang.launch_server` processes. Same fixtures, different
invocation surface — the request goes through each server's full stack
(chat-template materialization, tokenizer round-trip, request shaping,
streaming chunk boundaries, response assembly).

What Method 3 catches that Method 2 doesn't
-------------------------------------------
- Tokenizer round-trip bugs: model emits tokens X, detokenizer recovers
  Y, parser sees Y. Method 2 skips both halves.
- Chat-template materialization: assistant-turn injection, tool-message
  handling, system-prompt merging.
- Streaming chunk-boundary edge cases at the SSE layer: partial-token
  deltas, finish_reason timing, choice-index handling.
- Request preprocessing: `tool_choice="none"` actually stripping tools
  from the prompt; guided-decoding setup propagating correctly.
- HTTP-shape divergences: vLLM's vs SGLang's `tool_calls[i].id` format,
  function-arg JSON encoding, `index` field semantics.

Component checklist
-------------------
1. Server boot: subprocess wrapper for `vllm serve` /
   `python -m sglang.launch_server` with `--load-format dummy --device
   cpu --enforce-eager`. CPU-only, no model weights, random init.
   Health-check + ready-wait. Pytest session-scoped fixture so we boot
   once per session, not per test. Cleanup on teardown (kill subprocess,
   reap zombies, free ports).

2. Forcing exact output via constrained decoding: each fixture's
   `model_text` becomes a `guided_regex` (escape special chars, anchor
   with `^...$`). vLLM and SGLang ship different regex engines (outlines
   vs xgrammar) with different escape rules; some fixtures may need
   per-engine variants. Verify tokenizer round-trip recovers the exact
   fixture text byte-for-byte. Special tokens (`<|tool_calls_section_begin|>`
   etc.) must be in the dummy tokenizer's vocab — fall back to splitting
   the regex around them when not.

3. HTTP client + chat shape: build request bodies with `messages` +
   `tools` from the fixture. POST to `/v1/chat/completions` (and
   `/v1/completions` as fallback when the chat template fights
   constrained decoding). Parse response: `choices[0].message.tool_calls`
   for non-streaming, accumulated SSE deltas for streaming.

4. Wrapper integration: `e2e/vllm.py` and `e2e/sglang.py` expose the
   same `parse(family, model_text, tools) -> ParseResult` interface as
   `impls/parser/`. Same fixtures, same KNOWN_DIVERGENCES registry —
   though entries may need an extra dimension (`mode = "method2"
   | "method3"`) since divergences can differ by axis.

5. Container + CI strategy: vLLM and SGLang run in separate containers
   today. Either (a) run the test orchestrator in a third container that
   talks to both via Docker network, or (b) gate Method 3 to whichever
   container's server is local. Long startup (~30s/server) means
   session-scoped boot. CI cost: ~1-2 min boot + per-test HTTP latency.

6. Streaming variant (separate sub-track): same servers, SSE responses,
   `parse_streaming_increment` semantics. Cross-impl streaming-chunk
   boundary divergences are likely abundant — Method 2 can't reach them.

Estimated effort
----------------
2-3 weeks for non-streaming Method 3 across vLLM and SGLang. Stack:

    PR-A   _serve.py session fixture (vLLM only)
    PR-B   vllm e2e wrapper + first 3-5 fixtures green
    PR-C   sglang variant
    PR-D   full coverage parity with Method 2
    PR-E   streaming variant (parallel track)
"""
