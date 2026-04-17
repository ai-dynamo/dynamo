---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Kimi K2 Tool Calling: Reference Parser Bug"
---

## Reference Python implementation

Moonshot publishes a reference tool-call parser in the Kimi-K2-Instruct
HuggingFace repo:

- **File:** [`docs/tool_call_guidance.md`](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/docs/tool_call_guidance.md)
- **Also adopted by:** vLLM (`vllm/entrypoints/openai/tool_parsers/kimi_k2_tool_parser.py`)
  and SGLang (`sglang/srt/openai_api/kimi_k2_tool_parser.py`), both of which
  copy the same extraction logic.

## Current extraction logic (problematic)

This is the actual `extract_tool_call_info` function from Moonshot's reference,
copied verbatim from `tool_call_guidance.md`:

```python
def extract_tool_call_info(tool_call_rsp: str):
    if '<|tool_calls_section_begin|>' not in tool_call_rsp:
        return []
    import re
    pattern = r"<\|tool_calls_section_begin\|>(.*?)<\|tool_calls_section_end\|>"

    tool_calls_sections = re.findall(pattern, tool_call_rsp, re.DOTALL)

    func_call_pattern = r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w\.]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>.*?)\s*<\|tool_call_end\|>"
    tool_calls = []
    for match in re.findall(func_call_pattern, tool_calls_sections[0], re.DOTALL):
        function_id, function_args = match
        function_name = function_id.split('.')[1].split(':')[0]
        tool_calls.append(
            {
                "id": function_id,
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": function_args
                }
            }
        )
    return tool_calls
```

The outer regex on line 5 requires **both** `<|tool_calls_section_begin|>` and
`<|tool_calls_section_end|>` to be present for `re.findall` to match anything.

## Why this is problematic

When the model hits `max_tokens`, emits EOS, or is stopped by a stop sequence,
the output is truncated *before* the `<|tool_calls_section_end|>` marker is
generated. In this case:

1. `re.findall` on line 5 returns `[]` — zero matches.
2. Line 10 (`tool_calls_sections[0]`) would IndexError, but in practice the
   caller checks `len(tool_calls) == 0` before using the result, so the bug
   silently returns an empty list.
3. **All** complete individual tool calls inside the section are silently
   dropped, even when they have valid `<|tool_call_begin|>` + arguments +
   `<|tool_call_end|>` markers.
4. The raw special-token text is returned as normal content, which is useless
   to the caller and breaks downstream tool-execution logic.

This manifests as **TC-001 (tool call missing)** in customer reports: the model
clearly intended to call a tool, but the API response contains
`tool_call_chunks=0` and `finish_reason=length`.

## Proposed fix (Python)

The only change is on line 5 — the outer regex gets a `(?:...|$)` fallback so a
missing `section_end` is treated as "section extends to end-of-string". The
inner `func_call_pattern` (line 8) is unchanged and continues to require
`<|tool_call_end|>`, so truly truncated individual calls are still discarded.

```python
def extract_tool_call_info(tool_call_rsp: str):
    if '<|tool_calls_section_begin|>' not in tool_call_rsp:
        return []
    import re
    pattern = r"<\|tool_calls_section_begin\|>(.*?)(?:<\|tool_calls_section_end\|>|$)"  # CHANGED

    tool_calls_sections = re.findall(pattern, tool_call_rsp, re.DOTALL)
    if not tool_calls_sections:  # ADDED: guard against empty match
        return []

    func_call_pattern = r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w\.]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>.*?)\s*<\|tool_call_end\|>"
    tool_calls = []
    for match in re.findall(func_call_pattern, tool_calls_sections[0], re.DOTALL):
        function_id, function_args = match
        function_name = function_id.split('.')[1].split(':')[0]
        tool_calls.append(
            {
                "id": function_id,
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": function_args
                }
            }
        )
    return tool_calls
```

Two changes from the original:

1. **Line 5:** `(?:<\|tool_calls_section_end\|>|$)` — falls back to
   end-of-string when `section_end` is absent.
2. **Line 7:** `if not tool_calls_sections: return []` — the original blindly
   indexes `tool_calls_sections[0]`, which would IndexError if the early-return
   guard on line 2 were ever bypassed.

## Dynamo (Rust) status

This fix is implemented in `lib/parsers/src/tool_calling/xml/kimi_k2_parser.rs`
(`extract_tool_calls`). The streaming jail in `jail.rs` also needed a
corresponding change: `find_tool_call_end_position_kimi_k2` returns `None` when
`section_end` is missing, preventing premature early-exit that would swallow
subsequent parallel tool calls before the stream ends.

Covered by unit tests (`test_parse_malformed_no_section_end`,
`test_parse_multiple_calls_no_section_end`,
`test_parse_complete_plus_truncated_no_section_end`) and streaming integration
tests (`test_kimi_k2_streaming_missing_section_end_max_tokens`,
`test_kimi_k2_streaming_multiple_calls_missing_section_end`).

## TODO — upstream

- [ ] Open a PR / HuggingFace Discussion on
      [`moonshotai/Kimi-K2-Instruct`](https://huggingface.co/moonshotai/Kimi-K2-Instruct)
      to fix `tool_call_guidance.md`.
- [ ] Open a PR against vLLM (`kimi_k2_tool_parser.py`) with the same regex
      fix.
- [ ] Open a PR against SGLang (`kimi_k2_tool_parser.py`) with the same regex
      fix.
