<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Coding Trace Export

Rust-native exporters for privacy-preserving coding-agent traces.

## Claude Exporter

The Claude exporter lives in `dynamo-bench` and is invoked as:

```bash
cargo run -p dynamo-bench --bin claude_trace_export -- \
  --output-file /tmp/claude_trace.jsonl
```

That command writes two files:

- Mooncake JSONL: benchmark rows with `session_id`, `input_length`, `output_length`, `hash_ids`, and `timestamp` or `delay`
- Sidecar JSONL: text-free structural metadata such as context shape, top-level tool calls, and nested progress-derived timing

The sidecar path is derived from the output path by inserting `.sidecar` before the extension.

## Default Discovery

If `--input-path` is omitted, the exporter:

1. Starts from the current working directory.
2. Walks every ancestor directory upward to `/`.
3. For each ancestor, checks the matching encoded Claude project directory under `~/.claude/projects/<encoded-absolute-path>`.
4. Also scans the home-level Claude root `~/.claude/projects`.

Ignored during discovery:

- `history.jsonl`
- anything under `subagents/`

## Optional Input Override

Use `--input-path` to restrict the export to a specific file or directory:

```bash
cargo run -p dynamo-bench --bin claude_trace_export -- \
  --input-path ~/.claude/projects/<encoded-path> \
  --output-file /tmp/claude_trace.jsonl
```

`--input-path` may point to:

- a specific Claude session JSONL file
- an encoded Claude project directory under `~/.claude/projects`
- a repo root whose encoded Claude project directory should be used
- a directory containing Claude session JSONL files

## Important Flags

```bash
cargo run -p dynamo-bench --bin claude_trace_export -- \
  --input-path ~/.claude/projects/<encoded-path> \
  --tokenizer deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --block-size 64 \
  --delta-overlap-words 50 \
  --tokenizer-workers 8 \
  --output-file /tmp/claude_trace.jsonl
```

- `--anonymize-session-id`: replace Claude session IDs with stable anonymized IDs
- `--delta-overlap-words`: approximate tokenization by re-tokenizing only the final `N` words of the previous prompt plus the new delta; default is `50`
- `--tokenizer-workers`: number of worker threads used for session-parallel tokenization

## Parsing Semantics

The exporter:

- uses top-level non-sidechain `user`, `assistant`, and `system` rows for the main transcript
- groups adjacent assistant fragments by `requestId`, then `message.id`
- excludes `thinking` and `redacted_thinking`
- resets transcript state on `compact_boundary`
- starts post-compaction turns from the injected `isCompactSummary` row
- skips local command wrapper noise such as `<local-command-caveat>`, `<local-command-stdout>`, and command wrapper rows
- preserves top-level tool-use and tool-result structure in hashed text form
- mines `progress` rows for text-free sidecar metrics only

## Output Semantics

- The first turn in each session gets `timestamp`.
- Later turns get `delay`.
- Source Claude timestamps are parsed as UTC and normalized to millisecond replay timing.
- Each Mooncake row represents the full prompt prefix for that assistant turn.
- After compaction, future rows are built from the compact summary forward, not from pre-compact raw history.
- Rows are written incrementally as turns are merged across sessions.
