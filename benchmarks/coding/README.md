# Coding Trace Export

Privacy-preserving exporters for coding-agent traces.

## Current Scope

The canonical Claude exporter lives under:

- [`benchmarks/coding/claude/export_trace.py`](/Users/peabrane/Documents/codes/dynamo/benchmarks/coding/claude/export_trace.py)
- [`benchmarks/coding/claude/discovery.py`](/Users/peabrane/Documents/codes/dynamo/benchmarks/coding/claude/discovery.py)
- [`benchmarks/coding/claude/parser.py`](/Users/peabrane/Documents/codes/dynamo/benchmarks/coding/claude/parser.py)

## What It Produces

For each exported run, the Claude exporter writes two files:

- Mooncake JSONL:
  standard benchmark rows with `session_id`, `input_length`, `output_length`, `hash_ids`, and `timestamp` or `delay`
- Sidecar JSONL:
  text-free structural metadata such as context shape, tool counts, and nested progress-derived timing

Raw prompt text is used only in memory to derive hashes. It is not written to disk.

## Usage

Use the project venv:

Default behavior:

- if `--input-path` is omitted, the exporter walks upward from `benchmarks/coding/claude/` to filesystem root
- for each ancestor, it checks the matching encoded Claude project path under `~/.claude/projects/`
- after that, it also scans the normal Claude home-level root `~/.claude/projects`

Autodiscover traces and export:

```bash
.venv/bin/python benchmarks/coding/claude/export_trace.py \
  --output-file /tmp/claude_trace.jsonl
```

Optional `--input-path` overrides autodiscovery and restricts the export to a specific file or directory:

```bash
.venv/bin/python benchmarks/coding/claude/export_trace.py \
  --input-path /Users/peabrane/.claude/projects/-Users-peabrane-Documents-codes-dynamo \
  --output-file /tmp/claude_trace.jsonl
```

Custom tokenizer and block size:

```bash
.venv/bin/python benchmarks/coding/claude/export_trace.py \
  --input-path /Users/peabrane/.claude/projects/-Users-peabrane-Documents-codes-dynamo \
  --tokenizer deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --block-size 64 \
  --output-file /tmp/claude_trace.jsonl
```

Anonymize session ids:

```bash
.venv/bin/python benchmarks/coding/claude/export_trace.py \
  --input-path /Users/peabrane/.claude/projects/-Users-peabrane-Documents-codes-dynamo \
  --anonymize-session-id \
  --output-file /tmp/claude_trace.jsonl
```

## Discovery

If `--input-path` is provided, autodiscovery is skipped.

If `--input-path` is omitted, discovery does this:

1. Start from the Claude exporter script directory.
2. Walk every ancestor directory upward until filesystem root.
3. For each ancestor, look for the encoded Claude project directory under `~/.claude/projects/<encoded-absolute-path>`.
4. Also scan the normal Claude home-level root `~/.claude/projects`.

Ignored during discovery:

- `history.jsonl`
- anything under `subagents/`

## Parsing Rules

The exporter currently:

- parses top-level non-sidechain `user`, `assistant`, and `system` rows
- groups adjacent assistant fragments by `requestId` or `message.id`
- excludes `thinking` and `redacted_thinking`
- resets transcript state on `compact_boundary`
- starts post-compaction turns from the injected `isCompactSummary` row
- skips local command wrapper noise such as `<local-command-caveat>`, `<local-command-stdout>`, and command wrapper rows
- preserves top-level tool use and tool result structure in hashed text form
- mines `progress` rows for text-free sidecar metrics

## Output Semantics

Timing:

- first turn in each session gets `timestamp`
- later turns get `delay`
- source Claude timestamps are parsed as UTC and normalized to millisecond replay timing

Prompt representation:

- each Mooncake row represents the full prompt prefix for that assistant turn
- after compaction, future rows are built from the compact summary forward, not from the pre-compact raw history
