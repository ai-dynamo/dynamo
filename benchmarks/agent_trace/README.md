# Agent Trace Utilities

Utilities for working with Dynamo agent trace files emitted by
`DYN_AGENT_TRACE_SINKS=jsonl` or `jsonl_gz`.

## Convert to Perfetto

```bash
python3 benchmarks/agent_trace/convert_to_perfetto.py \
  "/tmp/dynamo-agent-trace.*.jsonl.gz" \
  --output /tmp/dynamo-agent-trace.perfetto.json
```

Open the output JSON in [Perfetto UI](https://ui.perfetto.dev/).

Inputs may be `.jsonl`, `.jsonl.gz`, a directory containing trace shards, or a
glob pattern. The converter emits Chrome Trace Event JSON:

- one workflow per Perfetto process
- one program lane per Perfetto thread
- one LLM request slice per Dynamo `request_end`
- prefill wait, prefill, and decode stage slices stacked under the request by
  default
- one tool slice per harness `tool_end`/`tool_error` when tool duration is
  available; otherwise the converter pairs `tool_start` with the terminal tool
  event when both records are present
- optional first-token markers with `--include-markers`

Use `--no-stages` for a compact request-only view. Use
`--separate-stage-tracks` to place stage slices on adjacent stage tracks when
debugging Perfetto nesting or label rendering.

Stage slice boundaries are normalized to avoid same-thread overlap caused by
independent metric rounding. Raw timing fields remain available in event args.

## Convert to Mooncake Replay

Agent traces with request replay hashes can be converted to Mooncake JSONL for
the Dynamo replay/mocker path:

```bash
cargo run -p dynamo-bench --bin agent_trace_to_mooncake -- \
  --input-path /tmp/dynamo-agent-trace.*.jsonl.gz \
  --output-file /tmp/dynamo-agent-trace.mooncake.jsonl
```

The converter accepts `.jsonl`, `.jsonl.gz`, repeated `--input-path` flags, and
recorder-envelope records of the form `{"timestamp": ..., "event": ...}`. It
groups turns by
`workflow_id:program_id`, emits `timestamp` on the first turn in each session,
and emits `delay` on later turns based on the gap from the previous request end
to the next request start. Stable trace `input_sequence_hashes` are compacted to
Mooncake `hash_ids` during conversion.

Aggregate mocker replay uses each row's `input_length`, `output_length`, and the
next turn's `hash_ids` to infer any full KV blocks materialized by decode. This
models same-session prefix reuse across agent turns without storing output token
IDs or prompt/response text.

Replay the output with the same block size recorded in the converter output:

```bash
python -m dynamo.replay \
  --trace-file /tmp/dynamo-agent-trace.mooncake.jsonl \
  --trace-format mooncake \
  --trace-block-size 64
```

## Validate Converter

The converter has a local self-check that is intentionally not wired into the
main pytest suite:

```bash
python3 benchmarks/agent_trace/validate_convert_to_perfetto.py
```
