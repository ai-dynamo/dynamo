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
- one tool slice per harness `tool_end`/`tool_error`; explicit
  `started_at_unix_ms`/`ended_at_unix_ms` are preferred, then `duration_ms`,
  then paired `tool_start` timing when both records are present
- optional first-token markers with `--include-markers`

Use `--no-stages` for a compact request-only view. Use
`--separate-stage-tracks` to place stage slices on adjacent stage tracks when
debugging Perfetto nesting or label rendering.

Stage slice boundaries are normalized to avoid same-thread overlap caused by
independent metric rounding. Raw timing fields remain available in event args.

## Validate Converter

The converter has a local self-check that is intentionally not wired into the
main pytest suite:

```bash
python3 benchmarks/agent_trace/validate_convert_to_perfetto.py
```
