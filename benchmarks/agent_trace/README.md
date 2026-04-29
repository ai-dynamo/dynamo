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
- optional first-token markers with `--include-markers`

Use `--no-stages` for a compact request-only view. Use
`--separate-stage-tracks` to place stage slices on adjacent stage tracks when
debugging Perfetto nesting or label rendering.

Stage slice boundaries are normalized to avoid same-thread overlap caused by
independent metric rounding. Raw timing fields remain available in event args.
