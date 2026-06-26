# Request Trace Utilities

Utilities for working with Dynamo `dynamo.request.trace.v1` files emitted by
`DYN_REQUEST_TRACE_SINKS=jsonl` or `jsonl_gz`.

## Convert to Perfetto

```bash
python3 benchmarks/request_trace/convert_to_perfetto.py \
  "/tmp/dynamo-request-trace.*.jsonl.gz" \
  --output /tmp/dynamo-request-trace.perfetto.json
```

Open the output JSON in [Perfetto UI](https://ui.perfetto.dev/).

Inputs may be `.jsonl`, `.jsonl.gz`, a directory containing trace shards, or a
glob pattern. The converter emits Chrome Trace Event JSON:

- one request trace per Perfetto process
- one session lane per Perfetto thread
- one LLM request slice per Dynamo `request_end`
- prefill wait, prefill, and decode stage slices stacked under the request by
  default
- one tool slice per harness `tool_end`/`tool_error`; explicit
  `started_at_unix_ms`/`ended_at_unix_ms` are preferred, then `duration_ms`,
  then paired `tool_start` timing when both records are present
- inferred tool slices from request `finish_reason_metadata.tool_calls` when
  no matching tool event was traced; when the next request in the same session
  starts after the tool-call response, the slice spans that gap, otherwise the
  converter emits a short `duration_unknown` synthetic slice at the response
  end, including when there is no following request in the same session
- finish metadata on request slices, including final finish reason, backend
  finish reason, stop reason, per-choice finish summaries, and complete
  tool-call names
- optional first-token markers with `--include-markers`

Use `--no-stages` for a compact request-only view.

Stage slice boundaries are normalized to avoid same-thread overlap caused by
independent metric rounding. Raw timing fields remain available in event args.

## Replay Dynamo Request Traces

Traces captured with Dynamo request tracing include replay hashes whenever a
request can be replayed by Dynamo mock workers. Pass the original JSONL or
JSONL.GZ shards directly to the replay harness:

```bash
python -m dynamo.replay /tmp/dynamo-request-trace.*.jsonl.gz \
  --trace-format dynamo \
  --replay-mode offline \
  --router-mode kv_router \
  --num-workers 4 \
  --report-json /tmp/dynamo-request-trace.replay-report.json
```

Replay derives the trace block size from the request records and rejects mixed
block sizes across shards. If you pass `--trace-block-size`, its value must
match the embedded value. Context-free traces use standard replay. Traces in
which every request has `agent_context` use agentic replay. Mixed traces are
rejected.

`kv_router` requires more than one mock worker. For a single aggregated-worker
sanity check, use `--router-mode round_robin --num-workers 1`.

## Validate Converter

The converter has a local self-check that is intentionally not wired into the
main pytest suite:

```bash
python3 benchmarks/request_trace/validate_convert_to_perfetto.py
```
