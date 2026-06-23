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

- one session per Perfetto process
- one trajectory lane per Perfetto thread
- one LLM request slice per Dynamo `request_end`
- prefill wait, prefill, and decode stage slices stacked under the request by
  default
- one tool slice per harness `tool_end`/`tool_error`; explicit
  `started_at_unix_ms`/`ended_at_unix_ms` are preferred, then `duration_ms`,
  then paired `tool_start` timing when both records are present
- inferred tool slices from request `finish_reason_metadata.tool_calls` when
  no matching tool event was traced; when the next same-trajectory request
  starts after the tool-call response, the slice spans that gap, otherwise the
  converter emits a short `duration_unknown` synthetic slice at the response
  end, including when there is no following same-trajectory request
- finish metadata on request slices, including final finish reason, backend
  finish reason, stop reason, per-choice finish summaries, and complete
  tool-call names
- optional first-token markers with `--include-markers`

Use `--no-stages` for a compact request-only view.

Stage slice boundaries are normalized to avoid same-thread overlap caused by
independent metric rounding. Raw timing fields remain available in event args.

## Convert to Mooncake Replay

Traces captured with Dynamo request tracing include replay hashes whenever a
request can be represented as one Mooncake replay row and can be converted to
Mooncake JSONL for the Dynamo replay/mocker path:

```bash
cargo run -p dynamo-bench --bin request_trace_to_mooncake -- \
  --input-path /tmp/dynamo-request-trace.*.jsonl.gz \
  --output-file /tmp/dynamo-request-trace.mooncake.jsonl
```

The converter accepts `.jsonl`, `.jsonl.gz`, repeated `--input-path` flags, and
recorder-envelope records of the form `{"timestamp": ..., "event": ...}`. It
emits one independent Mooncake request row per Dynamo `request_end`, with an
absolute `timestamp` from Dynamo's request-arrival time and no `session_id`.
Stable trace `input_sequence_hashes` are compacted to Mooncake `hash_ids`
during conversion.

Context-free `dynamo.request.trace.v1` rows convert to ordinary Mooncake rows.
Rows with `agent_context` can be converted with `--agentic`; context-free rows
are rejected with `--agentic` because the converter cannot infer trajectories.

Mooncake replay intentionally ignores non-replay request fields such as
`finish_reason_metadata`. Use the Perfetto converter when you want to inspect
whether a request stopped for tool calls, hit a backend stop reason, or emitted
complete tool-call metadata.

For what replay covers today and what remains on the roadmap (cache movement
fidelity, output token reconstruction, causal tool/turn dependencies, end-to-end
agent re-run), see
[Replay Scope and Follow-ups](../../docs/agents/agent-tracing.md#replay-scope-and-follow-ups).

Replay the output with the same trace block size used when the trace was
captured. The converter prints this value after writing the Mooncake JSONL.
Use the same value for the mock engine block size when you want the replay hash
granularity to match the live backend page size.

```bash
TRACE_BLOCK_SIZE=128
uv run --no-sync python -m dynamo.replay /tmp/dynamo-request-trace.mooncake.jsonl \
  --trace-format mooncake \
  --trace-block-size "${TRACE_BLOCK_SIZE}" \
  --replay-mode offline \
  --router-mode kv_router \
  --num-workers 4 \
  --extra-engine-args "{\"block_size\":${TRACE_BLOCK_SIZE}}" \
  --report-json /tmp/dynamo-request-trace.replay-report.json
```

`kv_router` requires more than one mock worker. For a single aggregated-worker
sanity check, use `--router-mode round_robin --num-workers 1`.

## Existing Text Traces To Mooncake

There is no generic "arbitrary text trace to Mooncake" CLI today. Current
helpers are specialized:

- `request_trace_to_mooncake` converts Dynamo request traces that already contain
  replay hashes.
- `claude_trace_export` converts local Claude traces into privacy-preserving
  Mooncake JSONL plus a sidecar.
- `dynamo-data-gen` owns the shared `MooncakeRow`, `MooncakeJsonlWriter`,
  `RollingHashIdMapper`, and token-block hashing primitives used by exporters.
- AIPerf has related hash/trace helpers for benchmark workloads.

A generic converter should be a follow-up: parse JSONL/CSV/text logs, select the
text/timestamp/session/output-length fields, tokenize with a configured model,
hash token blocks, write Mooncake JSONL, and report the trace block size needed
for replay.

## Validate Converter

The converter has a local self-check that is intentionally not wired into the
main pytest suite:

```bash
python3 benchmarks/request_trace/validate_convert_to_perfetto.py
```
