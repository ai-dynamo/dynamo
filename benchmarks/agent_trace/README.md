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

## Convert to Mooncake Replay

Agent traces captured with Dynamo agent tracing include replay hashes by default
and can be converted to Mooncake JSONL for the Dynamo replay/mocker path. Set
`DYN_AGENT_TRACE_REPLAY_HASHES=0` only when you need to suppress replay metadata:

```bash
cargo run -p dynamo-bench --bin agent_trace_to_mooncake -- \
  --input-path /tmp/dynamo-agent-trace.*.jsonl.gz \
  --output-file /tmp/dynamo-agent-trace.mooncake.jsonl
```

The converter accepts `.jsonl`, `.jsonl.gz`, repeated `--input-path` flags, and
recorder-envelope records of the form `{"timestamp": ..., "event": ...}`. It
emits one independent Mooncake request row per Dynamo `request_end`, with an
absolute `timestamp` from Dynamo's request-arrival time and no `session_id`.
Stable trace `input_sequence_hashes` are compacted to Mooncake `hash_ids`
during conversion.

For what replay covers today and what remains on the roadmap (cache movement
fidelity, output token reconstruction, causal tool/turn dependencies, end-to-end
agent re-run), see
[Replay Scope and Follow-ups](../../docs/agents/agent-context.md#replay-scope-and-follow-ups).

Replay the output with the same trace block size used when the trace was
captured. The converter prints this value after writing the Mooncake JSONL.
Use the same value for the mock engine block size when you want the replay hash
granularity to match the live backend page size.

```bash
TRACE_BLOCK_SIZE=128
uv run --no-sync python -m dynamo.replay /tmp/dynamo-agent-trace.mooncake.jsonl \
  --trace-format mooncake \
  --trace-block-size "${TRACE_BLOCK_SIZE}" \
  --replay-mode offline \
  --router-mode kv_router \
  --num-workers 4 \
  --extra-engine-args "{\"block_size\":${TRACE_BLOCK_SIZE}}" \
  --report-json /tmp/dynamo-agent-trace.replay-report.json
```

`kv_router` requires more than one mock worker. For a single aggregated-worker
sanity check, use `--router-mode round_robin --num-workers 1`.

## Validate Converter

The converter has a local self-check that is intentionally not wired into the
main pytest suite:

```bash
python3 benchmarks/agent_trace/validate_convert_to_perfetto.py
```
