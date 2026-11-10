<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

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
request can be replayed by Dynamo mock workers. Save `/tmp/request-trace-prediction.yaml` with the
original JSONL or JSONL.GZ shards:

```yaml
traffic:
  source:
    type: trace
    format: dynamo
    paths: [/tmp/dynamo-request-trace.0000.jsonl.gz]
  load: {type: trace_timestamps, speedup: 1.0}
engine:
  mode: aggregated
  model: meta-llama/Meta-Llama-3.1-8B-Instruct
  hardware: h200_sxm
  backend: vllm
  context_length: max
  workers:
    aggregated:
      parallelism: {replicas: 4, tensor: 1, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
      scheduler: {max_batched_tokens: 8192, max_sequences: 256}
      kv_cache: {block_size: 64, prefix_caching: true, capacity: {type: default, memory_fraction: 0.9}}
      timing: {type: default}
router:
  policy: kv_router
  prefill_load_model: {type: none}
planner: {policy: disabled}
```

Add every shard to `traffic.source.paths`, then run the offline prediction:

```bash
aisimulate predict \
  --stack dynamo \
  --config /tmp/request-trace-prediction.yaml \
  --output-dir /tmp/request-trace-prediction
```

No format conversion or intermediate Mooncake file is required.

AISimulate derives the trace block size from the request records and rejects mixed
block sizes across shards. Context-free traces use standard replay. Traces in
which every request has `agent_context` use agentic replay. Mixed traces are
rejected.

`router.policy: kv_router` requires more than one mock worker. For a single aggregated-worker
sanity check, set `router.policy: round_robin` and
`engine.workers.aggregated.parallelism.replicas: 1`.

## Validate Converter

The converter has a local self-check that is intentionally not wired into the
main pytest suite:

```bash
python3 benchmarks/request_trace/validate_convert_to_perfetto.py
```

## Join One Request Across nsys Reports and OTel

`nsys_request_timeline.py` answers "what happened to this one request?" when
Dynamo processes were captured with `DYN_NVTX=1` under Nsight Systems (one
report per process or per node, possibly on different hosts) and, optionally,
traces were exported to Tempo. Request-scoped NVTX ranges and markers use a
static stage name as their message (for example
`kv_router.route_request:prefill`) and carry the request in the NVTX 64-bit
payload (`NVTX_EVENTS.uint64Value`), `rid_key(rid)`: FNV-1a-64 of the UTF-8
request id with the top bit cleared. Each process also emits identity marks
`dynamo.request rid=<rid>[ trace=..][ xrid=..][ engine_rid=..]` with the same
payload, which map the key back to the request id, trace id and x-request-id.
The tool looks the request up by key, joins it with the request's OTel spans,
and prints one cross-process timeline.

```bash
python3 benchmarks/request_trace/nsys_request_timeline.py \
  frontend.nsys-rep prefill.nsys-rep decode.nsys-rep \
  --request chatcmpl-11111111-2222-3333-4444-555555555555 \
  --tempo http://localhost:3200
```

- `--request` takes a request id, `chatcmpl-<id>`, a W3C trace id (32 hex), or
  an `x-request-id`. A request id (or `chatcmpl-<id>`) is hashed to its payload
  key directly; a trace id or x-request-id is looked up in the identity marks
  (the x-request-id is sanitized like the runtime does) and resolved to its
  request id. The tool reports how the key was resolved and fails if it matches
  no request or several.
- Inputs are `.sqlite` exports, `.nsys-rep` files (exported with
  `nsys export --type sqlite`, selectable with `--nsys`, reusing an export that
  is newer than the report), directories, or globs. Use a plain export, not
  `--ts-normalize`.
- OTel comes from `--tempo URL` or `--otel-json FILE` (a saved Tempo v1/v2
  response or OTLP JSON). Tempo only keeps sampled traces for a limited time
  (about 1h in the local stack), so a missing trace is reported as a warning.
- Output sections: resolution summary, chronological table (relative time,
  duration, process, stage, source `nvtx`/`otel`/`both`, OTel-minus-NVTX start
  delta), TTFT breakdown with unattributed gaps, and per-report zoom windows
  such as `zoom to 12.3456s-12.5123s in decode.sqlite` (report-relative seconds,
  as shown in the Nsight timeline). `open` durations are ranges that never
  closed in the capture.
- `--json` emits the same data as JSON; `--perfetto OUT` writes Chrome Trace
  JSON with one process row per captured process plus one per OTel service.
- Paired OTel/NVTX start deltas are microseconds on one host; larger values
  across hosts reveal clock skew between the machines.

### Per-stage summary across all requests

```bash
python3 benchmarks/request_trace/nsys_request_timeline.py \
  frontend.sqlite prefill.sqlite decode.sqlite --summary [--json]
```

`--summary` needs no `--request`. It prints, per stage, the count, mean, median,
p99 (nearest rank) and max duration in milliseconds over every closed
request-scoped range (a range with a payload) in the given reports. Marks, static
ranges (no payload) and ranges that never closed are not aggregated. Because the
names are static, `nsys stats --report nvtx_sum report.nsys-rep` now aggregates
these ranges natively too, one row per stage instead of one per request.

### Raw-trace queries

The same data is directly queryable in the SQLite export:

```sql
-- per-stage aggregation
SELECT text AS stage, count(*) AS n, avg("end" - start) / 1e6 AS avg_ms
FROM NVTX_EVENTS WHERE eventType = 60 AND uint64Value IS NOT NULL
  AND "end" <= (SELECT duration FROM ANALYSIS_DETAILS)   -- skip ranges that never closed
GROUP BY text;

-- request id -> key
SELECT DISTINCT uint64Value FROM NVTX_EVENTS
WHERE eventType = 34
  AND (text = 'dynamo.request rid=<rid>' OR instr(text, 'dynamo.request rid=<rid> ') = 1);

-- every event of that request
SELECT text, start, "end", globalTid FROM NVTX_EVENTS WHERE uint64Value = :key ORDER BY start;
```

Ranges registered through `StringIds` have `text` NULL; join `StringIds` on
`textId` (the tool does this automatically).

Nsys exports embed the profiled process environment, which can contain secrets;
treat them like logs before sharing.

Run the tests (no network, synthetic nsys SQLite fixtures):

```bash
python3 -m pytest -m unit benchmarks/request_trace/tests
```
