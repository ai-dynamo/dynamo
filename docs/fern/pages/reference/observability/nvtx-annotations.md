---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: NVTX Annotations
subtitle: Request-scoped NVTX ranges, the message format, the stage catalog, and nsys queries
---

Dynamo can annotate its own code with NVIDIA Tools Extension (NVTX) ranges and markers so that
Nsight Systems (`nsys`) shows where one request spent its time inside the Dynamo processes. Every
range that belongs to a request carries a 64-bit key derived from the request id, so you can start
from a request that misbehaved, such as a time-to-first-token (TTFT) spike, and find its frontend,
routing, queueing, prefill, decode, and response-streaming segments in each process's `nsys`
report.

This page is the reference. For the workflow, see
[Follow One Request into Nsight Systems](../../cli/operations/observability.mdx#follow-one-request-into-nsight-systems).

## What this covers

Request-scoped annotations answer "which Dynamo stage was slow for this request?" They cover only
code that Dynamo controls: HTTP handling, preprocessing, routing, worker admission, handler code,
and response publishing.

> [!NOTE]
> Engine-internal attribution is not covered. Scheduler decisions, individual kernels, and KV
> transfer inside vLLM, SGLang, TensorRT-LLM, or NIXL are not tagged with the request id. The
> `worker.engine.first_output` range tells you how long the engine took from the generate call to
> its first output, but not why. Attributing engine internals to a request is future work.

## Enable annotations

One variable switches on every NVTX annotation in both the Rust runtime and the Python layer:

| Variable | Default | Description |
| --- | --- | --- |
| `DYN_NVTX` | unset (off) | Enable NVTX annotations. |

Accepted values are case-insensitive and whitespace-trimmed:

| Meaning | Values |
| --- | --- |
| On | `1`, `true`, `on`, `yes` |
| Off | `0`, `false`, `off`, `no`, empty, or unset |

Any other value, such as `DYN_NVTX=maybe`, is rejected at startup rather than read as off, so a
profiling run cannot half-start. Set the variable on every process you want to profile.

Shipped wheels and containers are built with the `nvtx` Cargo feature. Dynamo compiles a small C
shim against vendored NVTX v3 headers; the profiler's injection library is loaded at runtime, so
there is no `libnvToolsExt` dependency. A source build needs the feature explicitly:

```bash
cargo build --profile profiling --features dynamo-runtime/nvtx
```

The Python `nvtx` package is needed only for the older static Python ranges (the `nvtx_utils`
decorators and context managers, for example in multimodal loading). Request-scoped ranges are
emitted by the Rust runtime through `dynamo._core` and work without the package.

With `DYN_NVTX` off, nothing is emitted. If the `nvtx` feature is compiled in, each annotation site
costs one inlined atomic load and a branch. If it is not compiled in, the sites compile to nothing.

## What gets annotated

| Kind | Source | Request key | Example |
| --- | --- | --- | --- |
| Static ranges | Rust `dynamo_nvtx_range!` and similar macros; Python `nvtx_utils` decorators | No | `preprocess.tokenize`, `detokenize`, `pybridge.invoke_generator` |
| Mirrored spans | A `tracing` layer that copies every INFO-or-above span belonging to a request (as enabled by `DYN_LOG`) into an NVTX start/end range | Yes, in the payload | `http-request`, `kv_router.schedule` |
| Explicit markers and ranges | Hand-placed Rust sites and Python handler code | Yes, in the payload | `frontend.first_token`, `route.kv_match`, `worker.egress.chunk` |
| Identity marks | The mirror and explicit sites, once per request per process | Yes, in the payload | `dynamo.request rid=<id> trace=<trace-id>` |

The mirror is installed only when `DYN_NVTX` is on. It reads the request id from the span's own
`request_id` or `dynamo.request.id` field or inherits it from an ancestor span, and ignores spans
without one. It uses the same filter as console logging, so it mirrors the spans that `DYN_LOG`
enables: the default `info` covers every stage in the catalog, and the `http-request`,
`handle_payload`, and route spans are always on. A stricter setting such as `DYN_LOG=warn` drops the
lifecycle and `kv_router.*` stages from the mirror. It does not touch OpenTelemetry export.

Static ranges have a fixed name and no payload. They are useful for whole-process profiles but
cannot be attributed to a request. Request-scoped ranges also have fixed names (see the next
section), so both kinds aggregate per name in `nsys` summaries.

## Message format

Request-scoped events are in the default NVTX domain. The request is carried in the NVTX 64-bit
payload, not in the name.

**Range or marker.** The NVTX message is only the stage name, for example
`kv_router.route_request:prefill`, `worker.engine.first_output:decode`, or `frontend.first_token`.
A stage is a span or marker name, optionally followed by a lowercase `:<qualifier>`. The event's
unsigned 64-bit payload is the **request key**. In the `nsys` SQLite export these are
`NVTX_EVENTS.text` and `NVTX_EVENTS.uint64Value`.

**Request key.** `rid_key(rid)` is the FNV-1a 64-bit hash of the UTF-8 request id with the top bit
cleared (offset basis `0xcbf29ce484222325`, prime `0x100000001b3`, result masked with
`0x7FFFFFFFFFFFFFFF`). Clearing the top bit keeps the value a positive SQLite integer. Any request
id string works. Dynamo's Rust and Python code pin the same test vectors, so both sides agree.

```python
def rid_key(rid: str) -> int:
    h = 0xCBF29CE484222325
    for b in rid.encode("utf-8"):
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h & 0x7FFFFFFFFFFFFFFF
```

**Identity mark.** Each process also emits instant markers (`eventType` 34) with the same payload
that map the key back to the identifiers:

```text
dynamo.request rid=<rid>[ trace=<trace-id>][ xrid=<x-request-id>][ engine_rid=<engine-id>]
```

- `rid` is the request id Dynamo assigns, the same value the client sees as `chatcmpl-<rid>` in the
  response `id`, and the same value OpenTelemetry spans carry as `request_id` or
  `dynamo.request.id`.
- `trace` is the 32-character lowercase hex W3C trace id, present when the client sent a
  `traceparent`. `xrid` is the client's `x-request-id`. `engine_rid` is the engine's own request id
  when it differs from `rid`. Other attributes may follow; a parser must ignore unknown keys.
- Tokens are separated by one space. In values, whitespace becomes `_`, control characters are
  dropped, values are cut to 128 bytes, and empty values are omitted.

The tracing mirror emits an identity mark when a span takes the request id from its own field and
its parent does not already carry it. In practice that is once for `http-request` on the frontend
and once for `handle_payload` on each worker. An explicit marker or range emits an extra identity
mark only when it carries attributes, such as `engine_rid`. A request can therefore have several
identity marks in one process; merge their attributes.

Example identity mark and a range that carries its key:

```text
dynamo.request rid=6f1c2b0e-4a57-4c0c-9b3f-0d5e8a1b7c42 trace=4bf92f3577b34da6a3ce929d0e0e4736 xrid=slow-ttft-001
worker.engine.first_output:decode        (payload = rid_key of the same rid)
```

> [!NOTE]
> In the Nsight Systems GUI, request-scoped range names no longer show the request id. To find a
> request, search the NVTX rows for the identity mark `dynamo.request rid=<id>`, and read the key
> from a range's payload in its tooltip. `trace=` and `xrid=` appear only on the identity mark, not
> on range names.

## Stage catalog

These are the stage names that appear as the NVTX message; each event also carries the request key
in its payload. Stages marked with an asterisk appear only when `DYN_LIFECYCLE_TRACE_ENABLED=1` is
set on that process. A mirrored span is a range; a marker is an instant. Any other INFO-level span created under
a request span is also mirrored, so a profile can contain stages not listed here.

### Frontend process

| Stage | Type | Meaning |
| --- | --- | --- |
| `http-request` | Range | Root HTTP span for the request. Its identity mark carries `trace` and `xrid` when the client sent them. |
| `request.lifecycle`*, `request.preprocessing`*, `response.streaming`* | Range | Lifecycle, preprocessing, and response-streaming phases. |
| `kv_router.select_worker` | Range | Worker selection, including the child stages below. |
| `kv_router.compute_block_hashes`, `kv_router.compute_seq_hashes`, `kv_router.find_matches`, `kv_router.shared_cache_check` | Range | Prefix hashing and cache matching inside selection. |
| `kv_router.schedule` | Range | Scheduling decision. Includes time waiting in the scheduler queue. |
| `route.select_worker`, `route.kv_match` | Range | The KV router's own ranges for the whole selection and for the overlap match inside it. They cover the same work as `kv_router.select_worker` and its children, but do not depend on `DYN_LOG`. |
| `kv_router.route_request:<phase>` | Range | Dispatch to a worker through the end of that leg's response stream. One per leg, for example `prefill` and `decode`. |
| `router.route_request` | Range | The equivalent for non-KV router modes. It has no phase qualifier; the join tool names each leg after the worker it reached. |
| `transport.tcp.request`, `transport.nats.request` | Range | Sending one leg to a worker over that request plane, until the worker's response stream is established. |
| `transport.tcp.send` | Range | Sending the request over TCP until the worker acknowledges it. |
| `transport.response.wait_backend` | Range | Waiting for the worker to open its response stream. |
| `frontend.first_token` | Marker | The frontend observes the first output token. This is the TTFT point. |

### Worker processes

| Stage | Type | Meaning |
| --- | --- | --- |
| `handle_payload` | Range | Worker-side root span. Its identity mark carries `trace` and `xrid` when propagated. |
| `worker.ingress.request` | Range | From the request's arrival at the worker to its last response frame. |
| `worker.ingress.parse_request`, `worker.ingress.open_response_stream` | Range | Decoding the request, and opening the response stream back to the frontend. |
| `worker.ingress.generate` | Range | Dispatch into the engine. For a Python worker this is the call that builds the response generator, not token generation. |
| `worker.ingress.prologue_sent` | Marker | The worker told the frontend that the response stream is live. |
| `worker.admission`*, `worker.operation`*, `request.dispatch`*, `response.streaming.<role>`* | Range | Worker lifecycle phases. `worker.operation` may carry a `.<role>` suffix. |
| `worker.admission_gate.wait` | Range | Time the request queued in the backend admission gate before the handler ran. |
| `worker.handler.prepare:<role>` | Range | From handler entry to the engine generate call. |
| `worker.engine.first_output:<role>` | Range | From the engine generate call to the first engine output, or to handler exit if none. An extra identity mark carries `engine_rid` when the engine's id differs from `rid`. |
| `worker.egress.first_chunk` | Marker | The first response chunk is published back to the frontend. |
| `worker.egress.chunk`, `worker.egress.encode`, `worker.egress.publish` | Range | One per response chunk: the worker's egress work, and its encode and publish parts. The gap between consecutive `worker.egress.chunk` ranges is the engine producing the next output. |

`<role>` is `prefill`, `decode`, `aggregated`, or `encode`. The handler ranges are emitted by the
vLLM, SGLang, and TensorRT-LLM handlers. The worker lifecycle stages need
`DYN_LIFECYCLE_TRACE_ENABLED=1` on both the frontend and the worker, and appear only for workers
registered through the Python bindings with a disaggregation role; the mocker emits
`handle_payload`, `worker.admission_gate.wait`, and the `worker.ingress.*` and `worker.egress.*`
stages only. `worker.ingress.request` and `worker.ingress.parse_request` take the key from the
request id the transport carries, so they have no key when it carries none.

## Cost

| State | Cost |
| --- | --- |
| `DYN_NVTX` off | Nothing is emitted. Compiled-in sites cost one atomic load and a branch. |
| `DYN_NVTX` on, no profiler attached | About 0.1 µs per range. |
| `DYN_NVTX` on, under `nsys` | About 1.3 µs per range, measured on mock workers at 4096 concurrent streams. |

Request-scoped names are static, so `nsys stats --report nvtx_sum` aggregates them natively: one
row per stage (for example `:http-request`; the leading `:` is how `nsys` prints the empty domain),
not one row per request.

**High concurrency.** One run per arm, with 4096 streams, a mock worker, the KV router, and
lifecycle spans on (about 535 requests/s and 190k tokens/s):

- With `DYN_NVTX=1` and no profiler, there was no measurable change in throughput or CPU.
- Under `nsys`, per-token ranges (about 154k/s on the frontend and 447k/s on the worker) dominated
  event volume and `nsys` overhead. This run predates the key on the worker's ingress and per-chunk
  egress ranges and on the router's transport and selection ranges. At the time, request-scoped
  events were 3.6% of frontend events (about 11 per request) and 0.4% of worker events. The
  worker's per-chunk ranges now carry the key, which raises that share without adding events.
- A full-length capture at that load buffered about 25 GB of raw `nsys` data in 2.5 minutes. A
  bounded 10-second capture produced 35 MB (frontend) and 85 MB (worker) reports.

These numbers were measured with an earlier name-based encoding. The event counts are the same with
the payload encoding; no byte sizes are claimed for it.

> [!TIP]
> At high concurrency, capture a window instead of the whole run, for example
> `nsys profile ... --delay=45 --duration=10 --kill=none`. `--kill=none` leaves the process running
> when the window closes.

## Query the nsys SQLite export

The join CLI does this for you, including a per-stage summary across all requests with
`--summary`. To query a report yourself, export it:

```bash
nsys export --type sqlite --output decode.sqlite decode.nsys-rep
```

Facts you need:

- **Table and columns.** Request events are in `NVTX_EVENTS`. The stage is in `text` and the request
  key is in `uint64Value`. Event types: `34` mark, `59` push/pop range, `60` start/end range.
- **Absolute time.** `unix_ns = TARGET_INFO_SESSION_START_TIME.utcEpochNs + NVTX_EVENTS.start`. Use
  this to compare reports from different processes or hosts. Cross-host comparison is only as good
  as the hosts' clock sync.
- **Process and thread.** `pid = (globalTid >> 24) & 0xFFFFFF` and `tid = globalTid & 0xFFFFFF`.
- **Open ranges.** A range that never ended in the capture has an `end` beyond
  `ANALYSIS_DETAILS.duration`. Treat its end as unknown.
- **Matching identity marks.** Use `instr(text, ...)` or exact equality, not `LIKE`: `LIKE` treats
  `_` as a wildcard and is case-insensitive.
- **Registered strings.** Ranges registered through `StringIds` have a NULL `text`; join
  `StringIds` on `textId` to read the name.

**Per-stage aggregation** across all requests in the report:

```bash
sqlite3 decode.sqlite "
  SELECT text AS stage, count(*) AS n, avg(\"end\" - start) / 1e6 AS avg_ms
  FROM NVTX_EVENTS
  WHERE eventType = 60 AND uint64Value IS NOT NULL
    AND \"end\" <= (SELECT duration FROM ANALYSIS_DETAILS)
  GROUP BY text;"
```

The `end` filter drops ranges that never closed. Without it, their raw end timestamps swamp the
averages.

`nsys stats --report nvtx_sum decode.nsys-rep` gives a similar per-stage table without SQL for a
capture that covers whole requests. For a time-window capture (`--delay`/`--duration`), requests in
flight at the window edges leave many ranges open, and `nvtx_sum` in Nsight Systems 2026.3 fails
with `integer overflow`. Use the query above or the join CLI's `--summary`, which skip open
ranges. Either way, a window only measures requests that started and finished inside it, so long
requests are under-represented.

**Request id to key.** The identity mark maps a request id to its key. To start from a trace id or
`x-request-id` instead, match `trace=<id>` or `xrid=<id>` in the identity mark's text.

```bash
sqlite3 decode.sqlite "
  SELECT DISTINCT uint64Value FROM NVTX_EVENTS
  WHERE eventType = 34
    AND (text = 'dynamo.request rid=6f1c2b0e-4a57-4c0c-9b3f-0d5e8a1b7c42'
         OR instr(text, 'dynamo.request rid=6f1c2b0e-4a57-4c0c-9b3f-0d5e8a1b7c42 ') = 1);"
```

You can also compute the key directly with `rid_key` above, which does not need the identity mark.

**Every event for a key**, ordered by absolute start time:

```bash
sqlite3 decode.sqlite "
  SELECT (SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME) + start AS start_unix_ns,
         \"end\", eventType, (globalTid >> 24) & 16777215 AS pid, text
  FROM NVTX_EVENTS
  WHERE uint64Value = <key>
  ORDER BY start;"
```

## Operational warnings

> [!WARNING]
> `nsys` reports and exports embed the profiled process's environment, which can include API keys,
> tokens, and other secrets. Treat `.nsys-rep` and `.sqlite` files like logs, and review them before
> sharing.

> [!WARNING]
> A child process forked without `exec` loses its NVTX events unless you pass
> `--trace-fork-before-exec=true` to `nsys profile`. If a worker spawns engine subprocesses by
> forking, launch it with that flag.

## Related

- [Follow One Request into Nsight Systems](../../cli/operations/observability.mdx#follow-one-request-into-nsight-systems)
- [Request Trace Reference](request-traces.mdx)
- [Logging Reference](logging.mdx)
- [Environment Variables](environment-variables.mdx)
- [Request timeline join tool](https://github.com/ai-dynamo/dynamo/blob/main/benchmarks/request_trace/README.md)
