# DIS-1643: Consistent Error Tracing — Implementation Plan

## Overview

4 stacked PRs implementing consistent structured logging across all frontend error paths and worker trace propagation.

## PR1: Trace Infrastructure (#7733)

**Problem:** Request spans use `info_span!` which gets filtered out by `DYN_LOG=warn/error`, losing request context on error logs. No UUID validation on `x-dynamo-request-id`. System and inference endpoints share the same span/log treatment.

**Solution:**
- Custom span targets: `request_span` (inference, always on via `request_span=trace` directive) and `system_span` (health/metrics, debug level)
- `make_inference_request_span()` generates UUID when client doesn't provide one — captured by `DistributedTraceIdLayer` and propagated to workers
- `get_or_create_request_id()` returns `Result<String, String>` — validates UUID header, callers format errors for their API (OpenAI via `from_http_error`, Anthropic via `anthropic_error`)
- Router split in `service_v2.rs`: system routes get `make_system_request_span`, inference routes get `make_inference_request_span`

**Files:** `logging.rs`, `openai.rs`, `anthropic.rs`, `service_v2.rs`, `system_status_server.rs`, `push_endpoint.rs`

## PR2: Request Lifecycle Logging (#7734)

**Problem:** No consistent "request received" / "request completed" log. Error paths have different log formats. Worker has no request-level logs at INFO.

**Solution:**
- `InflightGuard` is single source of truth: logs "request received" (INFO) on creation, "request completed" (INFO/ERROR) on Drop
- `create_inflight_guard()` takes `request_id`, records `model` on span — no separate setup calls
- All inference errors at ERROR level with `status=error`, `error_type`, `error_detail`
- `on_response` renamed to "http response sent" — system at DEBUG, inference at INFO/ERROR
- Worker `push_handler.rs` logs "request received" / "request completed" at INFO

**Files:** `metrics.rs`, `openai.rs`, `anthropic.rs`, `service_v2.rs`, `push_handler.rs`, `grpc/{openai,tensor}.rs`

## PR3: Token Counts, TTFT, ITL, Worker IDs (#7735)

**Problem:** No token counts, latency metrics, or worker identification on request completion logs.

**Solution:**
- `ResponseMetricCollector::Drop` records on span: `input_tokens`, `output_tokens`, `ttft_ms`, `avg_itl_ms`, `prefill_worker_id`, `decode_worker_id`
- TTFT stored from already-computed value; ITL accumulated from per-chunk computation
- WARN log at cancellation point in `disconnect.rs`
- Connection monitor upgraded from TRACE to WARN

**Performance:** All additions are on cleanup path (Drop), not streaming hot path. Two f64/u64 accumulations per chunk (negligible alongside existing histogram publish).

**Files:** `metrics.rs`, `disconnect.rs`, `logging.rs` (Empty fields)

## PR4: E2E Tests (#7766)

11 parallel-safe pytest tests (25s with `-n auto`):

| Category | Tests |
|----------|-------|
| Aggregated success | unary, streaming, request ID propagation |
| Aggregated errors | 404, 400 invalid UUID, cancellation, worker crash |
| Disaggregated success | unary, streaming (both workers verified) |
| Disaggregated crashes | prefill crash, decode crash |

## Request ID Propagation Flow

```
Client → x-dynamo-request-id header (optional, 400 if invalid)
  ↓
make_inference_request_span() → generates UUID if absent
  ↓
DistributedTraceIdLayer::on_new_span() → captures into trace context
  ↓
get_or_create_request_id() → validates + reads from trace context
  ↓
create_inflight_guard() → logs "request received", records model on span
  ↓
addressed_router.rs → inject_trace_headers_into_map() → worker gets UUID
  ↓
Worker span → x_dynamo_request_id, trace_id correlation
  ↓
InflightGuard::Drop → logs "request completed" with all fields
```

## Log Messages

| Message | When | Success | Error |
|---|---|---|---|
| "request received" | Request starts | INFO | INFO |
| "http response sent" | HTTP headers sent | INFO | ERROR |
| "request completed" | Request fully done | INFO | ERROR |
| "request cancelled by client" | Client disconnect | — | WARN |

## Error Details

| error_type | error_detail |
|---|---|
| cancelled | client disconnected before completion |
| internal | internal server error during processing |
| validation | invalid request parameters |
| not_found | model or resource not found |
| overload | service overloaded or rate limited |
| not_implemented | requested feature not implemented |

## "request completed" Fields (streaming)

```json
{
  "level": "INFO",
  "message": "request completed",
  "status": "success",
  "request_id": "32691d61-...",
  "model": "qwen/qwen3-0.6b",
  "endpoint": "chat_completions",
  "request_type": "stream",
  "elapsed_ms": "20",
  "input_tokens": "9",
  "output_tokens": "50",
  "ttft_ms": "5.85",
  "avg_itl_ms": "0.29",
  "trace_id": "bca97f5e..."
}
```

## Before / After

### Before — Cancellation
```
(nothing — only a metric counter increment)
```

### Before — HTTP 500
```
request completed with server error  status=500  latency_ms=1508
```

### Before — HTTP 400
```
request completed with client request error  status=400  latency_ms=23
```

### After — Streaming Success
```json
{"level":"INFO","message":"request received","request_id":"284f18c7-...","model":"qwen/qwen3-0.6b","endpoint":"chat_completions","request_type":"stream"}
{"level":"INFO","message":"http response sent","status":"200","latency_ms":"4"}
{"level":"INFO","message":"request completed","status":"success","elapsed_ms":"14","input_tokens":"9","output_tokens":"50","ttft_ms":"5.85","avg_itl_ms":"0.29"}
```

### After — 404 Error
```json
{"level":"INFO","message":"request received","request_id":"4644979b-...","model":"nonexistent-model"}
{"level":"ERROR","message":"request completed","status":"error","error_type":"not_found","error_detail":"model or resource not found","elapsed_ms":"0"}
{"level":"ERROR","message":"http response sent","status":"404","latency_ms":"0"}
```

### After — Worker Crash (partial generation)
```json
{"level":"ERROR","message":"request completed","status":"error","error_type":"internal","error_detail":"internal server error during processing","elapsed_ms":"556","input_tokens":"9","output_tokens":"4"}
```

### After — Cancellation
```json
{"level":"ERROR","message":"request completed","status":"error","error_type":"cancelled","error_detail":"client disconnected before completion","elapsed_ms":"230"}
```

### After — Worker (trace_id correlation)
```json
{"level":"INFO","message":"request received","request_id":"6ec6ddd9-...","component":"backend","endpoint":"generate","instance_id":"2221573453717914121","trace_id":"3fe59d20..."}
{"level":"INFO","message":"request completed","request_id":"6ec6ddd9-...","trace_id":"3fe59d20..."}
```

## Span Targets

| Target | Level | Always on? | Used for |
|---|---|---|---|
| `request_span` | info | Yes (`request_span=trace` in filters) | Inference HTTP spans, worker payload spans |
| `system_span` | debug | No (follows DYN_LOG) | Health, metrics, models endpoints |

Both overridable via `DYN_LOG=request_span=off` or `DYN_LOG=system_span=trace`.

## Follow-up Issues

- **DIS-1652** — Propagate model name to worker via transport headers
- **DIS-1653** — Token counts/TTFT missing on unary requests (collector Drop ordering)
