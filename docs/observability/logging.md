---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Logging
subtitle: Structured text and JSONL logging for local Dynamo processes, with optional OTLP export to Loki via the OpenTelemetry Collector.
---

> [!IMPORTANT]
> Set `OTEL_EXPORT_ENABLED=true` on every Dynamo process. Without it, logs never leave the process and Loki will be silent regardless of `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT`.

## Overview

Dynamo provides structured logging in both text as well as JSONL. When
JSONL is enabled, logs support `trace_id` and `span_id` fields for
distributed tracing. Span creation and exit events can be optionally
enabled via the `DYN_LOGGING_SPAN_EVENTS` environment variable.

## Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `DYN_LOGGING_JSONL` | Enable JSONL logging format | `false` | `true` |
| `DYN_LOGGING_SPAN_EVENTS` | Enable span entry/close event logging (`SPAN_FIRST_ENTRY`, `SPAN_CLOSED` messages) | `false` | `true` |
| `DYN_LOG` | Log levels per target `<default_level>,<module_path>=<level>,<module_path>=<level>` | `info` | `DYN_LOG=info,dynamo_runtime::system_status_server:trace` |
| `DYN_LOG_USE_LOCAL_TZ` | Use local timezone for timestamps (default is UTC) | `false` | `true` |
| `DYN_LOGGING_CONFIG_PATH` | Path to custom TOML logging configuration | none | `/path/to/config.toml` |
| `VLLM_LOGGING_LEVEL` | vLLM backend log level (independent of `DYN_LOG`) | `INFO` | `DEBUG` |
| `TLLM_LOG_LEVEL` | TensorRT-LLM backend log level (independent of `DYN_LOG`) | `INFO` | `DEBUG` |
| `DYN_SKIP_SGLANG_LOG_FORMATTING` | Disable Dynamo's SGLang log configuration | `false` | `true` |
| `OTEL_SERVICE_NAME` | Service name for trace and span information | `dynamo` | `dynamo-frontend` |
| `OTEL_EXPORT_ENABLED` | Enable OTLP export of both traces and logs | `false` | `true` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Default OTLP endpoint for traces and logs when signal-specific endpoints are unset | `http://localhost:4317` | `http://collector:4317` |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | Override endpoint for traces, used as-is | unset | `http://tempo:4317` |
| `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT` | Override endpoint for logs, used as-is | unset | `http://loki-collector:4317` |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | Default OTLP protocol for traces and logs | `grpc` | `http/protobuf` |
| `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` | Override protocol for traces | unset | `grpc` |
| `OTEL_EXPORTER_OTLP_LOGS_PROTOCOL` | Override protocol for logs | unset | `grpc` |

## OTLP Log Export

When `OTEL_EXPORT_ENABLED=true`, Dynamo exports both **traces and logs** via OTLP. Logs are sent to an OpenTelemetry Collector which routes them to Grafana Loki for aggregation and querying.

Set `OTEL_EXPORTER_OTLP_ENDPOINT` to configure the default OTLP collector endpoint for both traces and logs. To send logs somewhere else, set `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT`; to send traces somewhere else, set `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`.

Signal-specific endpoints are used as-is. When using the generic `OTEL_EXPORTER_OTLP_ENDPOINT` with `http/protobuf`, Dynamo appends the signal path (`/v1/logs` or `/v1/traces`). With `grpc`, the endpoint is used without a path.

```bash
export OTEL_EXPORT_ENABLED=true
# Generic endpoint applies to both traces and logs:
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
# Optional per-signal overrides:
# export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://tempo:4317
# export OTEL_EXPORTER_OTLP_LOGS_ENDPOINT=http://loki-collector:4317
```

The local observability stack (see [Getting Started](README.md#getting-started-quickly)) includes an OpenTelemetry Collector that receives OTLP on `localhost:4317` and routes traces to Tempo and logs to Loki. In Grafana, the Loki datasource is pre-configured with a derived field that links `trace_id` labels to Tempo, so you can jump directly from a log line to its corresponding trace.

## Getting Started Quickly

### Start Observability Stack

For collecting and visualizing logs with Grafana Loki, or viewing trace context in logs alongside Grafana Tempo, start the observability stack. See [Observability Getting Started](README.md#getting-started-quickly) for instructions. The stack includes Loki, an OpenTelemetry Collector, and Tempo — all pre-wired together.

### Enable Structured Logging

Enable structured JSONL logging:

```bash
export DYN_LOGGING_JSONL=true
export DYN_LOG=debug

# Start your Dynamo components (default port 8000, override with --http-port or DYN_HTTP_PORT env var)
python -m dynamo.frontend &
python -m dynamo.vllm --model Qwen/Qwen3-0.6B --enforce-eager &
```

Logs will be written to stderr in JSONL format with trace context.

## Available Logging Levels

| **Logging Levels (Least to Most Verbose)** | **Description**                                                                 |
|-------------------------------------------|---------------------------------------------------------------------------------|
| **ERROR**                                 | Critical errors (e.g., unrecoverable failures, resource exhaustion)              |
| **WARN**                                  | Unexpected or degraded situations (e.g., retries, recoverable errors)           |
| **INFO**                                  | Operational information (e.g., startup/shutdown, major events)                 |
| **DEBUG**                                 | General debugging information (e.g., variable values, flow control)            |
| **TRACE**                                 | Very low-level, detailed information (e.g., internal algorithm steps)           |

## Example Readable Format

Environment Setting:

```
export DYN_LOG="info,dynamo_runtime::system_status_server:trace"
export DYN_LOGGING_JSONL="false"
```

Resulting Log format:

```
2025-09-02T15:50:01.770028Z  INFO main.init: VllmWorker for Qwen/Qwen3-0.6B has been initialized
2025-09-02T15:50:01.770195Z  INFO main.init: Reading Events from tcp://127.0.0.1:21555
2025-09-02T15:50:01.770265Z  INFO main.init: Getting engine runtime configuration metadata from vLLM engine...
2025-09-02T15:50:01.770316Z  INFO main.get_engine_cache_info: Cache config values: {'num_gpu_blocks': 24064}
2025-09-02T15:50:01.770358Z  INFO main.get_engine_cache_info: Scheduler config values: {'max_num_seqs': 256, 'max_num_batched_tokens': 2048}
```

## Example JSONL Format

Environment Setting:

```
export DYN_LOG="info,dynamo_runtime::system_status_server:trace"
export DYN_LOGGING_JSONL="true"
```

Resulting Log format:

```
{"time":"2025-09-02T15:53:31.943377Z","level":"INFO","target":"log","message":"VllmWorker for Qwen/Qwen3-0.6B has been initialized","log.file":"/opt/dynamo/venv/lib/python3.12/site-packages/dynamo/vllm/main.py","log.line":191,"log.target":"main.init"}
{"time":"2025-09-02T15:53:31.943550Z","level":"INFO","target":"log","message":"Reading Events from tcp://127.0.0.1:26771","log.file":"/opt/dynamo/venv/lib/python3.12/site-packages/dynamo/vllm/main.py","log.line":212,"log.target":"main.init"}
{"time":"2025-09-02T15:53:31.943636Z","level":"INFO","target":"log","message":"Getting engine runtime configuration metadata from vLLM engine...","log.file":"/opt/dynamo/venv/lib/python3.12/site-packages/dynamo/vllm/main.py","log.line":220,"log.target":"main.init"}
{"time":"2025-09-02T15:53:31.943701Z","level":"INFO","target":"log","message":"Cache config values: {'num_gpu_blocks': 24064}","log.file":"/opt/dynamo/venv/lib/python3.12/site-packages/dynamo/vllm/main.py","log.line":267,"log.target":"main.get_engine_cache_info"}
{"time":"2025-09-02T15:53:31.943747Z","level":"INFO","target":"log","message":"Scheduler config values: {'max_num_seqs': 256, 'max_num_batched_tokens': 2048}","log.file":"/opt/dynamo/venv/lib/python3.12/site-packages/dynamo/vllm/main.py","log.line":268,"log.target":"main.get_engine_cache_info"}
```

## Logging of Trace and Span IDs

When `DYN_LOGGING_JSONL` is enabled, all logs include `trace_id` and `span_id` fields, and spans are automatically created for requests. This is useful for short debugging sessions where you want to examine trace context in logs without setting up a full tracing backend and for correlating log messages with traces.

The trace and span information uses the OpenTelemetry format and libraries, which means the IDs are compatible with OpenTelemetry-based tracing backends like Tempo or Jaeger if you later choose to enable trace export.

**Note:** This section has overlap with [Distributed Tracing with Tempo](tracing.md). For trace visualization in Grafana Tempo and persistent trace analysis, see [Distributed Tracing with Tempo](tracing.md).

### Configuration for Logging

To see trace information in logs:

```bash
export DYN_LOGGING_JSONL=true
export DYN_LOG=debug  # Set to debug to see detailed trace logs

# Start your Dynamo components (e.g., frontend and worker) (default port 8000, override with --http-port or DYN_HTTP_PORT env var)
python -m dynamo.frontend &
python -m dynamo.vllm --model Qwen/Qwen3-0.6B --enforce-eager &
```

This enables JSONL logging with `trace_id` and `span_id` fields. Traces appear in logs but are not exported to any backend.

### Example Request

Send a request to generate logs with trace context:

```bash
curl -H 'Content-Type: application/json' \
-H 'x-request-id: test-trace-001' \
-d '{
  "model": "Qwen/Qwen3-0.6B",
  "max_completion_tokens": 100,
  "messages": [
    {"role": "user", "content": "What is the capital of France?"}
  ]
}' \
http://localhost:8000/v1/chat/completions
```

Check the logs (stderr) for JSONL output containing `trace_id`, `span_id`, and `x_request_id` fields.

## Trace and Span Information in Logs

This section shows how trace and span information appears in JSONL logs. These logs can be used to understand request flows even without a trace visualization backend.

### Example Disaggregated Trace in Grafana

When viewing the corresponding trace in Grafana, you should be able to see something like the following:

![Disaggregated Trace Example](../assets/img/grafana-disagg-trace.png)
### Trace Overview

Dynamo creates distributed traces that span across multiple services in a disaggregated serving setup. The following sections describe the key spans you'll see in Grafana when viewing traces for chat completion requests.

#### Available Spans in Disaggregated Mode

When running Dynamo in disaggregated mode, a typical request creates the following spans:

##### 1. `http-request` (Frontend - Root Span)

The root span for the entire request lifecycle, created in the **dynamo-frontend** service.

**Key Attributes:**
- **Service**: `dynamo-frontend`
- **Operation**: Handles the HTTP request from client to completion
- **Duration**: Total end-to-end request time (includes prefill + decode)
- **Method**: HTTP method (typically `POST`)
- **URI**: Request endpoint (e.g., `/v1/chat/completions`)
- **Status**: Request completion status
- **Children**: Typically 2-3 child spans (routing span + worker spans)

This span represents the complete request flow from when the frontend receives the HTTP request until the final response is sent back to the client.

##### 2. `prefill_routing` (Frontend - Routing Span)

A child span of `http-request`, created in the **dynamo-frontend** service during the routing phase.

**Key Attributes:**
- **Service**: `dynamo-frontend`
- **Operation**: Routes the prefill request to an appropriate prefill worker
- **Duration**: Time spent selecting and the span of prefill.
- **Parent**: `http-request` span

This span captures the routing logic and decision-making process and the request sent to the prefill worker.

##### 3. `handle_payload` (Prefill Worker Span)

A child span of `http-request`, created in the **dynamo-worker-vllm-prefill** service.

**Key Attributes:**
- **Service**: `dynamo-worker-vllm-prefill` (or `dynamo-worker-sglang-prefill` for SGLang)
- **Operation**: Processes the prefill phase of generation
- **Duration**: Time to compute prefill (typically milliseconds to seconds)
- **Component**: `prefill`
- **Endpoint**: `generate`
- **Parent**: `http-request` span

This span represents the actual prefill computation on a prefill-specialized worker, including prompt processing and initial KV cache generation.

##### 4. `handle_payload` (Decode Worker Span)

A child span of `http-request`, created in the **dynamo-worker-vllm-decode** service.

**Key Attributes:**
- **Service**: `dynamo-worker-vllm-decode` (or `dynamo-worker-sglang-decode` for SGLang)
- **Operation**: Processes the decode phase of generation
- **Duration**: Time to generate all output tokens (typically seconds)
- **Component**: `decode` or `backend`
- **Endpoint**: `generate`
- **Parent**: `http-request` span

This span represents the iterative token generation phase on a decode-specialized worker, which consumes the KV cache from prefill and produces output tokens.


#### Understanding Span Metrics

Each span provides several useful metrics:

| Metric | Description |
|--------|-------------|
| **Duration** | Total time from span start to end |
| **Busy Time** | Time actively processing (excluding waiting) |
| **Idle Time** | Time spent waiting (e.g., for network, other services) |
| **Start Time** | When the span began |
| **Child Count** | Number of direct child spans |

The relationship **Duration = Busy Time + Idle Time** helps identify where time is spent and potential bottlenecks.

## Custom Request IDs in Logs

You can provide a custom request ID using the `x-request-id` header. This ID will be attached to all spans and logs for that request, making it easier to correlate traces with application-level request tracking.

### Example Request with Custom Request ID

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'x-request-id: 8372eac7-5f43-4d76-beca-0a94cfb311d0' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {
        "role": "user",
        "content": "Explain why Roger Federer is considered one of the greatest tennis players of all time"
      }
    ],
    "stream": false,
    "max_tokens": 1000
  }'
```

All spans and logs for this request will include the `x_request_id` attribute with value `8372eac7-5f43-4d76-beca-0a94cfb311d0`.

### Frontend Logs with Custom Request ID

Notice how the `x_request_id` field appears in all log entries, alongside the `trace_id` (`80196f3e3a6fdf06d23bb9ada3788518`) and `span_id`:

```
{"time":"2025-10-31T21:06:45.397194Z","level":"DEBUG","file":"/opt/dynamo/lib/runtime/src/pipeline/network/tcp/server.rs","line":230,"target":"dynamo_runtime::pipeline::network::tcp::server","message":"Registering new TcpStream on 10.0.4.65:41959","method":"POST","span_id":"f7e487a9d2a6bf38","span_name":"http-request","trace_id":"80196f3e3a6fdf06d23bb9ada3788518","uri":"/v1/chat/completions","version":"HTTP/1.1","x_request_id":"8372eac7-5f43-4d76-beca-0a94cfb311d0"}
{"time":"2025-10-31T21:06:45.418584Z","level":"DEBUG","file":"/opt/dynamo/lib/llm/src/kv_router/prefill_router.rs","line":232,"target":"dynamo_llm::kv_router::prefill_router","message":"Prefill succeeded, using disaggregated params for decode","method":"POST","span_id":"f7e487a9d2a6bf38","span_name":"http-request","trace_id":"80196f3e3a6fdf06d23bb9ada3788518","uri":"/v1/chat/completions","version":"HTTP/1.1","x_request_id":"8372eac7-5f43-4d76-beca-0a94cfb311d0"}
{"time":"2025-10-31T21:06:45.418854Z","level":"DEBUG","file":"/opt/dynamo/lib/runtime/src/pipeline/network/tcp/server.rs","line":230,"target":"dynamo_runtime::pipeline::network::tcp::server","message":"Registering new TcpStream on 10.0.4.65:41959","method":"POST","span_id":"f7e487a9d2a6bf38","span_name":"http-request","trace_id":"80196f3e3a6fdf06d23bb9ada3788518","uri":"/v1/chat/completions","version":"HTTP/1.1","x_request_id":"8372eac7-5f43-4d76-beca-0a94cfb311d0"}
```



## Backend Engine Log Levels

Dynamo's `DYN_LOG` environment variable controls Dynamo's own logging. Each
inference backend has its own log level control that is **independent** of
`DYN_LOG`.

### vLLM

vLLM log level is controlled by the `VLLM_LOGGING_LEVEL` environment variable.
It defaults to `INFO` and is completely independent of `DYN_LOG`.

```bash
# Set vLLM to debug while keeping Dynamo at info
export DYN_LOG=info
export VLLM_LOGGING_LEVEL=DEBUG
```

Valid values: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

### TensorRT-LLM

TensorRT-LLM log level is controlled by the `TLLM_LOG_LEVEL` environment
variable. It defaults to `INFO` and is completely independent of `DYN_LOG`.

```bash
# Set TRT-LLM to info while keeping Dynamo at warn
export DYN_LOG=warn
export TLLM_LOG_LEVEL=INFO
```

Valid values: `TRACE`, `DEBUG`, `INFO`, `WARNING`, `ERROR`, `INTERNAL_ERROR`.

**Note:** `TLLM_LOG_LEVEL` is read once at TensorRT-LLM import time. It must
be set before the process starts.

### SGLang

SGLang logging is currently configured through Dynamo and follows the
`DYN_LOG` level by default. To disable Dynamo's SGLang log configuration
and manage it independently, set:

```bash
export DYN_SKIP_SGLANG_LOG_FORMATTING=true
```

Alternatively, pass the `--log-level` argument to the SGLang worker
command to set the SGLang engine's log level directly (e.g.
`--log-level DEBUG`). This is independent of `DYN_LOG`.

## Request Payload Logging (OTLP)

Dynamo emits per-request `/v1/chat/completions` payloads through request trace.
The `request_payload` row carries the client request and, when one completes,
the response. The same `DYN_REQUEST_TRACE_SINKS` setting selects local
files, stderr, NATS, and OTLP log export.

Request payload logging is opt-in and independent of `OTEL_EXPORT_ENABLED`
(which controls application log and trace export). By default, Dynamo does not
emit request or response payload rows, even when the OpenAI request sets
`store=true`. Set `DYN_REQUEST_TRACE_RECORDS=request_payload` to emit payload
rows for every eligible chat request, or include `request_payload` alongside
other selected record types.

> [!WARNING]
> Audit payload logging has migrated to request trace. This is a breaking
> migration for legacy audit consumers: the `store=true` audit trigger and old
> audit wire schema are not preserved. Payload rows use
> `dynamo.request.trace.v1` with `event_type=request_payload`.

> [!IMPORTANT]
> The OTLP payload path is enabled by `DYN_REQUEST_TRACE_SINKS=otel`, or
> by combining sinks such as `DYN_REQUEST_TRACE_SINKS=file,otel`.
> Setting `DYN_REQUEST_TRACE_SINKS=stderr` together with
> `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT` writes request trace JSON directly to
> stderr only; it does not export payload records over OTLP.

### How records are emitted

- One `dynamo.request.trace.v1` record with `event_type=request_payload` is
  published per eligible chat request.
- When requests are cancelled or fail, the record is
  still emitted with `payload.response` omitted, so those cases remain
  inspectable. A hard process crash before emission can lose a record.
- Each `otel` sink record maps to one OTLP `LogRecord`: scope
  `dynamo.request_trace`, body `request_payload`, with attributes `schema`,
  `event_type`, `rid`, `endpoint`, `model`, `streaming`, `payload_complete`,
  and `payload` (the request trace row serialized as a JSON string).

### Configuration

Payload-related request trace variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DYN_REQUEST_TRACE` | Master switch for request trace. When enabled and `DYN_REQUEST_TRACE_RECORDS` is unset, emits `request_end,tool`. | unset (disabled) |
| `DYN_REQUEST_TRACE_RECORDS` | Comma-separated record types: `request_end`, `request_payload`, `tool`. Include `request_payload` to emit request and response payload bodies. | `request_end,tool` when `DYN_REQUEST_TRACE=1`; unset otherwise |
| `DYN_REQUEST_TRACE_SINKS` | Comma-separated sinks: `file`, `stderr`, `nats`, `otel`. | `file` when enabled |
| `DYN_REQUEST_TRACE_NATS_SUBJECT` | Subject used by the `nats` sink. | `dynamo.request_trace.v1` |
| `DYN_REQUEST_TRACE_OTEL_MAX_PAYLOAD_BYTES` | Max serialized OTLP payload size. Oversized payload rows emit a marker with `payload_complete=false` and `payload_drop_reason`. | `4194304` (4 MiB) |

> [!WARNING]
> Deprecated. `DYN_AUDIT_SINKS`, `DYN_AUDIT_FORCE_LOGGING`,
> `DYN_AUDIT_NATS_SUBJECT`, and `DYN_AUDIT_OTEL_MAX_PAYLOAD_BYTES` are
> migration shims, not legacy audit compatibility aliases.
> A truthy `DYN_AUDIT_FORCE_LOGGING` maps to
> `DYN_REQUEST_TRACE_RECORDS=request_payload`; `DYN_AUDIT_SINKS` only selects
> destinations and does not enable `request_end` replay metadata. Prefer the
> `DYN_REQUEST_TRACE_*` variables for new deployments.

The `otel` sink ships over OTLP using the standard
`OTEL_EXPORTER_OTLP_*` variables, resolved the same way as the runtime log and
trace exporter:

- **Protocol**: `OTEL_EXPORTER_OTLP_LOGS_PROTOCOL` →
  `OTEL_EXPORTER_OTLP_PROTOCOL` → `grpc`.
- **Endpoint**: `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT` (used as-is) →
  `OTEL_EXPORTER_OTLP_ENDPOINT` (`/v1/logs` appended only for
  `http/protobuf`) → default (`http://localhost:4317` for `grpc`,
  `http://localhost:4318/v1/logs` for `http/protobuf`).
- `OTEL_SERVICE_NAME` sets the service name on exported records (default
  `dynamo`).

> [!NOTE]
> Keep the endpoint consistent with the protocol: a `grpc` endpoint is used
> verbatim, while `http/protobuf` expects an HTTP URL and gets `/v1/logs`
> appended. Because the request trace OTLP sink and the runtime exporter
> share these variables and the same `grpc` default, request payload records and
> application telemetry resolve to the same destination unless you override the
> logs signal explicitly.

### Example

Export chat-completion payload rows over OTLP gRPC to a collector, logging every
request:

```bash
export DYN_REQUEST_TRACE_RECORDS=request_payload
export DYN_REQUEST_TRACE_SINKS=otel
export OTEL_EXPORTER_OTLP_LOGS_ENDPOINT=http://otel-collector:4317
export OTEL_EXPORTER_OTLP_LOGS_PROTOCOL=grpc
```

To also write request trace JSON to stderr:

```bash
export DYN_REQUEST_TRACE_SINKS=stderr,otel
```

For local files, set `DYN_REQUEST_TRACE_SINKS=file` and configure
`DYN_REQUEST_TRACE_FILE_PATH`, `DYN_REQUEST_TRACE_FILE_FORMAT`, and
`DYN_REQUEST_TRACE_FILE_COMPRESSION`. See
[Request Replay Tracing](request-tracing.md) for the full request trace
configuration and record schema.

## Related Documentation

- [Distributed Tracing with Tempo](tracing.md)
- [Log Aggregation in Kubernetes](../kubernetes/observability/logging.md)
- [Observability Getting Started](README.md)
- [Distributed Runtime Architecture](../design-docs/distributed-runtime.md)
- [Dynamo Architecture Overview](../design-docs/architecture.md)
- [Backend Guide](../development/backend-guide.md)
