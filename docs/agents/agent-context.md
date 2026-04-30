---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Agent Context and Tracing
subtitle: Attach workflow identity to agentic requests
---

Dynamo supports passive agent request tracing. An agent harness can attach
identity metadata to each LLM request, and Dynamo can write normalized
`request_end` records to configured trace sinks.

This is observability only. It does not change routing, scheduling, or cache
behavior.

## Request Metadata

Set `nvext.agent_context` on chat completion requests:

```json
{
  "model": "my-model",
  "messages": [{"role": "user", "content": "Research Dynamo agent tracing."}],
  "nvext": {
    "agent_context": {
      "workflow_type_id": "deep_research",
      "workflow_id": "research-run-42",
      "program_id": "research-run-42:researcher",
      "parent_program_id": "research-run-42:planner"
    }
  }
}
```

For per-call correlation, set the HTTP `x-request-id` header to the harness LLM
call ID:

```text
x-request-id: llm-call-42
```

`x-request-id` is not Dynamo's internal inference request ID. It is copied into
the trace record as `request.x_request_id`.

| Field | Required | Meaning |
|-------|:--------:|---------|
| `workflow_type_id` | Yes | Reusable workload/profile class, such as `deep_research` or `coding_agent`. |
| `workflow_id` | Yes | Top-level run identifier. |
| `program_id` | Yes | One schedulable reasoning/tool trajectory. |
| `parent_program_id` | No | Parent program for subagents. |

## Enabling Trace Output

Set `DYN_AGENT_TRACE_SINKS` before starting Dynamo. Use `jsonl` for local
trace files, `jsonl_gz` for rotating compressed trace segments, `stderr` for
development logging, or a comma-separated list:

```bash
export DYN_AGENT_TRACE_SINKS=jsonl_gz,stderr
export DYN_AGENT_TRACE_OUTPUT_PATH=/tmp/dynamo-agent-trace
export DYN_AGENT_TRACE_CAPACITY=1024
```

Minimum setup for rotating compressed traces:

```bash
export DYN_AGENT_TRACE_SINKS=jsonl_gz
export DYN_AGENT_TRACE_OUTPUT_PATH=/tmp/dynamo-agent-trace
```

| Environment Variable | Required | Default | Description |
|----------------------|:--------:|---------|-------------|
| `DYN_AGENT_TRACE_SINKS` | Yes | unset | Enables agent tracing and selects sinks. Supported values: `jsonl`, `jsonl_gz`, `stderr`, or a comma-separated list such as `jsonl_gz,stderr`. |
| `DYN_AGENT_TRACE_OUTPUT_PATH` | If `jsonl` or `jsonl_gz` is selected | unset | Local trace output path. For `jsonl`, this is the literal `.jsonl` file path. For `jsonl_gz`, this is the segment prefix used to derive `.jsonl.gz` files. |
| `DYN_AGENT_TRACE_CAPACITY` | No | `1024` | In-process trace bus capacity. |
| `DYN_AGENT_TRACE_JSONL_BUFFER_BYTES` | No | `1048576` | JSONL writer buffer size. For `jsonl_gz`, this is the max uncompressed batch size before appending a complete gzip member. |
| `DYN_AGENT_TRACE_JSONL_FLUSH_INTERVAL_MS` | No | `1000` | JSONL periodic flush interval. For `jsonl_gz`, each flush appends a complete gzip member. |
| `DYN_AGENT_TRACE_JSONL_GZ_ROLL_BYTES` | No | `268435456` | `jsonl_gz` segment roll threshold in uncompressed bytes. |
| `DYN_AGENT_TRACE_JSONL_GZ_ROLL_LINES` | No | unset | Optional `jsonl_gz` segment roll threshold in records. |

The `jsonl` sink writes one recorder JSON object per line:
`{"timestamp": <elapsed_ms>, "event": <normalized trace event>}`. The
`jsonl_gz` sink writes the same JSONL records into numbered compressed segments
derived from `DYN_AGENT_TRACE_OUTPUT_PATH`, such as
`/tmp/dynamo-agent-trace.000000.jsonl.gz` and
`/tmp/dynamo-agent-trace.000001.jsonl.gz`. Each flush appends a complete gzip
member, so standard gzip tools can read the concatenated stream. The `stderr`
sink logs the normalized trace event as a structured `agent_trace` log record.
All sinks are best-effort telemetry for debugging and offline profiling. They
are not durable audit logs.

## ms-agent End-to-End Smoke

To see this in action, use a fork of the ModelScope ms-agent DeepResearch
agent framework with Dynamo trace hooks. Until those hooks land upstream, this
branch injects `nvext.agent_context` and `x-request-id` on LLM requests:

```bash
uv pip install -e "git+ssh://git@github.com/ishandhanani/ms-agent.git@idhanani/dynamo-agent-trace#egg=ms-agent"
```

Start Dynamo with a local compressed trace sink:

```bash
export DYN_AGENT_TRACE_SINKS=jsonl_gz
export DYN_AGENT_TRACE_OUTPUT_PATH=/tmp/dynamo-agent-trace

# Launch any Dynamo OpenAI-compatible backend on :8000.
```

Run ms-agent against Dynamo. Set a stable workflow ID if you want to grep or
query one smoke run:

```bash
export DYNAMO_AGENT_WORKFLOW_ID=ms-agent-smoke-$(date +%Y%m%d-%H%M%S)

ms-agent run \
  --config /path/to/agent.yaml \
  --query "What is 2 + 2? Answer with just the number." \
  --trust_remote_code true
```

Read the resulting compressed trace records:

```bash
gzip -cd /tmp/dynamo-agent-trace.*.jsonl.gz | jq .
```

Expected records should contain `event.event_type = "request_end"`,
`event.agent_context.workflow_id` matching `DYNAMO_AGENT_WORKFLOW_ID`, the
caller `x_request_id`, token counts, TTFT, average ITL, cache metrics, queue
depth, and worker IDs when available.

## Perfetto Timeline Conversion

Convert Dynamo agent trace shards to Chrome Trace JSON for Perfetto UI:

```bash
python3 benchmarks/agent_trace/convert_to_perfetto.py \
  "/tmp/dynamo-agent-trace.*.jsonl.gz" \
  --output /tmp/dynamo-agent-trace.perfetto.json
```

Open `/tmp/dynamo-agent-trace.perfetto.json` in
[Perfetto UI](https://ui.perfetto.dev/). Each LLM request becomes a timeline
slice grouped by workflow and program lane. The slice args include request IDs,
model, token counts, cache metrics, TTFT, average ITL, queue depth, and worker
IDs. By default, the converter stacks prefill wait, prefill, and decode slices
under each request when those timings are present. Add `--include-markers` to
emit first-token instant markers, `--no-stages` for a compact request-only
view, or `--separate-stage-tracks` to place stages on adjacent tracks when
debugging Perfetto nesting or label rendering. Stage slice boundaries are
normalized to avoid same-thread overlap caused by independent metric rounding;
raw timing fields remain available in event args.

## Operator Notes

- Agent request trace emission is currently wired for `/v1/chat/completions`.
- `DYN_AGENT_TRACE_SINKS` is the enable switch. Setting
  `DYN_AGENT_TRACE_OUTPUT_PATH` alone does not enable tracing.
- The `jsonl` sink appends to the configured path and does not rotate or enforce
  a maximum file size. Enable it for bounded debug/profiling runs, not as a
  long-running production sink.
- The `jsonl_gz` sink rotates compressed segments and is the preferred local
  file sink for long profiling or RL runs.

## Request-End Record

Dynamo emits `request_end` after the response stream completes or is dropped.
Nullable fields are omitted when the serving path did not record them.

```json
{
  "schema": "dynamo.agent.trace.v1",
  "event_type": "request_end",
  "event_time_unix_ms": 1777312801000,
  "event_source": "dynamo",
  "agent_context": {
    "workflow_type_id": "deep_research",
    "workflow_id": "research-run-42",
    "program_id": "research-run-42:researcher",
    "parent_program_id": "research-run-42:planner"
  },
  "request": {
    "request_id": "dynamo-request-id",
    "x_request_id": "llm-call-42",
    "model": "my-model",
    "input_tokens": 4096,
    "output_tokens": 512,
    "cached_tokens": 3584,
    "request_received_ms": 1777312800000,
    "prefill_wait_time_ms": 12.1,
    "prefill_time_ms": 70.3,
    "ttft_ms": 82.4,
    "total_time_ms": 1000.1,
    "avg_itl_ms": 1.8,
    "kv_hit_rate": 0.875,
    "kv_transfer_estimated_latency_ms": 4.2,
    "queue_depth": 3,
    "worker": {
      "prefill_worker_id": 0,
      "prefill_dp_rank": 0,
      "decode_worker_id": 1,
      "decode_dp_rank": 0
    }
  }
}
```

The `request` object captures Dynamo-owned request performance fields:

| Field | Meaning |
|-------|---------|
| `request_id` | Dynamo request ID for the LLM call. |
| `x_request_id` | Caller-provided logical request ID when present. |
| `model` | Requested model name. |
| `input_tokens` | Prompt/input token count when known. |
| `output_tokens` | Final output token count when known. |
| `cached_tokens` | Prompt tokens served from prefix/KV cache when known. |
| `request_received_ms` | Request receive time in Unix epoch milliseconds. |
| `prefill_wait_time_ms` | Time from request receipt to prefill start. |
| `prefill_time_ms` | Time from prefill start to first token. |
| `ttft_ms` | Time from request receipt to first token. |
| `total_time_ms` | Time from request receipt to request completion. |
| `avg_itl_ms` | Average inter-token latency after first token. |
| `kv_hit_rate` | Effective KV-cache hit rate observed by the router. |
| `kv_transfer_estimated_latency_ms` | Upper-bound estimated disaggregated KV transfer latency. |
| `queue_depth` | Router queue depth observed when routing the request. |
| `worker` | Prefill/decode worker IDs and DP ranks when recorded. |

This trace does not include prompt/response content, sampling parameters,
finish reason, error status, or OpenTelemetry/OpenInference attributes. Use the
audit sink for request/response payload capture and OTEL export for span-based
observability.

## Current Scope

- `agent_context` is passive metadata.
- Dynamo emits request-end trace records when agent tracing is enabled.
- `jsonl`, `jsonl_gz`, and `stderr` are local debug/profiling sinks.
- Trace records are best-effort profiling data, not durable audit records.
- Future scheduler/profiler consumers should read the normalized trace bus.
