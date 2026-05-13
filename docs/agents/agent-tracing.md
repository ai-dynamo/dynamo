---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Agent Tracing
subtitle: Attach trajectory identity and export Dynamo request and tool-event telemetry
---

Agent tracing records **who** called (`nvext.agent_context`), **what Dynamo measured** on each LLM request (`request_end`), and optional **harness tool spans** (`tool_*`). Context is passive—it does not steer routing or caching. Output is best-effort profiling data, not an audit log.

**Flow:** Harness sends chat completions with `agent_context` → Dynamo emits `request_end` to trace sinks. Harness sends tool events over ZMQ → same sinks.

## Adding trace context to each LLM call

**Direct LLM call**

Inject `agent_context` into each LLM request

```json
{
  "model": "my-model",
  "messages": [{ "role": "user", "content": "..." }],
  "nvext": {
    "agent_context": {
      "session_type_id": "deep_research",
      "session_id": "research-run-42",
      "trajectory_id": "research-run-42:researcher",
      "parent_trajectory_id": "research-run-42:planner"
    }
  }
}
```

| Field                  | Required | Meaning                                  |
| ---------------------- | :------: | ---------------------------------------- |
| `session_type_id`      |   Yes    | Workload class (e.g. `deep_research`).   |
| `session_id`           |   Yes    | Whole agent run.                         |
| `trajectory_id`        |   Yes    | One reasoning/tool chain inside the run. |
| `parent_trajectory_id` |    No    | Parent trajectory when using subagents.  |

**OpenAI client:** merge into `extra_body` / `extra_headers`:

```python
import uuid

def instrument_llm_request(kwargs, agent_context):
    body = dict(kwargs.get("extra_body") or {})
    nvext = dict(body.get("nvext") or {})
    nvext["agent_context"] = dict(agent_context)
    body["nvext"] = nvext

    headers = dict(kwargs.get("extra_headers") or {})
    headers.setdefault("x-request-id", str(uuid.uuid4()))

    out = dict(kwargs)
    out["extra_body"] = body
    out["extra_headers"] = headers
    return out
```

`x-request-id` is your logical per-call id; Dynamo stores it as `request.x_request_id` (distinct from Dynamo's internal `request_id`). No Dynamo imports are required in the harness. Keep context in a contextvar, attach before each completion, and propagate across threads/processes when those paths call the model or emit tools.

## Enable output

Minimal local setup (rotating gzip shards):

```bash
export DYN_AGENT_TRACE_SINKS=jsonl_gz
export DYN_AGENT_TRACE_OUTPUT_PATH=/tmp/dynamo-agent-trace
```

Optional tool ingestion (Dynamo **binds**; harness **connects** PUSH):

```bash
export DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT=tcp://127.0.0.1:20390
```

`DYN_AGENT_TRACE_SINKS` must be set for files/stderr; output path alone does not enable tracing. ZMQ alone ingests tools but still needs a sink if you want JSONL.

<details>
<summary>All agent trace environment variables</summary>

| Variable                                   |        Required         | Default     | Notes                                                                                |
| ------------------------------------------ | :---------------------: | ----------- | ------------------------------------------------------------------------------------ |
| `DYN_AGENT_TRACE_SINKS`                    | For local files/stderr  | unset       | `jsonl`, `jsonl_gz`, `stderr`, or comma-separated (e.g. `jsonl_gz,stderr`).          |
| `DYN_AGENT_TRACE_OUTPUT_PATH`              | If `jsonl` / `jsonl_gz` | unset       | File path for `jsonl`; segment **prefix** for `jsonl_gz` → `prefix.NNNNNN.jsonl.gz`. |
| `DYN_AGENT_TRACE_CAPACITY`                 |           No            | `1024`      | Trace bus capacity.                                                                  |
| `DYN_AGENT_TRACE_JSONL_BUFFER_BYTES`       |           No            | `1048576`   | Buffer / gzip batch threshold.                                                       |
| `DYN_AGENT_TRACE_JSONL_FLUSH_INTERVAL_MS`  |           No            | `1000`      | Flush interval.                                                                      |
| `DYN_AGENT_TRACE_JSONL_GZ_ROLL_BYTES`      |           No            | `268435456` | Roll gzip segment by uncompressed bytes.                                             |
| `DYN_AGENT_TRACE_JSONL_GZ_ROLL_LINES`      |           No            | unset       | Optional roll by line count.                                                         |
| `DYN_AGENT_TRACE_REPLAY_HASHES`            |           No            | on          | Falsey (`0`, `no`, …) disables `replay` hashes on requests.                          |
| `DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT` |           No            | unset       | PULL bind address for tool records.                                                  |
| `DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_TOPIC`    |           No            | unset       | If set, first ZMQ frame must match.                                                  |

</details>

## Tool events (ZMQ)

Wire format: `[topic, seq_be_u64, msgpack(AgentTraceRecord)]`. To publish to Dynamo, use a background publisher, bounded queue, monotonic sequence, and PUSH with HWM. **Terminal** `tool_end` / `tool_error` rows should carry timing (`started_at_unix_ms`, `ended_at_unix_ms`, `duration_ms`) even if `tool_start` was dropped.

Same `agent_context` as the surrounding LLM calls; `tool_call_id` unique per trajectory. Join offline on `session_id`, `trajectory_id`, `tool_call_id`.

Example `tool_end`:

```json
{
  "schema": "dynamo.agent.trace.v1",
  "event_type": "tool_end",
  "event_time_unix_ms": 1777312801500,
  "event_source": "harness",
  "agent_context": {
    "session_type_id": "deep_research",
    "session_id": "research-run-42",
    "trajectory_id": "research-run-42:researcher"
  },
  "tool": {
    "tool_call_id": "call-abc",
    "tool_class": "web_search",
    "status": "succeeded",
    "started_at_unix_ms": 1777312801080,
    "ended_at_unix_ms": 1777312801500,
    "duration_ms": 420.5
  }
}
```

Optional `tool` keys: `output_tokens`, `output_bytes`, `tool_name_hash`, `error_type` (useful on `tool_error`). Status values: `running`, `succeeded`, `error`, `cancelled`; synonyms `ok`/`success`, `failed`, `timeout`/`canceled` also deserialize.

## Dynamo `request_end` record

Emitted after the response stream finishes or is dropped. Omitted keys were not recorded on that path; see `AgentTraceRecord` / `AgentRequestMetrics` in `lib/llm/src/agents/trace/types.rs` for the full Rust schema.

```json
{
  "schema": "dynamo.agent.trace.v1",
  "event_type": "request_end",
  "event_time_unix_ms": 1777312801000,
  "event_source": "dynamo",
  "agent_context": {
    "session_type_id": "deep_research",
    "session_id": "research-run-42",
    "trajectory_id": "research-run-42:researcher",
    "parent_trajectory_id": "research-run-42:planner"
  },
  "request": {
    "request_id": "dynamo-request-id",
    "x_request_id": "llm-call-42",
    "model": "my-model",
    "output_tokens": 16,
    "replay": {
      "trace_block_size": 64,
      "input_length": 128,
      "input_sequence_hashes": [14879255164371896291, 274632075616497421]
    }
  }
}
```

By default we do not save the input/ouput payloads. In order to view these, use the built in Dynamo `audit_sink` functionality.

**Audit side-by-side** (same gzip/jsonl machinery):

```bash
# enable agent trace sinks
export DYN_AGENT_TRACE_SINKS=jsonl_gz
export DYN_AGENT_TRACE_OUTPUT_PATH=/tmp/dynamo-trace
# enable audit sinks
export DYN_AUDIT_SINKS=jsonl_gz
export DYN_AUDIT_OUTPUT_PATH=/tmp/dynamo-audit
export DYN_AUDIT_FORCE_LOGGING=true
```

After the run, correlate by id:

```bash
gzip -cd /tmp/dynamo-audit.*.jsonl.gz | jq -c '.event' > /tmp/audit.jsonl
gzip -cd /tmp/dynamo-trace.*.jsonl.gz | jq -c '.event' > /tmp/trace.jsonl
jq -s 'group_by(.request_id // .request.request_id)' /tmp/audit.jsonl /tmp/trace.jsonl
```

The result is a JSONL file where each line wraps the record:

```json
{
  "timestamp": 1234,
  "event": { "schema": "dynamo.agent.trace.v1", "...": "..." }
}
```

`timestamp` is sink-relative elapsed ms; use `event.event_time_unix_ms` for wall-clock ordering.

## Viewing traces in Perfetto

In order to visualize and optimize your agentic graph, we provide a utility to convert the agent trace JSONL files into a [Perfetto](https://ui.perfetto.dev/) trace file. We have found this to be extremely useful to pipeline agents that our team writes!

```bash
uv run --no-project python benchmarks/agent_trace/convert_to_perfetto.py \
  "${DYN_AGENT_TRACE_OUTPUT_PATH}".*.jsonl.gz \
  --output "${DYN_AGENT_TRACE_OUTPUT_PATH}.perfetto.json"
```

Open in [Perfetto UI](https://ui.perfetto.dev/). Flags: `--include-markers`, `--no-stages`, `--separate-stage-tracks`.

## [Experimental] Replaying agent traces using Mocker and Mooncake replay

You can use our offline engine mocker and replay fuctionality when you want to replay collected agent traces. Each trace saves hash-ids that can be used to simulate routing and KV cache behavior. Replay here means **synthesizing Mooncake-style request streams** for `python -m dynamo.replay`: only **`request_end`** events are converted. Being able to simulate tool loops and agent policies is a work in progress.

```bash
# convert agent trace to mooncake
cargo run -p dynamo-bench --bin agent_trace_to_mooncake -- \
  --input-path "${DYN_AGENT_TRACE_OUTPUT_PATH}".*.jsonl.gz \
  --output-file /tmp/dynamo-agent-trace.mooncake.jsonl
```

The binary prints **`trace_block_size`** — use that exact value for replay so hash segmentation matches what was recorded. Align mock engine block size with the same number (below: `TRACE_BLOCK_SIZE` in both `--trace-block-size` and `--extra-engine-args`). For more flags (online replay, reports), see [Mocker trace replay](../benchmarks/mocker-trace-replay.md).

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

`kv_router` needs **at least two** mock workers; for a single-worker smoke test use `--router-mode round_robin --num-workers 1`.

Converted rows use **per-request timestamps** as wall-clock arrivals; Mooncake `session_id` is left unset so overlapping LLM calls from the same trajectory replay in parallel. Mocker **simulates** KV behavior from engine/router settings.

ATIF alignment</summary>

Dynamo emits `dynamo.agent.trace.v1`, not full ATIF logs—but identifiers match [ATIF][atif-rfc] / [Harbor](https://github.com/harbor-framework/harbor) so you can join harness trajectories to Dynamo rows on `session_id` + `trajectory_id`. Dynamo omits conversational payload by design.

| Dynamo                 | Role                    |
| ---------------------- | ----------------------- |
| `session_id`           | Shared run id           |
| `trajectory_id`        | Branch within run       |
| `parent_trajectory_id` | Subagent link           |
| `session_type_id`      | Profile / workload type |

</details>

[atif-rfc]: https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md
