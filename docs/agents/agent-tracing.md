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

The fast path is one environment variable:

```bash
export DYN_AGENT_TRACE=1
```

That picks `jsonl_gz` output at `/tmp/dynamo-agent-trace.*.jsonl.gz` and binds
the harness tool-event ZMQ endpoint at `tcp://127.0.0.1:20390`. Any of the
per-knob variables below still wins when set explicitly, so you only need to
reach for them to relocate output, add `stderr`, or tune buffers.

To relocate captures only:

```bash
export DYN_AGENT_TRACE=1
export DYN_AGENT_TRACE_OUTPUT_PATH=/mnt/captures/run-42
```

<details>
<summary>All agent trace environment variables</summary>

| Variable                                   |        Required         | Default (when `DYN_AGENT_TRACE=1`) | Notes                                                                                |
| ------------------------------------------ | :---------------------: | ---------------------------------- | ------------------------------------------------------------------------------------ |
| `DYN_AGENT_TRACE`                          |        Master switch    | unset                              | Truthy (`1`, `true`, `on`, `yes`) enables tracing with all defaults below.            |
| `DYN_AGENT_TRACE_SINKS`                    |           No            | `jsonl_gz`                         | `jsonl`, `jsonl_gz`, `stderr`, or comma-separated (e.g. `jsonl_gz,stderr`).          |
| `DYN_AGENT_TRACE_OUTPUT_PATH`              |           No            | `/tmp/dynamo-agent-trace`          | File path for `jsonl`; segment **prefix** for `jsonl_gz` → `prefix.NNNNNN.jsonl.gz`. |
| `DYN_AGENT_TRACE_CAPACITY`                 |           No            | `1024`                             | Trace bus capacity.                                                                  |
| `DYN_AGENT_TRACE_JSONL_BUFFER_BYTES`       |           No            | `1048576`                          | Buffer / gzip batch threshold.                                                       |
| `DYN_AGENT_TRACE_JSONL_FLUSH_INTERVAL_MS`  |           No            | `1000`                             | Flush interval.                                                                      |
| `DYN_AGENT_TRACE_JSONL_GZ_ROLL_BYTES`      |           No            | `268435456`                        | Roll gzip segment by uncompressed bytes.                                             |
| `DYN_AGENT_TRACE_JSONL_GZ_ROLL_LINES`      |           No            | unset                              | Optional roll by line count.                                                         |
| `DYN_AGENT_TRACE_REPLAY_HASHES`            |           No            | on                                 | Falsey (`0`, `no`, …) disables `replay` hashes on requests.                          |
| `DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT` |           No            | `tcp://127.0.0.1:20390`            | PULL bind address for tool records.                                                  |
| `DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_TOPIC`    |           No            | unset                              | If set, first ZMQ frame must match.                                                  |

Without `DYN_AGENT_TRACE=1`, tracing is off unless you explicitly set
`DYN_AGENT_TRACE_SINKS` or `DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT` -- both
the legacy enable paths still work.

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

## [Experimental] Replaying agent traces using agentic Mooncake replay

You can convert a collected agent trace into an **agentic Mooncake** trace and replay it with
`python -m dynamo.replay`. The converter uses Dynamo `request_end` rows for request timing, token
lengths, worker placement, and replay hashes. It also uses terminal harness tool rows
(`tool_end` / `tool_error`) to preserve tool-wait time between dependent LLM requests.

```bash
cargo run -p dynamo-bench --bin agent_trace_to_mooncake -- \
  --agentic \
  --input-path "${DYN_AGENT_TRACE_OUTPUT_PATH}".*.jsonl.gz \
  --output-file /tmp/dynamo-agent-trace.agentic-mooncake.jsonl
```

The binary prints **`trace_block_size`**. Use that exact value for replay so hash segmentation
matches what Dynamo recorded. Align the mock engine block size with the same number in
`--extra-engine-args`.

```bash
TRACE_BLOCK_SIZE=128
uv run --no-sync python -m dynamo.replay /tmp/dynamo-agent-trace.agentic-mooncake.jsonl \
  --trace-format agentic_mooncake \
  --trace-block-size "${TRACE_BLOCK_SIZE}" \
  --replay-mode offline \
  --router-mode kv_router \
  --num-workers 4 \
  --extra-engine-args "{\"block_size\":${TRACE_BLOCK_SIZE}}" \
  --report-json /tmp/dynamo-agent-trace.replay-report.json
```

`kv_router` needs **at least two** mock workers; for a single-worker smoke test use
`--router-mode round_robin --num-workers 1`.

Agentic Mooncake rows preserve:

- `request_id`: the LLM request row identity.
- `session_id`: the Dynamo `trajectory_id`.
- `wait_for`: request ids that must complete before this row becomes eligible.
- `branches`: child request ids spawned from this row.
- `prefix_reset`: first request in a trajectory.
- `delay`: non-tool delay after dependencies finish.
- `tool_wait_ms`: tool time after dependencies finish, parallel-aware (the union
  of overlapping spans rather than their sum).
- `tool_events`: per-tool spans attributed to this LLM request, each carrying
  `tool_call_id`, `tool_class`, `status`, `started_at_unix_ms`, `ended_at_unix_ms`,
  `duration_ms`, and optional `output_bytes` / `output_tokens` / `error_type`.
- `hash_ids`, `input_length`, and `output_length`: prompt-prefix and length data for mocker replay.

Rows with no `wait_for` use their `timestamp` as the replay start time. Rows with dependencies wait
for all listed requests to complete, then wait `delay + tool_wait_ms` before dispatch. For more
flags and engine settings, see [Mocker trace replay](../benchmarks/mocker-trace-replay.md).

<details>
<summary>ATIF alignment</summary>

Dynamo emits `dynamo.agent.trace.v1`, not full ATIF logs—but identifiers match [ATIF][atif-rfc] / [Harbor](https://github.com/harbor-framework/harbor) so you can join harness trajectories to Dynamo rows on `session_id` + `trajectory_id`. Dynamo omits conversational payload by design.

| Dynamo                 | Role                    |
| ---------------------- | ----------------------- |
| `session_id`           | Shared run id           |
| `trajectory_id`        | Branch within run       |
| `parent_trajectory_id` | Subagent link           |
| `session_type_id`      | Profile / workload type |

</details>

[atif-rfc]: https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md
