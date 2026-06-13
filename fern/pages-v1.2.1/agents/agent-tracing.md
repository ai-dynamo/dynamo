---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Agent Tracing
subtitle: Attach trajectory identity and export Dynamo request and tool-event telemetry
---

Agent tracing records **who** called (`nvext.agent_context`) and **what Dynamo measured** on each LLM request (`request_end`). Tool-call understanding is built in: Dynamo **autodetects** tool calls and finish reasons from the response stream and records them as `finish_reason_metadata` on every request — no harness instrumentation. Richer **harness tool spans** (`tool_*`: tool timing, status, output sizes) are an optional add-on. Context is passive—it does not steer routing or caching. Output is best-effort profiling data, not an audit log.

**Flow:** Harness sends chat completions with `agent_context` → Dynamo emits `request_end` (with autodetected `finish_reason_metadata`) to trace sinks. *Optionally*, a harness also publishes its own tool events over ZMQ → same sinks.

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
| `trajectory_final`     |    No    | `true` marks the trajectory's last request — a cleanup hint. |

`trajectory_final` is an optional terminal marker: set it to `true` to signal that a
trajectory is finished. Lifecycle-aware backends use it to release whatever
per-trajectory state they hold (scheduling bookkeeping, routing affinity, cached
identity) right away instead of waiting for an idle timeout; backends that don't track
per-trajectory lifecycle ignore it.

Send it as a **dedicated minimal request** (e.g. `max_tokens: 1` with a placeholder
message), not piggybacked on a real turn. A reactive agent loop only learns a turn was
terminal from its *response*, so the run's end is typically known only after the last
real turn already returned — there is no live turn left to flag. (A harness with a hard
turn budget may know earlier, but early termination still leaves the real last turn
unflagged, so a post-hoc close is the robust contract.) Because a backend that acts on
the marker may skip generation entirely, the request body is just a carrier — keep it
minimal.

No Dynamo imports are required in the harness — `agent_context` is plain JSON under `nvext`; just propagate it across threads/processes wherever those paths call the model.

## Enable output

The fast path is one environment variable:

```bash
export DYN_AGENT_TRACE=1
```

That picks `jsonl_gz` output at `/tmp/dynamo-agent-trace.*.jsonl.gz`. Tool-call
understanding works immediately from `request_end` finish metadata — no harness
tooling and no sockets (the optional ZMQ tool-event ingress is opt-in; see
[Tool call observability](#tool-call-observability)). Any of the per-knob variables
below still wins when set explicitly, so you only need to reach for them to relocate
output, add `stderr`, or tune buffers.

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
| `DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT` |           No            | unset (opt-in)                     | Set a PULL bind address (e.g. `tcp://127.0.0.1:20390`) to enable tool-event ingress. |
| `DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_TOPIC`    |           No            | unset                              | If set, first ZMQ frame must match.                                                  |

Without `DYN_AGENT_TRACE=1`, tracing is off; the other variables only
take effect once the master switch is on.

</details>

## Dynamo `request_end` record

Emitted after the response stream finishes or is dropped. Carries `agent_context`,
`output_tokens`, and the autodetected `finish_reason_metadata` (tool-call names +
finish reasons). `request_id` correlates with audit rows; the `replay` block feeds
Mooncake replay (disable with `DYN_AGENT_TRACE_REPLAY_HASHES=0`). Tool-call metadata
is ids and names only — arguments are intentionally not stored.

<details>
<summary>Full <code>request_end</code> record</summary>

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
    "model": "my-model",
    "output_tokens": 16,
    "finish_reason_metadata": {
      "finish_reason": "tool_calls",
      "backend_finish_reason": "stop",
      "stop_reason": "END",
      "tool_calls": [
        {
          "choice_index": 0,
          "tool_call_index": 0,
          "id": "call-abc",
          "name": "web_search"
        }
      ],
      "choices": [
        {
          "choice_index": 0,
          "finish_reason": "tool_calls",
          "backend_finish_reason": "stop",
          "stop_reason": "END"
        }
      ]
    },
    "replay": {
      "trace_block_size": 64,
      "input_length": 128,
      "input_sequence_hashes": [14879255164371896291, 274632075616497421]
    }
  }
}
```

`finish_reason_metadata` is optional. `finish_reason` is the final OpenAI-compatible
reason after parser rewrites (e.g. `tool_calls`); `backend_finish_reason` /
`stop_reason` come from the backend stop path. Top-level finish fields summarize the
single-choice case; `choices` keeps per-choice finish fields when `n > 1`. For chat
streams, finish metadata is recorded after parser/jail rewrites; completion streams
record the final OpenAI-compatible completion finish reason. See `AgentTraceRecord` /
`AgentRequestMetrics` in `lib/llm/src/agents/trace/types.rs` for the full Rust schema.

</details>

## Tool call observability

**Default — autodetected, no harness work.** Dynamo parses each response stream and records the
tool calls the model made into [`request_end.finish_reason_metadata`](#dynamo-request_end-record):
the per-turn `finish_reason` and each call's `name` and `id` (arguments are never stored). Active
whenever `DYN_AGENT_TRACE=1` and the worker runs a tool-call parser (`--dyn-tool-call-parser …`).
This tells you *what* the agent called and *when each turn ended*.

You can also recover **tool-wait time offline, without any tool events**. Within a trajectory the
agent is sequential, so the gap between one turn finishing and the next arriving is the tool +
agent-overhead time:

```
tool_wait(turn N) ~= next.request_received_ms - this.event_time_unix_ms
```

`request_received_ms` is stamped at the frontend **before** the request enters the router
queue/pause, so server wait time lands in each request's own duration, not in the inter-turn gap —
the estimate holds under load. For agentic replay that gap *is* the inter-request delay you would
inject, so autodetect alone reproduces realistic arrival timing. It cannot *split* tool execution
from agent overhead (you get the sum, as the wall-clock union of any parallel calls).

<details>
<summary>Optional — explicit tool events (ZMQ)</summary>

Opt-in: set `DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT` to bind the ingress, and have the harness
publish. Use it **only** when you need what autodetection and the timing gap can't give: the
*attribution* of tool time, per-tool `duration_ms`, `status` (succeeded/error/cancelled), and output
sizes. Nothing emits tool events on its own.

Wire format: `[topic, seq_be_u64, msgpack(AgentTraceRecord)]`. Use a background publisher, bounded queue, monotonic sequence, and PUSH with HWM. **Terminal** `tool_end` / `tool_error` rows should carry timing (`started_at_unix_ms`, `ended_at_unix_ms`, `duration_ms`) even if `tool_start` was dropped. Same `agent_context` as the surrounding LLM calls; `tool_call_id` unique per trajectory. Join offline on `session_id`, `trajectory_id`, `tool_call_id`.

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

Optional `tool` keys: `output_tokens`, `output_bytes`, `tool_name_hash`, `error_type`. Status values: `running`, `succeeded`, `error`, `cancelled`; synonyms `ok`/`success`, `failed`, `timeout`/`canceled` also deserialize.

</details>

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

Request slices include flattened finish metadata when present, such as `finish.finish_reason`,
`finish.backend_finish_reason`, `finish.stop_reason`, `finish.tool_call_count`,
`finish.tool_call_names`, and per-choice summaries like `finish.choice_finish_reasons`.

## [Experimental] Replaying agent traces using agentic Mooncake replay

You can convert a collected agent trace into an **agentic Mooncake** trace and replay it with
`python -m dynamo.replay`. The converter uses Dynamo `request_end` rows for request timing, token
lengths, worker placement, and replay hashes. It also uses terminal harness tool rows
(`tool_end` / `tool_error`) to preserve tool-wait time between dependent LLM requests.

Replay ignores non-replay request fields such as `finish_reason_metadata`; use the
Perfetto view above when you want to inspect final finish reasons, backend stop
signals, or complete tool-call metadata inside the trace.

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
flags and engine settings, see [DynoSim Runs](../dynosim/runs.md).

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
