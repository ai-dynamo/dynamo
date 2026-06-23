---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Agent Simulation
subtitle: Capture an agent workload and replay it with DynoSim or AIPerf
---

Capture request shape and timing once. Replay it offline with DynoSim or against a live NVIDIA
Dynamo endpoint with AIPerf.

Request traces contain token lengths, prompt hashes, trajectory identity, and timing. They do not
contain prompts, responses, or tool arguments. Use the audit sink described in
[Agent Tracing](agent-tracing.md#audit-payloads) when you need payloads for inspection.

## Choose a Replay Path

| Goal | Path |
|---|---|
| Compare mock engine, KV router, or scheduling behavior without an HTTP endpoint | DynoSim offline replay |
| Exercise frontend parsing, routing, request rejection, and worker registration without GPUs | AIPerf against [live Mocker](../dynosim/mocker.md) |
| Measure a real model and transport stack | AIPerf against a live model endpoint |
| Inspect exact prompts, responses, or tool arguments | Audit sink; audit records are not replay inputs |

Both replay paths start from the same `dynamo.request.trace.v1` capture:

```mermaid
flowchart LR
    A["Agent workload through Dynamo"] --> T["Dynamo request trace"]
    T --> C["request_trace_to_mooncake --agentic"]
    C --> D["DynoSim offline replay"]
    T --> F["Trace with final markers"]
    F --> W["aiperf synthesize dynamo-trace"]
    W --> P["AIPerf live replay"]
    P --> M["Mocker-backed endpoint"]
    P --> R["Real model endpoint"]
```

The live AIPerf path requires a terminal marker on each trajectory. The offline DynoSim path does
not.

See [Trace Format Reference](../dynosim/trace-formats.md) for other inputs and mode constraints.

## Capture an Agent Workload

Run the agent through Dynamo with a supported trajectory header. Claude Code, Codex, OpenCode, and
generic Dynamo clients use the mappings in [Trajectory IDs](trajectory-ids.md#trajectory-id-inputs).

Enable the request trace sink on the frontend:

```bash
export DYN_REQUEST_TRACE=1
export DYN_REQUEST_TRACE_SINKS=jsonl_gz
export DYN_REQUEST_TRACE_OUTPUT_PATH=/tmp/dynamo-agent-trace
```

Run the workload, then inspect the captured request rows:

```bash
gzip -cd /tmp/dynamo-agent-trace.*.jsonl.gz | \
  jq -c '
    select(.event.event_type == "request_end")
    | {request_id: .event.request.request_id, agent_context: .event.agent_context}
  ' | \
  head
```

Each request row passed to the agentic converter must include `agent_context.trajectory_id`. Set
`parent_trajectory_id` for child trajectories. For precise tool duration and status, publish the
optional tool events described in [Tool Call Observability](agent-tracing.md#tool-call-observability).
Without those events, the converter preserves the elapsed gap between adjacent LLM requests.

## Replay Offline with DynoSim

### Convert the Capture

Convert the request rows to Agentic Mooncake:

```bash
cargo run -p dynamo-bench --bin request_trace_to_mooncake -- \
  --agentic \
  --input-path "${DYN_REQUEST_TRACE_OUTPUT_PATH}".*.jsonl.gz \
  --output-file /tmp/dynamo-agent-trace.agentic-mooncake.jsonl
```

The converter prints the row count and trace block size:

```text
Wrote 15 Agentic Mooncake rows to /tmp/dynamo-agent-trace.agentic-mooncake.jsonl
Trace block size: 64
```

Set `TRACE_BLOCK_SIZE` to the printed value so replay expands each `hash_id` correctly.

### Run a One-Worker Smoke Test

Start with round-robin routing and one worker:

```bash
TRACE_BLOCK_SIZE=64
uv run --no-sync python -m dynamo.replay \
  /tmp/dynamo-agent-trace.agentic-mooncake.jsonl \
  --trace-format agentic_mooncake \
  --trace-block-size "${TRACE_BLOCK_SIZE}" \
  --replay-mode offline \
  --router-mode round_robin \
  --num-workers 1 \
  --extra-engine-args "{\"block_size\":${TRACE_BLOCK_SIZE}}" \
  --report-json /tmp/dynamo-agent-trace.replay-report.json
```

The command prints a summary and writes run metrics to the report file.

### Compare Router Behavior

After the smoke test passes, run the same trace through the KV router:

```bash
TRACE_BLOCK_SIZE=64
uv run --no-sync python -m dynamo.replay \
  /tmp/dynamo-agent-trace.agentic-mooncake.jsonl \
  --trace-format agentic_mooncake \
  --trace-block-size "${TRACE_BLOCK_SIZE}" \
  --replay-mode offline \
  --router-mode kv_router \
  --num-workers 4 \
  --extra-engine-args "{\"block_size\":${TRACE_BLOCK_SIZE}}" \
  --report-json /tmp/dynamo-agent-trace.kv-router-report.json
```

Change one router or engine setting at a time and compare the report JSON files. Agentic Mooncake
currently supports offline aggregated replay with trace timestamps. It does not support online
DynoSim mode, disaggregated replay, or `--replay-concurrency`.

## Replay Against a Live Endpoint with AIPerf

**Experimental.** Use AIPerf to test frontend parsing, request rejection, routing, transport, and
live worker behavior.

This path requires:

- an AIPerf build with `aiperf synthesize dynamo-trace` and
  `--use-dynamo-conv-aware-routing`
- a Dynamo build that derives backend session lifecycle from trajectory headers

Older Dynamo builds that require client-generated `nvext.session_control` are not compatible with
this header-only path.

> [!IMPORTANT]
> The AIPerf converter requires exactly one request with `trajectory_final=true` at the end of each
> trajectory. Set `x-dynamo-trajectory-final: true` on the final request. Native Claude Code,
> Codex, and OpenCode identity headers do not supply this marker, so use DynoSim for those captures
> unless the harness adds it.

The AIPerf converter reads uncompressed JSONL. Merge the captured segments, then convert them to a
Weka trace directory. The output directory must be absent or empty.

```bash
gzip -cd /tmp/dynamo-agent-trace.*.jsonl.gz > /tmp/dynamo-agent-trace.jsonl

aiperf synthesize dynamo-trace /tmp/dynamo-agent-trace.jsonl \
  --output /tmp/dynamo-agent-weka
```

Replay the converted workload against a Dynamo endpoint:

```bash
AIPERF_DATASET_WEKA_SPLIT_FLATTENED_AGENTS=false \
aiperf profile \
  --url localhost:8000 \
  --model my-model \
  --endpoint-type chat \
  --input-file /tmp/dynamo-agent-weka \
  --custom-dataset-type weka_trace \
  --fixed-schedule \
  --fixed-schedule-auto-offset \
  --use-dynamo-conv-aware-routing
```

Set `AIPERF_DATASET_WEKA_SPLIT_FLATTENED_AGENTS=false` to preserve trajectory boundaries.
`--use-dynamo-conv-aware-routing` sends trajectory, parent, and final markers as headers without
changing request bodies. Point `--url` at a Mocker-backed frontend or a real model deployment.

The converter requires one final request per trajectory. It supports one level of child trajectories
and rejects deeper trees.

## How the Request DAG Is Built

Each node is one LLM request:

- Requests in one trajectory run in recorded order.
- A child trajectory starts after the last parent request that finished before the child arrived.
- The next parent request waits when it arrived after the child completed.
- The edge to the next request carries tool-execution and agent-overhead delays.

DynoSim schedules requests from `wait_for`, `delay`, and `tool_wait_ms`. `branches`, `prefix_reset`,
and `tool_events` preserve structure for analysis but do not control scheduling. `hash_ids`,
`input_length`, and `output_length` preserve prompt shape without storing prompt text.

The converter infers spawn and join edges from request timing. It does not recreate application
logic that the trace never observed.

For Claude Code, Codex, OpenCode, and text inputs, see
[Trace Format Reference](../dynosim/trace-formats.md#coding-agent-and-text-inputs).
