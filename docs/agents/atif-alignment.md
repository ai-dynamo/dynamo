---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: ATIF Alignment
subtitle: Join Dynamo serving telemetry with agent trajectory traces
---

The [Agent Trajectory Interchange Format (ATIF)][atif-rfc] is a JSON format for
complete agent trajectories: user inputs, agent steps, tool calls, observations,
subagents, rewards, and evaluation metadata. ATIF is maintained as the
[Harbor framework][harbor] data format and is the canonical source of truth for
the schema referenced below.

Dynamo does not emit ATIF in the current trace path. Dynamo emits
`dynamo.agent.trace.v1`, a serving-oriented trace that records request timing,
token counts, cache metrics, queue depth, worker placement, and harness tool
events. The two formats are complementary.

[atif-rfc]: https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md
[harbor]: https://github.com/harbor-framework/harbor

## Identifier Alignment

Dynamo uses ATIF-aligned identifier names in `nvext.agent_context`:

| Dynamo field | ATIF role | Meaning |
|--------------|-----------|---------|
| `session_id` | `session_id` | Agent run identity. Multiple trajectories can share one session. |
| `trajectory_id` | `trajectory_id` | One parent or child trajectory within the run. |
| `parent_trajectory_id` | subagent relationship metadata | Optional parent trajectory for subagents. |
| `session_type_id` | producer-specific metadata | Reusable workload or profile class. |

This lets an ATIF file from a harness or eval framework join with Dynamo's
serving trace without renaming fields.

```text
ATIF trajectory
  session_id = research-run-42
  trajectory_id = research-run-42:searcher
        |
        | join on session_id + trajectory_id
        v
Dynamo trace records
  request timing, tokens, cache hit rate, queue depth, worker placement
```

## Ownership Boundary

Harnesses and eval frameworks should own semantic trajectory capture. They see
the prompts, observations, actions, tool arguments, tool outputs, rewards, and
subagent structure needed for debugging, SFT, RL, and evaluation.

Dynamo should own inference-serving telemetry. It sees the queueing, prefill,
decode, token, cache, routing, and worker-placement data needed to explain how
the inference engine served each trajectory.

Keeping these boundaries separate avoids forcing Dynamo's low-level serving
trace to become a payload-heavy trajectory log. Full-fidelity ATIF export should
come from the harness or from an offline join of harness/audit data with Dynamo
trace records.

## Worked Pattern: Joining a Harness Trajectory With Dynamo Telemetry

A harness that already writes ATIF (Harbor's `terminus-2`, OpenHands, Mini-SWE,
etc.) can be joined with Dynamo serving telemetry without changes to the ATIF
schema:

1. Run Dynamo with the shared sinks enabled so audit and trace land as
   `.jsonl.gz` segments side by side:

   ```bash
   export DYN_AGENT_TRACE_SINKS=jsonl_gz
   export DYN_AGENT_TRACE_OUTPUT_PATH=/tmp/dynamo-trace
   export DYN_AUDIT_SINKS=jsonl_gz
   export DYN_AUDIT_OUTPUT_PATH=/tmp/dynamo-audit
   export DYN_AUDIT_FORCE_LOGGING=true
   ```

2. Make sure each LLM request reaching Dynamo carries
   `nvext.agent_context.{session_type_id, session_id, trajectory_id}`. If the
   harness does not natively know about `nvext`, a thin OpenAI-passthrough
   that injects `agent_context` from environment variables keeps the harness
   source untouched.

3. After the run, merge Dynamo's audit and trace streams by `request_id` to
   recover request payload + serving metrics in a single record per LLM call.

4. Attach the merged stream to the harness trajectory in
   [`trajectory.extra`][atif-rfc] (and, when the harness exposes per-step
   token counts, fold per-call serving metrics into
   `step.metrics.extra`):

   ```json
   {
     "schema_version": "ATIF-v1.7",
     "session_id": "...",
     "steps": [...],
     "extra": {
       "dynamo": {
         "summary": { "n_records": 2, "total_input_tokens": 1632,
                      "total_output_tokens": 365, "mean_kv_hit_rate": 0.36 },
         "joined_records": [
           { "request_id": "...", "x_request_id": "...",
             "input": { "messages": [...], "input_tokens": 693, "cached_tokens": 0 },
             "output": { "content": "...", "output_tokens": 207,
                         "ttft_ms": 121.98, "kv_hit_rate": 0.0 },
             "worker": { "prefill_worker_id": ..., "decode_worker_id": ... }
           }
         ]
       }
     }
   }
   ```

The result still parses against ATIF v1.7 because all additions live under
optional `extra` fields. Consumers that do not understand the `dynamo` key can
ignore it; downstream RL/SFT/debugging tools that do can read TTFT, KV hit
rate, worker placement, and replay hashes per LLM call without re-running the
workload.

## Current Dynamo Scope

- Dynamo aligns identifier names with ATIF.
- Dynamo does not write native ATIF files in this trace path.
- Dynamo trace records intentionally omit prompt and response content.
- Full ATIF reconstruction requires harness trajectory data, audit payload data,
  or both.
- Future exporters can store Dynamo-specific metrics in [ATIF `extra` fields][atif-rfc]
  or produce a separate sidecar joined by `session_id` and `trajectory_id`.
