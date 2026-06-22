---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Agent Quickstart
subtitle: Send your first agent-aware request to a running Dynamo deployment
---

This is the fastest path to an agent-aware request. You start an ordinary Dynamo
deployment, send a normal OpenAI-compatible request, then add Dynamo's agent
metadata to that same request. The metadata is plain JSON under `nvext` - no SDK,
client library, or harness changes required. For a full coding-agent harness
wired end to end, see [Use Pi-Mono with Dynamo](pi-mono.md).

## Prerequisites

A running Dynamo frontend and worker. If you don't have one, start the smallest
possible deployment (same as the [main Quick Start](https://github.com/ai-dynamo/dynamo#quick-start)):

```bash
python3 -m dynamo.frontend --http-port 8000 --discovery-backend file > /dev/null 2>&1 &
python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --discovery-backend file &
```

Agent metadata is backend-agnostic - substitute `dynamo.vllm` or `dynamo.trtllm`
and the requests below work unchanged.

## 1. Send a Normal Request

Confirm the deployment answers a plain request first:

```bash
curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen3-0.6B",
  "messages": [{"role": "user", "content": "Hello!"}],
  "max_tokens": 100
}' | jq
```

## 2. Add Agent Context (Trace Identity)

`agent_context` is passive identity that ties every LLM call in a run to a
session and a trajectory. It does not change routing or output - it tags the
request so traces can be joined later, across LLM calls, tool calls, and external
trajectory files.

```bash
curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen3-0.6B",
  "messages": [{"role": "user", "content": "Plan the next step."}],
  "max_tokens": 100,
  "nvext": {
    "agent_context": {
      "session_type_id": "deep_research",
      "session_id": "research-run-42",
      "trajectory_id": "research-run-42:researcher"
    }
  }
}' | jq
```

The three context fields are required when you send `agent_context`. See
[Agent Tracing](agent-tracing.md#adding-trace-context-to-each-llm-call) for the
full field reference, including `parent_trajectory_id` for subagents.

## 3. Add Agent Hints (Serving Intent)

`agent_hints` is active intent the deployment can act on - request `priority`,
expected output length (`osl`), and `speculative_prefill`. Send hints alongside
the same context:

```bash
curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen3-0.6B",
  "messages": [{"role": "user", "content": "Continue the report."}],
  "max_tokens": 1024,
  "nvext": {
    "agent_context": {
      "session_type_id": "deep_research",
      "session_id": "research-run-42",
      "trajectory_id": "research-run-42:writer"
    },
    "agent_hints": {
      "priority": 5,
      "osl": 1024
    }
  }
}' | jq
```

Hints are accepted on any deployment, but each one only takes effect where the
matching feature is enabled - for example, `priority` needs router and backend
priority support, and `osl` needs `--router-track-output-blocks`. See
[Agent Hints](agent-hints.md) for the full hint list and the per-backend support
matrix, and [Priority Scheduling](priority-scheduling.md) for priority semantics.

## 4. (Optional) Capture a Trace

To see what Dynamo measured for these requests, enable tracing **before** you
start the worker:

```bash
export DYN_AGENT_TRACE=1
python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --discovery-backend file &
```

Dynamo writes one `request_end` record per request - carrying your
`agent_context`, output tokens, and autodetected tool-call and finish metadata -
to `/tmp/dynamo-agent-trace.*.jsonl.gz`. From there you can visualize the run in
Perfetto. See [Agent Tracing → Enable output](agent-tracing.md#enable-output) for
the sink options and the Perfetto converter.

## Next Steps

| Go deeper | Page |
|-----------|------|
| The full hint list and backend support matrix | [Agent Hints](agent-hints.md) |
| The trace protocol, `request_end` schema, and Perfetto view | [Agent Tracing](agent-tracing.md) |
| Priority semantics across router, engine, and cache | [Priority Scheduling](priority-scheduling.md) |
| A real coding agent wired through Dynamo end to end | [Use Pi-Mono with Dynamo](pi-mono.md) |
| Backend-specific behavior (SGLang) | [SGLang for Agentic Workloads](../backends/sglang/agents.md) |
