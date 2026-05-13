---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Use Pi-Mono with Dynamo
subtitle: Drive a real coding agent through Dynamo with agent context and tool tracing
---

[Pi-Mono](https://github.com/mariozechner/pi-mono) is an open-source coding-agent harness whose clean plugin architecture has made it a popular substrate for patterns like subagents and planner/implementer loops. The [`pi-dynamo-provider`](https://github.com/ai-dynamo/pi-dynamo-provider) extension uses that plugin surface to register Dynamo as a Pi model provider — it is **not** a proxy or interceptor. It runs in-process, adds Dynamo's [`agent_context`](agent-tracing.md) and [`agent_hints`](agent-hints.md) to each request, and emits Pi's tool lifecycle events to Dynamo over ZMQ. No harness patches, no sidecar.

This is the recommended way to exercise agent tracing and agent hints end-to-end.

## Why run Pi through Dynamo

You can already point Pi at any OpenAI-compatible endpoint (Ollama, vLLM, a hosted API). Routing through Dynamo gives you two things you don't get from a plain hosting layer:

- **Harness-aware observability.** Pi's session and trajectory IDs flow into Dynamo's `request_end` traces, and Pi's tool spans land on the same timeline. One Perfetto view shows LLM requests, prefill/decode stages, and tool calls together.
- **Harness-aware orchestration.** Once Dynamo knows which trajectory a request belongs to, it can act on agent hints (priority, expected output length, speculative prefill) for smarter scheduling and KV-aware routing. That same trajectory awareness is what lets backends like [SGLang](../backends/sglang/agents.md) apply priority-based radix eviction and session-scoped KV isolation.

The integration works against any Dynamo backend — vLLM, SGLang, or TRT-LLM — without backend-specific glue.

## What the extension does

- Registers a `dynamo` provider in Pi: `pi --model dynamo/<model-id>`.
- Discovers models from Dynamo's `/v1/models`.
- Injects `nvext.agent_context` (session/trajectory IDs) into every chat-completion request.
- Adds `x-request-id` when one is not already set.
- Relays Pi's `tool_start` / `tool_end` / `tool_error` events to Dynamo over ZMQ so LLM and tool spans share one trace.

```mermaid
sequenceDiagram
    participant Pi as Pi-Mono
    participant Provider as pi-dynamo-provider
    participant Dynamo as Dynamo frontend
    participant Trace as Agent trace sink

    Pi->>Provider: streamSimple(model, context)
    Provider->>Dynamo: POST /v1/chat/completions<br/>nvext.agent_context, x-request-id
    Dynamo-->>Provider: SSE chunks
    Dynamo->>Trace: request_end
    Pi->>Provider: tool_execution_start / _end
    Provider->>Dynamo: ZMQ PUSH tool record
    Dynamo->>Trace: tool_start / tool_end
```

## Quickstart

### 1. Install the provider

```bash
pi install npm:pi-dynamo-provider
```

Or from source:

```bash
git clone git@github.com:ai-dynamo/pi-dynamo-provider.git
cd pi-dynamo-provider
npm install && npm run build
pi install /absolute/path/to/pi-dynamo-provider
```

### 2. Launch Dynamo with tracing enabled

The repo ships a launcher that builds Dynamo, starts an OpenAI-compatible frontend plus one SGLang worker, and turns on JSONL agent tracing and the ZMQ tool-event ingest:

```bash
./scripts/install-dynamo.sh
./scripts/launch-agg-agent.sh
```

By default this serves `zai-org/GLM-4.7-Flash` on one GPU. When it's ready it prints the exact Pi environment variables to paste into another shell. The launcher is bundled for convenience — the provider works equally well against any Dynamo backend (vLLM, SGLang, TRT-LLM) started by your own deployment, as long as the agent-trace env vars below are set.

<Note>
Equivalent manual launch: set `DYN_AGENT_TRACE_SINKS=jsonl`, `DYN_AGENT_TRACE_OUTPUT_PATH=<path>`, and `DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT=tcp://127.0.0.1:20390` before starting `dynamo.frontend` and your worker. See [Agent Tracing → Enable output](agent-tracing.md#enable-output) for the full variable reference.
</Note>

### 3. Point Pi at Dynamo

```bash
export DYNAMO_BASE_URL=http://127.0.0.1:18083/v1
export DYNAMO_API_KEY=dummy

export DYN_AGENT_SESSION_TYPE_ID=pi_coding_agent
export DYN_AGENT_SESSION_ID=pi-demo-001
export DYN_AGENT_TOOL_EVENTS_ZMQ_ENDPOINT=tcp://127.0.0.1:20390

pi --model dynamo/zai-org/GLM-4.7-Flash \
   -p "Run the tests in this folder, fix the smallest bug, and rerun the tests."
```

`DYN_AGENT_SESSION_ID` becomes the trace's `session_id`; if `DYN_AGENT_TRAJECTORY_ID` is unset, Pi's session id is used as the trajectory id.

### 4. View the trace in Perfetto

```bash
python benchmarks/agent_trace/convert_to_perfetto.py \
  /tmp/dynamo-agent-trace.jsonl \
  --include-markers \
  --separate-stage-tracks \
  --output /tmp/dynamo-agent-trace.perfetto.json
```

Open the result at [ui.perfetto.dev](https://ui.perfetto.dev). You'll see:

- `dynamo.llm` spans for each LLM request.
- `dynamo.llm.stage` spans for prefill/decode when Dynamo records them.
- `dynamo.agent.tool` spans for every Pi tool invocation.

## Pi environment variables

| Variable                             | Default                    | Purpose                                                                |
| ------------------------------------ | -------------------------- | ---------------------------------------------------------------------- |
| `DYNAMO_BASE_URL`                    | `http://127.0.0.1:8000/v1` | Dynamo OpenAI-compatible endpoint root.                                |
| `DYNAMO_API_KEY`                     | `dynamo-local`             | Bearer token. Local Dynamo usually accepts any value.                  |
| `DYN_AGENT_SESSION_TYPE_ID`          | `pi_coding_agent`          | Stable workload class for the trace.                                   |
| `DYN_AGENT_SESSION_ID`               | unset                      | Session/run id. Falls back to Pi's session id for tool events.         |
| `DYN_AGENT_TRAJECTORY_ID`            | unset                      | Trajectory id override; defaults to Pi's session id per request.       |
| `DYN_AGENT_PARENT_TRAJECTORY_ID`     | unset                      | Parent trajectory id for nested or subagent workflows.                 |
| `DYN_AGENT_TOOL_EVENTS_ZMQ_ENDPOINT` | unset                      | Dynamo-bound ZMQ PULL endpoint Pi connects to for tool events.         |
| `DYN_AGENT_TOOL_EVENTS_ZMQ_TOPIC`    | `agent-tool-events`        | First ZMQ frame; must match `DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_TOPIC`.   |

See the [provider README](https://github.com/ai-dynamo/pi-dynamo-provider) for the full variable list, aliases, and ZMQ wire format.

## Troubleshooting

| Symptom                                   | Likely cause                                            | Fix                                                                                       |
| ----------------------------------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `pi` reports the model is unknown         | `/v1/models` returned empty during Pi startup           | `curl -s "$DYNAMO_BASE_URL/models"` to confirm; restart Pi after Dynamo is ready.         |
| LLM spans appear, tool spans do not       | Tool-event endpoint not configured on both sides        | Set `DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT` on Dynamo and `DYN_AGENT_TOOL_EVENTS_ZMQ_ENDPOINT` on Pi to the same value. |
| Tool spans appear, request spans do not   | Dynamo trace sinks not enabled                          | Set `DYN_AGENT_TRACE_SINKS=jsonl` and `DYN_AGENT_TRACE_OUTPUT_PATH` on Dynamo.             |
| Authentication fails                      | Dynamo expects a specific token                         | Set `DYNAMO_API_KEY` to match your deployment. Local Dynamo usually accepts `dummy`.      |

## Further reading

- [pi-dynamo-provider repo](https://github.com/ai-dynamo/pi-dynamo-provider) — install, scripts, and source.
- [Agent Tracing](agent-tracing.md) — the underlying trace protocol and `request_end` schema.
- [Agent Hints](agent-hints.md) — per-request hints (`priority`, `osl`, `speculative_prefill`) Pi-Mono can forward via `nvext.agent_hints`.
