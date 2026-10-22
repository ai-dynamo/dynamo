---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Run an Agent on Dynamo
subtitle: Point a coding agent at a local Dynamo endpoint and see its requests in a trace
---

By the end of this guide, a coding agent is sending requests to a local Dynamo deployment, and you can see those requests grouped by session in a request trace.

You need a Dynamo source checkout (run commands from the repository root) and 2 GPUs.

## Start the Endpoint

```bash
bash examples/backends/sglang/launch/agg_agent.sh
```

This serves `zai-org/GLM-4.7-Flash` with SGLang at TP2 on port 8000. It also turns on the settings that matter for agent workloads:

| Setting | What it does for agents |
|---|---|
| `--router-mode kv` | Sends each request to the worker that already holds the most reusable KV cache. |
| `--kv-events-config` | Lets the worker report its cached blocks so the router can find reuse. |
| `--radix-eviction-policy priority` | When GPU cache fills, evicts low-priority entries first. It only has an effect once you [send priorities](priority-scheduling.mdx). |
| `--dyn-tool-call-parser`, `--dyn-reasoning-parser` | Turn model output into structured tool calls and reasoning. The values (`glm47`, `glm45`) are specific to GLM; change them with the model. |
| `--enable-anthropic-api` | Serves `/v1/messages` so Claude Code can connect. |
| `DYN_REQUEST_TRACE=1` | Records every request so you can analyze the run. |

With one worker, KV routing has nothing to choose between. It starts to matter when you add workers.

The script enables request tracing with the default gzip sink. It writes segments named `/tmp/dynamo-request-trace.NNNNNN.jsonl.gz`, or uses `DYN_REQUEST_TRACE_FILE_PATH` as the prefix if you set it, and prints the prefix at startup.

Check that the endpoint is up:

```bash
curl http://localhost:8000/v1/models
```

## Connect a Harness

Point your agent at `http://localhost:8000`. For Claude Code:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_MODEL=zai-org/GLM-4.7-Flash
export ANTHROPIC_SMALL_FAST_MODEL=zai-org/GLM-4.7-Flash
export CLAUDE_CODE_ATTRIBUTION_HEADER=0 # preserve KV cache hits
export ANTHROPIC_API_KEY=

claude
```

Setup for Codex, Pi, OpenCode, OpenClaw, and Hermes is in [Agent Harnesses](agent-harnesses.mdx). These harnesses send their own session headers, so you do not add `x-dynamo-session-id` yourself.

## Send a Labeled Request

To see the same behavior without a harness, send a request with a session ID:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'x-dynamo-session-id: quickstart:main' \
  -d '{"model":"zai-org/GLM-4.7-Flash","messages":[{"role":"user","content":"Say hello."}]}'
```

## Check the Trace

List the requests by session:

```bash
gzip -cd "${DYN_REQUEST_TRACE_FILE_PATH:-/tmp/dynamo-request-trace}".*.jsonl.gz \
  | jq -c '.event | select(.event_type=="request_end")
       | {session: .agent_context.session_id,
          input_tokens: .request.input_tokens,
          output_tokens: .request.output_tokens,
          cached_tokens: .request.cached_tokens}'
```

Each line is one LLM request. A coding agent produces many lines per session. `cached_tokens` is the number of input tokens served from KV cache, so it is the number to watch as a session grows. If you see requests with a null `session`, the harness is not sending a session header. See [Identify agent sessions](session-ids.mdx).

## Next

[Identify agent sessions](session-ids.mdx) explains the headers and how subagents are linked.
