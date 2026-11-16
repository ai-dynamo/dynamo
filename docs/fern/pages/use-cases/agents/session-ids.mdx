---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Identify Agent Sessions
subtitle: Tell Dynamo which requests belong to which agent session
---

By the end of this guide, every LLM request from your agent carries a session ID, and subagent requests point to their parent. Dynamo uses that identity to group requests in traces and replay, to keep a session on one worker when you enable [session affinity](session-affinity.md), and for any session-aware routing policy you configure.

Most coding agents send a session ID on their own. See [Connect an Agent Harness](agent-harnesses.mdx) to confirm yours does. If you are building a custom client, send the header below on every request in a reasoning chain:

```bash title="Custom client with session ID" {4}
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-dummy' \
  -H 'x-dynamo-session-id: research-run-42:researcher' \
  -d '{"model":"my-model","messages":[{"role":"user","content":"..."}]}'
```

## Use One ID Per Reasoning Chain

Use one `session_id` for one agent reasoning and tool-use chain. A root agent, planner, researcher subagent, or OpenCode subtask can each have its own ID. Child sessions can also send a parent ID so traces and replay tools can rebuild the tree. Some academic papers call this a `program_id`.

<a id="session-id-inputs"></a>

## Session ID Inputs

Custom clients should send the canonical Dynamo headers. When `X-Dynamo-Session-ID` is present, Dynamo uses it and `X-Dynamo-Parent-Session-ID` instead of any agent-native identity values.

<ParamField path="X-Dynamo-Session-ID" type="string" required={true}>
  One reasoning or tool chain inside the run. Maps to `agent_context.session_id`.
</ParamField>

<ParamField path="X-Dynamo-Parent-Session-ID" type="string">
  Parent session when using subagents. Maps to `agent_context.parent_session_id`.
</ParamField>

<ParamField path="X-Dynamo-Session-Final" type="boolean">
  Set to `true` on a dedicated minimal request when the session ends. Lifecycle-aware consumers can release per-session state immediately instead of waiting for an idle timeout. This works with canonical or agent-native session identity and maps to `agent_context.session_final`.
</ParamField>

## Native Agent Headers

Dynamo also recognizes the current stable identity headers emitted by the following coding agents. The [frontend API surface compliance test](https://github.com/ai-dynamo/dynamo/blob/main/tests/frontend/test_frontend_api_surface_compliance.py) catches header changes as coding agents evolve. See [Connect an Agent Harness](agent-harnesses.mdx) for packaged setup.

<Tabs>
  <Tab title="Claude Code">
    Dynamo reads Claude Code headers and normalizes them to `session_id` and `parent_session_id`:

    - `x-claude-code-session-id` for root turns
    - `x-claude-code-agent-id` for child-agent turns
    - `x-claude-code-parent-agent-id` for nested children; top-level children fall back to `x-claude-code-session-id`
  </Tab>
  <Tab title="Codex">
    Dynamo maps Codex's per-agent thread headers:

    - `thread-id` to `session_id`
    - `x-codex-parent-thread-id` to `parent_session_id` for child threads

  </Tab>
  <Tab title="OpenCode">
    Dynamo maps:

    - `x-session-id` to `session_id`
    - `x-parent-session-id` to `parent_session_id` when present
  </Tab>
  <Tab title="Custom client">
    Send the canonical headers directly:

    ```bash title="Session and parent session" {4-5}
    curl http://localhost:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -H 'Authorization: Bearer sk-dummy' \
      -H 'x-dynamo-session-id: research-run-42:researcher' \
      -H 'x-dynamo-parent-session-id: research-run-42:planner' \
      -d '{"model":"my-model","messages":[{"role":"user","content":"..."}]}'
    ```
  </Tab>
</Tabs>

<Warning>
Session identity is passive metadata. Sending a session ID does not enable sticky sessions or change request placement. Tracing records the identity when `DYN_REQUEST_TRACE=1` is enabled, and a session-aware routing policy can consume it only when that policy is configured separately. To keep a session on one worker, see [Maintain session affinity](session-affinity.md).
</Warning>

<a id="agent-headers"></a>

## Captured Agent Headers

Dynamo carries raw coding-agent HTTP headers in `AgentContext.agent_headers` without interpreting them. Capture includes the open `x-claude-code-*` and `x-codex-*` families alongside supported session and subagent headers. Names within those families are not hardcoded: new headers are captured without a Dynamo schema change. Header names and values can change as coding harnesses evolve. Plugin authors and trace consumers own any parsing or normalization between harnesses.

Names are lowercase. Values remain strings, including malformed JSON, percent encoding, whitespace, empty values, and repeated values in their received order. Core code only normalizes session identity and the explicit Dynamo terminal marker.

Capture requires a recognized session identity and is bounded to 64 header values, 16 KiB per value, and 32 KiB total, counting the name once per value. Sensitive-marked and non-text values are skipped. Values exceeding the limits are omitted whole; retained values are never truncated. Missing metadata means no observation was captured, not that an event did not occur. Ordinary authentication, cookie, and unrelated protocol headers are excluded. Do not place credentials in an agent metadata header; these values are exposed to plugins and request traces.

The map travels through the frontend and worker request envelope and is available through plugin `SessionContext.agent_headers()`. It contains HTTP headers only; request-body metadata is not captured. The default routing policy does not interpret the map. See [Trace an Agent Run](agent-tracing.md#dynamo-request_end-record) for the request-trace format and [Custom Worker Selection](../../developer-guide/knowledge-base/modular-components/router/custom-worker-selection.mdx#session-context) for plugin access.

For Claude Code custom endpoints, enable gateway hints with `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1`. Its [gateway protocol](https://code.claude.com/docs/en/llm-gateway-protocol#gateway-hint-headers) defines the header semantics. Preserving observations in `AgentContext` does not implement transparent Anthropic request-body or header forwarding to upstream providers.

## Next

- [Maintain session affinity](session-affinity.md) keeps a session's turns on one worker.
- [Trace an Agent Run](agent-tracing.md) joins LLM and tool spans on `session_id`.
- [Schedule Agents with ThunderAgent](thunderagent-program-scheduler.md) schedules whole agent programs keyed by `session_id`.
