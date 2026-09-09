<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Subagent Group Affinity

The `subagent-group-affinity` policy co-locates the subagents of one parent session so they reuse the prompt prefix they share with their parent.

Dynamo binds session affinity to each request's own session id. A subagent carries its own session id, so siblings receive independent bindings and scatter across the worker pool even though they all replay the parent's system prompt, tool definitions, and context. This policy keeps a second, policy-local binding keyed by the *parent* session id, and prefers it for any request that carries one.

The group is placed on the least-loaded eligible worker the first time it is seen. It is not pinned to the parent's own worker: a parent typically holds its worker for the whole session, so sending its subagent fan-out to the same worker adds a burst of siblings to a worker that is already busy.

The policy implements only `WorkerPicker`. It requests `WorkerInputs::LOAD`; Dynamo continues to own worker discovery, eligibility, reservations, accounting, dispatch, and its own session binding state.

> [!IMPORTANT]
> Run this policy with `--router-session-affinity-mode soft`. Dynamo's default `hard` mode passes a bound session as a *pinned* target, which narrows the candidate set to one worker before any policy runs. A subagent whose own session is already bound would then never be free to join its parent's group.

## Policy Behavior

| Request | Decision |
|---|---|
| No parent session id | Dynamo's advisory session target if eligible and at or below `max_active_requests`, otherwise the least-loaded worker |
| Parent session id, group not yet bound | The least-loaded worker, ignoring the subagent's own session target, then bind the group to it |
| Parent session id, group bound and at or below `max_active_requests` | Retain the group's worker |
| Parent session id, group bound and above `max_active_requests` | The least-loaded non-group worker if it is *strictly* less loaded, then rebind the group to it; otherwise retain |
| Group's worker absent from the eligible set | The least-loaded worker, then rebind the group to it |
| One eligible candidate | Select it without rebinding the group |

A subagent is steered only by its group binding. Its own session target is ignored even when the group is unbound, because that per-subagent binding is exactly what scatters siblings: honoring it on the first request would place the whole new group on whichever worker that one subagent happened to land on. Every worker the policy freely selects becomes the group's new binding, so a group that moves stays moved.

The move requires a *strictly* less loaded alternative. Moving to an equally loaded worker would relocate the group on every sibling once the threshold is crossed, so the group would oscillate across the pool and lose the shared prefix it exists to reuse.

A single-candidate set means the host narrowed the choice — most often a pinned session target — so the policy selects that worker but leaves the group binding alone rather than dragging every sibling onto a worker it never chose. This is a heuristic: the picker cannot see *why* the set was narrowed, so a pool that is genuinely down to one eligible worker is treated the same way. Such a request still marks its group as in use, so an actively used group does not expire while it is being served.

Load ties use `WorkerWithDpRank`, so selection does not depend on Dynamo's unspecified candidate-row order. The binding records a worker and its data-parallel rank, so a group shares one rank rather than only one worker.

Group bindings idle for longer than `group_idle_ttl_secs` are dropped. The sweep runs at most once per TTL, on a request that carries a parent session id. The map mirrors Dynamo's own session-affinity bounds of 65,536 groups and 256-byte ids, so a client cannot grow it without limit by varying the parent header.

## Recognized Headers

Dynamo resolves the parent session id at the HTTP boundary, so the policy works with any agent framework Dynamo already recognizes:

| Framework | Session header | Parent header |
|---|---|---|
| Dynamo | `X-Dynamo-Session-ID` | `X-Dynamo-Parent-Session-ID` |
| Claude Code | `x-claude-code-agent-id` | `x-claude-code-parent-agent-id` |
| Codex | `thread-id` | `x-codex-parent-thread-id` |
| OpenCode | `x-session-id` | `x-parent-session-id` |

A request whose parent id equals its own session id is treated as a main-agent request.

## Attention-DP Backends

The binding records a worker and its data-parallel rank, so grouping is rank-exact as far as Dynamo's router is concerned. A backend that routes again internally can still split a group across ranks.

TensorRT-LLM is the case to watch. When its `attention_dp_config.kv_cache_routing_conversation_affinity` is enabled, the engine's own `ConversationAwareADPRouter` pins a conversation to an attention-DP rank keyed on `agent_context.session_id` — each subagent's own id, not the parent's. With the default `--conversation-affinity-dp-rank-source engine`, the engine load-balances every new conversation independently, so siblings share a worker but scatter across the ranks inside it, which is where the shared prefix would have been reused. Set `--conversation-affinity-dp-rank-source dynamo` so the rank this policy selects is the one the engine records.

## Configuration

[`worker-selection.yaml`](worker-selection.yaml) configures the Mocker demonstration:

```yaml
worker_selection:
  aggregated: subagent-group-affinity
  instances:
    - name: subagent-group-affinity
      type: subagent-group-affinity
      parameters:
        max_active_requests: 4
        group_idle_ttl_secs: 300
```

`max_active_requests` is inclusive and controls how much load a group tolerates before it moves. Raise it to keep siblings together more aggressively; lower it to favor load balance. Setting it to `0` moves the group as soon as its worker has any in-flight request and a strictly less loaded worker exists, which largely disables grouping under concurrency.

Build the Python extension against the example catalog before starting the frontend. Follow [Run With the Python Frontend](../README.md#run-with-the-python-frontend) for the catalog-link command.

## Run With Two Mockers

Run each command from the Dynamo repository root in its own terminal. These source-development commands use Dynamo's private Mocker launcher because Mocker is not currently exposed as a public CLI.

### 1. Start the Frontend

```bash
DYN_ROUTER_WORKER_SELECTION_POLICY=subagent-group-affinity \
python -m dynamo.frontend \
  --router-mode kv \
  --router-policy-config examples/router/custom-policy-example/subagent-group-affinity/worker-selection.yaml \
  --router-session-affinity-ttl-secs 60 \
  --router-session-affinity-mode soft \
  --discovery-backend file \
  --http-port 8000
```

### 2. Start Two Mocker Workers

```bash
python3 -m dynamo.mocker \
  --model-path Qwen/Qwen3-0.6B \
  --discovery-backend file \
  --decode-speedup-ratio 0.1 \
  --num-workers 2
```

The decode slowdown leaves enough time to send later requests while the first request remains active.

### 3. Open the First Subagent and Hold It

Start a long streaming request for subagent `child-1` of parent `parent-1`. It must still be streaming when you send both later requests: once it finishes, every worker is idle again and the third request has no load to steer it away from the group's worker.

```bash
curl --fail --silent --show-error --no-buffer \
  http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Dynamo-Session-ID: child-1' \
  -H 'X-Dynamo-Parent-Session-ID: parent-1' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "hold the first subagent"}],
    "max_tokens": 4096,
    "stream": true,
    "nvext": {"extra_fields": ["worker_id"]}
  }'
```

Wait for the first `data:` line, then read `nvext.worker_id.decode_worker_id` as worker A. Both workers are idle, so the group is placed on the lower worker id. Leave the request running while you send the next two.

### 4. Send a Sibling Subagent

In another terminal, send a different subagent of the same parent:

```bash
curl --fail --silent --show-error \
  http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Dynamo-Session-ID: child-2' \
  -H 'X-Dynamo-Parent-Session-ID: parent-1' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "join the sibling group"}],
    "max_tokens": 4,
    "stream": false,
    "nvext": {"extra_fields": ["worker_id"]}
  }' | jq --exit-status --raw-output '.nvext.worker_id.decode_worker_id'
```

`child-2` has its own session id and no binding of its own, and worker B is idle while A is busy. It still selects A, because `parent-1` is bound to A and A's one active request is at or below the configured threshold.

### 5. Send a Subagent of a Different Parent

```bash
curl --fail --silent --show-error \
  http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Dynamo-Session-ID: child-3' \
  -H 'X-Dynamo-Parent-Session-ID: parent-2' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "start a second group"}],
    "max_tokens": 4,
    "stream": false,
    "nvext": {"extra_fields": ["worker_id"]}
  }' | jq --exit-status --raw-output '.nvext.worker_id.decode_worker_id'
```

`parent-2` has no binding yet, so it is placed on the least-loaded worker. A still holds the streaming request, so the new group starts on B.

The worker sequence must be:

```text
A -> A -> B
```

The second request proves that a sibling joins its parent's group instead of following load. The third request proves that each parent forms an independent group and that a new group is placed by load.

## Test the Policy

```bash
cargo test -p dynamo-custom-policy-example-subagent-group-affinity
```
