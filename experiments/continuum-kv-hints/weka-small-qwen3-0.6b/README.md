<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Small WEKA Qwen3-0.6B lifecycle policy experiment

This experiment runs a Qwen3-0.6B-compatible subset of [`semianalysisai/cc-traces-weka-062126-256k`](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k) through Dynamo vLLM with constrained G1 capacity. It evaluates sparse lifecycle-aware retention and eviction while leaving ordinary requests to vLLM's existing LRU policy.

The canonical policy rationale and future considerations are recorded in the [initial WEKA policy matrix](https://gitlab-master.nvidia.com/karenc/dynamo-workflows/-/blob/1d9fb2a2e04ffb866ec5b4965fcf0d60343f0e5c/dynamo/routing/agentic-kv-management/continuum-implementation-plan.md#L541-572).

## Initial matrix

| Configuration | Retain policy | Evict policy |
| --- | --- | --- |
| A. No hints | None | None |
| B. Parent-at-spawn retention | Retain the parent when its request spawns one or more subagents | None |
| C. Root-final eviction | None | Evict a root session after its final request completes successfully |
| D. Combined lifecycle policy | Retain the parent at subagent spawn | Evict a root session after its final request completes successfully |

```text
if session_final and parent_session_id is None:
    emit deferred kv.evict for the root session's selected-worker lineage
elif subagent_spawn:
    emit deferred bounded kv.retain for the parent session's selected-worker lineage
else:
    emit no hint and use ordinary LRU
```

Parent retention uses a bounded TTL and modest priority. Its primary signals are parent-resume cache hit rate and TTFT. Root-final eviction is evaluated under concurrent, cache-constrained traffic; its expected benefit is releasing capacity for other live sessions rather than improving reuse for the completed session.

## Future considerations

- Evict only child-exclusive blocks when a subagent terminates, preserving prefixes shared with the parent or another live session.
- For an inferred `reset_context`, compare the old and replacement lineages and consider only the obsolete old suffix for eviction after the replacement request succeeds.
- Deliver eviction to every worker holding targeted blocks when a session spans multiple workers.
- Release parent retention explicitly at subagent join instead of relying only on TTL expiration.

## Preconditions

- Add `X-Dynamo-Subagent-Spawn` normalization to `AgentContext` so the policy can identify configuration B.
- Restrict final-session eviction to roots by checking that `parent_session_id` is absent.
- Use the experiment image and AIPerf branch documented in [`../BUILD.md`](../BUILD.md).

Run commands and result locations will be added after the subset, concurrency, G1 capacity, retention TTL, and trial protocol are frozen.
