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

Parent retention uses the recorded pause until the next parent request as its oracle TTL and a modest priority. Its primary signals are parent-resume cache hit rate and TTFT. Root-final eviction is evaluated under concurrent, cache-constrained traffic; its expected benefit is releasing capacity for other live sessions rather than improving reuse for the completed session.

## Future considerations

- Evict only child-exclusive blocks when a subagent terminates, preserving prefixes shared with the parent or another live session.
- For an inferred `reset_context`, compare the old and replacement lineages and consider only the obsolete old suffix for eviction after the replacement request succeeds.
- Deliver eviction to every worker holding targeted blocks when a session spans multiple workers.
- Release parent retention explicitly at subagent join instead of relying only on TTL expiration.

## Preconditions

- Add `X-Dynamo-Subagent-Spawn` normalization to `AgentContext` so the policy can identify configuration B.
- Restrict final-session eviction to roots by checking that `parent_session_id` is absent.
- Use the experiment image and AIPerf branch documented in [`../BUILD.md`](../BUILD.md).

## Workstation protocol

The public corpus contains no complete trace below Qwen3-0.6B's 40,960-token context limit. [`common/prepare_dataset.py`](./common/prepare_dataset.py) therefore selects four complete lifecycle-rich traces and reproducibly scales them for the workstation. It keeps real subagent and session-end boundaries, groups every eight complete source hash blocks into one logical 64-token block, scales timing by 0.01, and rejects any transformed trace above 32,768 tokens. Incomplete hash groups become unhashed tails, which conservatively understates reuse across a branch inside one eight-block group.

The generated loadable trace directory, combined JSONL, and provenance manifest are written under the ignored run artifact directory. They are never committed. Every ablation in one matrix run consumes that same generated dataset.

Defaults:

| Parameter | Value |
| --- | --- |
| Model | `Qwen/Qwen3-0.6B` |
| Traces | First 4 with completed subagents, parent resume, and at most 120 requests |
| Token scale | 8 source blocks to 1 logical block |
| Time scale | 0.01, with think time capped at 2 seconds |
| Maximum transformed context | 32,768 tokens |
| Concurrency | 4 |
| Benchmark duration | 300 seconds |
| G1 capacity override | 1,024 blocks of 64 tokens |
| Retention | priority 10, per-request oracle TTL, at most 25% of G1 blocks |

Each configuration has an independent launcher under [`ablations/`](./ablations/). Run the full matrix with:

```bash
export CONTINUUM_IMAGE=<local-image-with-the-current-Dynamo-policy>
./experiments/continuum-kv-hints/weka-small-qwen3-0.6b/run_matrix.sh
```

Aggregate repeated matrix runs with:

```bash
python experiments/continuum-kv-hints/weka-small-qwen3-0.6b/common/summarize_replicates.py \
  experiments/continuum-kv-hints/weka-small-qwen3-0.6b/artifacts/<aggregate-run-id> \
  experiments/continuum-kv-hints/weka-small-qwen3-0.6b/artifacts/<matrix-run-id-1> \
  experiments/continuum-kv-hints/weka-small-qwen3-0.6b/artifacts/<matrix-run-id-2>
```

Generated files are ignored under `artifacts/<run-id>/<case>/`. Each case retains its exact command and revisions, subset manifest, AIPerf aggregate exports, per-request JSONL metrics, server metrics, frontend and worker logs, normalized KV-event evidence, and policy-action trace. Set `CAPTURE_RAW_KV_EVENTS=1` for a diagnostic raw vLLM ZMQ capture; leave it disabled for performance comparisons. See [`REPORT.md`](./REPORT.md) for the current workstation results and limitations.

The source-clock, ten-minute follow-up is isolated under [`time-faithful-20260921/`](./time-faithful-20260921/). It preserves the earlier compressed matrix and writes to a distinct dated artifact namespace.
