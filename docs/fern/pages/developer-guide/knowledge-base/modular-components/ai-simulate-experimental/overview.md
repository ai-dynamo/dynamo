---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: AI Simulate (Experimental)
subtitle: Backend-neutral simulation and configuration-search tools
---

> [!WARNING]
> **Experimental.** AI Simulate is intended for evaluation and feedback, not production capacity
> planning. Its Python APIs, configuration schemas, search results, and deployment output may
> change without a standard deprecation period.

AI Simulate is a standalone Python distribution. It provides inference-engine forward-pass
simulation, deployment simulation, and search without depending on `ai-dynamo`.

For an engine-only single replay run, use `python -m aisimulate.replay`. For a replay with Dynamo
Router, Planner, or online adapters, use `python -m dynamo.replay`. Both commands share their base
replay configuration; Dynamo extends it with adapter options. The selected runtime validates each
`--*-engine-args` JSON payload, so runtime-specific fields can differ. For configuration search, call
`Sweeper(runner_factory=...).run(config)` or start from an example under
[`aisimulate/examples/sweeper`](https://github.com/ai-dynamo/dynamo/tree/main/aisimulate/examples/sweeper).

## Engine

The [AI Simulate Engine](engine-experimental/architecture.md) owns scheduler models for vLLM,
SGLang, and TensorRT-LLM, native GPU KV-cache accounting, prefix reuse, preemption, timing, and
attention data-parallel barriers. It emits neutral pass observations and leaves the clock, workload,
and runtime effects to its caller.

Offline Replayer and Dynamo Live Mocker construct this same generalized engine. Replayer supplies a
virtual clock; Live Mocker supplies Tokio and Dynamo transport.

## Replayer

The [AI Simulate Replayer](replayer-experimental/architecture.md) owns deterministic virtual time,
logical workers, aggregated and disaggregated replay, placement and scaling contracts, and canonical
reports. Built-in round-robin composition runs without Dynamo.

Use `python -m aisimulate.replay` for one engine-only run. The
[AI Simulate Replay CLI Reference](replayer-experimental/cli-reference.md) documents its shared
workload, engine, topology, service-level agreement (SLA), and output arguments.

Dynamo extends the same Replayer with KV router placement and Planner scaling. Those adapters remain
outside AI Simulate.

## Sweeper

[Sweeper](sweeper-experimental/overview.md) searches backend deployment settings against an injected replay runner.
Its core owns backend search, candidate orchestration, scoring, and the versioned `ReplaySpec`
contract.

Optional adapters extend the search without adding a Dynamo dependency to AI Simulate. The
`ai-dynamo` wheel registers the `dynamo.planner` and `dynamo.router` adapters. Selecting either
adapter imports its Dynamo implementation and adds a versioned runtime hook to the replay
specification.

KVBM search settings are deprecated and are not supported by the AI Simulate engine and replay
path. They have no adapter migration.

## Install

The `dynamo-planner` image builds and installs AI Simulate and Dynamo from the same source revision.
The AI Simulate wheel remains inside the image and is not published as a standalone release
artifact.

For source development, install AI Simulate, Dynamo, and the Planner dependencies from the
repository root:

```bash
uv pip install -e ./aisimulate
uv pip install --no-deps -e .
uv pip install -r container/deps/requirements.planner.txt
```
