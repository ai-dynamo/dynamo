<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Hello World Workflow Comparison

This comparison composes the same fixed stage behavior in three ways:

- `HelloStage` produces `Hello, `.
- `WorldStage` produces `World!`.
- `MergeStage` concatenates both values.

The implementations differ in who owns orchestration.

## Bespoke Orchestration

The bespoke gateway owns the OpenAI endpoint, fixed worker URLs, concurrent
fan-out, timeout and error handling, response validation, and response shaping.
It calls the Hello and World HTTP workers concurrently, then runs the merge stage
inline.

```text
OpenAI client -> aiohttp gateway --+--> Hello HTTP worker --+
                                   +--> World HTTP worker --+--> inline merge
```

Run the gateway and both workers:

```bash
examples/custom_backend/workflow_hello_world/bespoke/launch.sh
```

## Manual Dynamo Orchestration

The manual Dynamo implementation uses existing endpoint discovery, request
transport, and clients. Its orchestrator expresses fan-out, join, cancellation,
response validation, and the Merge dependency as ordinary Python control flow.
The example intentionally stops at the Dynamo endpoint boundary and does not add
another OpenAI frontend adapter or model registration path.

```text
Dynamo caller -> manual orchestrator --+--> remote Hello --+
                                       +--> remote World --+--> remote Merge
```

See
[`dynamo_manual/worker.py`](dynamo_manual/worker.py)
for the conceptual implementation.

## Dynamo Orchestration

The Dynamo implementation declares the same all-remote graph and binds its three
stages to discovery-backed endpoints. The existing frontend owns the OpenAI
protocol, while `WorkflowOrchestrator` owns dependency scheduling, fan-out,
join, cancellation, and result validation.

```text
OpenAI client -> Dynamo frontend --+--> remote Hello --+
                                   +--> remote World --+--> remote Merge
```

Run the frontend and three workers:

```bash
examples/custom_backend/workflow_hello_world/dynamo/launch.sh
```

## Send a Request

The bespoke and Dynamo workflow launchers listen on port 8000 by default. Send
the same request to either implementation:

```bash
python3 -m examples.custom_backend.workflow_hello_world.common.client
```

Override `--base-url` when the selected launcher uses another port.

## Responsibility Comparison

| Concern | Bespoke | Manual Dynamo | Dynamo workflow |
| --- | --- | --- | --- |
| OpenAI request and response handling | Gateway code | Outside comparison | Existing frontend |
| Worker location | Configured URLs | Discovery endpoint IDs in code | Discovery endpoint bindings |
| Fan-out and join | Gateway tasks | Orchestrator tasks | Graph scheduler |
| Merge placement | Inline code | Remote endpoint call | Remote binding |
| Cancellation and stage failure | Gateway code | Orchestrator code | Workflow attempt |
| Stage input and output checks | HTTP adapter code | Orchestrator code | Stage contracts |
| Graph representation | Python control flow | Python control flow | Validated workflow IR |
