<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Hello World Workflow Comparison

This comparison accepts an OpenAI chat-completions request, ignores its content,
and returns `Hello, World!`. All three implementations reuse the same stage behavior:

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

The manual Dynamo implementation uses the existing frontend, endpoint discovery,
request transport, and model registration. A public orchestrator worker writes
the concurrent fan-out, cancellation, response validation, and inline merge as
ordinary Python control flow. The Hello and World workers remain private Dynamo
endpoints and only the orchestrator registers a public model deployment card.

```text
OpenAI client -> Dynamo frontend -> orchestrator --+--> private Hello --+
                                                   +--> private World --+--> inline merge
```

Run the frontend, orchestrator, and two private workers:

```bash
examples/custom_backend/workflow_hello_world/dynamo_manual/launch.sh
```

## Dynamo Orchestration

The Dynamo implementation declares the graph and binds all three stages to
discovery-backed remote endpoints. The existing frontend owns the OpenAI
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

All launchers listen on port 8000 by default. Send the same request to any
implementation:

```bash
python3 -m examples.custom_backend.workflow_hello_world.common.client
```

Override `--base-url` when the selected launcher uses another port.

## Responsibility Comparison

| Concern | Bespoke | Manual Dynamo | Dynamo workflow |
| --- | --- | --- | --- |
| OpenAI request and response handling | Gateway code | Existing frontend | Existing frontend |
| Worker location | Configured URLs | Discovery endpoint IDs in code | Discovery endpoint bindings |
| Fan-out and join | Gateway tasks | Orchestrator tasks | Graph scheduler |
| Merge placement | Inline code | Inline code | Remote binding |
| Cancellation and stage failure | Gateway code | Orchestrator code | Workflow attempt |
| Stage input and output checks | HTTP adapter code | Orchestrator code | Stage contracts |
| Graph representation | Python control flow | Python control flow | Validated workflow IR |
