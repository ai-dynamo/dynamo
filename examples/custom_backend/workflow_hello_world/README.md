<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Hello World Workflow Comparison

This comparison accepts an OpenAI chat-completions request, ignores its content,
and returns `Hello, World!`. All implementations reuse the same stage behavior:

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

## Dynamo with Bespoke Worker Orchestration

This implementation keeps Dynamo's existing frontend, model discovery, and
routing, but registers one aggregated worker at `dyn://my.endpoint.generate`.
The worker owns the application control flow: it runs Hello and World
concurrently, then runs Merge locally. It does not construct a `WorkflowIR`.

```text
OpenAI client -> Dynamo frontend -> aggregated worker --+--> Hello --+
                                                       +--> World --+--> Merge
```

Run the frontend and aggregated worker:

```bash
examples/custom_backend/workflow_hello_world/dynamo_bespoke/launch.sh
```

## Send a Request

All launchers listen on port 8000 by default. Send the same request to any
implementation:

```bash
python3 -m examples.custom_backend.workflow_hello_world.common.client
```

Override `--base-url` when the selected launcher uses another port.

## Responsibility Comparison

| Concern | Bespoke HTTP | Dynamo workflow | Dynamo bespoke |
| --- | --- | --- | --- |
| OpenAI request and response handling | Gateway code | Existing frontend | Existing frontend |
| Worker location | Configured URLs | Three discovery endpoint IDs | One discovery endpoint ID |
| Fan-out and join | Gateway tasks | Graph scheduler | Aggregated worker code |
| Merge placement | Inline gateway code | Remote binding | Inline worker code |
| Cancellation and stage failure | Gateway code | Workflow attempt | Aggregated worker code |
| Stage input and output checks | HTTP adapter code | Stage contracts | User code |
