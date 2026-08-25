<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Hello World Workflow

> Experimental: This workflow API is under active development and may evolve
> based on feedback. It is an optional orchestration layer over Dynamo's
> discoverable workers and endpoints; applications may compose those primitives
> directly when they prefer to own the orchestration logic.

This example serves a declarative workflow through a discovered orchestrator
worker. It accepts an OpenAI chat-completions request, ignores its content, and
returns `Hello, World!`:

- `HelloStage` produces `Hello, `.
- `WorldStage` produces `World!`.
- `MergeStage` concatenates both values.

The frontend remains generic. It discovers the orchestrator as an ordinary
model worker and can continue serving other discovered models. The orchestrator
owns dependency scheduling, fan-out, joins, cancellation, and stage-result
validation; all three stages run behind discovery-backed remote endpoints.

```text
OpenAI client -> generic Dynamo frontend -> orchestrator worker --+--> remote Hello --+
                                                                +--> remote World --+--> remote Merge
```

## Run

Start the frontend, orchestrator, and three stage workers:

```bash
examples/experimental/workflow/hello_world/launch.sh
```

The frontend listens on port 8000 by default. Send a request with:

```bash
python3 -m examples.experimental.workflow.hello_world.client
```

Set `DYN_HTTP_PORT` to change the frontend port and pass the matching
`--base-url` to the client.
