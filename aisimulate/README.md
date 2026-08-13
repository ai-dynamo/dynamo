<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AI Simulate

> [!WARNING]
> **Experimental.** AI Simulate and Sweeper are intended for evaluation and feedback, not production
> capacity planning. Their APIs, configuration schemas, search behavior, and output may change
> without a standard deprecation period. They provide no SLA, accuracy, or configuration-optimality
> guarantees.

AI Simulate is a standalone Python distribution in the Dynamo repository. Its first package,
`aisimulate.sweeper`, searches backend deployment settings by evaluating serializable replay
specifications. The package does not depend on `ai-dynamo`.

Sweeper accepts a replay `RunnerFactory` through its Python API. Optional feature adapters own
their search spaces and runtime hooks. A backend-only sweep can use a Dynamo-free replay runner;
a sweep configured with Dynamo Planner or Router adapters uses Dynamo's runner composition.

Install AI Simulate by itself for backend-only development:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ./aisimulate
```

The `dynamo-planner` image builds and installs AI Simulate from the same source revision as Dynamo.
The AI Simulate wheel is an image-local artifact, not a standalone release artifact. For Dynamo
feature development from a source checkout, also install `ai-dynamo` and the Planner requirements:

```bash
uv pip install --no-deps -e .
uv pip install -r container/deps/requirements.planner.txt
```

Run a sweep from Python with an explicit runner:

```python
from aisimulate.sweeper import SmartSearchConfig, Sweeper
from dynamo.replay.simulation import DynamoReplayRunnerFactory

config = SmartSearchConfig.from_yaml("smart_sweep.yaml")
candidates = Sweeper(
    runner_factory=DynamoReplayRunnerFactory(),
).run(config)
```

The standalone module validates the backend-neutral core schema but intentionally has no implicit
replay runtime. Adapter-owned search spaces are validated when the selected adapters are resolved
by `Sweeper.run`.
KVBM sweep fields have been removed; native G2 is their replacement.

Read the canonical [Sweeper documentation](docs/sweeper/overview.md) for its configuration,
search-space, and replay behavior. Backend-neutral and Dynamo integration examples live under
[`aisimulate/examples/sweeper`](examples/sweeper/README.md).
