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

AI Simulate is a standalone Python distribution in the Dynamo repository. Its engine-only replay
CLI and `aisimulate.sweeper` package evaluate serializable replay specifications without depending
on `ai-dynamo`.

Sweeper accepts a replay `RunnerFactory` through its Python API. Optional feature adapters own
their search spaces and runtime hooks. An engine-only sweep can use a Dynamo-free replay runner;
a sweep configured with Dynamo Planner or Router adapters uses Dynamo's runner composition.

For an engine-only single replay run, use `python -m aisimulate.replay`. For a replay with Dynamo
Router, Planner, or online adapters, use `python -m dynamo.replay`. Both commands share the
engine, topology, traffic, replay-mode, SLA, and output arguments; Dynamo adds its adapter options.
They share option names and the base `ReplaySpec` schema; the selected runtime validates each
`--*-engine-args` JSON payload, which can therefore contain runtime-specific fields.
For configuration search, call `Sweeper(runner_factory=...).run(config)` or start from an example
under `aisimulate/examples/sweeper`.

For example, run one engine-only synthetic replay with fixed timing:

```bash
python -m aisimulate.replay \
  --extra-engine-args '{"engine_type":"vllm","num_gpu_blocks":1024,"block_size":16,"timing_model":{"type":"fixed","prefill_ms":10,"decode_ms":2}}' \
  --input-tokens 1024 \
  --output-tokens 128 \
  --request-count 16 \
  --replay-concurrency 4
```

Install AI Simulate by itself for engine-only development:

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

The `ai-dynamo` wheel registers the `dynamo.planner` and `dynamo.router` Sweeper provider entry
points. Its Dynamo runner composes the materialized runtime hooks with the shared AI Simulate
Replayer.

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
KVBM sweep fields have been removed and have no adapter migration.

Read the canonical documentation for the [engine](docs/engine/architecture.md),
[Replayer](docs/replay/architecture.md), [replay CLI](docs/replay/cli-reference.md), and
[Sweeper](docs/sweeper/overview.md). Backend-neutral and Dynamo integration examples live under
[`aisimulate/examples/sweeper`](examples/sweeper/README.md).
