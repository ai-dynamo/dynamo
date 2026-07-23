<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AI Simulate

> [!WARNING]
> **Experimental.** AI Simulate and Spica are intended for evaluation and feedback, not production
> capacity planning. Their APIs, configuration schemas, search behavior, and output may change
> without a standard deprecation period. They provide no SLA, accuracy, or configuration-optimality
> guarantees.

AI Simulate is a standalone Python distribution in the Dynamo repository. Its first package,
`spica`, searches engine and Dynamo deployment settings by evaluating candidates with Dynamo
Replay.

Spica uses replay APIs and Rust features from the same Dynamo revision. For local source use,
build and install the matching runtime before AI Simulate:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install pip "maturin[patchelf]"
cd lib/bindings/python
maturin develop --uv --release --features aic-forward-pass,mocker-kvbm-offload
cd ../../..
uv pip install --no-deps -e .
uv pip install -e ./aisimulate
```

Do not combine a source checkout of AI Simulate with an older released `ai-dynamo` runtime. The
`dynamo-planner` image builds and installs both wheels from the same commit.

Run Spica with a YAML configuration:

```bash
python -m spica --config aisimulate/examples/spica/configs/smart_sweep.yaml
```

Read the [Spica documentation](https://github.com/ai-dynamo/dynamo/tree/main/docs/components/aisimulate/spica)
for its configuration, search-space, and replay behavior. Runnable configurations and tools live
under [`aisimulate/examples/spica`](examples/spica/README.md).
