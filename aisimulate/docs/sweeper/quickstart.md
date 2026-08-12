---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Sweeper Quickstart
subtitle: Run a backend-neutral sweep with an injected replay runtime
---

> [!WARNING]
> **Experimental.** Sweeper is intended for evaluation and feedback, not production capacity
> planning.

From a source checkout, install AI Simulate and run its neutral example:

```bash
python -m pip install -e ./aisimulate
python aisimulate/examples/sweeper/run_sweep.py \
  --config aisimulate/examples/sweeper/sweep.yaml
```

The example runner returns deterministic metrics so you can inspect orchestration without importing
an application framework. A production composition supplies a `RunnerFactory` that executes real
replay:

```python
from aisimulate.sweeper import SmartSearchConfig, Sweeper

config = SmartSearchConfig.from_yaml("sweep.yaml")
sweeper = Sweeper(runner_factory=my_runner_factory, show_progress=False)
candidates = sweeper.run(config)
```

Set `sweep.parallel_evals` above one to use spawned worker processes. Scripts using that mode must
guard their entrypoint with `if __name__ == "__main__":`.

Next, read the [Tutorial](tutorial.md) for the complete configuration flow or [Sweep Configuration
Providers](sweep-config-provider.md) to add feature-specific search dimensions.
