---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Sweeper Quickstart
subtitle: Run a backend-neutral sweep with an injected replay runtime
---

<!--
Generated from `aisimulate/docs/sweeper/quickstart.md` by `docs/fern/scripts/sync_aisimulate_docs.py`.
Edit the canonical source instead of this Fern copy.
-->

> [!WARNING]
> **Experimental.** Sweeper is intended for evaluation and feedback, not production capacity
> planning.

From a source checkout, install AI Simulate and run the engine-only CLI:

```bash
python -m pip install -e ./aisimulate
aisimulate recommend --stack engine \
  --config aisimulate/examples/sweeper/configs/smart_sweep.yaml \
  --output-dir results
```

The command uses the engine-only replay stack by default. Select `--stack dynamo` to load the
optional Dynamo replay composition and configured Dynamo providers. The command prints top
recommendations and writes lossless JSON plus `best_config_topn.csv`; Pareto searches write
`pareto.csv` instead.

Applications can supply a `RunnerFactory` directly:

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
