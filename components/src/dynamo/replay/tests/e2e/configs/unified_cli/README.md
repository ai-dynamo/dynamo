<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Public CLI cases

`test_unified_cli_dynamo.py` runs every Predict and Recommend YAML through the
installed AISimulate CLI with the real Dynamo replay stack. The tiny local model
contains metadata only, and fixed timing makes these CPU tests independent of
model weights, GPUs, and external performance databases.

Recommend cases with a `dgd` section also request the installed DGD output adapter.
The tests check its manifest or Kustomize artifacts against the selected Candidates
in `recommendation.json`, including model, role-specific replicas, scheduler/cache
limits, image, namespace, runtime override, and total GPU use. The Pareto case keeps
different scheduler limits on its front so emitting the winner repeatedly fails.

The fixed aggregated and disaggregated cases additionally compare the complete DGD
to `goldens/dgd/`. Run those cases normally to check the reviewed output:

```bash
python -m pytest \
  components/src/dynamo/replay/tests/e2e/test_unified_cli_dynamo.py \
  -k '01-no-adapters-throughput or 07-disagg-dgd'
```

When a golden is missing or differs semantically, the test writes the actual CLI
output beside it as `<stem>.new.yaml` and fails. The reviewed golden is never
rewritten. The failure prints the exact per-file diff and acceptance commands.
For example, from the repository root:

```bash
diff -u \
  components/src/dynamo/replay/tests/e2e/configs/unified_cli/goldens/dgd/01-no-adapters-throughput.yaml \
  components/src/dynamo/replay/tests/e2e/configs/unified_cli/goldens/dgd/01-no-adapters-throughput.new.yaml
```

Inspect each proposal separately. If that change is intended, replace only its
corresponding golden with the reviewed `.new.yaml` file and rerun the case. If the
change is unintended, fix the regression and discard that proposal. Proposal files
remain visible in the working tree; there is no batch-update mode.

The broader comparison corpus under `dynamo/aisimulate/tests/dgd/comparison` remains
a separate, manually invoked recipe/performance-database comparison.
