<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Recommend-to-DGD CLI tests

`test_dgd_cli.py` runs every YAML in `configs/` through the installed AISimulate
CLI with `recommend --stack dynamo --output dgd`. The real Dynamo replay stack,
search, AIC renderer, and DGD output adapter run together. The local tiny-model
fixture contains metadata only; fixed timings keep these CPU tests independent
of model weights, GPUs, and external performance databases.

The table covers aggregated and disaggregated manifests, router configuration
with Kustomize output, and a Pareto front with different scheduler limits. It
checks emitted manifests and `index.json` against the selected Candidates in
`recommendation.json`: model, role-specific replicas, scheduler/cache limits,
runtime image, namespace, runtime override, and total GPU use. The Pareto checks
reject rendering the same Candidate repeatedly under different names.

The fixed aggregated and disaggregated cases also compare the complete DGD to
`goldens/`. Run the table from the repository root:

```bash
python -m pytest components/src/dynamo/aisimulate/tests/dgd/e2e/test_dgd_cli.py -v
```

When a golden is missing or differs semantically, the test writes the actual CLI
output beside it as `<stem>.new.yaml` and fails. The reviewed golden is never
rewritten. Golden comparison runs before detailed Candidate assertions so a
changed model or runtime image also produces a reviewable proposal. The failure
prints exact per-file diff and acceptance commands. For example:

```bash
diff -u \
  components/src/dynamo/aisimulate/tests/dgd/e2e/goldens/aggregated.yaml \
  components/src/dynamo/aisimulate/tests/dgd/e2e/goldens/aggregated.new.yaml
```

Inspect each proposal separately. If the change is intended, replace only its
corresponding golden with the reviewed `.new.yaml` file and rerun the case. If the
change is unintended, fix the regression and discard that proposal. Proposal
files remain visible in the working tree; there is no batch-update mode. Passing
tests never rewrite a golden.

The [comparison corpus](../comparison/README.md) is a separate, manually invoked
recipe/performance-database comparison. Replay CLI prediction and recommendation
round trips remain in the Replay test suite.
