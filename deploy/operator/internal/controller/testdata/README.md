<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Reviewing clusterenv goldens

> [!WARNING]
> Every change to an existing clusterenv golden needs a detailed semantic review.
> Replacing expected output with newly observed output can hide a production
> regression. A passing test after regeneration does not establish correctness.

These rules apply to all controller goldens here, including DCD, DGD, DGDR, and
upgrade scenarios. Review every changed field against the intended behavior and
identify the production change that caused it. Do not bulk-accept `.new` files.

Pay particular attention to worker-generation hashes, generated resource names,
selectors, and pod-template labels, annotations, environment, and specs. Changes
can replace or roll running workloads even when the DGD intent and runtime images
have not changed. Distinguish an internal bookkeeping digest from a value that
selects an active generation or triggers a rollout; do not assume a hash-only diff
is harmless.

Never update, recompute, normalize away, or wildcard rollout-relevant expectations
merely to accommodate changed controller output. Investigate the production
change first and preserve compatibility. An intentional behavior change needs an
explanation of affected workloads, upgrade consequences, and any migration or
opt-in, with focused validation. Explain each semantic golden change in the PR;
regeneration alone is not a justification.

Keep independent safety assertions intact. Historical inputs remain frozen and
advance only through the documented [upgrade history workflow](upgrade/README.md).
