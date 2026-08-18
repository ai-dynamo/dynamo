<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Worker Hash Semantics

The Dynamo Operator stores versioned worker hashes on each `DynamoGraphDeployment` (DGD). These
hashes drive managed worker rollouts and form the suffix of generated worker
`DynamoComponentDeployment` (DCD) names and labels.

A worker hash does not identify a DGD, DCD, or running set of Pods. It is a comparison value for
worker-generation inputs. The DCDs and rollout status remain the source of truth for which workloads
exist and whether a rollout has completed.

## Stored State

The controller owns two DGD annotations:

| Version | Annotation | Meaning |
| --- | --- | --- |
| v1 | `nvidia.com/current-worker-hash` | The suffix of an active DCD generation created under the legacy hash contract |
| v2 | `nvidia.com/current-worker-hash-v2` | The hash used for worker change detection and, for v2 generations, the DCD suffix |

Worker DCDs carry the selected generation suffix in the `nvidia.com/dynamo-worker-hash` label. The
controller keeps the DGD annotations on the previously completed generation while a managed rollout
is in progress. It commits the new annotation state only after the rollout completes.

The hash versions are typed values with different semantics. Code must keep v1 and v2 in separate
fields and compare values only within the same version. A helper that treats either value as an
interchangeable string loses this distinction.

## Annotation States

| DGD annotations | Rollout comparison | Active DCD suffix |
| --- | --- | --- |
| v1 only | Legacy migration state; record v2 without rolling workers | v1 |
| v1 and v2 | v2 only | v1 until the next worker rollout completes |
| v2 only | v2 only | v2 |

Once v2 exists, v1 is constant. The controller does not use v1 equality to decide whether workers
need a rollout. It retains v1 only to locate the active DCDs whose names contain that suffix.

The dual-annotation state does not mean that v1 and v2 are equivalent hashes. It records two facts
about one completed generation: v2 describes the worker inputs for change detection, while v1 names
the already-created DCDs.

## Migration Semantics

The v1-only state is compatibility state for a DGD whose active DCDs predate v2 tracking. Migration
runs at the beginning of reconciliation, before steady-state rollout decisions:

1. Keep the stored v1 value unchanged.
2. Compute v2 from the current DGD spec.
3. Store v2 without rolling workers or changing the active DCD suffix.
4. Continue reconciliation using only v2 for change detection.

The migration deliberately avoids an operator-upgrade-induced rollout. If an operator upgrade and a
v2-relevant DGD edit happen together, recording v2 from the DGD can absorb that edit without a
rollout. Dynamo 1.4 accepted this narrow compatibility tradeoff. A normally reconciled DGD from 1.2
or later already has a v2 annotation, so this migration is relevant only to pre-v2 or incomplete
annotation state.

The semantic model does not require v1 to be recalculated. The 1.4 implementation still computes the
legacy v1 hash only while recognizing such v1-only state. It never computes v1 after a v2 annotation
exists. This compatibility check is not needed for normal upgrades from 1.2 or later and is isolated
migration machinery, not part of the steady-state rollout contract.

## Steady-State Rollout Semantics

After v2 exists, rollout decisions follow one functional sequence:

1. Compute v2 for the desired DGD worker state.
2. Compare it with `nvidia.com/current-worker-hash-v2`.
3. If the values match, do not start a worker rollout.
4. If the values differ, roll workers to DCDs with the desired v2 suffix.
5. After the rollout completes, store the desired v2 value and remove the v1 annotation if it still
   exists.

The first real worker rollout after migration therefore moves a dual-annotation DGD to v2-only state.
All later rollouts remain v2-only.

## Failure and Retry Model

Reconciliation is idempotent. The Kubernetes API server update that stores v2 is the migration's
commit boundary:

- If storing v2 fails, reconciliation returns an error and retries the migration.
- If child DCD or status updates succeed before a later operation fails, the next reconciliation
  observes those resources and continues.
- The controller can recompute the desired v2 value on every reconciliation.

No third annotation is needed to store an intended next hash. Such an annotation would duplicate
recomputable state and introduce a second state machine without improving recovery.

## Hash Function Contract

The v2 hash summarizes rendered worker DCD inputs that require a new worker generation. Scaling-only
inputs such as replica count do not participate because the controller applies them to the active
generation without a worker rollout.

Hash-function changes must be conservative. Changing the value for an existing DGD causes a rollout,
even when the rendered worker workload is unchanged. Any new hash input must therefore preserve
results for existing release states or use an explicit compatibility boundary. Never redefine v1;
stored v1 values are opaque generation suffixes.

## Release History

| Release | Hash behavior |
| --- | --- |
| 1.0 and 1.1 | [#6110](https://github.com/ai-dynamo/dynamo/pull/6110) introduced managed worker rollouts with v1 as the only hash and DCD suffix. |
| 1.2 | [#9210](https://github.com/ai-dynamo/dynamo/pull/9210) preserved v1 across the v1beta1 conversion. [#9235](https://github.com/ai-dynamo/dynamo/pull/9235), including the hash compatibility work in [#9385](https://github.com/ai-dynamo/dynamo/pull/9385), introduced v2. The controller computed and stored both versions, but continued to select v1 for changes visible to v1. A v2-only change could already trigger a v2-named rollout and remove v1. |
| 1.3 | The 1.2 dual-hash algorithm remained unchanged: both hashes were computed, ordinary rollouts remained v1-named, and v2 detected changes outside the v1 input. |
| 1.4 | [#11529](https://github.com/ai-dynamo/dynamo/pull/11529) made v2 authoritative in the steady state. New DGDs start v2-only. Existing dual-annotation DGDs keep their constant v1 suffix until the next real rollout, then become v2-only. v1 computation remains only in the incomplete v1-only compatibility path. |
| 1.5 | [#12633](https://github.com/ai-dynamo/dynamo/pull/12633) added the canonical resolved runtime version to v2 for runtimes at or above 1.5.0. Older and unresolved runtime versions keep their previous v2 value. The annotation lifecycle remains unchanged. |

The v2 annotation therefore first shipped in 1.2, participated in change detection in 1.2 and 1.3,
and became the sole steady-state rollout comparison in 1.4. The controller stopped computing v1 for
normal reconciliations in 1.4; the remaining legacy computation applies only to pre-v2 or incomplete
v1-only annotation state, not to normal 1.3-to-1.5 or 1.4-to-1.5 upgrades.
