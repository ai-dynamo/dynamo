<!--
SPDX-FileCopyrightText: Copyright (c) 2026 BlaiseAI / ai-blaise. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CRIU Snapshots Runbook

Operational playbook for the `criu-snapshots` addon. The full hard-constraint
list lives upstream at
[ai-blaise/criu-snapshots/docs/hard-constraints.md](https://github.com/ai-blaise/criu-snapshots/blob/main/docs/hard-constraints.md);
this runbook covers the local production use.

## Pre-flight

Before flipping `snapshots.ai-blaise.io/enabled: "true"` on any DGD:

1. The `criu-snapshots` addon is healthy. Both the controller Deployment
   and the per-node DaemonSet are ready.

   ```bash
   kubectl -n criu-snapshots get pods
   kubectl get crd dynamographdeploymentsnapshots.snapshots.ai-blaise.io
   ```

2. The GHCR push secret exists in the `criu-snapshots` namespace. External
   Secrets Operator at `addons/external-secrets/` reconciles it from the
   vault.

   ```bash
   kubectl -n criu-snapshots get secret ghcr-push ghcr-pull
   ```

3. The target DGD is healthy and serving requests through the SMG gateway.
   Snapshotting an unhealthy DGD is undefined behaviour.

4. The runtime image used by the target DGD's worker Pods has the
   SGLang snapshot hooks compiled in (env `SGLANG_SNAPSHOT_HOOKS=1` and
   the `snapshot_hooks.py` module on the import path). See
   `examples/deepseek-v32-reap-sglang-snapshotted.yaml` for the exact
   shape.

## Hard constraints — load-bearing

| Constraint                                            | What breaks if you violate it                                              |
| ----------------------------------------------------- | -------------------------------------------------------------------------- |
| NVIDIA driver R565+ on every snapshot-eligible node   | `cuda-checkpoint` is unavailable; CRIU plugin disables itself              |
| GPU SKU + count + slot order match on restore         | Restore SIGFAULTs on first kernel launch                                   |
| Host CPU instruction set matches                      | Restore SIGILLs on first uncommon-instruction code path                    |
| NVIDIA driver version matches                         | Restored process talks to a different driver ABI; opaque crash             |
| Runtime container image digest matches                | TLS / TCB / linker state differs; restore deadlock or segfault             |
| `pre_snapshot` hook destroys NCCL process groups      | Restored process deadlocks on first collective                             |
| `post_restore` hook re-initializes NCCL               | Same                                                                       |
| MIG disabled on snapshot-eligible nodes               | cuda-checkpoint refuses to operate on MIG instances                        |
| MPS disabled on snapshot-eligible nodes               | cuda-checkpoint refuses to operate on MPS                                  |
| UVM (managed memory) not in use                       | cuda-checkpoint does not handle UVM                                        |
| Single-node multi-GPU only                            | NCCL inter-node state cannot be snapshotted                                |

## Take a snapshot

From the operator host (or any host with kubectl + the
`deploy-a4-snapshots.sh` wrapper):

```bash
# One specific rank
OPERATION=take \
SNAPSHOT_COMPONENT=prefill \
SNAPSHOT_RANKS=0:0 \
scripts/dynamo-reap/deploy-a4-snapshots.sh

# All ranks of a component (parallel fanout, one per GPU)
OPERATION=take \
SNAPSHOT_COMPONENT=prefill \
scripts/dynamo-reap/deploy-a4-snapshots.sh

OPERATION=take \
SNAPSHOT_COMPONENT=decode \
scripts/dynamo-reap/deploy-a4-snapshots.sh
```

Watch progress:

```bash
kubectl -n criu-snapshots get dgds -w
kubectl -n criu-snapshots describe dgds <name>
kubectl -n criu-snapshots logs -l app.kubernetes.io/component=controller --tail=100
kubectl -n criu-snapshots logs -l app.kubernetes.io/component=daemon --field-selector=spec.nodeName=<node> --tail=100
```

Expected phase progression: `Pending` → `Snapshotting` → `Packaging` →
`Pushing` → `Validating` → `Ready`. Total wall time on the
`deepseek-v32-reap-sglang` deployment runs ~30-90 s per rank depending
on which scratch buffers and cached graphs are resident.

## Promote a snapshot

The take operation produces sequenced tags
(`deepseek-v32-reap-prefill-tp0-dp0-fp<X>-seq<Y>`). Promote the latest
to the `-current` tag that the SGLang Pods reference in their
`snapshot-pull` initContainer:

```bash
OPERATION=promote \
SNAPSHOT_COMPONENT=prefill \
PROMOTE_FROM_SEQ=<Y> \
PROMOTE_TO_TAG=current \
scripts/dynamo-reap/deploy-a4-snapshots.sh
```

The Argo CD application sees the next reconcile but no Pods restart
until an explicit rollout: the snapshot tag is read at Pod start, so
in-flight Pods continue with whatever snapshot they pulled at boot.
For scale-up, the new replica picks the promoted tag.

## Restore-verify before going live

Always probe a fresh snapshot before promoting:

```bash
OPERATION=restore-verify \
SNAPSHOT_COMPONENT=prefill \
scripts/dynamo-reap/deploy-a4-snapshots.sh
```

This spins up a one-off Pod with the snapshot-pull initContainer and a
probe inference request. The Pod is deleted on success; retained for
inspection on failure.

## Rotate old snapshots

GHCR storage is cheap but not free; rotate after a deploy stabilises:

```bash
OPERATION=rotate \
SNAPSHOT_COMPONENT=prefill \
ROTATE_KEEP=5 \
scripts/dynamo-reap/deploy-a4-snapshots.sh
```

Keeps the 5 most-recent sequenced tags plus `-current` and deletes the
rest.

## Failure modes and what to do

| Symptom                                          | Likely cause                                                            | Fix                                                                          |
| ------------------------------------------------ | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| DGDS stuck in `Snapshotting`                     | Source replica did not drain in `preStopGracePeriodSeconds`             | Increase `drain.preStopGracePeriodSeconds`; or drain the replica's request queue manually |
| DGDS transitions to `Failed` with `pre_snapshot timed out` | Application's `pre_snapshot` handler crashed or took > 5 s              | Inspect the worker logs at the time of the snapshot; the application hook must be idempotent and complete in < 5 s |
| `cuda-checkpoint lock failed` in agent logs     | In-flight CUDA stream callbacks didn't quiesce in 10 s                  | Retry; persistent failures mean the workload has a long-running callback that needs revisiting |
| Snapshot-fast-start Pod CrashLoops               | Fingerprint drift (driver bump, runtime image bump, kernel bump)         | Compare `manifest.json` fingerprint to current node fingerprint; re-take if drift is intentional |
| Snapshot-fast-start Pod CrashLoops with SIGILL  | Host CPU instruction set differs from the snapshot host                  | Restrict snapshot consumers via nodeSelector to the original CPU class       |
| `NCCL deadlock` after restore                   | Application's `post_restore` hook didn't re-init NCCL                    | Inspect `snapshot_hooks.py` install; ensure `SGLANG_SNAPSHOT_HOOKS=1` is set |
| Restore validation probe fails                  | KV router didn't re-attach, or NIXL P/D connection didn't re-establish  | Check `post_restore.done` sentinel; check `kv_router.attach_current_replica()` logs |
| GHCR push 401                                    | `ghcr-push` secret missing or stale                                     | `kubectl -n criu-snapshots delete secret ghcr-push` and let ESO re-reconcile |

## Decommission a deployment's snapshots

When retiring a DGD, also delete its snapshot history:

```bash
OPERATION=rotate \
SNAPSHOT_COMPONENT=prefill \
ROTATE_KEEP=0 \
scripts/dynamo-reap/deploy-a4-snapshots.sh
```

`ROTATE_KEEP=0` deletes every tag for the given (DGD, component) tuple
including `-current`. Re-runnable for `decode`.

## What this runbook does NOT cover

- Megatron-LM training snapshots (use `MegatronTrainingSnapshot`; see
  the upstream repo's `examples/`).
- Cross-cluster snapshot migration. Snapshots are valid only on hosts
  in the same fingerprint class. There is no notion of "moving" a
  snapshot to a different driver / kernel / GPU SKU.
- Restoring from a snapshot taken before the cuda-checkpoint or CRIU
  version was bumped. Always re-take after any upgrade in the addon's
  `image.*.tag` or `targetRevision`.
