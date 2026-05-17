<!--
SPDX-FileCopyrightText: Copyright (c) 2026 BlaiseAI / ai-blaise. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# criu-snapshots — Production Addon

Drop-in installer for the [ai-blaise/criu-snapshots](https://github.com/ai-blaise/criu-snapshots)
controller, per-node snapshot-agent DaemonSet, and the two CRDs
(`DynamoGraphDeploymentSnapshot`, `MegatronTrainingSnapshot`). Gives Dynamo
+ SGLang workers and Megatron-LM training jobs CRIU + `cuda-checkpoint`
snapshot/restore so a fresh replica can resume from a cold-cached snapshot
in seconds instead of minutes.

The full constraint list and operational rules live in
[`../../runbooks/criu-snapshots.md`](../../runbooks/criu-snapshots.md). Read
that before flipping `snapshots.ai-blaise.io/enabled: "true"` on any DGD.

## What's wired on by default

| Capability                              | Where                                   | Knob                                                |
| --------------------------------------- | --------------------------------------- | --------------------------------------------------- |
| `DynamoGraphDeploymentSnapshot` CRD     | `chart/crds/` of upstream repo          | `installCRDs: true`                                 |
| `MegatronTrainingSnapshot` CRD          | `chart/crds/` of upstream repo          | `installCRDs: true`                                 |
| Controller Deployment                   | namespace `criu-snapshots`              | `controller.replicas`                               |
| Per-node DaemonSet (drops criu, plugins, cuda-checkpoint onto `/opt/criu-snapshots/`) | every GPU node | `daemon.nodeSelector`                               |
| OCI snapshot artifacts                  | `ghcr.io/ai-blaise/dynamo-snapshots`    | `ghcr.pushSecretName`, `ghcr.pullSecretName`        |
| Prometheus ServiceMonitor + PrometheusRule | namespace `monitoring`              | `prometheus.serviceMonitor.enabled`, `prometheus.prometheusRule.enabled` |
| NetworkPolicy                           | Controller and daemon                   | `networkPolicy.enabled`                             |

## What this addon does NOT do

- Build the snapshot artifact images. CI in
  `ai-blaise/criu-snapshots` publishes
  `ghcr.io/ai-blaise/criu-snapshots-controller` and
  `ghcr.io/ai-blaise/criu-snapshots-daemon` on every release tag. Bump
  the image tag in `values.yaml` to upgrade.
- Source the GHCR push token. External Secrets Operator at
  `addons/external-secrets/` is responsible for materialising
  `ghcr-pull` and `ghcr-push` secrets in the `criu-snapshots`
  namespace from the org's secret vault.
- Take snapshots automatically. A `DynamoGraphDeploymentSnapshot` is
  always created explicitly (manual `kubectl apply`, the
  `deploy-a4-snapshots.sh` wrapper, or a future scheduled CronJob).
- Provide a cold-start fallback. Per the runbook, failed restores
  surface as Pod failures; the Dynamo KV router routes around the
  degraded replica and the alert fires.

## Why sync wave 55

Order matters:

| Wave | App                       | Why                                                                  |
| ---- | ------------------------- | -------------------------------------------------------------------- |
| 00   | gpu-operator              | Driver must exist before the daemon can read driver version          |
| 05   | external-secrets          | Required for `ghcr-pull` / `ghcr-push`                               |
| 10   | prometheus                | ServiceMonitor + PrometheusRule consumers must exist                 |
| 50   | kai-scheduler             | Snapshot-fast-start Pods schedule through KAI                        |
| 51   | grove                     | Gang scheduling participates in restored Pod placement               |
| 55   | **criu-snapshots**        | Provides CRDs that the dynamo-platform Pods reference at admission   |
| 70   | smg                       | Routes around degraded replicas during a failed restore              |
| 90   | dynamo-platform           | The actual DGD operator that owns the workers we snapshot            |

## Pulling in upstream updates

`renovate.json5` at the repo root watches the chart's `targetRevision`
in `gitops/apps/55-criu-snapshots.yaml` and the controller + daemon
image tags in this `values.yaml`. Patch and minor versions auto-merge
after CI; majors land with a `needs-human-review` label.

To bump manually:

1. Edit `targetRevision` in `gitops/apps/55-criu-snapshots.yaml` to the
   new chart tag.
2. Edit `image.controller.tag` and `image.daemon.tag` in this
   `values.yaml` to the matching image tags. Both must change together;
   the chart's `appVersion` and the image tag are kept in lockstep by
   the upstream release CI.
3. Re-read `../../runbooks/criu-snapshots.md` for any new hard
   constraints introduced by the upgrade.

## Verify after deploy

```bash
kubectl -n criu-snapshots get pods
kubectl get crd dynamographdeploymentsnapshots.snapshots.ai-blaise.io
kubectl get crd megatrontrainingsnapshots.snapshots.ai-blaise.io
kubectl get dgds,mts -A
```

Take a no-op snapshot of an existing DGD to confirm the take path:

```bash
OPERATION=take \
SNAPSHOT_COMPONENT=prefill \
SNAPSHOT_RANKS=0:0 \
~/infrastructure/scripts/dynamo-reap/deploy-a4-snapshots.sh
```

And the restore-verify path:

```bash
OPERATION=restore-verify \
SNAPSHOT_COMPONENT=prefill \
~/infrastructure/scripts/dynamo-reap/deploy-a4-snapshots.sh
```
