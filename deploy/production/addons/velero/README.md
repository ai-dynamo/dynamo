<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Velero - Production Addon

Velero is the baseline backup and restore addon for Dynamo production clusters. It backs up Dynamo custom resources, selected Kubernetes resources, and persistent volume data needed to recover the serving control plane.

## Deployment

| Field | Value |
|---|---|
| Argo CD app | [`gitops/apps/40-velero.yaml`](../../gitops/apps/40-velero.yaml) |
| Chart | `velero` |
| Source | `https://vmware-tanzu.github.io/helm-charts` |
| Version | `12.0.0` |
| Namespace | `velero` |
| Values | [`values.yaml`](values.yaml) |
| Sync wave | `4` |
| Baseline | Yes |

## What This Addon Owns

| Capability | Local setting |
|---|---|
| Backup storage location | AWS provider, bucket `dynamo-velero-backups`, region `us-east-1`. |
| Volume snapshot location | AWS provider, region `us-east-1`. |
| Node agent | `deployNodeAgent: true` for filesystem-level volume backup support. |
| Metrics | ServiceMonitor enabled with `release: kube-prometheus-stack`. |
| Dynamo daily backup | Schedule `dynamo-daily` at `0 7 * * *`. |
| Backup scope | `dynamo-system` namespace, Dynamo CRDs, ConfigMaps, Secrets, and PVCs. |

## What This Addon Does Not Own

- It does not create the object-storage bucket.
- It does not create cloud IAM roles or provider credentials.
- It does not replace database-native backups for external services.
- It does not back up namespaces that are not listed in the schedule template.

## Production Contract

Velero is the recovery layer for Dynamo Kubernetes state. Any production change that adds persistent state outside `dynamo-system` should either extend this schedule or define a separate backup policy in GitOps.

## Verify

```bash
kubectl -n velero get pods
kubectl -n velero get schedules.velero.io
kubectl -n velero get backupstoragelocations.velero.io
deploy/pre-deployment/pre-deployment-check.sh --require velero,velero-schedule
```

## Upgrade Notes

Validate backup and restore after chart upgrades. A successful controller rollout is not enough; run at least one manual backup and restore against a non-production namespace before promotion.

Upstream reference: [Velero documentation](https://velero.io/docs/).
