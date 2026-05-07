<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SMG Postgres - Production Addon

SMG Postgres is the local history and audit-log database for the SMG router. It is a small hand-written Kubernetes manifest, not a Helm chart, because the production profile only needs one Postgres pod, one PVC, and one service for SMG history.

## Deployment

| Field | Value |
|---|---|
| Argo CD app | [`gitops/apps/65-smg-postgres.yaml`](../../gitops/apps/65-smg-postgres.yaml) |
| Source | This repository, [`postgres.yaml`](postgres.yaml) |
| Namespace | `smg` |
| Image | `postgres:16.6-alpine` |
| Kustomization | [`kustomization.yaml`](kustomization.yaml) |
| Sync wave | `6` |
| Baseline | Yes, as part of the SMG path |

## What This Addon Owns

| Resource | Purpose |
|---|---|
| `Namespace/smg` | Restricted pod-security namespace for SMG resources. |
| `Secret/smg-history-postgres` | Password consumed by Postgres and the SMG router. |
| `PersistentVolumeClaim/smg-postgres-data` | 20 GiB history storage. |
| `Service/smg-postgres` | Stable cluster DNS for SMG. |
| `StatefulSet/smg-postgres` | Single Postgres pod pinned to the A4 node. |

## What This Addon Does Not Own

- It does not provide HA Postgres.
- It does not create managed database infrastructure.
- It does not back up the PVC by itself; Velero or an external database must handle retention.

## Production Caveat

The default password in `postgres.yaml` is a bootstrap/dev value. Production should overwrite `Secret/smg-history-postgres` via External Secrets or replace this local Postgres with a managed database and update `../smg/values.yaml`.

## Verify

```bash
kubectl -n smg get statefulset,pod,pvc,svc,secret smg-postgres
kubectl -n smg exec smg-postgres-0 -- pg_isready -U smg -d smg
kubectl -n smg exec smg-postgres-0 -- psql -U smg -d smg -c '\dt'
```

## Upgrade Notes

Postgres image upgrades need a backup and rollback plan. Check PostgreSQL major-version compatibility before changing the image tag.

Upstream reference: [PostgreSQL Docker Official Image documentation](https://www.docker.com/blog/how-to-use-the-postgres-docker-official-image/).
