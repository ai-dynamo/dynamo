<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Loki - Production Addon

Loki stores and serves logs collected by Fluentd. This profile uses a single-binary filesystem-backed Loki deployment for the current single-cluster production lane.

## Deployment

| Field | Value |
|---|---|
| Argo CD app | [`gitops/apps/20-loki.yaml`](../../gitops/apps/20-loki.yaml) |
| Chart | `loki` |
| Source | `https://grafana.github.io/helm-charts` |
| Version | `7.0.0` |
| Namespace | `logging` |
| Values | [`values.yaml`](values.yaml) |
| Sync wave | `2` |
| Baseline | Yes |

## What This Addon Owns

| Capability | Local setting |
|---|---|
| Deployment mode | `deploymentMode: SingleBinary`. |
| Storage | Filesystem storage with TSDB schema v13. |
| Replication | `replication_factor: 1`. |
| Gateway | Enabled; Fluentd pushes to `loki-gateway`. |
| Persistence | Single binary PVC, `100Gi`. |
| Read/write/backend replicas | Set to `0` because this is not simple-scalable mode. |
| Auth | `auth_enabled: false` for in-cluster use. |

## What This Addon Does Not Own

- It does not collect logs; Fluentd does.
- It does not provide HA or object-store durability in this profile.
- It does not expose logs outside the cluster by default.

## Production Caveat

This is a pragmatic single-binary deployment. Before multi-cluster or long-retention production use, move Loki to object storage and a scalable or distributed deployment mode.

## Verify

```bash
kubectl -n logging get pods,svc -l app.kubernetes.io/name=loki
kubectl -n logging port-forward svc/loki-gateway 3100:80
curl -sS http://127.0.0.1:3100/ready
deploy/pre-deployment/pre-deployment-check.sh --require loki
```

## Upgrade Notes

Coordinate Loki and Fluentd changes. If the gateway name, API path, or storage schema changes, update [`../fluentd/values.yaml`](../fluentd/values.yaml) and validate log ingestion before promoting.

Upstream reference: [Grafana Loki Helm installation documentation](https://grafana.com/docs/loki/latest/setup/install/helm/).
