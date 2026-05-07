<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Fluentd - Production Addon

Fluentd collects container logs from every node and forwards them to Loki. It is the log shipper for this production profile; Loki owns storage and query.

## Deployment

| Field | Value |
|---|---|
| Argo CD app | [`gitops/apps/21-fluentd.yaml`](../../gitops/apps/21-fluentd.yaml) |
| Chart | `fluentd` |
| Source | `https://fluent.github.io/helm-charts` |
| Version | `0.5.3` |
| Namespace | `logging` |
| Values | [`values.yaml`](values.yaml) |
| Sync wave | `3` |
| Baseline | Yes |

## What This Addon Owns

| Capability | Local setting |
|---|---|
| Node log collection | Runs as a `DaemonSet`. |
| Container log source | Tails `/var/log/containers/*.log`. |
| Parsing | Accepts JSON logs first, then CRI-style regexp logs. |
| Kubernetes labels | Adds Kubernetes metadata to each record. |
| Loki output | Sends to `http://loki-gateway.logging.svc.cluster.local/loki/api/v1/push`. |
| Metrics | Exposes Fluentd Prometheus metrics on `:24231/metrics`. |

## What This Addon Does Not Own

- It does not store logs.
- It does not configure Loki retention or object storage.
- It does not guarantee application logs are structured; applications still need to emit useful fields.

## Label Contract

Logs are labeled with `cluster`, `namespace`, `pod`, `container`, and `app`. Keep label cardinality controlled. Do not add request IDs, user IDs, prompts, or model outputs as Loki labels.

## Verify

```bash
kubectl -n logging get daemonset,pods -l app.kubernetes.io/name=fluentd
kubectl -n logging logs -l app.kubernetes.io/name=fluentd --tail=50
kubectl -n logging port-forward daemonset/fluentd 24231:24231
curl -sS http://127.0.0.1:24231/metrics | head
deploy/pre-deployment/pre-deployment-check.sh --require fluentd
```

## Upgrade Notes

Keep Fluentd and Loki changes coordinated. If the Loki gateway service name or namespace changes, update `LOKI_URL` here before syncing Fluentd.

Upstream reference: [Fluentd Kubernetes deployment documentation](https://docs.fluentd.org/container-deployment/kubernetes).
