<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo Platform - Production Addon

This values file configures the Dynamo platform chart for the production GitOps stack. It is the bridge between upstream Dynamo-owned Kubernetes resources and the fork-owned production add-ons under `deploy/production`.

## Deployment

| Field | Value |
|---|---|
| Argo CD app | [`gitops/apps/90-dynamo-platform.yaml`](../../gitops/apps/90-dynamo-platform.yaml) |
| Chart | `deploy/helm/charts/platform` from this repository |
| Version | Chart-local, currently `1.2.0` |
| Namespace | `dynamo-system` |
| Values | [`values.yaml`](values.yaml) |
| Sync wave | `9` |
| Baseline | Yes |

## What This Addon Owns

| Capability | Local setting |
|---|---|
| Dynamo discovery | `dynamo-operator.discoveryBackend: kubernetes`. |
| Admission posture | Webhook `failurePolicy: Fail`, `timeoutSeconds: 10`. |
| Metrics integration | Operator points at `http://kube-prometheus-stack-prometheus.monitoring.svc:9090`. |
| Scheduler integration | Grove and KAI are enabled for the platform chart but not installed by it. |
| Embedded dependencies | `global.etcd.install: false`; external etcd is not installed by this profile. |

## What This Addon Does Not Own

- It does not install GPU Operator, Prometheus, Loki, Fluentd, Falco, Trivy, Velero, External Secrets, Grove, KAI, or SMG. Those are separate Argo CD apps.
- It does not own the production DeepSeek REAP `DynamoGraphDeployment`; that lives under [`examples/`](../../examples/).
- It does not own secrets or external model storage.

## Drift Handling

The Argo CD app has narrow ignore rules for Kubernetes-defaulted fields in Dynamo CRDs and the bundled NATS StatefulSet. Those ignores are there to keep GitOps status meaningful without hiding application drift.

## Operator Image Pin

The values file pins `dynamo-operator.controllerManager.manager.image.tag` to `1.0.2`. Do not remove this pin only because the chart version is newer. First verify that the matching `nvcr.io/nvidia/ai-dynamo/kubernetes-operator:<chart appVersion>` image exists and that CRD conversion behavior is compatible with this fork.

## Verify

```bash
kubectl -n argocd get application dynamo-platform
kubectl -n dynamo-system get pods
kubectl get crd | grep dynamo
deploy/pre-deployment/pre-deployment-check.sh --require dynamo-crds,dynamo-webhooks,kai-queue
```

## Upgrade Notes

When upstream Dynamo changes operator APIs or chart defaults, update this values file only as needed to preserve the production contract. Keep third-party add-ons out of the platform chart so future upstream merges remain small and reviewable.

Upstream reference: [NVIDIA Dynamo Kubernetes documentation](https://docs.nvidia.com/dynamo/latest/kubernetes-deployment/).
