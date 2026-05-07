<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KEDA - Optional Production Addon

KEDA adds event-driven autoscaling for Kubernetes workloads. Dynamo already has native scaling adapter integration, so KEDA is optional and should be installed only for trigger patterns that need KEDA scalers.

## Deployment

| Field | Value |
|---|---|
| Argo CD app | [`gitops/optional/keda.yaml`](../../gitops/optional/keda.yaml) |
| Chart | `keda` |
| Source | `https://kedacore.github.io/charts` |
| Version | `2.19.0` |
| Namespace | `keda` |
| Values | [`values.yaml`](values.yaml) |
| Sync wave | `8` |
| Baseline | No |

## What This Addon Owns

| Capability | Local setting |
|---|---|
| KEDA CRDs | `crds.install: true`. |
| Metric server | `prometheus.metricServer.enabled: true`. |
| Prometheus Operator integration | `prometheus.operator.enabled: true`. |
| ServiceMonitor | Enabled with `release: kube-prometheus-stack`. |

## What This Addon Does Not Own

- It does not replace Dynamo Planner or `DynamoGraphDeploymentScalingAdapter`.
- It does not create `ScaledObject` resources by default.
- It is not part of the root production app.

## Dynamo Contract

Use KEDA when a Dynamo deployment needs external triggers, scale-to-zero, or a scaler not covered by Planner/HPA. The example [`examples/keda-dgdsa-scaledobject.yaml`](../../examples/keda-dgdsa-scaledobject.yaml) shows the intended integration path.

## Verify

```bash
kubectl -n keda get pods
kubectl get crd scaledobjects.keda.sh scaledjobs.keda.sh
kubectl get servicemonitor -A | grep keda
deploy/pre-deployment/pre-deployment-check.sh --require keda
```

## Upgrade Notes

Review scaler and CRD changes before major KEDA bumps. Trigger authentication and fallback behavior are common sources of production regressions.

Upstream reference: [KEDA documentation](https://keda.sh/docs/).
