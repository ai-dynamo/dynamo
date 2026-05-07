<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OpenTelemetry Operator - Optional Production Addon

The OpenTelemetry Operator manages `OpenTelemetryCollector` resources and optional auto-instrumentation. It is optional because the baseline profile already exposes metrics and logs; install it when traces or managed collectors are required.

## Deployment

| Field | Value |
|---|---|
| Argo CD app | [`gitops/optional/opentelemetry.yaml`](../../gitops/optional/opentelemetry.yaml) |
| Chart | `opentelemetry-operator` |
| Source | `https://open-telemetry.github.io/opentelemetry-helm-charts` |
| Version | `0.110.0` |
| Namespace | `opentelemetry-operator` |
| Values | [`values.yaml`](values.yaml) |
| Sync wave | `8` |
| Baseline | No |

## What This Addon Owns

| Capability | Local setting |
|---|---|
| Operator lifecycle | Installs the OpenTelemetry Operator. |
| Collector image default | `otel/opentelemetry-collector-contrib`. |
| Webhook certificates | `certManager.enabled: false`, `autoGenerateCert.enabled: true`. |

## What This Addon Does Not Own

- It does not create a collector by default.
- It does not configure a trace backend.
- It does not replace Prometheus or Loki.

## Dynamo Contract

The example [`examples/opentelemetry-collector.yaml`](../../examples/opentelemetry-collector.yaml) creates the collector endpoint used by SMG values: `http://dynamo-collector.observability.svc.cluster.local:4317`. Keep that endpoint stable if SMG tracing remains enabled.

## Verify

```bash
kubectl -n opentelemetry-operator get pods
kubectl get crd opentelemetrycollectors.opentelemetry.io
deploy/pre-deployment/pre-deployment-check.sh --require opentelemetry
```

## Upgrade Notes

Review collector and auto-instrumentation CRD changes before major upgrades. Webhook certificate behavior is especially important because this profile does not require cert-manager.

Upstream reference: [OpenTelemetry Helm chart documentation](https://opentelemetry.io/docs/platforms/kubernetes/helm/).
