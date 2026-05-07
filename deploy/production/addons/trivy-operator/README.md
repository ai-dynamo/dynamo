<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Trivy Operator - Production Addon

Trivy Operator scans Kubernetes workloads and cluster resources for vulnerabilities, misconfigurations, RBAC issues, infrastructure findings, and exposed secrets. It is the baseline cluster scanning addon.

## Deployment

| Field | Value |
|---|---|
| Argo CD app | [`gitops/apps/31-trivy-operator.yaml`](../../gitops/apps/31-trivy-operator.yaml) |
| Chart | `trivy-operator` |
| Source | `https://aquasecurity.github.io/helm-charts` |
| Version | `0.32.1` |
| Namespace | `trivy-system` |
| Values | [`values.yaml`](values.yaml) |
| Sync wave | `3` |
| Baseline | Yes |

## What This Addon Owns

| Capability | Local setting |
|---|---|
| Vulnerability scanner | Enabled. |
| Config audit scanner | Enabled. |
| RBAC assessment scanner | Enabled. |
| Infra assessment scanner | Enabled. |
| Exposed secret scanner | Enabled. |
| Concurrent scans | `scanJobsConcurrentLimit: 5`. |
| Metrics | ServiceMonitor enabled with `release: kube-prometheus-stack`. |

## What This Addon Does Not Own

- It does not block deployments by default.
- It does not replace image signing or admission policy.
- It does not scan host-level runtime behavior; Falco owns runtime detection.

## Operating Notes

Treat Trivy reports as findings to triage, not as automatic rollout gates. If you want blocking behavior, add an explicit admission controller or CI policy rather than changing this operator silently.

## Verify

```bash
kubectl -n trivy-system get pods
kubectl get vulnerabilityreports.aquasecurity.github.io -A
kubectl get configauditreports.aquasecurity.github.io -A
deploy/pre-deployment/pre-deployment-check.sh --require trivy,trivy-reports
```

## Upgrade Notes

Scanner output and CRDs can change between chart versions. Review report schema changes before consuming reports in automation.

Upstream reference: [Trivy Operator documentation](https://aquasecurity.github.io/trivy-operator/).
