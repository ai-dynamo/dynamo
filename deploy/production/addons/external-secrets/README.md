<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# External Secrets Operator - Production Addon

External Secrets Operator (ESO) syncs secrets from an external provider into Kubernetes `Secret` resources. In this profile it is the baseline secret bridge for Hugging Face tokens, SMG secrets, and any future cloud-provider credentials that should not live directly in Git.

## Deployment

| Field | Value |
|---|---|
| Argo CD app | [`gitops/apps/05-external-secrets.yaml`](../../gitops/apps/05-external-secrets.yaml) |
| Chart | `external-secrets` |
| Source | `https://charts.external-secrets.io` |
| Version | `2.4.0` |
| Namespace | `external-secrets` |
| Values | [`values.yaml`](values.yaml) |
| Sync wave | `0` |
| Baseline | Yes |

## What This Addon Owns

| Capability | Local setting |
|---|---|
| ESO CRDs | `installCRDs: true`. |
| Prometheus scraping | `serviceMonitor.enabled: true` with `release: kube-prometheus-stack`. |
| Webhook certificates | `webhook.certManager.enabled: false`; chart-managed cert path is used. |

## What This Addon Does Not Own

- It does not create a `ClusterSecretStore`.
- It does not create cloud IAM roles, cloud secret paths, or provider credentials.
- It does not decide secret names for application charts; individual apps own their `ExternalSecret` objects.

## Production Contract

Any addon that needs secrets should reference ESO through explicit `ExternalSecret` manifests. The current SMG secret mirror expects a `ClusterSecretStore` named `aws-secrets`; see [`examples/clustersecretstore-aws.yaml`](../../examples/clustersecretstore-aws.yaml).

## Verify

```bash
kubectl -n external-secrets get pods
kubectl get crd externalsecrets.external-secrets.io clustersecretstores.external-secrets.io
kubectl get externalsecrets -A
deploy/pre-deployment/pre-deployment-check.sh --require external-secrets
```

## Upgrade Notes

Review major ESO bumps before merging. CRD version or provider behavior changes can break existing `ExternalSecret` resources even when the controller pod itself upgrades cleanly.

Upstream reference: [External Secrets Operator documentation](https://external-secrets.io/).
