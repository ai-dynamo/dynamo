<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Actions Runner Controller - Production Addon

Actions Runner Controller (ARC) is the optional control plane for running GitHub Actions runners inside the Kubernetes cluster. It is a CI integration only; it is not part of the Dynamo serving path and is intentionally kept outside the baseline root app.

## Deployment

| Field | Value |
|---|---|
| Argo CD app | [`gitops/optional/actions-runner-controller.yaml`](../../gitops/optional/actions-runner-controller.yaml) |
| Chart | `gha-runner-scale-set-controller` |
| Source | `ghcr.io/actions/actions-runner-controller-charts` |
| Version | `0.14.1` |
| Namespace | `arc-systems` |
| Values | [`values.yaml`](values.yaml) |
| Baseline | No |

## What This Addon Owns

| Capability | Local setting |
|---|---|
| ARC controller lifecycle | Installed through the upstream controller chart. |
| Controller logging | `flags.logLevel: info`. |
| Metrics endpoints | Controller and listener metrics bind to `:8080`; listener path is `/metrics`. |

## What This Addon Does Not Own

- It does not create runner scale sets.
- It does not create GitHub App, PAT, or organization credentials.
- It does not run model-serving CI jobs by itself.
- It does not change Dynamo, SMG, SGLang, or GPU scheduling resources.

## Operating Notes

Deploy this only when GitHub Actions work must run inside the cluster. Keep production serving and CI capacity separated with node selectors, taints, or runner scale-set limits before adding actual runners.

Runner scale-set definitions should live in a separate GitOps app or a tightly scoped directory, because they contain repository/org policy and usually reference credentials.

## Verify

```bash
kubectl -n arc-systems get pods
kubectl -n arc-systems get svc
kubectl -n arc-systems port-forward svc/arc-gha-rs-controller-metrics 8080:8080
curl -sS http://127.0.0.1:8080/metrics | head
```

## Upgrade Notes

Patch and minor chart bumps can be handled by Renovate. Major ARC bumps should be reviewed manually because GitHub runner scale-set APIs and authentication flows can change across major releases.

Upstream reference: [GitHub Actions Runner Controller documentation](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners-with-actions-runner-controller).
