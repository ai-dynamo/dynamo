<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Grove - Production Addon

Grove provides Dynamo's preferred topology-aware gang scheduling and autoscaling APIs for complex AI inference deployments. In this profile it is paired with KAI Scheduler and enabled in the Dynamo platform values.

## Deployment

| Field | Value |
|---|---|
| Argo CD app | [`gitops/apps/51-grove.yaml`](../../gitops/apps/51-grove.yaml) |
| Chart | `grove-charts` |
| Source | `ghcr.io/ai-dynamo/grove` |
| Version | `v0.1.0-alpha.8` |
| Namespace | `grove-system` |
| Values | [`values.yaml`](values.yaml) |
| Sync wave | `5` |
| Baseline | Yes |

## What This Addon Owns

| Capability | Local setting |
|---|---|
| Grove controller | `controllerManager.replicas: 1`. |
| Grove CRDs | Installed by the Grove chart. |
| Webhook secret path | `webhookServerSecret.enabled: false`. |

## What This Addon Does Not Own

- It does not install KAI Scheduler.
- It does not replace the Dynamo operator.
- It does not decide per-model GPU topology; Dynamo workload specs and scheduler policy do that.

## Dynamo Contract

The Dynamo platform values set `global.grove.enabled: true` and `global.grove.install: false`. That means Dynamo is allowed to use Grove integration, but this standalone addon owns the Grove installation lifecycle.

## Verify

```bash
kubectl -n grove-system get pods
kubectl get crd | grep grove.io
deploy/pre-deployment/pre-deployment-check.sh --require grove-kai
```

## Upgrade Notes

Grove is still alpha in this profile. Review CRD and controller changes before updating because Dynamo multinode scheduling depends on API compatibility.

Upstream reference: [Grove repository](https://github.com/ai-dynamo/grove).
