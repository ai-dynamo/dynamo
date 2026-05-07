<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KAI Scheduler - Production Addon

KAI Scheduler is the baseline AI workload scheduler for this production profile. It provides queueing and scheduling APIs used with Grove and Dynamo multinode orchestration.

## Deployment

| Field | Value |
|---|---|
| Argo CD app | [`gitops/apps/50-kai-scheduler.yaml`](../../gitops/apps/50-kai-scheduler.yaml) |
| Chart | `kai-scheduler` |
| Source | `ghcr.io/kai-scheduler/kai-scheduler` |
| Version | `v0.14.0` |
| Namespace | `kai-scheduler` |
| Values | [`values.yaml`](values.yaml) |
| Sync wave | `5` |
| Baseline | Yes |

## What This Addon Owns

| Capability | Local setting |
|---|---|
| KAI controller/scheduler stack | Installed from the upstream chart. |
| Namespace binding | `global.namespace: kai-scheduler`. |
| Queue CRDs | Installed by the chart and checked by preflight. |

## What This Addon Does Not Own

- It does not install Grove.
- It does not install Volcano or LWS.
- It does not decide Dynamo queues by itself; Dynamo platform integration creates/uses the Dynamo queues.

## Dynamo Contract

The Dynamo platform values set `global.kai-scheduler.enabled: true` and `global.kai-scheduler.install: false`. Keep KAI as a separate Argo CD app so chart upgrades remain isolated from Dynamo platform chart merges.

## Verify

```bash
kubectl -n kai-scheduler get pods
kubectl get queues.scheduling.run.ai
deploy/pre-deployment/pre-deployment-check.sh --require grove-kai,kai-queue
```

## Upgrade Notes

KAI changes affect queue semantics and scheduler behavior. Validate queue creation and at least one Dynamo deployment before promoting a scheduler update.

Upstream reference: [KAI Scheduler repository](https://github.com/kai-scheduler/KAI-Scheduler).
