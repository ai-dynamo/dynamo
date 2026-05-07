<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Volcano - Optional Production Addon

Volcano is the optional batch scheduler used only with the alternate LWS multinode orchestration path. The baseline Dynamo production profile uses Grove plus KAI Scheduler, so Volcano is intentionally kept outside the root app.

## Deployment

| Field | Value |
|---|---|
| Argo CD app | [`gitops/optional/volcano.yaml`](../../gitops/optional/volcano.yaml) |
| Chart | `volcano` |
| Source | `https://volcano-sh.github.io/helm-charts` |
| Version | `1.14.1` |
| Namespace | `volcano-system` |
| Values | [`values.yaml`](values.yaml) |
| Sync wave | `8` |
| Baseline | No |

## What This Addon Owns

| Capability | Local setting |
|---|---|
| Volcano scheduler stack | Installed from the upstream chart. |
| Controller metrics | `custom.controller_metrics_enable: true`. |
| Scheduler metrics | `custom.scheduler_metrics_enable: true`. |
| General metrics | `custom.metrics_enable: true`. |

## What This Addon Does Not Own

- It does not install LWS; deploy [`../lws`](../lws/) separately.
- It does not replace the Grove plus KAI baseline path.
- It does not migrate existing Dynamo workloads to Volcano scheduling.

## Operating Notes

Use Volcano only when validating or operating the LWS plus Volcano path. Keep it separate from Grove plus KAI testing so scheduler behavior changes can be isolated.

## Verify

```bash
kubectl -n volcano-system get pods
kubectl get crd queues.scheduling.volcano.sh podgroups.scheduling.volcano.sh
deploy/pre-deployment/pre-deployment-check.sh --require lws-volcano
```

## Upgrade Notes

Review Volcano scheduler and CRD changes with the matching LWS version. Promote only after a multinode workload has completed scheduling and rollout checks on the target cluster class.

Upstream reference: [Volcano documentation](https://volcano.sh/en/docs/).
