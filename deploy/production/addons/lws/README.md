<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LeaderWorkerSet - Optional Production Addon

LeaderWorkerSet (LWS) is an optional alternative multinode orchestration path. The baseline Dynamo profile uses Grove plus KAI; install LWS only when intentionally testing or operating the LWS plus Volcano path.

## Deployment

| Field | Value |
|---|---|
| Argo CD app | [`gitops/optional/lws.yaml`](../../gitops/optional/lws.yaml) |
| Chart path | `charts/lws` |
| Source | `https://github.com/kubernetes-sigs/lws` |
| Version | `v0.8.0` |
| Namespace | `lws-system` |
| Values | [`values.yaml`](values.yaml) |
| Sync wave | `8` |
| Baseline | No |

## What This Addon Owns

| Capability | Local setting |
|---|---|
| LWS controller | Installed from the upstream LWS chart. |
| Prometheus metrics | `enablePrometheus: true`. |
| Manager image | `image.manager.tag: v0.8.0`, `pullPolicy: IfNotPresent`. |
| Gang scheduling integration | `gangSchedulingManagement.schedulerProvider: volcano`. |

## What This Addon Does Not Own

- It does not install Volcano; deploy [`../volcano`](../volcano/) separately.
- It does not replace Grove/KAI in the baseline profile.
- It does not change existing DynamoGraphDeployment specs automatically.

## Operating Notes

LWS and Volcano should be treated as a paired optional stack. The preflight check `lws-volcano` expects CRDs from both.

## Verify

```bash
kubectl -n lws-system get pods
kubectl get crd leaderworkersets.leaderworkerset.x-k8s.io
deploy/pre-deployment/pre-deployment-check.sh --require lws-volcano
```

## Upgrade Notes

LWS API changes can affect rolling updates and topology placement. Test multinode inference before promoting a chart or CRD update.

Upstream reference: [LeaderWorkerSet documentation](https://lws.sigs.k8s.io/docs/).
