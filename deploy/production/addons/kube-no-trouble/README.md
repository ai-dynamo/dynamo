<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# kube-no-trouble - Optional Production Addon

kube-no-trouble (`kubent`) checks a cluster for deprecated Kubernetes API usage before a Kubernetes minor-version upgrade. It is an operational check, not a long-running Dynamo runtime dependency.

## Deployment

| Field | Value |
|---|---|
| Argo CD app | None |
| Example manifest | [`examples/kube-no-trouble-job.yaml`](../../examples/kube-no-trouble-job.yaml) |
| Image | `ghcr.io/doitintl/kube-no-trouble:0.7.3` |
| Namespace | `kube-no-trouble` |
| Baseline | No |

## What This Addon Owns

| Capability | Local setting |
|---|---|
| One-shot API scan | Kubernetes `Job` that runs `kubent`. |
| Read-only cluster access | ServiceAccount, ClusterRole, and ClusterRoleBinding with `get` and `list`. |
| Failing on findings | Job argument `--exit-error`. |

## What This Addon Does Not Own

- It does not install a controller.
- It does not run continuously.
- It does not replace Kubernetes release-note review.
- It does not remediate deprecated resources automatically.

## Operating Notes

Run this before Kubernetes minor upgrades and before promoting GitOps changes into a newer cluster version. Delete and recreate the Job for repeated checks.

## Verify

```bash
kubectl apply -f deploy/production/examples/kube-no-trouble-job.yaml
kubectl -n kube-no-trouble logs job/kube-no-trouble
deploy/pre-deployment/pre-deployment-check.sh --require kubent
```

## Upgrade Notes

Confirm that the kubent rules cover the Kubernetes target version. For newer Kubernetes releases, cross-check with the Kubernetes deprecated API migration guide before treating a clean run as sufficient.

Upstream reference: [kube-no-trouble repository](https://github.com/doitintl/kube-no-trouble).
