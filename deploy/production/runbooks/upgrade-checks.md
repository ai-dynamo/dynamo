<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Kubernetes Upgrade Checks

Run kube-no-trouble before Kubernetes minor upgrades and before promoting rendered manifest changes:

```bash
kubectl apply -f deploy/production/examples/kube-no-trouble-job.yaml
kubectl logs -n kube-no-trouble job/kube-no-trouble
```

Confirm the check was installed:

```bash
deploy/pre-deployment/pre-deployment-check.sh --require kubent --output json
```
