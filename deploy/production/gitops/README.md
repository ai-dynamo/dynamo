<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GitOps Bootstrap

Apply the Argo CD project first, then the root application:

```bash
kubectl apply -f deploy/production/gitops/project.yaml
kubectl apply -f deploy/production/gitops/root-app.yaml
```

The root application syncs the child applications in `gitops/apps`. The child applications install platform add-ons as independent Helm releases and deploy Dynamo from this repository.

Optional integrations live in `gitops/optional` and are not included by the root app. Apply them explicitly when the cluster needs that capability:

```bash
kubectl apply -f deploy/production/gitops/optional/keda.yaml
kubectl apply -f deploy/production/gitops/optional/opentelemetry.yaml
kubectl apply -f deploy/production/gitops/optional/actions-runner-controller.yaml
kubectl apply -f deploy/production/gitops/optional/parca.yaml
kubectl apply -f deploy/production/gitops/optional/volcano.yaml
kubectl apply -f deploy/production/gitops/optional/lws.yaml
```

The manifests default to `targetRevision: main` for `https://github.com/ai-blaise/k8s-playground.git`. Change that revision when validating a feature branch.

For k3s clusters, add `deploy/production/addons/gpu-operator/values-k3s.yaml` to the `gpu-operator` application value files before syncing. NVIDIA GPU Operator needs the k3s containerd socket and config-template paths to configure the NVIDIA runtime.
