<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Dynamo Kubernetes Platform

Deploy and manage Dynamo inference graphs on Kubernetes with automated orchestration and scaling, using the Dynamo Kubernetes Platform.

## Quick Start Paths

**Path A: Production Install**
Install from published artifacts on your existing cluster → [Jump to Path A](#path-a-production-install)

**Path B: Local Development**
Set up Minikube first → [Minikube Setup](minikube.md) → Then follow Path A

**Path C: Custom Development**
Build from source for customization → [Jump to Path C](#path-c-custom-development)

## Prerequisites

```bash
# Required tools
kubectl version --client  # v1.24+
helm version             # v3.0+
docker version           # Running daemon

# Set your inference runtime image
export DYNAMO_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.5.0
# Also available: sglang-runtime, tensorrtllm-runtime
```

> [!TIP]
> No cluster? See [Minikube Setup](minikube.md) for local development.

## Path A: Production Install

Install from [NGC published artifacts](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo/artifacts) in 3 steps.

```bash
# 1. Set environment
export NAMESPACE=dynamo-kubernetes
export RELEASE_VERSION=0.5.0 # any version of Dynamo 0.3.2+

# 2. Install CRDs
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-crds-${RELEASE_VERSION}.tgz
helm install dynamo-crds dynamo-crds-${RELEASE_VERSION}.tgz --namespace default

# 3. Install Platform
kubectl create namespace ${NAMESPACE}
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${RELEASE_VERSION}.tgz
helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz --namespace ${NAMESPACE}
```

> [!TIP]
> By default, Grove and Kai Scheduler are NOT installed. You can enable them by setting the following flags in the helm install command:

```bash
--set "grove.enabled=true"
--set "kai-scheduler.enabled=true"
```

> [!TIP]
> By default, Model Express Server is not used.
> If you wish to use an existing Model Express Server, you can set the modelExpressURL to the existing server's URL in the helm install command:

```bash
--set "dynamo-operator.modelExpressURL=http://model-express-server.model-express.svc.cluster.local:8080"
```


→ [Verify Installation](#verify-installation)

## Path C: Custom Development

Build and deploy from source for customization.

### Quick Deploy Script

```bash
# 1. Set environment
export NAMESPACE=dynamo-cloud
export DOCKER_SERVER=nvcr.io/nvidia/ai-dynamo/  # or your registry
export DOCKER_USERNAME='$oauthtoken'
export DOCKER_PASSWORD=<YOUR_NGC_CLI_API_KEY>
export IMAGE_TAG=0.5.0

# 2. Build operator
cd deploy/cloud/operator
earthly --push +docker --DOCKER_SERVER=$DOCKER_SERVER --IMAGE_TAG=$IMAGE_TAG
cd -

# 3. Create namespace and secrets
kubectl create namespace ${NAMESPACE}
kubectl create secret docker-registry docker-imagepullsecret \
  --docker-server=${DOCKER_SERVER} \
  --docker-username=${DOCKER_USERNAME} \
  --docker-password=${DOCKER_PASSWORD} \
  --namespace=${NAMESPACE}

# 4. Deploy
helm repo add bitnami https://charts.bitnami.com/bitnami
./deploy.sh --crds
```

### Manual Steps (Alternative)

<details>
<summary>Click to expand manual installation steps</summary>

**Step 1: Install CRDs**
```bash
helm install dynamo-crds ./crds/ --namespace default
```

**Step 2: Install Platform**
```bash
helm dep build ./platform/
helm install dynamo-platform ./platform/ \
  --namespace ${NAMESPACE} \
  --set "dynamo-operator.controllerManager.manager.image.repository=${DOCKER_SERVER}/dynamo-operator" \
  --set "dynamo-operator.controllerManager.manager.image.tag=${IMAGE_TAG}" \
  --set "dynamo-operator.imagePullSecrets[0].name=docker-imagepullsecret"
```
</details>

→ [Verify Installation](#verify-installation)

## Verify Installation

```bash
# Check CRDs
kubectl get crd | grep dynamo

# Check operator and platform pods
kubectl get pods -n ${NAMESPACE}
# Expected: dynamo-operator-* and etcd-* pods Running
```

## Next Steps

1. **Deploy Model/Workflow**
   ```bash
   # Example: Deploy a vLLM workflow with Qwen3-0.6B using aggregated serving
   kubectl apply -f components/backends/vllm/deploy/agg.yaml -n ${NAMESPACE}

   # Port forward and test
   kubectl port-forward svc/agg-vllm-frontend 8000:8000 -n ${NAMESPACE}
   curl http://localhost:8000/v1/models
   ```

2. **Explore Backend Guides**
   - [vLLM Deployments](../../../components/backends/vllm/deploy/README.md)
   - [SGLang Deployments](../../../components/backends/sglang/deploy/README.md)
   - [TensorRT-LLM Deployments](../../../components/backends/trtllm/deploy/README.md)

3. **Optional:**
   - [Set up Prometheus & Grafana](metrics.md)
   - [SLA Planner Deployment Guide](sla_planner_deployment.md) (for advanced SLA-aware scheduling and autoscaling)

## Troubleshooting

**Pods not starting?**
```bash
kubectl describe pod <pod-name> -n ${NAMESPACE}
kubectl logs <pod-name> -n ${NAMESPACE}
```

**HuggingFace model access?**
```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE}
```

**Clean uninstall?**
```bash
./uninstall.sh  # Removes all CRDs and platform
```

## Advanced Options

- [Helm Chart Configuration](../../../deploy/cloud/helm/platform/README.md)
- [GKE-specific setup](gke_setup.md)
- [Create custom deployments](create_deployment.md)
- [Dynamo Operator details](dynamo_operator.md)
- [Model Express Server details](https://github.com/ai-dynamo/modelexpress)
