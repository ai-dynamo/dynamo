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

# Installation Guide for Dynamo Kubernetes Platform

Deploy and manage Dynamo inference graphs on Kubernetes with automated orchestration and scaling, using the Dynamo Kubernetes Platform.

## Quick Start Paths

Platform is installed using Dynamo Kubernetes Platform [helm chart](/deploy/cloud/helm/platform/README.md).

**Path A: Production Install**
Install from published artifacts on your existing cluster → [Jump to Path A](#path-a-production-install)

**Path B: Local Development**
Set up Minikube first → [Minikube Setup](minikube.md) → Then follow Path A

**Path C: Custom Development**
Build from source for customization → [Jump to Path C](#path-c-custom-development)

All helm install commands could be overridden by either setting the values.yaml file or by passing in your own values.yaml:

```bash
helm install ...
  -f your-values.yaml
```

and/or setting values as flags to the helm install command, as follows:

```bash
helm install ...
  --set "your-value=your-value"
```

## Prerequisites

```bash
# Required tools
kubectl version --client  # v1.24+
helm version             # v3.0+
docker version           # Running daemon

# Set your inference runtime image
export RELEASE_VERSION=0.x.x # any version of Dynamo 0.3.2+ listed at https://github.com/ai-dynamo/dynamo/releases
export DYNAMO_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:${RELEASE_VERSION}
# Also available: sglang-runtime, tensorrtllm-runtime
```

> [!TIP]
> No cluster? See [Minikube Setup](minikube.md) for local development.

## Path A: Production Install

Install from [NGC published artifacts](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo/artifacts) in 3 steps.

```bash
# 1. Set environment
export NAMESPACE=dynamo-system
export RELEASE_VERSION=0.x.x # any version of Dynamo 0.3.2+ listed at https://github.com/ai-dynamo/dynamo/releases

# 2. Install CRDs
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-crds-${RELEASE_VERSION}.tgz
helm install dynamo-crds dynamo-crds-${RELEASE_VERSION}.tgz --namespace default

# 3. Install Platform
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${RELEASE_VERSION}.tgz
helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz --namespace ${NAMESPACE} --create-namespace
```

> [!TIP]
> For multinode deployments, you need to enable Grove and Kai Scheduler.
> You might chose to install them manually or through the dynamo-platform helm install command.
> When using the dynamo-platform helm install command, Grove and Kai Scheduler are NOT installed by default. You can enable their installation by setting the following flags in the helm install command:

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

> [!TIP]
> By default, Dynamo Operator is installed cluster-wide and will monitor all namespaces.
> If you wish to restrict the operator to monitor only a specific namespace (the helm release namespace by default), you can set the namespaceRestriction.enabled to true.
> You can also change the restricted namespace by setting the targetNamespace property.

```bash
--set "dynamo-operator.namespaceRestriction.enabled=true"
--set "dynamo-operator.namespaceRestriction.targetNamespace=dynamo-namespace" # optional
```

→ [Verify Installation](#verify-installation)

## Path C: Custom Development

Build and deploy from source for customization.

```bash
# 1. Set environment
export NAMESPACE=dynamo-system
export DOCKER_SERVER=nvcr.io/nvidia/ai-dynamo/  # or your registry
export DOCKER_USERNAME='$oauthtoken'
export DOCKER_PASSWORD=<YOUR_NGC_CLI_API_KEY>
export IMAGE_TAG=${RELEASE_VERSION}

# 2. Build operator
cd deploy/cloud/operator

# 2.1 Alternative 1 : Build and push the operator image for multiple platforms
docker buildx create --name multiplatform --driver docker-container --bootstrap
docker buildx use multiplatform
docker buildx build --platform linux/amd64,linux/arm64 -t $DOCKER_SERVER/dynamo-operator:$IMAGE_TAG --push .

# 2.2 Alternative 2 : Build and push the operator image for a single platform
docker build -t $DOCKER_SERVER/dynamo-operator:$IMAGE_TAG . && docker push $DOCKER_SERVER/dynamo-operator:$IMAGE_TAG

cd -

# 3. Create namespace and secrets to be able to pull the operator image (only needed if you pushed the operator image to a private registry)
kubectl create namespace ${NAMESPACE}
kubectl create secret docker-registry docker-imagepullsecret \
  --docker-server=${DOCKER_SERVER} \
  --docker-username=${DOCKER_USERNAME} \
  --docker-password=${DOCKER_PASSWORD} \
  --namespace=${NAMESPACE}

# 4. Install CRDs
helm upgrade --install dynamo-crds ./crds/ --namespace default

# 5. Install Platform
helm dep build ./platform/
helm install dynamo-platform ./platform/ \
  --namespace ${NAMESPACE} \
  --set "dynamo-operator.controllerManager.manager.image.repository=${DOCKER_SERVER}/dynamo-operator" \
  --set "dynamo-operator.controllerManager.manager.image.tag=${IMAGE_TAG}" \
  --set "dynamo-operator.imagePullSecrets[0].name=docker-imagepullsecret"
```

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
   - [vLLM Deployments](/components/backends/vllm/deploy/README.md)
   - [SGLang Deployments](/components/backends/sglang/deploy/README.md)
   - [TensorRT-LLM Deployments](/components/backends/trtllm/deploy/README.md)

3. **Optional:**
   - [Set up Prometheus & Grafana](metrics.md)
   - [SLA Planner Quickstart Guide](sla_planner_quickstart.md) (for SLA-aware scheduling and autoscaling)

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

**Bitnami etcd "unrecognized" image?**

```bash
ERROR: Original containers have been substituted for unrecognized ones. Deploying this chart with non-standard containers is likely to cause degraded security and performance, broken chart features, and missing environment variables.
```
This error that you might encounter during helm install is due to bitnami changing their docker repository to a [secure one](https://github.com/bitnami/charts/tree/main/bitnami/etcd#%EF%B8%8F-important-notice-upcoming-changes-to-the-bitnami-catalog).

just add the following to the helm install command:
```bash
--set "etcd.image.repository=bitnamilegacy/etcd" --set "etcd.global.security.allowInsecureImages=true"
```

**Clean uninstall?**
```bash
./uninstall.sh  # Removes all CRDs and platform
```

## Advanced Options

- [Helm Chart Configuration](/deploy/cloud/helm/platform/README.md)
- [GKE-specific setup](gke_setup.md)
- [Create custom deployments](create_deployment.md)
- [Dynamo Operator details](dynamo_operator.md)
- [Model Express Server details](https://github.com/ai-dynamo/modelexpress)
