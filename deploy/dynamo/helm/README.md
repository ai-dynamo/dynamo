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

# 🚀 Deploy Dynamo Cloud to Kubernetes

## 🏗️ Building Docker images for Dynamo Cloud components

You can build and push Docker images for the Dynamo cloud components (API server, API store, and operator) to any container registry of your choice. Here's how to build each component:

### 📋 Prerequisites
- [Earthly](https://earthly.dev/) installed
- Docker installed and running
- Access to a container registry of your choice

### ⚙️ Building and Pushing Images

First, set the required environment variables:
```bash
export DOCKER_SERVER=<CONTAINER_REGISTRY>
export IMAGE_TAG=<TAG>
```

As a description of the placeholders:
- `<CONTAINER_REGISTRY>`: Your container registry (e.g., `nvcr.io`, `docker.io/<your-username>`, etc.)
- `<TAG>`: The tag you want to use for the image (e.g., `latest`, `0.0.1`, etc.)

Note: Make sure you're logged in to your container registry before pushing images. For example:
```bash
docker login <CONTAINER_REGISTRY>
```

You can build each component individually or build all components at once:

#### 🛠️ Build and push platform components
```bash
earthly --push +all-docker --DOCKER_SERVER=$DOCKER_SERVER --IMAGE_TAG=$IMAGE_TAG
```

## 🚀 Deploy Dynamo Cloud Platform

### 📋 Prerequisites
Before deploying Dynamo Cloud, ensure your Kubernetes cluster meets the following requirements:

#### 1. 🛡️ Istio Installation
Dynamo Cloud requires Istio for service mesh capabilities. Verify Istio is installed and running:

```bash
# Check if Istio is installed
kubectl get pods -n istio-system

# Expected output should show running Istio pods
# istiod-* pods should be in Running state
```

#### 2. 💾 PVC Support with Default Storage Class
Dynamo Cloud requires Persistent Volume Claim (PVC) support with a default storage class. Verify your cluster configuration:

```bash
# Check if default storage class exists
kubectl get storageclass

# Expected output should show at least one storage class marked as (default)
# Example:
# NAME                 PROVISIONER             RECLAIMPOLICY   VOLUMEBINDINGMODE      ALLOWVOLUMEEXPANSION   AGE
# standard (default)   kubernetes.io/gce-pd    Delete          Immediate              true                   1d
```

> [!TIP]
> Don't have a Kubernetes cluster? Check out our [Minikube setup guide](../../../docs/guides/dynamo_deploy/minikube.md) to set up a local environment! 🏠

### 📥 Installation

1. Set the required environment variables:
```bash
export PROJECT_ROOT=($pwd)
export DOCKER_USERNAME=<your-docker-username>
export DOCKER_PASSWORD=<your-docker-password>
export DOCKER_SERVER=<your-docker-server>
export IMAGE_TAG=<TAG>  # Use the same tag you used when building the images
export NAMESPACE=dynamo-cloud    # change this to whatever you want!
export DYNAMO_INGRESS_SUFFIX=dynamo-cloud.com # change this to whatever you want!
```

2. [One-time Action] Create a new kubernetes namespace and set it as your default. Create image pull secrets if needed.

```bash
cd $PROJECT_ROOT/deploy/dynamo/helm
kubectl create namespace $NAMESPACE
kubectl config set-context --current --namespace=$NAMESPACE

kubectl create secret docker-registry docker-imagepullsecret \
  --docker-server=$DOCKER_SERVER \
  --docker-username=$DOCKER_USERNAME \
  --docker-password=$DOCKER_PASSWORD \
  --namespace=$NAMESPACE
```

3. Deploy the helm chart using the deploy script:

```bash
./deploy.sh
```
