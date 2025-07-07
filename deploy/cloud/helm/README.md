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

Dynamo Cloud acts as an orchestration layer between the end user and Kubernetes, handling the complexity of deploying your graphs for you.
Before you can deploy your graphs, you need to deploy the Dynamo Runtime and Dynamo Cloud images. This is a one-time action, only necessary the first time you deploy a DynamoGraph.


## 🏗️ Building Docker images for Dynamo Cloud components

You can build and push Docker images for the Dynamo cloud components (API server, API store, and operator) to any container registry of your choice. Here's how to build each component:

### 📋 Prerequisites
- [Earthly](https://earthly.dev/) installed
- Docker installed and running
- Access to a container registry of your choice

#### 🏗️ Build Dynamo inference runtime.

[One-time Action]
For basic cases you could use the prebuilt image for the Dynamo Inference Runtime.
Just export the environment variable. This will be the image used by your individual components. You pick whatever dynamo version you want or use the latest (default)

```bash
export DYNAMO_IMAGE=nvcr.io/nvidia/dynamo:latest-vllm
```

For advanced examples make sure you have first built and pushed to your registry Dynamo Base Image for Dynamo inference runtime. This is a one-time operation.

```bash
# Run the script to build the default dynamo:latest-vllm image.
./container/build.sh
export IMAGE_TAG=<TAG>
# retag the image
docker tag dynamo:latest-vllm <your-registry>/dynamo:${IMAGE_TAG}
docker push <your-registry>/dynamo:${IMAGE_TAG}
```


#### 🛠️ Build and push Dynamo Cloud platform components

[One-time Action]
You should also setup or build (advanced case) the images for the Dynamo Cloud Platform.
If you are a **👤 Dynamo User** you would use the standard images, so just export the appropriate tag:
```bash
export IMAGE_TAG=<TAG>
```

If you are a **🧑‍💻 Dynamo Contributor** you would have to rebuild the dynamo platform images as the code evolves.To do so please use the steps below.
Set the required environment variables:

```bash
export DOCKER_SERVER=<CONTAINER_REGISTRY>
export IMAGE_TAG=<TAG>
```

As a description of the placeholders:
- `<CONTAINER_REGISTRY>`: Your container registry (e.g., `nvcr.io`, `docker.io/<your-username>`, etc.)
- `<TAG>`: The tag you want to use for the images of the Dynamo cloud components (e.g., `latest`, `0.0.1`, etc.)
If the runtime image tag is not explicitly set, the default is the `latest`.

The tag will go into the dynamo-operator:<IMAGE_TAG> image for the Operator.  The runtime (base) image handles the inference toolchain and the sdk and built by the (`build.sh`). The tags do not have to match the runtime  image tag but the images must be compatible.


Note: Make sure you're logged in to your container registry before pushing images. For example:
```bash
docker login <CONTAINER_REGISTRY>
```

You can build each component individually or build all components at once:


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
export PROJECT_ROOT=$(pwd)
export DOCKER_USERNAME=<your-docker-username>
export DOCKER_PASSWORD=<your-docker-password>
export DOCKER_SERVER=<your-docker-server>
export IMAGE_TAG=<TAG>  # Use the same tag you used when building the images
export NAMESPACE=dynamo-cloud    # change this to whatever you want!
export DYNAMO_INGRESS_SUFFIX=dynamo-cloud.com # change this to whatever you want!
```

2. [One-time Action] Create a new kubernetes namespace and set it as your default. Create image pull secrets if needed.

```bash
cd $PROJECT_ROOT/deploy/cloud/helm
kubectl create namespace $NAMESPACE
kubectl config set-context --current --namespace=$NAMESPACE

kubectl create secret docker-registry docker-imagepullsecret \
  --docker-server=$DOCKER_SERVER \
  --docker-username=$DOCKER_USERNAME \
  --docker-password=$DOCKER_PASSWORD \
  --namespace=$NAMESPACE
```

3. Deploy Dynamo Cloud using the Helm chart via the provided deploy script:
To deploy the Dynamo Cloud Platform on Kubernetes, run:

```bash
./deploy_dynamo_cloud.sh
```

This will validate tools, configure your environment, generate generated-values.yaml, and deploy the platform using Helm.

If you'd like to only generate the generated-values.yaml file without deploying to Kubernetes (e.g., for inspection, CI workflows, or dry-run testing), use:

```bash
./deploy_dynamo_cloud.py --yaml-only
```



