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

# Dynamo Cloud Kubernetes Platform (Dynamo Deploy)

The Dynamo Cloud platform is a comprehensive solution for deploying and managing Dynamo inference graphs (also referred to as pipelines) in Kubernetes environments. It provides a streamlined experience for deploying, scaling, and monitoring your inference services. You can interface with Dynamo Cloud using the `deploy` subcommand available in the Dynamo CLI (for example, `dynamo deploy`)

## Overview

The Dynamo cloud platform consists of several key components:

- **Dynamo Operator**: A Kubernetes operator that manages the lifecycle of Dynamo inference graphs from build ➡️ deploy. For more information on the operator, see [Dynamo Kubernetes Operator Documentation](../dynamo_deploy/dynamo_operator.md)
- **Custom Resources**: Kubernetes custom resources for defining and managing Dynamo services

These components work together to provide a seamless deployment experience, handling everything from containerization to scaling and monitoring.

![Dynamo Deploy system deployment diagram.](../../images/dynamo-deploy.png)

## Prerequisites

Before getting started with the Dynamo cloud platform, ensure you have:

- A Kubernetes cluster (version 1.24 or later)
- [Earthly](https://earthly.dev/) installed for building components
- Docker installed and running
- Access to a container registry (e.g., Docker Hub, NVIDIA NGC, etc.)
- `kubectl` configured to access your cluster
- Helm installed (version 3.0 or later)


> [!TIP]
> Don't have a Kubernetes cluster? Check out our [Minikube setup guide](../../../docs/guides/dynamo_deploy/minikube.md) to set up a local environment! 🏠

#### 🏗️ Build Dynamo inference runtime.

[One-time Action]
Before you could use Dynamo make sure you have setup the Inference Runtime Image.
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

## Building Docker Images for Dynamo Cloud Components

The Dynamo cloud platform components need to be built and pushed to a container registry before deployment. You can build these components individually or all at once.

### Setting Up Environment Variables

First, set the required environment variables for building and pushing images:

```bash
# Set your container registry
export DOCKER_SERVER=<CONTAINER_REGISTRY>
# Set the image tag (e.g., latest, 0.0.1, etc.)
export IMAGE_TAG=<TAG>
```

As a description of the placeholders:
- `<CONTAINER_REGISTRY>`: Your container registry (e.g., `nvcr.io`, `docker.io/<your-username>`, etc.)
- `<TAG>`: The tag you want to use for the images of the Dynamo cloud components (e.g., `latest`, `0.0.1`, etc.)
If the runtime image tag is not explicitly set, the default is the `latest`.

The tag will go into the dynamo-operator:<IMAGE_TAG> image for the Operator.  The runtime (base) image handles the inference toolchain and the sdk and built by the (`build.sh`). The tags do not have to match the runtime  image tag but the images must be compatible.

**Important** Make sure you're logged in to your container registry before pushing images. For example:

```bash
docker login <CONTAINER_REGISTRY>
```

### Building Components

You can build and push all platform components at once:

```bash
earthly --push +all-docker --DOCKER_SERVER=$DOCKER_SERVER --IMAGE_TAG=$IMAGE_TAG
```

### 🚀 Deploying the Dynamo Cloud Platform

Once you've built and pushed the components, you can deploy the platform to your Kubernetes cluster.

### Prerequisites

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



### Installation using the helper script

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

``` {note}
DOCKER_USERNAME and DOCKER_PASSWORD are optional and only needed if you want to pull docker images from a private registry.
A docker image pull secret is created automatically if these variables are set. Its name is `docker-imagepullsecret` unless overridden by the `DOCKER_SECRET_NAME` environment variable.
```

The Dynamo Cloud Platform auto-generates docker images for pipelines and pushes them to a container registry.
By default, the platform uses the same container registry as the platform components (specified by `DOCKER_SERVER`).
However, you can use a different container registry for the platform components by making sure an associated kubernetes secret is present:

```bash
kubectl create secret docker-registry dynamo-components-imagepullsecret \
  --docker-server=<docker-registry-for-dynamo-components> \
  --docker-username=<username> \
  --docker-password=<password> \
  --namespace=${NAMESPACE}
```

If you wish to expose your Dynamo Cloud Platform externally, you can setup the following environment variables:

```bash
# if using ingress
export INGRESS_ENABLED="true"
export INGRESS_CLASS="nginx" # or whatever ingress class you have configured

# if using istio
export ISTIO_ENABLED="true"
export ISTIO_GATEWAY="istio-system/istio-ingressgateway" # or whatever istio gateway you have configured
```

Running the installation script with `--interactive` guides you through the process of exposing your Dynamo Cloud Platform externally if you don't want to set these environment variables manually.

2. [One-time Action] Create a new kubernetes namespace and set it as your default.

```bash
cd deploy/cloud/helm
kubectl create namespace $NAMESPACE
kubectl config set-context --current --namespace=$NAMESPACE
```

3. Deploy the Helm charts (install CRDs first, then platform) using the deployment script:

```bash
./deploy.sh --crds
```

if you want guidance during the process, run the deployment script with the `--interactive` flag:

```bash
./deploy.sh --crds --interactive
```

omitting `--crds` will skip the CRDs installation/upgrade. This is useful when installing on a shared cluster as CRDs are cluster-scoped resources.

If you'd like to only generate the generated-values.yaml file without deploying to Kubernetes (e.g., for inspection, CI workflows, or dry-run testing), use:

```bash
./deploy_dynamo_cloud.py --yaml-only
```


### Installation using published helm chart

To install Dynamo Cloud using the published Helm chart, you'll need to configure Docker registry credentials and image settings.


#### Environment Setup

Set the required environment variables:

```bash
# Docker registry configuration
export DOCKER_SERVER="your-registry.com"                    # Docker registry server where images of dynamo cloud services (operator) are available
export IMAGE_TAG="v1.0.0"                                   # Image tag to deploy
export NAMESPACE="dynamo-cloud"                             # Target namespace

# Components-specific Docker registry (if different from DOCKER_SERVER)
export COMPONENTS_DOCKER_SERVER="your-pipeline-registry.com" # Registry for Dynamo components images

# Image pull secret for the operator itself
export DOCKER_SECRET_NAME="my-pull-secret"                       # Secret for pulling images of dynamo cloud services (operator) operator images
export COMPONENTS_DOCKER_SECRET_NAME="my-components-pull-secret" # Secret for pulling images of dynamo components images (if needed)
```

you can easily create an image pull secret with the following command :

```bash
kubectl create secret docker-registry ${DOCKER_SECRET_NAME} \
  --docker-server=${DOCKER_SERVER} \
  --docker-username=<docker-server-username> \
  --docker-password=<docker-server-password> \
  --namespace=${NAMESPACE}

# Only if using a different registry for Dynamo components
kubectl create secret docker-registry ${COMPONENTS_DOCKER_SECRET_NAME} \
  --docker-server=${COMPONENTS_DOCKER_SERVER} \
  --docker-username=<components-docker-server-username> \
  --docker-password=<components-docker-server-password> \
  --namespace=${NAMESPACE}

```

#### Installation Commands

**Step 1: Install Custom Resource Definitions (CRDs)**

```bash
helm install dynamo-crds dynamo-crds-helm-chart.tgz \
  --namespace default \
  --wait \
  --atomic
```

**Step 2: Install Dynamo Platform**

Run the following helm command:

```bash
helm install dynamo-platform dynamo-platform-helm-chart.tgz \
  --namespace ${NAMESPACE} \
  --set "dynamo-operator.controllerManager.manager.image.repository=${DOCKER_SERVER}/dynamo-operator" \
  --set "dynamo-operator.controllerManager.manager.image.tag=${IMAGE_TAG}" \
  --set "dynamo-operator.imagePullSecrets[0].name=${DOCKER_SECRET_NAME}"
```

### Cloud Provider-Specific deployment

#### Google Kubernetes Engine (GKE) deployment

You can find detailed instructions for deployment in GKE [here](../dynamo_deploy/gke_setup.md)

## Next Steps

After deploying the Dynamo cloud platform, you can:

1. Deploy your first inference graph using the [Dynamo CLI](operator_deployment.md)
2. Deploy Dynamo LLM graphs to Kubernetes using the [Dynamo CLI](../../examples/llm_deployment.md)
3. Manage your deployments using the Dynamo CLI

For more detailed information about deploying inference graphs, see the [Dynamo Deploy Guide](README.md).
