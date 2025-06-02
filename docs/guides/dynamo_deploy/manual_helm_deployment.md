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

<a id="k8-helm-deploy"></a>
# Deploying Dynamo Inference Graphs to Kubernetes using Helm

This guide describes the deployment process of an inference graph created using the Dynamo SDK onto a Kubernetes cluster.

While this guide covers deployment of Dynamo inference graphs using Helm, the preferred method to deploy an inference graph is to [deploy with the Dynamo cloud platform](operator_deployment.md). The [Dynamo cloud platform](dynamo_cloud.md) simplifies the deployment and management of Dynamo inference graphs. It includes a set of components (Operator, Kubernetes Custom Resources, etc.) that work together to streamline the deployment and management process.

Once an inference graph is defined using the Dynamo SDK, it can be deployed onto a Kubernetes cluster using a simple `dynamo deploy` command that orchestrates the following deployment steps:

1. Building docker images from inference graph components on the cluster
2. Intelligently composing the encoded inference graph into a complete deployment on Kubernetes
3. Enabling autoscaling, monitoring, and observability for the inference graph
4. Easy administration of deployments via UI

## Helm Deployment Guide

### Setting up MicroK8s

Follow these steps to set up a local Kubernetes cluster using MicroK8s:

1. Install MicroK8s:
```bash
sudo snap install microk8s --classic
```

2. Configure user permissions:
```bash
sudo usermod -a -G microk8s $USER
sudo chown -R $USER ~/.kube
```

3. **Important**: Log out and log back in for the permissions to take effect

4. Start MicroK8s:
```bash
microk8s start
```

5. Enable required addons:
```bash
# Enable GPU support
microk8s enable gpu

# Enable storage support
# See: https://microk8s.io/docs/addon-hostpath-storage
microk8s enable storage
```

6. Configure kubectl:
```bash
mkdir -p ~/.kube
microk8s config >> ~/.kube/config
```

After completing these steps, you should be able to use the `kubectl` command to interact with your cluster.

### Installing Required Dependencies

Follow these steps to set up the namespace and install required components:

1. Set environment variables:
```bash
export NAMESPACE=dynamo-playground
export RELEASE_NAME=dynamo-platform
export PROJECT_ROOT=$(pwd)
```

2. Install NATS messaging system:
```bash
# Navigate to dependencies directory
cd $PROJECT_ROOT/deploy/helm/dependencies

# Add and update NATS Helm repository
helm repo add nats https://nats-io.github.io/k8s/helm/charts/
helm repo update

# Install NATS with custom values
helm install --namespace ${NAMESPACE} ${RELEASE_NAME}-nats nats/nats \
    --values nats-values.yaml
```

3. Install etcd key-value store:
```bash
# Install etcd using Bitnami chart
helm install --namespace ${NAMESPACE} ${RELEASE_NAME}-etcd \
    oci://registry-1.docker.io/bitnamicharts/etcd \
    --values etcd-values.yaml
```

After completing these steps, your cluster has the necessary messaging and storage infrastructure for running Dynamo inference graphs.

### Building and Deploying the Pipeline

Follow these steps to containerize and deploy your inference pipeline:

1. Build and containerize the pipeline:

``` {note}
For instructions on building and pushing the Dynamo base image, see [Building the Dynamo Base Image](../../get_started.md#building-the-dynamo-base-image).
```

```bash
# Navigate to example directory
cd $PROJECT_ROOT/examples/hello_world

# Set runtime image name
export DYNAMO_IMAGE=<dynamo_base_image>

# Build and containerize the Frontend service
dynamo build --containerize hello_world:Frontend
```

2. Push container to registry:
```bash
# Tag the built image for your registry
docker tag <BUILT_IMAGE_TAG> <TAG>

# Push to your container registry
docker push <TAG>
```

3. Deploy using Helm:
```bash
# Navigate to the deployment directory
cd $PROJECT_ROOT/deploy/helm

# Set release name for Helm
export HELM_RELEASE=hello-world-manual

# Generate Helm values file from Frontend service
dynamo get frontend > pipeline-values.yaml

# Install/upgrade Helm release
helm upgrade -i "$HELM_RELEASE" ./chart \
    -f pipeline-values.yaml \
    --set image=<TAG> \
    --set dynamoIdentifier="hello_world:Frontend" \
    -n "$NAMESPACE"
```

4. Test the deployment:
```bash
# Forward the service port to localhost
kubectl -n ${NAMESPACE} port-forward svc/${HELM_RELEASE}-frontend 3000:80

# Test the API endpoint
curl -X 'POST' 'http://localhost:3000/generate' \
    -H 'accept: text/event-stream' \
    -H 'Content-Type: application/json' \
    -d '{"text": "test"}'
```

### Using the Deployment Script

For convenience, you can use the deployment script at `deploy/helm/deploy.sh` that automates all of these steps:

```bash
export DYNAMO_IMAGE=<dynamo_docker_image_name>
./deploy.sh <docker_registry> <k8s_namespace> <path_to_dynamo_directory> <dynamo_identifier> [<dynamo_config_file>]

# Example: export DYNAMO_IMAGE=nvcr.io/nvidian/nim-llm-dev/dynamo-base-worker:0.0.1
# Example: ./deploy.sh nvcr.io/nvidian/nim-llm-dev my-namespace ../../../examples/hello_world/ hello_world:Frontend
# Example: ./deploy.sh nvcr.io/nvidian/nim-llm-dev my-namespace ../../../examples/llm graphs.disagg_router:Frontend ../../../examples/llm/configs/disagg_router.yaml
```

This script handles:
1. Building and pushing the Docker image
2. Setting up the Helm values
3. Installing/upgrading the Helm release
4. Configuring the necessary Kubernetes resources
