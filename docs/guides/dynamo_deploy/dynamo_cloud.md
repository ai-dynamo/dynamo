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

# Dynamo Cloud Kubernetes Platform

The Dynamo Cloud platform is a comprehensive solution for deploying and managing Dynamo inference graphs (also referred to as pipelines) in Kubernetes environments. It provides a streamlined experience for deploying, scaling, and monitoring your inference services.

## Overview

The Dynamo cloud platform consists of several key components:

- **Dynamo Operator**: A Kubernetes operator that manages the lifecycle of Dynamo inference graphs from build ➡️ deploy. For more information on the operator, see [Dynamo Kubernetes Operator Documentation](../dynamo_deploy/dynamo_operator.md)
- **Custom Resources**: Kubernetes custom resources for defining and managing Dynamo services


## Deployment Prerequisites

Before getting started with the Dynamo cloud platform, ensure you have:

- A Kubernetes cluster (version 1.24 or later)
- **Make** and standard build tools
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

For a custom setup build and push to your registry Dynamo Base Image for Dynamo inference runtime. This is a one-time operation.

```bash
# Run the script to build the default dynamo:latest-vllm image.
./container/build.sh
export IMAGE_TAG=<TAG>
# Tag the image
docker tag dynamo:latest-vllm <your-registry>/dynamo:${IMAGE_TAG}
docker push <your-registry>/dynamo:${IMAGE_TAG}
```

## 🚀 Deploying the Dynamo Cloud Platform

## Prerequisites

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

## Installation

Follow [Quickstart Guide](./quickstart.md) to install the Dynamo Cloud

⚠️ **Note:** that omitting `--crds` will skip the CRDs installation/upgrade. This is useful when installing on a shared cluster as CRDs are cluster-scoped resources.

⚠️ **Note:** If you'd like to only generate the generated-values.yaml file without deploying to Kubernetes (e.g., for inspection, CI workflows, or dry-run testing), use:

```bash
./deploy_dynamo_cloud.py --yaml-only
```



### Cloud Provider-Specific deployment

#### Google Kubernetes Engine (GKE) deployment

You can find detailed instructions for deployment in GKE [here](../dynamo_deploy/gke_setup.md)

