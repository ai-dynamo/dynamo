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

# 🏠 Minikube Setup Guide

Don't have a Kubernetes cluster? No problem! You can set up a local development environment using Minikube. This guide will walk you through setting up everything you need to run Dynamo Cloud locally.

## Setting Up Minikube

### 1. Install Minikube
First things first! You'll need to install Minikube. Follow the official [Minikube installation guide](https://minikube.sigs.k8s.io/docs/start/) for your operating system.

### 2. Configure GPU Support (Optional)
Planning to use GPU-accelerated workloads? You'll need to configure GPU support in Minikube. Follow the [Minikube GPU guide](https://minikube.sigs.k8s.io/docs/tutorials/nvidia/) to set up NVIDIA GPU support before proceeding.

> [!TIP]
> Make sure to configure GPU support before starting Minikube if you plan to use GPU workloads!

### 3. Start Minikube
Time to launch your local cluster!

```bash
# Start Minikube with GPU support (if configured)
minikube start --driver docker --container-runtime docker --gpus all --memory=16000mb --cpus=8

# Enable required addons
minikube addons enable istio-provisioner
minikube addons enable istio
minikube addons enable storage-provisioner-rancher
```

### 4. Verify Installation
Let's make sure everything is working correctly!

```bash
# Check Minikube status
minikube status

# Verify Istio installation
kubectl get pods -n istio-system

# Verify storage class
kubectl get storageclass
```

## Next Steps

Once your local environment is set up, you can proceed with the [Dynamo Cloud deployment guide](./dynamo_cloud.md) to deploy the platform to your local cluster.

## Coming Soon

- MicroK8s setup guide
- Kind setup guide
- More local development tips and tricks
