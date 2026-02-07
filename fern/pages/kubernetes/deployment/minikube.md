---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# Minikube Setup Guide

Don't have a Kubernetes cluster? No problem! You can set up a local development environment using Minikube. This guide walks through the set up of everything you need to run Dynamo Kubernetes Platform locally.

## 1. Install Minikube
First things first! Start by installing Minikube. Follow the official [Minikube installation guide](https://minikube.sigs.k8s.io/docs/start/) for your operating system.

## 2. Configure GPU Support (Optional)
Planning to use GPU-accelerated workloads? You'll need to configure GPU support in Minikube. Follow the [Minikube GPU guide](https://minikube.sigs.k8s.io/docs/tutorials/nvidia/) to set up NVIDIA GPU support before proceeding.

> [!TIP]
> Make sure to configure GPU support before starting Minikube if you plan to use GPU workloads!


## 3. Start Minikube
Time to launch your local cluster!

```bash
# Start Minikube with GPU support (if configured)
minikube start --driver docker --container-runtime docker --gpus all --memory=16000mb --cpus=8

# Enable required addons
minikube addons enable istio-provisioner
minikube addons enable istio
minikube addons enable storage-provisioner-rancher
```

## 4. Verify Installation
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

Once your local environment is set up, you can proceed with the [Dynamo Kubernetes Platform installation guide](../installation-guide.md) to deploy the platform to your local cluster.

