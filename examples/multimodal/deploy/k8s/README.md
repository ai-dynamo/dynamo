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

# Deploying Multimodal Examples on Kubernetes

This guide will help you quickly deploy and clean up the multimodal example services in Kubernetes.

## Prerequisites

- **Dynamo Cloud** is already deployed in your target Kubernetes namespace.
- You have `kubectl` access to your cluster and the correct namespace set in `$NAMESPACE`.


## Create a secret with huggingface token

```bash
export HF_TOKEN="huggingfacehub token with read permission to models"
kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN=$HF_TOKEN -n $KUBE_NS || true
```

---

Choose the example you want to deploy or delete. The YAML files are located in `examples/multimodal/deploy/k8s/`.

## Deploy the Multimodal Example

```bash
kubectl apply -f examples/multimodal/deploy/k8s/<Example yaml file> -n $NAMESPACE
```

## Uninstall the Multimodal Example


```bash
kubectl delete -f examples/multimodal/deploy/k8s/<Example yaml file> -n $NAMESPACE
```