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

# Deploying Dynamo inference graphs to Kubernetes

## Deployment Paths in Dynamo

Dynamo provides two distinct deployment paths, each serving different purposes:

1. **Dynamo Cloud Platform** (`deploy/dynamo/helm/`)
   - Contains the infrastructure components required for the Dynamo cloud platform
   - Used when deploying with the `dynamo deploy` CLI commands
   - Provides a managed deployment experience
   - This README focuses on setting up this platform infrastructure
   - For Dynamo cloud installation instructions, see [Installing Dynamo Cloud](./helm/README.md), which walks through installing and configuring the Dynamo cloud components on your Kubernetes cluster.

2. **Manual Deployment with Helm Charts** (`deploy/Kubernetes/`)
   - Used for manually deploying inference graphs to Kubernetes
   - Contains Helm charts and configurations for deploying individual inference pipelines
   - Documentation:
        - [Deploying Dynamo Inference Graphs to Kubernetes using Helm](../Kubernetes/pipeline/README.md)
        - [Dynamo Deploy Guide](../../docs/guides/dynamo_deploy.md)

Choose the appropriate deployment path based on your needs:
- Use `deploy/Kubernetes/` if you want to manually manage your inference graph deployments
- Use `deploy/dynamo/helm/` if you want to use the Dynamo cloud platform and CLI tools

## Hello World example
See [examples/hello_world/README.md#deploying-to-kubernetes-using-dynamo-cloud-and-dynamo-deploy-cli](../../examples/hello_world/README.md#deploying-to-kubernetes-using-dynamo-cloud-and-dynamo-deploy-cli)