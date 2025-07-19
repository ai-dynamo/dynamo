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

# Deploying Inference Graphs to Kubernetes

We expect users to deploy their inference graphs using CRDs or helm charts.

Prior to deploying an inference graph the user should deploy the Dynamo Cloud Platform.
Dynamo Cloud acts as an orchestration layer between the end user and Kubernetes, handling the complexity of deploying your graphs for you. This is a one-time action, only necessary the first time you deploy a DynamoGraph.


# 1. Please follow [Installing Dynamo Cloud](./dynamo_cloud.md) for steps to install.
For details about the Dynamo Cloud Platform, see the [Dynamo Operator Guide](dynamo_operator.md)

# 2. Follow [Examples](../../examples/README.md) to see how you can deploy your Inference Graphs.


## Manual Deployment with Helm Charts

Users who need more control over their deployments can use the manual deployment path (`deploy/helm/`):

- Used for manually deploying inference graphs to Kubernetes
- Contains Helm charts and configurations for deploying individual inference pipelines
- Provides full control over deployment parameters
- Requires manual management of infrastructure components
- Documentation:
  - [Using the Deployment Script](manual_helm_deployment.md#using-the-deployment-script): all-in-one script for manual deployment
  - [Helm Deployment Guide](manual_helm_deployment.md#helm-deployment-guide): detailed instructions for manual deployment
