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

# Dynamo Deployment Guide

This directory contains all the necessary files and instructions for deploying Dynamo in various environments. Choose the deployment method that best suits your needs:

## Directory Structure

```
deploy/
├── cloud/                    # Cloud deployment configurations and tools
├── helm/                     # Helm charts for manual Kubernetes deployment
├── metrics/                  # Monitoring and metrics configuration
├── sdk/                      # Dynamo SDK and related tools
└── README.md                 # This file
```

## Deployment Options

### 1. 🚀 Dynamo Cloud Platform [PREFERRED]

The Dynamo Cloud Platform provides a managed deployment experience with:
- Automated infrastructure management
- Built-in monitoring and metrics
- Simplified deployment process via `dynamo deploy` CLI commands
- Production-ready configurations
- Managed NATS and etcd dependencies

For detailed instructions, see:
- [Dynamo Cloud Platform Guide](../docs/guides/dynamo_deploy/dynamo_cloud.md)
- [Operator Deployment Guide](../docs/guides/dynamo_deploy/operator_deployment.md)

### 2. Manual Deployment with Helm Charts

For users who need more control over their deployments:
- Full control over deployment parameters
- Manual management of infrastructure
- Customizable monitoring setup
- Flexible configuration options
- Manual management of NATS and etcd dependencies

Documentation:
- [Manual Helm Deployment Guide](../docs/guides/dynamo_deploy/manual_helm_deployment.md)
- [Minikube Setup Guide](../docs/guides/dynamo_deploy/minikube.md)

## Choosing the Right Deployment Method

- **Dynamo Cloud Platform**: Best for most users, provides managed deployment with built-in monitoring
  - See [Dynamo Cloud Platform Guide](../docs/guides/dynamo_deploy/dynamo_cloud.md)
  - Recommended for production deployments
  - Simplifies dependency management
  - Provides infrastructure for user management

- **Manual Helm Deployment**: For users who need full control over their deployment
  - See [Manual Helm Deployment Guide](../docs/guides/dynamo_deploy/manual_helm_deployment.md)
  - Suitable for custom deployments
  - Requires manual management of dependencies
  - Provides maximum flexibility for users

## Example Deployments

To help you get started, we provide several example deployments:

### Hello World Example
A basic example to learn Dynamo deployment: [Hello World Example](../examples/hello_world/README.md#deploying-to-and-running-the-example-in-kubernetes)
- Shows how to deploy a simple three-service pipeline that processes text
- Provides step-by-step instructions for building your service and testing with port forwarding
- Includes sample output showing the text flow between services

### LLM Examples
Example for deploying LLM services: [LLM Example](../examples/llm/README.md#deploy-to-kubernetes)
- Demonstrates deploying and making inference requests against LLM models
- Includes examples for both aggregated and disaggregated serving
- Provides detailed deployment steps and testing instructions
