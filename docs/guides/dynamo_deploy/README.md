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

# 1. Install Dynamo Cloud.

Prior to deploying an inference graph the user should deploy the Dynamo Cloud Platform. Dynamo Cloud acts as an orchestration layer between the end user and Kubernetes, handling the complexity of deploying your graphs for you. This is a one-time action, only necessary the first time you deploy a DynamoGraph.


Please the [Quickstart Guide](quickstart.md) for steps to install Dynamo Cloud.

# 2. Deploy your inference graph.

You can deploy your graphs using the same command:

```bash
kubectl apply -f CRD.yaml
```

Consult the examples below for the CRD mathcin
vLLM K8s (hyperlink to /deploy)

SGLang k8s (hyperlink to /deploy)

TRTLLM k8s (hyperlink to /deploy)
