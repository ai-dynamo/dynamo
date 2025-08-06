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

Prior to deploying an inference graph the user should deploy the Dynamo Cloud Platform. Reference the [Quickstart Guide](quickstart.md) for steps to install Dynamo Cloud with Helm.

Dynamo Cloud acts as an orchestration layer between the end user and Kubernetes, handling the complexity of deploying your graphs for you. This is a one-time action, only necessary the first time you deploy a DynamoGraph.

# 2. Deploy your inference graph.

We provide a Custom Resource YAML file for many examples under the components/backends/{engine}/deploy folders. Consult the examples below for the CRs for a specific inference backend.

[View SGLang K8s](../../../components/backends/sglang/deploy/README.md)

[View vLLM K8s](../../../components/backends/vllm/deploy/README.md)

[View TRT-LLM K8s](../../../components/backends/trtllm/deploy/README.md)

### Deploying a particular example

```bash
# Set your dynamo root directory
cd <root-dynamo-folder>
export PROJECT_ROOT=$(pwd)
export NAMESPACE=<your-namespace> # the namespace you used to deploy Dynamo cloud to.
```

Deploying an example consists of the simple `kubectl apply -f ... -n ${NAMESPACE}` command. For example:

```bash
kubectl apply -f components/backends/vllm/deploy/agg.yaml -n ${NAMESPACE}
```

You can use `kubectl get dynamoGraphDeployment -n ${NAMESPACE}` to view your deployment.
You can use `kubectl delete dynamoGraphDeployment <your-dep-name> -n ${NAMESPACE}` to delete the deployment.

We provide a Custom Resource YAML file for many examples under the `deploy/` folder.
Use [VLLM YAML](../../components/backends/vllm/deploy/agg.yaml) for an example.

**Note 1** Example Image

The examples use a prebuilt image from the `nvcr.io` registry.
You can utilize public images from [Dynamo NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo) or build your own image and update the image location in your CR file prior to applying. Either way, you will need to overwrite the image in the example YAML.

To build your own image:

```bash
./container/build.sh --framework <your-inference-framework>
```

For example for the `sglang` run
```bash
./container/build.sh --framework sglang
```

To overwrite the image in the example:

```bash
extraPodSpec:
        mainContainer:
          image: <image-in-your-$DYNAMO_IMAGE>
```

**Note 2**
Setup port forward if needed when deploying to Kubernetes.

List the services in your namespace:

```bash
kubectl get svc -n ${NAMESPACE}
```
Look for one that ends in `-frontend` and use it for port forward.

```bash
SERVICE_NAME=$(kubectl get svc -n ${NAMESPACE} -o name | grep frontend | sed 's|.*/||' | sed 's|-frontend||' | head -n1)
kubectl port-forward svc/${SERVICE_NAME}-frontend 8080:8080 -n ${NAMESPACE}
```

Additional Resources:
- [Port Forward Documentation](https://kubernetes.io/docs/tasks/access-application-cluster/port-forward-access-application-cluster/)
- [Examples Deployment Guide](../../examples/README.md#deploying-a-particular-example)

