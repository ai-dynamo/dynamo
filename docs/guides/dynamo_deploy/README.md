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

[Deploying a particular example](../../examples/README.md#deploying-a-particular-example)

Set your dynamo root directory
cd
export PROJECT_ROOT=$(pwd)
export NAMESPACE= # the namespace you used to deploy Dynamo cloud to.
Deploying an example consists of the simple kubectl apply -f ... -n ${NAMESPACE} command. For example:

kubectl apply -f components/backends/vllm/deploy/agg.yaml -n ${NAMESPACE}
You can use kubectl get dynamoGraphDeployment -n ${NAMESPACE} to view your deployment. You can use kubectl delete dynamoGraphDeployment -n ${NAMESPACE} to delete the deployment.

We provide a Custom Resource YAML file for many examples under the components/backends/{engine}/deploy folders. Consult the examples below for the CRs for a specific inference backend.

[View SGLang K8s](../../components/backends/sglang/deploy/README.md)

[View vLLM K8s](../../components/backends/vllm/deploy/README.md)

[View TRT-LLM K8s](../../components/backends/trtllm/deploy/README.md)

The examples use a prebuilt image from the nvcr.io registry. You can point to the public images on [Dynamo NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo) or build your own image and update the image location in your CR YAML prior to applying. You could build your own image using

./container/build.sh --framework
For example for the sglang run

./container/build.sh --framework sglang
Then you would need to overwrite the image in the examples.

extraPodSpec:
mainContainer:
image: <image-in-your-$DYNAMO_IMAGE>
Note 2 Setup port forward if needed when deploying to Kubernetes.

List the services in your namespace:

kubectl get svc -n ${NAMESPACE}
Look for one that ends in -frontend and use it for port forward.

SERVICE_NAME=$(kubectl get svc -n ${NAMESPACE} -o name | grep frontend | sed 's|.*/||' | sed 's|-frontend||' | head -n1)
kubectl port-forward svc/${SERVICE_NAME}-frontend 8080:8080 -n ${NAMESPACE}
Consult the Port Forward Documentation
