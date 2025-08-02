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

# 🚀 Deploy Dynamo Cloud to Kubernetes

Dynamo Cloud acts as an orchestration layer between the end user and Kubernetes, handling the complexity of deploying your graphs for you.
Before you can deploy your graphs, you need to deploy the Dynamo Runtime and Dynamo Cloud images. This is a one-time action, only necessary the first time you deploy a DynamoGraph.

[See Dynamo Cloud Guide](../../../docs/guides/dynamo_deploy/dynamo_cloud.md) for advanced cases and details on how to install and use Dynamo Cloud. For a quick start follow the steps below.


## 🏗️ Building Docker images for Dynamo Cloud components

You can build and push Docker images for the Dynamo cloud components to any container registry of your choice.

**Important** Make sure you're logged in to your container registry before pushing images. For example:

```bash
docker login <CONTAINER_REGISTRY>
```

#### 🛠️ Build and push images for the Dynamo Cloud platform components

[One-time Action]
You should build the image(s) for the Dynamo Cloud Platform.
If you are a **👤 Dynamo User** you would do this step once.

```bash
export DOCKER_SERVER=<your-docker-server>
export IMAGE_TAG=<TAG>
cd deploy/cloud/operator
earthly --push +docker --DOCKER_SERVER=$DOCKER_SERVER --IMAGE_TAG=$IMAGE_TAG
```

If you are a **🧑‍💻 Dynamo Contributor** you would have to rebuild the dynamo platform images as the code evolves. To do so please look at the [Cloud Guide](../../../docs/guides/dynamo_deploy/dynamo_cloud.md).


### 🚀 Deploying the Dynamo Cloud Platform

1. Set the required environment variables:
```bash
export PROJECT_ROOT=$(pwd)
export DOCKER_USERNAME=<your-docker-username>
export DOCKER_PASSWORD=<your-docker-password>
export DOCKER_SERVER=<your-docker-server>
export IMAGE_TAG=<TAG>  # Use the same tag you used when building the images
export NAMESPACE=dynamo-cloud    # change this to whatever you want!
export DYNAMO_INGRESS_SUFFIX=dynamo-cloud.com # change this to whatever you want!
```

2. [One-time Action] Create a new kubernetes namespace and set it as your default. Create image pull secrets if needed.

```bash
cd $PROJECT_ROOT/deploy/cloud/helm
kubectl create namespace $NAMESPACE
kubectl config set-context --current --namespace=$NAMESPACE
```

3. Deploy Dynamo Cloud using the Helm chart via the provided deploy script:
To deploy the Dynamo Cloud Platform on Kubernetes, run:

```bash
./deploy.sh --crds
```

if you want guidance during the process, run the deployment script with the `--interactive` flag:

```bash
./deploy.sh --crds --interactive
```

omitting `--crds` will skip the CRDs installation/upgrade. This is useful when installing on a shared cluster as CRDs are cluster-scoped resources.




