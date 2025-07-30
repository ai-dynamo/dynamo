# Quickstart

Your onboarding includes 2 steps.
1. Before deploying your inference graphs you need to install the Dynamo Inference Platform and the Dynamo Cloud.
Dynamo Cloud acts as an orchestration layer between the end user and Kubernetes, handling the complexity of deploying your graphs for you.
You could install from [Published Artifacts](#1-installing-dynamo-cloud-from-published-artifacts) or [Source](#2-installing-dynamo-cloud-from-source)
2. Once you install the Dynamo Cloud, proceed to the [Examples](../../examples/README.md) to deploy an inference graph.

## 1. Installing Dynamo Cloud from Published Artifacts

Use this approach when installing from pre-built helm charts and docker images published to NGC.

### Prerequisites

```bash
export NAMESPACE=dynamo-cloud
export RELEASE_VERSION=0.3.2
```

Install `envsubst`, `kubectl`, `helm`

### Authenticate with NGC

Go to  https://ngc.nvidia.com/org to get your NGC_CLI_API_KEY.

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia --username='$oauthtoken' --password=<YOUR_NGC_CLI_API_KEY>
```

### Fetch Helm Charts

```bash
# Fetch the CRDs helm chart
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-crds-${RELEASE_VERSION}.tgz

# Fetch the platform helm chart
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${RELEASE_VERSION}.tgz
```

### Install Dynamo Cloud

**Step 1: Install Custom Resource Definitions (CRDs)**

```bash
helm install dynamo-crds dynamo-crds-${RELEASE_VERSION}.tgz \
  --namespace default \
  --wait \
  --atomic
```

**Step 2: Install Dynamo Platform**

```bash
kubectl create namespace ${NAMESPACE}

helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz --namespace ${NAMESPACE}
```

## 2. Installing Dynamo Cloud from Source

Use this approach when developing or customizing Dynamo as a contributor, or using local helm charts from the source repository.

### Prerequisites

Ensure you have the source code checked out and are in the `dynamo` directory:


### Set Environment Variables

Our examples use the [`nvcr.io`](https://nvcr.io/nvidia/ai-dynamo/) but you can setup your own values if you use another docker registry.

```bash
export NAMESPACE=dynamo-cloud # or whatever you prefer.
export DOCKER_SERVER=nvcr.io/nvidia/ai-dynamo/  # your-docker-registry.com
export DOCKER_USERNAME='$oauthtoken'  # your-username if not using nvcr.io
export DOCKER_PASSWORD=YOUR_NGC_CLI_API_KEY  # your-password if not using nvcr.io
```

### Pick the Dynamo Inference Image

Export the tag of the Dynamo Runtime Image.
If you are using a pre-defined release:

```bash
export IMAGE_TAG=RELEASE_VERSION # i.e. 0.3.2 - the release you are using
```

Or build your own image first and tag it with IMAGE_TAG

```bash
export IMAGE_TAG=<your-pick>
./container/build.sh
docker tag dynamo:latest-vllm <your-registry>/dynamo-base:$IMAGE_TAG
docker login <your-registry>
docker push <your-registry>/dynamo-base:$IMAGE_TAG
```

### Install Dynamo Cloud

You need to build and push the Dynamo Cloud Operator Image by running

```bash
cd deploy/cloud/operator
earthly --push +docker --DOCKER_SERVER=$DOCKER_SERVER --IMAGE_TAG=$IMAGE_TAG
```

The  Nvidia Cloud Operator image will be pulled from the `$DOCKER_SERVER/dynamo-operator:$IMAGE_TAG`.

You could run the `deploy.sh` or use the manual commands under Step 1 and Step 2.

**Installing with a script (alternative to the Step 1 and Step 2)**

Create the namespace and the docker registry secret.

```bash
kubectl create namespace ${NAMESPACE}
kubectl create secret docker-registry docker-imagepullsecret \
  --docker-server=${DOCKER_SERVER} \
  --docker-username=${DOCKER_USERNAME} \
  --docker-password=${DOCKER_PASSWORD} \
  --namespace=${NAMESPACE}
```

You need to add the bitnami helm repository by running:

```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
```

```bash
./deploy.sh --crds
```

if you want guidance during the process, run the deployment script with the `--interactive` flag:

```bash
./deploy.sh --crds --interactive
```

**Installing CRDs manually  (alternative to the script deploy.sh)**

***Step 1: Install Custom Resource Definitions (CRDs)**

```bash
dynamo/deploy/cloud/helm

helm install dynamo-crds ./crds/ \
  --namespace default \
  --wait \
  --atomic
```

***Step 2: Build Dependencies and Install Platform**

```bash
helm dep build ./platform/

kubectl create namespace ${NAMESPACE}

# Create docker registry secret
kubectl create secret docker-registry docker-imagepullsecret \
  --docker-server=${DOCKER_SERVER} \
  --docker-username=${DOCKER_USERNAME} \
  --docker-password=${DOCKER_PASSWORD} \
  --namespace=${NAMESPACE}

# Install platform
helm install dynamo-platform ./platform/ \
  --namespace ${NAMESPACE} \
  --set "dynamo-operator.controllerManager.manager.image.repository=${DOCKER_SERVER}/dynamo-operator" \
  --set "dynamo-operator.controllerManager.manager.image.tag=${IMAGE_TAG}" \
  --set "dynamo-operator.imagePullSecrets[0].name=docker-imagepullsecret"
```

[More on Deploying to Dynamo Cloud](./dynamo_cloud.md)

## Uninstall CRDs for a clean start

We provide a script to uninstall CRDs should you need a clean start.

```bash
./uninstall.sh
```

## Explore Examples

If deploying to Kubernetes, create a Kubernetes secret containing your sensitive values if needed:

```bash
export HF_TOKEN=your_hf_token
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE}
```

Follow the [Examples](../../examples/README.md)
For more details on how to create your own deployments follow [Create Deployment Guide](create_deployment.md)
