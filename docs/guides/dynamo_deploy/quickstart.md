# Quickstart

Before deploying your inference graphs you need to install the Dynamo Inference Platform and the Dynamo Cloud.

## 1. Installing from Published Artifacts

Use this approach when installing from pre-built helm charts and docker images published to NGC.

### Prerequisites

```bash
export NAMESPACE=dynamo-cloud
export RELEASE_VERSION=0.3.2
```

Install `envsubst`, `kubectl`, `helm`

### Authenticate with NGC

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia --username='$oauthtoken' --password=<YOUR_NGC_CLI_API_KEY>
```

### Fetch Helm Charts

```bash
# Fetch the CRDs helm chart
helm fetch https://helm.ngc.nvidia.com/nvidia/charts/dynamo-crds-v${RELEASE_VERSION}.tgz

# Fetch the platform helm chart
helm fetch https://helm.ngc.nvidia.com/nvidia/charts/dynamo-platform-v${RELEASE_VERSION}.tgz
```

### Install Dynamo Cloud

**Step 1: Install Custom Resource Definitions (CRDs)**

```bash
helm install dynamo-crds dynamo-crds-v${RELEASE_VERSION}.tgz \
  --namespace default \
  --wait \
  --atomic
```

**Step 2: Install Dynamo Platform**

```bash
kubectl create namespace ${NAMESPACE}

helm install dynamo-platform dynamo-platform-v${RELEASE_VERSION}.tgz --namespace ${NAMESPACE}
```

## 2. Installing from Source

Use this approach when developing or customizing Dynamo as a contributor, or using local helm charts from the source repository.

### Prerequisites

Ensure you have the source code checked out and are in the `dynamo` directory:

```bash
cd deploy/cloud/helm/
```

### Set Environment Variables

```bash
export NAMESPACE=dynamo-cloud
export DOCKER_USERNAME=your-username
export DOCKER_PASSWORD=your-password
export DOCKER_SERVER=your-docker-registry.com
export IMAGE_TAG=your-image-tag
```

The operator image will be pulled from `$DOCKER_SERVER/dynamo-operator:$IMAGE_TAG`.

### Install Dynamo Cloud

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

**Step 1: Install Custom Resource Definitions (CRDs)**

```bash
helm install dynamo-crds ./crds/ \
  --namespace default \
  --wait \
  --atomic
```

**Step 2: Build Dependencies and Install Platform**

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


## Explore Examples

### Hello World

For a basic example that doesn't require a GPU, see the [Hello World](../../examples/hello_world.md)

### LLM

Create a Kubernetes secret containing your sensitive values if needed:

```bash
export HF_TOKEN=your_hf_token
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE}
```


Pick your deployment destination.

If local

```bash
export DYNAMO_CLOUD=http://localhost:8080
```

If kubernetes
```bash
export DYNAMO_CLOUD=https://dynamo-cloud.nvidia.com
```

```bash
# Go to your main dynamo directory.
cd ../../../
kubectl apply -f examples/llm/deploy/agg.yaml -n $NAMESPACE
```

