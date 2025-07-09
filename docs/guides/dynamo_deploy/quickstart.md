# Quickstart

## Set Environment Variables

```bash
export NAMESPACE=dynamo-cloud
```


if the charts are published.

## Install with helm.

### Authenticate with NGC

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia --username='$oauthtoken' --password=<YOUR_NGC_API_KEY>
```

### Fetch helm charts

```bash
# fetch the crds helm chart
helm fetch https://helm.ngc.nvidia.com/nvidia/charts/dynamo-crds-v0.3.2.tgz

# fetch the platform helm chart
helm fetch https://helm.ngc.nvidia.com/nvidia/charts/dynamo-platform-v0.3.2.tgz
```

## Install Dynamo Cloud

**Step 1: Install Custom Resource Definitions (CRDs)**

```bash
helm install dynamo-crds dynamo-crds-v0.3.2.tgz \
  --namespace default \
  --wait \
  --atomic
```

**Step 2: Install Dynamo Platform**

```bash
export NAMESPACE=your-kubernetes-namespace

export DOCKER_USERNAME=your-name
export DOCKER_PASSWORD=your-password
export DOCKER_SERVER=gitlab-master.nvidia.com:5005/aire/microservices/compoundai
export IMAGE_TAG=3a4adda55d8ed0c0d03f280b6644566002bec026
export DOCKER_SECRET_NAME="my-pull-secret"

kubectl create namespace ${NAMESPACE}

kubectl create secret docker-registry ${DOCKER_SECRET_NAME} \
  --docker-server=${DOCKER_SERVER} \
  --docker-username=${DOCKER_USERNAME} \
  --docker-password=${DOCKER_PASSWORD} \
  --namespace=${NAMESPACE}

helm install dynamo-platform ./deploy/cloud/helm/platform   --namespace ${NAMESPACE}   --set "dynamo-operator.controllerManager.manager.image.repository=${DOCKER_SERVER}/dynamo-operator"   --set "dynamo-operator.controllerManager.manager.image.tag=${IMAGE_TAG}"   --set "dynamo-operator.imagePullSecrets[0].name=${DOCKER_SECRET_NAME}"
```


## Explore Examples

### Hello World

For a basic example that doesn't require a GPU, see the [Hello World](../../examples/hello_world.md)

### LLM example
Create a Kubernetes secret containing your sensitive values if needed:

```bash
export HF_TOKEN=your_hf_token
kubectl create secret generic dynamo-env-secrets \
  --from-literal=huggingface.token=$HF_TOKEN \
  --from-literal=another_secret.key=value \
  -n $NAMESPACE
```

```bash
kubectl apply -f examples/llm/deploy/agg.yaml
```

