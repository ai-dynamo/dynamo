# Quickstart


## Set Environment Variables

```bash
export NAMESPACE=dynamo-cloud

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

Run the following helm command to install from published docker images:

```bash
helm install dynamo-platform dynamo-platform-v0.3.2.tgz --namespace ${NAMESPACE}
```

Run the following helm commands to install from custom docker images:

```bash
# create a secret for the docker registry (you can reuse an existing secret if you have one)
kubectl create secret docker-registry <docker-secret-name> \
  --docker-server=<docker-registry> \
  --docker-username=<docker-username> \
  --docker-password=<docker-password> \
  --namespace=${NAMESPACE}

helm install dynamo-platform dynamo-platform-v0.3.2.tgz --namespace ${NAMESPACE} --set "dynamo-operator.controllerManager.manager.image.repository=<docker-registry>/<image-name>" --set "dynamo-operator.controllerManager.manager.image.tag=<image-tag>" --set "dynamo-operator.imagePullSecrets[0].name=<docker-secret-name>"
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

