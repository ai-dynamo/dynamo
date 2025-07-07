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

Run the following helm command:

```bash
helm install dynamo-platform dynamo-platform-v0.3.2.tgz --namespace ${NAMESPACE}
```

## Install Dynamo Components

```bash
kubectl apply -f examples/llm/deploy/agg.yaml
```

