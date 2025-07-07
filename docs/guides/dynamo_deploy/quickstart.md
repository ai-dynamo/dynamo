# Quickstart

## Set Environment Variables

```bash
export NAMESPACE=dynamo-cloud
```

## Install Dynamo Cloud

**Step 1: Install Custom Resource Definitions (CRDs)**

```bash
helm install dynamo-crds dynamo-crds-helm-chart.tgz \
  --namespace default \
  --wait \
  --atomic
```

**Step 2: Install Dynamo Platform**

Run the following helm command:

```bash
helm install dynamo-platform dynamo-platform-helm-chart.tgz --namespace ${NAMESPACE}
```

## Install Dynamo Components

```bash
kubectl apply -f examples/llm/deploy/agg.yaml
```

