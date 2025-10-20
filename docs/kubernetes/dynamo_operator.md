# Working with Dynamo Kubernetes Operator

## Overview

Dynamo operator is a Kubernetes operator that simplifies the deployment, configuration, and lifecycle management of DynamoGraphs. It automates the reconciliation of custom resources to ensure your desired state is always achieved. This operator is ideal for users who want to manage complex deployments using declarative YAML definitions and Kubernetes-native tooling.

## Architecture

- **Operator Deployment:**
  Deployed as a Kubernetes `Deployment` in a specific namespace.

- **Controllers:**
  - `DynamoGraphDeploymentController`: Watches `DynamoGraphDeployment` CRs and orchestrates graph deployments.
  - `DynamoComponentDeploymentController`: Watches `DynamoComponentDeployment` CRs and handles individual component deployments.

- **Workflow:**
  1. A custom resource is created by the user or API server.
  2. The corresponding controller detects the change and runs reconciliation.
  3. Kubernetes resources (Deployments, Services, etc.) are created or updated to match the CR spec.
  4. Status fields are updated to reflect the current state.

## Custom Resource Definitions (CRDs)

For the complete technical API reference for Dynamo Custom Resource Definitions, see:

**📖 [Dynamo CRD API Reference](./api_reference.md)**

## Installation

### Quick Install with Helm

```bash
# Set environment
export NAMESPACE=dynamo-system
export RELEASE_VERSION=0.x.x # any version of Dynamo 0.3.2+ listed at https://github.com/ai-dynamo/dynamo/releases

# Install Platform (includes operator)
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${RELEASE_VERSION}.tgz
helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz --namespace ${NAMESPACE} --create-namespace
```

For namespace-restricted installations (shared clusters):
```bash
helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz \
  --namespace ${NAMESPACE} \
  --create-namespace \
  --set dynamo-operator.namespaceRestriction.enabled=true
```

### Building from Source

```bash
# Set environment
export NAMESPACE=dynamo-system
export DOCKER_SERVER=your-registry.com/  # your container registry
export IMAGE_TAG=latest

# Build operator image
cd deploy/cloud/operator
docker build -t $DOCKER_SERVER/dynamo-operator:$IMAGE_TAG .
docker push $DOCKER_SERVER/dynamo-operator:$IMAGE_TAG
cd -

# Install CRDs
cd deploy/cloud/helm
helm install dynamo-crds ./crds/ --namespace default

# Install platform with custom operator image
helm install dynamo-platform ./platform/ \
  --namespace ${NAMESPACE} \
  --create-namespace \
  --set "dynamo-operator.controllerManager.manager.image.repository=${DOCKER_SERVER}/dynamo-operator" \
  --set "dynamo-operator.controllerManager.manager.image.tag=${IMAGE_TAG}"
```

For detailed installation options, see the [Installation Guide](./installation_guide.md)


## Development

- **Code Structure:**

The operator is built using Kubebuilder and the operator-sdk, with the following structure:

- `controllers/`: Reconciliation logic
- `api/v1alpha1/`: CRD types
- `config/`: Manifests and Helm charts


## References

- [Kubernetes Operator Pattern](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/)
- [Custom Resource Definitions](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/)
- [Operator SDK](https://sdk.operatorframework.io/)
- [Helm Best Practices for CRDs](https://helm.sh/docs/chart_best_practices/custom_resource_definitions/)
