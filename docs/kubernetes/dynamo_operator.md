# Working with Dynamo Kubernetes Operator

## Overview

Dynamo operator is a Kubernetes operator that simplifies the deployment, configuration, and lifecycle management of DynamoGraphs. It automates the reconciliation of custom resources to ensure your desired state is always achieved. This operator is ideal for users who want to manage complex deployments using declarative YAML definitions and Kubernetes-native tooling.

## Architecture

- **Operator Deployment:**
  Deployed as a Kubernetes `Deployment` in a specific namespace.

- **Controllers:**
  - `DynamoGraphDeploymentController`: Watches `DynamoGraphDeployment` CRs and orchestrates graph deployments.
  - `DynamoComponentDeploymentController`: Watches `DynamoComponentDeployment` CRs and handles individual component deployments.
  - `DynamoModelController`: Watches `DynamoModel` CRs and manages model lifecycle (e.g., loading LoRA adapters).

- **Workflow:**
  1. A custom resource is created by the user or API server.
  2. The corresponding controller detects the change and runs reconciliation.
  3. Kubernetes resources (Deployments, Services, etc.) are created or updated to match the CR spec.
  4. Status fields are updated to reflect the current state.

## Deployment Modes

The Dynamo operator supports three deployment modes to accommodate different cluster environments and use cases:

### 1. Cluster-Wide Mode (Default)

The operator monitors and manages DynamoGraph resources across **all namespaces** in the cluster.

**When to Use:**
- You have full cluster admin access
- You want centralized management of all Dynamo workloads
- Standard production deployment on a dedicated cluster

---

### 2. Namespace-Scoped Mode

The operator monitors and manages DynamoGraph resources **only in a specific namespace**. A lease marker is created to signal the operator's presence to any cluster-wide operators.

**When to Use:**
- You're on a shared/multi-tenant cluster
- You only have namespace-level permissions
- You want to test a new operator version in isolation
- You need to avoid conflicts with other operators

**Installation:**
```bash
helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz \
  --namespace my-namespace \
  --create-namespace \
  --set dynamo-operator.namespaceRestriction.enabled=true
```

---

### 3. Hybrid Mode

A **cluster-wide operator** manages most namespaces, while **one or more namespace-scoped operators** run in specific namespaces (e.g., for testing new versions). The cluster-wide operator automatically detects and excludes namespaces with namespace-scoped operators using lease markers.

**When to Use:**
- Running production workloads with a stable operator version
- Testing new operator versions in isolated namespaces without affecting production
- Gradual rollout of operator updates
- Development/staging environments on production clusters

**How It Works:**
1. Namespace-scoped operator creates a lease named `dynamo-operator-namespace-scope` in its namespace
2. Cluster-wide operator watches for these lease markers across all namespaces
3. Cluster-wide operator automatically excludes any namespace with a lease marker
4. If namespace-scoped operator stops, its lease expires (TTL: 30s by default)
5. Cluster-wide operator automatically resumes managing that namespace

**Setup Example:**

```bash
# 1. Install cluster-wide operator (production, v1.0.0)
helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz \
  --namespace dynamo-system \
  --create-namespace

# 2. Install namespace-scoped operator (testing, v2.0.0-beta)
helm install dynamo-test dynamo-platform-${RELEASE_VERSION}.tgz \
  --namespace test-namespace \
  --create-namespace \
  --set dynamo-operator.namespaceRestriction.enabled=true \
  --set dynamo-operator.controllerManager.manager.image.tag=v2.0.0-beta

**Observability:**

```bash
# List all namespaces with local operators
kubectl get lease -A --field-selector metadata.name=dynamo-operator-namespace-scope

# Check which operator version is running in a namespace
kubectl get lease -n my-namespace dynamo-operator-namespace-scope \
  -o jsonpath='{.spec.holderIdentity}'
```


## Custom Resource Definitions (CRDs)

Dynamo provides the following Custom Resources:

- **DynamoGraphDeployment (DGD)**: Deploys complete inference pipelines
- **DynamoComponentDeployment (DCD)**: Deploys individual components
- **DynamoModel**: Manages model lifecycle (e.g., loading LoRA adapters)

For the complete technical API reference for Dynamo Custom Resource Definitions, see:

**📖 [Dynamo CRD API Reference](./api_reference.md)**

For a user-focused guide on deploying and managing models with DynamoModel, see:

**📖 [Managing Models with DynamoModel Guide](./deployment/dynamomodel-guide.md)**

## Webhooks

The Dynamo Operator uses **Kubernetes admission webhooks** for real-time validation of custom resources before they are persisted to the cluster. Webhooks are **enabled by default** and ensure that invalid configurations are rejected immediately at the API server level.

**Key Features:**
- ✅ Shared certificate infrastructure across all webhook types
- ✅ Automatic certificate generation (for testing/development)
- ✅ cert-manager integration (for production)
- ✅ Multi-operator support with lease-based coordination
- ✅ Immutability enforcement for critical fields

For complete documentation on webhooks, certificate management, and troubleshooting, see:

**📖 [Webhooks Guide](./webhooks.md)**

## Observability

The Dynamo Operator provides comprehensive observability through Prometheus metrics and Grafana dashboards. This allows you to monitor:

- **Controller Performance**: Reconciliation loop duration, success rates, and error rates by resource type
- **Webhook Activity**: Validation performance, admission rates, and denial patterns
- **Resource Inventory**: Current count of managed resources by state and namespace
- **Operational Health**: Success rates and health indicators for controllers and webhooks

### Metrics Collection

Metrics are automatically exposed on the operator's `/metrics` endpoint (port 8443 by default) and collected by Prometheus via a ServiceMonitor. The ServiceMonitor is automatically created when you install the operator via Helm (controlled by `metricsService.enabled`, which defaults to `true`).

### Grafana Dashboard

A pre-built Grafana dashboard is available for visualizing operator metrics. The dashboard includes:

- **Reconciliation Metrics**: Rate, duration (P95), and errors by resource type
- **Webhook Metrics**: Request rate, duration (P95), and denials by resource type and operation
- **Resource Inventory**: Count of DynamoGraphDeployments by state and namespace
- **Operational Health**: Success rate gauges for controllers and webhooks

For complete setup instructions and metrics reference, see:

**📖 [Operator Metrics Guide](./observability/operator-metrics.md)**

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

> **Note:** For shared/multi-tenant clusters or testing scenarios, see [Deployment Modes](#deployment-modes) above for namespace-scoped and hybrid configurations.

### Building from Source

```bash
# Set environment
export NAMESPACE=dynamo-system
export DOCKER_SERVER=your-registry.com/  # your container registry
export IMAGE_TAG=latest

# Build operator image
cd deploy/operator
docker build -t $DOCKER_SERVER/dynamo-operator:$IMAGE_TAG .
docker push $DOCKER_SERVER/dynamo-operator:$IMAGE_TAG
cd -

# Install CRDs
cd deploy/helm/charts
helm install dynamo-crds ./crds/ --namespace default

# Install platform with custom operator image
helm install dynamo-platform ./platform/ \
  --namespace ${NAMESPACE} \
  --create-namespace \
  --set "dynamo-operator.controllerManager.manager.image.repository=${DOCKER_SERVER}/dynamo-operator" \
  --set "dynamo-operator.controllerManager.manager.image.tag=${IMAGE_TAG}" \
  --set etcd.enabled=false \
  --set dynamo-operator.imagePullSecrets[0].name=docker-imagepullsecret
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
