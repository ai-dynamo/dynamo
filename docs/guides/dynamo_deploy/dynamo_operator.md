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

**📖 [Dynamo CRD API Reference](../../../deploy/cloud/operator/docs/api_reference.md)**

## Installation

[See installation steps](dynamo_cloud.md#overview)


## GitOps Deployment with FluxCD

This section describes how to use FluxCD for GitOps-based deployment of Dynamo inference graphs. GitOps enables you to manage your Dynamo deployments declaratively using Git as the source of truth. We'll use the [aggregated vLLM example](../../../components/backends/vllm/README.md) to demonstrate the workflow.

### Prerequisites

- A Kubernetes cluster with [Dynamo Cloud](dynamo_cloud.md) installed
- [FluxCD](https://fluxcd.io/flux/installation/) installed in your cluster
- A Git repository to store your deployment configurations

### Workflow Overview

The GitOps workflow for Dynamo deployments consists of three main steps:

1. Build and push the Dynamo Operator
2. Create and commit a DynamoGraphDeployment custom resource for initial deployment
3. Update the graph by building a new version and updating the CR for subsequent updates

### Step 1: Build and Push Dynamo Cloud Operator

First, follow to [See Install Dynamo Cloud](README.md).

### Step 2: Create Initial Deployment

Create a new file in your Git repository (e.g., `deployments/llm-agg.yaml`) with the following content:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: llm-agg
spec:
  services:
    Frontend:
      replicas: 1
      envs:
      - name: SPECIFIC_ENV_VAR
        value: some_specific_value
    Processor:
      replicas: 1
      envs:
      - name: SPECIFIC_ENV_VAR
        value: some_specific_value
    VllmWorker:
      replicas: 1
      envs:
      - name: SPECIFIC_ENV_VAR
        value: some_specific_value
      # Add PVC for model storage
      pvc:
        name: vllm-model-storage
        mountPath: /models
        size: 100Gi
```

Commit and push this file to your Git repository. FluxCD will detect the new CR and create the initial deployment in your cluster. The operator will:
- Create the specified PVCs
- Build container images for all components
- Deploy the services with the configured resources

### Step 3: Update Existing Deployment

To update your pipeline, just update the associated DynamoGraphDeployment CRD

The Dynamo operator will automatically reconcile it.

### Monitoring the Deployment

You can monitor the deployment status using:

```bash

export NAMESPACE=<namespace-with-the-dynamo-cloud-operator>

# Check the DynamoGraphDeployment status
kubectl get dynamographdeployment llm-agg -n $NAMESPACE
```

## Configuration


- **Environment Variables:**

| Name                                               | Description                          | Default                                                |
|----------------------------------------------------|--------------------------------------|--------------------------------------------------------|
| `LOG_LEVEL`                                        | Logging verbosity level              | `info`                                                 |
| `DYNAMO_SYSTEM_NAMESPACE`                          | System namespace                     | `dynamo`                                               |

- **Flags:**
  | Flag                  | Description                                | Default |
  |-----------------------|--------------------------------------------|---------|
  | `--natsAddr`          | Address of NATS server                     | ""      |
  | `--etcdAddr`          | Address of etcd server                     | ""      |



## Troubleshooting

| Symptom                | Possible Cause                | Solution                          |
|------------------------|-------------------------------|-----------------------------------|
| Resource not created   | RBAC missing                  | Ensure correct ClusterRole/Binding|
| Status not updated     | CRD schema mismatch           | Regenerate CRDs with kubebuilder  |
| Image build hangs      | Misconfigured DynamoComponent | Check image build logs            |


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
