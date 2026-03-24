# Dynamo Kubernetes Operator

A Kubernetes Operator to manage all Dynamo pipelines using custom resources.


## Overview

This operator automates the deployment and lifecycle management of Dynamo resources in Kubernetes clusters:

- **DynamoGraphDeploymentRequest (DGDR)** - Simplified SLA-driven deployment interface
- **DynamoGraphDeployment (DGD)** - Direct deployment configuration

Built with [Kubebuilder](https://book.kubebuilder.io/), it follows Kubernetes best practices and supports declarative configuration through CustomResourceDefinitions (CRDs).

### Custom Resources

- **DynamoGraphDeploymentRequest**: High-level interface for SLA-driven configuration generation. Automatically handles profiling and generates an optimized DGD spec based on your performance requirements.
- **DynamoGraphDeployment**: Lower-level interface for direct deployment configuration with full control over all parameters.


## Developer guide

### Pre-requisites

- [Go](https://go.dev/doc/install) >= 1.25
- [Kubebuilder](https://book.kubebuilder.io/quick-start.html)

### Build

```
make
```

### Local development with Tilt

[Tilt](https://docs.tilt.dev/install.html) provides a live-reload development loop for the operator. It compiles the Go binary locally, builds a minimal Docker image, renders the production Helm chart, and deploys everything to your cluster. On code changes, Tilt recompiles and live-updates the binary without a full image rebuild — giving fast iteration on controller logic against a real cluster.

#### Prerequisites

The following tools must be installed and available in your `PATH` before running `tilt up`:

| Tool | Version | Purpose | Install |
|------|---------|---------|---------|
| [Go](https://go.dev/doc/install) | ≥ 1.25 | Compiles the manager binary locally | [go.dev/doc/install](https://go.dev/doc/install) |
| [Tilt](https://docs.tilt.dev/install.html) | latest | Live-reload dev loop orchestrator | [docs.tilt.dev/install](https://docs.tilt.dev/install.html) |
| [Helm](https://helm.sh/docs/intro/install/) | v3 | Renders the platform Helm chart | [helm.sh/docs/intro/install](https://helm.sh/docs/intro/install/) |
| [kubectl](https://kubernetes.io/docs/tasks/tools/) | ≥ 1.29 | Applies CRDs and creates the namespace | [kubernetes.io/docs/tasks/tools](https://kubernetes.io/docs/tasks/tools/) |
| [Docker](https://docs.docker.com/get-docker/) | latest | Builds the live-update container image | [docs.docker.com/get-docker](https://docs.docker.com/get-docker/) |

**Conditional prerequisites** (only needed when `skip_codegen: false`, the default):

| Tool | Version | Purpose | Install |
|------|---------|---------|---------|
| [yq](https://github.com/mikefarah/yq) | v4+ | Post-processes generated CRD YAML | `make ensure-yq` or [github.com/mikefarah/yq](https://github.com/mikefarah/yq) |
| [Python 3](https://www.python.org/) + [pydantic](https://docs.pydantic.dev/) | 3.x | Generates Pydantic models from Go types (`make generate`) | `pip install pydantic` |

> **Tip:** Set `skip_codegen: true` in `tilt-settings.yaml` to skip CRD/code generation on every reload. This removes the yq/Python requirement and speeds up iteration when you haven't changed API types.

**Cluster:** You need a Kubernetes cluster (kind, minikube, GKE, EKS, bare-metal, etc.) with a kubeconfig context that Tilt can reach. If your cluster has GPUs and you want to test DGD/DGDR workloads end-to-end, the [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html) should be installed on the cluster.

#### Setup

1. **Create `tilt-settings.yaml`** in `deploy/operator/` with this minimal config:
   ```yaml
   allowed_contexts:
     - h100                 # Change to your Kubernetes context

   registry: docker.io/myuser  # Change to your Docker registry
   ```

2. **Run Tilt**:
   ```bash
   cd deploy/operator
   tilt up
   ```
   The Tilt UI will open at http://localhost:10350 showing resource status and logs.

#### Features

- **Fast iteration**: On code changes, Tilt recompiles the manager binary and live-updates it into the running container — no full image rebuild needed
- **Real cluster testing**: Reconciles against your actual Kubernetes cluster (kind, minikube, GKE, AKS, etc.)
- **CRD + Helm rendering**: Automatically applies CRDs and renders the platform Helm chart with your configuration
- **Infrastructure toggles**: Control NATS, etcd, KAI scheduler, and Grove via `tilt-settings.yaml`

#### Optional configuration

Additional settings available in `tilt-settings.yaml`:

```yaml
# Infrastructure toggles (control which components are deployed)
enable_nats: true              # Enable NATS messaging (default: true, required for DGD/DGDR)
enable_etcd: false             # Enable etcd service discovery (default: false)
enable_kai_scheduler: false    # Enable KAI GPU-aware scheduler (default: false)
enable_grove: false            # Enable Grove orchestrator (default: false)

# Other settings
namespace: dynamo-system       # Kubernetes namespace for operator deployment
skip_codegen: false            # Skip code generation for faster reloads if API unchanged
image_pull_secret: ""          # Name of Secret for private Docker registries
helm_values: {}                # Extra Helm value overrides for platform chart
operator_version: "0.0.0-dev"  # Override operator version (default: from Chart.yaml)
```

### Install

See [Dynamo Kubernetes Platform Installation Guide](/docs/kubernetes/installation-guide.md) for installation instructions.
