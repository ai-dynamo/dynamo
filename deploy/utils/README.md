# Kubernetes utilities for Dynamo Benchmarking and Profiling

This directory contains utilities and manifests for Dynamo benchmarking and profiling workflows.

## Prerequisites

**Before using these utilities, you must first set up Dynamo Cloud following the main installation guide:**

👉 **[Follow the Dynamo Cloud installation guide](/docs/kubernetes/installation_guide.md) to install the Dynamo Kubernetes Platform first.**

This includes:
1. Installing the Dynamo CRDs
2. Installing the Dynamo Platform (operator, etcd, NATS)
3. Setting up your target namespace

## Contents

- `setup_benchmarking_resources.sh` — Sets up benchmarking and profiling resources in your existing Dynamo namespace
- `manifests/`
  - `pvc.yaml` — PVC `dynamo-pvc` for storing profiler results and configurations
  - `pvc-access-pod.yaml` — short‑lived pod for copying profiler results from the PVC
- `kubernetes.py` — helper used by tooling to apply/read resources (e.g., access pod for PVC access)
- `dynamo_deployment.py` — utilities for working with DynamoGraphDeployment resources
- `requirements.txt` — Python dependencies for benchmarking utilities

## Quick start

### Benchmarking Resource Setup

After setting up Dynamo Cloud, use this script to prepare your namespace with the additional resources needed for benchmarking and profiling workflows:

The setup script creates a `dynamo-pvc` with `ReadWriteMany` (RWX). If your cluster's default `storageClassName` does not support RWX, set `storageClassName` in `deploy/utils/manifests/pvc.yaml` to an RWX-capable class before running the script.

Example (add under `spec` in `deploy/utils/manifests/pvc.yaml`):
```yaml
...
spec:
  accessModes:
  - ReadWriteMany
  storageClassName: <your-rwx-storageclass>
...
```

> [!TIP]
> **Check your clusters storage classes**
>
> - List storage classes and provisioners:
> ```bash
> kubectl get sc -o wide
> ```

```bash
export NAMESPACE=your-dynamo-namespace
export HF_TOKEN=<HF_TOKEN>  # Optional: for HuggingFace model access

deploy/utils/setup_benchmarking_resources.sh
```

This script applies the following manifests to your existing Dynamo namespace:

- `deploy/utils/manifests/pvc.yaml` - PVC `dynamo-pvc`

If `HF_TOKEN` is provided, it also creates a secret for HuggingFace model access.

After running the setup script, verify the resources by checking:

```bash
kubectl get pvc dynamo-pvc -n $NAMESPACE
```

### Working with the PVC

The Persistent Volume Claim (PVC) stores configuration files and benchmark/profiling results. Use `kubectl cp` to copy files to and from the PVC.

#### Setting Up PVC Access

First, create a temporary access pod to interact with the PVC:

```bash
# Create access pod
kubectl apply -f deploy/utils/manifests/pvc-access-pod.yaml -n $NAMESPACE

# Wait for pod to be ready
kubectl wait --for=condition=Ready pod/pvc-access-pod -n $NAMESPACE --timeout=60s
```

#### Copying Files to the PVC

**Copy deployment configurations for profiling:**

```bash
# Copy a single file
kubectl cp ./my-disagg.yaml $NAMESPACE/pvc-access-pod:/data/configs/disagg.yaml

# Copy an entire directory
kubectl cp ./configs/ $NAMESPACE/pvc-access-pod:/data/configs/
```

#### Downloading Files from the PVC

**Download benchmark results:**

```bash
# Download entire results directory
kubectl cp $NAMESPACE/pvc-access-pod:/data/results ./benchmarks/results

# Download a specific subdirectory
kubectl cp $NAMESPACE/pvc-access-pod:/data/results/benchmark-name ./benchmarks/results/benchmark-name
```

**Download profiling results (optional, for local inspection):**

```bash
# Optional: Download profiling data for local analysis
# The planner reads directly from the PVC, so this is only needed for inspection
kubectl cp $NAMESPACE/pvc-access-pod:/data ./profiling_data
```

> **Note on Profiling Results**: When using DGDR (DynamoGraphDeploymentRequest) for SLA-driven profiling, profiling data is stored in `/data/` on the PVC. The planner component reads this data directly from the PVC, so downloading is **optional** - only needed if you want to inspect the profiling results locally (e.g., view performance plots, check configurations).

#### Cleanup Access Pod

When finished, delete the access pod:

```bash
kubectl delete pod pvc-access-pod -n $NAMESPACE
```

#### Path Structure

**Common path patterns in the PVC:**
- `/data/configs/` - Configuration files (DGD manifests)
- `/data/results/` - Benchmark results (for download after benchmarking jobs)
- `/data/` - Profiling data (used directly by planner, typically not downloaded)
- `/data/benchmarking/` - Benchmarking artifacts

#### Next Steps

For complete benchmarking and profiling workflows:
- **Benchmarking Guide**: See [docs/benchmarks/benchmarking.md](../../docs/benchmarks/benchmarking.md) for comparing DynamoGraphDeployments and external endpoints
- **Pre-Deployment Profiling**: See [docs/benchmarks/sla_driven_profiling.md](../../docs/benchmarks/sla_driven_profiling.md) for optimizing configurations before deployment

## Notes

- This setup is focused on benchmarking and profiling resources only - the main Dynamo platform must be installed separately.
