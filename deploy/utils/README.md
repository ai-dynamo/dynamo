# Kubernetes utilities for Dynamo Benchmarking and Profiling

This directory contains utilities and manifests for Dynamo benchmarking and profiling workflows.

## Prerequisites

**Before using these utilities, you must first set up Dynamo Kubernetes Platform following the main installation guide:**

👉 **[Follow the Dynamo Kubernetes Platform installation guide](../../docs/fern/pages/kubernetes/installation/install-dynamo.md) to install the Dynamo Kubernetes Platform first.**

This includes:
1. Installing the Dynamo CRDs
2. Installing the Dynamo Platform (operator, etcd, NATS)
3. Setting up your target namespace

## Contents

- `setup_benchmarking_resources.sh` — Sets up benchmarking and profiling resources in your existing Dynamo namespace
- `manifests/`
  - `pvc.yaml` — PVC `dynamo-pvc`
  - `pvc-access-pod.yaml` — short‑lived pod for copying profiler results from the PVC
- `kubernetes.py` — helper used by tooling to apply/read resources (e.g., access pod for PVC access)
- `dynamo_deployment.py` — utilities for working with DynamoGraphDeployment resources
- `requirements.txt` — Python dependencies for benchmarking utilities

## Quick start

### Benchmarking Resource Setup

After setting up Dynamo Kubernetes Platform, use this script to prepare your namespace with the additional resources needed for benchmarking and profiling workflows:

The setup script creates a `dynamo-pvc` with `ReadWriteOnce` (RWO) access mode using your cluster's default storage class. This is sufficient for profiling workflows where only one job writes at a time.

If you want to use `ReadWriteMany` (RWX) for concurrent access, modify `deploy/utils/manifests/pvc.yaml` before running the script:

```yaml
spec:
  accessModes:
  - ReadWriteMany
  storageClassName: <your-rwx-capable-storageclass>  # e.g., NFS-based storage
  resources:
    requests:
      storage: 50Gi
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

**Inspect profiling results (optional, for local inspection):**

```bash
# View the generated DGD configuration from profiling
kubectl get configmap dgdr-output-<dgdr-name> -n $NAMESPACE -o yaml

# List generated planner profiling data ConfigMaps
kubectl get configmap -n $NAMESPACE -l dgdr.nvidia.com/name=<dgdr-name>
```

> **Note on Profiling Results**: When using DGDR (DynamoGraphDeploymentRequest) for SLA-driven profiling, profiling data is automatically stored in ConfigMaps:
> - `dgdr-output-<dgdr-name>`: Contains the generated DynamoGraphDeployment YAML
> - `planner-profile-data-XXXX`: Contains profiling performance data in JSON format for the planner, with a generated suffix. It is created only for thorough sweeping when profile data is needed.
>
> The planner component reads this data directly from the mounted ConfigMap, so no PVC is needed.

Creating `dynamo-pvc` does not automatically persist profiling artifacts. To
capture plots, logs, and raw data, replace the profiling Job's
`profiling-output` volume with that claim through
`spec.overrides.profilingJob`; see the
[Profiler Guide](../../docs/fern/pages/developer-guide/knowledge-base/modular-components/profiler/profiler-guide.md#accessing-profiling-artifacts).

#### Cleanup Access Pod

When finished, delete the access pod:

```bash
kubectl delete pod pvc-access-pod -n $NAMESPACE
```

#### Path Structure

**Common path patterns in the PVC:**
- `/data/configs/` - Configuration files (DGD manifests)
- `/data/results/` - Benchmark results (for download after benchmarking jobs)
- `/data/benchmarking/` - Benchmarking artifacts

#### Next Steps

For complete benchmarking and profiling workflows:
- **Benchmarking Guide**: See [docs/benchmarks/benchmarking.md](../../docs/fern/pages/recipes/feature-benchmarks/benchmarking-guide.md) for comparing DynamoGraphDeployments and external endpoints
- **Pre-Deployment Profiling**: See [docs/components/profiler/profiler-guide.md](../../docs/fern/pages/developer-guide/knowledge-base/modular-components/profiler/profiler-guide.md) for optimizing configurations before deployment

## Notes

- This setup is focused on benchmarking and profiling resources only - the main Dynamo platform must be installed separately.

## convert_api_version.py — nvidia.com apiVersion converter

Converts Dynamo CRD manifests (DynamoGraphDeployment, DynamoComponentDeployment,
DynamoGraphDeploymentRequest, DynamoGraphDeploymentScalingAdapter) between served
apiVersions by driving the operator's conversion webhook — no conversion logic is
duplicated, so output always matches the operator.

**Requirements:** `kubectl` configured against a cluster running the Dynamo
operator (the conversion webhook must be installed). Multi-document files and
non-nvidia.com documents (passed through unchanged) are supported.

```bash
python deploy/utils/convert_api_version.py -i my-dgd.v1alpha1.yaml -o my-dgd.v1beta1.yaml
# explicit target version:
python deploy/utils/convert_api_version.py -i in.yaml -o out.yaml --to nvidia.com/v1beta1
```

The converter POSTs an `apiextensions.k8s.io/v1` ConversionReview to the CRD's
conversion webhook through the API server service proxy. Nothing is created in
the cluster (no reconcile, no pods, no cleanup).
