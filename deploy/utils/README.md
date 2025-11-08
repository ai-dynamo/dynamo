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
- `kubernetes.py` — helper used by tooling to apply/read resources (e.g., access pod for PVC downloads)
- `inject_manifest.py` — utility for injecting deployment configurations into the PVC for profiling
- `download_pvc_results.py` — utility for downloading benchmark/profiling results from the PVC
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

### PVC Manipulation Scripts

These scripts interact with the Persistent Volume Claim (PVC) that stores configuration files and benchmark/profiling results. They're essential for the Dynamo benchmarking and profiling workflows.

#### Why These Scripts Are Needed

1. **For Pre-Deployment Profiling**: The profiling job needs access to your Dynamo deployment configurations (DGD manifests) to test different parallelization strategies
2. **For Retrieving Results**: Both benchmarking and profiling jobs write their results to the PVC, which you need to download for analysis

#### Script Usage

**Inject deployment configurations for profiling:**

```bash
# The profiling job reads your DGD config from the PVC
# IMPORTANT: All paths must start with /data/ for security reasons
python3 -m deploy.utils.inject_manifest \
  --namespace $NAMESPACE \
  --src ./my-disagg.yaml \
  --dest /data/configs/disagg.yaml
```

**Download benchmark results:**

```bash
# After benchmarking completes, download results
python3 -m deploy.utils.download_pvc_results \
  --namespace $NAMESPACE \
  --output-dir ./benchmarks/results \
  --folder /data/results \
  --no-config   # optional: skip *.yaml/*.yml in the download
```

**Download profiling results (optional, for local inspection):**

```bash
# Optional: Download profiling data for local analysis
# The planner reads directly from the PVC, so this is only needed for inspection
python3 -m deploy.utils.download_pvc_results \
  --namespace $NAMESPACE \
  --output-dir ./profiling_data \
  --folder /data
```

> **Note on Profiling Results**: When using DGDR (DynamoGraphDeploymentRequest) for SLA-driven profiling, profiling data is stored in `/data/` on the PVC. The planner component reads this data directly from the PVC, so downloading is **optional** - only needed if you want to inspect the profiling results locally (e.g., view performance plots, check configurations).

#### Path Requirements

**Important**: The PVC is mounted at `/data` in the access pod for security reasons. All destination paths must start with `/data/`.

**Common path patterns:**
- `/data/configs/` - Configuration files (DGD manifests)
- `/data/results/` - Benchmark results (for download after benchmarking jobs)
- `/data/` - Profiling data (used directly by planner, typically not downloaded)
- `/data/benchmarking/` - Benchmarking artifacts

**User-friendly error messages**: If you forget the `/data/` prefix, the script will show a helpful error message with the correct path and example commands.

#### Next Steps

For complete benchmarking and profiling workflows:
- **Benchmarking Guide**: See [docs/benchmarks/benchmarking.md](../../docs/benchmarks/benchmarking.md) for comparing DynamoGraphDeployments and external endpoints
- **Pre-Deployment Profiling**: See [docs/benchmarks/sla_driven_profiling.md](../../docs/benchmarks/sla_driven_profiling.md) for optimizing configurations before deployment

## Notes

- This setup is focused on benchmarking and profiling resources only - the main Dynamo platform must be installed separately.
