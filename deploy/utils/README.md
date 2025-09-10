# Kubernetes utilities for Dynamo

This directory contains small utilities and manifests used by benchmarking and profiling flows.

## Contents

- `setup_k8s_namespace.sh` — **fully encapsulated deployment setup** that provides one-time per Kubernetes namespace setup. Creates namespace (if missing), applies common manifests, installs CRDs, and deploys the Dynamo operator. If `DOCKER_SERVER`/`IMAGE_TAG` are provided, it installs your custom operator image; otherwise it installs the default published image. If your registry is private, provide `DOCKER_USERNAME`/`DOCKER_PASSWORD` or respond to the prompt to create an image pull secret.
- `manifests/`
  - `serviceaccount.yaml` — ServiceAccount `dynamo-sa`
  - `role.yaml` — Role `dynamo-role`
  - `rolebinding.yaml` — RoleBinding `dynamo-binding`
  - `pvc.yaml` — PVC `dynamo-pvc`
  - `pvc-access-pod.yaml` — short‑lived pod for copying profiler results from the PVC
- `kubernetes.py` — helper used by tooling to apply/read resources (e.g., access pod for PVC downloads).

## Quick start

### Kubernetes Setup (one-time per namespace)

Use the helper script to prepare a Kubernetes namespace with the common manifests and install the operator. This provides a **fully encapsulated deployment setup**.

This script creates a Kubernetes namespace with the given name if it does not yet exist. It then applies common manifests (serviceaccount, role, rolebinding, pvc), installs CRDs, creates secrets, and deploys the Dynamo Cloud Operator to your namespace.
If your namespace is already set up, you can skip this step.

```bash
export HF_TOKEN=<HF_TOKEN>
export DOCKER_SERVER=<YOUR_DOCKER_SERVER>

NAMESPACE=benchmarking HF_TOKEN=$HF_TOKEN DOCKER_SERVER=$DOCKER_SERVER deploy/utils/setup_k8s_namespace.sh

# IF you want to build and push a new Docker image for the Dynamo Cloud Operator, include an IMAGE_TAG
# NAMESPACE=benchmarking HF_TOKEN=$HF_TOKEN DOCKER_SERVER=$DOCKER_SERVER IMAGE_TAG=latest deploy/utils/setup_k8s_namespace.sh
```

This script applies the following manifests:

- `deploy/utils/manifests/serviceaccount.yaml` - ServiceAccount `dynamo-sa`
- `deploy/utils/manifests/role.yaml` - Role `dynamo-role`
- `deploy/utils/manifests/rolebinding.yaml` - RoleBinding `dynamo-binding`
- `deploy/utils/manifests/pvc.yaml` - PVC `dynamo-pvc`

If `DOCKER_SERVER` and `IMAGE_TAG` are not both provided, the script deploys the operator using the default published image `nvcr.io/nvidia/ai-dynamo/kubernetes-operator:0.4.0`.
To build/push and use a new image instead, pass both `DOCKER_SERVER` and `IMAGE_TAG`.

This script also installs the Dynamo CRDs if not present.

If the registry is private, either pass credentials or respond to the prompt:

```bash
NAMESPACE=benchmarking \
DOCKER_SERVER=my-registry.example.com \
IMAGE_TAG=latest \
DOCKER_USERNAME="$oauthtoken" \
DOCKER_PASSWORD=<token> \
deploy/utils/setup_k8s_namespace.sh
```

If `DOCKER_SERVER`/`IMAGE_TAG` are omitted, the script installs the default operator image `nvcr.io/nvidia/ai-dynamo/kubernetes-operator:0.4.0`.

After running the setup script, verify the installation by checking the pods:

```bash
kubectl get pods -n $NAMESPACE
```

The output should look something like:

```
NAME                                                            READY   STATUS    RESTARTS   AGE
dynamo-platform-dynamo-operator-controller-manager-xxxxx       2/2     Running   0          5m
dynamo-platform-etcd-0                                          1/1     Running   0          5m
dynamo-platform-nats-0                                          2/2     Running   0          5m
dynamo-platform-nats-box-xxxxx                                  1/1     Running   0          5m
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

**Download benchmark/profiling results:**

```bash
# After benchmarking or profiling completes, download results
python3 -m deploy.utils.download_pvc_results \
  --namespace $NAMESPACE \
  --output-dir ./pvc_files \
  --folder /data/results \
  --no-config   # optional: skip *.yaml/*.yml in the download
```

#### Path Requirements

**Important**: The PVC is mounted at `/data` in the access pod for security reasons. All destination paths must start with `/data/`.

**Common path patterns:**
- `/data/configs/` - Configuration files (DGD manifests)
- `/data/results/` - Benchmark results
- `/data/profiling_results/` - Profiling data
- `/data/benchmarking/` - Benchmarking artifacts

**User-friendly error messages**: If you forget the `/data/` prefix, the script will show a helpful error message with the correct path and example commands.

#### Next Steps

For complete benchmarking workflows:
- **Benchmarking Guide**: See [docs/benchmarks/benchmarking.md](../../docs/benchmarks/benchmarking.md) for comparing DynamoGraphDeployments and external endpoints
- **Pre-Deployment Profiling**: See [docs/benchmarks/pre_deployment_profiling.md](../../docs/benchmarks/pre_deployment_profiling.md) for optimizing configurations before deployment

## Notes

- Benchmarking scripts (`benchmarks/benchmark.sh`, `benchmarks/deploy_benchmark.sh`) call this setup automatically when present.
- Profiling job manifest remains in `benchmarks/profiler/deploy/profile_sla_job.yaml` and now relies on the common ServiceAccount/PVC here.
