# Dynamo test cluster prerequisites

This directory contains a first-pass bootstrap plan for the target cluster described by `cxl/.codex-tmp/user-kubeconfig.yaml`.

Current cluster facts observed on 2026-04-02:

- Kubernetes version: `v1.35.0`
- Runtime: `containerd://1.7.30`
- GPU-named nodes exist:
  - `yj-testaiinfragpu-01`
  - `yj-testaiinfragpu-02`
  - `yj-testaiinfragpu-03`
- No `StorageClass` exists
- No `gpu-operator` pods exist
- GPU nodes do not currently expose `nvidia.com/gpu.present=true`

Planned install order:

1. Install a default `StorageClass` for this test cluster using local-path-provisioner.
2. Install NVIDIA GPU Operator using Helm with containerd-oriented settings.
3. Verify GPU resources and node labels.
4. If the label `nvidia.com/gpu.present=true` is still missing after GPU Operator is healthy, label the three GPU nodes manually.
5. Re-run `./deploy/pre-deployment/pre-deployment-check.sh`.

Apply the storage manifest:

```bash
kubectl apply -f cxl/k8s-prereqs/local-path-provisioner.yaml
kubectl get sc
```

Install GPU Operator:

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

helm upgrade --install gpu-operator nvidia/gpu-operator \
  -n gpu-operator \
  --create-namespace \
  -f cxl/k8s-prereqs/gpu-operator-values.yaml
```

Verify GPU operator:

```bash
kubectl get pods -n gpu-operator
kubectl get nodes -o wide
kubectl get nodes --show-labels | grep nvidia.com/gpu.present || true
kubectl describe node yj-testaiinfragpu-01 | grep -A5 -E "Capacity:|Allocatable:"
```

If GPU nodes are healthy but the expected label is still absent, label them explicitly:

```bash
kubectl label node yj-testaiinfragpu-01 nvidia.com/gpu.present=true --overwrite
kubectl label node yj-testaiinfragpu-02 nvidia.com/gpu.present=true --overwrite
kubectl label node yj-testaiinfragpu-03 nvidia.com/gpu.present=true --overwrite
```

Then rerun:

```bash
./deploy/pre-deployment/pre-deployment-check.sh
```

Notes:

- `local-path-provisioner` is a pragmatic test-cluster choice. It is not a shared RWX storage backend.
- The manifest uses the `default` ServiceAccount in namespace `local-path-storage`, but still keeps ClusterRole and ClusterRoleBinding because dynamic PV provisioning needs cluster-scoped permissions.
- The GPU Operator install here assumes the nodes are real GPU hosts and that the cluster can pull NVIDIA images.
- We intentionally keep GPU Operator as Helm values instead of a giant rendered manifest so you can review the intent cleanly before installation.
