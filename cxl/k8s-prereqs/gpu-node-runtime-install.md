# GPU Node Runtime Installation Checklist

This checklist is for the three GPU nodes:

- `yj-testaiinfragpu-01`
- `yj-testaiinfragpu-02`
- `yj-testaiinfragpu-03`

Current observed state:

- `nvidia-smi` exists on all three nodes
- `nvidia-container-runtime` is missing
- `nvidia-ctk` is missing
- `/etc/containerd/config.toml` exists
- `containerd` is not configured with an NVIDIA runtime

Goal:

- Install `nvidia-container-toolkit`
- Configure `containerd` with NVIDIA runtime support
- Restart `containerd`
- Verify containers can request GPU through Kubernetes

## 1. Verify current state on each node

Run on each GPU node as root:

```bash
hostname
nvidia-smi
which nvidia-container-runtime || true
which nvidia-ctk || true
grep -n nvidia /etc/containerd/config.toml || true
```

Expected before fix:

- `nvidia-smi` works
- `nvidia-container-runtime` not found
- `nvidia-ctk` not found
- no `nvidia` runtime block in `containerd` config

## 2. Install NVIDIA Container Toolkit

Use the package source approved in your environment. The exact repo setup can differ by OS image and internal mirror policy.

On CentOS/RHEL-like systems, the target package is typically:

```bash
nvidia-container-toolkit
```

If your environment already has an internal YUM repo or offline RPM bundle, install from there. Example shape:

```bash
yum install -y nvidia-container-toolkit
```

After install, verify:

```bash
which nvidia-container-runtime
which nvidia-ctk
```

Expected:

- `nvidia-container-runtime` exists
- `nvidia-ctk` exists

## 3. Configure containerd to use NVIDIA runtime

Back up the existing config first:

```bash
cp /etc/containerd/config.toml /etc/containerd/config.toml.bak.$(date +%Y%m%d%H%M%S)
```

If `nvidia-ctk` is available, prefer using it to patch `containerd`:

```bash
nvidia-ctk runtime configure --runtime=containerd
```

If your environment requires explicit config path:

```bash
nvidia-ctk runtime configure --runtime=containerd --config=/etc/containerd/config.toml
```

Then restart containerd:

```bash
systemctl restart containerd
systemctl status containerd --no-pager
```

## 4. Validate containerd runtime wiring

Check the config contains NVIDIA runtime entries:

```bash
grep -n nvidia /etc/containerd/config.toml
```

You should see a runtime stanza similar in intent to:

```toml
[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia]
  runtime_type = "io.containerd.runc.v2"

[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia.options]
  BinaryName = "/usr/bin/nvidia-container-runtime"
```

The exact generated config can vary by toolkit version.

## 5. Post-change node verification

Run on each node:

```bash
which nvidia-container-runtime
which nvidia-ctk
grep -n nvidia /etc/containerd/config.toml
crictl info | grep -i runtime -A3 || true
```

Expected:

- toolkit binaries exist
- `containerd` config references NVIDIA runtime
- `containerd` is healthy after restart

## 6. Kubernetes-side verification after node changes

After all three nodes are updated, run from a machine with cluster access:

```bash
kubectl get nodes -o wide
kubectl apply -f cxl/k8s-prereqs/nvidia-device-plugin.yaml
kubectl rollout status ds/nvidia-device-plugin-daemonset -n nvidia-device-plugin --timeout=120s
kubectl get pods -n nvidia-device-plugin -o wide
kubectl describe node yj-testaiinfragpu-01 | grep -A8 -E "Capacity:|Allocatable:"
kubectl describe node yj-testaiinfragpu-02 | grep -A8 -E "Capacity:|Allocatable:"
kubectl describe node yj-testaiinfragpu-03 | grep -A8 -E "Capacity:|Allocatable:"
```

Expected:

- device-plugin pods are `Running`
- node capacity or allocatable shows `nvidia.com/gpu`

## 7. Optional labels for Dynamo scheduling checks

If you still want nodes to match the current Dynamo precheck behavior, add:

```bash
kubectl label node yj-testaiinfragpu-01 nvidia.com/gpu.present=true --overwrite
kubectl label node yj-testaiinfragpu-02 nvidia.com/gpu.present=true --overwrite
kubectl label node yj-testaiinfragpu-03 nvidia.com/gpu.present=true --overwrite
```

## Notes

- This checklist assumes the GPU driver is already installed because `nvidia-smi` works.
- If package installation must use an internal mirror, the ops team should substitute your approved repo or RPM source.
- If `nvidia-ctk runtime configure` produces a config merge conflict, stop and review `/etc/containerd/config.toml` instead of overwriting it blindly.
