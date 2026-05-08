---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: RDMA / InfiniBand on AKS
---

# RDMA / InfiniBand on AKS

This guide covers setting up RDMA over InfiniBand on AKS for high-performance disaggregated inference with Dynamo. RDMA enables direct memory access between GPUs across nodes, bypassing CPU and kernel overhead — critical for low-latency KV cache transfer between prefill and decode workers.

Without RDMA, disaggregated inference falls back to TCP with severe performance degradation (~98s TTFT vs ~200-500ms with RDMA). See the [Disaggregated Communication Guide](../../disagg-communication-guide.md) for details on transport options and performance expectations.

> [!NOTE]
> The Network Operator and NicClusterPolicy steps in this guide are based on the [Azure AKS RDMA InfiniBand](https://github.com/Azure/aks-rdma-infiniband) repository. That project is open-source and not covered by Microsoft Azure support — file issues on the GitHub repository.

## Prerequisites

**AKS cluster with RDMA-capable nodes:**

- At least **2 GPU nodes** to enable cross-node RDMA communication
- **ND-series VMs** with Mellanox ConnectX InfiniBand NICs (e.g., `Standard_ND96asr_v4`, `Standard_ND96isr_H100_v5`)
- **Ubuntu OS** on the node pool (required for NVIDIA driver compatibility)
- GPU driver installation **skipped** on the node pool (`--skip-gpu-driver-install`) — see [GPU Node Pool Setup](aks.md#step-2-add-a-gpu-node-pool)

**Register the AKS InfiniBand feature** to ensure nodes land on the same physical InfiniBand network:

```bash
az feature register --namespace Microsoft.ContainerService --name AKSInfinibandSupport
az feature show --namespace Microsoft.ContainerService --name AKSInfinibandSupport --query "properties.state"
# Wait until "Registered"

az provider register --namespace Microsoft.ContainerService
```

## Overview

The RDMA setup involves five components installed in this order:

1. **Network Operator** — Deploys the Mellanox OFED driver and Node Feature Discovery
2. **NicClusterPolicy** — Configures the OFED driver on InfiniBand-capable nodes
3. **IB Node Configuration** — Loads InfiniBand kernel modules and sets memlock limits
4. **RDMA Shared Device Plugin** — Exposes InfiniBand NICs to pods as a Kubernetes resource
5. **GPU Operator** — Installed with RDMA-specific settings (NFD disabled, GPUDirect RDMA enabled, host MOFED)

## Step 1: Install the NVIDIA Network Operator

The [NVIDIA Network Operator](https://docs.nvidia.com/networking/display/cokan10/network+operator) automates deployment of networking components including Mellanox OFED drivers for InfiniBand support.

Create the namespace and label it for privileged workloads:

```bash
kubectl create ns network-operator
kubectl label --overwrite ns network-operator pod-security.kubernetes.io/enforce=privileged
```

Add the NVIDIA Helm repo (if not already added):

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update
```

Create a `network-operator-values.yaml`:

```yaml
nfd:
  deployNodeFeatureRules: false
```

Install the Network Operator:

```bash
helm install network-operator nvidia/network-operator \
  --namespace network-operator \
  -f network-operator-values.yaml \
  --version v26.1.0
```

Verify the Network Operator pod is running:

```bash
kubectl get pods -n network-operator
```

## Step 2: Apply the NicClusterPolicy

The NicClusterPolicy configures the OFED driver (Mellanox OFED / DOCA driver) as a DaemonSet on all InfiniBand-capable nodes.

Apply the base NicClusterPolicy using kustomize:

```bash
kubectl apply -k https://github.com/Azure/aks-rdma-infiniband/configs/nicclusterpolicy/base
```

This targets nodes with Mellanox NICs (`feature.node.kubernetes.io/pci-15b3.present`) and installs the DOCA/OFED driver as a DaemonSet.

Wait for the MOFED driver DaemonSet to finish installing on all nodes (this may take several minutes):

```bash
kubectl get pods -n network-operator -l app=mofed-ubuntu22.04-ds -w
# Wait until all pods show Running
```

## Step 3: Deploy the IB Node Configuration DaemonSet

This DaemonSet loads InfiniBand kernel modules and sets unlimited memlock limits on GPU nodes. This is required for RDMA to function — without it, InfiniBand device files may not exist and memory pinning for RDMA transfers will fail.

> [!IMPORTANT]
> This step is not covered in the Azure RDMA repo but is required for a working setup. The DaemonSet loads `ib_umad` and `rdma_ucm` kernel modules, sets unlimited memlock limits for containerd and kubelet, and restarts both services to apply the changes.

Create `ib-node-config.yaml`:

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: ib-node-config
  namespace: kube-system
spec:
  selector:
    matchLabels:
      app: ib-node-config
  template:
    metadata:
      labels:
        app: ib-node-config
    spec:
      hostPID: true
      nodeSelector:
        kubernetes.azure.com/agentpool: <GPU_NODE_POOL_NAME>
      tolerations:
      - operator: Exists
      initContainers:
      - name: ib-setup
        image: busybox:1.36
        securityContext:
          privileged: true
        command:
        - sh
        - -c
        - |
          echo "=== IB Node Configuration ==="

          nsenter -t 1 -m -u -i -n -- modprobe ib_umad
          nsenter -t 1 -m -u -i -n -- modprobe rdma_ucm 2>/dev/null || true
          nsenter -t 1 -m -u -i -n -- modprobe ib_ucm 2>/dev/null || true
          nsenter -t 1 -m -u -i -n -- lsmod | grep ib_umad && echo "OK: ib_umad" || echo "FAIL: ib_umad"
          nsenter -t 1 -m -u -i -n -- ls /dev/infiniband/rdma_cm && echo "OK: rdma_cm device" || echo "WARN: no rdma_cm device"

          nsenter -t 1 -m -u -i -n -- sh -c 'printf "ib_umad\nrdma_ucm\n" > /etc/modules-load.d/ib-umad.conf'
          nsenter -t 1 -m -u -i -n -- sh -c 'printf "* - memlock unlimited\nroot - memlock unlimited\n" > /etc/security/limits.d/99-ib-memlock.conf'

          nsenter -t 1 -m -u -i -n -- sh -c 'mkdir -p /etc/systemd/system/containerd.service.d && printf "[Service]\nLimitMEMLOCK=infinity\n" > /etc/systemd/system/containerd.service.d/memlock.conf'
          nsenter -t 1 -m -u -i -n -- sh -c 'mkdir -p /etc/systemd/system/kubelet.service.d && printf "[Service]\nLimitMEMLOCK=infinity\n" > /etc/systemd/system/kubelet.service.d/memlock.conf'

          nsenter -t 1 -m -u -i -n -- systemctl daemon-reload
          nsenter -t 1 -m -u -i -n -- systemctl restart containerd
          nsenter -t 1 -m -u -i -n -- systemctl restart kubelet

          sleep 10

          nsenter -t 1 -m -u -i -n -- systemctl is-active containerd && echo "OK: containerd active" || echo "FAIL: containerd"
          nsenter -t 1 -m -u -i -n -- systemctl is-active kubelet && echo "OK: kubelet active" || echo "FAIL: kubelet"

          echo "=== Setup Complete ==="
      containers:
      - name: keepalive
        image: busybox:1.36
        command: ["sh", "-c", "echo IB node config active; sleep infinity"]
```

> [!NOTE]
> Replace `<GPU_NODE_POOL_NAME>` with your GPU node pool name (e.g., `ndh100pool`).

```bash
kubectl apply -f ib-node-config.yaml
```

Wait for all pods to complete initialization:

```bash
kubectl get pods -n kube-system -l app=ib-node-config -w
```

**What this does:**
- **`ib_umad`** — InfiniBand user-space management datagram module, required for RDMA device access
- **`rdma_ucm`** — RDMA user-space connection manager
- **Memlock limits** — RDMA requires pinning memory pages; without unlimited memlock, large transfers fail
- **Service restarts** — containerd and kubelet must be restarted to pick up the new memlock limits

## Step 4: Deploy the RDMA Shared Device Plugin

The RDMA Shared Device Plugin exposes InfiniBand NICs as a Kubernetes extended resource so pods can request RDMA access.

Create the ConfigMap with the device plugin configuration:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rdma-devices
  namespace: kube-system
data:
  config.json: |
    {
        "periodicUpdateInterval": 300,
        "configList": [{
             "resourceName": "hca_shared_devices_a",
             "rdmaHcaMax": 1000,
             "selectors": {
               "vendors": ["15b3"],
               "drivers": ["mlx5_core"]
             }
           }
        ]
    }
```

Create the DaemonSet:

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: rdma-shared-dp-ds
  namespace: kube-system
spec:
  selector:
    matchLabels:
      name: rdma-shared-dp-ds
  template:
    metadata:
      labels:
        name: rdma-shared-dp-ds
    spec:
      hostNetwork: true
      nodeSelector:
        kubernetes.azure.com/agentpool: <GPU_NODE_POOL_NAME>
      tolerations:
      - operator: Exists
      containers:
      - name: k8s-rdma-shared-dp-ds
        image: ghcr.io/mellanox/k8s-rdma-shared-dev-plugin:v1.5.3
        securityContext:
          privileged: true
        volumeMounts:
        - name: device-plugin
          mountPath: /var/lib/kubelet/device-plugins
        - name: plugins-registry
          mountPath: /var/lib/kubelet/plugins_registry
        - name: config
          mountPath: /k8s-rdma-shared-dev-plugin
        - name: devs
          mountPath: /dev/
      volumes:
      - name: device-plugin
        hostPath:
          path: /var/lib/kubelet/device-plugins
      - name: plugins-registry
        hostPath:
          path: /var/lib/kubelet/plugins_registry
      - name: config
        configMap:
          name: rdma-devices
      - name: devs
        hostPath:
          path: /dev/
```

> [!NOTE]
> Replace `<GPU_NODE_POOL_NAME>` with your GPU node pool name (e.g., `ndh100pool`).

```bash
kubectl apply -f rdma-configmap.yaml
kubectl apply -f rdma-shared-dp-ds.yaml
```

Wait for the device plugin pods to start:

```bash
kubectl get pods -n kube-system -l name=rdma-shared-dp-ds -w
```

## Step 5: Install the GPU Operator (RDMA-Enabled)

Install the GPU Operator with RDMA-specific values:

```bash
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator --create-namespace \
  --set nfd.enabled=false \
  --set driver.rdma.enabled=true \
  --set driver.rdma.useHostMofed=true
```

Key differences from a standard GPU Operator install:

- `nfd.enabled=false` — Network Operator already deploys Node Feature Discovery; running two NFD instances causes conflicts
- `driver.rdma.enabled=true` — enables GPUDirect RDMA support; causes the driver daemonset to build and load `nvidia_peermem`
- `driver.rdma.useHostMofed=true` — tells the GPU Operator to use the MOFED driver installed by the Network Operator (Step 1) rather than its own; required when the Network Operator manages OFED

Wait for the GPU Operator pods to reach `Running` state:

```bash
kubectl get pods -n gpu-operator -w
```

## Verification

**1. Check that MOFED driver pods are running on all InfiniBand nodes:**

```bash
kubectl get pods -n network-operator -l app=mofed-ubuntu22.04-ds
```

**2. Check that IB node config pods completed initialization:**

```bash
kubectl get pods -n kube-system -l app=ib-node-config
```

**3. Check that the RDMA Shared Device Plugin is running:**

```bash
kubectl get pods -n kube-system -l name=rdma-shared-dp-ds
```

**4. Verify RDMA resources are available on GPU nodes:**

```bash
kubectl get nodes -o json | jq '.items[] | select(.status.allocatable["rdma/hca_shared_devices_a"] != null) | {name: .metadata.name, rdma: .status.allocatable["rdma/hca_shared_devices_a"], gpu: .status.allocatable["nvidia.com/gpu"]}'
```

Each InfiniBand-capable node should report `rdma/hca_shared_devices_a` resources (typically `1k` based on `rdmaHcaMax: 1000`).

**5. Check GPU Operator pods are healthy:**

```bash
kubectl get pods -n gpu-operator
```

## Pod Resource Requests

Dynamo pods that need RDMA access should request the `rdma/hca_shared_devices_a` resource. When using the Dynamo operator with DGDR, this is handled automatically for disaggregated deployments on RDMA-capable clusters.

For manual DGD specs, add the resource request to your container:

```yaml
resources:
  limits:
    nvidia.com/gpu: 8
    rdma/hca_shared_devices_a: 1
```

> [!NOTE]
> **`IPC_LOCK` capability is not required** when this setup is followed. `IPC_LOCK` is historically needed for RDMA because `ibv_reg_mr` calls `mlock()` to pin memory pages — but `mlock()` only needs the capability if the memlock rlimit would otherwise block it. The `ib-node-config` DaemonSet (Step 3) sets `LimitMEMLOCK=infinity` on the kubelet and containerd systemd units, so all pods on GPU nodes inherit an unlimited memlock limit and RDMA memory pinning works without any capability in the pod spec.
>
> If you see `ENOMEM` errors from `ibv_reg_mr` and `ib-node-config` is running, verify that containerd and kubelet were restarted after the limits were applied (check the init container logs). If `ib-node-config` is not deployed, add `IPC_LOCK` to your pod's `securityContext.capabilities.add`.

## Troubleshooting

**MOFED pods stuck in `Init` or `CrashLoopBackOff`:**
- Verify nodes are Ubuntu OS: `kubectl get nodes -o custom-columns="NAME:.metadata.name,OS:.status.nodeInfo.osImage"`
- Check MOFED pod logs: `kubectl logs -n network-operator <mofed-pod> -c mofed-container`

**`rdma/hca_shared_devices_a` not showing on nodes:**
- Check the RDMA device plugin pods are running: `kubectl get pods -n kube-system -l name=rdma-shared-dp-ds`
- Check device plugin logs: `kubectl logs -n kube-system <rdma-shared-dp-pod>`
- Verify the `rdma-devices` ConfigMap exists: `kubectl get configmap rdma-devices -n kube-system`

**IB kernel modules not loading:**
- Check the ib-node-config init container logs: `kubectl logs -n kube-system <ib-node-config-pod> -c ib-setup`
- Verify the MOFED driver is installed first (Step 2 must complete before Step 3)

**Memlock errors during RDMA transfers (`ENOMEM` from `ibv_reg_mr`):**
- Verify the ib-node-config DaemonSet has run on all GPU nodes and init containers completed
- Check that containerd and kubelet were restarted: `kubectl logs -n kube-system <ib-node-config-pod> -c ib-setup`
- Confirm the limits took effect on the kubelet process:
  ```bash
  # On a GPU node (via kubectl debug or ssh)
  cat /proc/$(pgrep -x kubelet)/limits | grep -i memlock
  # Should show: Max locked memory  unlimited  unlimited
  ```
- If limits are not unlimited, the ib-node-config DaemonSet needs to be re-applied and services restarted

**GPUDirect RDMA not working — `nvidia_peermem` module missing:**

ND-series nodes (including ND H100 v5) do **not** ship `nvidia_peermem` in the host OS. This module is required for InfiniBand adapters to directly read/write GPU memory — without it, RDMA transfers fall back to staging through host memory.

Verify whether the module is loaded:

```bash
# Check on a GPU node via a privileged pod or node shell
lsmod | grep nvidia_peermem
# If empty, the module is not loaded

modinfo nvidia_peermem
# If "Module not found", it is also not present in the host's /lib/modules
```

With the GPU Operator managing drivers (`driver.rdma.enabled=true`), `nvidia_peermem` is built and loaded by the `nvidia-driver-daemonset` — it lives in the driver pod's `/lib/modules`, not the host's native kernel modules. Verify the driver daemonset is loading it:

```bash
kubectl exec -n gpu-operator $(kubectl get pod -n gpu-operator -l app=nvidia-driver-daemonset -o jsonpath='{.items[0].metadata.name}') -- lsmod | grep nvidia_peermem
```

If this returns empty, ensure `driver.rdma.enabled=true` and `driver.rdma.useHostMofed=true` are set in your GPU Operator Helm values (see [Step 5](#step-5-install-the-gpu-operator-rdma-enabled) above), then restart the driver daemonset:

```bash
kubectl rollout restart daemonset/nvidia-driver-daemonset -n gpu-operator
```

> [!NOTE]
> The [nvidia-peermem-reloader](https://github.com/Azure/aks-rdma-infiniband/tree/main/configs/nvidia-peermem-reloader) DaemonSet from the Azure RDMA repo is designed for clusters using **AKS-managed GPU drivers** (without the GPU Operator). It simply runs `modprobe nvidia-peermem` — which will fail on ND H100 v5 nodes because the host OS doesn't include the module. When using the GPU Operator (recommended), the operator handles `nvidia_peermem` automatically via `driver.rdma.enabled=true`.

## See Also

- [Azure AKS RDMA InfiniBand — GitHub](https://github.com/Azure/aks-rdma-infiniband)
- [Set up InfiniBand on Azure HPC VMs — Microsoft Learn](https://learn.microsoft.com/en-us/azure/virtual-machines/setup-infiniband)
- [Enable InfiniBand VM extension — Microsoft Learn](https://learn.microsoft.com/en-us/azure/virtual-machines/extensions/enable-infiniband)
- [NVIDIA Network Operator Documentation](https://docs.nvidia.com/networking/display/cokan10/network+operator)
- [Disaggregated Communication Guide](../../disagg-communication-guide.md) — transport options, UCX configuration, performance expectations

