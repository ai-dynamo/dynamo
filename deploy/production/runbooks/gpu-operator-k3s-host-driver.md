<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GPU Operator on k3s with Host Drivers

Use this path for k3s nodes where the host OS owns NVIDIA driver installation. This is useful when GPU Operator driver containers cannot resolve the host kernel packages, such as custom Rocky or CIQ kernel builds.

## Install the host driver

```bash
sudo dnf install -y dnf-plugins-core

if [[ ! -f /etc/yum.repos.d/cuda-rhel9.repo ]]; then
  sudo dnf config-manager addrepo \
    --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
fi

sudo dnf install -y \
  "kernel-devel-$(uname -r)" \
  "kernel-headers-$(uname -r)" \
  dkms \
  nvidia-driver \
  nvidia-driver-cuda

sudo reboot
```

After the node returns:

```bash
nvidia-smi -L
```

## Sync GPU Operator

Add both k3s overlays to the `gpu-operator` Argo CD application:

```yaml
valueFiles:
  - $values/deploy/production/addons/gpu-operator/values.yaml
  - $values/deploy/production/addons/gpu-operator/values-k3s.yaml
  - $values/deploy/production/addons/gpu-operator/values-k3s-preinstalled-driver.yaml
```

The first k3s overlay points GPU Operator at the k3s containerd template and socket. The preinstalled-driver overlay sets `driver.enabled=false` and maps the driver install directory to `/`, leaving GPU Operator to manage the toolkit, device plugin, GFD, and DCGM exporter.

## Verify

```bash
kubectl -n gpu-operator get pods
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}'
```
