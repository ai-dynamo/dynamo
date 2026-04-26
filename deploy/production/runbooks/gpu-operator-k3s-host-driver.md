<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GPU Operator on k3s with CIQ Host Drivers

Use this path for k3s nodes where the host OS needs to install the NVIDIA driver before GPU Operator runs. This is useful when GPU Operator driver containers cannot resolve the host kernel packages, such as custom Rocky or CIQ kernel builds.

## Install the host driver

```bash
sudo dnf install -y dnf-plugins-core
sudo dnf install -y epel-release

if [[ ! -f /etc/yum.repos.d/cuda-rhel9.repo ]]; then
  sudo dnf config-manager --add-repo \
    https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
fi

sudo sed -i 's/gpgcheck=1/gpgcheck=0/' /etc/yum.repos.d/cuda-rhel9.repo

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

Add the k3s overlay to the `gpu-operator` Argo CD application:

```yaml
valueFiles:
  - $values/deploy/production/addons/gpu-operator/values.yaml
  - $values/deploy/production/addons/gpu-operator/values-k3s.yaml
```

The k3s overlay points GPU Operator at the k3s containerd template and socket. Leave `driver.enabled` at the chart default so the GPU Operator driver pod can detect the preinstalled driver and proceed with toolkit, device plugin, GFD, and DCGM exporter reconciliation.

## Verify

```bash
kubectl -n gpu-operator get pods
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}'
```
