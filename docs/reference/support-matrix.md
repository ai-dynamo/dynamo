<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo Support Matrix

This document provides the support matrix for Dynamo, including hardware, software and build instructions.

## Hardware Compatibility

| **CPU Architecture** | **Status**   |
| :------------------- | :----------- |
| **x86_64**           | Supported    |
| **ARM64**            | Experimental |

> [!Warning]
> While **x86_64** architecture is supported on systems with a minimum of 32 GB RAM and at least 4 CPU cores,
> the **ARM64** support is experimental and may have limitations.

### GPU Compatibility

If you are using a **GPU**, the following GPU models and architectures are supported:

| **GPU Architecture**                 | **Status** |
| :----------------------------------- | :--------- |
| **NVIDIA Blackwell Architecture**    | Supported  |
| **NVIDIA Hopper Architecture**       | Supported  |
| **NVIDIA Ada Lovelace Architecture** | Supported  |
| **NVIDIA Ampere Architecture**       | Supported  |

## Platform Architecture Compatibility

**Dynamo** is compatible with the following platforms:

| **Operating System** | **Version** | **Architecture** | **Status**   |
| :------------------- | :---------- | :--------------- | :----------- |
| **Ubuntu**           | 22.04       | x86_64           | Supported    |
| **Ubuntu**           | 24.04       | x86_64           | Supported    |
| **Ubuntu**           | 24.04       | ARM64            | Experimental |
| **CentOS Stream**    | 9           | x86_64           | Experimental |

> [!Note]
> For **Linux**, the **ARM64** support is experimental and may have limitations.
> Wheels are built using a manylinux_2_28-compatible environment and they have been validated on CentOS 9 and Ubuntu (22.04, 24.04).
>
> Compatibility with other Linux distributions is expected but has not been officially verified yet.

> [!Caution]
> KV Block Manager is supported only with Python 3.12. Python 3.12 support is currently limited to Ubuntu 24.04.

## Software Compatibility

### Runtime Dependency

| **Python Package** | **Version** | glibc version                         | CUDA Version |
| :----------------- | :---------- | :------------------------------------ | :----------- |
| ai-dynamo          | 0.7.0       | >=2.28                                |              |
| ai-dynamo-runtime  | 0.7.0       | >=2.28 (Python 3.12 has known issues) |              |
| NIXL               | 0.7.0       | >=2.27                                | >=11.8       |

### Build Dependency

| **Build Dependency** | **Version**                                                                      |
| :------------------- | :------------------------------------------------------------------------------- |
| **TensorRT-LLM**     | 1.1.0rc5                                                                         |
| **NIXL**             | 0.7.0                                                                            |
| **vLLM**             | 0.10.1.1                                                                         |
| **SGLang**           | 0.5.3rc0                                                                         |

> [!Important]
> Specific versions of TensorRT-LLM supported by Dynamo are subject to change. Currently TensorRT-LLM does not support Python 3.11 so installation of the ai-dynamo[trtllm] will fail.

## Cloud Service Provider Compatibility

### AWS

| **Host Operating System** | **Version** | **Architecture** | **Status** |
| :------------------------ | :---------- | :--------------- | :--------- |
| **Amazon Linux**          | 2023        | x86_64           | Supported¹ |

> [!Caution]
> ¹ There is a known issue with the TensorRT-LLM framework when running the AL2023 container locally with `docker run --network host ...` due to a [bug](https://github.com/mpi4py/mpi4py/discussions/491#discussioncomment-12660609) in mpi4py. To avoid this issue, replace the `--network host` flag with more precise networking configuration by mapping only the necessary ports (e.g., 4222 for nats, 2379/2380 for etcd, 8000 for frontend).

## Build Support

**Dynamo** currently provides build support in the following ways:

- **Wheels**: Pre-built Python wheels are only available for **x86_64 Linux**.
  No wheels are available for other platforms at this time.

- **Runtime Container Images**: We distribute only **AMD64** images of the runtime target on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo) for [TensorRT-LLM](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/tensorrtllm-runtime), [vLLM](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/vllm-runtime), and [SGLang](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/sglang-runtime).
  Users must build the container image from source if they require an **ARM64** image.

- **Deployment-supportive Images**: [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo) hosts the [Dynamo kubernetes-operator](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/kubernetes-operator) to simplify deployments of Dynamo Graphs.
  It is currently provided as an **AMD64** image only.

- **Helm Charts**: [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo) hosts the helm charts supporting Kubernetes deployments of Dynamo. [Dynamo CRDs](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/helm-charts/dynamo-crds), [Dynamo Platform](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/helm-charts/dynamo-platform), and [Dynamo Graph](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/helm-charts/dynamo-graph) are available.

Once you've confirmed that your platform and architecture are compatible, you can install **Dynamo** by following the instructions in the [Quick Start Guide](https://github.com/ai-dynamo/dynamo/blob/main/README.md#installation).
