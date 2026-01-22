<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo Support Matrix

This document provides the support matrix for Dynamo, including hardware, software and build instructions.

> **See also:** [Feature Compatibility Matrix](../../feature-matrix.md) for backend-specific feature support (vLLM, TensorRT-LLM, SGLang).

## Hardware Compatibility

| **CPU Architecture** | **Status**   |
| :------------------- | :----------- |
| **x86_64**           | Supported    |
| **ARM64**            | Supported    |


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
| **Ubuntu**           | 24.04       | ARM64            | Supported    |
| **CentOS Stream**    | 9           | x86_64           | Experimental |

> [!Note]
> Wheels are built using a manylinux_2_28-compatible environment and they have been validated on CentOS 9 and Ubuntu (22.04, 24.04).
>
> Compatibility with other Linux distributions is expected but has not been officially verified yet.

> [!Caution]
> KV Block Manager is supported only with Python 3.12. Python 3.12 support is currently limited to Ubuntu 24.04.

## Software Compatibility

### Runtime Dependency

| **Python Package** | **Version** | glibc version                         | CUDA Version |
| :----------------- | :---------- | :------------------------------------ | :----------- |
| ai-dynamo          | 0.9.0       | >=2.28                                |              |
| ai-dynamo-runtime  | 0.9.0       | >=2.28 (Python 3.12 has known issues) |              |
| NIXL               | 0.9.0       | >=2.27                                | >=11.8       |

### Build Dependency

The following table shows the dependency versions included with each Dynamo release:

| **Dependency** | **main (ToT)** | **v0.9.0 (unreleased)** | **v0.8.1 (latest)** | **v0.8.0** | **v0.7.1** | **v0.7.0.post1** | **v0.7.0** |
| :------------- | :------------- | :---------------------- | :------------------ | :--------- | :--------- | :--------------- | :--------- |
| SGLang         | 0.5.8          | 0.5.8      | 0.5.6.post2 | 0.5.6.post2 | 0.5.3.post4| 0.5.3.post4      | 0.5.3.post4|
| TensorRT-LLM   | 1.3.0rc1       | 1.3.0rc1   | 1.2.0rc6.post1 | 1.2.0rc6.post1 | 1.2.0rc3   | 1.2.0rc3         | 1.2.0rc2   |
| vLLM           | 0.14.0         | 0.14.0     | 0.12.0     | 0.12.0     | 0.11.0     | 0.11.0           | 0.11.0     |
| NIXL           | 0.9.0          | 0.9.0      | 0.8.0      | 0.8.0      | 0.8.0      | 0.8.0            | 0.8.0      |

> [!Note]
> **main (ToT)** reflects the current development branch. **v0.9.0** is currently in QA and not yet released.


> [!Important]
> Specific versions of TensorRT-LLM supported by Dynamo are subject to change. Currently TensorRT-LLM does not support Python 3.11 so installation of the ai-dynamo[trtllm] will fail.

### CUDA Support by Framework
| **Dynamo Version**   | **SGLang**                        | **TensorRT-LLM**        | **vLLM**                          |
| :------------------- | :-------------------------------- | :-----------------------| :-------------------------------- |
| **Dynamo 0.9.0**     | CUDA 12.9.1, CUDA 13.0            | CUDA 13.0               | CUDA 12.9, CUDA 13.0              |
| **Dynamo 0.8.1**     | CUDA 12.9, CUDA 13.0 (🧪)         | CUDA 13.0               | CUDA 12.9, CUDA 13.0 (🧪)         |
| **Dynamo 0.8.0**     | CUDA 12.9, CUDA 13.0 (🧪)         | CUDA 13.0               | CUDA 12.9, CUDA 13.0 (🧪)         |
| **Dynamo 0.7.1**     | CUDA 12.8                         | CUDA 13.0               | CUDA 12.9                         |

> 🧪 = Experimental

## Cloud Service Provider Compatibility

### AWS

| **Host Operating System** | **Version** | **Architecture** | **Status** |
| :------------------------ | :---------- | :--------------- | :--------- |
| **Amazon Linux**          | 2023        | x86_64           | Supported¹ |

> [!Caution]
> There is a known issue with the TensorRT-LLM framework when running the AL2023 container locally with `docker run --network host ...` due to a [bug](https://github.com/mpi4py/mpi4py/discussions/491#discussioncomment-12660609) in mpi4py. To avoid this issue, replace the `--network host` flag with more precise networking configuration by mapping only the necessary ports (e.g., 4222 for nats, 2379/2380 for etcd, 8000 for frontend).

## Build Support

**Dynamo** currently provides build support in the following ways:

- **Wheels**: We distribute Python wheels of Dynamo and KV Block Manager:
  - [ai-dynamo](https://pypi.org/project/ai-dynamo/)
  - [ai-dynamo-runtime](https://pypi.org/project/ai-dynamo-runtime/)
  - [kvbm](https://pypi.org/project/kvbm/) as a standalone implementation.

- **Dynamo Container Images**: We distribute multi-arch images (x86 & ARM64 compatible) on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo):
  - [Dynamo Frontend](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/dynamo-frontend) *(New in v0.8.0)*
  - [SGLang Runtime](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/sglang-runtime)
  - [SGLang Runtime (CUDA 13)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/sglang-runtime-cu13)
  - [TensorRT-LLM Runtime](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/tensorrtllm-runtime)
  - [vLLM Runtime](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/vllm-runtime)
  - [vLLM Runtime (CUDA 13)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/vllm-runtime-cu13)
  - [Kubernetes Operator](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/kubernetes-operator)

- **Helm Charts**: [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo) hosts the helm charts supporting Kubernetes deployments of Dynamo:
  - [Dynamo CRDs](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/helm-charts/dynamo-crds)
  - [Dynamo Platform](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/helm-charts/dynamo-platform)
  - [Dynamo Graph](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/helm-charts/dynamo-graph)

- **Rust Crates**:
  - [dynamo-runtime](https://crates.io/crates/dynamo-runtime/)
  - [dynamo-llm](https://crates.io/crates/dynamo-llm/)
  - [dynamo-async-openai](https://crates.io/crates/dynamo-async-openai/)
  - [dynamo-parsers](https://crates.io/crates/dynamo-parsers/)
  - [dynamo-config](https://crates.io/crates/dynamo-config/) *(New in v0.8.0)*
  - [dynamo-memory](https://crates.io/crates/dynamo-memory/) *(New in v0.8.0)*
  - [dynamo-tokens](https://crates.io/crates/dynamo-tokens/) *(New in v0.9.0)*

Once you've confirmed that your platform and architecture are compatible, you can install **Dynamo** by following the instructions in the [Quick Start Guide](https://github.com/ai-dynamo/dynamo/blob/main/README.md#installation).
