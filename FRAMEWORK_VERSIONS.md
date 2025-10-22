<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Dynamo Framework & Dependency Versions

> **⚠️ AUTO-GENERATED** - Last updated: 2025-10-21
> 
> This document is automatically generated from [dependency extraction](.github/reports/dependency_versions_latest.csv).
> To update, run: `python3 .github/scripts/dependency-extraction/generate_framework_versions.py`

This document tracks the major dependencies and critical versions used in the NVIDIA Dynamo project.

## Core Framework Dependencies

### vLLM
- **Version**: `v0.11.0`
- **Description**: High-throughput LLM serving engine
- **Component**: `vllm`

### TensorRT-LLM
- **Version**: See dependency reports

### SGLang
- **Version**: `0.5.3.post2`
- **Description**: Structured generation language for LLMs
- **Component**: `sglang`

### Additional Critical Dependencies

#### Aiperf
- **Version**: `unspecified`
- **Category**: Python Package
- **Component**: `shared`
- **Source**: https://pypi.org/project/aiperf/

#### Aiperf
- **Version**: `70af59489df2`
- **Category**: Python Git Package
- **Component**: `shared`
- **Source**: https://pypi.org/project/aiperf/

#### Aiperf
- **Version**: `70af59489df2`
- **Category**: Python Git Package
- **Component**: `shared`
- **Source**: https://pypi.org/project/aiperf/

#### Aiperf
- **Version**: `70af59489df2`
- **Category**: Python Git Package
- **Component**: `shared`
- **Source**: https://pypi.org/project/aiperf/

#### Aiperf
- **Version**: `70af59489df2`
- **Category**: Python Git Package
- **Component**: `shared`
- **Source**: https://pypi.org/project/aiperf/

#### bitnamilegacy/etcd
- **Version**: `3.6.1`
- **Category**: Docker Compose Service
- **Component**: `shared`
- **Source**: https://hub.docker.com/r/bitnamilegacy/etcd

#### Dynamo Operator
- **Version**: `0.5.0`
- **Category**: Helm Chart Dependency
- **Component**: `shared`
- **Source**: https://artifacthub.io/packages/search?ts_query_web=dynamo-operator

#### etcd
- **Version**: `12.0.18`
- **Category**: Helm Chart Dependency
- **Component**: `shared`
- **Source**: https://artifacthub.io/packages/search?ts_query_web=etcd

#### Genai Perf
- **Version**: `==0.0.15`
- **Category**: Python Package
- **Component**: `shared`
- **Source**: https://pypi.org/project/genai-perf/

#### github.com/NVIDIA/grove/operator/api
- **Version**: `v0.1.0-alpha.3`
- **Category**: Go Module
- **Component**: `operator`
- **Source**: https://pkg.go.dev/github.com/NVIDIA/grove/operator/api

#### go.etcd.io/etcd/api/v3
- **Version**: `v3.5.21`
- **Category**: Go Module
- **Component**: `operator`
- **Source**: https://pkg.go.dev/go.etcd.io/etcd/api/v3

#### go.etcd.io/etcd/client/pkg/v3
- **Version**: `v3.5.21`
- **Category**: Go Module
- **Component**: `operator`
- **Source**: https://pkg.go.dev/go.etcd.io/etcd/client/pkg/v3

#### go.etcd.io/etcd/client/v3
- **Version**: `v3.5.21`
- **Category**: Go Module
- **Component**: `operator`
- **Source**: https://pkg.go.dev/go.etcd.io/etcd/client/v3

#### Grove Charts
- **Version**: `v0.1.0-alpha.3`
- **Category**: Helm Chart Dependency
- **Component**: `shared`
- **Source**: https://artifacthub.io/packages/search?ts_query_web=grove-charts

#### Kai Scheduler
- **Version**: `v0.9.4`
- **Category**: Helm Chart Dependency
- **Component**: `shared`
- **Source**: https://artifacthub.io/packages/search?ts_query_web=kai-scheduler

#### Kubernetes
- **Version**: `==32.0.1`
- **Category**: Python Package
- **Component**: `shared`
- **Source**: https://pypi.org/project/kubernetes/

#### Kubernetes
- **Version**: `>=32.0.1,<33.0.0`
- **Category**: Python Package
- **Component**: `shared`
- **Source**: https://pypi.org/project/kubernetes/

#### Kubernetes_asyncio
- **Version**: `unspecified`
- **Category**: Python Package
- **Component**: `shared`
- **Source**: https://pypi.org/project/kubernetes_asyncio/

#### NATS
- **Version**: `2.11.4`
- **Category**: Docker Compose Service
- **Component**: `shared`

#### NATS
- **Version**: `1.3.2`
- **Category**: Helm Chart Dependency
- **Component**: `shared`
- **Source**: https://artifacthub.io/packages/search?ts_query_web=nats

#### NATS-py
- **Version**: `unspecified`
- **Category**: Python Package (Test)
- **Component**: `shared`
- **Source**: https://pypi.org/project/nats-py/

#### NATSio/prometheus-NATS-exporter
- **Version**: `0.17.3`
- **Category**: Docker Compose Service
- **Component**: `shared`
- **Source**: https://hub.docker.com/r/NATSio/prometheus-NATS-exporter

#### Nixl
- **Version**: `0.6.0`
- **Category**: System
- **Component**: `shared`

#### Nixl
- **Version**: `<=0.6.0`
- **Category**: Python Package (vllm)
- **Component**: `shared`
- **Source**: https://pypi.org/project/nixl/

#### Nixl
- **Version**: `<=0.6.0`
- **Category**: Python Package (sglang)
- **Component**: `shared`
- **Source**: https://pypi.org/project/nixl/

#### Python
- **Version**: `3.12`
- **Category**: Framework
- **Component**: `shared`
- **Source**: https://www.python.org/downloads/

#### Rust
- **Version**: `1.90.0`
- **Category**: Language
- **Component**: `shared`
- **Source**: https://www.rust-lang.org/tools/install

#### Sglang [all]
- **Version**: `==0.5.3.post2`
- **Category**: Python Package (sglang)
- **Component**: `shared`
- **Source**: https://pypi.org/project/sglang/

#### Ucx PY Cu12
- **Version**: `unspecified`
- **Category**: Python Package (Standard)
- **Component**: `shared`
- **Source**: https://pypi.org/project/ucx-py-cu12/

#### Uvicorn
- **Version**: `unspecified`
- **Category**: Python Package
- **Component**: `shared`
- **Source**: https://pypi.org/project/uvicorn/

#### Vllm [flashinfer]
- **Version**: `==0.10.2`
- **Category**: Python Package (vllm)
- **Component**: `shared`
- **Source**: https://pypi.org/project/vllm/

## Base & Runtime Images

### OPERATOR Container Images

#### Base
- **Tag**: `latest`
- **Category**: Base Image
- **Source File**: `deploy/cloud/operator/Dockerfile`

#### Base
- **Tag**: `latest`
- **Category**: Base Image
- **Source File**: `deploy/cloud/operator/Dockerfile`

#### Base
- **Tag**: `latest`
- **Category**: Base Image
- **Source File**: `deploy/cloud/operator/Dockerfile`

#### NVIDIA Go
- **Tag**: `v3.1.13`
- **Category**: Base Image
- **Source File**: `deploy/cloud/operator/Dockerfile`

### SGLANG Container Images

#### NVIDIA CUDA
- **Tag**: `12.8.1-runtime-ubuntu24.04`
- **Category**: Base Image
- **Source File**: `container/Dockerfile.sglang`
- **CUDA Version**: Extracted from tag `12.8.1-runtime-ubuntu24.04`

#### NVIDIA CUDA-dl-base
- **Tag**: `25.01-cuda12.8-devel-ubuntu24.04`
- **Category**: Base Image
- **Source File**: `container/Dockerfile.sglang`
- **CUDA Version**: Extracted from tag `25.01-cuda12.8-devel-ubuntu24.04`

#### Sglang
- **Tag**: `v0.5.3.post2`
- **Category**: Base Image
- **Source File**: `container/Dockerfile.sglang-wideep`

#### Dynamo:latest None
- **Tag**: `latest`
- **Category**: Base Image
- **Source File**: `container/Dockerfile.sglang`

#### Runtime
- **Tag**: `latest`
- **Category**: Base Image
- **Source File**: `container/Dockerfile.sglang`

#### Scratch
- **Tag**: `latest`
- **Category**: Base Image
- **Source File**: `container/Dockerfile.sglang-wideep`

### SHARED Container Images

#### NVIDIA CUDA-dl-base
- **Tag**: `25.01-cuda12.8-devel-ubuntu24.04`
- **Category**: Base Image
- **Source File**: `container/Dockerfile`
- **CUDA Version**: Extracted from tag `25.01-cuda12.8-devel-ubuntu24.04`

#### Base
- **Tag**: `latest`
- **Category**: Base Image
- **Source File**: `container/Dockerfile`

#### Manylinux 2 28 X86 64
- **Tag**: `latest`
- **Category**: Base Image
- **Source File**: `container/Dockerfile`

### TRTLLM Container Images

#### NVIDIA CUDA
- **Tag**: `12.9.1-runtime-ubuntu24.04`
- **Category**: Base Image
- **Source File**: `container/Dockerfile.trtllm`
- **CUDA Version**: Extracted from tag `12.9.1-runtime-ubuntu24.04`

#### NVIDIA PyTorch
- **Tag**: `25.06-py3`
- **Category**: Base Image
- **Source File**: `container/Dockerfile.trtllm`

#### Dynamo:latest None
- **Tag**: `latest`
- **Category**: Base Image
- **Source File**: `container/Dockerfile.trtllm`

#### Runtime
- **Tag**: `latest`
- **Category**: Base Image
- **Source File**: `container/Dockerfile.trtllm`

### VLLM Container Images

#### NVIDIA CUDA
- **Tag**: `12.8.1-runtime-ubuntu24.04`
- **Category**: Base Image
- **Source File**: `container/Dockerfile.vllm`
- **CUDA Version**: Extracted from tag `12.8.1-runtime-ubuntu24.04`

#### NVIDIA CUDA-dl-base
- **Tag**: `25.01-cuda12.8-devel-ubuntu24.04`
- **Category**: Base Image
- **Source File**: `container/Dockerfile.vllm`
- **CUDA Version**: Extracted from tag `25.01-cuda12.8-devel-ubuntu24.04`

#### Dynamo:latest None
- **Tag**: `latest`
- **Category**: Base Image
- **Source File**: `container/Dockerfile.vllm`

#### Runtime
- **Tag**: `latest`
- **Category**: Base Image
- **Source File**: `container/Dockerfile.vllm`

## Framework-Specific Configurations

### vLLM Configuration
**Critical Dependencies:**
- Cuda: `12.8`
- Cuda: `12.8`
- Deepgemm: ``
- Flashinf: `v0.3.1`
- Flashinf: `v0.3.1`
- Python: `3.12`
- Vllm: `v0.11.0`
- Vllm: `v0.11.0`

**Build Location**: `container/deps/vllm/install_vllm.sh`
**Dockerfile**: `container/Dockerfile.vllm`

### TensorRT-LLM Configuration
**Critical Dependencies:**
- Flash Attn: `2.7.4.post1`
- Python: `3.12`
- Pytorch Triton: `3.3.0+git96316ce52.nvinternal`
- Ucx: `v1.18.1`

**Build Location**: `container/build_trtllm_wheel.sh`
**Dockerfile**: `container/Dockerfile.trtllm`

### SGLang Configuration
**Critical Dependencies:**
- Nats Server: `v2.10.28`
- Python: `3.12`
- Sglang: `0.5.3.post2`
- Sglang Image: `v0.5.3.post2`

**Build Location**: `container/Dockerfile.sglang`
**Dockerfile**: `container/Dockerfile.sglang`

## Dependency Management

### Automated Tracking
Dependency versions are automatically extracted and tracked nightly.

**Reports**:
- Latest versions: [`.github/reports/dependency_versions_latest.csv`](.github/reports/dependency_versions_latest.csv)
- Release snapshots: [`.github/reports/releases/`](.github/reports/releases/)
- Documentation: [`.github/reports/README.md`](.github/reports/README.md)

### Build Scripts
- **Main Build Script**: `container/build.sh`
- **vLLM Installation**: `container/deps/vllm/install_vllm.sh`
- **TensorRT-LLM Wheel**: `container/build_trtllm_wheel.sh`
- **NIXL Installation**: `container/deps/trtllm/install_nixl.sh`

### Python Dependencies
- **Core Requirements**: `container/deps/requirements.txt`
- **Standard Requirements**: `container/deps/requirements.standard.txt`
- **Test Requirements**: `container/deps/requirements.test.txt`

## Statistics

- **Total Dependencies Tracked**: 262
- **Critical Dependencies**: 55
- **NVIDIA Products**: 34

## Notes

- Different frameworks may use slightly different CUDA versions for runtime images
- NIXL and UCX are primarily used for distributed inference scenarios
- FlashInfer integration varies by build type (source builds, ARM64)
- Dependency versions are centrally managed through Docker build arguments and shell script variables
- Version discrepancies across components are automatically detected and reported

## Container Documentation

For detailed information about container builds and usage, see:
- [Container README](container/README.md)
- [Container Build Script](container/build.sh)
- [Container Run Script](container/run.sh)

## Related Documentation

- [Support Matrix](docs/support_matrix.md) - Supported platforms and versions
- [Dependency Extraction System](.github/scripts/dependency-extraction/README.md) - How dependencies are tracked
- [Dependency Reports](.github/reports/README.md) - CSV structure and workflows

---

_This document is automatically generated. Do not edit manually._
_To update, run: `python3 .github/scripts/dependency-extraction/generate_framework_versions.py`_