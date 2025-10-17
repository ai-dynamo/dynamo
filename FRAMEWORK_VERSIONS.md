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

# Dynamo Framework Versions

This document tracks the major dependencies and versions used in the NVIDIA Dynamo project.

## Core Framework Dependencies

### vLLM
- **Version**: v0.11.0

### TensorRT-LLM
- **Version**: 1.1.0rc5

### FlashInfer
- **Version**: v0.3.1
- **Description**: High-performance attention kernel library

### NIXL
- **Version**: 0.6.0
- **Description**: NVIDIA's high-performance networking library for distributed LLM inference

### UCX (Unified Communication X)
- **Version**: v1.19.0
- **Description**: Communication framework used by NIXL

## Base Images

### CUDA Development Image
- **Version**: CUDA 12.8
- **Base Image Tag**: `25.01-cuda12.8-devel-ubuntu24.04`
- **Description**: NVIDIA CUDA development environment

### CUDA Runtime Images
- **Description**: NVIDIA CUDA runtime environment for production deployments
- **Default**: CUDA 12.8.1 (`12.8.1-runtime-ubuntu24.04`)
- **TensorRT-LLM**: CUDA 12.9.1 (`12.9.1-runtime-ubuntu24.04`)

## Framework-Specific Configurations

### vLLM Configuration
- **CUDA Version**: 12.8
- **FlashInfer Integration**: Enabled for source builds or ARM64 builds
- **Build Location**: `container/deps/vllm/install_vllm.sh`

### TensorRT-LLM Configuration
- **Runtime Image**: `12.9.1-runtime-ubuntu24.04` (CUDA 12.9.1)
- **Build Location**: `container/Dockerfile.trtllm`

### SGLang Configuration
- **CUDA Version**: 12.8
- **Base Image**: Same as vLLM (`25.01-cuda12.8-devel-ubuntu24.04`)
- **Build Location**: `container/Dockerfile.sglang`

## Dependency Management

### Build Scripts
- **Main Build Script**: `container/build.sh`
- **vLLM Installation**: `container/deps/vllm/install_vllm.sh`
- **TensorRT-LLM Wheel**: `container/build_trtllm_wheel.sh`
- **NIXL Installation**: `container/deps/trtllm/install_nixl.sh`

### Python Dependencies
- **Requirements File**: `container/deps/requirements.txt`
- **Standard Requirements**: `container/deps/requirements.standard.txt`
- **Test Requirements**: `container/deps/requirements.test.txt`

## Notes

- FlashInfer is only used when building vLLM from source or for ARM64 builds
- Different frameworks may use slightly different CUDA versions for runtime images
- NIXL and UCX are primarily used for distributed inference scenarios
- The dependency versions are centrally managed through Docker build arguments and shell script variables

## Container Documentation

For detailed information about container builds and usage, see:
- [Container README](container/README.md)
- [Container Build Script](container/build.sh)
- [Container Run Script](container/run.sh)
