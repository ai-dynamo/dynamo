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

> **⚠️ AUTO-GENERATED** - Last updated: 2025-10-21
> 
> This document is automatically generated from [dependency extraction](.github/reports/dependency_versions_latest.csv).
> To update, run: `python3 .github/scripts/dependency-extraction/generate_framework_versions.py`

This document tracks the major framework dependencies and versions used in NVIDIA Dynamo.

## Quick Reference

| Component | Latest (main) | Release |
|-----------|---------------|---------|
| vLLM | `v0.11.0` | `N/A` |
| SGLang | `0.5.3.post2` | `N/A` |
| FlashInfer | `v0.3.1` | `N/A` |
| CUDA (base) | `N/A` | N/A |

## Core Framework Dependencies

### vLLM
- **Version**: `v0.11.0`
- **Component**: `vllm`
- **FlashInfer**: `v0.3.1` (high-performance attention kernels)

### SGLang
- **Version**: `0.5.3.post2`
- **Component**: `sglang`

## Base Images

### CUDA Runtime Images
- **Description**: NVIDIA CUDA runtime environment for production deployments


## Framework-Specific Configurations

### vLLM Configuration
- **Build Location**: `container/deps/vllm/install_vllm.sh`
- **Dockerfile**: `container/Dockerfile.vllm`

### TensorRT-LLM Configuration
- **Build Location**: `container/Dockerfile.trtllm`
- **Wheel Builder**: `container/build_trtllm_wheel.sh`

### SGLang Configuration
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

## See Also

- [Support Matrix](docs/support_matrix.md) - Supported platforms and versions
- [Dependency Reports](.github/reports/README.md) - Full dependency tracking (262 total dependencies)
- [Dependency Extraction System](.github/scripts/dependency-extraction/README.md) - How this doc is generated
- [Container README](container/README.md) - Container build and usage details

---

_This document is automatically generated. Do not edit manually._
_Last generated: 2025-10-21_

**Update Instructions:**
```bash
python3 .github/scripts/dependency-extraction/generate_framework_versions.py
```