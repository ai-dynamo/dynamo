# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Constants for dependency extraction.

This module contains all hardcoded values that may need updating as the project evolves.
"""

# NVIDIA product indicators for auto-detection
# Add new NVIDIA product keywords here as they are introduced
NVIDIA_INDICATORS = [
    "nvidia",
    "nvcr.io",
    "cuda",
    "tensorrt",
    "triton",
    "nccl",
    "nvshmem",
    "dcgm",
    "cutlass",
    "cudf",
    "rapids",
    "dali",
    "tao",
    "nvtabular",
    "merlin",
    "trt",
    "nemo",
]

# Dependency name normalizations for version discrepancy detection
# Maps variations of dependency names to a canonical name
# Add entries when you discover dependencies with inconsistent naming
NORMALIZATIONS = {
    "tensorrt-llm": "tensorrt-llm",
    "trtllm": "tensorrt-llm",
    "tensorrt": "tensorrt",
    "pytorch": "pytorch",
    "torch": "pytorch",
    "tensorflow": "tensorflow",
    "cuda": "cuda",
    "cudnn": "cudnn",
    "nccl": "nccl",
    "nixl": "nixl",
}

# PyTorch-related packages that should NOT be normalized to "pytorch"
# e.g., "pytorch triton" is the Triton compiler, not PyTorch itself
PYTORCH_EXCEPTIONS = ["pytorch triton", "pytorch_triton", "triton"]

# Component sort order for CSV output
# Lower numbers appear first in the CSV
# Add new components here with appropriate sort priority
COMPONENT_ORDER = {
    "trtllm": 0,
    "vllm": 1,
    "sglang": 2,
    "operator": 3,
    "shared": 4,
}

# CSV column order
CSV_COLUMNS = [
    "Component",
    "Category",
    "Dependency Name",
    "Version",
    "Source File",
    "GitHub URL",
    "Package Source URL",
    "Status",
    "Diff from Latest",
    "Diff from Release",
    "Critical",
    "NVIDIA Product",
    "Notes",
]

# Default critical dependencies if not specified in config
DEFAULT_CRITICAL_DEPENDENCIES = [
    {"name": "CUDA", "reason": "Core compute platform"},
    {"name": "PyTorch", "reason": "Primary ML framework"},
    {"name": "Python", "reason": "Runtime language"},
    {"name": "Kubernetes", "reason": "Orchestration platform"},
]

# Baseline dependency count for warnings (updated dynamically from previous CSV)
DEFAULT_BASELINE_COUNT = 251

