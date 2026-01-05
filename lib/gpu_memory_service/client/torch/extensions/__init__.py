# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service CUDA extensions for PyTorch integration.

These extensions are built at install time using torch.utils.cpp_extension.

- _allocator_ext: CUDAPluggableAllocator backend (my_malloc/my_free)
- _tensor_from_pointer: Create torch.Tensor from CUDA pointer
"""

# These are built by setup.py build_ext --inplace
# Import will fail until extensions are built
try:
    from gpu_memory_service.client.torch.extensions import _allocator_ext  # noqa: F401
    from gpu_memory_service.client.torch.extensions._allocator_ext import *  # noqa: F401, F403
except ImportError:
    _allocator_ext = None  # type: ignore

try:
    from gpu_memory_service.client.torch.extensions import (  # noqa: F401
        _tensor_from_pointer,
    )
    from gpu_memory_service.client.torch.extensions._tensor_from_pointer import *  # noqa: F401, F403
except ImportError:
    _tensor_from_pointer = None  # type: ignore
