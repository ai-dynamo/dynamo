# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service C++ extensions for PyTorch integration.

These extensions are built at install time using setuptools.

- _allocator_ext: CUDAPluggableAllocator backend (my_malloc/my_free)
"""

# Built by setup.py build_ext --inplace
# Import will fail until extensions are built
try:
    from gpu_memory_service.client.torch.extensions import _allocator_ext  # noqa: F401
    from gpu_memory_service.client.torch.extensions._allocator_ext import *  # noqa: F401, F403
except ImportError:
    _allocator_ext = None  # type: ignore
