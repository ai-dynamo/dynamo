# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service CUDA extensions.

These extensions are built at install time using torch.utils.cpp_extension.
After building, import _rpc_cumem_ext and _tensor_from_pointer from this module.
"""

# These are built by setup.py build_ext --inplace
# Import will fail until extensions are built
try:
    from gpu_memory_service.extensions import _rpc_cumem_ext  # noqa: F401
    from gpu_memory_service.extensions._rpc_cumem_ext import *  # noqa: F401, F403
except ImportError:
    _rpc_cumem_ext = None  # type: ignore

try:
    from gpu_memory_service.extensions import _tensor_from_pointer  # noqa: F401
    from gpu_memory_service.extensions._tensor_from_pointer import *  # noqa: F401, F403
except ImportError:
    _tensor_from_pointer = None  # type: ignore
