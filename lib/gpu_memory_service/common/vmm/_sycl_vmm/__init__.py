# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""SYCL VMM native extension for GMS XPU backend.

The compiled pybind11 module (_sycl_vmm.<ext>.so) is loaded from this
package.  If the extension has not been built, import raises a clear
error rather than a cryptic shared-library load failure.
"""

try:
    from gpu_memory_service.common.vmm._sycl_vmm._sycl_vmm import *  # noqa: F401,F403
    from gpu_memory_service.common.vmm._sycl_vmm._sycl_vmm import (  # noqa: F401
        HAS_SYCL_FREE_MEMORY,
        HAS_SYCL_HOST_REGISTER,
        HAS_SYCL_IPC,
        ONEAPI_VERSION,
    )
except ImportError as exc:
    raise ImportError(
        "The _sycl_vmm native extension is not built. "
        "Build it with: cd _sycl_vmm && cmake -B build -DCMAKE_CXX_COMPILER=icpx . "
        "&& cmake --build build"
    ) from exc
