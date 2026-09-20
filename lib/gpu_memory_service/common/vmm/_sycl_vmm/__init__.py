# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SYCL VMM native extension for GMS XPU backend.

The compiled pybind11 module (_sycl_vmm.<ext>.so) is loaded from this
package.  If the extension has not been built, import raises a clear
error rather than a cryptic shared-library load failure.

The module is found in one of two locations:
  1. Installed in-place (this directory): after ``cmake --install build --prefix .``
  2. Left in the CMake build directory: ``build/_sycl_vmm*.so``
"""

import atexit as _atexit
import importlib.util
import os
import sys


def _try_load_from_build_dir():
    """Attempt to load _sycl_vmm from the adjacent build/ directory."""
    build_dir = os.path.join(os.path.dirname(__file__), "build")
    if not os.path.isdir(build_dir):
        return None
    for fname in os.listdir(build_dir):
        if fname.startswith("_sycl_vmm") and fname.endswith(".so"):
            so_path = os.path.join(build_dir, fname)
            spec = importlib.util.spec_from_file_location("_sycl_vmm", so_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
    return None


# Try 1: standard package import (works after install or symlink)
try:
    from gpu_memory_service.common.vmm._sycl_vmm._sycl_vmm import *  # noqa: F401,F403
    from gpu_memory_service.common.vmm._sycl_vmm._sycl_vmm import (  # noqa: F401
        HAS_SYCL_HOST_REGISTER,
        HAS_SYCL_IPC,
        ONEAPI_VERSION,
    )
except ImportError:
    # Try 2: load from build/ subdirectory (development workflow)
    _mod = _try_load_from_build_dir()
    if _mod is None:
        raise ImportError(
            "The _sycl_vmm native extension is not built. "
            "Build it with: cd _sycl_vmm && cmake -B build "
            "-DCMAKE_CXX_COMPILER=icpx . && cmake --build build"
        )
    # Inject into sys.modules so subsequent imports work
    sys.modules[__name__] = _mod
    # Re-export key attributes at package level
    globals().update({k: getattr(_mod, k) for k in dir(_mod) if not k.startswith("__")})


# Register finalize() as an atexit handler so SYCL/L0 resources are
# destroyed before C++ static destructors run (avoids abort on exit).


def _finalize_if_available():
    import sys

    mod = sys.modules.get(__name__)
    if mod is not None and hasattr(mod, "finalize"):
        try:
            mod.finalize()
        except Exception:
            pass


_atexit.register(_finalize_if_available)
